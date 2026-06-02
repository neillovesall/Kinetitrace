from __future__ import annotations

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import threading, time, json, os

# ── PySerial ──────────────────────────────────────────────
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ── Numba ─────────────────────────────────────────────────
try:
    from numba import njit as _njit
    NUMBA = True
except ImportError:
    def _njit(fn): return fn          # transparent no-op
    NUMBA = False

# ─── App ──────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "kinetrace-v8"
sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ─── Constants ────────────────────────────────────────────
BUFFER_MAX   = 600
TARGET_HZ    = 50.0
CUTOFF_HZ    = 10.0
FILT_ORDER   = 2
TAU          = 50.0
SAFE_MAX_DEG = 130.0
SCORE_EVERY  = 25           # emit score_update every N telemetry frames

SERIAL_PORT  = os.environ.get("KINETRACE_PORT", "COM3")
BAUD_RATE    = 115200

# ─── Thread pool for batch pipeline (max 2 workers) ───────
_batch_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kt_batch")

# ─── Gold-standard template ───────────────────────────────
_t_g   = np.linspace(0, 2, 100)
_ang_g = 45.0 * (1.0 - np.cos(np.pi * _t_g))
_vel_g = np.gradient(_ang_g, _t_g)
_acc_g = np.gradient(_vel_g, _t_g)
GOLD   = np.column_stack([_ang_g, _vel_g, _acc_g]).astype(np.float64)

# ─── Shared state ─────────────────────────────────────────
_lock = threading.Lock()
_state: dict = {
    "t_buf":        deque(maxlen=BUFFER_MAX),
    "ang_buf":      deque(maxlen=BUFFER_MAX),
    "frame_ctr":    0,
    "cal_thigh":    np.array([1., 0., 0., 0.]),
    "cal_shin":     np.array([1., 0., 0., 0.]),
    "cal_locked":   False,
    "score":        100.0,
    "device_state": "DISCONNECTED",
}


# ══════════════════════════════════════════════════════════
# 1. QUATERNION MATH  (explicit NumPy — Python 3.8 compatible)
# ══════════════════════════════════════════════════════════

def qnorm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1., 0., 0., 0.])

def qconj(q: np.ndarray) -> np.ndarray:
    """Unit quaternion inverse: conjugate == inverse for ‖q‖=1."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product — all 8 multiplications explicit.
      w = w1w2 − x1x2 − y1y2 − z1z2
      x = w1x2 + x1w2 + y1z2 − z1y2
      y = w1y2 − x1z2 + y1w2 + z1x2
      z = w1z2 + x1y2 − y1x2 + z1w2
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def qcal(q_live: np.ndarray, q_static: np.ndarray) -> np.ndarray:
    """Remove static attachment offset: q_cal = q_live ⊗ q_static⁻¹"""
    return qnorm(qmul(q_live, qconj(qnorm(q_static))))

def q2rot(q: np.ndarray) -> np.ndarray:
    """Unit quaternion → 3×3 rotation matrix (all 9 terms explicit)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])

def seg_vec(q: np.ndarray) -> np.ndarray:
    """Apply rotation to neutral bone axis [0,0,-1] → world-space direction."""
    v = q2rot(q) @ np.array([0., 0., -1.])
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def flex_angle(vt: np.ndarray, vs: np.ndarray) -> float:
    """θ = arccos(v_thigh · v_shin) — clamped to [-1,1] for float safety."""
    return float(np.degrees(np.arccos(np.clip(np.dot(vt, vs), -1.0, 1.0))))


# ══════════════════════════════════════════════════════════
# 2. NUMBA JIT — DTW accumulated cost matrix
# ══════════════════════════════════════════════════════════

@_njit
def dtw_numba(a: np.ndarray, b: np.ndarray) -> float:
    """
    Multidimensional DTW via explicit dynamic programming.
    a, b : float64 arrays (N, F) and (M, F).
    Returns scalar warping distance D[N-1, M-1].
    """
    n = a.shape[0]
    m = b.shape[0]
    INF = 1e18
    D = np.full((n, m), INF)

    # Seed
    s = 0.0
    for f in range(a.shape[1]):
        d = a[0, f] - b[0, f];  s += d * d
    D[0, 0] = s ** 0.5

    # First column
    for i in range(1, n):
        s = 0.0
        for f in range(a.shape[1]):
            d = a[i, f] - b[0, f];  s += d * d
        D[i, 0] = D[i-1, 0] + s ** 0.5

    # First row
    for j in range(1, m):
        s = 0.0
        for f in range(a.shape[1]):
            d = a[0, f] - b[j, f];  s += d * d
        D[0, j] = D[0, j-1] + s ** 0.5

    # Inner DP
    for i in range(1, n):
        for j in range(1, m):
            s = 0.0
            for f in range(a.shape[1]):
                d = a[i, f] - b[j, f];  s += d * d
            cost = s ** 0.5
            best = D[i-1, j]
            if D[i, j-1]   < best: best = D[i, j-1]
            if D[i-1, j-1] < best: best = D[i-1, j-1]
            D[i, j] = cost + best

    return D[n-1, m-1]


def _warmup_jit():
    try:
        tiny = np.zeros((4, 3), dtype=np.float64)
        dtw_numba(tiny, tiny)
    except Exception:
        pass

threading.Thread(target=_warmup_jit, daemon=True).start()


# ══════════════════════════════════════════════════════════
# 3. SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════

def butterworth_lp(sig: np.ndarray,
                   cutoff: float = CUTOFF_HZ,
                   fs: float = TARGET_HZ,
                   order: int = FILT_ORDER) -> np.ndarray:
    nyq  = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    pad  = 3 * max(len(a), len(b))
    return filtfilt(b, a, sig) if len(sig) > pad else sig.copy()

def resample_cubic(t_raw: np.ndarray, y_raw: np.ndarray,
                   hz: float = TARGET_HZ):
    if len(t_raw) < 4:
        return t_raw, y_raw
    dur = t_raw[-1] - t_raw[0]
    if dur < 1e-6:
        return t_raw, y_raw
    t_u = np.linspace(t_raw[0], t_raw[-1], max(4, int(round(dur * hz))))
    return t_u, CubicSpline(t_raw, y_raw)(t_u)

def zscore(arr: np.ndarray) -> np.ndarray:
    mu = arr.mean(axis=0);  sg = arr.std(axis=0)
    sg[sg < 1e-9] = 1.0
    return (arr - mu) / sg

def adherence(D: float) -> float:
    return float(100.0 * np.exp(-D / TAU))


# ══════════════════════════════════════════════════════════
# 4. BATCH PIPELINE
#    Receives snapshot arrays — does NOT hold _lock during compute.
#    Emits score_update after computation completes.
# ══════════════════════════════════════════════════════════

def run_batch(angles_snap: np.ndarray, times_snap: np.ndarray,
              sid: Optional[str]) -> None:
    if len(angles_snap) < 12:
        return

    t_u, ang_u = resample_cubic(times_snap, angles_snap)
    if len(ang_u) < 12:
        return

    ang_f = butterworth_lp(ang_u)
    dt    = 1.0 / TARGET_HZ
    vel   = np.gradient(ang_f, dt)
    acc   = np.gradient(vel,   dt)
    feat  = zscore(np.column_stack([ang_f, vel, acc])).astype(np.float64)
    L     = min(len(feat), len(GOLD))
    if L < 4:
        return

    D     = dtw_numba(feat[:L], GOLD[:L])
    score = adherence(D)

    with _lock:
        _state["score"] = round(score, 1)

    # Emit OUTSIDE lock
    if sid:
        sio.emit("score_update", {"score": round(score, 1)}, room=sid)
    else:
        sio.emit("score_update", {"score": round(score, 1)})


# ══════════════════════════════════════════════════════════
# 5. TELEMETRY PROCESSOR
#    FIX: sio.emit() is now called AFTER releasing _lock.
# ══════════════════════════════════════════════════════════

def process_frame(raw: dict, sid: Optional[str] = None) -> None:
    try:
        ts = float(raw["timestamp"])
        qt = qnorm(np.array([
            float(raw["thigh"]["w"]), float(raw["thigh"]["x"]),
            float(raw["thigh"]["y"]), float(raw["thigh"]["z"])]))
        qs = qnorm(np.array([
            float(raw["shin"]["w"]),  float(raw["shin"]["x"]),
            float(raw["shin"]["y"]),  float(raw["shin"]["z"])]))
    except (KeyError, TypeError, ValueError):
        return

    # ── Compute under lock, snapshot what we need, release ──
    with _lock:
        if not _state["cal_locked"]:
            _state["cal_thigh"] = qt.copy()
            _state["cal_shin"]  = qs.copy()
            _state["cal_locked"] = True

        qt_c = qcal(qt, _state["cal_thigh"])
        qs_c = qcal(qs, _state["cal_shin"])
        vt   = seg_vec(qt_c)
        vs   = seg_vec(qs_c)
        ang  = flex_angle(vt, vs)

        _state["t_buf"].append(ts)
        _state["ang_buf"].append(ang)
        fc    = _state["frame_ctr"] = _state["frame_ctr"] + 1
        score = _state["score"]

        # Snapshot for batch (copy while locked, cheap for deque→array)
        if fc % SCORE_EVERY == 0:
            ang_snap  = np.array(_state["ang_buf"])
            time_snap = np.array(_state["t_buf"])
        else:
            ang_snap = time_snap = None
    # ── Lock released ────────────────────────────────────

    payload = {
        "angle":     round(ang, 3),
        "alert":     bool(ang > SAFE_MAX_DEG),
        "score":     score,
        "thigh_q":   qt_c.tolist(),
        "shin_q":    qs_c.tolist(),
        "thigh_vec": vt.tolist(),
        "shin_vec":  vs.tolist(),
    }

    # Emit OUTSIDE lock — no deadlock risk
    if sid:
        sio.emit("frame_update", payload, room=sid)
    else:
        sio.emit("frame_update", payload)

    # Submit batch to pool (non-blocking)
    if ang_snap is not None:
        _batch_pool.submit(run_batch, ang_snap, time_snap, sid)


# ══════════════════════════════════════════════════════════
# 6. PYSERIAL DAEMON
# ══════════════════════════════════════════════════════════

class SerialDaemon(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        if not HAS_SERIAL:
            sio.emit("device_event", {
                "type": "error",
                "msg": "pyserial not installed — use WebSocket sim mode"})
            return

        sio.emit("device_event", {
            "type": "scanning",
            "msg": f"Opening {SERIAL_PORT} @ {BAUD_RATE} baud..."})

        while not self._stop_evt.is_set():
            try:
                with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.0) as ser:
                    sio.emit("device_event", {
                        "type": "connected",
                        "msg": f"ESP32 detected on {SERIAL_PORT}"})
                    with _lock:
                        _state["device_state"] = "CALIBRATING"
                    sio.emit("state_change", {"state": "CALIBRATING"})

                    buf = b""
                    while not self._stop_evt.is_set():
                        waiting = ser.in_waiting
                        chunk   = ser.read(waiting if waiting > 0 else 1)
                        if not chunk:
                            continue
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            try:
                                raw = json.loads(
                                    line.decode("utf-8", errors="ignore"))
                                process_frame(raw)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                pass

            except Exception as e:
                sio.emit("device_event", {"type": "error", "msg": str(e)})
                time.sleep(3.0)


# Daemon is created lazily on first start_scanning to avoid
# premature emit() calls before SocketIO is ready.
_serial_daemon: Optional[SerialDaemon] = None
_daemon_lock = threading.Lock()


def _get_or_start_daemon() -> None:
    global _serial_daemon
    with _daemon_lock:
        if _serial_daemon is None:
            _serial_daemon = SerialDaemon()
        if not _serial_daemon.is_alive():
            # Threads cannot be restarted; create a new one each reconnect
            _serial_daemon = SerialDaemon()
            _serial_daemon.start()


# ══════════════════════════════════════════════════════════
# 7. SOCKET EVENTS
# ══════════════════════════════════════════════════════════

@sio.on("connect")
def on_connect():
    with _lock:
        ds = _state["device_state"]
    emit("state_change",  {"state": ds})
    emit("numba_status",  {"active": NUMBA})
    emit("serial_status", {"has_serial": HAS_SERIAL, "port": SERIAL_PORT})

@sio.on("disconnect")
def on_disconnect():
    pass

@sio.on("start_scanning")
def on_scan():
    with _lock:
        _state["device_state"] = "SCANNING"
    emit("state_change", {
        "state": "SCANNING",
        "msg":   f"Scanning {SERIAL_PORT}..."})
    _get_or_start_daemon()

@sio.on("trigger_calibration")
def on_calibrate(data):
    try:
        qt = np.array([float(data["thigh"]["w"]), float(data["thigh"]["x"]),
                       float(data["thigh"]["y"]), float(data["thigh"]["z"])])
        qs = np.array([float(data["shin"]["w"]),  float(data["shin"]["x"]),
                       float(data["shin"]["y"]),  float(data["shin"]["z"])])
    except (KeyError, TypeError, ValueError):
        emit("error", {"msg": "Bad calibration payload"})
        return

    with _lock:
        _state["cal_thigh"]   = qnorm(qt)
        _state["cal_shin"]    = qnorm(qs)
        _state["cal_locked"]  = True
        _state["t_buf"].clear()
        _state["ang_buf"].clear()
        _state["frame_ctr"]   = 0
        _state["device_state"] = "LIVE"

    emit("state_change", {"state": "LIVE"})

@sio.on("submit_telemetry")
def on_telemetry(data):
    from flask_socketio import request as sr
    process_frame(data, sid=sr.sid)

@sio.on("reset_calibration")
def on_reset_cal():
    with _lock:
        _state["cal_locked"]   = False
        _state["device_state"] = "CALIBRATING"
    emit("state_change", {"state": "CALIBRATING"})


# ── HTTP ──────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           numba_active=NUMBA,
                           has_serial=HAS_SERIAL,
                           serial_port=SERIAL_PORT)

if __name__ == "__main__":
    sio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)