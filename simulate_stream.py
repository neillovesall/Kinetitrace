"""
simulate_serial.py — KineTrace V7 Hardware Simulator
=======================================================
Generates synthetic 50 Hz BNO085-style quaternion JSON and delivers
it via ONE of three modes:

  Mode A — Virtual serial port pair (Linux/Mac with socat):
    socat -d -d pty,raw,echo=0 pty,raw,echo=0
    # note the two /dev/pts/N paths printed, then:
    python simulate_serial.py --mode serial --port /dev/pts/3

  Mode B — Write to a real COM port (Windows loopback with com0com):
    python simulate_serial.py --mode serial --port COM5

  Mode C — WebSocket direct (no serial, connects straight to Flask-SocketIO):
    python simulate_serial.py --mode ws --url http://127.0.0.1:5000

Usage:
  python simulate_serial.py [--mode serial|ws] [--port PORT]
                            [--url URL] [--hz HZ] [--danger]
"""

import argparse, math, time, json, sys, threading
import numpy as np

# ── PySerial ───────────────────────────────────────────────
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ── python-socketio client ─────────────────────────────────
try:
    import socketio as sio_client
    HAS_SIO = True
except ImportError:
    HAS_SIO = False

# ─── Constants ────────────────────────────────────────────
DEFAULT_HZ     = 50
REP_PERIOD_S   = 3.5     # seconds per full knee extension rep
MAX_ANGLE_DEG  = 88.0    # normal peak flexion
DANGER_ANGLE   = 148.0   # spike angle for danger mode
NOISE_STD      = 0.0035  # quaternion noise (simulates BNO085 noise floor)
THIGH_LEAN_DEG = 11.0    # static anatomical forward lean of thigh


# ════════════════════════════════════════════════════════════
# QUATERNION HELPERS
# ════════════════════════════════════════════════════════════

def euler_to_quat_x(angle_rad: float) -> dict:
    """
    Pure rotation around X axis (sagittal plane).
    q = [cos(a/2), sin(a/2), 0, 0]
    Already unit length for single-axis rotation.
    """
    c = math.cos(angle_rad / 2)
    s = math.sin(angle_rad / 2)
    return {"w": c, "x": s, "y": 0.0, "z": 0.0}


def add_noise(q: dict, std: float) -> dict:
    """Perturb all four components with Gaussian noise, re-normalise."""
    a = np.array([q["w"], q["x"], q["y"], q["z"]]) + np.random.normal(0, std, 4)
    a /= np.linalg.norm(a)
    return {"w": float(a[0]), "x": float(a[1]), "y": float(a[2]), "z": float(a[3])}


# ════════════════════════════════════════════════════════════
# MOTION PROFILE
# ════════════════════════════════════════════════════════════

def knee_flexion_profile(t: float, period: float,
                         peak_deg: float, danger: bool) -> float:
    """
    Smooth sinusoidal knee extension profile.
      0 deg at t=0, peak_deg at t=period/2, 0 deg at t=period.

    Danger mode: every 4th rep spikes to DANGER_ANGLE during
    the mid-extension window to test the hyperflexion alert path.
    """
    base = peak_deg * (1.0 - math.cos(2.0 * math.pi * t / period)) / 2.0

    if danger:
        # Spike window: 2nd quarter of every 4th rep
        rep_t = t % (period * 4)
        if 0.45 * period < rep_t < 0.55 * period:
            spike = DANGER_ANGLE * (1.0 - math.cos(math.pi * (rep_t - 0.45*period) / (0.10*period))) / 2.0
            base = max(base, spike)

    return base


def make_payload(elapsed: float, period: float,
                 peak_deg: float, danger: bool) -> dict:
    """
    Build one telemetry packet matching the server's expected schema:
      {"timestamp": float, "thigh": {w,x,y,z}, "shin": {w,x,y,z}}
    """
    flex_deg = knee_flexion_profile(elapsed, period, peak_deg, danger)
    flex_rad = math.radians(flex_deg)
    lean_rad = math.radians(THIGH_LEAN_DEG)

    # Thigh: constant forward lean (represents pelvis tilt)
    q_thigh = add_noise(euler_to_quat_x(lean_rad), NOISE_STD)

    # Shin: thigh lean + knee flexion (shin rotates backward from thigh)
    q_shin  = add_noise(euler_to_quat_x(lean_rad - flex_rad), NOISE_STD)

    return {
        "timestamp": round(elapsed, 5),
        "thigh":     q_thigh,
        "shin":      q_shin,
        "_flex_deg": round(flex_deg, 2),   # debug field; server ignores unknown keys
    }


# ════════════════════════════════════════════════════════════
# MODE A / B — SERIAL OUTPUT
# ════════════════════════════════════════════════════════════

def run_serial(port: str, hz: int, danger: bool):
    if not HAS_SERIAL:
        print("[!] pyserial not installed. Run: pip install pyserial"); sys.exit(1)

    dt = 1.0 / hz
    t0 = time.time()
    count = 0

    print(f"\nKineTrace V7 Serial Simulator")
    print(f"  Port    : {port} @ 115200 baud")
    print(f"  Rate    : {hz} Hz  |  Rep period: {REP_PERIOD_S}s")
    print(f"  Danger  : {danger}\n")

    while True:
        try:
            with serial.Serial(port, 115200, timeout=1.0) as ser:
                print(f"[sim] Port {port} open. Streaming...")
                while True:
                    loop_t = time.perf_counter()
                    elapsed = time.time() - t0
                    pkt = make_payload(elapsed, REP_PERIOD_S, MAX_ANGLE_DEG, danger)
                    line = (json.dumps(pkt) + "\n").encode("utf-8")
                    ser.write(line)
                    count += 1
                    if count % (hz * 2) == 0:
                        print(f"  t={elapsed:7.2f}s  θ={pkt['_flex_deg']:6.2f}°  pkts={count}")
                    sleep = dt - (time.perf_counter() - loop_t)
                    if sleep > 0:
                        time.sleep(sleep)
        except serial.SerialException as e:
            print(f"[!] {e} — retrying in 3s...")
            time.sleep(3.0)
        except KeyboardInterrupt:
            print("\n[sim] Stopped."); break


# ════════════════════════════════════════════════════════════
# MODE C — WEBSOCKET DIRECT
# ════════════════════════════════════════════════════════════

def run_ws(url: str, hz: int, danger: bool):
    if not HAS_SIO:
        print("[!] python-socketio not installed. Run: pip install 'python-socketio[client]'")
        sys.exit(1)

    dt = 1.0 / hz
    t0 = time.time()
    count = 0
    connected = threading.Event()
    state_machine = {"state": "DISCONNECTED"}

    sio = sio_client.Client()

    @sio.event
    def connect():
        print(f"[sim] WebSocket connected to {url}")
        connected.set()

    @sio.event
    def disconnect():
        print("[sim] Disconnected.")
        connected.clear()

    @sio.on("state_change")
    def on_state(data):
        s = data.get("state", "")
        state_machine["state"] = s
        print(f"[sim] Server state → {s}: {data.get('msg','')}")
        if s == "CALIBRATING":
            # Wait 3 seconds then send calibration trigger
            def _send_cal():
                time.sleep(3.2)
                print("[sim] Sending calibration trigger...")
                sio.emit("trigger_calibration", {
                    "thigh": {"w":1,"x":0,"y":0,"z":0},
                    "shin":  {"w":1,"x":0,"y":0,"z":0},
                })
            threading.Thread(target=_send_cal, daemon=True).start()

    @sio.on("score_update")
    def on_score(data):
        pass   # score updates logged by server

    @sio.on("frame_update")
    def on_frame(data):
        pass   # no-op on simulator side

    print(f"\nKineTrace V7 WebSocket Simulator")
    print(f"  Server  : {url}")
    print(f"  Rate    : {hz} Hz  |  Danger: {danger}\n")

    sio.connect(url)
    connected.wait(timeout=5.0)
    if not connected.is_set():
        print("[!] Could not connect. Is Flask-SocketIO running?"); sys.exit(1)

    # Trigger pairing state machine
    time.sleep(0.3)
    sio.emit("start_scanning")

    # Wait until LIVE
    print("[sim] Waiting for LIVE state...")
    for _ in range(60):
        if state_machine["state"] == "LIVE":
            break
        time.sleep(0.3)

    if state_machine["state"] != "LIVE":
        print("[!] Server did not reach LIVE state."); sys.exit(1)

    print("[sim] LIVE — streaming telemetry at", hz, "Hz")

    try:
        while True:
            loop_t  = time.perf_counter()
            elapsed = time.time() - t0
            pkt     = make_payload(elapsed, REP_PERIOD_S, MAX_ANGLE_DEG, danger)
            sio.emit("submit_telemetry", pkt)
            count += 1
            if count % (hz * 2) == 0:
                print(f"  t={elapsed:7.2f}s  θ={pkt['_flex_deg']:6.2f}°  pkts={count}")
            sleep = dt - (time.perf_counter() - loop_t)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[sim] Stopped.")
        sio.disconnect()


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="KineTrace V7 Hardware Simulator")
    p.add_argument("--mode",   choices=["serial","ws"], default="ws",
                   help="serial = write to COM/tty port; ws = direct WebSocket (default: ws)")
    p.add_argument("--port",   default="COM3",
                   help="Serial port for --mode serial (default: COM3 / /dev/pts/N)")
    p.add_argument("--url",    default="http://127.0.0.1:5000",
                   help="Flask-SocketIO server URL for --mode ws")
    p.add_argument("--hz",     type=int, default=DEFAULT_HZ,
                   help=f"Sample rate in Hz (default: {DEFAULT_HZ})")
    p.add_argument("--danger", action="store_true",
                   help="Periodically spike angle > 130° to test hyperflexion alerts")
    args = p.parse_args()

    if args.mode == "serial":
        run_serial(args.port, args.hz, args.danger)
    else:
        run_ws(args.url, args.hz, args.danger)