"""
tinySA device layer: USB-CDC shell protocol, a simulator, and a worker thread.

Protocol notes (tinySA / tinySA Ultra firmware shell):
  * Virtual COM port (STM32 VCP, VID 0x0483 PID 0x5740). Baud rate is ignored.
  * Send "<command>\r"; the device echoes the line, prints the reply, then the
    prompt "ch> ".
  * scanraw <start> <stop> <points>  ->  '{' + points * ('x' + uint16 LE) + '}'
    level_dBm = raw / 32 - offset   (offset 128 for tinySA, 174 for Ultra)
  * capture  ->  width*height*2 bytes of RGB565 (big endian) screen dump.
"""
import math
import queue
import random
import threading
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

try:
    import serial
    import serial.tools.list_ports
except ImportError:  # pragma: no cover
    serial = None

TINYSA_VID, TINYSA_PID = 0x0483, 0x5740
TINYSA_ULTRA_VID, TINYSA_ULTRA_PID = 0x1781, 0x0A44
PROMPT = b"ch> "
SIM_PORT = "Simulator"

MODELS = {
    "basic": dict(name="tinySA", scale=128.0, fmax=960e6, low_max=350e6,
                  rbw=[3, 10, 30, 100, 300, 600], screen=(320, 240), att_max=31),
    "ultra": dict(name="tinySA Ultra", scale=174.0, fmax=6.0e9, low_max=800e6,
                  rbw=[0.2, 1, 3, 10, 30, 100, 300, 600, 850], screen=(480, 320), att_max=31),
}


def list_ports():
    """Return list of (device, description, is_tinysa)."""
    out = []
    if serial is None:
        return out
    for p in serial.tools.list_ports.comports():
        is_t = (p.vid == TINYSA_VID and p.pid == TINYSA_PID) or (p.vid == TINYSA_ULTRA_VID and p.pid == TINYSA_ULTRA_PID)  
        desc = p.description or ""
        out.append((p.device, desc, is_t))
    out.sort(key=lambda x: (not x[2], x[0]))
    return out


# --------------------------------------------------------------------------
class SerialTransport:
    def __init__(self, port):
        if serial is None:
            raise RuntimeError("pyserial is not installed")
        self.scale = 174.0
        self.ser = serial.Serial(port, 115200, timeout=0.05, write_timeout=2)
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        self.ser.write(b"\r")
        try:
            self._read_until(PROMPT, 1.5)
        except TimeoutError:
            pass
        time.sleep(0.05)
        self.rx = b""
        self.ser.reset_input_buffer()

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    # low level ----------------------------------------------------------
    # A persistent receive buffer so bytes that arrive in the same USB packet
    # as a delimiter are never lost.
    rx = b""

    def _fill(self):
        chunk = self.ser.read(max(1, self.ser.in_waiting))
        if chunk:
            self.rx += chunk
            return True
        return False

    def _read_until(self, token, idle_timeout):
        """Read until token (inclusive); timeout resets whenever bytes arrive."""
        last = time.monotonic()
        start = 0
        while True:
            idx = self.rx.find(token, start)
            if idx >= 0:
                out, self.rx = self.rx[:idx + len(token)], self.rx[idx + len(token):]
                return out
            start = max(0, len(self.rx) - len(token) + 1)
            if self._fill():
                last = time.monotonic()
            elif time.monotonic() - last > idle_timeout:
                raise TimeoutError("timeout waiting for %r (got %r)" % (token, self.rx[-60:]))

    def _read_exact(self, n, idle_timeout):
        last = time.monotonic()
        while len(self.rx) < n:
            if self._fill():
                last = time.monotonic()
            elif time.monotonic() - last > idle_timeout:
                raise TimeoutError("timeout reading %d bytes (got %d)" % (n, len(self.rx)))
        out, self.rx = self.rx[:n], self.rx[n:]
        return out

    def resync(self):
        time.sleep(0.2)
        self.rx = b""
        self.ser.reset_input_buffer()
        self.ser.write(b"\r")
        try:
            self._read_until(PROMPT, 1.0)
        except TimeoutError:
            pass
        self.rx = b""
        self.ser.reset_input_buffer()

    # commands -------------------------------------------------------------
    def command(self, cmd, timeout=3.0):
        self.ser.write((cmd + "\r").encode())
        resp = self._read_until(PROMPT, timeout)[:-len(PROMPT)]
        lines = resp.decode(errors="replace").replace("\r", "").split("\n")
        if lines and lines[0].strip() == cmd.strip():
            lines = lines[1:]
        return "\n".join(lines).strip()

    def scanraw(self, start, stop, points):
        cmd = "scanraw %d %d %d\r" % (int(start), int(stop), int(points))
        self.ser.write(cmd.encode())
        self._read_until(b"{", 5.0)
        data = self._read_exact(points * 3, 8.0)   # idle timeout: narrow RBW sweeps are slow
        self._read_until(PROMPT, 3.0)              # trailing '}' + prompt
        arr = np.frombuffer(data, dtype=np.uint8).reshape(points, 3)
        if not np.all(arr[:, 0] == ord("x")):
            raise ValueError("scanraw framing error")
        raw = arr[:, 1].astype(np.uint16) | (arr[:, 2].astype(np.uint16) << 8)
        return raw.astype(np.float64) / 32.0 - self.scale

    def capture(self, w, h):
        self.ser.write(b"capture\r")
        self._read_until(b"capture\r\n", 2.0)
        data = self._read_exact(w * h * 2, 5.0)
        self._read_until(PROMPT, 2.0)
        return data


# --------------------------------------------------------------------------
class SimTransport:
    """Synthetic tinySA for running the UI without hardware."""

    def __init__(self, model="ultra"):
        self.model = model
        self.scale = MODELS[model]["scale"]
        self.rbw = None
        self.att = None
        self.mode = "low input"
        self.output = False
        self.t0 = time.monotonic()
        rnd = random.Random(7)
        self.fm = [(rnd.uniform(88e6, 108e6), rnd.uniform(-75, -45)) for _ in range(9)]

    def close(self):
        pass

    def resync(self):
        pass

    def command(self, cmd, timeout=3.0):
        a = cmd.split()
        if not a:
            return ""
        c = a[0]
        if c == "version":
            return ("tinySA4_v1.4-SIM\nHW Version:SIMULATOR" if self.model == "ultra"
                    else "tinySA_v1.4-SIM\nHW Version:SIMULATOR")
        if c == "info":
            return "tinySA simulator (no hardware)"
        if c == "rbw" and len(a) > 1:
            self.rbw = None if a[1] == "auto" else float(a[1])
            return ""
        if c == "rbw":
            return "usage: rbw 0.2..850|auto"
        if c == "attenuate" and len(a) > 1:
            self.att = None if a[1] == "auto" else float(a[1])
            return ""
        if c == "mode" and len(a) > 2:
            self.mode = a[1] + " " + a[2]
            return ""
        if c == "output" and len(a) > 1:
            self.output = a[1] == "on"
            return ""
        if c in ("lna", "spur", "freq", "level", "pause", "resume", "sweep", "lna2"):
            return ""
        if c == "help":
            return "Commands: version info rbw attenuate mode output lna spur freq level scanraw capture"
        return "%s?" % c

    def _signals(self, t):
        sig = []
        # comb: FM carrier at 700.01 MHz, 20 kHz rate (Bessel sidebands, beta 2.4)
        J = [0.0025, 0.52, 0.43, 0.20, 0.064, 0.016, 0.003]
        for n in range(-6, 7):
            a = abs(J[abs(n)])
            if a > 1e-4:
                sig.append((700.01e6 + n * 20e3, -16 + 20 * math.log10(a / 0.52), 0))
        sig.append((100e6, -30 + 0.5 * math.sin(t), 0))
        sig.append((200e6, -58, 0))
        sig.append((300e6, -64, 0))
        sig.append((433.92e6 + 3e3 * math.sin(t * 0.3), -45, 0))
        if int(t * 2) % 3:
            sig.append((915.2e6, -52, 150e3))
        for f, a in self.fm:
            sig.append((f, a + 2 * math.sin(t * 0.7 + f / 1e7), 100e3))
        sig.append((1.5754e9, -95, 2e6))
        if self.model == "ultra":
            sig.append((2.437e9, -58 + 6 * math.sin(t * 3), 18e6))
            sig.append((1.2e9, -48, 0))
        return sig

    def scanraw(self, start, stop, points):
        t = time.monotonic() - self.t0
        f = np.linspace(start, stop, points)
        binw = (stop - start) / max(points - 1, 1)
        rbw_min = MODELS[self.model]["rbw"][0] * 1e3
        rbw = self.rbw * 1e3 if self.rbw else max(rbw_min, binw * 1.5)
        floor_dbm = -106 + 10 * math.log10(rbw / 3e3)
        if self.att:
            floor_dbm += self.att
        p = 10 ** (floor_dbm / 10) * np.random.gamma(3.0, 1 / 3.0, points)
        p += 10 ** (-20 / 10) * np.exp(-0.5 * (f / max(rbw, 30e3)) ** 2)  # 0 Hz LO feedthrough
        for fc, a, bw in self._signals(t):
            sigma = max(rbw, binw, bw) / 2.355
            amp = a if bw == 0 else a + 10 * math.log10(max(rbw, binw) / max(rbw, binw, bw))
            if self.att:
                pass
            p += 10 ** (amp / 10) * np.exp(-0.5 * ((f - fc) / sigma) ** 2)
        time.sleep(0.015 + points * 4e-5 + (0.2 if rbw < 3e3 else 0))
        return 10 * np.log10(p)

    def capture(self, w, h):
        y, x = np.mgrid[0:h, 0:w]
        r = (x * 31 // w).astype(np.uint16)
        g = (y * 63 // h).astype(np.uint16)
        b = np.full_like(r, 12)
        px = (r << 11) | (g << 5) | b
        return px.astype(">u2").tobytes()


# --------------------------------------------------------------------------
class DeviceWorker(QThread):
    sweep_ready = pyqtSignal(object, object, float)   # freqs, dBm, seconds
    connected = pyqtSignal(str, str)                   # model key, version text
    disconnected = pyqtSignal(str)                     # reason
    reply = pyqtSignal(int, object)                    # token, result
    status = pyqtSignal(str)

    def __init__(self, port, model_pref="auto", sim_model="ultra"):
        super().__init__()
        self.port = port
        self.model_pref = model_pref
        self.sim_model = sim_model
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.sweep = (0.0, 800e6, 450)
        self.running = True
        self.single = False
        self._stop = False
        self.model = "ultra"

    # called from GUI thread ------------------------------------------------
    def set_sweep(self, start, stop, points):
        with self.lock:
            self.sweep = (float(start), float(stop), int(points))

    def set_running(self, run):
        self.running = bool(run)

    def single_sweep(self):
        self.single = True

    def send(self, cmd, token=None):
        self.q.put(("cmd", cmd, token))

    def capture(self, token):
        self.q.put(("capture", None, token))

    def set_scale(self, scale):
        self.q.put(("scale", scale, None))

    def stop(self):
        self._stop = True

    # thread ------------------------------------------------------------------
    def run(self):
        tr = None
        try:
            if self.port == SIM_PORT:
                tr = SimTransport(self.sim_model if self.model_pref == "auto" else self.model_pref)
            else:
                tr = SerialTransport(self.port)
            ver = tr.command("version")
            if self.model_pref in MODELS:
                self.model = self.model_pref
            else:
                self.model = "ultra" if "tinySA4" in ver else "basic"
            tr.scale = MODELS[self.model]["scale"]
            self.connected.emit(self.model, ver)
        except Exception as e:
            if tr:
                tr.close()
            self.disconnected.emit("Connect failed: %s" % e)
            return

        reason = "Disconnected"
        errors = 0
        try:
            while not self._stop:
                try:
                    while True:
                        kind, arg, token = self.q.get_nowait()
                        if kind == "cmd":
                            r = tr.command(arg)
                        elif kind == "capture":
                            w, h = MODELS[self.model]["screen"]
                            r = (w, h, tr.capture(w, h))
                        elif kind == "scale":
                            tr.scale = float(arg)
                            r = None
                        if token is not None:
                            self.reply.emit(token, r)
                except queue.Empty:
                    pass

                if self.running or self.single:
                    with self.lock:
                        s, e, n = self.sweep
                    t0 = time.perf_counter()
                    try:
                        vals = tr.scanraw(s, e, n)
                        errors = 0
                    except (ValueError, TimeoutError) as ex:
                        errors += 1
                        self.status.emit("Sweep error: %s (resyncing)" % ex)
                        tr.resync()
                        if errors > 5:
                            raise
                        continue
                    dt = time.perf_counter() - t0
                    self.single = False
                    self.sweep_ready.emit(np.linspace(s, e, n), vals, dt)
                else:
                    time.sleep(0.02)
        except Exception as e:
            reason = "Connection lost: %s" % e
        finally:
            tr.close()
        self.disconnected.emit(reason)
