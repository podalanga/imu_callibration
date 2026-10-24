"""Frame codec for firmware/main/protocol.h. Frame: A5 5A | type | len | payload | crc16 (LE)."""
import struct

SYNC = b"\xA5\x5A"
T_RAW, T_MARK, T_EST, T_STAT, T_CTRL, T_REPLAY = 0x01, 0x02, 0x03, 0x04, 0x10, 0x11
CMD_LIVE, CMD_REPLAY, CMD_RESET = 0, 1, 2

FMT = {
    T_RAW: struct.Struct("<Iq7h"),                 # seq, t_us, ax ay az temp gx gy gz
    T_MARK: struct.Struct("<qH"),                  # t_us, id
    T_EST: struct.Struct("<q16fIBB"),              # t_us, p3 v3 q4 ba3 bg3, cycles, still, flags
    T_STAT: struct.Struct("<8I"),                  # samples, fifo_ovf, queue_drops, seq_gaps, max_cycles, gated, nan, mhz
}
ACC_LSB_PER_G, GYRO_LSB_PER_DPS, G0 = 8192.0, 65.5, 9.80665


def crc16(data: bytes) -> int:
    c = 0xFFFF
    for b in data:
        c ^= b << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
    return c


def encode(ftype: int, payload: bytes) -> bytes:
    body = bytes([ftype, len(payload)]) + payload
    return SYNC + body + struct.pack("<H", crc16(body))


def encode_ctrl(cmd):
    return encode(T_CTRL, bytes([cmd]))


def encode_replay(t_us, acc, gyro):
    return encode(T_REPLAY, struct.pack("<q6f", int(t_us), *acc, *gyro))


class Parser:
    """Feed bytes, get (type, fields) tuples. Resynchronises on errors."""
    def __init__(self):
        self.buf = bytearray()
        self.crc_errors = 0

    def feed(self, data: bytes):
        self.buf += data
        out = []
        while True:
            i = self.buf.find(SYNC)
            if i < 0:
                del self.buf[:-1]
                return out
            del self.buf[:i]
            if len(self.buf) < 4:
                return out
            ftype, n = self.buf[2], self.buf[3]
            if len(self.buf) < 6 + n:
                return out
            body = bytes(self.buf[2:4 + n])
            crc = struct.unpack("<H", self.buf[4 + n:6 + n])[0]
            if crc != crc16(body) or ftype not in FMT or FMT[ftype].size != n:
                self.crc_errors += 1
                del self.buf[:2]
                continue
            out.append((ftype, FMT[ftype].unpack(body[2:])))
            del self.buf[:6 + n]


def raw_to_si(fields):
    """T_RAW fields -> (seq, t_us, gyro rad/s [3], acc m/s^2 [3], temp degC)."""
    seq, t_us, ax, ay, az, temp, gx, gy, gz = fields
    acc = [v / ACC_LSB_PER_G * G0 for v in (ax, ay, az)]
    gyro = [v / GYRO_LSB_PER_DPS * 3.141592653589793 / 180 for v in (gx, gy, gz)]
    return seq, t_us, gyro, acc, temp / 340.0 + 36.53
