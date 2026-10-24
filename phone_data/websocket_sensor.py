"""Quick console check of the phone stream: every event, with its hardware timestamp and the measured rate.

  adb forward tcp:8080 tcp:8080
  python phone_data/websocket_sensor.py

The earlier version kept only the latest value per sensor and polled it at ~60 Hz, which discarded most
samples and replaced hardware timestamps with laptop arrival time. To record data, use phone_logger.py.
"""
import asyncio, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phone_data import stream


async def main():
    count, last, t_print = {}, {}, time.monotonic()
    async for name, t_ns, v, _ in stream.phone_events(types=(stream.ACC, stream.GYRO, stream.MAG)):
        count[name] = count.get(name, 0) + 1
        last[name] = (t_ns, v[:3])
        if time.monotonic() - t_print >= 1.0:
            dt = time.monotonic() - t_print
            t_print = time.monotonic()
            print(" | ".join(f"{n.upper()} {count[n] / dt:5.0f} Hz t={last[n][0] / 1e9:.4f} s "
                             f"[{last[n][1][0]:7.3f} {last[n][1][1]:7.3f} {last[n][1][2]:7.3f}]" for n in sorted(count)))
            count = {}


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
