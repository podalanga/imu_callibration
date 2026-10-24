/* Binary UART protocol shared by firmware and tools/protocol.py. Little-endian, packed.
 * Frame: 0xA5 0x5A | type u8 | len u8 | payload[len] | crc16 (CCITT-FALSE over type,len,payload) */
#ifndef PROTOCOL_H
#define PROTOCOL_H
#include <stdint.h>

#define SYNC0 0xA5
#define SYNC1 0x5A

enum {
    T_RAW = 0x01,      /* device -> host: one MPU6050 sample */
    T_MARK = 0x02,     /* device -> host: marker button press */
    T_EST = 0x03,      /* device -> host: filter output */
    T_STAT = 0x04,     /* device -> host: counters, once per second */
    T_CTRL = 0x10,     /* host -> device: command */
    T_REPLAY = 0x11,   /* host -> device: replay sample (HIL) */
};

enum { CMD_LIVE = 0, CMD_REPLAY = 1, CMD_RESET = 2 };

typedef struct __attribute__((packed)) {
    uint32_t seq; int64_t t_us;
    int16_t ax, ay, az, temp, gx, gy, gz;      /* raw counts, MPU6050 register order */
} pl_raw_t;                                    /* 26 bytes */

typedef struct __attribute__((packed)) { int64_t t_us; uint16_t id; } pl_mark_t;     /* 10 bytes */

typedef struct __attribute__((packed)) {
    int64_t t_us;
    float p[3], v[3], q[4], ba[3], bg[3];
    uint32_t cycles;                           /* CPU cycles of the last nav_step */
    uint8_t still, flags;
} pl_est_t;                                    /* 78 bytes */

typedef struct __attribute__((packed)) {
    uint32_t samples, fifo_overflows, queue_drops, seq_gaps, max_cycles, gated, nan_resets, cpu_mhz;
} pl_stat_t;                                   /* 32 bytes */

typedef struct __attribute__((packed)) { uint8_t cmd; } pl_ctrl_t;

typedef struct __attribute__((packed)) { int64_t t_us; float acc[3], gyro[3]; } pl_replay_t;   /* 32 bytes, SI */

static inline uint16_t crc16_ccitt(const uint8_t *d, int n)
{
    uint16_t c = 0xFFFF;
    for (int i = 0; i < n; i++) {
        c ^= (uint16_t)(d[i] << 8);
        for (int b = 0; b < 8; b++) c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x1021) : (uint16_t)(c << 1);
    }
    return c;
}

#endif
