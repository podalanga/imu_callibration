/* Fixed-point pipeline for the Tier 3 bonus (task D5): calibration correction, Mahony attitude,
 * strapdown propagation, stop detection with velocity reset and gyro-bias update. Integer only.
 * Gravity (Mahony) correction is applied only at stops, where it also absorbs the horizontal
 * accelerometer bias into tilt (the same combined estimate a ZUPT gives).
 *
 * Formats (Qm.n = value * 2^n in a signed integer):
 *   quaternion, unit vectors  Q2.30  int32
 *   angular rate [rad/s]      Q5.27  int32   (+-16 rad/s)
 *   specific force [m/s^2]    Q7.25  int32   (+-64 m/s^2)
 *   velocity [m/s], pos [m]   Q32.32 int64   (increments of a*dt are ~1e-5; Q16.16 would truncate them)
 *   time step                 integer microseconds
 * Mirrors nav/fixedpoint.py (float64 version of the same algorithm) for validation. */
#ifndef MAHONY_Q_H
#define MAHONY_Q_H
#include <stdint.h>

typedef struct {
    int32_t q[4];          /* Q2.30 */
    int32_t b[3];          /* Q5.27, MINUS the gyro bias estimate (Mahony convention) */
    int64_t v[3], p[3];    /* Q32.32 */
    int32_t kp_q16, ki_q16;           /* gains Q16.16 */
    int32_t g_q25;                    /* local gravity */
    int32_t still_count, still_need;
    int64_t ew;                       /* low-passed |w|^2 (Q54), 1/16 per sample */
    int32_t a_ref[3], da[3];          /* accel at the last stop, low-passed deviation from it (Q25) */
    int32_t init;
    int32_t w_still_q27, a_still_q25; /* stop thresholds */
    uint8_t still;
} mq_t;

void mq_init(mq_t *s, const int32_t q0[4], float kp, float ki, float g, float w_still, float a_still, int still_need);
void mq_step(mq_t *s, const int32_t acc_q25[3], const int32_t gyr_q27[3], int32_t dt_us);

static inline int32_t mq_from_float(float x, int frac) { return (int32_t)(x * (float)(1u << frac) + (x >= 0 ? 0.5f : -0.5f)); }
#endif
