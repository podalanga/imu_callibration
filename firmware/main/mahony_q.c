/* See mahony_q.h. Every product is formed in int64 and shifted back with rounding. */
#include "mahony_q.h"

#define US_PER_S 1000000LL

static inline int32_t mul30(int32_t a, int32_t b) { return (int32_t)(((int64_t)a * b + (1LL << 29)) >> 30); }

static uint64_t isqrt64(uint64_t x)
{
    uint64_t r = 0, bit = 1ULL << 62;
    while (bit > x) bit >>= 2;
    while (bit) {
        if (x >= r + bit) { x -= r + bit; r = (r >> 1) + bit; } else r >>= 1;
        bit >>= 2;
    }
    return r;
}

static void normalize(int32_t q[4])
{
    uint64_t n2 = 0;
    for (int i = 0; i < 4; i++) n2 += (uint64_t)((int64_t)q[i] * q[i]);       /* Q60 */
    int64_t n = (int64_t)isqrt64(n2);                                         /* Q30 */
    if (n == 0) return;
    for (int i = 0; i < 4; i++) q[i] = (int32_t)(((int64_t)q[i] << 30) / n);
}

void mq_init(mq_t *s, const int32_t q0[4], float kp, float ki, float g, float w_still, float a_still, int still_need)
{
    for (int i = 0; i < 4; i++) s->q[i] = q0[i];
    for (int i = 0; i < 3; i++) { s->b[i] = 0; s->v[i] = 0; s->p[i] = 0; }
    s->kp_q16 = mq_from_float(kp, 16); s->ki_q16 = mq_from_float(ki, 16);
    s->g_q25 = mq_from_float(g, 25);
    s->w_still_q27 = mq_from_float(w_still, 27); s->a_still_q25 = mq_from_float(a_still, 25);
    s->still_count = 0; s->still_need = still_need; s->still = 0; s->ew = 0; s->init = 0;
    for (int i = 0; i < 3; i++) { s->a_ref[i] = 0; s->da[i] = 0; }
}

void mq_step(mq_t *s, const int32_t a[3], const int32_t w[3], int32_t dt_us)
{
    const int32_t *q = s->q;
    /* ---- Mahony correction from gravity */
    int64_t a2 = 0;
    for (int i = 0; i < 3; i++) a2 += (int64_t)a[i] * a[i];                   /* Q50 */
    int32_t an = (int32_t)isqrt64((uint64_t)a2);                              /* Q25 */
    int32_t e[3] = {0, 0, 0};
    int32_t dev = an - s->g_q25;
    if (dev < 0) dev = -dev;
    /* gravity correction only while stopped: during motion the trolley accelerates and the
     * accelerometer no longer points along gravity (false tilt a/g). Gyro-only between stops. */
    if (s->still && an > 0 && dev < s->g_q25 / 20) {
        int32_t u[3], vh[3];
        for (int i = 0; i < 3; i++) u[i] = (int32_t)(((int64_t)a[i] << 30) / an);   /* Q30 unit */
        vh[0] = 2 * (mul30(q[1], q[3]) - mul30(q[0], q[2]));
        vh[1] = 2 * (mul30(q[0], q[1]) + mul30(q[2], q[3]));
        vh[2] = mul30(q[0], q[0]) - mul30(q[1], q[1]) - mul30(q[2], q[2]) + mul30(q[3], q[3]);
        e[0] = mul30(u[1], vh[2]) - mul30(u[2], vh[1]);
        e[1] = mul30(u[2], vh[0]) - mul30(u[0], vh[2]);
        e[2] = mul30(u[0], vh[1]) - mul30(u[1], vh[0]);
    }
    int32_t wc[3], dth[3];
    for (int i = 0; i < 3; i++) {
        s->b[i] += (int32_t)((((int64_t)s->ki_q16 * e[i]) * dt_us / US_PER_S) >> 19);   /* b = -bias: db/dt = +ki e */
        wc[i] = w[i] + s->b[i] + (int32_t)(((int64_t)s->kp_q16 * e[i]) >> 19);
        dth[i] = (int32_t)((int64_t)wc[i] * dt_us * 8 / US_PER_S);          /* Q27 rad/s * s -> Q30 rad */
    }
    /* ---- strapdown, using the attitude at the start of the step */
    int32_t R[3][3];
    R[0][0] = (1 << 30) - 2 * (mul30(q[2], q[2]) + mul30(q[3], q[3]));
    R[0][1] = 2 * (mul30(q[1], q[2]) - mul30(q[0], q[3]));
    R[0][2] = 2 * (mul30(q[1], q[3]) + mul30(q[0], q[2]));
    R[1][0] = 2 * (mul30(q[1], q[2]) + mul30(q[0], q[3]));
    R[1][1] = (1 << 30) - 2 * (mul30(q[1], q[1]) + mul30(q[3], q[3]));
    R[1][2] = 2 * (mul30(q[2], q[3]) - mul30(q[0], q[1]));
    R[2][0] = 2 * (mul30(q[1], q[3]) - mul30(q[0], q[2]));
    R[2][1] = 2 * (mul30(q[2], q[3]) + mul30(q[0], q[1]));
    R[2][2] = (1 << 30) - 2 * (mul30(q[1], q[1]) + mul30(q[2], q[2]));
    for (int i = 0; i < 3; i++) {
        int64_t aw = 0;
        for (int j = 0; j < 3; j++) aw += (int64_t)R[i][j] * a[j];            /* Q55 */
        int32_t awq = (int32_t)((aw + (1LL << 29)) >> 30);                   /* Q25 */
        if (i == 2) awq -= s->g_q25;
        s->p[i] += s->v[i] * dt_us / US_PER_S;                                /* Q32 */
        s->v[i] += (int64_t)awq * dt_us * 128 / US_PER_S;                     /* Q25 -> Q32 */
    }
    /* ---- attitude: q += 0.5 q (x) [0, dtheta], then normalise */
    int32_t dq[4];
    dq[0] = -(mul30(q[1], dth[0]) + mul30(q[2], dth[1]) + mul30(q[3], dth[2]));
    dq[1] = mul30(q[0], dth[0]) + mul30(q[2], dth[2]) - mul30(q[3], dth[1]);
    dq[2] = mul30(q[0], dth[1]) - mul30(q[1], dth[2]) + mul30(q[3], dth[0]);
    dq[3] = mul30(q[0], dth[2]) + mul30(q[1], dth[1]) - mul30(q[2], dth[0]);
    for (int i = 0; i < 4; i++) s->q[i] += dq[i] / 2;
    normalize(s->q);
    /* ---- stop detection, velocity reset (ZUPT) and gyro-bias update (ZARU) */
    /* stop detector: gyro energy, and deviation of the accel vector from its value at the last
     * stop (sees horizontal acceleration, which |a| - g cannot) */
    if (!s->init) { for (int i = 0; i < 3; i++) s->a_ref[i] = a[i]; s->init = 1; }
    int64_t w2 = 0, ea = 0;
    for (int i = 0; i < 3; i++) {
        int32_t x = w[i] + s->b[i];
        w2 += (int64_t)x * x;                                                     /* Q54 */
        s->da[i] += (a[i] - s->a_ref[i] - s->da[i]) / 16;
        ea += (int64_t)s->da[i] * s->da[i];                                       /* Q50 */
    }
    s->ew += (w2 - s->ew) / 16;
    int64_t wt = (int64_t)s->w_still_q27 * s->w_still_q27;
    int64_t at = (int64_t)s->a_still_q25 * s->a_still_q25;
    s->still_count = (s->ew < wt && ea < at) ? s->still_count + 1 : 0;
    s->still = s->still_count >= s->still_need;
    if (s->still) {
        for (int i = 0; i < 3; i++) {
            s->v[i] = 0;
            s->b[i] += (-w[i] - s->b[i]) / 1024;     /* b -> -bias, tau ~ 5 s: slow, so the last few
                                                         samples before motion is detected barely matter */
            s->a_ref[i] += (a[i] - s->a_ref[i]) / 64;
        }
    }
}
