/* 15-state error-state Kalman filter, float32. Mirrors nav/eskf.py line by line.
 * Portable C99: no ESP-IDF dependency, so the host can compile it and compare with Python.
 * Error state: dp(0..2) dv(3..5) dtheta(6..8) dba(9..11) dbg(12..14).
 * Quaternion: Hamilton, scalar first, body -> world; local (right) attitude error. */
#ifndef ESKF_H
#define ESKF_H
#include <stdint.h>

#define ESKF_N 15

typedef struct {
    float p[3], v[3], q[4], ba[3], bg[3];
    float P[ESKF_N][ESKF_N];
    float Na, Ng, Kba, Kbg;   /* noise densities and bias random walks */
    float g;
} eskf_t;

void eskf_init(eskf_t *f, const float q0[4], const float bg0[3], const float P0_sigma[ESKF_N],
               float Na, float Ng, float Kba, float Kbg, float g);
void eskf_predict(eskf_t *f, const float f_m[3], const float w_m[3], float dt);
/* Generic update, m <= 3. nu[m], H[m][15] row-major, Rz[m][m]. Returns 1 if applied, 0 if gated.
 * nis_out receives the normalised innovation squared. gate <= 0 disables gating. */
int eskf_update(eskf_t *f, int m, const float *nu, const float *H, const float *Rz, float gate, float *nis_out);
int eskf_zupt(eskf_t *f, float sigma, float gate, float *nis);
int eskf_zaru(eskf_t *f, const float w_m[3], float sigma, float gate, float *nis);
int eskf_nhc(eskf_t *f, float sigma, float gate, float *nis);
int eskf_line(eskf_t *f, float sigma, float y0, float gate, float *nis);

/* small helpers, also used by nav.c */
void quat_mul(const float a[4], const float b[4], float out[4]);
void quat_exp(const float phi[3], float q[4]);
void quat_to_R(const float q[4], float R[3][3]);
void quat_normalize(float q[4]);

#endif
