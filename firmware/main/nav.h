/* Causal navigation pipeline: calibration, initial alignment, ESKF, stop detector, aiding.
 * Mirrors nav/runner.py. Portable C99 (host-testable). */
#ifndef NAV_H
#define NAV_H
#include <stdint.h>
#include "eskf.h"

#define NAV_MAX_WIN 64

typedef struct {
    float Ta[3][3], Ka[3], ba[3];     /* a_O = Ta diag(Ka) (a_S + ba) */
    float Tg[3][3], Kg[3], bg[3];     /* w_O = Tg diag(Kg) (w_S + bg) */
} nav_calib_t;

typedef struct {
    float Na, Ng, Kba, Kbg, g;
    float t_init, fs_nominal;
    int win; float sigma_a, sigma_w, gamma, min_still;
    int use_zupt; float zupt_sigma, zaru_sigma; int zupt_every;
    int use_nhc; float nhc_sigma; int nhc_every; float nhc_max_wz;
    int use_line; float line_sigma; int line_every;
    int gate;
} nav_config_t;

typedef struct {
    uint32_t zupt, zaru, nhc, line;   /* applied */
    uint32_t gated;                   /* rejected by the chi-square gate */
    uint32_t nan_resets;
} nav_counters_t;

typedef struct {
    nav_config_t cfg;
    nav_calib_t cal;
    eskf_t f;
    int64_t t_prev_us;
    uint32_t k;                       /* absolute sample index, as in the Python loop */
    uint32_t n0;                      /* samples used for alignment */
    float acc_sum[3], gyr_sum[3];     /* alignment accumulators */
    int aligned;
    float buf_f[NAV_MAX_WIN][3], buf_w[NAV_MAX_WIN][3];
    uint32_t det_n, still_count, still_need, k_still;
    int still;
    nav_counters_t cnt;
    eskf_t last_good;
} nav_t;

void nav_default_config(nav_config_t *c);
void nav_identity_calib(nav_calib_t *c);
void nav_init(nav_t *n, const nav_config_t *cfg, const nav_calib_t *cal);
/* One raw sample in SI units (sensor frame). Returns 1 once the filter is running. */
int nav_step(nav_t *n, int64_t t_us, const float acc_raw[3], const float gyr_raw[3]);

#endif
