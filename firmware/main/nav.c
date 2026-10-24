/* See nav.h. Same step order as nav/runner.py so HIL replay can be compared sample by sample. */
#include "nav.h"
#include <math.h>
#include <string.h>

#define DEG2RAD 0.017453292519943295f

void nav_default_config(nav_config_t *c)
{
    c->Na = 400e-6f * 9.80665f; c->Ng = 0.005f * DEG2RAD; c->Kba = 1e-4f; c->Kbg = 2e-5f; c->g = 9.80665f;
    c->t_init = 8.0f; c->fs_nominal = 200.0f;
    c->win = 20; c->sigma_a = 0.05f; c->sigma_w = 0.005f; c->gamma = 5.0f; c->min_still = 0.3f;
    c->use_zupt = 1; c->zupt_sigma = 0.01f; c->zaru_sigma = 0.002f; c->zupt_every = 4;
    c->use_nhc = 1; c->nhc_sigma = 0.05f; c->nhc_every = 10; c->nhc_max_wz = 0.3f;
    c->use_line = 0; c->line_sigma = 0.05f; c->line_every = 20;
    c->gate = 1;
}

void nav_identity_calib(nav_calib_t *c)
{
    memset(c, 0, sizeof(*c));
    for (int i = 0; i < 3; i++) { c->Ta[i][i] = 1.0f; c->Tg[i][i] = 1.0f; c->Ka[i] = 1.0f; c->Kg[i] = 1.0f; }
}

void nav_init(nav_t *n, const nav_config_t *cfg, const nav_calib_t *cal)
{
    memset(n, 0, sizeof(*n));
    n->cfg = *cfg;
    n->cal = *cal;
    n->n0 = (uint32_t)lrintf(cfg->t_init * cfg->fs_nominal);
    n->still_need = (uint32_t)lrintf(cfg->min_still * cfg->fs_nominal);
    if (n->cfg.win > NAV_MAX_WIN) n->cfg.win = NAV_MAX_WIN;
}

static void apply(const float T[3][3], const float K[3], const float b[3], const float in[3], float out[3])
{
    float u[3];
    for (int i = 0; i < 3; i++) u[i] = K[i] * (in[i] + b[i]);
    for (int i = 0; i < 3; i++) out[i] = T[i][0]*u[0] + T[i][1]*u[1] + T[i][2]*u[2];
}

static float chi2(int m) { return m == 1 ? 6.63f : (m == 2 ? 9.21f : 11.34f); }

static int detector_step(nav_t *n, const float f[3], const float w[3])
{
    const nav_config_t *c = &n->cfg;
    int i = (int)(n->det_n % (uint32_t)c->win);
    memcpy(n->buf_f[i], f, 3 * sizeof(float));
    memcpy(n->buf_w[i], w, 3 * sizeof(float));
    n->det_n++;
    if (n->det_n < (uint32_t)c->win) return 0;
    float m[3] = {0, 0, 0};
    for (int k = 0; k < c->win; k++) for (int j = 0; j < 3; j++) m[j] += n->buf_f[k][j];
    for (int j = 0; j < 3; j++) m[j] /= (float)c->win;
    float mn = sqrtf(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
    float T = 0.0f;
    for (int k = 0; k < c->win; k++) {
        float da = 0.0f, dw = 0.0f;
        for (int j = 0; j < 3; j++) {
            float d = n->buf_f[k][j] - c->g * m[j] / mn;
            da += d * d;
            dw += n->buf_w[k][j] * n->buf_w[k][j];
        }
        T += da / (c->sigma_a * c->sigma_a) + dw / (c->sigma_w * c->sigma_w);
    }
    T /= (float)c->win;
    n->still_count = (T < c->gamma) ? n->still_count + 1 : 0;
    return n->still_count >= n->still_need;
}

static int finite_state(const eskf_t *f)
{
    for (int i = 0; i < 3; i++) if (!isfinite(f->p[i]) || !isfinite(f->v[i])) return 0;
    for (int i = 0; i < ESKF_N; i++) if (!(f->P[i][i] > 0.0f) || !isfinite(f->P[i][i])) return 0;
    return 1;
}

int nav_step(nav_t *n, int64_t t_us, const float acc_raw[3], const float gyr_raw[3])
{
    const nav_config_t *c = &n->cfg;
    float f[3], w[3], nis;
    apply(n->cal.Ta, n->cal.Ka, n->cal.ba, acc_raw, f);
    apply(n->cal.Tg, n->cal.Kg, n->cal.bg, gyr_raw, w);
    uint32_t k = n->k++;

    if (k < n->n0) {                               /* initial static alignment */
        for (int j = 0; j < 3; j++) { n->acc_sum[j] += f[j]; n->gyr_sum[j] += w[j]; }
        n->t_prev_us = t_us;
        if (k + 1 == n->n0) {
            float fm[3], bg[3];
            for (int j = 0; j < 3; j++) { fm[j] = n->acc_sum[j] / (float)n->n0; bg[j] = n->gyr_sum[j] / (float)n->n0; }
            float roll = atan2f(fm[1], fm[2]);
            float pitch = atan2f(-fm[0], sqrtf(fm[1]*fm[1] + fm[2]*fm[2]));
            float qy[4] = {cosf(0.5f*pitch), 0, sinf(0.5f*pitch), 0};
            float qx[4] = {cosf(0.5f*roll), sinf(0.5f*roll), 0, 0};
            float q0[4];
            quat_mul(qy, qx, q0);                      /* yaw = 0 by start-fixture definition */
            const float d = DEG2RAD;
            float s[ESKF_N] = {0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.2f*d, 0.2f*d, 0.5f*d,
                               0.02f, 0.02f, 0.02f, 0.01f*d, 0.01f*d, 0.01f*d};
            eskf_init(&n->f, q0, bg, s, c->Na, c->Ng, c->Kba, c->Kbg, c->g);
            n->last_good = n->f;
            n->aligned = 1;
        }
        return 0;
    }

    float dt = (float)(t_us - n->t_prev_us) * 1e-6f;
    n->t_prev_us = t_us;
    eskf_predict(&n->f, f, w, dt);
    float gate = c->gate ? 1.0f : -1.0f;

    n->still = detector_step(n, f, w);
    if (n->still && c->use_zupt) {
        n->k_still++;
        if (n->k_still % (uint32_t)c->zupt_every == 0) {
            if (eskf_zupt(&n->f, c->zupt_sigma, gate * chi2(3), &nis)) n->cnt.zupt++; else n->cnt.gated++;
            if (eskf_zaru(&n->f, w, c->zaru_sigma, gate * chi2(3), &nis)) n->cnt.zaru++; else n->cnt.gated++;
        }
    } else {
        n->k_still = 0;
        if (c->use_nhc && k % (uint32_t)c->nhc_every == 0 && fabsf(w[2] - n->f.bg[2]) < c->nhc_max_wz) {
            if (eskf_nhc(&n->f, c->nhc_sigma, gate * chi2(2), &nis)) n->cnt.nhc++; else n->cnt.gated++;
        }
    }
    if (c->use_line && k % (uint32_t)c->line_every == 0) {
        if (eskf_line(&n->f, c->line_sigma, 0.0f, gate * chi2(1), &nis)) n->cnt.line++; else n->cnt.gated++;
    }

    if (!finite_state(&n->f)) { n->f = n->last_good; n->cnt.nan_resets++; }
    else if ((k & 255u) == 0) n->last_good = n->f;
    return 1;
}
