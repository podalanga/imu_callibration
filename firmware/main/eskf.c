/* See eskf.h. All arithmetic is float32; compile with -Wdouble-promotion to catch slips. */
#include "eskf.h"
#include <math.h>
#include <string.h>

#define N ESKF_N

void quat_mul(const float a[4], const float b[4], float o[4])
{
    float w = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    float x = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    float y = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    float z = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
    o[0] = w; o[1] = x; o[2] = y; o[3] = z;
}

void quat_normalize(float q[4])
{
    float n = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    for (int i = 0; i < 4; i++) q[i] /= n;
}

void quat_exp(const float phi[3], float q[4])
{
    float a = sqrtf(phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2]);
    if (a < 1e-9f) {
        q[0] = 1.0f; q[1] = 0.5f*phi[0]; q[2] = 0.5f*phi[1]; q[3] = 0.5f*phi[2];
        quat_normalize(q);
        return;
    }
    float s = sinf(0.5f*a) / a;
    q[0] = cosf(0.5f*a); q[1] = s*phi[0]; q[2] = s*phi[1]; q[3] = s*phi[2];
}

void quat_to_R(const float q[4], float R[3][3])
{
    float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0][0] = 1.0f - 2.0f*(y*y + z*z); R[0][1] = 2.0f*(x*y - w*z);        R[0][2] = 2.0f*(x*z + w*y);
    R[1][0] = 2.0f*(x*y + w*z);        R[1][1] = 1.0f - 2.0f*(x*x + z*z); R[1][2] = 2.0f*(y*z - w*x);
    R[2][0] = 2.0f*(x*z - w*y);        R[2][1] = 2.0f*(y*z + w*x);        R[2][2] = 1.0f - 2.0f*(x*x + y*y);
}

static void skew(const float a[3], float S[3][3])
{
    S[0][0] = 0.0f;  S[0][1] = -a[2]; S[0][2] = a[1];
    S[1][0] = a[2];  S[1][1] = 0.0f;  S[1][2] = -a[0];
    S[2][0] = -a[1]; S[2][1] = a[0];  S[2][2] = 0.0f;
}

static void symmetrize(float P[N][N])
{
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            float s = 0.5f * (P[i][j] + P[j][i]);
            P[i][j] = s; P[j][i] = s;
        }
}

void eskf_init(eskf_t *f, const float q0[4], const float bg0[3], const float s[N],
               float Na, float Ng, float Kba, float Kbg, float g)
{
    memset(f, 0, sizeof(*f));
    memcpy(f->q, q0, sizeof(f->q));
    memcpy(f->bg, bg0, sizeof(f->bg));
    for (int i = 0; i < N; i++) f->P[i][i] = s[i] * s[i];
    f->Na = Na; f->Ng = Ng; f->Kba = Kba; f->Kbg = Kbg; f->g = g;
}

/* scratch buffers: static to keep the task stack small */
static float F[N][N], T1[N][N], T2[N][N];

void eskf_predict(eskf_t *f, const float f_m[3], const float w_m[3], float dt)
{
    float R[3][3], fb[3], w[3], a[3], S[3][3], dq[4], qn[4], Rdq[3][3];
    quat_to_R(f->q, R);
    for (int i = 0; i < 3; i++) { fb[i] = f_m[i] - f->ba[i]; w[i] = (w_m[i] - f->bg[i]) * dt; }
    for (int i = 0; i < 3; i++) a[i] = R[i][0]*fb[0] + R[i][1]*fb[1] + R[i][2]*fb[2];
    a[2] -= f->g;
    for (int i = 0; i < 3; i++) {
        f->p[i] += f->v[i]*dt + 0.5f*a[i]*dt*dt;
        f->v[i] += a[i]*dt;
    }
    quat_exp(w, dq);
    quat_mul(f->q, dq, qn);
    memcpy(f->q, qn, sizeof(qn));
    quat_normalize(f->q);

    /* F = I + blocks (Chapter 12) */
    memset(F, 0, sizeof(F));
    for (int i = 0; i < N; i++) F[i][i] = 1.0f;
    skew(fb, S);
    quat_to_R(dq, Rdq);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float RS = R[i][0]*S[0][j] + R[i][1]*S[1][j] + R[i][2]*S[2][j];
            F[i][3 + j] = (i == j) ? dt : 0.0f;
            F[3 + i][6 + j] = -RS * dt;
            F[3 + i][9 + j] = -R[i][j] * dt;
            F[6 + i][6 + j] = Rdq[j][i];                  /* transpose */
            F[6 + i][12 + j] = (i == j) ? -dt : 0.0f;
        }
    /* P = F P F^T + Qd */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++) s += F[i][k] * f->P[k][j];
            T1[i][j] = s;
        }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++) s += T1[i][k] * F[j][k];
            f->P[i][j] = s;
        }
    float qv = f->Na*f->Na*dt, qt = f->Ng*f->Ng*dt, qa = f->Kba*f->Kba*dt, qg = f->Kbg*f->Kbg*dt;
    for (int i = 0; i < 3; i++) {
        f->P[3 + i][3 + i] += qv; f->P[6 + i][6 + i] += qt;
        f->P[9 + i][9 + i] += qa; f->P[12 + i][12 + i] += qg;
    }
}

/* invert a symmetric positive m x m matrix, m <= 3 */
static int inv_small(int m, const float A[3][3], float B[3][3])
{
    if (m == 1) { if (A[0][0] <= 0.0f) return 0; B[0][0] = 1.0f / A[0][0]; return 1; }
    if (m == 2) {
        float d = A[0][0]*A[1][1] - A[0][1]*A[1][0];
        if (d <= 0.0f) return 0;
        B[0][0] = A[1][1]/d; B[1][1] = A[0][0]/d; B[0][1] = -A[0][1]/d; B[1][0] = -A[1][0]/d;
        return 1;
    }
    float c00 = A[1][1]*A[2][2] - A[1][2]*A[2][1];
    float c01 = A[1][2]*A[2][0] - A[1][0]*A[2][2];
    float c02 = A[1][0]*A[2][1] - A[1][1]*A[2][0];
    float d = A[0][0]*c00 + A[0][1]*c01 + A[0][2]*c02;
    if (d <= 0.0f) return 0;
    B[0][0] = c00/d; B[1][0] = c01/d; B[2][0] = c02/d;
    B[0][1] = (A[0][2]*A[2][1] - A[0][1]*A[2][2])/d;
    B[1][1] = (A[0][0]*A[2][2] - A[0][2]*A[2][0])/d;
    B[2][1] = (A[0][1]*A[2][0] - A[0][0]*A[2][1])/d;
    B[0][2] = (A[0][1]*A[1][2] - A[0][2]*A[1][1])/d;
    B[1][2] = (A[0][2]*A[1][0] - A[0][0]*A[1][2])/d;
    B[2][2] = (A[0][0]*A[1][1] - A[0][1]*A[1][0])/d;
    return 1;
}

static void inject(eskf_t *f, const float dx[N])
{
    float dq[4], qn[4];
    for (int i = 0; i < 3; i++) {
        f->p[i] += dx[i]; f->v[i] += dx[3 + i]; f->ba[i] += dx[9 + i]; f->bg[i] += dx[12 + i];
    }
    quat_exp(&dx[6], dq);
    quat_mul(f->q, dq, qn);
    memcpy(f->q, qn, sizeof(qn));
    quat_normalize(f->q);
    /* reset: P = G P G^T, G = I except G[6:9,6:9] = I - skew(dtheta/2) */
    float h[3] = {0.5f*dx[6], 0.5f*dx[7], 0.5f*dx[8]}, S[3][3], Gb[3][3];
    skew(h, S);
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Gb[i][j] = (i == j ? 1.0f : 0.0f) - S[i][j];
    for (int c = 0; c < N; c++) {               /* rows 6..8: G P */
        float r[3];
        for (int i = 0; i < 3; i++) r[i] = Gb[i][0]*f->P[6][c] + Gb[i][1]*f->P[7][c] + Gb[i][2]*f->P[8][c];
        for (int i = 0; i < 3; i++) f->P[6 + i][c] = r[i];
    }
    for (int rr = 0; rr < N; rr++) {            /* cols 6..8: (G P) G^T */
        float r[3];
        for (int i = 0; i < 3; i++) r[i] = f->P[rr][6]*Gb[i][0] + f->P[rr][7]*Gb[i][1] + f->P[rr][8]*Gb[i][2];
        for (int i = 0; i < 3; i++) f->P[rr][6 + i] = r[i];
    }
}

int eskf_update(eskf_t *f, int m, const float *nu, const float *H, const float *Rz, float gate, float *nis_out)
{
    float PHt[N][3], S[3][3], Si[3][3], K[N][3], dx[N];
    for (int i = 0; i < N; i++)
        for (int a = 0; a < m; a++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++) s += f->P[i][k] * H[a*N + k];
            PHt[i][a] = s;
        }
    for (int a = 0; a < m; a++)
        for (int b = 0; b < m; b++) {
            float s = Rz[a*m + b];
            for (int k = 0; k < N; k++) s += H[a*N + k] * PHt[k][b];
            S[a][b] = s;
        }
    if (!inv_small(m, S, Si)) return 0;
    float nis = 0.0f;
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) nis += nu[a] * Si[a][b] * nu[b];
    if (nis_out) *nis_out = nis;
    if (gate > 0.0f && nis > gate) return 0;
    for (int i = 0; i < N; i++)
        for (int b = 0; b < m; b++) {
            float s = 0.0f;
            for (int a = 0; a < m; a++) s += PHt[i][a] * Si[a][b];
            K[i][b] = s;
        }
    for (int i = 0; i < N; i++) {
        float s = 0.0f;
        for (int a = 0; a < m; a++) s += K[i][a] * nu[a];
        dx[i] = s;
    }
    /* Joseph form: P = (I-KH) P (I-KH)^T + K R K^T */
    for (int i = 0; i < N; i++)                  /* T2 = I - K H */
        for (int j = 0; j < N; j++) {
            float s = (i == j) ? 1.0f : 0.0f;
            for (int a = 0; a < m; a++) s -= K[i][a] * H[a*N + j];
            T2[i][j] = s;
        }
    for (int i = 0; i < N; i++)                  /* T1 = T2 P */
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++) s += T2[i][k] * f->P[k][j];
            T1[i][j] = s;
        }
    for (int i = 0; i < N; i++)                  /* P = T1 T2^T + K R K^T */
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++) s += T1[i][k] * T2[j][k];
            for (int a = 0; a < m; a++)
                for (int b = 0; b < m; b++) s += K[i][a] * Rz[a*m + b] * K[j][b];
            f->P[i][j] = s;
        }
    symmetrize(f->P);
    inject(f, dx);
    return 1;
}

int eskf_zupt(eskf_t *f, float sigma, float gate, float *nis)
{
    float H[3*N] = {0}, nu[3], Rz[9] = {0};
    for (int i = 0; i < 3; i++) { H[i*N + 3 + i] = 1.0f; nu[i] = -f->v[i]; Rz[i*3 + i] = sigma*sigma; }
    return eskf_update(f, 3, nu, H, Rz, gate, nis);
}

int eskf_zaru(eskf_t *f, const float w_m[3], float sigma, float gate, float *nis)
{
    float H[3*N] = {0}, nu[3], Rz[9] = {0};
    for (int i = 0; i < 3; i++) { H[i*N + 12 + i] = 1.0f; nu[i] = w_m[i] - f->bg[i]; Rz[i*3 + i] = sigma*sigma; }
    return eskf_update(f, 3, nu, H, Rz, gate, nis);
}

int eskf_nhc(eskf_t *f, float sigma, float gate, float *nis)
{
    float R[3][3], vb[3], S[3][3], H[2*N] = {0}, nu[2], Rz[4] = {sigma*sigma, 0, 0, sigma*sigma};
    quat_to_R(f->q, R);
    for (int i = 0; i < 3; i++) vb[i] = R[0][i]*f->v[0] + R[1][i]*f->v[1] + R[2][i]*f->v[2];
    skew(vb, S);
    for (int a = 0; a < 2; a++) {                /* rows y, z of body velocity */
        int r = a + 1;
        for (int j = 0; j < 3; j++) {
            H[a*N + 3 + j] = R[j][r];                /* (R^T)[r][j] */
            H[a*N + 6 + j] = S[r][j];
        }
        nu[a] = -vb[r];
    }
    return eskf_update(f, 2, nu, H, Rz, gate, nis);
}

int eskf_line(eskf_t *f, float sigma, float y0, float gate, float *nis)
{
    float H[N] = {0}, nu = y0 - f->p[1], Rz = sigma*sigma;
    H[1] = 1.0f;
    return eskf_update(f, 1, &nu, H, &Rz, gate, nis);
}
