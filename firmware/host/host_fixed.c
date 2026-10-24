/* Host build of the fixed-point pipeline: reads imu.csv (SI units), converts to Q formats,
 * runs mahony_q.c, writes t_us,px,py,pz,vx,vy,vz,qw,qx,qy,qz,still.
 *   ./host_fixed imu.csv est_fixed.csv qw qx qy qz        (initial attitude from alignment) */
#include <stdio.h>
#include <stdlib.h>
#include "../main/mahony_q.h"

int main(int argc, char **argv)
{
    if (argc < 7) { fprintf(stderr, "usage: %s imu.csv out.csv qw qx qy qz\n", argv[0]); return 1; }
    int32_t q0[4];
    for (int i = 0; i < 4; i++) q0[i] = mq_from_float((float)atof(argv[3 + i]), 30);
    mq_t s;
    mq_init(&s, q0, 1.0f, 0.02f, 9.80665f, 0.004f, 0.04f, 60);
    FILE *in = fopen(argv[1], "r"), *out = fopen(argv[2], "w");
    char line[512];
    if (!in || !out || !fgets(line, sizeof line, in)) return 1;
    fprintf(out, "t_us,px,py,pz,vx,vy,vz,qw,qx,qy,qz,still\n");
    long long t, seq, t_prev = -1; double g[3], a[3], temp;
    while (fgets(line, sizeof line, in)) {
        if (sscanf(line, "%lld,%lld,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &t, &seq, &g[0], &g[1], &g[2], &a[0], &a[1], &a[2], &temp) != 9) continue;
        if (t_prev < 0) { t_prev = t; continue; }
        int32_t aq[3], wq[3];
        for (int i = 0; i < 3; i++) { aq[i] = mq_from_float((float)a[i], 25); wq[i] = mq_from_float((float)g[i], 27); }
        mq_step(&s, aq, wq, (int32_t)(t - t_prev));
        t_prev = t;
        const double P = 4294967296.0, Q = 1073741824.0;
        fprintf(out, "%lld,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n", t,
                (double)s.p[0] / P, (double)s.p[1] / P, (double)s.p[2] / P, (double)s.v[0] / P, (double)s.v[1] / P,
                (double)s.v[2] / P, s.q[0] / Q, s.q[1] / Q, s.q[2] / Q, s.q[3] / Q, s.still);
    }
    fclose(in); fclose(out);
    return 0;
}
