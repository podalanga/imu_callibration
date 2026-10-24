/* Host build of the embedded filter: runs firmware/main/nav.c on a dataset's imu.csv.
 *   make -C firmware/host
 *   ./firmware/host/host_replay data/synth_track/imu.csv est_c.csv [--line] [--no-nhc] [--no-zupt]
 * Output columns match est.csv from the Python runner, so tests/test_c_vs_python.py can compare them. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../main/nav.h"

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s imu.csv est.csv [--line] [--no-nhc] [--no-zupt]\n", argv[0]); return 1; }
    nav_config_t cfg; nav_calib_t cal; nav_t *n = malloc(sizeof(nav_t));
    nav_default_config(&cfg);
    nav_identity_calib(&cal);
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--line")) cfg.use_line = 1;
        if (!strcmp(argv[i], "--no-nhc")) cfg.use_nhc = 0;
        if (!strcmp(argv[i], "--no-zupt")) cfg.use_zupt = 0;
    }
    nav_init(n, &cfg, &cal);
    FILE *in = fopen(argv[1], "r"), *out = fopen(argv[2], "w");
    if (!in || !out) { perror("open"); return 1; }
    char line[512];
    if (!fgets(line, sizeof line, in)) return 1;      /* header */
    fprintf(out, "t_us,px,py,pz,vx,vy,vz,qw,qx,qy,qz,bax,bay,baz,bgx,bgy,bgz,stationary\n");
    long long t_us, seq; double g[3], a[3], temp;
    while (fgets(line, sizeof line, in)) {
        if (sscanf(line, "%lld,%lld,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &t_us, &seq, &g[0], &g[1], &g[2],
                   &a[0], &a[1], &a[2], &temp) != 9) continue;
        float af[3] = {(float)a[0], (float)a[1], (float)a[2]}, gf[3] = {(float)g[0], (float)g[1], (float)g[2]};
        if (nav_step(n, (int64_t)t_us, af, gf)) {
            eskf_t *f = &n->f;
            fprintf(out, "%lld,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%d\n",
                    t_us, f->p[0], f->p[1], f->p[2], f->v[0], f->v[1], f->v[2], f->q[0], f->q[1], f->q[2], f->q[3],
                    f->ba[0], f->ba[1], f->ba[2], f->bg[0], f->bg[1], f->bg[2], n->still);
        }
    }
    fclose(in); fclose(out);
    fprintf(stderr, "zupt %u zaru %u nhc %u line %u gated %u nan_resets %u\n", n->cnt.zupt, n->cnt.zaru,
            n->cnt.nhc, n->cnt.line, n->cnt.gated, n->cnt.nan_resets);
    return 0;
}
