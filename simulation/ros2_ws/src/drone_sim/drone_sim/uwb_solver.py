"""ROS-free UWB multilateration: Gauss-Newton least squares with z-mirror rejection."""
import numpy as np


def solve(anchors, ranges, x0, z_prior=1.0, z_min=0.0, max_iter=20, tol=1e-6, sigma=0.05, huber=0.0):
    """Estimate a 3D position from ranges to known anchors.

    anchors: (N,3); ranges: (N,); x0: initial guess (3,). Needs N >= 4 (3 anchors would leave a
    two-fold ambiguity; coplanar anchors always leave a mirror about their plane, resolved by the z prior).
    Returns (position (3,), covariance (3,3), rms residual, iterations) or None if it fails.
    The covariance is sigma^2 (J^T J)^-1 (3x3, world frame).
    """
    A = np.asarray(anchors, dtype=float)
    r = np.asarray(ranges, dtype=float)
    ok = np.isfinite(r) & (r > 0.0)
    A, r = A[ok], r[ok]
    if len(r) < 4:
        return None

    def gn(start):
        x = np.array(start, dtype=float)
        it = 0
        for it in range(1, max_iter + 1):
            d = x - A
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            res = dist - r
            J = d / dist[:, None]
            w = np.ones_like(res)
            if huber > 0.0:
                big = np.abs(res) > huber
                w[big] = huber / np.abs(res[big])
            JW = J * w[:, None]
            H = J.T @ JW
            try:
                step = np.linalg.solve(H + 1e-9 * np.eye(3), -(JW.T @ res))
            except np.linalg.LinAlgError:
                return None
            x = x + step
            if np.linalg.norm(step) < tol:
                break
        d = x - A
        res = np.linalg.norm(d, axis=1) - r
        return x, float(np.sqrt(np.mean(res ** 2))), it

    best = None
    x0 = np.asarray(x0, dtype=float)
    starts = [np.array([x0[0], x0[1], max(x0[2], z_prior)])]
    # also start from the mirror image so a coplanar-anchor mirror can be compared explicitly
    starts.append(np.array([x0[0], x0[1], -max(abs(x0[2]), z_prior)]))
    for s in starts:
        out = gn(s)
        if out is None:
            continue
        x, rms, it = out
        if x[2] < z_min:
            x = x.copy()
            x[2] = 2.0 * float(A[:, 2].mean()) - x[2]   # reflect through the (coplanar) anchor plane
            if x[2] < z_min:
                continue
            d = np.linalg.norm(x - A, axis=1) - r
            rms = float(np.sqrt(np.mean(d ** 2)))
        if best is None or rms < best[1] - 1e-12:
            best = (x, rms, it)
    if best is None:
        return None
    x, rms, it = best
    d = x - A
    J = d / np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
    try:
        cov = sigma ** 2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.inf)
    return x, cov, rms, it
