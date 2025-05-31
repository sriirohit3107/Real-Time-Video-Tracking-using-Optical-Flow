import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, rect):
    """
    Affine Lucas–Kanade Tracker (translation + linear terms).

    Inputs:
      It, It1 – Grayscale template and current image (2D arrays).
      rect    – [x1, y1, x2, y2] defining the template window in It.

    Output:
      M       – 2×3 affine warp matrix mapping points from It → It1.
    """
    # Convergence parameters
    threshold = 0.01875
    maxIters  = 100

    # Unpack rectangle
    x1, y1, x2, y2 = rect

    # Build interpolators
    h, w = It.shape
    ys = np.arange(h)
    xs = np.arange(w)
    splineT = RectBivariateSpline(ys, xs, It)
    splineI = RectBivariateSpline(ys, xs, It1)

    # Create sampling grid — ensure non-negative count
    dx = x2 - x1
    dy = y2 - y1
    nx = max(int(abs(dx)) + 1, 1)
    ny = max(int(abs(dy)) + 1, 1)
    coordsX = np.linspace(x1, x2, nx)
    coordsY = np.linspace(y1, y2, ny)
    X, Y = np.meshgrid(coordsX, coordsY)
    Xf = X.flatten()
    Yf = Y.flatten()

    # Template pixel values
    T = splineT.ev(Yf, Xf)

    # Initialize affine parameters [a, b, tx, c, d, ty]^T = 0
    p = np.zeros((6, 1), dtype=np.float64)

    for _ in range(maxIters):
        # Warp coordinates by current affine parameters
        a, b, tx, c, d, ty = p.ravel()
        x_w = (1 + a) * Xf + b * Yf + tx
        y_w = c * Xf + (1 + d) * Yf + ty

        # Interpolate I1 at warped coords
        Iw = splineI.ev(y_w, x_w)

        # Compute error vector
        error = (T - Iw)

        # Gradients at warped coords
        Ix = splineI.ev(y_w, x_w, dx=0, dy=1).flatten()
        Iy = splineI.ev(y_w, x_w, dx=1, dy=0).flatten()

        # Steepest-descent images (Jacobian wrt p)
        SD = np.vstack([
            Ix * Xf,
            Ix * Yf,
            Ix,
            Iy * Xf,
            Iy * Yf,
            Iy
        ]).T  # shape: (N_pixels, 6)

        # Solve for delta_p via least-squares to handle singular cases
        delta_p, *_ = np.linalg.lstsq(SD, error.flatten(), rcond=None)
        delta_p = delta_p.reshape(6, 1)

        # Update parameters
        p += delta_p

        # Termination check
        if np.linalg.norm(delta_p) < threshold:
            break

    # Assemble final affine warp matrix
    a, b, tx, c, d, ty = p.ravel()
    M = np.array([
        [1 + a, b,     tx],
        [c,     1 + d, ty]
    ], dtype=np.float64)

    return M
