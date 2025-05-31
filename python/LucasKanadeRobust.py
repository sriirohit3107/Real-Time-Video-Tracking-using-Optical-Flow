import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffineRobust(It, It1, rect, mtype='huber'):
    threshold = 0.01875
    maxIters  = 100

    x1, y1, x2, y2 = rect
    h, w = It.shape
    ys, xs = np.arange(h), np.arange(w)
    splineT = RectBivariateSpline(ys, xs, It)
    splineI = RectBivariateSpline(ys, xs, It1)

    dx, dy = x2-x1, y2-y1
    nx = max(int(abs(dx))+1, 1)
    ny = max(int(abs(dy))+1, 1)
    coordsX = np.linspace(x1, x2, nx)
    coordsY = np.linspace(y1, y2, ny)
    Xf, Yf = np.meshgrid(coordsX, coordsY)
    Xf, Yf = Xf.ravel(), Yf.ravel()
    T = splineT.ev(Yf, Xf)

    p = np.zeros((6,1), dtype=np.float64)
    huber_delta = 1.0

    for _ in range(maxIters):
        a, b, tx, c, d, ty = p.ravel()
        x_w = (1+a)*Xf + b*Yf + tx
        y_w = c*Xf + (1+d)*Yf + ty

        Iw = splineI.ev(y_w, x_w)
        error = (T - Iw).ravel()

        # compute robust weights
        if mtype=='huber':
            abs_e = np.abs(error)
            w = np.ones_like(abs_e)
            mask = abs_e > huber_delta
            w[mask] = huber_delta / abs_e[mask]
        elif mtype=='tukey':
            c_t = 4.685
            abs_e = np.abs(error)
            w = np.zeros_like(error)
            mask = abs_e < c_t
            w[mask] = (1 - (error[mask]/c_t)**2)**2
        else:
            raise ValueError(f"Unknown mtype '{mtype}'")

        # image gradients
        Ix = splineI.ev(y_w, x_w, dx=0, dy=1).ravel()
        Iy = splineI.ev(y_w, x_w, dx=1, dy=0).ravel()

        # steepest-descent images
        SD = np.vstack([
            Ix*Xf,
            Ix*Yf,
            Ix,
            Iy*Xf,
            Iy*Yf,
            Iy
        ]).T   # shape (N_pixels, 6)

        # weighted least squares
        WSD = SD * w[:, np.newaxis]
        Rw  = error * w
        delta_p, *_ = np.linalg.lstsq(WSD, Rw, rcond=None)
        delta_p = delta_p.reshape(6,1)

        p += delta_p
        if np.linalg.norm(delta_p) < threshold:
            break

    a, b, tx, c, d, ty = p.ravel()
    M = np.array([[1+a, b,    tx],
                  [c,   1+d,  ty]], dtype=np.float64)
    return M
