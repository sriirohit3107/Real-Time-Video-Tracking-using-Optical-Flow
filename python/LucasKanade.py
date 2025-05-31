import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect):
    """
    Lucas-Kanade Tracker (Translation only)
    
    Input:
    - It: Template image at time t
    - It1: Current image at time t+1
    - rect: (x1, y1, x2, y2) coordinates of the template in It
    
    Output:
    - p: np.array([dx, dy]) movement vector
    """
    # Parameters
    threshold = 0.01875
    maxIters = 100
    npDtype = np.float64
    p = np.zeros(2, dtype=npDtype)  # Initialize displacement (dx, dy)
    
    x1, y1, x2, y2 = rect

    # Set up interpolation functions for both images
    height, width = It.shape
    _x = np.arange(width)
    _y = np.arange(height)
    splineT = RectBivariateSpline(_y, _x, It)   # Notice: (y,x) order
    splineI = RectBivariateSpline(_y, _x, It1)

    # Grid points within the rectangle
    coordsX = np.linspace(x1, x2, int(x2-x1)+1)
    coordsY = np.linspace(y1, y2, int(y2-y1)+1)
    X, Y = np.meshgrid(coordsX, coordsY)

    # Template values
    T = splineT.ev(Y, X)

    for _ in range(maxIters):
        # Current guess of the rectangle position
        x_warp = X + p[0]
        y_warp = Y + p[1]

        # Check if points are inside image boundaries (optional)

        # Warp I1 to the template coordinate frame
        I_warp = splineI.ev(y_warp, x_warp)

        # Error image
        error = (T - I_warp).flatten()

        # Gradients of It1
        Ix = splineI.ev(y_warp, x_warp, dx=0, dy=1)
        Iy = splineI.ev(y_warp, x_warp, dx=1, dy=0)

        # Stack gradients
        A = np.vstack((Ix.flatten(), Iy.flatten())).T  # Shape: (num_points, 2)

        # Solve least squares for delta p
        delta_p, _, _, _ = np.linalg.lstsq(A, error, rcond=None)

        # Update p
        p += delta_p

        # Termination check
        if np.linalg.norm(delta_p) < threshold:
            break

    return p
