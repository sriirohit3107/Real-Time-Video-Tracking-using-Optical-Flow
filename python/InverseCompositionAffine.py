import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
    """
    Q3.3
        [I] It: Template image
            It1: Current image
            rect: Current position of the object
                (top left, bottom right coordinates: x1, y1, x2, y2)
        [O] M: the Affine warp matrix [2x3 numpy array]
    """

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    # p = np.zeros((6,1))
    npDtype = np.float64
    W = np.eye(3, dtype=npDtype)    # This might be a better format than p
    x1, y1, x2, y2 = rect
    height, width = It.shape

    # Clamp and sort
    x1_, x2_ = sorted([np.clip(x1, 0, width - 1), np.clip(x2, 0, width - 1)])
    y1_, y2_ = sorted([np.clip(y1, 0, height - 1), np.clip(y2, 0, height - 1)])

    template_x = np.arange(int(x1_), int(x2_) + 1)
    template_y = np.arange(int(y1_), int(y2_) + 1)
    splineT = RectBivariateSpline(np.arange(height), np.arange(width), It)
    template = splineT(template_y, template_x, grid=True)

    # Compute the gradients of the template image
    grad_x = splineT.ev(template_y[:, None], template_x[None, :], dx=0, dy=1).flatten()
    grad_y = splineT.ev(template_y[:, None], template_x[None, :], dx=1, dy=0).flatten()


    # Compute gradient of template image
    coords_x, coords_y = np.meshgrid(template_x, template_y)
    coords = np.stack((coords_x.flatten(), coords_y.flatten(), np.ones(coords_x.size)))

    # Compute Jacobian
    jacobian = np.zeros((len(grad_x), 6))
    jacobian[:, 0] = grad_x * coords_x.flatten()
    jacobian[:, 1] = grad_y * coords_x.flatten()
    jacobian[:, 2] = grad_x * coords_y.flatten()
    jacobian[:, 3] = grad_y * coords_y.flatten()
    jacobian[:, 4] = grad_x
    jacobian[:, 5] = grad_y
    # Compute Hessian
    H = jacobian.T @ jacobian
    H_inv = np.linalg.pinv(H)
    # Finish after maxIters or [at the end] when deltaP < threshold
    for _ in range(maxIters):

        # Warp image
        warp_matrix = W[:2, :]
        warped_coords = warp_matrix @ coords
        x_warped, y_warped = warped_coords[0], warped_coords[1]

        splineI = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
        warped_image = splineI.ev(y_warped, x_warped).reshape(template.shape)
        # Compute error image
        error = (warped_image - template).flatten()
        # Compute deltaP
        deltaP = H_inv @ (jacobian.T @ error)
        # Compute new W
        deltaP_matrix = np.array([[1 + deltaP[0], deltaP[2], deltaP[4]],
                                  [deltaP[1], 1 + deltaP[3], deltaP[5]],
                                  [0, 0, 1]])
        W = W @ np.linalg.inv(deltaP_matrix)
        
        # Continue unless below threshold
        if np.linalg.norm(deltaP) < threshold:
            break


    # reshape the output affine matrix
    # M = np.array([[1.0+p[0], p[1],    p[2]],
    #              [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)
    M = W[:2, :]

    return M