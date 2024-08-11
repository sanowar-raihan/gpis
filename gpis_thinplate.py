import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gpytoolbox import png2poly, random_points_on_mesh, edge_indices


def thin_plate_covariance_with_gradient(X1, X2, R=10):
    """thin plate covariance function with gradient obesrvations"""
    n1, d = X1.shape
    n2, _ = X2.shape

    X_1 = X1[:, np.newaxis, :]  # Shape (n1, 1, d)
    X_2 = X2[np.newaxis, :, :]  # Shape (1, n2, d)

    # Compute norm values from pairwise differences
    r = np.linalg.norm(X_1 - X_2, axis=2)  # Shape (n1, n2)
    r = np.where(r == 0, np.finfo(float).eps, r) # Avoid log(0) by setting zero norms to a small positive number
    
    # k(xi, xj) = 2*r^2*log(r) - (1 + 2*log(R))*r^2 + R^2
    K_values = 2*r**2*np.log(r) - (1 + 2*np.log(R))*r**2 + R**2 # Shape (n1, n2)

    # \frac{\partial}{\partial x_j} k(xi, xj) = - 4*(log(r) - log(R))*(xi-xj)
    log_diffs = np.log(r) - np.log(R)  # Shape (n1, n2)
    K_values_derivatives = - 4 * log_diffs[:, :, np.newaxis] * (X_1 - X_2)  # Shape (n1, n2, d)
    K_values_derivatives = K_values_derivatives.reshape(n1, n2*d) # Shape (n1, n2*d)

    # \frac{\partial}{\partial x_i} k(xi, xj) = - 4*(log(r) - log(R))*(xi-xj)
    K_derivatives_values = 4 * log_diffs[:, :, np.newaxis] * (X_1 - X_2)  # Shape (n1, n2, d)
    K_derivatives_values = K_derivatives_values.transpose(0, 2, 1).reshape(n1*d, n2) # Shape (n1*d, n2)

    # \frac{\partial^2}{\partial xi \partial xj} k(xi, xj) = -4*((xi-xj)*(xi-xj)^T/r**2 + (log r - logR)I)
    
    # term1 = (xi-xj)*(xi-xj)^T/r**2 for all pairs
    diffs = X_1 - X_2 # Shape (n1, n2, d)
    diffs_expanded = diffs[:, :, :, np.newaxis]  # shape (n1, n2, d, 1)
    diffs_transposed = diffs[:, :, np.newaxis, :]  # shape (n1, n2, 1, d)
    term1 = (diffs_expanded * diffs_transposed) / (r[:, :, np.newaxis, np.newaxis] ** 2) # Shape (n1, n2, d, d)

    # term2 = (log r - logR)I for all pairs
    term2 = (np.log(r) - np.log(R))[:, :, np.newaxis, np.newaxis] * np.eye(d) # Shape (n1, n2, d, d)

    K_derivatives = - 4 * (term1 + term2) # (n1, n2, d, d)
    K_derivatives = K_derivatives.transpose(0, 2, 1, 3).reshape(n1 * d, n2 * d) # (n1*d, n2*d)

    # Combine into the block matrix
    K = np.block([
            [K_values, K_values_derivatives],
            [K_derivatives_values, K_derivatives]
        ])
        
    return K


# Gaussian Process Class
class GaussianProcess:
    def __init__(self, covariance_function):
        self.X_train = None
        self.y_train = None
        self.covariance_function = covariance_function

    def fit(self, X_train, y_train, R=3):
        self.X_train = X_train
        self.y_train = y_train
        self.K3 = self.covariance_function(X_train, X_train)

    def predict(self, X_pred):
        K1 = self.covariance_function(X_pred, X_pred)
        K2 = self.covariance_function(X_train, X_pred)
        
        Q, R = np.linalg.qr(self.K3)
        
        X = np.linalg.solve(R, Q.T @ self.y_train)
        mu = K2.T @ X
        
        Y = np.linalg.solve(R, Q.T @ K2)
        cov = K1 - K2.T @ Y
                
        return mu, cov


rng = np.random.default_rng(0)

"""
# Vertices of a 2D circle
theta = 2*np.pi*rng.random(20)
x = np.cos(theta)
y = np.sin(theta)
P = 2*np.vstack((x, y)).T

N = np.vstack((x, y)).T # Normals are the same as positions on a circle
z = np.zeros(P.shape[0])  # Implicit surface values (zero level set)
"""

"""
poly = png2poly("images/illustrator.png")[0]
poly = poly - np.min(poly)
poly = poly/np.max(poly)
poly = 0.5*poly + 0.25
poly = 3*poly - 1.5
poly = poly * 2.5
num_samples = 20

EC = edge_indices(poly.shape[0],closed=False)
P,I,_ = random_points_on_mesh(poly, EC, num_samples, return_indices=True,rng=rng)
vecs = poly[EC[:,0],:] - poly[EC[:,1],:]
vecs /= np.linalg.norm(vecs, axis=1)[:,None]
J = np.array([[0., -1.], [1., 0.]])
N = vecs @ J.T
N = N[I,:]

z = np.zeros(P.shape[0])
"""


poly = png2poly("images/springer.png")[0]
poly = poly - np.min(poly)
poly = poly/np.max(poly)
poly = 2.5*poly
poly[:, 0] -= 0.5
poly[:, 1] -= 1.5
poly = 2*poly
 
num_samples = 50
EC = edge_indices(poly.shape[0],closed=False)
P,I,_ = random_points_on_mesh(poly, EC, num_samples, return_indices=True,rng=rng)
vecs = poly[EC[:,0],:] - poly[EC[:,1],:]
vecs /= np.linalg.norm(vecs, axis=1)[:,None]
J = np.array([[0., -1.], [1., 0.]])
N = vecs @ J.T
N = N[I,:]
z = np.zeros(P.shape[0]) # Implicit surface values (zero level set)


X_train = P
y_train = np.concatenate((z, N.ravel()), axis=0)

# Fit the Gaussian Process model
gp = GaussianProcess(covariance_function=thin_plate_covariance_with_gradient)
gp.fit(X_train, y_train)

# Create a grid for predictions
x_pred = np.linspace(-3, 3, 50)
y_pred = np.linspace(-3, 3, 50)
X_pred, Y_pred = np.meshgrid(x_pred, y_pred)
X_grid = np.vstack((X_pred.ravel(), Y_pred.ravel())).T

# Predict the scalar field over the grid
Z_pred, Z_cov = gp.predict(X_grid)
Z_pred, Z_cov = Z_pred[:50**2], Z_cov[:50**2, :50**2]
Z_mean = Z_pred.reshape(X_pred.shape)
Z_std = np.sqrt(np.diag(Z_cov)).reshape(X_pred.shape)

# Visualize the mean and variance
fig, ax = plt.subplots(1, 2)

img0 = ax[0].pcolormesh(X_pred, Y_pred, Z_mean, vmin=-np.max(np.abs(Z_mean)), vmax=np.max(np.abs(Z_mean)), cmap='RdBu', shading='gouraud')
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes('right', size='5%', pad=0.05)
fig.colorbar(img0, cax=cax0, orientation='vertical')

ax[0].scatter(P[:, 0], P[:, 1], color = 'brown')
ax[0].quiver(P[:,0], P[:,1], N[:,0], N[:,1], angles='xy', scale_units='xy', scale=2.5)
ax[0].contour(X_pred, Y_pred, Z_mean, levels=[0], colors='blue')
ax[0].set_aspect('equal')
ax[0].set_title('Mean')

img1 = ax[1].pcolormesh(X_pred, Y_pred, Z_std, vmin=0, vmax=np.max(Z_std), cmap='plasma',shading='gouraud')
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes('right', size='5%', pad=0.05)
fig.colorbar(img1, cax=cax1, orientation='vertical')

ax[1].scatter(P[:, 0], P[:, 1], color = 'brown')
ax[1].quiver(P[:,0], P[:,1], N[:,0], N[:,1], angles='xy', scale_units='xy', scale=2.5)
ax[1].set_aspect('equal')
ax[1].set_title('Standard Deviation')

plt.show()