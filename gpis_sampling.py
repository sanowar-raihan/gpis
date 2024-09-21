import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from gpytoolbox import png2poly, random_points_on_mesh, edge_indices


def exponential_covariance(X1, X2, sigma_f=1, length=0.5):
    """exponential covariance function"""
    # k(xi, xj) = \sigma_f**2 \exp(-r**2/2l**2)
    r_sq = cdist(X1, X2, "sqeuclidean")
    return sigma_f**2 * np.exp(-r_sq/(2*length**2))

def exponential_covariance_with_gradient(X1, X2, length=0.5):
    """exponential covariance function with gradient observations"""
    n1, d = X1.shape
    n2, _ = X2.shape

    X_1 = X1[:, np.newaxis, :]  # Shape (n1, 1, d)
    X_2 = X2[np.newaxis, :, :]  # Shape (1, n2, d)

    K_values = exponential_covariance(X1, X2, length=length) # Shape (n1, n2)

    # \frac{\partial}{\partial x_j} k(xi, xj) = k(xi, xj)/l**2 (xi - xj)
    K_values_derivatives = K_values[:, :, np.newaxis]/length**2 * (X_1 - X_2) # Shape (n1, n2, d)
    K_values_derivatives = K_values_derivatives.reshape(n1, n2*d) # Shape (n1, n2*d)

    # \frac{\partial}{\partial x_i} k(xi, xj) = -k(xi, xj)/l**2 (xi - xj)
    K_derivatives_values = -K_values[:, :, np.newaxis]/length**2 * (X_1 - X_2) # Shape (n1, n2, d)
    K_derivatives_values = K_derivatives_values.transpose(0, 2, 1).reshape(n1*d, n2) # Shape (n1*d, n2)

    # \frac{\partial^2}{\partial xi \partial xj} k(xi, xj) = -k(xi, xj)/l**4 * (xi-xj)*(xi-xj)^T + k(xi, xj)/l**2 * I

    # term1 = -k(xi, xj)/l**4 * (xi-xj)*(xi-xj)^T for all pairs
    diffs = X_1 - X_2 # Shape (n1, n2, d)
    diffs_expanded = diffs[:, :, :, np.newaxis]  # shape (n1, n2, d, 1)
    diffs_transposed = diffs[:, :, np.newaxis, :]  # shape (n1, n2, 1, d)
    term1 = -K_values[:, :, np.newaxis, np.newaxis]/length**4 * (diffs_expanded * diffs_transposed) # Shape (n1, n2, d, d)

    # term2 = k(xi, xj)/l**2 * I for all pairs
    term2 = K_values[:, :, np.newaxis, np.newaxis]/length**2 * np.eye(d) # Shape (n1, n2, d, d)

    K_derivatives = term1 + term2 # (n1, n2, d, d)
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

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.K3 = self.covariance_function(X_train, X_train)

    def predict(self, X_pred):
        K1 = self.covariance_function(X_pred, X_pred)
        K2 = self.covariance_function(self.X_train, X_pred)
        
        Q, R = np.linalg.qr(self.K3)
        
        X = np.linalg.solve(R, Q.T @ self.y_train)
        mu = K2.T @ X
        
        Y = np.linalg.solve(R, Q.T @ K2)
        cov = K1 - K2.T @ Y
                
        return mu, cov


rng = np.random.default_rng(0)


# Vertices of a 2D circle
theta = 2*np.pi*rng.random(20)
x = np.cos(theta)
y = np.sin(theta)
P = 2*np.vstack((x, y)).T

N = np.vstack((x, y)).T # Normals are the same as positions on a circle
z = np.zeros(P.shape[0])  # Implicit surface values (zero level set)


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
"""

X_train = P
y_train = np.concatenate((z, N.ravel()), axis=0)

# Fit the Gaussian Process model
gp = GaussianProcess(covariance_function=exponential_covariance_with_gradient)
gp.fit(X_train, y_train)

# Create a grid for predictions
x_pred = np.linspace(-3, 3, 25)
y_pred = np.linspace(-3, 3, 25)
X_pred, Y_pred = np.meshgrid(x_pred, y_pred)
X_grid = np.vstack((X_pred.ravel(), Y_pred.ravel())).T

# Predict the scalar field over the grid
Z_mean, Z_cov = gp.predict(X_grid)

# Sample the entire gaussian process
num_surfaces = 3
Z_samples = rng.multivariate_normal(Z_mean, Z_cov, num_surfaces)
Z_samples = Z_samples[:, :25**2] # only the scalar field values


fig, axes = plt.subplots(1, num_surfaces + 1, figsize=(24,5))
# Plot the mean curve
Z_mean = Z_mean[:25**2].reshape(X_pred.shape)
axes[0].scatter(P[:, 0], P[:, 1], color = 'brown')
axes[0].quiver(P[:,0], P[:,1], N[:,0], N[:,1], angles='xy', scale_units='xy', scale=2.5)
axes[0].contour(X_pred, Y_pred, Z_mean, levels=[0], colors='red')
axes[0].set_title('Mean Curve')
axes[0].set_aspect('equal')

# Plot the sampled surfaces
for i in range(num_surfaces):
    Z_pred = Z_samples[i]
    Z_pred = Z_pred.reshape(X_pred.shape)
    ax = axes[i + 1]
    ax.scatter(P[:, 0], P[:, 1], color = 'brown')
    ax.quiver(P[:,0], P[:,1], N[:,0], N[:,1], angles='xy', scale_units='xy', scale=2.5)
    ax.contour(X_pred, Y_pred, Z_pred, levels=[0], colors='blue')
    ax.set_title(f'Sample {i+1}')
    ax.set_aspect('equal')

plt.show()