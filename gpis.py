import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from gpytoolbox import png2poly, random_points_on_mesh, edge_indices

# Define the Squared Exponential Covariance function (Fig.2)
def squared_exponential_covariance(X1, X2, alpha=2.0):
    """squared exponential covariance function."""
    r_sq = cdist(X1, X2, "sqeuclidean")
    return np.exp(-alpha * r_sq)

# Define the Thin Plate Covariance function for 2D curevs (Equation 11.a)
def thin_plate_covariance(X1, X2, R=2.0):
    """thin plate covariance function"""
    r = cdist(X1, X2, "euclidean")
    term1 = np.where(np.abs(r) > 1e-5, 2*r**2*np.log(np.abs(r)), 0)
    return term1 - (1 + 2*np.log(R))*r**2 + R**2

# Gaussian Process Class
class GaussianProcess:
    def __init__(self, covariance_function):
        self.covariance_function = covariance_function
        self.X_train = None
        self.y_train = None

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


rng = np.random.default_rng(20)

# Vertices of a 2D circle
theta = 2*np.pi*rng.random(20)
x = np.cos(theta)
y = np.sin(theta)
vertices = np.vstack((x, y)).T
z = np.zeros(vertices.shape[0])  # Implicit surface values (zero level set)


"""
rng = np.random.default_rng(85)

poly = png2poly("springer.png")[0]
poly = poly - np.min(poly)
poly = poly/np.max(poly)
poly = 2.5*poly
poly[:, 0] -= 0.5
poly[:, 1] -= 1.5
 
num_samples = 50
EC = edge_indices(poly.shape[0],closed=False)
vertices,I,_ = random_points_on_mesh(poly, EC, num_samples, return_indices=True,rng=rng)
z = np.zeros(vertices.shape[0]) # Implicit surface values (zero level set)
"""

# Add a point inside the shape with negative one value
inside_points = np.array([[0.0, 0.0]])
inside_values = -np.ones(inside_points.shape[0])

# Add some points outside the shape with positive one values
outside_theta = 2*np.pi*rng.random(10)[:,None]
outside_points = 1.2*np.concatenate((np.cos(outside_theta),np.sin(outside_theta)),axis=1)
outside_values = np.ones(outside_points.shape[0])

X_train = np.vstack((vertices, inside_points, outside_points))
y_train = np.concatenate((z, inside_values, outside_values))

# Fit the Gaussian Process model
gp = GaussianProcess(covariance_function=thin_plate_covariance)
gp.fit(X_train, y_train)

# Create a grid for predictions
x_pred = np.linspace(-1.5, 1.5, 50)
y_pred = np.linspace(-1.5, 1.5, 50)
X_pred, Y_pred = np.meshgrid(x_pred, y_pred)
X_grid = np.vstack((X_pred.ravel(), Y_pred.ravel())).T

# Predict the scalar field over the grid
Z_mean, Z_cov = gp.predict(X_grid)
Z_samples = rng.multivariate_normal(Z_mean, Z_cov, 3)

fig, axes = plt.subplots(1, Z_samples.shape[0] + 1, figsize=(24,5))

# Plot the mean curve
Z_mean = Z_mean.reshape(X_pred.shape)
ax = axes[0]
ax.contour(X_pred, Y_pred, Z_mean, levels=[0], colors='red')
ax.plot(vertices[:, 0], vertices[:, 1], '.k')
ax.set_title('Mean Curve')
ax.axis('equal')

# Plot each sample
for i in range(Z_samples.shape[0]):
    Z_pred = Z_samples[i]
    Z_pred = Z_pred.reshape(X_pred.shape)
    ax = axes[i + 1]
    ax.contour(X_pred, Y_pred, Z_pred, levels=[0], colors='blue')
    ax.plot(vertices[:, 0], vertices[:, 1], '.k')
    ax.set_title(f'Sample {i+1}')
    ax.axis('equal')

plt.show()