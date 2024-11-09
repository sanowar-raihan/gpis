import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import cdist
from gpytoolbox import png2poly, random_points_on_mesh, edge_indices


def exponential_covariance(X1, X2, sigma_f=1, length=1):
    """exponential covariance function"""
    # k(xi, xj) = \sigma_f**2 \exp(-r**2/2l**2)
    r_sq = cdist(X1, X2, "sqeuclidean")
    return sigma_f**2 * np.exp(-r_sq/(2*length**2))

def exponential_covariance_with_gradient(X1, X2, length=1):
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
    def __init__(self, covariance_function, noise_variance):
        self.X_train = None
        self.y_train = None
        self.covariance_function = covariance_function
        self.noise_variance = noise_variance

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.K3 = self.covariance_function(X_train, X_train)
        self.K3 = self.K3 + self.noise_variance * np.eye(self.K3.shape[0])

    def predict(self, X_pred):
        K1 = self.covariance_function(X_pred, X_pred)
        K2 = self.covariance_function(X_train, X_pred)
        
        Q, R = np.linalg.qr(self.K3)
        
        X = np.linalg.solve(R, Q.T @ self.y_train)
        mu = K2.T @ X
        
        Y = np.linalg.solve(R, Q.T @ K2)
        cov = K1 - K2.T @ Y
                
        return mu, cov


rng = np.random.default_rng(42)
noise_variance = 1e-3
grid_size = 25


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
num_samples = 35

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
 
num_samples = 40
EC = edge_indices(poly.shape[0],closed=False)
P,I,_ = random_points_on_mesh(poly, EC, num_samples, return_indices=True,rng=rng)
vecs = poly[EC[:,0],:] - poly[EC[:,1],:]
vecs /= np.linalg.norm(vecs, axis=1)[:,None]
J = np.array([[0., -1.], [1., 0.]])
N = vecs @ J.T
N = N[I,:]
z = np.zeros(P.shape[0]) # Implicit surface values (zero level set)
"""

point_noise = noise_variance * rng.standard_normal(size=(P.shape[0], 2))
P += point_noise

normal_noise = noise_variance * rng.standard_normal(size=(N.shape[0], 2))
N += normal_noise
N /= np.linalg.norm(N, axis=1)[:, None] # Ensure the noisy normals remain unit vectors

X_train = P
y_train = np.concatenate((z, N.ravel()), axis=0)

# Fit the Gaussian Process model
gp = GaussianProcess(exponential_covariance_with_gradient, noise_variance)
gp.fit(X_train, y_train)

# Create a grid for predictions
x_pred = np.linspace(-3, 3, grid_size)
y_pred = np.linspace(-3, 3, grid_size)
X_pred, Y_pred = np.meshgrid(x_pred, y_pred)
X_grid = np.vstack((X_pred.ravel(), Y_pred.ravel())).T

# Prediction over the grid
Z_mean, Z_cov = gp.predict(X_grid)

# Visualize the mean of the scalar field
fig, ax = plt.subplots(1, 2)

img0 = ax[0].pcolormesh(X_pred, Y_pred, Z_mean[:grid_size**2].reshape(X_pred.shape), 
                        vmin=-np.max(np.abs(Z_mean)), vmax=np.max(np.abs(Z_mean)), cmap='RdBu', shading='gouraud')
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes('right', size='5%', pad=0.05)
fig.colorbar(img0, cax=cax0, orientation='vertical')

ax[0].scatter(P[:, 0], P[:, 1], color = 'brown')
ax[0].quiver(P[:,0], P[:,1], N[:,0], N[:,1], angles='xy', scale_units='xy', scale=2.5)
ax[0].contour(X_pred, Y_pred, Z_mean[:grid_size**2].reshape(X_pred.shape), levels=[0], colors='blue')
ax[0].set_aspect('equal')
ax[0].set_title('Mean')

# Visualize the standard deviation of the scalar field
Z_std = np.sqrt(np.diag(Z_cov[:grid_size**2, :grid_size**2]))
img1 = ax[1].pcolormesh(X_pred, Y_pred, Z_std.reshape(X_pred.shape), 
                        vmin=0, vmax=np.max(Z_std), cmap='plasma',shading='gouraud')
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes('right', size='5%', pad=0.05)
fig.colorbar(img1, cax=cax1, orientation='vertical')

ax[1].scatter(P[:, 0], P[:, 1], color = 'brown')
ax[1].quiver(P[:,0], P[:,1], N[:,0], N[:,1], angles='xy', scale_units='xy', scale=2.5)
ax[1].set_aspect('equal')
ax[1].set_title('Standard Deviation')


# Sample the entire gaussian process
num_surfaces = 4
Z_cov_scaled = 5e-4 * Z_cov
Z_samples = rng.multivariate_normal(Z_mean, Z_cov_scaled, num_surfaces)

# Visualize the sampled vector field
contour_levels = [-0.6, -0.3, 0, 0.3, 0.6]
contour_colors = ['blue', 'magenta', 'red', 'green', 'cyan']
for i in range(num_surfaces):
    fig_sample, ax_sample = plt.subplots()

    scalar_values = Z_samples[i][:grid_size**2]

    # Remaining values are gradients, with pairs representing (dx, dy) at each point
    gradient_values = Z_samples[i][grid_size**2:].reshape(grid_size**2, 2)
    gradient_x = gradient_values[:, 0].reshape(grid_size, grid_size)  # dx for each point
    gradient_y = gradient_values[:, 1].reshape(grid_size, grid_size)  # dy for each point

    scalar_field_grid = scalar_values.reshape(grid_size, grid_size)

    contours_sample= ax_sample.contour(X_pred, Y_pred, scalar_field_grid, levels=contour_levels, colors=contour_colors)

    # Extract contour line coordinates and plot the gradient vectors with quiver
    for seg in contours_sample.allsegs:
        for vertices in seg:
            contour_x, contour_y = vertices[:, 0], vertices[:, 1]

            # Find the nearest grid points for quiver
            idx_x = np.searchsorted(x_pred, contour_x)
            idx_y = np.searchsorted(y_pred, contour_y)
            
            # Make sure the indices stay within bounds
            idx_x = np.clip(idx_x, 0, gradient_x.shape[1] - 1)
            idx_y = np.clip(idx_y, 0, gradient_y.shape[0] - 1)
            
            contour_gradient_x = gradient_x[idx_y, idx_x]
            contour_gradient_y = gradient_y[idx_y, idx_x]

            ax_sample.quiver(contour_x, contour_y, contour_gradient_x, contour_gradient_y, 
                    angles='xy', scale_units='xy', scale=2, color='black')
            
    # Add legend for contour values            
    handles = [mlines.Line2D([], [], color=color, label=f'Level {level}') 
               for color, level in zip(contour_colors, contour_levels)]
    ax_sample.legend(handles=handles, title='Contour Levels', loc='upper right', bbox_to_anchor=(1.2, 1))

    ax_sample.set_title(f'Sampled vector field {i+1}')
    ax_sample.set_aspect('equal')

plt.show()