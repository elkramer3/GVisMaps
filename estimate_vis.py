import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def rotate_surface(x, y, scores, rotation_angle):
    """
    Optimized Python equivalent of the MATLAB rotate_surface function
    
    Inputs:
    x, y: n x n matrices of Cartesian coordinates
    scores: n x n matrix of visibility scores
    rotation_angle: rotation angle in radians
    
    Returns:
    rotated_scores: n x n matrix of interpolated scores on rotated grid
    """
    # Handle zero rotation case
    if abs(rotation_angle) < 1e-10:
        return scores.copy()
    
    # Convert Cartesian to polar coordinates
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Apply the rotation
    theta_rot = theta + rotation_angle
    
    # Convert back to Cartesian coordinates
    x_rot = rho * np.cos(theta_rot)
    y_rot = rho * np.sin(theta_rot)
    
    # Use scipy's RegularGridInterpolator for much faster interpolation
    from scipy.interpolate import RegularGridInterpolator
    
    # Create regular grid coordinates (assuming x, y are regular grids)
    rows, cols = scores.shape
    x_coords = x[0, :]  # First row (assuming regular grid)
    y_coords = y[:, 0]  # First column (assuming regular grid)
    
    # Create interpolator
    interpolator = RegularGridInterpolator(
        (y_coords, x_coords), scores,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    
    # Interpolate at rotated points
    points = np.column_stack([y_rot.flatten(), x_rot.flatten()])
    rotated_scores_flat = interpolator(points)
    
    # Reshape back to original shape
    rotated_scores = rotated_scores_flat.reshape(scores.shape)
    
    return rotated_scores

# %% Load relevant visibility data
vm_folder = r"C:\Users\evank\Documents\repos\GVisMaps\data\4_1_genVM"

# Load CSV files using numpy
num_vm_map = np.genfromtxt(f"{vm_folder}/vm/stoVM.csv", delimiter=',')
num_az_map = np.genfromtxt(f"{vm_folder}/vm/az.csv", delimiter=',')
num_inc_map = np.genfromtxt(f"{vm_folder}/vm/inc.csv", delimiter=',')
# num_map_stats not available as CSV - commenting out for now

vm_folder = r"C:\Users\evank\Documents\repos\GVisMaps\data\nyc_dsm"

# Load CSV files using numpy
dsm_vm_map = np.genfromtxt(f"{vm_folder}/vm/stoVM.csv", delimiter=',')
dsm_az_map = np.genfromtxt(f"{vm_folder}/vm/az.csv", delimiter=',')
dsm_inc_map = np.genfromtxt(f"{vm_folder}/vm/inc.csv", delimiter=',')
# dsm_map_stats not available as CSV - commenting out for now

# %% Load GeoTIFF (road network raster)
pth = r"C:\Users\evank\Documents\repos\GVisMaps\data\nyc_dsm\road_network_rasterized_clip.tif"

with rasterio.open(pth) as src:
    roadData = src.read(1)  # first band
roadData = roadData.astype(float)

# Make sure raster is square by padding the smaller dimension
raster_rows, raster_cols = roadData.shape
raster_dims = np.array([raster_rows, raster_cols])

dim_idx = np.argmax(raster_dims)
tot_add_cell = raster_dims[dim_idx] - raster_dims[1 - dim_idx]

# Initialize square padded array
scaled_roadData = np.zeros((raster_dims[dim_idx], raster_dims[dim_idx]))
if dim_idx == 0:  # more rows than cols
    start = tot_add_cell // 2
    scaled_roadData[:, start:start + raster_cols] = roadData
else:  # more cols than rows
    start = tot_add_cell // 2
    scaled_roadData[start:start + raster_rows, :] = roadData

# %% Compute the 2D Fourier Transform
fft2dataGrid = np.fft.fftshift(np.fft.fft2(scaled_roadData))
rows, cols = fft2dataGrid.shape

# Create coordinate grids
X, Y = np.meshgrid(np.arange(-cols//2, cols//2), np.arange(-rows//2, rows//2))

# Convert to polar coordinates
theta, rho = np.arctan2(Y, X), np.sqrt(X**2 + Y**2)

# Magnitude of FFT
fftMagdataGrid = np.abs(fft2dataGrid)

# Angular spectrum
angular_spectrum = np.zeros(360)
for angle in range(360):
    mask = (theta >= np.deg2rad(angle - 0.5)) & (theta < np.deg2rad(angle + 0.5))
    angular_spectrum[angle] = np.sum(fftMagdataGrid[mask])

# Normalize
angular_spectrum /= np.max(angular_spectrum)

# Dominant directions
dom_angles = np.where(angular_spectrum > 0.4)[0]

# %% Rotate visibility map according to dominant orientations
est_map = np.zeros_like(num_vm_map)
for i in range(len(dom_angles)):
    rotation_angle = np.deg2rad(dom_angles[i])
    temp_map = rotate_surface(num_az_map, num_inc_map, num_vm_map, rotation_angle)
    
    est_map = est_map + temp_map

# Normalize map
if np.nanmax(est_map) > 0:
    est_map = est_map / np.nanmax(est_map)
else:
    est_map = np.zeros_like(est_map)

# %% Plot results
plt.figure()
plt.imshow(np.log1p(fftMagdataGrid), cmap="pink", origin="lower")
c = plt.colorbar()
c.set_label(r"$|F(u,v)|$", fontsize=14)
plt.title("Road network Fourier Spectrum", fontsize=16)
plt.xlabel("u"); plt.ylabel("v")
plt.gca().set_aspect("equal")
plt.show()

plt.figure()
ax = plt.subplot(111, polar=True)
ax.plot(np.deg2rad(360-np.arange(0, 360, 1)), angular_spectrum, linewidth=1.5, color="k")
ax.set_title("Directional Analysis of Road Orientations", fontsize=16)
ax.set_theta_zero_location("N")
ax.set_theta_direction(1)  # counterclockwise
plt.show()

# %% Visualize all three visibility maps using visualize.py
from visualize import plot_visibility_map

# Plot the numerical visibility map
print("Plotting numerical visibility map...")
num_fig, num_ax = plot_visibility_map(num_vm_map, num_az_map, num_inc_map, 
                                     'Numerical Visibility Map')

# Plot the DSM visibility map  
print("Plotting DSM visibility map...")
dsm_fig, dsm_ax = plot_visibility_map(dsm_vm_map, dsm_az_map, dsm_inc_map,
                                     'DSM Visibility Map')

# Plot the estimated visibility map
print("Plotting estimated visibility map...")
est_fig, est_ax = plot_visibility_map(est_map, num_az_map, num_inc_map,
                                     'Estimated Visibility Map (Road-Oriented)')

plt.show()