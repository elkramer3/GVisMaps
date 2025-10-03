import numpy as np
import rasterio
import matplotlib.pyplot as plt

# %% Load relevant visibility data
vm_folder = r"C:\Users\evank\Documents\repos\GVisMaps\data\4_1_genVM"

# Load CSV files using numpy
num_vm_map = np.genfromtxt(f"{vm_folder}/vm/stoVM.csv", delimiter=',')
num_az_map = np.genfromtxt(f"{vm_folder}/vm/az.csv", delimiter=',')
num_inc_map = np.genfromtxt(f"{vm_folder}/vm/inc.csv", delimiter=',')
# num_map_stats not available as CSV - commenting out for now

vm_folder = r"C:\Users\evank\MIT Dropbox\Evan Kramer\PhD\Research\orbit design\nyc_dsm_1lsmask"

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
if dim_idx == 0:  # more rows
    start = tot_add_cell // 2
    scaled_roadData[start:start + raster_cols, :] = roadData.T
    scaled_roadData = scaled_roadData.T
else:  # more cols
    start = tot_add_cell // 2
    scaled_roadData[:, start:start + raster_rows] = roadData

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
ax.plot(np.deg2rad(np.arange(359, -1, -1)), angular_spectrum, linewidth=1.5, color="k")
ax.set_title("Directional Analysis of Road Orientations", fontsize=16)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)  # counterclockwise
plt.show()