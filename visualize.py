import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def custom_colormap(n_colors, min_color, mid_color, max_color):
    """Create a custom colormap similar to MATLAB's approach"""
    colors = [min_color, mid_color, max_color]
    n_bins = n_colors
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    return cmap

def show_map(f=None, ax=None, query_az=None, query_in=None, query_snr=None, params=None):
    """
    
    Parameters:
    - f: matplotlib figure (None to create new)
    - ax: matplotlib axes (None to create new)
    - query_az: azimuth data array
    - query_in: incidence data array  
    - query_snr: SNR/visibility data array
    - params: list of parameters [iso_inc_lines, iso_az_lines, title, ...]
    
    Returns:
    - f: matplotlib figure
    - ax: matplotlib axes
    """
    
    # Replace NaN values with 0
    query_snr = np.nan_to_num(query_snr, nan=0.0)
    
    if f is None and ax is None:
        f = plt.figure(figsize=(7.5, 9.6), facecolor='white')
        f.canvas.manager.set_window_title(str(params[2]))
        ax = f.add_subplot(111, projection='3d', facecolor='white')
    
    # Set white background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    z_offset = np.max(query_snr)
    
    if query_snr.ndim == 2:  # Single 2D array
        ax.clear()
        
        # Set white background after clear
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False  
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        
        # Mask data outside 70 degrees radius in polar coordinates
        query_snr_masked = query_snr.copy()
        mask = query_in > 70  # Mask points beyond 70 degrees incidence
        query_snr_masked[mask] = np.nan
        
        # Additional masking to remove corner artifacts in polar coordinates
        # Convert to Cartesian coordinates to identify corner regions
        x_cart = query_in * np.cos(np.radians(query_az))
        y_cart = query_in * np.sin(np.radians(query_az))
        r_cart = np.sqrt(x_cart**2 + y_cart**2)
        
        # Mask data outside the maximum valid polar radius (corners of rectangular grid)
        max_polar_radius = 70  # Same as incidence limit
        corner_mask = r_cart > max_polar_radius
        query_snr_masked[corner_mask] = np.nan
        
        # Create surface plot with proper color mapping based on z-values (no grid lines)
        surf = ax.plot_surface(query_az, query_in, query_snr_masked, 
                              linewidth=0, antialiased=False, alpha=0.8,
                              cmap='viridis', vmin=np.nanmin(query_snr_masked), vmax=np.nanmax(query_snr_masked),
                              rcount=50, ccount=50)  # Reduce grid density for smoother appearance
        
        # Iso-incidence lines (concentric circles) - on z=0 plane
        r = np.linspace(15, 60, params[0])
        theta = np.linspace(0, 2*np.pi, 200)
        
        # Iso-azimuth lines (radial lines) - on z=0 plane
        ang = np.linspace(0, 360, params[1])
        
        pad_in = 1
        pad_az = 3
        z_grid = 0  # Place grid lines on z=0 plane
        
        # Draw iso-incidence circles on z=0 plane
        for i in range(params[0]):
            xi = r[i] * np.cos(theta)
            yi = r[i] * np.sin(theta)
            zi = np.ones(len(xi)) * z_grid
            
            ax.plot(xi, yi, zi, linewidth=1, color='k', linestyle='-', alpha=0.5)
            
            # Add incidence angle labels on z=0 plane
            lab_ang = np.radians(-45)
            xlab = r[i] * np.cos(lab_ang)
            ylab = r[i] * np.sin(lab_ang)
            
            ax.text(xlab-pad_in, -ylab-pad_in, z_grid, 
                   f'{int(round(r[i]))}°',
                   fontsize=16, color='k', style='normal', weight='normal')
        
        # Near-nadir exclusion area on z=0 plane
        ex_r = 15
        xex = ex_r * np.cos(theta)
        yex = ex_r * np.sin(theta)
        zex = np.ones(len(xex)) * z_grid
        
        # Create exclusion circle on z=0 plane
        ax.plot(xex, yex, zex, color='gray', linewidth=2, alpha=0.7)
        
        # Draw iso-azimuth lines on z=0 plane
        for j in range(params[1]):
            if j != params[1] - 1:  # Skip the last one to avoid overlap
                rho = np.max(query_in)
                
                xa = rho * np.cos(np.radians(ang[j]))
                ya = rho * np.sin(np.radians(ang[j]))
                
                # Draw line from center to edge and back on z=0 plane
                xa_line = np.array([xa, -xa])
                ya_line = np.array([ya, -ya])
                za_line = np.ones(2) * z_grid
                
                ax.plot(xa_line, ya_line, za_line, linewidth=1, color='k', linestyle='-', alpha=0.5)
                
                # Add azimuth angle labels on z=0 plane with simpler positioning
                if ya > 0 and xa > 0:  # First quadrant
                    ax.text(xa+pad_az, ya+pad_az, z_grid, f'{int(round(ang[j]))}°',
                           fontsize=14, color='k', ha='right', style='normal', weight='normal')
                elif ya > 0 and xa < 0:  # Second quadrant
                    ax.text(xa-pad_az, ya+pad_az, z_grid, f'{int(round(ang[j]))}°',
                           fontsize=14, color='k', ha='right', style='normal', weight='normal')
                elif ya < 0 and xa < 0:  # Third quadrant
                    ax.text(xa-pad_az, ya-pad_az, z_grid, f'{int(round(ang[j]))}°',
                           fontsize=14, color='k', ha='left', style='normal', weight='normal')
                elif ya < 0 and xa > 0:  # Fourth quadrant
                    ax.text(xa+pad_az, ya-pad_az, z_grid, f'{int(round(ang[j]))}°',
                           fontsize=14, color='k', ha='left', style='normal', weight='normal')
                elif abs(ya) < 1e-6 and xa < 0:  # Negative x-axis
                    ax.text(xa-pad_az, ya, z_grid, f'{int(round(ang[j]))}°',
                           fontsize=14, color='k', ha='center', style='normal', weight='normal')
                elif abs(ya) < 1e-6 and xa > 0:  # Positive x-axis
                    ax.text(xa+pad_az, ya, z_grid, f'{int(round(ang[j]))}°',
                           fontsize=14, color='k', ha='center', style='normal', weight='normal')
                elif ya > 0 and abs(xa) < 1e-6:  # Positive y-axis
                    ax.text(xa, ya+pad_az, z_grid, f'{int(round(ang[j]))}°',
                           fontsize=14, color='k', ha='right', style='normal', weight='normal')
                elif ya < 0 and abs(xa) < 1e-6:  # Negative y-axis
                    ax.text(xa, ya-pad_az, z_grid, f'{int(round(ang[j]))}°',
                           fontsize=14, color='k', ha='left', style='normal', weight='normal')
        
        # Set view to top-down with zero azimuth pointing upwards
        ax.view_init(elev=90, azim=180)
        
        # Configure plot for Variance-Adjusted Stochastic VM map
        title = params[2]
        surf.set_cmap('viridis')
        cbar = plt.colorbar(surf, ax=ax, location='bottom', shrink=0.8, aspect=30, pad=0.05)
        cbar.set_label(r'$VM_{sto(adj)}$', fontsize=24)
        ax.set_title(title, fontsize=26)
        
        # Turn off grid and axes completely
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Hide axis lines and spines
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Make axis panes invisible
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # Remove axis labels completely
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,0.1])
        
    else:
        # Handle 3D array case (multiple subplots)
        n_plots = query_snr.shape[2]
        
        for i in range(n_plots):
            ax_sub = plt.subplot(n_plots, 1, i+1, projection='3d')
            
            # Set white background for subplot
            ax_sub.xaxis.pane.fill = False
            ax_sub.yaxis.pane.fill = False
            ax_sub.zaxis.pane.fill = False
            ax_sub.xaxis.pane.set_edgecolor('white')
            ax_sub.yaxis.pane.set_edgecolor('white')
            ax_sub.zaxis.pane.set_edgecolor('white')
            
            surf = ax_sub.plot_surface(query_az, query_in, query_snr[:,:,i], 
                                      linewidth=0, antialiased=False, 
                                      cmap='viridis', 
                                      vmin=np.min(query_snr[:,:,i]), 
                                      vmax=np.max(query_snr[:,:,i]),
                                      rcount=50, ccount=50)
            
            # Apply similar grid and labeling logic as above
            # (Abbreviated for space - would follow same pattern as 2D case)
            
            ax_sub.view_init(elev=90, azim=180)
            ax_sub.grid(False)
            ax_sub.set_xticks([])
            ax_sub.set_yticks([])
            ax_sub.set_zticks([])
            
            # Hide axis lines and spines for subplot
            ax_sub.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax_sub.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax_sub.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            
            # Make axis panes invisible for subplot
            ax_sub.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax_sub.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax_sub.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    plt.tight_layout()
    return f, ax

# %% Load relevant visibility data
vm_folder = r"C:\Users\evank\Documents\repos\GVisMaps\data\1_1_genVM"

# Load CSV files using numpy and replace NaN values with 0
num_vm_map = np.genfromtxt(f"{vm_folder}/vm/stoVM.csv", delimiter=',')
num_vm_map = np.nan_to_num(num_vm_map, nan=0.0)

num_az_map = np.genfromtxt(f"{vm_folder}/vm/az.csv", delimiter=',')
num_az_map = np.nan_to_num(num_az_map, nan=0.0)

num_inc_map = np.genfromtxt(f"{vm_folder}/vm/inc.csv", delimiter=',')
num_inc_map = np.nan_to_num(num_inc_map, nan=0.0)

# %% Example usage (equivalent to MATLAB call)
def plot_visibility_map(est_map, az_map, inc_map, title='Variance-Adjusted Stochastic VM map'):
    """
    Equivalent to the MATLAB plotting call (no thresholding applied)
    """
    # Replace any NaN values with 0
    est_map_clean = np.nan_to_num(est_map, nan=0.0)
    
    params = [4, 13, title]
    
    map_f, map_ax = show_map(None, None, az_map, inc_map, est_map_clean, params)
    
    return map_f, map_ax

# %% Example of how to call it (equivalent to your MATLAB code)
if __name__ == "__main__":

    map_f, map_ax = plot_visibility_map(num_vm_map, num_az_map, num_inc_map, 
                                       'Variance-Adjusted Stochastic VM map')
    plt.show()

    pass
