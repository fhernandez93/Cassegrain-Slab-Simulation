import h5py
import xarray as xr
import imageio
import matplotlib.pyplot as plt
import os
import numpy as np
import tidy3d as td
import matplotlib
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from scipy.fft import fftn, ifftn, fftshift
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema



def get_sphere(n):
    """get_sphere(n) discretized a sphere using a Fibonacci lattice with midpoint intertion (and poles added by hand)
       input parameters:
       n (int) number of points in the discretization (including poles)
       returns:       
       sphere <class 'scipy.spatial._qhull.ConvexHull'> with points and triangulation according to convex hull
              (check https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html)
       phi (numpy array, len=n) array with phi angles 
       theta (numpy array, len=n) array with theta angles        
       
       more info in "Measurement of Areas on a Sphere Using Fibonacci and Latitude–Longitude Lattices" by
        Alvaro Gonzalez, Math Geosci (2010) 42: 49–64, DOI 10.1007/s11004-009-9257-x. 
        Check also https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    """
    import numpy as np
    import scipy.spatial as sp
    #Golden ratio
    gr=(1.+np.sqrt(5.))/2.
    #ng=number of points by Fibonacci lattice with midpoint insertion
    ng=n-2
    i=np.arange(ng,dtype=int)
    phi=np.zeros(n)
    theta=np.zeros(n)
    phi[1:ng+1]=2*np.pi*i/gr
    theta[1:ng+1]=np.arccos(1.-2*(i+0.5)/ng)
    #adding poles by hand
    phi[0]=0.
    theta[0]=0.
    phi[n-1]=0.
    theta[n-1]=np.pi
    #getting Cartesian coordinates
    points=np.zeros((n,3))    
    sin_arr=np.sin(theta)
    points[:,0]=np.cos(phi)*sin_arr
    points[:,1]=np.sin(phi)*sin_arr
    points[:,2]=np.cos(theta)
    #getting convex hull
    sphere=sp.ConvexHull(points)
    
    
    return sphere,phi,theta

def moving_average(x, w=3):
    if w == 0:
        return x
    return np.convolve(x, np.ones(w), 'same') / w  # 'same' preserves length

def moving_average_with_sem(data, window):
    """Compute moving average and standard error of the mean"""
    if window <= 1:
        return data, np.zeros_like(data)
    
    mean = moving_average(data, window)  # now same length as data
    rolling_std = np.array([np.std(data[max(0, i-window//2):min(len(data), i+window//2+1)]) 
                            for i in range(len(data))])
    sem = rolling_std / np.sqrt(window)
    return mean, sem



def _write_dict_to_hdf5(data, hdf_group):
    for key, value in data.items():
        key = str(key)
        if isinstance(value, dict):
            # If the value is a nested dictionary, create a group and call recursively
            subgroup = hdf_group.create_group(key)
            _write_dict_to_hdf5(value, subgroup)
        else:
            # If the value is not a dictionary, store it as a dataset
            hdf_group.create_dataset(key, data=value)
def create_hdf5_from_dict(data, filename):
    """ Data must have a structure like this, and all items are arrays 
     data = {
                        'transmission_right':transmission_flux_right,
                        'transmission_left':transmission_flux_left,
                        'transmission0':transmission0,
                        'lambdas':monitor_lambdas,
                        'decay':structure_1.final_decay
                  }
    
    """
    with h5py.File(filename, 'w') as hdf_file:
        # Recursively traverse the dictionary and write data to the HDF5 file
        _write_dict_to_hdf5(data, hdf_file)





def create_movie(sim_result, monitor_lambdas,name='',type='t',log=False,path="",frames_per_second=1,rem_frames=False, normalize=True, a=1, movie=False):
    # Define the colors: Blue -> White -> Yellow -> Green -> Red
    colors = [
              (1, 1, 1),  # White
              (0, 0, 1),  # Blue
              (1, 1, 0),  # Yellow
              (0, 1, 0),  # Green
              (1, 0, 0), #Red
              (0, 0, 0), # Black
              ]  
    
    # Create a custom colormap
    n_bins = 500  # Number of bins for smooth transition
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors, N=n_bins)
    field_time_out = sim_result.get_intensity("time_monitorFieldOut")
    fig1, ax_1 = plt.subplots()
    ax1=sim_result.simulation.plot_structures_eps(freq=monitor_lambdas[0], cbar=False, z=0, ax=ax_1,
                reverse=False)

    # Create a new figure for plotting the shapes
    patches_list = ax1.patches
    plt.close(fig1)
    frames = []
    max_values = field_time_out.max(axis=(0,1,2)) if normalize else 1

    field_time_out = field_time_out/max_values
    field_time_out['x'] = field_time_out['x']/a
    field_time_out['y'] = field_time_out['y']/a
    field_time_out['z'] = field_time_out['z']/a


    if log:
        field_log = np.log10((field_time_out))
        folder_pics = "logPics"
        
    else:
        field_log = field_time_out
        folder_pics = "linPics"

    if not os.path.exists(f'{path}/{folder_pics}'):
            os.makedirs(f'{path}/{folder_pics}')
            print(f"Folder {path}/{folder_pics} created successfully.")

    for i, element in enumerate(range(np.shape(field_time_out)[-1])):
        if i%5 != 0:
            continue
        if os.path.isfile(f'{path}/{folder_pics}/frame_{i}.png'):
            frames.append(f'{path}/{folder_pics}/frame_{i}.png')
            continue
        fig, ax = plt.subplots(figsize=(5, 7))
        if type=="t":
            pcolormesh = (field_log).isel(t=i).squeeze().plot.pcolormesh(ax=ax,cmap=cmap)
           
        else:
            pcolormesh = (field_log).isel(f=i).squeeze().plot.pcolormesh(ax=ax,cmap=cmap)

        ax.set_aspect('equal', adjustable='box')
        try:
            plt.title(f'Time: {(np.array(field_time_out['t'][()][i])*1e12):.4g} ps')
        except:
            plt.title(f'$\\nu$: {(a/np.array(td.C_0/field_time_out['f'][()][i])):.4g}')
        ax.set_ylabel(rf"x",fontsize=16)
        ax.set_xlabel(rf"y",fontsize=16)
        ax.tick_params(which='major', labelsize=14)
        
        for patch in patches_list:
            path_patch = patch.get_path()  
            new_patch = patches.PathPatch(path_patch, edgecolor= (0,0,0, 0.08), facecolor='none')
            t2 = matplotlib.transforms.Affine2D().rotate_deg(90) + ax.transData
            new_patch.set_transform(t2)
            ax.add_patch(new_patch)
        # Save the frame
        plt.savefig(f'{path}/{folder_pics}/frame_{i}.pdf')
        plt.close(fig)
        frames.append(f'{path}/{folder_pics}/frame_{i}.png')
        

        
    if movie:
        name_movie = f'{path}/{name}.mp4' if name else f'output/anderson/d(t) analysis/Diameter d(t) at output of the structure Range - {monitor_lambdas[0]:.3g} - {monitor_lambdas[-1]:.3g}.mp4'
        with imageio.get_writer(name_movie, fps=frames_per_second) as writer:
            for frame in frames:
                image = imageio.imread(frame)
                writer.append_data(image)

    # Optionally, remove the individual frames if no longer needed
    if rem_frames:
        for frame in frames:
            os.remove(frame)
    
    return False

### Calculate the charactieristic lenght from the correlations 

def compute_normalized_autocorrelation_fft(density_field_3d):
    # Subtract mean to center the data
    rho = density_field_3d - np.mean(density_field_3d)
    
    # Compute autocorrelation via FFT
    fft_rho = fftn(rho)
    autocorr = np.real(ifftn(np.abs(fft_rho)**2))
    
    # Normalize
    autocorr /= np.prod(rho.shape)
    autocorr /= np.var(density_field_3d)  # Now C(0) = 1

    return fftshift(autocorr)

# Extract radial profile C_r(δr)
def radial_profile(data, L:float, bins=50):
    nx, ny, nz = data.shape
    dx =L / nx

    # Build coordinate grid in physical units
    x = np.arange(nx) - (nx - 1) / 2
    y = np.arange(ny) - (ny - 1) / 2
    z = np.arange(nz) - (nz - 1) / 2
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    r = dx * np.sqrt(X**2 + Y**2 + Z**2)

    # Flatten
    r = r.flatten()
    data = data.flatten()
    # Bin and average
    r_bins = np.linspace(0, r.max(), bins + 1)
    r_vals = 0.5 * (r_bins[1:] + r_bins[:-1])
    digitized = np.digitize(r, r_bins)
    radial_C = np.array([data[digitized == i].mean() if np.any(digitized == i) else 0.0
                         for i in range(1, bins + 1)])
    return r_vals, radial_C

# Compute normalized autocorrelation
def get_a_from_h5_eps(file:str, L:float,plot_correlation:bool=True):
    with h5py.File(file, 'r') as f:
        permittivity_raw = np.array(f['epsilon'])
        density_normalized = (permittivity_raw - permittivity_raw.min()) / (permittivity_raw.max() - permittivity_raw.min())

        autocorr = compute_normalized_autocorrelation_fft(density_normalized)
        r_vals, C_r = radial_profile(autocorr,L=L)
        r_vals = r_vals[~np.isnan(C_r)]
        C_r = C_r[~np.isnan(C_r)]
        f_interp = interp1d(r_vals, C_r, kind='cubic')  # 'cubic' or 'quadratic' for smooth curves
        x_fine = np.linspace(r_vals.min(), r_vals.max(), 20000)
        y_fine = f_interp(x_fine)
        local_maxima_indices = argrelextrema(y_fine, np.greater)[0]

        print(f"First maximum at x = {x_fine[local_maxima_indices[0]]}, y = {y_fine[local_maxima_indices[0]]}")
        a=x_fine[local_maxima_indices[0]]
        if plot_correlation:
            plt.plot(r_vals/a, C_r , 'o', label='Original points')
            plt.plot(x_fine/a, y_fine, '-', label='Interpolated curve')
            plt.plot(x_fine[local_maxima_indices]/a, y_fine[local_maxima_indices], 'rx', label='Local maxima')
            plt.xlabel(r'$r/a$')
            plt.ylabel(r'$C_r(\delta r)$')
            plt.title(f'Normalized Radial Autocorrelation a={a:.2f}')
            plt.grid(True)
            plt.legend()
            plt.show()

        return a,x_fine,y_fine
    

def unwrap_h5(obj):
    """Automatically convert HDF5 objects to appropriate Python types."""
    if isinstance(obj, h5py.Dataset):
        data = obj[()]
        # If it's bytes, decode to string
        if isinstance(data, (bytes, bytearray)):
            return data.decode()
        # If it's a scalar NumPy type, convert to native Python
        if np.isscalar(data):
            return data.item()
        # If it's a numpy array of bytes (e.g. for string arrays)
        if hasattr(data, "dtype") and data.dtype.kind in {"S", "O"}:
            try:
                return data.astype(str).tolist()
            except Exception:
                return data.tolist()
        return data
    elif isinstance(obj, h5py.Group):
        # For groups, recursively parse contents
        return {key: unwrap_h5(obj[key]) for key in obj.keys()}
    else:
        return obj

    
def read_hdf5_as_dict(filename):
    with h5py.File(filename, 'r') as hdf_file:
        return unwrap_h5(hdf_file)