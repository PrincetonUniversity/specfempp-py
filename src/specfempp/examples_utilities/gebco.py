"""This file just defines a function to download bathymetry data from GEBCO
and save it to a file. The link is static and the file is saved to the
current working directory.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import cartopy.geodesic as geodesic
import cartopy.crs as ccrs
from scipy.interpolate import RegularGridInterpolator
from . import mapping
from . import plot as sfplot

import matplotlib.gridspec as gridspec

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.dates as mdates



def download(remove_zip: bool = False, force: bool = False,
                              progress: bool = False):
    """Download bathymetry data from GEBCO and save it to a file.

    Parameters
    ----------
    remove_zip : bool, optional
        Removes the original after the download, by default False
    force : bool, optional
        forces redownload even though the output bathymetry file exists, by default False
    progress : bool, optional
        shows download progress, by default False
    """
    from .download_file import download_with_progress
    import requests
    import zipfile
    import os
    link = "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2024/zip/"
    
    
    # Download bathymetry data
    if not os.path.exists("gebco/GEBCO_2024.nc") or force:
        
        if force:
            print("Forcing redownload of bathymetry data.")
        else:
            print("Downloading bathymetry data from GEBCO.")
            
        # The progress bar implementation seems to be working only
        # sometimes. On the plus side, you can resume the download.
        if progress:
            download_with_progress(link, "gebco_2024.zip", chunksize=1024**2 * 10)
        else:
            r = requests.get(link)
            with open("gebco_2024.zip", "wb") as f:
                f.write(r.content)
                
        print("Download complete.")
        
        
        # Unzip bathymetry data
        with zipfile.ZipFile("gebco_2024.zip", 'r') as zip_ref:
            # Extract to gebco subdirectory
            zip_ref.extractall("gebco")
        
        # Remove zip file
        if remove_zip:
            os.remove("gebco_2024.zip")

    else:
        print("Bathymetry data already exists. Skipping download.")


def get_bathymetry(extent, split=False):
    """Extract a rectangular area from a global grid that may cross the dateline.
    
    Parameters:
    -----------
    extent : list or tuple
        [minlon, maxlon, minlat, maxlat]
        If minlon > maxlon, it's assumed the area crosses the dateline
    
    Returns:
    --------
    extracted_grid : 2D numpy array
        The extracted portion of the grid
    extracted_lons : 1D numpy array
        Longitudes of the extracted grid
    extracted_lats : 1D numpy array
        Latitudes of the extracted grid
    """
    
    import numpy as np
    import netCDF4 as nc 
    
    # Load bathymetry data    
    dataset = nc.Dataset('gebco/gebco_2024.nc')

    # Extract variables
    grid_lons = dataset.variables['lon'][:]
    grid_lats = dataset.variables['lat'][:]
    
    minlon, maxlon, minlat, maxlat = extent
    
    # Check if we're crossing the dateline
    dateline_crossing = minlon > maxlon
    
    # Find latitude indices (this is straightforward)
    lat_indices = np.where((grid_lats >= minlat) & (grid_lats <= maxlat))[0]
    min_lat_idx, max_lat_idx = lat_indices[0], lat_indices[-1]
    
    bathymetry = dict(
        dateline_crossing=dateline_crossing,
        
    )
    
    # Extract longitude indices (handles dateline crossing)
    if dateline_crossing:
        # For dateline crossing, we need two segments: minlon to 180 and -180 to maxlon
        west_indices = np.where(grid_lons >= minlon)[0]
        east_indices = np.where(grid_lons <= maxlon)[0]
        lon_indices = np.concatenate([west_indices, east_indices])
        
        bathymetry['split'] = split
        
        if split:
            # Split the dateline crossing into two segments        
            lon_indices = np.array(lon_indices)
            # print(lon_indices)
            lons = grid_lons[lon_indices]
            west_indices = np.where(lons >= 0)[0]
            east_indices = np.where(lons < 0)[0]
            
            print(west_indices, east_indices)
            
            # print(west_indices, east_indices)
            west_longitudes = np.array(lons[lons >= 0])
            east_longitudes = np.array(lons[lons < 0])
            
            # print(west_indices, east_indices)
            # print(np.arange(min_lat_idx, max_lat_idx+1, skip))
            # Fix to get every
            west_segment = np.array(dataset['elevation'][min_lat_idx:max_lat_idx+1, west_indices])
            east_segment = np.array(dataset['elevation'][min_lat_idx:max_lat_idx+1, east_indices])      
            
            # Get values at the dateline
            west_longitudes = np.append(west_longitudes, 180)
            east_longitudes = np.append(-180, east_longitudes)
            
            # Concatenate the segments
            west_segment = np.concatenate([west_segment, dataset['elevation'][min_lat_idx:max_lat_idx+1, 0:1]], axis=1)
            east_segment = np.concatenate([dataset['elevation'][min_lat_idx:max_lat_idx+1:, 0:1], east_segment], axis=1)
            
            bathymetry['elevation_west'] = np.array(west_segment)
            bathymetry['elevation_east'] = np.array(east_segment)
            bathymetry['longitudes_west'] = np.array(west_longitudes)
            bathymetry['longitudes_east'] = np.array(east_longitudes)
        else:
            
            west_segment = dataset['elevation'][min_lat_idx:max_lat_idx+1, west_indices]
            east_segment = dataset['elevation'][min_lat_idx:max_lat_idx+1, east_indices]
            
            extracted_grid = np.concatenate([west_segment, east_segment], axis=1)
            extracted_lons = np.concatenate([grid_lons[west_indices], grid_lons[east_indices]])
            
            bathymetry['elevation'] = extracted_grid
            bathymetry['longitudes'] = extracted_lons
        
    else:
        # Regular case (no dateline crossing)
        lon_indices = np.where((grid_lons >= minlon) & (grid_lons <= maxlon))[0]
        min_lon_idx, max_lon_idx = lon_indices[0], lon_indices[-1]
        
        extracted_grid = dataset['elevation'][min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]
        
        extracted_lons = grid_lons[min_lon_idx:max_lon_idx+1]
        
        bathymetry['elevation'] = extracted_grid
        bathymetry['longitudes'] = extracted_lons
    
    # Extract latitudes
    bathymetry['latitudes'] = np.array(grid_lats[min_lat_idx:max_lat_idx+1])
    
    return bathymetry


def plot_bathymetry(bathymetry, ax=None, cmap=None, norm=None, **kwargs):
    
    from matplotlib.colors import Normalize, TwoSlopeNorm
    import matplotlib.pyplot as plt
    import numpy as np
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # Create topographic colormap and norm
    if norm == None and cmap == None:
        cmap, norm = topography_cmap_and_norm()
        
    if norm is None:
        norm = TwoSlopeNorm(vmin=-6000, vcenter=0, vmax=8000)
    elif norm == 'linear':
        if bathymetry['dateline_crossing'] and bathymetry['split']:
            vmin = np.min([np.min(bathymetry['elevation_west']), np.min(bathymetry['elevation_east'])])
            vmax = np.max([np.max(bathymetry['elevation_west']), np.max(bathymetry['elevation_east'])])
        else:
            vmin = np.min(bathymetry['elevation'])
            vmax = np.max(bathymetry['elevation'])
        
        norm = Normalize(vmin=vmin, vmax=vmax)
        
    if cmap is None:
        cmap = plt.get_cmap('terrain')
    
    
    if bathymetry['dateline_crossing'] and bathymetry['split']:
        # Plot west segment
        im = ax.pcolormesh(bathymetry['longitudes_west'], bathymetry['latitudes'], bathymetry['elevation_west'], cmap=cmap, norm=norm, **kwargs)
        # Plot east segment
        ax.pcolormesh(bathymetry['longitudes_east'], bathymetry['latitudes'], bathymetry['elevation_east'], cmap=cmap, norm=norm, **kwargs)
    else:
        im = ax.pcolormesh(bathymetry['longitudes'], bathymetry['latitudes'], bathymetry['elevation'], cmap=cmap, norm=norm, **kwargs)
    
    return im


def to_meters(bathymetry):
    """Basic conversion of longitudes and latitudes to meters."""
    
    
    bathymetry_m = dict()
    
    lons = bathymetry['longitudes']
    lats = bathymetry['latitudes']
        
    if bathymetry["dateline_crossing"]:
    
        # convert negative longitudes to positive
        lons[lons < 0] += 360
        
        # degrees to meter conversion factor
        degree2meters = 111319.9
        
        # Convert to meters
        x = (lons - lons[0]) * 111319.9
        y = (lats - lats[0]) * 111319.9
        
    else:
        
        x = (lons - lons[0]) * 111319.9
        y = (lats - lats[0]) * 111319.9
        
    bathymetry_m['x'] = x
    bathymetry_m['y'] = y
    bathymetry_m['elevation'] = bathymetry['elevation']
    
    return bathymetry_m
    


def bathymetryprofile(width, npts, lonlat, az, input_lons=None, input_lats=None, input_bathy=None):
    """
    Generate a bathymetry profile centered at [lon lat] along a straight path
    in a specified azimuthal direction. The azimuth will be the direction
    from the left to the right of the profile.
    
    Parameters
    ----------
    width : float
        Length of the path in meters
    npts : int
        Number of points on the path
    lonlat : list or tuple
        [longitude, latitude] of the center of the path
    az : float
        Azimuthal direction of the path in degrees
    input_lons : numpy.ndarray, optional
        Array of input longitudes for bathymetry data
    input_lats : numpy.ndarray, optional
        Array of input latitudes for bathymetry data
    input_bathy : numpy.ndarray, optional
        Array of bathymetry values corresponding to input_lons and input_lats
    
    Returns
    -------
    x : numpy.ndarray
        x-coordinate along the path from 0 to width
    z : numpy.ndarray
        elevation
    
    Example
    -------
    # basic example call with default bathymetry source
    lon = -171.9965
    lat = -12.0744
    az = 90
    npts = 401
    width = 20000
    x, z = bathymetryprofile(width, npts, [lon, lat], az)
    
    # Example with custom bathymetry data
    lons = np.array([...])  # longitude points
    lats = np.array([...])  # latitude points
    bathy = np.array([...])  # corresponding bathymetry values
    x, z = bathymetryprofile(width, npts, [lon, lat], az, lons, lats, bathy)
    
    # plot the bathymetry
    plt.plot(x, z)
    plt.show()
    """

    # Unpack the input
    lon, lat = lonlat
    
    # Calculate the profile line
    x, lats, lons = mapping.get_line(lat, lon, width, npts, az)
    
    # This handles cases where lons span across 180 E/W longitude
    # Determine if there is any jump in lons
    difflons = lons[1:] - lons[:-1]
    if not (np.all(difflons < 0) or np.all(difflons > 0)) and not np.all(difflons == 0):
        lons = np.mod(lons, 360)
        is_across180 = True
    else:
        is_across180 = False
        
    # If custom bathymetry data is provided, use it
    if input_lons is not None and input_lats is not None and input_bathy is not None:
        # Use the provided bathymetry data
        longrid = copy.deepcopy(input_lons)
        latgrid = copy.deepcopy(input_lats)
        elev = input_bathy

        # Check whether bathymetry data is in [-180 180] or [0 360]
        difflons = longrid[1:] - longrid[:-1]
        if not (np.all(difflons < 0) or np.all(difflons > 0)) and not np.all(difflons == 0):
            longrid = np.mod(longrid, 360)
            bathy_is_across180 = True
        else:
            bathy_is_across180 = False            
            
    else:
        # Request gridded elevation from the default bathymetry source
        longrid, latgrid, elev = bathymetry(None, 
                                            [min(lons) - 0.1, max(lons) + 0.1], 
                                            [min(lats) - 0.1, max(lats) + 0.1], 
                                            False)
        
        # convert longitude to [-180 180] when lons do not span across 180 E/W
        if not is_across180:
            longrid = np.mod(longrid + 180, 360) - 180
        
        # handle edge cases where longrid span across 180 E/W longitude
        if longrid[0] > longrid[-1]:
            longrid = np.mod(longrid, 360)
            lons = np.mod(lons, 360)
    
    # Use scipy's griddata for 2D interpolation
    RGI = RegularGridInterpolator((longrid, latgrid), elev.T, method='linear')
    z = RGI(np.vstack((lons, lats)).T)
    
    # shift x-coordinate to [0 width]
    x = x - x[0]
    
    return x, z, lats, lons


def _reckon(lat1, lon1, distance, azimuth):
    """
    Calculate the destination point given a starting point, 
    distance, and azimuth using Cartopy.
    
    Parameters
    ----------
    lat1 : float
        Starting latitude in degrees
    lon1 : float
        Starting longitude in degrees
    distance : numpy.ndarray or float
        Distance in degrees
    azimuth : float
        Azimuth in degrees
    
    Returns
    -------
    lats : numpy.ndarray
        Destination latitudes in degrees
    lons : numpy.ndarray
        Destination longitudes in degrees
    """
    # Convert azimuth to bearing in geodesic convention (clockwise from north)
    bearing = 90 - azimuth
    
    # Cartopy's geodesic calculator
    geod = geodesic.Geodesic()
    
    # For scalar inputs
    if isinstance(distance, (int, float)):
        # Convert degrees to meters (approx 111.32 km per degree)
        dist_m = distance * 111320  # meters
        
        # Calculate the destination point
        result = geod.direct(np.array([[lon1, lat1]]), bearing, dist_m)
        return result[0, 1], result[0, 0]
    
    # For array inputs
    # Convert degrees to meters
    dist_m = distance * 111320  # meters
    
    # Initialize arrays for results
    lats = np.zeros_like(distance)
    lons = np.zeros_like(distance)
    
    # Process each point
    for i, d in enumerate(dist_m):
        result = geod.direct(np.array([[lon1, lat1]]), bearing, d)
        lons[i] = result[0, 0]
        lats[i] = result[0, 1]
    
    return lats, lons

def bathymetry(data=None, lon_range=None, lat_range=None, plot_map=False, ax=None):
    """
    Get bathymetry data for a specified region.
    
    This is a stub function that should be implemented to retrieve actual bathymetry data.
    In a complete implementation, this would connect to a bathymetry database
    or load from local data files.
    
    Parameters
    ----------
    data : None or str
        Path to data file or None to use default dataset
    lon_range : list
        [min_longitude, max_longitude]
    lat_range : list
        [min_latitude, max_latitude]
    plot_map : bool
        Whether to plot the bathymetry map
    ax : matplotlib.axes.Axes
        Axes to plot on
    
    Returns
    -------
    longrid : numpy.ndarray
        Grid of longitudes
    latgrid : numpy.ndarray
        Grid of latitudes
    elev : numpy.ndarray
        Grid of elevations
    """
    # This is a placeholder - in a real implementation, you would:
    # 1. Load or retrieve bathymetry data
    # 2. Process it to the specified region
    # 3. Return the gridded data
    
    # For this example, we'll create some dummy data
    if lon_range is None:
        lon_range = [-180, 180]
    if lat_range is None:
        lat_range = [-90, 90]
    
    # Create a grid
    lons = np.linspace(lon_range[0], lon_range[1], 100)
    lats = np.linspace(lat_range[0], lat_range[1], 100)
    longrid, latgrid = np.meshgrid(lons, lats)
    
    # Create some dummy elevation data - this should be replaced with real data
    # This creates a simple pattern with deeper water away from land
    elev = -5000 + 5000 * np.exp(-((longrid - lon_range[0] - (lon_range[1]-lon_range[0])/2)**2 + 
                                   (latgrid - lat_range[0] - (lat_range[1]-lat_range[0])/2)**2) / 1000)
    
    # Plot if requested
    if plot_map:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        c = ax.contourf(longrid, latgrid, elev, cmap='Blues_r')
        plt.colorbar(c, ax=ax)
        ax.set_xlabel('Longitude (degrees)')
        ax.set_ylabel('Latitude (degrees)')
        ax.set_title('Bathymetry')
        
        xoffset = 0  # No offset for longitude in this simple implementation
        return longrid, latgrid, elev, None, c, xoffset
    
    return longrid, latgrid, elev


"""
Below are functions that are used to adjust the domain size and bathymetry
for the dynamic bathymetry example. The functions are used to adjust the
domain size and bathymetry based on the input extent of the domain.
"""


def element_number_from_extent(offset_in_m, depth_in_m, mean_bathymetry_in_m):
    """Computes element numbers based on ratio of elements to meters from Ivy's 
    example. The values are hard coded in the function

    Parameters
    ----------
    offset_in_m : float
        new domain offset in meters
    depth_in_m : float
        new domain in meters
    mean_bathymetry_in_m : float
        new mean bathymetry in meters

    Returns
    -------
    tuple of 4 int
        nx, ny, nlower, nupper
    """
    # Values from Ivy's example to adjust the simulation
    nx_for_ratio = 250         # elements
    nz_for_ratio = 120         # elements
    offset_for_ratio = 20000.0 # meters
    depth_for_ratio = 9600.0   # meters

    offset_elements_per_m = nx_for_ratio / offset_for_ratio
    depth_elements_per_m = nz_for_ratio / depth_for_ratio
    
    # Adjust the domain size
    nx = int(np.ceil(offset_elements_per_m * offset_in_m))
    nz = int(np.ceil(depth_elements_per_m * depth_in_m))
    
    # Adjust the acoustic elastic domain split based on mean bathymetry
    depth_split_element_number = \
      int(np.ceil(depth_elements_per_m * mean_bathymetry_in_m))
      
    nlower = nz - depth_split_element_number
    nupper = depth_split_element_number
    
    return nx, nz, nlower, nupper


def update_par_file_extent(par_dict, offset_in_m, depth_in_m, bathymetry_in_m,
                           verbose: bool = False):
    """This updates the par_file based on the input extent of the domain. It does
    not update the 

    Parameters
    ----------
    par_dict : OrderedDict
        Par_file in dictionary format
    offset_in_m : float
        new domain offset in meters
    depth_in_m : float
        new domain in meters
    bathymetry_in_m : np.ndarray
        new domain mean bathymetry in meters

    Raises
    ------
    ValueError
        if par_dict['regions'] does not have 2 regions
    ValueError
        if region name is not recognized
    """
    
    # Values from Ivy's example to adjust the simulation
    nx, ny, nlower, nupper = element_number_from_extent(
        offset_in_m, depth_in_m, np.mean(bathymetry_in_m))
    
    if verbose:
        print(f"nx: {nx}, ny: {ny}, nlower: {nlower}, nupper: {nupper}")
    
    # Check length of regions
    if len(par_dict["regions"]) != 2:
        raise ValueError("There should be two regions in the parameter file for this example.")
      
    for region in par_dict["regions"]:
        # Acoustic
        if region[4] == 2:
            region[0] = 1
            region[1] = nx
            region[2] = nlower + 1
            region[3] = ny
            
        # Elastic
        elif region[4] == 1:
            region[0] = 1
            region[1] = nx
            region[2] = 1
            region[3] = nlower
            
        else:
            raise ValueError("Region number not recognized.")


def write_topography_file(topography_file, offset_in_m, depth_in_m, bathymetry_in_m, bathymetry_offset_in_m):
    """Writes the topography file in the format required by the dynamic bathymetry
    example.

    Parameters
    ----------
    topography_file : str
        name of the topography file
    bathymetry : np.ndarray
        bathymetry data
    """
    
    # Get bathymetry data
    N_bathymetry = len(bathymetry_in_m)
    
    # Get the element numbers
    nx, ny, nlower, nupper = element_number_from_extent(
        offset_in_m, depth_in_m, np.mean(bathymetry_in_m))
    
    with open(topography_file, 'w') as f:

        # First header
        f.write("#\n# number of interfaces\n#\n")
        
        # Write the number of interfaces in the file
        number_of_interfaces = 3
        f.write(f" {number_of_interfaces}\n")
        
        # Write the interface header
        f.write("#\n# for each interface below, we give the number of points "
                "and then x,z for each point\n#\n")
        
        # Write interface 1 header
        f.write("#\n# interface number 1 (bottom of the mesh)\n#\n")
        
        # Write number of nodes for the bottom of the mesh
        f.write(f" {2}\n")
        
        # Write the bottom of the mesh (x,z) combination in integer format
        f.write(f" {0:>7d} {0:>7d}\n")
        f.write(f" {int(offset_in_m):>7d} {0:>7d}\n")
        
        # Write interface 2 header
        f.write("#\n# interface number 2 (bathymetry)\n#\n")
        
        # Write number of nodes for the top of the mesh
        f.write(f" {N_bathymetry}\n")
        
        # Write the bathymetry data
        for i in range(N_bathymetry):
            f.write(f" {int(bathymetry_offset_in_m[i]):>7d} {int(bathymetry_in_m[i]):>7d}\n")
        
        # Write interface 3 header
        f.write("#\n#write interface 3 (top of the mesh)\n")
        
        # Write interface 3 (top of the mesh)
        f.write(f" {2}\n")
        f.write(f" {0:>7d} {int(depth_in_m):>7d}\n")
        f.write(f" {int(offset_in_m):>7d} {int(depth_in_m):>7d}\n")
        
        # Write element header
        f.write("#\n# for each layer, we give the number of spectral "
                "elements in the vertical direction\n#\n")
        
        # Write the element header for layer 1
        f.write("#\n# layer number 1 (bottom, elastic layer)\n#\n")
        
        # Write the elements for layer 1
        f.write(f" {nlower}\n")
        
        # Write the element header for layer 2
        f.write("#\n# layer number 2 (top, acoustic layer)\n#\n")
        f.write(f" {nupper}\n")
      
def test_cmap_and_norm():
        
    # invent some data (height in meters relative to sea level)
    data = np.linspace(-10000,8000,15**2).reshape((15,15))

    cmap, norm = topography_cmap_and_norm()
    # blues = plt.get_cmap()
    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(data, norm=norm, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
      
    
def topography_cmap_and_norm():
    colors_undersea = plt.cm.Blues_r(np.linspace(0, 0.85,173)) #512
    colors_land = plt.cm.terrain(np.linspace(0.25, 1, 87))
    # combine them and build a new colormap
    colors = np.vstack((colors_undersea, colors_land))
    cut_terrain_map = mcolors.LinearSegmentedColormap.from_list('cut_terrain', colors)
    norm = sfplot.FixPointNormalize(sealevel=0, vmin=-8000, vmax=4000, col_val=0.5)
    return cut_terrain_map, norm


def plot_summary(station_lat, station_lon, event_lat, event_lon,
                 mermaid_metadata,
                 mermaid_bathymetry, local_bathymetry,
                 line_offset, line_bathymetry, line_latitudes, line_longitudes):

    # Create the figure
    fig = plt.figure(figsize=(8, 10))

    # Setup the plot grid
    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 0.025, 0.025], wspace=0.5)

    # Get the map midpoint
    _, mlon = mapping.get_midpoint(station_lat, station_lon, event_lat, event_lon)

    # Plot the track bathymetry
    ax = fig.add_subplot(gs[0,0], projection=ccrs.PlateCarree(mlon))

    # Get extent of the track bathymetry
    mermaid_extent = mapping.get_extent(
        mermaid_bathymetry["latitudes"][0], mermaid_bathymetry["longitudes"][0],
        mermaid_bathymetry["latitudes"][-1], mermaid_bathymetry["longitudes"][-1])

    # Adjust the extent if it crosses the dateline
    if mermaid_extent[0] > 0 and mermaid_extent[1] < 0:
        mermaid_extent[1] += 360

    # Set the extent
    ax.set_extent(mermaid_extent)

    # Plot the bathymetry
    bathy_plot = plot_bathymetry(mermaid_bathymetry, ax = ax, transform=ccrs.PlateCarree())    

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, zorder=-10,)
    gl.left_labels = True
    gl.right_labels = False
    gl.top_labels = True
    gl.bottom_labels = False

    # Create colornorm for the trajectory
    norm = Normalize(vmin=np.min(mermaid_metadata['datetime']), 
                    vmax=np.max(mermaid_metadata['datetime']))

    # Plot the trajectory
    _, sm = sfplot.xyz_line(mermaid_metadata['lon'], mermaid_metadata['lat'],
                        mermaid_metadata['datetime'], cmap='grey', norm=norm,
                        linewidth=4,
                        transform=ccrs.Geodetic())    

    # Plot the mermaid location at the time of the event
    ax.plot(station_lon, station_lat, 'rv', label='Mermaid', markeredgecolor='black',
            markersize=10, transform=ccrs.Geodetic())

    # Bathymetry colorbar
    bathy_cax = fig.add_subplot(gs[0,1])
    plt.colorbar(bathy_plot, label='Elevation (m)', cax=bathy_cax)

    # Mermaid colorbar
    mermaid_cax = fig.add_subplot(gs[0,2])
    fig.colorbar(sm, cax=mermaid_cax, label='Time', 
                 format=mdates.DateFormatter("%Y-%m-%d"))

    # Local bathymetry axes
    local_ax = fig.add_subplot(gs[1,0], projection=ccrs.PlateCarree())
    local_cax = fig.add_subplot(gs[1,1])

    # Get extent of the local bathymetry
    bathymetry_extent = mapping.get_extent(
        local_bathymetry["latitudes"][0], local_bathymetry["longitudes"][0],
        local_bathymetry["latitudes"][-1], local_bathymetry["longitudes"][-1])

    # Adjust the extent if it crosses the dateline
    if bathymetry_extent[0] > 0 and bathymetry_extent[1] < 0:
        bathymetry_extent[1] += 360

    # Set the extent
    local_ax.set_extent(bathymetry_extent)

    # Plot the bathymetry
    lbathy = plot_bathymetry(local_bathymetry, ax=local_ax,
                                norm='linear', cmap='grey')

    # Add gridlines
    gl = local_ax.gridlines(draw_labels=True, zorder=-10,)
    gl.left_labels = True
    gl.right_labels = False
    gl.top_labels = False
    gl.bottom_labels = True

    # Plot the line along the back azimuth of the event
    local_ax.plot(line_longitudes, line_latitudes, 
            'r-', label='Mermaid', linewidth=2, transform=ccrs.Geodetic())
    
    # Plot the mermaid location at the time of the event
    local_ax.plot(station_lon, station_lat, 'rv', label='Mermaid', markeredgecolor='black',
                markersize=10, transform=ccrs.Geodetic())

    # Bathymetry colorbar
    fig.colorbar(lbathy, label='Elevation (m)', cax=local_cax)

    # Plot bathymetry along Offset
    offset_ax = fig.add_subplot(gs[2,:])
    
    # Set colors for ocean an crust
    ocean_color = np.array([175, 205, 240])/255 # light blue
    crust_color = np.array([175, 175, 175])/255 # light gray 
    
    # Plot the bathymetry along the offset
    offset_ax.fill_between(line_offset, 0, -line_bathymetry, color=ocean_color, alpha=0.5)
    offset_ax.fill_between(line_offset, -line_bathymetry, 9600, color=crust_color, alpha=0.5)
    
    # Plot the bathymetry line
    offset_ax.plot(line_offset, -line_bathymetry, 'k', linewidth=2)
    
    # Plot the mermaid location at the time of the event
    offset_ax.plot(line_offset[len(line_offset)//2], 1500, 'rv', 
                   markeredgecolor='black', markersize=10)
    
    # Set the limits
    plt.xlim(0, line_offset[-1])
    plt.ylim(9600, 0)
    
    # Set the labels
    plt.xlabel('Offset (m)')
    plt.ylabel('Depth (m)')

    plt.show()