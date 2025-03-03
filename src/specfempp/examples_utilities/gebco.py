"""This file just defines a function to download bathymetry data from GEBCO
and save it to a file. The link is static and the file is saved to the
current working directory.
"""


def download_gebco_bathymetry(remove_zip: bool = False, force: bool = False,
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
    else:
        print("Bathymetry data already exists. Skipping download.")
        
    # Unzip bathymetry data
    with zipfile.ZipFile("gebco_2024.zip", 'r') as zip_ref:
        # Extract to gebco subdirectory
        zip_ref.extractall("gebco")
    
    # Remove zip file
    if remove_zip:
        os.remove("gebco_2024.zip")
        
        
  
def get_bathymetry(lonmin: float, lonmax: float, latmin: float, latmax: float):
    """Get bathymetry data from GEBCO and return it as a numpy array.
    Assumes that the data has been downloaded and is in 
    ``./gebco/gebco_2024_tid.nc``.

    Parameters
    ----------
    extent : list
        region to extract bathymetry data from [lonmin, lonmax, latmin, latmax]

    Raises
    ------
    ValueError
        raised if extent is invalid
    """
    
    import numpy as np
    import netCDF4 as nc 
    
    if lonmin >= lonmax or latmin >= latmax:
        raise ValueError("Invalid extent. Must be lonmin < lonmax & latmin < latmax.")
    
    # Load bathymetry data    
    dataset = nc.Dataset('gebco/gebco_2024_TID.nc')

    # Extract variables
    lon = dataset.variables['lon'][:]
    lat = dataset.variables['lat'][:]
    
    # Get indices for extent
    nlonmin = np.argmin(np.abs(lon-lonmin)) - 1
    nlonmax = np.argmin(np.abs(lon-lonmax)) + 1
    nlatmin = np.argmin(np.abs(lat-latmin)) - 1
    nlatmax = np.argmin(np.abs(lat-latmax)) + 1
    
    # Extract bathymetry data
    bathymetry = dataset.variables['z'][nlatmin:nlatmax,nlonmin:nlonmax]
    
    return bathymetry