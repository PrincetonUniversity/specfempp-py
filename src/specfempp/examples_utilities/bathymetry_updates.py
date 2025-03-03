"""File: bathtymetry_updates.py

Description:
------------

This file contains functions that handle the file updates that need to be made
to update the extent of the mesh for the dynamic bathymetry problem.

"""

import numpy as np

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


def update_par_file_extent(par_dict, offset_in_m, depth_in_m, bathymetry_in_m):
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
        f.write(f" 0 0\n")
        f.write(f" {int(offset_in_m)} 0\n")
        
        # Write interface 2 header
        f.write("#\n# interface number 2 (bathymetry)\n#\n")
        
        # Write number of nodes for the top of the mesh
        f.write(f" {N_bathymetry}\n")
        
        # Write the bathymetry data
        for i in range(N_bathymetry):
            f.write(f" {int(bathymetry_offset_in_m[i])} {int(bathymetry_in_m[i])}\n")
        
        # Write interface 3 header
        f.write("#\n#write interface 3 (top of the mesh)\n")
        
        # Write interface 3 (top of the mesh)
        f.write(f" {2}\n")
        f.write(f" 0 {int(depth_in_m)}\n")
        f.write(f" {int(offset_in_m)} {int(depth_in_m)}\n")
        
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
        

        
        