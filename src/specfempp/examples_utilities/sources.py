
import numpy as np


def create_sources(angle, vp: float, 
                   ix: float = 200.0, dx: float = 100.0, fx: float = 20000.0,
                   iz: float = 720.0, dz: float = 100.0, fz: float = 9600.0,
                   bathymetry_in_m: np.ndarray | None = None,
                   bathymetry_offset_in_m: np.ndarray | None = None,
                   ):
    """Create sources for the simulation.

    Parameters
    ----------
    params : dict
        Dictionary containing the simulation parameters.
    freq : float
        Dominant frequency of the source.
    angle : float
        Angle of the source in degrees.

    Returns
    -------
    sources : list
        List of sources to be used in the simulation.
    """
    # use multiple sources to imitate plane wave
    
    
    sources = []
    xcount = 0
    zcount = 0
    
    for i in range(197):
        # Compute time shift based on the angle and vp
        tshift = i * dx * np.sin(angle * np.pi / 180) / vp
        x = ix + i * dx
        
        if x > fx:
            break
        moment_tensor = {"moment-tensor": dict()}
        moment_tensor["moment-tensor"]["x"] = float(x)
        moment_tensor["moment-tensor"]["z"] = float(iz)
        moment_tensor["moment-tensor"]["Mxx"] = 1.0
        moment_tensor["moment-tensor"]["Mzz"] = 1.0
        moment_tensor["moment-tensor"]["Mxz"] = 0.0
        moment_tensor["moment-tensor"]["angle"] = 0.0
        moment_tensor["moment-tensor"]["Ricker"] = dict()
        moment_tensor["moment-tensor"]["Ricker"]["factor"] = float(1e-9 * np.cos(angle * np.pi / 180))
        moment_tensor["moment-tensor"]["Ricker"]["tshift"] = float(tshift)
        moment_tensor["moment-tensor"]["Ricker"]["f0"] = 1.0
        xcount += 1
        sources.append(moment_tensor)
    
    # Add more sources if angle is 5 degrees or above
    if angle >= 5:
        
        tshift = 0
        
        if bathymetry_in_m is not None and bathymetry_offset_in_m is not None:
            from scipy.interpolate import interp1d
            bathymetry_interp = interp1d(bathymetry_offset_in_m, bathymetry_in_m)

        if bathymetry_in_m is not None and bathymetry_offset_in_m is not None:
            bathy_at_x = bathymetry_interp(x)
        
        for ii in range(50):
            z = iz + ii * dz

            
            # Interpolate the bathymetry if available
            if bathymetry_in_m is not None and bathymetry_offset_in_m is not None:
                
                # For debugging
                # print(f"z: {z}, ip_z: {bathy_at_x}")
            
                # Stop of the source is above the bathymetry 
                # Since z is counted upwards, we need to check if the source is 
                # we need to subtract the bathymetry from the top interface
                # to get zoffset from bottom interface
                # The dz buffer is added to make sure the source is not too 
                # close to the bathymetry
                if (fz - bathy_at_x - dz) < z:
                    break
            # Otherwise break if the source is above the top interface
            else:
                if z > fz:
                    break
            
            # Compute the vertical timeshift
            tshift = (ii+1) * dz * np.cos(angle * np.pi / 180) / vp
            
            moment_tensor = {"moment-tensor": dict()}
            moment_tensor["moment-tensor"]["x"] = float(ix)
            moment_tensor["moment-tensor"]["z"] = float(z)
            moment_tensor["moment-tensor"]["Mxx"] = 1.0
            moment_tensor["moment-tensor"]["Mzz"] = 1.0
            moment_tensor["moment-tensor"]["Mxz"] = 0.0
            moment_tensor["moment-tensor"]["angle"] = 0.0
            moment_tensor["moment-tensor"]["Ricker"] = dict()
            moment_tensor["moment-tensor"]["Ricker"]["factor"] = float(1e-10)
            moment_tensor["moment-tensor"]["Ricker"]["tshift"] = float(tshift)
            moment_tensor["moment-tensor"]["Ricker"]["f0"] = 1.0
            zcount += 1
            sources.append(moment_tensor)
            
    return {"sources": sources, "number-of-sources": len(sources)}


def plot_sources(sources: dict):
    
    import matplotlib.pyplot as plt

    # plot sources
    x = [source['moment-tensor']['x'] for source in sources['sources']]
    z = [source['moment-tensor']['z'] for source in sources['sources']]
    t = [source['moment-tensor']['Ricker']['tshift'] for source in sources['sources']]

    # Create and show figure
    plt.figure()
    plt.scatter(x, z, c=t, cmap='viridis')
    c = plt.colorbar()
    c.set_label('Time shift (s)')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.title('Source locations')
    plt.axis('equal')
    plt.show()