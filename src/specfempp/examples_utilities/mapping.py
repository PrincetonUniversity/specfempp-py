import numpy as np
import cartopy.geodesic as geodesic

def geo2cart(r: float or np.ndarray or list,
             theta: float or np.ndarray or list,
             phi: float or np.ndarray or list) \
    -> (float or np.ndarray or list,
        float or np.ndarray or list,
        float or np.ndarray or list):
    """Computes Cartesian coordinates from geographical coordinates.

    Parameters
    ----------
    r : float or numpy.ndarray or list
        Radius
    theta : float or numpy.ndarray or list
        Latitude (-90, 90)
    phi : float or numpy.ndarray or list
        Longitude (-180, 180)

    Returns
    -------
    float or np.ndarray or list, float or np.ndarray or list, float or np.ndarray or list
        (x, y, z)
    """

    if type(r) is list:
        r = np.array(r)
        theta = np.array(theta)
        phi = np.array(phi)

    # Convert to Radians
    thetarad = theta * np.pi/180.0
    phirad = phi * np.pi/180.0

    # Compute Transformation
    x = r * np.cos(thetarad) * np.cos(phirad)
    y = r * np.cos(thetarad) * np.sin(phirad)
    z = r * np.sin(thetarad)

    if type(r) is list:
        x = x.tolist()
        y = y.tolist()
        z = z.tolist()

    return x, y, z

def cart2geo(x: float or np.ndarray or list,
             y: float or np.ndarray or list,
             z: float or np.ndarray or list) \
    -> (float or np.ndarray or list,
        float or np.ndarray or list,
        float or np.ndarray or list):
    """Computes Cartesian coordinates from geographical coordinates.

    Parameters
    ----------
    x : float or numpy.ndarray or list
        Radius
    y : float or numpy.ndarray or list
        Latitude (-90, 90)
    z : float or numpy.ndarray or list
        Longitude (-180, 180)

    Returns
    -------
    float or np.ndarray or list, float or np.ndarray or list, float or np.ndarray or list
        (r, theta, phi)
    """

    if type(x) is list:
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

    # Compute Transformation
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)

    # Catch corner case of latitude computation
    theta = np.where((r == 0), 0, np.arcsin(z/r))

    # Convert to Radians
    theta *= 180.0/np.pi
    phi *= 180.0/np.pi

    if type(r) is list:
        r = r.tolist()
        theta = theta.tolist()
        phi = phi.tolist()

    return r, theta, phi


def get_midpoint(lat1, lon1, lat2, lon2):
    """This is dumb simple, but assumes a spherical Earth. It's not accurate for
    large distances, but it's good enough for small-ish distances."""
    
    # Gets cartesian coordinates
    x1 = geo2cart(1, lat1, lon1)
    x2 = geo2cart(1, lat2, lon2)

    # vector between x1 and x2
    dx = np.array(x2) - np.array(x1)

    # Add vector to point
    _, lat3, lon3 = cart2geo(*(x1 + 0.5 * dx))

    return lat3, lon3


def get_extent(event_lat, event_lon, station_lat, station_lon):
    """Tiny function to get the extent of the map to plot"""
    
    # Check if the map crosses the dateline
    dateline_crossing = event_lon > 0 and station_lon < 0 or event_lon < 0 and station_lon > 0
    if dateline_crossing:
        event_west_of_dateline = event_lon > 0
        if event_west_of_dateline:
            minlon = event_lon
            maxlon = station_lon
        else:
            minlon = station_lon
            maxlon = event_lon
    else:
        minlon = min(event_lon, station_lon)
        maxlon = max(event_lon, station_lon)
    
    minlat = min(event_lat, station_lat)
    maxlat = max(event_lat, station_lat)
            
    return [minlon, maxlon, minlat, maxlat]


def fix_extent(extent, fraction=0.05, min_ratio=0.5, max_ratio=2.0):
    """
    Fix map extent by adding a buffer around it and ensuring the aspect ratio
    is within the specified bounds (between 1:2 and 2:1 by default).
    
    The function:
    1. Applies a 5% buffer to the longer dimension
    2. Adjusts the shorter dimension to achieve a 2:1 (or 1:2) aspect ratio
    
    Parameters:
    -----------
    extent : list
        [minlon, maxlon, minlat, maxlat]
    fraction : float
        Fraction of the extent to add as buffer for the longer dimension (default: 0.05)
    min_ratio : float
        Minimum allowed aspect ratio (width:height) (default: 0.5, which is 1:2)
    max_ratio : float
        Maximum allowed aspect ratio (width:height) (default: 2.0, which is 2:1)
        
    Returns:
    --------
    list
        Fixed extent [minlon, maxlon, minlat, maxlat]
    """
    from copy import deepcopy
    
    # Get extent values
    minlon, maxlon, minlat, maxlat = deepcopy(extent)
    
    # Normalize input longitudes to -180 to 180 range
    minlon = ((minlon + 180) % 360) - 180
    maxlon = ((maxlon + 180) % 360) - 180
    
    # Check for dateline crossing
    dateline_crossing = minlon > maxlon
    
    # Calculate original width and height
    if dateline_crossing:
        width = 360 - (minlon - maxlon)
    else:
        width = maxlon - minlon
    
    height = maxlat - minlat
    
    # Calculate original aspect ratio (width:height)
    original_aspect_ratio = width / height
    
    # If width is very large (almost 360 degrees), it's likely a global extent
    # In that case, limit it to a reasonable value
    if width > 350:
        # This is almost a global extent, limit it to half the globe
        lon_center = 0  # Center on the prime meridian
        width = min(width, 180)  # Limit to half the globe
        minlon = lon_center - width/2
        maxlon = lon_center + width/2
        dateline_crossing = False  # Reset dateline crossing status
    
    # Step 1: Add buffer to the longer dimension
    if width >= height:  # Width is longer (or equal)
        # Add buffer to longitude (the longer dimension)
        if dateline_crossing:
            lonb = (360 - (minlon - maxlon)) * fraction
            minlon -= lonb
            maxlon += lonb
        else:
            lonb = (maxlon - minlon) * fraction
            minlon -= lonb
            maxlon += lonb
            
        # Update width after buffer
        if dateline_crossing:
            width = 360 - (minlon - maxlon)
        else:
            width = maxlon - minlon
            
        # Step 2: Adjust height to achieve 2:1 ratio (width:height)
        target_height = width / max_ratio  # For 2:1 ratio
        
        # If current height is less than target, expand it
        if height < target_height:
            height_diff = target_height - height
            lat_expansion = height_diff / 2
            
            # Expand latitude in both directions
            minlat -= lat_expansion
            maxlat += lat_expansion
    else:  # Height is longer
        # Add buffer to latitude (the longer dimension)
        latb = (maxlat - minlat) * fraction
        minlat -= latb
        maxlat += latb
        
        # Update height after buffer
        height = maxlat - minlat
        
        # Step 2: Adjust width to achieve 1:2 ratio (width:height)
        target_width = height * min_ratio  # For 1:2 ratio
        
        # If current width is less than target, expand it
        if width < target_width:
            width_diff = target_width - width
            lon_expansion = width_diff / 2
            
            # Expand longitude in both directions
            if dateline_crossing:
                minlon -= lon_expansion
                maxlon += lon_expansion
            else:
                minlon -= lon_expansion
                maxlon += lon_expansion
    
    # Ensure latitude values stay within valid range
    minlat = max(-90, minlat)
    maxlat = min(90, maxlat)
    
    # Normalize longitude values after all adjustments
    minlon = ((minlon + 180) % 360) - 180
    maxlon = ((maxlon + 180) % 360) - 180
    
    # Ensure longitudes are strictly within -180 to 180 range
    minlon = max(-180, min(180, minlon))
    maxlon = max(-180, min(180, maxlon))
    
    # Ensure proper dateline crossing status is maintained
    if dateline_crossing and minlon <= maxlon:
        # If we lost the dateline crossing, restore it
        if minlon >= 0:  # Both positive
            minlon = -180
            maxlon = 180
        else:  # Both negative or one on each side
            maxlon = 180
    elif not dateline_crossing and minlon > maxlon:
        # If we gained dateline crossing incorrectly, fix it
        if maxlon < 0 and minlon > 0:
            # One on each side of dateline
            maxlon += 360
            if maxlon > 180:
                maxlon = 180
    
    return [minlon, maxlon, minlat, maxlat]


def reckon(lat1, lon1, distance, azimuth):
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
    bearing = azimuth
    
    # Cartopy's geodesic calculator
    geod = geodesic.Geodesic()
    
    degrees_to_meters = 6371000 * np.pi / 360  # Approximate value
    
    # For scalar inputs
    if isinstance(distance, (int, float)):
        # Convert degrees to meters (approx 111.32 km per degree)
        dist_m = distance * degrees_to_meters  # meters
        
        # Calculate the destination point
        result = geod.direct(np.array([[lon1, lat1]]), bearing, dist_m)
        return result[0, 1], result[0, 0]
    
    # For array inputs
    # Convert degrees to meters
    dist_m = distance * degrees_to_meters  # meters
    
    # Initialize arrays for results
    lats = np.zeros_like(distance)
    lons = np.zeros_like(distance)
    
    # Process each point
    for i, d in enumerate(dist_m):
        result = geod.direct(np.array([[lon1, lat1]]), bearing, d)
        lons[i] = result[0, 0]
        lats[i] = result[0, 1]
    
    return lats, lons


def get_line(lat, lon, width, npts, az):
    # Earth's radius in meter
    R = 6371000.0
    
    # convert meters to degrees
    m2deg = 180.0 / (np.pi * R)
    
    # x-coordinate of the bathymetry
    x = np.linspace(-width/2, width/2, npts)
    
    # latitude and longitude of the bathymetry
    lats, lons = reckon(lat, lon, x * m2deg, az)
    
    return x, lats, lons


def plot_station_event_geometry(event_lat, event_lon, station_lat, station_lon):
    
    import pygmt
    import numpy as np
    
    # Compute the midpoint between the event and the station
    midpoint_lat, midpoint_lon = get_midpoint(
        event_lat, event_lon, station_lat, station_lon)

    # Create a new figure
    fig = pygmt.Figure()

    # Set the region to global and use an orthographic projection centered at
    # specified location
    projection = f"G{midpoint_lon}/{0.0}/15c"
    fig.basemap(region="g", projection=projection, frame=True)

    # Add Earth relief data (equivalent to stock_img in  )
    fig.grdimage(grid="@earth_relief_10m", shading=False, cmap='terra', 
                 transparency=10, projection=projection)

    # Plot geodesic line between event and station
    fig.plot(
        x=[event_lon, station_lon],
        y=[event_lat, station_lat],
        pen="1p,black",
        projection=projection
    )

    # Add event marker (star)
    fig.plot(
        x=event_lon,
        y=event_lat,
        style="a0.5c",  # 'a' for star, 0.5c is the size
        fill="blue",
        pen="1p,black",
        projection=projection
    )

    # Add station marker (triangle)
    fig.plot(
        x=station_lon,
        y=station_lat,
        style="t0.5c",  # '' for triangle, 0.5c is the size
        fill="red",
        pen="1p,black",  # black outline
        projection=projection
    )

    # Add colorbar
    fig.colorbar(
        # Position: right side of map with offset
        position="JMR+o1c/0c+w15c/0.5c",  
        # Frame settings with labels
        frame=["af", "x+lElevation (m)"],  
        box=False,  # Draw a box around the colorbar
    )

    # Show the figure
    return fig
