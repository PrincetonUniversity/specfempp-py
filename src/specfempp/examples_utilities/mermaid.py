header = """
# This is the header for the generic files
http://geoweb.princeton.edu/people/simons/SOM/P0004_all.txt
http://geoweb.princeton.edu/people/simons/SOM/P0049_all.txt
http://geoweb.princeton.edu/people/simons/SOM/P0040_030.txt

# A typical line is
P0023   08-Oct-2018 10:07:48   -24.061967  -140.922033   0.630  1.100    14626  13873    78711   328   30     7   0   0
P0050   29-Aug-2019 19:01:05   -11.599950  -136.425650   0.680  1.000    14722  14002    80967    22   10    12   5   0

# The files are generated from MERMAID vital files by the code VIT2TBL

# Formats are as follows:

%s      5 characters, e.g. P0023
%s      A datetime string, e.g. 27-Jul-2019 21:59:05
%11.6f  Instrument latitude in decimal degrees
%12.6f  Instrument longitude in decimal degrees
%7.3f   Horizontal dilution of precision
%7.3f   Vertical dilution of precision
%6i     Battery level in mV
%6i     Minimum voltage in mV
%6i     Internal pressure in Pa
%6i     External pressure in mbar
%5i     Pressure range in mbar
%3i     Number of commands received
%3i     Number of files queued for upload
%3i     Number of files uploaded
"""
get_month = {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12
        }

import numpy as np
import obspy

def flexfloat(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return np.nan

def flexint(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return np.nan


def download_position(mermaid_id: str, last_30_only: bool = False):    
    """
    
    
    
    Note
    ----
    
    The file is located at
    `https://geoweb.princeton.edu/people/simons/SOM/all.txt`.
    
    .. code-block:: text
    
        % This is the header for the generic files
        http://geoweb.princeton.edu/people/simons/SOM/P0004_all.txt
        http://geoweb.princeton.edu/people/simons/SOM/P0049_all.txt
        http://geoweb.princeton.edu/people/simons/SOM/P0040_030.txt

        % A typical line is 
        P0023   08-Oct-2018 10:07:48   -24.061967  -140.922033   0.630  1.100    14626 13873 78711   328   30     7   0   0 
        P0050   29-Aug-2019 19:01:05   -11.599950  -136.425650   0.680  1.000    14722 14002 80967    22   10    12   5   0

        % The files are generated from MERMAID vital files by the code
        VIT2TBL

        % Formats are as follows:

        %s      5 characters, e.g. P0023 
        %s      A datetime string, e.g. 27-Jul-2019 21:59:05 
        %11.6f  Instrument latitude in decimal degrees
        %12.6f  Instrument longitude in decimal degrees 
        %7.3f   Horizontal dilution of precision 
        %7.3f   Vertical dilution of precision 
        %6i     Battery level in mV 
        %6i     Minimum voltage in mV 
        %6i     Internal pressure in Pa 
        %6i     External pressure in mbar 
        %5i     Pressure range in mbar 
        %3i     Number of commands received 
        %3i     Number of files queued for upload 
        %3i     Number of files uploaded

    """
    
    import requests
    
    # Get URL for the data
    add = "all" if not last_30_only else "030"
    url = f"https://geoweb.princeton.edu/people/simons/SOM/{mermaid_id}_{add}.txt"
    
    # Get the data
    try: 
        r = requests.get(url)
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
        raise ValueError("Could not download the data.")    
    
    lines = r.content.decode("utf-8").strip().split('\n')
    
    data = dict()
    data["url"] = url
    data["header"] = header
    data["id"] = []
    data["datetime"] = []
    data["lat"] = []
    data["lon"] = []
    data["hdop"] = []
    data["vdop"] = []
    data["battery"] = []
    data["min_voltage"] = []
    data["int_pressure"] = []
    data["ext_pressure"] = []
    data["pressure_range"] = []
    data["commands"] = []
    data["queued"] = []
    data["uploaded"] = []
    
    for line in lines:
        
        id = line[0:7]
        date, time = line[8:28].split()
        day, month, year = date.split('-')
        
        # See dictionary above
        month = get_month[month]

        hour, minute, second = time.split(':')
        datetime = obspy.UTCDateTime(year=int(year), month=int(month), day=int(day), 
                                     hour=int(hour), minute=int(minute), second=int(second))
        lat = flexfloat(line[29:40])
        lon = flexfloat(line[41:53])
        hdop = flexfloat(line[54:61])
        vdop = flexfloat(line[62:69])
        battery = flexint(line[70:76])
        min_voltage = flexint(line[77:83])
        int_pressure = flexint(line[84:90])
        ext_pressure = flexint(line[91:97])
        pressure_range = flexint(line[98:103])
        commands = flexint(line[104:107])
        queued = flexint(line[108:111])
        uploaded = flexint(line[112:115])
        
        
        data["id"].append(id)
        data["datetime"].append(datetime.matplotlib_date)
        data["lat"].append(lat)
        data["lon"].append(lon)
        data["hdop"].append(hdop)
        data["vdop"].append(vdop)
        data["battery"].append(battery)
        data["min_voltage"].append(min_voltage)
        data["int_pressure"].append(int_pressure)
        data["ext_pressure"].append(ext_pressure)
        data["pressure_range"].append(pressure_range)
        data["commands"].append(commands)
        data["queued"].append(queued)
        data["uploaded"].append(uploaded)
        
        
    # Convert numeric data to numpy arrays
    data["datetime"] = np.array(data["datetime"])
    data["lat"] = np.array(data["lat"])
    data["lon"] = np.array(data["lon"])
    data["hdop"] = np.array(data["hdop"])
    data["vdop"] = np.array(data["vdop"])
    data["battery"] = np.array(data["battery"])
    data["min_voltage"] = np.array(data["min_voltage"])
    data["int_pressure"] = np.array(data["int_pressure"])
    data["ext_pressure"] = np.array(data["ext_pressure"])
    data["pressure_range"] = np.array(data["pressure_range"])
    data["commands"] = np.array(data["commands"])
    data["queued"] = np.array(data["queued"])
    data["uploaded"] = np.array(data["uploaded"])
    
    return data



def get_location(mermaid_data: dict, datetime: obspy.UTCDateTime):
    """
    Get the location of the MERMAID at a given time.
    
    Parameters
    ----------
    mermaid_data : dict
        dictionary with MERMAID data
    datetime : obspy.UTCDateTime
        datetime to get the location for
        
    Returns
    -------
    tuple(3) of float
        lon, lat, pressure
    """
    
    # Check if the datetime is within the range of the data
    if datetime.matplotlib_date < mermaid_data["datetime"][0] or datetime.matplotlib_date > mermaid_data["datetime"][-1]:
        raise ValueError("The datetime is not within the range of the data provided.")

    # Get the index of the datetime before and after the given datetime
    idx_before = np.where(mermaid_data["datetime"] < datetime.matplotlib_date)[0][-1]
    idx_after = np.where(mermaid_data["datetime"] > datetime.matplotlib_date)[0][0]
    
    # Interpolate the time and location to get the location at the given time
    pos1 = np.array([mermaid_data["lon"][idx_before], mermaid_data["lat"][idx_before]])
    pos2 = np.array([mermaid_data["lon"][idx_after], mermaid_data["lat"][idx_after]])
    
    time1 = mermaid_data["datetime"][idx_before]
    time2 = mermaid_data["datetime"][idx_after]
    
    # Interpolate the position
    new_pos = pos1 + (pos2 - pos1) * (datetime.matplotlib_date - time1) / (time2 - time1)
    
    return new_pos[0], new_pos[1], 1500.0


def _get_extent(lats, lons):
    """adapt above function for vector of lats and lons

    Parameters:
    ----------
    lats : np.ndarray
        latitudes
    lons : np.ndarray
        longitudes
        
    """
    
    # Check if the map crosses the dateline
    dateline_crossing = np.any(lons > 0) and np.any(lons < 0)
    if dateline_crossing:
        event_west_of_dateline = np.any(lons > 0)
        if event_west_of_dateline:
            minlon = np.min(lons[lons > 0])
            maxlon = np.max(lons[lons < 0])
        else:
            minlon = np.min(lons[lons < 0])
            maxlon = np.max(lons[lons > 0])
    else:
        minlon = np.min(lons)
        maxlon = np.max(lons)
    
    minlat = np.min(lats)
    maxlat = np.max(lats)
    
    return minlon, maxlon, minlat, maxlat
    
    

def get_extent(mermaid_data: dict):
    """
    Get the extent of the MERMAID data
    
    Parameters
    ----------
    mermaid_data : dict
        dictionary with MERMAID data
        
    Returns
    -------
    tuple(4) of float
        lonmin, lonmax, latmin, latmax
    """
    
    return _get_extent(mermaid_data["lat"], mermaid_data["lon"])