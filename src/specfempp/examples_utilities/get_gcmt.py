import obspy
import io

from obspy.clients.fdsn import Client as fdsnClient
from obspy.core import UTCDateTime
from obspy import read_events
import requests
import csv
import xml.etree.ElementTree as ET


def get_gcmt(starttime: UTCDateTime | None = None, 
             endtime: UTCDateTime | None = None, 
             minlon: float | None = None, maxlon: float | None = None, 
             minlat: float | None= None, maxlat: float | None = None, 
             minmag: float | None = None, maxmag: float | None = None,
             mindepth: float | None = None, maxdepth: float | None = None):
    
    # Get the events using the 
    base_url = 'http://ds.iris.edu/spudservice/momenttensor/ids?'
    
    if starttime is not None:
        base_url += f'evtstartdate={starttime.format_iris_web_service()}&'
        
    if endtime is not None:
        base_url += f'evtenddate={endtime.format_iris_web_service()}&'
        
    if minlon is not None:
        base_url += f'evtminlon={minlon}&'
        
    if maxlon is not None:
        base_url += f'evtmaxlon={maxlon}&'
        
    if minlat is not None:
        base_url += f'evtminlat={minlat}&'
        
    if maxlat is not None:
        base_url += f'evtmaxlat={maxlat}&'
        
    if minmag is not None:
        base_url += f'evtminmag={minmag}&'
        
    if maxmag is not None:
        base_url += f'evtmaxmag={maxmag}&'
        
    if mindepth is not None:
        base_url += f'evtmindepth={mindepth}&'
        
    if maxdepth is not None:
        base_url += f'evtmaxdepth={maxdepth}&'
        
    # Remove last apmersand
    base_url = base_url[:-1]
    
    # Get the events
    r = requests.get(base_url)
    
    # Get the event ids
    quake_ids = [int(line) for line in r.content.decode("utf-8").strip().split('\n')]
    
    # Get the events
    events = []
    
    # Get the event information
    for quake_id in quake_ids:
        url = f'http://ds.iris.edu/spudservice/momenttensor/{quake_id}/quakeml'
    
        try:
            r = requests.get(url)
            r.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(err)
            continue
        
        print(f'Getting event {quake_id}')
        quakeml = r.content.decode("utf-8")
        
        # Create in-memory file-like object
        f = io.StringIO(quakeml)
        
        # Read the event
        event = obspy.read_events(f)[0]
        
        # Append to the list
        events.append(event)
        
        
    return obspy.Catalog(events)