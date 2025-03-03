import requests
import sys
import os
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def download_with_progress(url, filename=None, chunksize=1024*1024, max_retries=5, timeout=30,
                           max_function_retry=5):
    """
    Download a file from url with a simple progress bar and robust connection handling
    
    Parameters:
    url (str): URL to download
    filename (str): Output filename
    chunksize (int): Size of chunks to download
    max_retries (int): Maximum number of retries for connection errors
    timeout (int): Connection timeout in seconds
    max_function_retry (int): Maximum number of retries for the function itself
    """
    # If no filename provided, use the last part of the URL
    if filename is None:
        filename = os.path.basename(url)
    
    # Create a session with retry capability
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=0.5,
        status_forcelist=[502, 503, 504, 429],
        allowed_methods=["GET"]
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    # Check if file exists to support resume
    file_size = 0
    headers = {}
    
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        headers['Range'] = f'bytes={file_size}-'
        print(f"Resuming download from {format_size(file_size)}")
    
    # Make the request with extended timeout for large files
    for current_try in range(max_function_retry):
        
        # Running the dowload once
        try:
            response = session.get(
                url, 
                stream=True, 
                headers=headers,
                timeout=timeout
            )
            
            # Handle non-206 response when resume is requested
            if file_size > 0 and response.status_code == 200:
                # Server doesn't support resume, start over
                file_size = 0
                print("Server doesn't support resume, starting from beginning")
            
            # Raise an error for bad responses
            response.raise_for_status()
            
            # Get the total file size
            if 'content-length' in response.headers:
                total_size = int(response.headers.get('content-length', 0)) + file_size
            else:
                # If content-length is not provided, we can't show progress percentage
                total_size = None
                print("Warning: Server did not provide content length, progress will be indeterminate")
            
            # Initialize variables for progress tracking
            downloaded = file_size
            start_time = time.time()
            last_update_time = start_time
            last_downloaded = downloaded
            progress_bar_length = 30
            
            # Open file in append mode if resuming, otherwise write mode
            mode = 'ab' if file_size > 0 else 'wb'
            
            # Open a file to write the content
            with open(filename, mode) as file:
                print(f"Downloading {filename}:")
                
                # Iterate over the response in chunks
                for chunk in response.iter_content(chunk_size=chunksize):
                    if chunk:  # filter out keep-alive chunks
                        file.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress bar at most once per 0.5 seconds
                        current_time = time.time()
                        if current_time - last_update_time > 0.5:
                            # Calculate speed
                            speed = (downloaded - last_downloaded) / (current_time - last_update_time)
                            last_update_time = current_time
                            last_downloaded = downloaded
                            
                            # Only show percentage if we know the total size
                            if total_size:
                                # Calculate progress percentage
                                percent = downloaded / total_size
                                
                                # Calculate estimated time remaining
                                elapsed_time = current_time - start_time
                                if speed > 0:
                                    remaining = (total_size - downloaded) / speed
                                else:
                                    remaining = 0
                                
                                # Create the progress bar visualization
                                filled_length = int(progress_bar_length * percent)
                                bar = '#' * filled_length + '-' * (progress_bar_length - filled_length)
                                
                                # Create progress message
                                progress_msg = f"\r|{bar}| {percent:6.1%} {format_size(downloaded)}/{format_size(total_size)} "
                                progress_msg += f"[{format_size(speed)}/s, ETA: {format_time(remaining)}]"
                            else:
                                # Indeterminate progress
                                # Create a moving animation
                                position = int((current_time * 2) % (progress_bar_length - 3))
                                bar = '-' * position + '###' + '-' * (progress_bar_length - position - 3)
                                
                                # Create progress message without percentage
                                progress_msg = f"\r|{bar}| {format_size(downloaded)} downloaded [{format_size(speed)}/s]"
                            
                            # Print the progress bar (overwriting the previous one)
                            sys.stdout.write(progress_msg)
                            sys.stdout.flush()
                
                # Print a newline after the download is complete
                sys.stdout.write('\n')
            
            print(f"Downloaded {filename} ({format_size(downloaded)}) in {time.time() - start_time:.1f}s")
            return filename
            
        except requests.exceptions.RequestException as e:
            print(f"\nError during download: {e}")
            if current_try == max_function_retry - 1:
              print(f"Downloaded {format_size(downloaded)} so far. You can resume the download later.")
              break
            else:
              print(f"Downloaded {format_size(downloaded)} so far retrying... {current_try + 1}/{max_function_retry}")
              time.sleep(2)
            # Don't delete the partial file so it can be resumed

        if downloaded == total_size:
            # Download completed successfully
            return filename
          
    # If we reach this point, the download failed
    print(f"Failed to download {filename} after {max_function_retry} retries")
    return None

def format_size(size_in_bytes):
    """Format size in bytes to human readable format"""
    if size_in_bytes is None:
        return "Unknown"
        
    size_in_bytes = max(0, size_in_bytes)
    for unit in ['B ', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024 or unit == 'TB':
            return f"{size_in_bytes:7.2f} {unit}"
        size_in_bytes /= 1024

def format_time(seconds):
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{int(seconds):>7}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
        return f"{minutes:>2}m {seconds:>2}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours:>2}h {minutes:>2}m"