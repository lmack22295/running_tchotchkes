import pandas as pd
import numpy as np
import polyline
import matplotlib.pyplot as plt
import folium
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import math
import matplotlib.patheffects as pe
from datetime import datetime
import pdb
import time
import os
import requests
import streamlit as st
from urllib.parse import urlparse, parse_qs

FILE_NAME = 'all_strava_runs_lance_mack_2025_02_04.pkl'
DISTANCE = 'distance'

def load_runs():
    with open(FILE_NAME, 'rb') as file: 
        df = pickle.load(file)
    
    # remove 0 distance runs
    df = df.loc[df[DISTANCE] > 0]

    # remove empty maps
    df['missing_polyline'] = df['map'].apply(lambda x: x['summary_polyline']=='')
    df = df.loc[df.missing_polyline==False]

    lats = []
    longs = []
    for i, row in df.iterrows():
        lat,long = zip(*polyline.decode(row['map']['summary_polyline']))
        lats.append(lat)
        longs.append(long)

    df['lats'] = lats
    df['longs'] = longs
    return df

class Visualize:
    def __init__(self):
        self.runs = load_runs() 

    def set_curr_run(self,id=0):
        self.curr_run = {'lats': self.runs.iloc[id]['lats'], 'lons': self.runs.iloc[id]['longs']}

    def get_curr_run(self):
        return self.curr_run

def get_run_data(run_df, run_id):
    curr_run = run_df.iloc[run_id]
    coords = polyline.decode(curr_run['map']['summary_polyline'])
    lats, lons = zip(*coords)
    return lats, lons

def plot_run_on_map(lats, longs, margin=0.002):
    # Create figure and projection
    fig, ax = plt.subplots(figsize=(12, 12),
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Calculate bounds for the map
    min_lon, max_lon = min(longs) - margin, max(longs) + margin
    min_lat, max_lat = min(lats) - margin, max(lats) + margin
    
    # Set map bounds
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])
    
    # Add OpenStreetMap tiles with higher zoom level
    osm_tiles = cimgt.OSM()
    ax.add_image(osm_tiles, 15)  # Increased zoom level for more detail
    
    # Plot the running route with a more prominent style
    ax.plot(longs, lats, 
           color='#FC4C02',  # Strava orange color
           linewidth=3,
           transform=ccrs.PlateCarree(),
           zorder=4,
           alpha=0.8)
    
    # Add start and end markers
    ax.plot(longs[0], lats[0], 'go', markersize=10, label='Start', 
           transform=ccrs.PlateCarree(), zorder=5)
    ax.plot(longs[-1], lats[-1], 'ro', markersize=10, label='End', 
           transform=ccrs.PlateCarree(), zorder=5)
    
    # Remove axis labels and ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend with a semi-transparent background
    ax.legend(framealpha=0.5)
    
    return fig

def format_pace(pace_minutes):
    """Convert decimal minutes to MM:SS format"""
    minutes = int(pace_minutes)
    seconds = int((pace_minutes - minutes) * 60)
    return f"{minutes}:{seconds:02d}"

def format_date(date_str):
    """Convert Strava date format to yyyy-mm-dd"""
    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    return date_obj.strftime('%Y-%m-%d')

def calculate_scale(run_data_dict, scale_factor=0.6):  # Reduced from 1.2 to 0.6 for tighter zoom
    """Calculate a consistent scale for all maps based on the largest run"""
    max_span = 0
    for lats, longs in run_data_dict.values():
        lat_span = max(lats) - min(lats)
        lon_span = max(longs) - min(longs)
        max_span = max(max_span, lat_span, lon_span)
    return max_span * scale_factor / 2

def create_run_grid(run_data_dict, run_info_dict, output_filename='run_grid.png'):
    """Create a grid of run visualizations with additional info"""
    import math
    
    # Calculate grid dimensions based on number of runs
    n_runs = len(run_data_dict)
    n_cols = min(6, n_runs)  # 6 columns max
    n_rows = math.ceil(n_runs / n_cols)
    
    # Create figure with controlled size
    fig = plt.figure(figsize=(5*n_cols, 5*n_rows))
    
    # Adjust margins to minimum
    plt.subplots_adjust(hspace=0, wspace=0)
    
    # Sort run IDs by date
    sorted_run_ids = sorted(run_data_dict.keys())
    
    for idx, run_id in enumerate(sorted_run_ids):
        # Create subplot with map projection
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection=ccrs.Mercator())
        ax.set_aspect('auto')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        ax.set_axis_off()
        
        # Calculate the center and range of coordinates
        center_long = (max(run_data_dict[run_id][1]) + min(run_data_dict[run_id][1])) / 2
        center_lat = (max(run_data_dict[run_id][0]) + min(run_data_dict[run_id][0])) / 2
        
        # Calculate the range with a buffer
        long_range = (max(run_data_dict[run_id][1]) - min(run_data_dict[run_id][1])) * 1.1  # 10% buffer
        lat_range = (max(run_data_dict[run_id][0]) - min(run_data_dict[run_id][0])) * 2
        
        # Use the larger range to maintain square aspect
        max_range = max(long_range, lat_range)
        
        # Set consistent bounds
        ax.set_extent([
            center_long - max_range/2,
            center_long + max_range/2,
            center_lat - max_range/2,
            center_lat + max_range/2
        ], crs=ccrs.PlateCarree())
        
        # Rest of the plotting code remains the same...
        osm_tiles = cimgt.OSM()
        ax.add_image(osm_tiles, 13)
        
        # Add a semi-transparent white overlay
        ax.patch.set_facecolor('white')
        ax.patch.set_alpha(0.4)
        
        # Plot lines
        ax.plot(run_data_dict[run_id][1], run_data_dict[run_id][0], 
               color='black',
               linewidth=8,
               transform=ccrs.PlateCarree(),
               zorder=5,
               alpha=1.0,
               solid_capstyle='round')
        
        ax.plot(run_data_dict[run_id][1], run_data_dict[run_id][0], 
               color='#FC4C02',
               linewidth=6,
               transform=ccrs.PlateCarree(),
               zorder=6,
               alpha=1.0,
               solid_capstyle='round')
        
        # Add text with consistent positioning
        distance, pace, date, name = run_info_dict[run_id]
        bbox_props = dict(facecolor='white', 
                         alpha=0.7,
                         edgecolor='none',
                         pad=3.0)
        
        # Add text elements
        ax.text(0.05, 0.95, date,
                transform=ax.transAxes,
                fontsize=14,
                fontweight='bold',
                color='#FC4C02',
                ha='left',
                va='top',
                bbox=bbox_props,
                zorder=7)
        
        ax.text(0.05, 0.90, name,
                transform=ax.transAxes,
                fontsize=14,
                fontweight='bold',
                color='#FC4C02',
                ha='left',
                va='top',
                bbox=bbox_props,
                zorder=7)
                
        ax.text(0.05, 0.85, f"{distance:.1f} mi | {pace}",
                transform=ax.transAxes,
                fontsize=12,
                color='#FC4C02',
                ha='left',
                va='top',
                bbox=bbox_props,
                zorder=7)
    
    # Save with consistent DPI
    plt.savefig(output_filename, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_test_date_range(test_df, start_date, end_date):
    """Generate a grid using runs between start_date and end_date (inclusive)"""
    # Add formatted columns

    test_df['date_formatted'] = pd.to_datetime(test_df['start_date']).dt.strftime('%Y-%m-%d')
    test_df['pace_formatted'] = test_df['pace_per_mi'].apply(format_pace)

    # Convert string dates to datetime objects
    from datetime import datetime
    if isinstance(start_date, str):
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start_dt = datetime.combine(start_date, datetime.min.time())
        
    if isinstance(end_date, str):
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_dt = datetime.combine(end_date, datetime.max.time())
    
    # Filter runs within date range using datetime objects
    mask = (pd.to_datetime(test_df['date_formatted']) >= start_dt) & \
           (pd.to_datetime(test_df['date_formatted']) <= end_dt)
    date_filtered_df = test_df[mask]
    
    # Prepare data for visualization
    run_data = {}
    run_info = {}

    for run_id in date_filtered_df.index:
        try:
            run_data[run_id] = (test_df.loc[run_id, 'lats'], test_df.loc[run_id, 'longs'])
            run_info[run_id] = (
                test_df.loc[run_id, 'distance_mi'],
                test_df.loc[run_id, 'pace_formatted'],
                test_df.loc[run_id, 'date_formatted'],
                test_df.loc[run_id, 'name']
            )
        except Exception as e:
            print(f"Skipping run {run_id} due to error: {str(e)}")
            continue
    
    if not run_data:
        print("No valid runs found in the specified date range")
        return
    
    create_run_grid(run_data, run_info, f'test_date_range.png')

def create_ui():
    """Create simple UI for run visualization using Streamlit"""
    import streamlit as st
    import time
    import os
    from datetime import datetime
    
    st.title("Run Visualization Generator")
    
    # Create date input fields
    start_date = st.date_input(
        "Start Date",
        value=datetime.now(),
        format="YYYY-MM-DD"
    )
    
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        format="YYYY-MM-DD"
    )
    
    # Create generate button
    if st.button("Generate Visualization"):
        try:
            if start_date > end_date:
                st.error("Start date must be before end date")
                return
                
            df = load_runs()
            
            # Convert dates to string format
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Generate visualization and get the output path
            generate_test_date_range(df, start_date_str, end_date_str)
            
            # Small delay to ensure file is saved
            time.sleep(1)
            
            # Check if file exists and display
            if os.path.exists('test_date_range.png'):
                st.success(f"Generated visualization for runs between {start_date_str} and {end_date_str}")
                st.image('test_date_range.png')
            else:
                st.error(f"Could not find generated image: test_date_range.png")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def fetch_strava_runs(access_token, start_date=None, end_date=None):
    """
    Fetch runs from Strava API for authenticated user
    
    Parameters:
    access_token: str, Strava API access token
    start_date: datetime.date, optional start date filter
    end_date: datetime.date, optional end date filter
    
    Returns:
    DataFrame containing run data
    """
    import requests
    import pandas as pd
    from datetime import datetime
    import polyline
    
    # API endpoint for activities
    url = "https://www.strava.com/api/v3/athlete/activities"
    
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {
        'per_page': 200,  # Max activities per page
        'page': 1
    }
    
    # Convert date objects to timestamps
    if start_date:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        params['after'] = int(start_datetime.timestamp())
    if end_date:
        end_datetime = datetime.combine(end_date, datetime.max.time())
        params['before'] = int(end_datetime.timestamp())
    
    all_activities = []
    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}")
            
        activities = response.json()
        if not activities:  # No more activities
            break
            
        # Filter for runs only
        runs = [activity for activity in activities if activity['type'] == 'Run']
        
        for run in runs:
            # Decode polyline to get coordinates
            coords = polyline.decode(run['map']['summary_polyline']) if run['map']['summary_polyline'] else []
            
            if coords:  # Only include runs with valid coordinates
                # Get distance in miles and moving time in seconds
                distance_miles = run['distance'] * 0.000621371  # Convert meters to miles
                moving_time_seconds = run['moving_time']
                
                # Calculate minutes per mile pace
                minutes_per_mile = (moving_time_seconds / 60) / distance_miles
                
                # Convert to minutes and seconds
                minutes = int(minutes_per_mile)
                seconds = int((minutes_per_mile - minutes) * 60)
                
                all_activities.append({
                    'id': run['id'],
                    'name': run['name'],
                    'start_date': run['start_date'],
                    'distance_mi': distance_miles,
                    'pace_per_mi': minutes_per_mile,  # Store raw pace
                    'pace_formatted': f"{minutes}:{seconds:02d}",  # Store formatted pace
                    'lats': [coord[0] for coord in coords],
                    'longs': [coord[1] for coord in coords]
                })
        
        params['page'] += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(all_activities)
    
    # Format dates
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['date_formatted'] = df['start_date'].dt.strftime('%Y-%m-%d')
    
    return df.set_index('id')

def setup_auth_callback():
    """Setup a local server to receive the OAuth callback"""
    import socket
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import parse_qs, urlparse
    import threading
    
    auth_code = []
    
    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            query_components = parse_qs(urlparse(self.path).query)
            if 'code' in query_components:
                auth_code.append(query_components['code'][0])
                
            # Send success page
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Authorization successful! You can close this window and return to the app.")
            
        def log_message(self, format, *args):
            # Suppress logging
            return
    
    # Find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    
    # Start server in a thread
    server = HTTPServer(('', port), CallbackHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    
    return server, port, auth_code

def get_strava_auth_url(client_id):
    """Generate Strava OAuth authorization URL"""
    auth_url = "https://www.strava.com/oauth/authorize"
    redirect_uri = "http://localhost:8501"  # Default Streamlit port
    scope = "activity:read_all"
    
    url = f"{auth_url}?client_id={client_id}&response_type=code&redirect_uri={redirect_uri}&scope={scope}&approval_prompt=force"
    return url

def get_strava_token(client_id, client_secret, code):
    """Exchange authorization code for access token"""
    token_url = "https://www.strava.com/oauth/token"
    data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'code': code,
        'grant_type': 'authorization_code'
    }
    
    response = requests.post(token_url, data=data)
    if response.status_code != 200:
        raise Exception(f"Token exchange failed: {response.status_code}")
        
    return response.json()

def create_ui_api():
    """Create simple UI for run visualization using Strava API"""
    import streamlit as st
    import time
    import os
    from datetime import datetime
    from urllib.parse import urlparse, parse_qs
    
    st.title("Run Visualization Generator (Strava API)")
    
    # Initialize session state for auth
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    
    # Authentication section
    if not st.session_state.access_token:
        st.write("Please authenticate with Strava")
        
        # Generate auth URL and show button
        auth_url = get_strava_auth_url(client_id='144695')
        
        if st.button("Connect with Strava"):
            st.write("Click this link to authorize:")
            st.markdown(f"[Authorize with Strava]({auth_url})", unsafe_allow_html=True)
            
        # Get the current URL parameters
        current_url = st.query_params
        
        # Check if code is in the URL
        if 'code' in current_url:
            code = current_url['code']
            try:
                # Exchange code for token
                token_response = get_strava_token('144695', '4dc0aaa84c9a5efdc75924cea5ad5fbae8c32e81', code)
                st.session_state.access_token = token_response['access_token']
                st.success("Successfully authenticated!")
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {str(e)}")
        else:
            st.write("Waiting for authorization...")
    
    # Only show the rest if authenticated
    if st.session_state.access_token:
        # Create date input fields
        start_date = st.date_input(
            "Start Date",
            value=datetime.now(),
            format="YYYY-MM-DD"
        )
        
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            format="YYYY-MM-DD"
        )
        
        # Create generate button
        if st.button("Generate Visualization"):
            try:
                if start_date > end_date:
                    st.error("Start date must be before end date")
                    return
                
                # Fetch runs directly from Strava
                df = fetch_strava_runs(st.session_state.access_token, start_date, end_date)
                
                if df.empty:
                    st.warning("No runs found in the selected date range")
                    return
                
                # Generate visualization using date range function
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                generate_test_date_range(df, start_date_str, end_date_str)
                
                # Small delay to ensure file is saved
                time.sleep(1)
                
                # Check if file exists and display
                if os.path.exists('test_date_range.png'):
                    st.success(f"Generated visualization for runs between {start_date_str} and {end_date_str}")
                    st.image('test_date_range.png')
                else:
                    st.error(f"Could not find generated image: test_date_range.png")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    create_ui_api()