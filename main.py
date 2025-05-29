import io
import tempfile
import uuid

from typing_extensions import Literal

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import subprocess
import json
import os
import zipfile
from datetime import datetime, timedelta
from io import StringIO, BytesIO
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, MaxNLocator
import numpy as np

# Get the full path to the directory containing the FastAPI script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the full path to the directory containing the NOA Workflow scripts
workflow_dir = script_dir.replace('/api', '')
app = FastAPI(
    openapi_tags=[
        {
            "name": "Run Workflow",
            "description": "Run the Ionospheric Peak Height Anomaly Monitor Workflow"
        },
        {
            "name": "Plot Data",
            "description": "Plot the data from the Ionospheric Peak Height Anomaly Monitor Workflow"
        }
    ],
    title="Ionospheric Peak Height Anomaly Monitor Workflow API",
    description="This workflow provides measured F2 peak height (FPeak) values obtained from GIRO ionosonde stations and compares them with modelled FPeak values in quiet conditions derived from a data collection developed by the Ebro Observatory. <br/><br/>The comparison is performed over a user-defined time interval, with plots showing the residuals (differences) between the observed and modelled FPeak values.<br/><br/>These residuals are displayed together with magnetic field components from the DSCOVR mission and the three-hourly geomagnetic Kp-index as possible drivers. <br/><br/>The workflow enables users to explore potential correlations between FPeak residuals and geomagnetic or interplanetary indices.",
    version="1.0.0",
    root_path="/wf-oe-f-peak"
)

# Configure CORS for all domains
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the stations from ./dataset/stations_workflow.csv
# Station,URSI,Lat,Lon
stations_df = pd.read_csv(f'{workflow_dir}/dataset/stations_workflow.csv', sep=',', header=0)
# Convert the stations to a list
stations_list = stations_df['URSI'].tolist()
# Convert the stations to a set
VALID_STATIONS = set(stations_list)
# Convert the stations to a sorted string
SORTED_VALID_STATIONS = ','.join(sorted(VALID_STATIONS))
# Create a Literal type for all valid stations
ValidStation = Literal.__getitem__(tuple(sorted(VALID_STATIONS)))
# Limit n_days to a fixed selection: 1 to 10
NDays = Literal["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# Validate the date range
def validate_dates(start_date_str, end_date_str):
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        if start_date > end_date:
            return False
    except ValueError:
        return False
    return True

# Validate the datetime range
def validate_datetimes(start_datetime_str, end_datetime_str):
    try:
        start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%dT%H:%M:%S')
        end_datetime = datetime.strptime(end_datetime_str, '%Y-%m-%dT%H:%M:%S')
        if start_datetime > end_datetime:
            return False
    except ValueError:
        return False
    return True

# Get Latitude and Longitude from the station name
def get_lat_lon(station):
    # Get the latitude and longitude from the station name
    try:
        lat = stations_df.loc[stations_df['URSI'] == station, 'Lat'].values[0]
        lon = stations_df.loc[stations_df['URSI'] == station, 'Lon'].values[0]
        return lat, lon
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Get t Ionosonde Characteristics data from https://lgdc.uml.edu/common/DIDBGetValues?ursiCode=EB040&charName=hmF2&DMUF=3000&fromDate=2025%2F04%2F29+09%3A00%3A00&toDate=2025%2F04%2F30+09%3A01%3A00
def get_iono_characteristics(ursi_code, start_datetime, end_datetime):
    # Get the ionosonde characteristics data from the URL
    url = f"https://lgdc.uml.edu/common/DIDBGetValues?ursiCode={ursi_code}&charName=hmF2&DMUF=3000&fromDate={start_datetime}&toDate={end_datetime}"
    try:
        response = requests.get(url)
        print(f"Ionosonde characteristics data URL: {url}")
        print(f"Response status code: {response.text}")
        # Read the plain text into a pandas dataframe
        df = pd.read_csv(
            io.StringIO(response.text),
                    comment="#",
                    sep='\s+',
                    names=["Time", "CS", "hmF2", "QD"])
        df = df.drop(columns=["QD"])
        # Drop the CS column
        df = df.drop(columns=["CS"])
        # Rename column "Time" to "Timestamp", and convert the value from "2025-01-01T00:00:00.000Z" to "2025-01-01 00:00:00"
        df.rename(columns={"Time": "Timestamp"}, inplace=True)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime('%Y-%m-%dT%H:%M:%S')
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

# Get the IMF B values
# https://electron.space.noa.gr/swnet_swif/api/v2/swifdb/solardb/magdata?start=2024-01-01T00%3A00%3A00&end=2024-01-03T00%3A00%3A00
def get_imf_b(start_datetime, end_datetime):
    # Get the IMF B values from the URL
    url = f"https://electron.space.noa.gr/swnet_swif/api/v2/swifdb/solardb/magdata?start={start_datetime}&end={end_datetime}"
    try:
        response = requests.get(url)
        response_json = response.json()
        # for each record, get the timestamp, bmag, bx, by, bz
        b_data = []
        for record in response_json:
            b_data.append({
                "Timestamp": record["timestamp"],
                "bmag": record["bmag"],
                "bx": record["bx"],
                "by": record["by"],
                "bz": record["bz"]
            })
        return b_data
    except Exception as e:
        print(f"Error: {e}")
        return None

# Get F Peak Data
# https://www.obsebre.es/hmf2_API/hmf2_time_series_interval_file/?year=2000&month=4&day_0=15&n_days=1&latitude=40.8&longitude=0.5
def get_f_peak_data(year, month, day, n_days, latitude, longitude):
    # Get the F Peak data from the URL
    url = f"https://www.obsebre.es/hmf2_API/hmf2_time_series_interval_file/?year={year}&month={month}&day_0={day}&n_days={n_days}&latitude={latitude}&longitude={longitude}"
    print(f"F Peak data URL: {url}")
    try:
        response = requests.get(url)
        # This is a CSV file, so we can read it directly into a pandas dataframe, line 1 is the header, year, month, day, hour, fpeak_height, read from line 2
        df = pd.read_csv(
            io.StringIO(response.text),
            comment="#",
            sep='\s+',
            names=["year", "month", "day", "hour", "fpeak_height"],
            skiprows=1
        )
        # Convert the year, month, day, hour to a Timestamp, format is YYYY-MM-DDTHH:MM:SS, as String
        df['Timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        # Drop the year, month, day, hour columns
        df = df.drop(columns=["year", "month", "day", "hour"])
        df = df[['Timestamp', 'fpeak_height']]
        return df

    except Exception as e:
        print(f"Error: {e}")
        return None

@app.get("/run_workflow/", response_class=StreamingResponse, responses={
    200: {"content": {"application/octet-stream": {}},
          "description": "**Important:** When selecting the 'zip' format, please remember to rename the downloaded file to have the extension '*.zip' before opening it.\n\n", }},
         summary="Run the workflow.",
         description="Return the results of the Ionospheric Peak Height Anomaly Monitor Workflow as a json object.",
         tags=["Run Workflow"])
async def run_workflow(start_datetime: str = Query(...,
                                                   description="Datetime in the format 'YYYY-MM-DD', e.g. 2024-01-01"),

                       n_days: NDays = Query(
                           ..., description="Select number of days (1 to 10). Default is 1 day."
                       ),
                       station: ValidStation = Query(...,
                                                                description="Comma-separated list of stations. Valid stations are: " + SORTED_VALID_STATIONS),
                       ):
    # convert n_days to int
    n_days = int(n_days)
    # Append the HH:MM:SS to the start_datetime
    start_datetime = start_datetime + 'T00:00:00'
    # Get the end datetime by adding n_days to the start datetime
    end_datetime = (datetime.strptime(start_datetime, '%Y-%m-%dT%H:%M:%S') + timedelta(days=n_days)).strftime('%Y-%m-%dT%H:%M:%S')
    print(f"Start datetime: {start_datetime}, End datetime: {end_datetime}")
    error_message = {"error": ""}
    # Validate the inputs
    if not validate_datetimes(start_datetime, end_datetime):
        error_message['error']="Invalid datetime range. Ensure the start datetime is before the end datetime and the format is YYYY-MM-DDTHH:MM:SS."
        return JSONResponse(
            status_code=200, content=error_message)

    # Workflow Step 1: Get KP data
    kp_script_path = f'{workflow_dir}/get_kp_data.sh'
    # Convert the start and end datetimes to dates
    start_date = start_datetime.split('T')[0]
    # Add one day to the end date to ensure the end date is included in the results
    end_date = (datetime.strptime(end_datetime.split('T')[0], '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    # Execute the shell script and capture the output
    try:
        process = subprocess.Popen(
            [kp_script_path, start_date, end_date, 'print-csv'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            error_message['error'] = stderr.decode() + " - DEBUG - get KP 1"  # Parse the JSON error message
            # Return the error message as json in the response, instead of raising an exception
            return JSONResponse(status_code=200, content=error_message)
    except subprocess.CalledProcessError as e:
        error_message['error'] = str(e) + " - DEBUG - get KP 2"
        return JSONResponse(status_code=200, content=error_message)

    # Create CSV files in memory for kp data
    try:
        kp_file = StringIO(stdout.decode())
        kp_file.seek(0)
        kp_df = pd.read_csv(kp_file, sep=',', header=0, index_col=0)
        # Convert to json, include the index
        kp_json = kp_df.to_json(orient='table', index=True)
        kp_json = json.loads(kp_json)["data"]
        # Convert to key-value pairs, timestamp as key, and the rest as value
        kp_json = {record['Timestamp']: {k: v for k, v in record.items() if k != 'Timestamp'} for record in kp_json}
    except Exception as e:
        error_message['error'] = f"Error processing output: {str(e)} - DEBUG - get KP 3"
        return JSONResponse(status_code=200, content=error_message)

    # Get Ionosonde characteristics data
    try:
        ic_df = get_iono_characteristics(station, start_datetime, end_datetime)
        ic_json = ic_df.to_json(orient='records')
        ic_json = json.loads(ic_json)
        # Convert to key-value pairs, timestamp as key, and the rest as value
        ic_json = {record['Timestamp']: {k: v for k, v in record.items() if k != 'Timestamp'} for record in ic_json}
    except Exception as e:
        error_message['error'] = f"Error processing output: {str(e)} - DEBUG - Fail to get Ionosonde characteristics"
        return JSONResponse(status_code=200, content=error_message)

    # Get the IMF B values
    try:
        b_df = get_imf_b(start_datetime, end_datetime)
        # Convert to key-value pairs, timestamp as key, and the rest as value
        b_json = {record['Timestamp']: {k: v for k, v in record.items() if k != 'Timestamp'} for record in b_df}
    except Exception as e:
        error_message['error'] = f"Error processing output: {str(e)} - DEBUG - Fail to get IMF B values"
        return JSONResponse(status_code=200, content=error_message)

    # Get the F Peak data
    try:
        latitude, longitude = get_lat_lon(station)
        # Get the year, month, day from the start_datetime
        year = start_datetime.split('-')[0]
        month = start_datetime.split('-')[1]
        day = start_datetime.split('-')[2].split('T')[0]
        # Get the F Peak data
        df_f_peak_data = get_f_peak_data(year, month, day, n_days, latitude, longitude)
        # Convert the F Peak data to JSON
        f_peak_data = json.loads(df_f_peak_data.to_json(orient='records'))
        # Convert to key-value pairs, timestamp as key, and the rest as value
        f_peak_json = {record['Timestamp']: {k: v for k, v in record.items() if k != 'Timestamp'} for record in f_peak_data}
    except Exception as e:
        error_message['error'] = f"Error processing output: {str(e)} - DEBUG - Fail to get F Peak data"
        return JSONResponse(status_code=200, content=error_message)

    # Find the difference between the f_peak_height and the hmF2 values where the timestamp is the same
    # Create a new dictionary to store the differences
    diff_json = {}
    for timestamp in f_peak_json.keys():
        if timestamp in ic_json:
            # Calculate the difference
            diff = ic_json[timestamp]['hmF2'] - f_peak_json[timestamp]['fpeak_height']
            # keep 2 decimal places
            diff = round(diff, 2)
            diff_json[timestamp] = {
                "fpeak_height": f_peak_json[timestamp]['fpeak_height'],
                "hmF2": ic_json[timestamp]['hmF2'],
                "difference": diff
            }
    output = {
        "ic_data": ic_json,
        "f_peak_data": f_peak_json,
        "diff_data": diff_json,
        "b_data": b_json,
        "kp_data": kp_json,
    }

    return JSONResponse(
        status_code=200,
        content={
            "message": "Workflow data retrieved successfully.",
            "input":{
                "start_datetime": start_datetime,
                "end_datetime": end_datetime,
                "n_days": n_days,
                "station": station
            },
            "output": output
        }
    )


@app.get("/plot_data/", response_class=StreamingResponse,
         summary="Plot the data from the workflow.",
         description="Plot the data from the Ionospheric Peak Height Anomaly Monitor Workflow and return the plot as a PNG image.",
         tags=["Plot Data"])
async def plot_data(start_datetime: str = Query(...,
                                                   description="Datetime in the format 'YYYY-MM-DD', e.g. 2024-01-01"),

                       n_days: NDays = Query(
                           ..., description="Select number of days (1 to 10). Default is 1 day."
                       ),
                       station: ValidStation = Query(...,
                                                                description="Comma-separated list of stations. Valid stations are: " + SORTED_VALID_STATIONS),
                       ):
    # convert n_days to int
    n_days = int(n_days)
    # Append the HH:MM:SS to the start_datetime
    start_datetime = start_datetime + 'T00:00:00'
    # Get the end datetime by adding n_days to the start datetime
    end_datetime = (datetime.strptime(start_datetime, '%Y-%m-%dT%H:%M:%S') + timedelta(days=n_days)).strftime('%Y-%m-%dT%H:%M:%S')
    print(f"Start datetime: {start_datetime}, End datetime: {end_datetime}")
    error_message = {"error": ""}
    # Validate the inputs
    if not validate_datetimes(start_datetime, end_datetime):
        error_message['error']="Invalid datetime range. Ensure the start datetime is before the end datetime and the format is YYYY-MM-DDTHH:MM:SS."
        return JSONResponse(
            status_code=200, content=error_message)

    # Workflow Step 1: Get KP data
    kp_script_path = f'{workflow_dir}/get_kp_data.sh'
    # Convert the start and end datetimes to dates
    start_date = start_datetime.split('T')[0]
    # Add one day to the end date to ensure the end date is included in the results
    end_date = (datetime.strptime(end_datetime.split('T')[0], '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    # Execute the shell script and capture the output
    try:
        process = subprocess.Popen(
            [kp_script_path, start_date, end_date, 'print-csv'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            error_message['error'] = stderr.decode()  # Parse the JSON error message
            # Return the error message as json in the response, instead of raising an exception
            return JSONResponse(status_code=200, content=error_message)
    except subprocess.CalledProcessError as e:
        error_message['error'] = str(e)
        return JSONResponse(status_code=200, content=error_message)

    # Create CSV files in memory for kp data
    try:
        kp_file = StringIO(stdout.decode())
        kp_file.seek(0)
        kp_df = pd.read_csv(kp_file, sep=',', header=0, index_col=0)
        # Convert to json, include the index
        kp_json = kp_df.to_json(orient='table', index=True)
        kp_json = json.loads(kp_json)["data"]
        # Convert to key-value pairs, timestamp as key, and the rest as value
        kp_json = {record['Timestamp']: {k: v for k, v in record.items() if k != 'Timestamp'} for record in kp_json}
    except Exception as e:
        error_message['error'] = f"Error processing output: {str(e)}"
        return JSONResponse(status_code=200, content=error_message)

    # Get Ionosonde characteristics data
    try:
        ic_df = get_iono_characteristics(station, start_datetime, end_datetime)
        ic_json = ic_df.to_json(orient='records')
        ic_json = json.loads(ic_json)
        # Convert to key-value pairs, timestamp as key, and the rest as value
        ic_json = {record['Timestamp']: {k: v for k, v in record.items() if k != 'Timestamp'} for record in ic_json}
    except Exception as e:
        error_message['error'] = f"Error processing output: {str(e)} - DEBUG - Fail to get Ionosonde characteristics"
        return JSONResponse(status_code=200, content=error_message)

    # Get the IMF B values
    try:
        b_df = get_imf_b(start_datetime, end_datetime)
        # Convert to key-value pairs, timestamp as key, and the rest as value
        b_json = {record['Timestamp']: {k: v for k, v in record.items() if k != 'Timestamp'} for record in b_df}
    except Exception as e:
        error_message['error'] = f"Error processing output: {str(e)} - DEBUG - Fail to get IMF B values"
        return JSONResponse(status_code=200, content=error_message)

    # Get the F Peak data
    try:
        latitude, longitude = get_lat_lon(station)
        # Get the year, month, day from the start_datetime
        year = start_datetime.split('-')[0]
        month = start_datetime.split('-')[1]
        day = start_datetime.split('-')[2].split('T')[0]
        # Get the F Peak data
        df_f_peak_data = get_f_peak_data(year, month, day, n_days, latitude, longitude)
        # Convert the F Peak data to JSON
        f_peak_data = json.loads(df_f_peak_data.to_json(orient='records'))
        # Convert to key-value pairs, timestamp as key, and the rest as value
        f_peak_json = {record['Timestamp']: {k: v for k, v in record.items() if k != 'Timestamp'} for record in f_peak_data}
    except Exception as e:
        error_message['error'] = f"Error processing output: {str(e)} - DEBUG - Fail to get F Peak data"
        return JSONResponse(status_code=200, content=error_message)

    # Find the difference between the f_peak_height and the hmF2 values where the timestamp is the same
    # Create a new dictionary to store the differences
    diff_json = {}
    for timestamp in f_peak_json.keys():
        if timestamp in ic_json:
            # Calculate the difference
            diff = ic_json[timestamp]['hmF2'] - f_peak_json[timestamp]['fpeak_height']
            # keep 2 decimal places
            diff = round(diff, 2)
            diff_json[timestamp] = {
                "fpeak_height": f_peak_json[timestamp]['fpeak_height'],
                "hmF2": ic_json[timestamp]['hmF2'],
                "difference": diff
            }
    output = {
        "ic_data": ic_json,
        "f_peak_data": f_peak_json,
        "diff_data": diff_json,
        "b_data": b_json,
        "kp_data": kp_json,
    }

    # Plot the data
    nrows = 4
    figsize = (16, 9)
    fig, (ax_ic_vs_fpeak, ax_ic_diff_fpeak, ax_b, ax_kp) = plt.subplots(nrows, 1, figsize=figsize, dpi=100)
    # Add the title to the fig
    fig.suptitle(f"{n_days} Days Results for station {station} from {start_datetime}", fontsize=16)
    # Add a divider line between the subplots and the title
    fig.subplots_adjust(hspace=0.5)

    start_datetime_offset = datetime.strptime(start_datetime, '%Y-%m-%dT%H:%M:%S') - timedelta(hours=2)
    end_datetime_offset = datetime.strptime(end_datetime, '%Y-%m-%dT%H:%M:%S') + timedelta(hours=2)
    ax_ic_vs_fpeak.set_xlim(start_datetime_offset, end_datetime_offset)
    ax_ic_diff_fpeak.set_xlim(start_datetime_offset, end_datetime_offset)
    ax_kp.set_xlim(start_datetime_offset, end_datetime_offset)
    ax_b.set_xlim(start_datetime_offset, end_datetime_offset)
    # Get the total hours between start and end datetime, and make it to be 8 major ticks, and get the dynamic frequency
    total_hours = (datetime.strptime(end_datetime, '%Y-%m-%dT%H:%M:%S') - datetime.strptime(start_datetime, '%Y-%m-%dT%H:%M:%S')).total_seconds() / 3600
    freq = int(total_hours / 8)
    # Set the major ticks to be every frequency hours
    major_ticks = pd.date_range(start=start_datetime, end=end_datetime, freq=f'{freq}h')
    for ax in (ax_ic_vs_fpeak, ax_ic_diff_fpeak, ax_b, ax_kp):
        ax.set_xticks(major_ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%dT%H'))
        # Set the minor ticks to be every 1 hour
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        # Plot the x-axis with grey color, linewidth 0.5, from top to bottom
        ax.xaxis.grid(True, which='major', color='grey', linestyle='-', linewidth=0.5)
        # Plot the x-axis for minor ticks without any text
        ax.xaxis.grid(True, which='minor', color='lightgrey', linestyle='-', linewidth=0.5)
        # Make all the lines in the plot to be under the grid lines, not cover the data
        ax.set_axisbelow(True)

    # Plot the fpeak_height data using line chart, solid line, red color, linewidth 2,
    # and all the hmF2 data using scatter plot, blue color, marker 'o', size 10
    # and the hmF2 date which has same timestamp as fpeak_height data using scatter plot, blue color, marker 'o', size 40
    vs_fpeak_timestamps = [datetime.fromisoformat(timestamp) for timestamp in f_peak_json.keys()]
    vs_fpeak_y_axis = [f_peak_json[t]["fpeak_height"] for t in f_peak_json]
    vs_hmf2_timestamps = [datetime.fromisoformat(timestamp) for timestamp in ic_json.keys()]
    vs_hmf2_y_axis = [ic_json[t]["hmF2"] for t in ic_json]
    same_timestamp = [datetime.fromisoformat(timestamp) for timestamp in diff_json.keys()]
    same_hmf2_y_axis = [diff_json[t]["hmF2"] for t in diff_json]
    # Plot the fpeak_height data using line chart
    ax_ic_vs_fpeak.plot(vs_fpeak_timestamps, vs_fpeak_y_axis, label='Model fpeak', linestyle='-', linewidth=2, color='red')
    # Plot the hmF2 data using scatter plot
    ax_ic_vs_fpeak.scatter(vs_hmf2_timestamps, vs_hmf2_y_axis, label='All GIRO hmF2 measurements', color='blue', marker='o', s=1)
    # Plot the hmF2 data which has same timestamp as fpeak_height data using scatter plot
    ax_ic_vs_fpeak.scatter(same_timestamp, same_hmf2_y_axis, label='Selected GIRO hmF2 (closest to hour)', color='green', marker='o', s=40)
    # Set the y-axis range from min to max of all vs_fpeak_y_axis and vs_hmf2_y_axis offset by 20% of the axis height, auto scale
    ax_ic_vs_fpeak.set_ylim(100, 700)
    # Set the major ticks, maximum 5 ticks
    ax_ic_vs_fpeak.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    # Draw the horizontal line at each major tick position
    ax_ic_vs_fpeak.yaxis.grid(True, linestyle='-', linewidth=0.5)
    ax_ic_vs_fpeak.set_ylabel('Height [km]')
    # Fix the legend position to the top right corner
    ax_ic_vs_fpeak.legend(ncol=4, loc='upper right')
    ax_ic_vs_fpeak.set_title(f'GIRO hmF2 measurements vs Model fpeak')

    # Plot the difference between the fpeak_height and hmF2 values
    fpeak_x_axis = [datetime.fromisoformat(timestamp) for timestamp in diff_json.keys()]
    fpeak_y_axis = [diff_json[t]["difference"] for t in diff_json]
    ax_ic_diff_fpeak.plot(fpeak_x_axis, fpeak_y_axis, color='green', label='Difference (hmF2 - fpeak)')
    ax_ic_diff_fpeak.fill_between(fpeak_x_axis, fpeak_y_axis, 0, color='green', alpha=0.2)
    # Get the max of absolute value of fpeak_y_axis
    max_fpeak_y_axis = max(fpeak_y_axis, key=abs)
    # Convert the max_fpeak_y_axis to int
    max_fpeak_y_axis = int(max_fpeak_y_axis)
    # Set the y-axis range from min to max of fpeak_y_axis offset by 20% of the axis height, auto scale
    ax_ic_diff_fpeak.set_ylim(-1.2*max_fpeak_y_axis, 1.2*max_fpeak_y_axis)
    ax_ic_diff_fpeak.invert_yaxis()  # Invert the y-axis
    # Set the major ticks, maximum 5 ticks
    ax_ic_diff_fpeak.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    # Draw the horizontal line at each major tick position
    ax_ic_diff_fpeak.yaxis.grid(True, linestyle='-', linewidth=0.5)
    ax_ic_diff_fpeak.set_ylabel('Difference [km]')
    # Fix the legend position to the top right corner
    ax_ic_diff_fpeak.legend(ncol=4, loc='upper right')
    ax_ic_diff_fpeak.set_title(f'Differences between GIRO hmF2 and Model fpeak')

    # Plot the BMAG data using line chart
    b_x_axis = pd.date_range(start_datetime, end_datetime, freq='1h')
    bmag_y_axis = []
    bx_y_axis = []
    by_y_axis = []
    bz_y_axis = []

    for timestamp in b_json.keys():
        bmag_y_axis.append(b_json[timestamp]['bmag'])
        bx_y_axis.append(b_json[timestamp]['bx'])
        by_y_axis.append(b_json[timestamp]['by'])
        bz_y_axis.append(b_json[timestamp]['bz'])

    # print(f"b_x_axis: {b_x_axis}, bmag_y_axis: {bmag_y_axis}, bx_y_axis: {bx_y_axis}, by_y_axis: {by_y_axis}, bz_y_axis: {bz_y_axis}")
    # bmag line style is solid bold, bx line style is dash, by line style is dot, bz line style is solid thin, all in black color
    ax_b.plot(b_x_axis, bmag_y_axis, label='Bmag', linestyle='-', linewidth=4, color='black')
    ax_b.plot(b_x_axis, bx_y_axis, label='Bx', linestyle='--', linewidth=3, color='black')
    ax_b.plot(b_x_axis, by_y_axis, label='By', linestyle=':', linewidth=2, color='black')
    ax_b.plot(b_x_axis, bz_y_axis, label='Bz', linestyle='-', linewidth=1, color='black')
    # Set the y-axis range from min to max of all bmag_y_axis, bx_y_axis, by_y_axis, bz_y_axis offset by 20% of the axis height, auto scale
    ax_b.set_ylim(min(bmag_y_axis+bx_y_axis+by_y_axis+bz_y_axis)-1, max(bmag_y_axis+bx_y_axis+by_y_axis+bz_y_axis)+int(0.4*(max(bmag_y_axis+bx_y_axis+by_y_axis+bz_y_axis)-min(bmag_y_axis+bx_y_axis+by_y_axis+bz_y_axis))))
    # Set the major ticks, maximum 5 ticks
    ax_b.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    # Draw the horizontal line at each major tick position
    ax_b.yaxis.grid(True, linestyle='-', linewidth=0.5)
    ax_b.set_ylabel('Bmag, Bx, By, Bz [nT]')
    # Fix the legend position to the top right corner
    ax_b.legend(ncol=4, loc='upper right')
    ax_b.set_title(f'DSCOVR mission Magdata records')

    # Plot the KP data using bar chart, skip the timestamp with 0 value
    kp_x_axis = []
    kp_y_axis = []
    for timestamp in kp_json.keys():
        if kp_json[timestamp]['Kp'] != 0:
            kp_x_axis.append(timestamp)
            kp_y_axis.append(kp_json[timestamp]['Kp'])
    # Plot the KP data using bar chart, skip the timestamp with 0 value
    ax_kp.bar(kp_x_axis, kp_y_axis, width=0.04, color='#156082', label='Kp')
    # Set the kp y-axis range from min to max of kp_y_axis offset by 0.5
    ax_kp.set_ylim(0, max(kp_y_axis) + 0.5)
    # Set the major ticks, maximum 5 ticks
    ax_kp.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    # Draw the horizontal line at each major tick position
    ax_kp.yaxis.grid(True, linestyle='-', linewidth=0.5)
    ax_kp.set_ylabel('Kp-index')
    ax_kp.set_title(f'Geomagnetic three-hourly Kp-index')

    plt.tight_layout()
    plot_filename = f"{station}_{start_datetime.split('T')[0]}_{end_datetime.split('T')[0]}.png"
    img_io = io.BytesIO()
    fig.savefig(img_io, format='png', bbox_inches='tight')
    img_io.seek(0)
    plt.close()

    return StreamingResponse(img_io,media_type="image/png")