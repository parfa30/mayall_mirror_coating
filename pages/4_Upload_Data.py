import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
import matplotlib as mpl
from typing import Tuple, List
from datetime import datetime

# -----------------------------#
# ----- STYLE / CONSTANTS -----#
# -----------------------------#
mpl.rcParams["font.family"] = "sans-serif"
COLORS = mpl.colormaps['tab20'].colors
spreadsheet_key = '1NVfhMm5GVn2o8nRfpzRPlQ1rBH_3WFEV2wFevAXYiF8'
# -----------------------------#
# -------- DATA LOADING -------#
# -----------------------------#
@st.cache_data(show_spinner=False)
def load_mayall_data() -> pd.DataFrame:
    """Load Mayall telescope reflectivity data and index by Date."""
    url=f'https://docs.google.com/spreadsheet/ccc?key={spreadsheet_key}&output=csv'
    df = pd.read_csv(url,skiprows=[1,2])
    wavelengths = df.columns[9:]
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime('%Y-%m-%d')
    #df.drop(columns="Date", inplace=True)
    headers = pd.read_csv(url,nrows=2)
    return wavelengths, df, headers

WAVELENGTHS, mayall_data, headers = load_mayall_data()

# Standard calibration reference
HCL_STAND_SCI = np.array([
    86.79, 86.88, 86.94, 86.97, 87.0, 86.99, 86.97, 86.94, 86.89, 86.85,
    86.8, 86.73, 86.65, 86.57, 86.47, 86.36, 86.24, 86.12, 85.98, 85.83,
    85.68, 85.51, 85.33, 85.14, 84.95, 84.75, 84.54, 84.33, 84.1, 83.86, 83.61
])

STANDARD_SCI = pd.DataFrame(HCL_STAND_SCI).T
STANDARD_SCI.columns = WAVELENGTHS

# Dropdown option lists
wash_types = sorted(mayall_data["Wash Type"].dropna().unique())
wash_types.append("None")
coatings = sorted(mayall_data["Coating Date"].dropna().unique())

# -----------------------------#
# --------- HELPERS -----------#
# -----------------------------#
def get_dates_by_filter(column: str, value: str) -> List[str]:
    """
    Return available measurement dates filtered by a column value.

    Parameters
    ----------
    column : str
        Column to filter on (e.g., 'Wash Type', 'Coating Date').
    value : str
        Desired value or 'All' for no filtering.

    Returns
    -------
    List[str] : sorted date strings (newest first).
    """
    if value == "All":
        idx = mayall_data.index
    else:
        idx = mayall_data.loc[mayall_data[column] == value].index
    return sorted(idx.strftime("%Y-%m-%d"), reverse=True)

def between(val, pair):
    return pair[0] <= val <= pair[1]

def load_data(csv_file, date,telescope,mirror,zone,coating_date,wash_type,cals,meas,bw=True, aw=True,bw_cals=True, aw_cals=True):
    new_df = pd.read_csv(csv_file, skiprows=[1,2])
    new_df.drop(columns=new_df.columns[0:3],inplace=True)
    for col in new_df.columns[0:31]:
        new_df.rename(columns={col:col.strip('nm')},inplace=True)
    new_df['Date'] = date
    new_df['Date'] = pd.to_datetime(new_df["Date"]).dt.strftime('%Y-%m-%d')
    new_df['Telescope'] = telescope
    new_df['Mirror'] = mirror
    new_df['Zone'] = zone
    new_df['Coating Date'] = coating_date
    new_df['Wash Type'] = wash_type
    tags = []
    for i in range(len(new_df)//2):
        tags.append(['SCI','SCE'])
    new_df['Tag'] = np.hstack(tags)
    if wash_type == 'None':
        new_df['BW/AW'] = "NW"
        cals_idx = [0,cals*2]
        empty_idx = [cals*2+1, cals*2+4]
        meas_idx = [cals*2+4+1, cals*2+4+meas*2]
        types = []
        for idx in range(len(new_df)):
            if between(idx, cals_idx):
                types.append("calibration")
            elif between(idx, empty_idx):
                types.append("empty")
            elif between(idx, meas_idx):
                types.append("measurement")
            else:
                types.append("None")
        if len(types) == len(new_df):
            new_df['Type'] = np.hstack(types)
        else:
            print("Problem")
    else:
        labels = []
        types = []
        if bw:
            if bw_cals:
                bw_cals_idx = [0,cals*2-1]
                bw_empty_idx = [cals*2, cals*2+4-1]
                bw_meas_idx = [cals*2+4, cals*2+4+meas*2-1]
                bw_empty_2_idx = [cals*2+4+meas*2, cals*2+4+meas*2+4-1]
                final_bw = cals*2+4+meas*2+4-1
            else:
                bw_cals_idx = [np.nan, np.nan]
                bw_empty_idx = [np.nan, np.nan]
                bw_meas_idx = [0, meas*2-1]
                bw_empty_2_idx = [meas*2, meas*2+4-1]
                final_bw = meas*2+4
        if aw:
            if aw_cals:
                aw_cals_idx = [0+final_bw,cals*2+final_bw]
                aw_empty_idx = [cals*2+final_bw, cals*2+4+final_bw]
                aw_meas_idx = [cals*2+4+final_bw, cals*2+4+meas*2+final_bw]
                aw_empty_2_idx = [cals*2+4+meas*2+final_bw, cals*2+4+meas*2+4+final_bw]
            else:
                aw_cals_idx = [np.nan, np.nan]
                aw_empty_idx = [np.nan, np.nan]
                aw_meas_idx = [0+final_bw, meas*2+final_bw]
                aw_empty_2_idx = [meas*2+final_bw, meas*2+4+final_bw]
        for idx in range(len(new_df)):
            if between(idx, bw_cals_idx):
                labels.append("BW")
                types.append("calibration")
            elif between(idx, bw_empty_idx):
                labels.append("BW")
                types.append("empty")
            elif between(idx, bw_meas_idx):
                labels.append("BW")
                types.append("measurement")
            elif between(idx, bw_empty_2_idx):
                labels.append("BW")
                types.append("empty")
            elif between(idx, aw_cals_idx):
                labels.append("AW")
                types.append("calibration")
            elif between(idx, aw_empty_idx):
                labels.append("AW")
                types.append("empty")
            elif between(idx, aw_meas_idx):
                labels.append("AW")
                types.append("measurement")
            elif between(idx, aw_empty_2_idx):
                labels.append("AW")
                types.append("empty")
        if len(types) == len(new_df):
            new_df['Type'] = np.hstack(types)
            new_df['BW/AW'] = np.hstack(labels)
        else:
            print(f"len of df ({len(new_df)} not the same as labels ({len(types)}))")
            print(np.stack(types))
    new_df = new_df[new_df['Type'] != "empty"]
    if not aw_cals:
        bw_cal_df = new_df[(new_df['Type'] == 'calibration')&(new_df['BW/AW'] == 'BW')]
        bw_cal_df['BW/AW'] = 'AW'
        new_df = pd.concat([new_df, bw_cal_df], ignore_index=True)
    
    new_df = new_df[['Date','Telescope', 'Mirror', 'Zone', 'Coating Date', 'Wash Type', 'BW/AW',
       'Type', 'Tag', '400', '410', '420', '430', '440', '450', '460', '470',
       '480', '490', '500', '510', '520', '530', '540', '550', '560', '570',
       '580', '590', '600', '610', '620', '630', '640', '650', '660', '670',
       '680', '690', '700']]

    # new_df.index = pd.to_datetime(new_df["Date"]).dt.strftime('%Y-%m-%d')
    # new_df.drop(columns="Date", inplace=True)
    
    return new_df

def append_data(new_df):
    print(type(mayall_data.index))
    print(type(new_df.index))
    x = pd.concat([headers,mayall_data, new_df])
    x = x.reset_index(drop=True)
    print(x)
    # Authenticate (you’ll need a service account JSON key)
    creds = Credentials.from_service_account_file("service_account.json", scopes=[
        "https://www.googleapis.com/auth/spreadsheets"
    ])
    client = gspread.authorize(creds)
    print('Here 2')
    # Open the sheet by its key (the long part after /d/ in the URL)
    sheet = client.open_by_key(spreadsheet_key).sheet1
    print('Here 3')
    # Replace the data with your new DataFrame
    set_with_dataframe(sheet, x)
    print('Here 4')


# -----------------------------#
# -------- STREAMLIT UI -------#
# -----------------------------#
st.header("Upload Data")

st.markdown(
    """
    Add measurement data to the master spreadsheet.

    Currently, this is only possible for the Mayall

    If mirror wash type is None, data will be labelled as "NW". 

    Otherwise, you will have to identify if there is a Before and After wash measurement. It will assume the same number of measurements and calibrations for both.
    If there are only one set of calibrations, it will copy this for both the Before and After Wash.

    Expected Format:
    CSV with the following (this is for BW/AW with Cals for both)
    * Header
    * 2 rows of empty
    * 2 x 'Number of Cals' rows for Before Wash 
    * 4 rows of empty
    * 2 x 'Number of Measurements' rows for Before Wash
    * 4 rows of empty
    * 2 x 'Number of Cals' rows for After Wash
    * 4 rows of empty
    * 2 x 'Number of Measurements' rows for Before Wash
    * 4 rows of empty
    
    """
)


# User selections
new_file = st.file_uploader("Upload a csv")
today = datetime.now()

coating_date = coatings[0]

col1, col2, col3, col4,col5 = st.columns(5)
with col1:
    new_date = st.text_input("Data Taken Date:", today.strftime("%Y-%m-%d"))
with col2:
    telescope = st.text_input("Telescope:","Mayall")
with col3:
    mirror = st.text_input("Mirror:","M1")
with col4:
    zone = st.text_input("Zone:",1)
with col5:
    wash_type = st.selectbox("Wash Type:",wash_types,index=len(wash_types)-1)

col6, col7 = st.columns(2)
with col6:
    n_cals = st.number_input("Number of Cals:",min_value=0,max_value=100,value=4)
with col7:
    n_meas = st.number_input("Number of Measurements:",min_value=0,max_value=100,value=9)

if wash_type != "None":
    st.divider()
    st.markdown("Confirm that there are Before and After Wash measurements:")
    col8, col9 = st.columns(2)
    with col8:
        bw = st.checkbox("BW",value=1)
        bw_cals = st.checkbox("BW cals",value=1)
    with col9:
        aw = st.checkbox("AW",value=1)
        aw_cals = st.checkbox("AW cals",value=1)
    st.divider()

if "new_df" not in st.session_state:
    st.session_state.new_df = None

if st.button("Load Data"):
    st.session_state.new_df = load_data(
        new_file, new_date, telescope, mirror, zone,
        coating_date, wash_type, n_cals, n_meas,
        bw, bw_cals, aw, aw_cals
    )

if st.session_state.new_df is not None:
    st.dataframe(st.session_state.new_df)

    if st.button("Add Data to Master"):
        append_data(st.session_state.new_df)
        st.success("Data appended to master")

