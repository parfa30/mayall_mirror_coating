import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
from typing import List, Dict, Union, Tuple

# -----------------------------#
# ----- STYLE / CONSTANTS -----#
# -----------------------------#
mpl.rcParams["font.family"] = "sans-serif"
COLORS = mpl.colormaps['tab20'].colors

# -----------------------------#
# -------- DATA LOADING -------#
# -----------------------------#
@st.cache_data(show_spinner=False)
def load_mayall_data() -> pd.DataFrame:
    """Load Mayall telescope reflectivity data and index by Date."""
    spreadsheet_key = '1NVfhMm5GVn2o8nRfpzRPlQ1rBH_3WFEV2wFevAXYiF8'

    url=f'https://docs.google.com/spreadsheet/ccc?key={spreadsheet_key}&output=csv'
    df = pd.read_csv(url,skiprows=[1,2])
    wavelengths = df.columns[9:]
    df.index = pd.to_datetime(df["Date"])
    df.drop(columns="Date", inplace=True)
    return wavelengths, df

WAVELENGTHS, mayall_data = load_mayall_data()

# Standard calibration reference
HCL_STAND_SCI = np.array([
    86.79, 86.88, 86.94, 86.97, 87.0, 86.99, 86.97, 86.94, 86.89, 86.85,
    86.8, 86.73, 86.65, 86.57, 86.47, 86.36, 86.24, 86.12, 85.98, 85.83,
    85.68, 85.51, 85.33, 85.14, 84.95, 84.75, 84.54, 84.33, 84.1, 83.86, 83.61
])

STANDARD_SCI = pd.DataFrame(HCL_STAND_SCI).T
STANDARD_SCI.columns = WAVELENGTHS


# -----------------------------#
# --------- HELPERS -----------#
# -----------------------------#
def get_dates_by_filters(filters: Dict[str, Union[str, List[str]]]) -> List[str]:
    """
    Return available measurement dates filtered by one or more columns and values.

    Parameters
    ----------
    filters : dict
        Dictionary mapping column names to allowed values.
        Each value can be:
          - a single string (e.g. "BW")
          - a list of strings (e.g. ["SCI", "SCE"])
          - the string "All" to disable filtering on that column

    Returns
    -------
    List[str]
        Sorted list of date strings (newest first).
    """
    df = mayall_data.copy()

    # Apply each filter
    for column, values in filters.items():
        if values == "All":
            continue
        if isinstance(values, str):
            values = [values]
        df = df[df[column].isin(values)]

    # Collect and format unique dates from the index
    dates = [pd.to_datetime(date).strftime("%Y-%m-%d") for date in np.unique(df.index)]
    return sorted(dates, reverse=True)


def reduce_data(df: pd.DataFrame, sc: str = "SCI", wash: str = "AW") -> Tuple[pd.Series, pd.Series]:
    """
    Aggregate a single-day measurement and scale by calibration.

    Returns
    -------
    reduced : Series
        Scaled mean values across wavelengths.
    sem : Series
        Standard error of the mean.
    """
    grouped = (
        df.groupby(["Type", "BW/AW", "Tag"])[WAVELENGTHS]
        .agg(["mean", "sem", "std", "min", "max"])
    )
    cal_mean = grouped.loc[("calibration", wash, "SCI")].xs("mean", level=1)
    meas = grouped.loc[("measurement", wash, sc)]
    meas_mean = meas.xs("mean", level=1)
    meas_sem = meas.xs("sem", level=1)
    scale = cal_mean / STANDARD_SCI.loc[0]
    return meas_mean / scale, meas_sem

def mean_segment(data_list, start: int, end: int = None) -> list:
    """
    Compute the mean of slices of pandas Series/DataFrames in a list.

    Parameters
    ----------
    data_list : list of pd.Series or pd.DataFrame
        List of data series to compute means for.
    start : int
        Start index for slicing.
    end : int or None
        End index for slicing. If None, slice to the end of the series.

    Returns
    -------
    List[float]
        List of mean values for each series in `data_list`.
    """
    if end is None:
        return [np.mean(d.iloc[start:]) for d in data_list]
    else:
        return [np.mean(d.iloc[start:end]) for d in data_list]


def before_after_plot(date_list: list, sc: str):
    """
    Plot before/after wash mean measurements for Blue, Green, and Red segments.

    Parameters
    ----------
    date_list : list of str or pd.Timestamp
        Dates to process and plot.
    sc : SCI (Reflectivity) or SCE (Scatter)

    Notes
    -----
    Assumes the existence of:
    - `mayall_data` : DataFrame indexed by date.
    - `reduce_data` : function to filter data by 'sc' and 'wash' parameters.
    """
    # Initialize arrays for B, G, R
    B, G, R = [], [], []

    for date in date_list:
        
        # Select data for the current date
        d_ = mayall_data.loc[mayall_data.index == date]

        try:
            # Reduce by SC and wash type
            d_bw = reduce_data(d_, sc=sc, wash='BW')
                        # Compute mean segments: 0-10, 10-20, 20-end
            B_bw, G_bw, R_bw = (
                mean_segment(d_bw, 0, 10),
                mean_segment(d_bw, 10, 20),
                mean_segment(d_bw, 20, None)
            )
        except:
            B_bw, G_bw, R_bw = ([np.nan], [np.nan], [np.nan])
        try:
            d_aw = reduce_data(d_, sc=sc, wash='AW')
            B_aw, G_aw, R_aw = (
                mean_segment(d_aw, 0, 10),
                mean_segment(d_aw, 10, 20),
                mean_segment(d_aw, 20, None)
            )
        except:
            B_baw, G_baw, R_aw = ([np.nan], [np.nan], [np.nan])
        # Collect Blue, Green, Red means (before/after)
        B.append([B_bw[0], B_aw[0]])
        G.append([G_bw[0], G_aw[0]])
        R.append([R_bw[0], R_aw[0]])


    # Convert to numpy arrays for plotting
    B = np.array(B)
    G = np.array(G)
    R = np.array(R)

    # Convert dates to pandas datetime
    dates = pd.to_datetime(date_list)

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6,10))

    # Plot Blue
    ax1.plot(dates, B[:, 0], 'x', label='Before Wash')
    ax1.plot(dates, B[:, 1], 'o', label='After Wash')
    ax1.set_title("Blue Mean Measurements (400-500nm)")
    ax1.legend()

    # Plot Green
    ax2.plot(dates, G[:, 0], 'x', label='Before Wash')
    ax2.plot(dates, G[:, 1], 'o', label='After Wash')
    ax2.set_title("Green Mean Measurements (500-600nm)")

    # Plot Red
    ax3.plot(dates, R[:, 0], 'x', label='Before Wash')
    ax3.plot(dates, R[:, 1], 'o', label='After Wash')
    ax3.set_title("Red Mean Measurements (600-700nm)")

    # Rotate x-axis tick labels 45 degrees for readability
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    st.pyplot(fig)


# -----------------------------#
# -------- STREAMLIT UI -------#
# -----------------------------#
st.header("Before/After Wash Comparison")

st.markdown(
    """
    These plots show the mean before and after measurements for a selection of dates.

    Note: To select several dates, it will require you to try several times. 
    However, you can change the Coating Date and Wash type to explore different comparisons.
    """
)
# Dropdown option lists
wash_types = sorted(mayall_data["Wash Type"].dropna().unique())
wash_types.append("All")
coatings = sorted(
    mayall_data["Coating Date"].dropna().unique(),
    key=lambda x: pd.to_datetime(x, format="%m/%d/%Y"), reverse=True)
coatings.append("All")


# Step 1: Dropdown filters
coatings_select = st.selectbox("Coating Date", coatings)
wash_select = st.selectbox("Wash Type", wash_types)

# Step 2: Filter data based on current selection
filters = {"Coating Date":coatings_select, "Wash Type":wash_select}
date_options = get_dates_by_filters(filters)
options_with_all = ["Select All"] + date_options
# Step 3: Use session state to remember selected dates
if "selected_dates" not in st.session_state:
    st.session_state.selected_dates = []

# Keep only valid selections from previous runs
valid_previous = [d for d in st.session_state.selected_dates if d in date_options]

selected_dates = st.multiselect(
    "Select Dates",
    options_with_all,
    default=valid_previous
)

sc_choice = st.selectbox(
        "SC",
        ['SCI','SCE'],
        index=0,
    )

# Update session state with new selection
st.session_state.selected_dates = selected_dates

if st.button("Generate Before/After Plot"):
    if "Select All" in selected_dates:
        selected = date_options
    else:
        selected = selected_dates
    before_after_plot(
        selected, sc_choice
    )