import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
from typing import Tuple, List

# -------------------------#
# -----  CONFIG / STYLE ---#
# -------------------------#
mpl.rcParams["font.family"] = "sans-serif"
COLORS = mpl.colormaps['tab20'].colors

# -------------------------#
# -----  DATA LOADING -----#
# -------------------------#
@st.cache_data(show_spinner=False)
def load_mayall_data() -> pd.DataFrame:
    """Load Mayall telescope reflectivity data and index by Date."""
    url = (
        "https://docs.google.com/spreadsheets/d/e/"
        "2PACX-1vSWF3pE1qBNcatq1Cm-H2z6mAGBPA-EbyUhujoOHXUU9-"
        "BfGYkmzP0YaFslu0bm2efn6AB9fNwUEUHj/pub?gid=0&single=true&output=csv"
    )
    df = pd.read_csv(url)
    wavelengths = df.columns[9:]
    df.index = pd.to_datetime(df["Date"])
    df.drop(columns="Date", inplace=True)
    return wavelengths, df

WAVELENGTHS, mayall_data = load_mayall_data()

# Standard reference (single-row DataFrame)
HCL_STAND_SCI = np.array(
    [
        86.79, 86.88, 86.94, 86.97, 87.0, 86.99, 86.97, 86.94, 86.89, 86.85,
        86.8, 86.73, 86.65, 86.57, 86.47, 86.36, 86.24, 86.12, 85.98, 85.83,
        85.68, 85.51, 85.33, 85.14, 84.95, 84.75, 84.54, 84.33, 84.1, 83.86,
        83.61,
    ]
)
STANDARD_SCI = pd.DataFrame(HCL_STAND_SCI).T
STANDARD_SCI.columns = WAVELENGTHS

# UI option lists
wash_types = sorted(mayall_data["Wash Type"].dropna().unique())
wash_types.append("All")
coating_dates = sorted(mayall_data["Coating Date"].dropna().unique())
coating_dates.append("All")


# -------------------------#
# -----  HELPERS ----------#
# -------------------------#
def get_dates_by_filter(
    column: str, value: str
) -> List[str]:
    """
    Return available dates (YYYY-MM-DD) filtered by a column (Wash Type or Coating Date).

    Parameters
    ----------
    column : str
        Column name to filter on (e.g., 'Wash Type' or 'Coating Date').
    value : str
        Value to match or 'All' for no filtering.

    Returns
    -------
    list of str
        Sorted list of date strings, newest first.
    """
    if value == "All":
        dates = mayall_data.index
    else:
        dates = mayall_data.loc[mayall_data[column] == value].index
    return sorted(dates.strftime("%Y-%m-%d"), reverse=True)


def reduce_data(
    df: pd.DataFrame, sc: str = "SCI", wash: str = "AW"
) -> Tuple[pd.Series, pd.Series]:
    """
    Aggregate measurement data and scale by calibration.

    Parameters
    ----------
    df : DataFrame
        Subset of mayall_data for a single date.
    sc : {'SCI', 'SCE'}
        Signal type: reflectance or scatter.
    wash : {'AW', 'BW'}
        After/Before wash.

    Returns
    -------
    reduced_data : Series
        Scaled mean values by wavelength.
    meas_sem : Series
        Standard error of the mean for those wavelengths.
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

# -------------------------#
# -----  PLOTTING ---------#
# -------------------------#
@st.cache_data(show_spinner=False)
def daily_plot(date_str: str) -> pd.DataFrame:
    """
    Generate daily plots of reflectance and scatter for a given date.

    Parameters
    ----------
    date_str : str
        Date in 'YYYY-MM-DD' format.

    Returns
    -------
    DataFrame
        Subset of mayall_data used in the plot.
    """
    date = pd.to_datetime(date_str)
    subset = mayall_data.loc[mayall_data.index.normalize() == date]

    if subset.empty:
        st.warning(f"No data available for {date_str}")
        return pd.DataFrame()

    wash_type = subset["Wash Type"].iloc[0]
    sc_map = {"Reflectance": "SCI", "Scatter": "SCE"}
    color_map = {"BW": "blue", "AW": "orange"}

    for label, sc in sc_map.items():
        fig = plt.figure(figsize=(10, 4))
        gs = GridSpec(2, 1, height_ratios=[1, 3], hspace=0.0)
        ax_ratio = fig.add_subplot(gs[0])
        ax_values = fig.add_subplot(gs[1], sharex=ax_ratio)

        has_bw = has_aw = False

        try:
            bw_mean, bw_err = reduce_data(subset, sc, "BW")
            bw_mean.plot(ax=ax_values, yerr=bw_err, alpha=0.6,
                         label="Before Wash", color=color_map["BW"])
            has_bw = True
        except Exception as exc:
            st.info(f"No BW {label} data: {exc}")

        try:
            aw_mean, aw_err = reduce_data(subset, sc, "AW")
            aw_mean.plot(ax=ax_values, yerr=aw_err, alpha=0.6,
                         label="After Wash", color=color_map["AW"])
            has_aw = True
        except Exception as exc:
            st.info(f"No AW {label} data: {exc}")

        if has_bw and has_aw:
            (aw_mean / bw_mean).plot(ax=ax_ratio, label="AW / BW", color="k", alpha=0.5)

        ax_ratio.set_title(f"{label} Curves for {date_str} ({wash_type} wash)")
        ax_ratio.set_ylabel("Wash % Increase")
        ax_ratio.legend()
        ax_ratio.tick_params(axis="x", labelbottom=False)

        ax_values.set_xlabel("Wavelength (nm)")
        ax_values.set_ylabel("Mean Value")
        ax_values.legend()

        plt.tight_layout()
        st.pyplot(fig)

    return subset


@st.cache_data(show_spinner=False)
def convert_for_download(df: pd.DataFrame) -> bytes:
    """Return CSV bytes for Streamlit download_button."""
    return df.to_csv().encode("utf-8")


# -------------------------#
# -----  STREAMLIT UI -----#
# -------------------------#
st.header("Mayall Daily Reflectivity & Scatter Plot")

st.markdown(
    """
    ## Daily Plot

    View the **reflectivity and scattering measurements** for a single observation date.

    * Choose a **wash type** to filter the dataset to specific days.
    * Select the specific **date of measurement** you want to inspect.
    * The plot displays the measured reflectivity/scattering curves across all recorded wavelengths.

    Use this page when you want a detailed snapshot of the mirror’s performance on a particular day,
    for example to check the effect of a recent wash or to compare different coating cycles.
    """
)


col1, col2 = st.columns(2)
with col1:
    wash_choice = st.selectbox("Wash Type", wash_types, index=len(wash_types) - 1)
with col2:
    date_choice = st.selectbox(
        "Measurement Date",
        get_dates_by_filter("Wash Type", wash_choice),
    )

if st.button("Generate Daily Plot"):
    data_for_day = daily_plot(date_choice)
    if not data_for_day.empty:
        st.download_button(
            label="Download CSV",
            data=convert_for_download(data_for_day),
            file_name=f"mayall_{date_choice}.csv",
            mime="text/csv",
            icon=":material/download:",
        )
