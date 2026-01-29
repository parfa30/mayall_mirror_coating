import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
from typing import Tuple, List
from astropy.time import Time

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

# Dropdown option lists
wash_types = sorted(mayall_data["Wash Type"].dropna().unique())
wash_types.append("All")
coatings = sorted(
    mayall_data["Coating Date"].dropna().unique(),
    key=lambda x: pd.to_datetime(x, format="%m/%d/%Y"), reverse=True)
coatings.append("All")

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
        dates_ = mayall_data.index
    else:
        dates_ = mayall_data.loc[mayall_data[column] == value].index
    dates = [pd.to_datetime(date).strftime("%Y-%m-%d") for date in np.unique(dates_)]
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


def lin_reg(x_dates: pd.Series,
            y_with_err: Tuple[pd.Series, pd.Series]) -> Tuple[pd.DatetimeIndex, np.ndarray, float]:
    """
    Weighted linear regression of y vs. date.

    Returns date range, predicted y values, and weighted R².
    """
    y, y_err = y_with_err
    weights = 1 / (np.array(y_err, float) ** 2)
    x_ord = pd.Series(x_dates).map(pd.Timestamp.toordinal).values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_ord, y, sample_weight=weights)

    date_range = pd.date_range(start=min(x_dates), end=max(x_dates), periods=len(y))
    x_range = pd.Series(date_range).map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y_pred = model.predict(x_range)

    ss_res = np.sum(weights * (y - model.predict(x_ord)) ** 2)
    ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return date_range, y_pred, r2


def change_plot(
    start_date: str,
    end_date: str,
    linreg: bool = False,
    sc: str = "SCI",
    outliers_input: str = "",
    coating_all: bool = False
) -> None:
    """
    Plot mean reflectivity in three wavelength bands (Blue/Green/Red) across time.

    Parameters
    ----------
    start_date : str
        Start of date range (YYYY-MM-DD).
    end_date : str
        End of date range (YYYY-MM-DD).
    linreg : bool
        If True, add weighted linear regression fits.
    sc : str
        Whether Reflectivity Measurmeent (SCI) or Scatter Measurement (SCE)
    outliers_input : str
        Comma-separated dates to exclude.
    """
    this_df = mayall_data.loc[start_date:end_date]

    # Filter out user-provided outliers
    outliers = [o.strip() for o in outliers_input.split(",") if o.strip()]
    dates = pd.Series(this_df.index.normalize().unique())
    if outliers:
        dates = dates[~dates.astype(str).isin(outliers)]

    results = []
    for d in dates:
        daily = this_df.loc[this_df.index.normalize() == d]
        try:
            reduced, err = reduce_data(daily, sc=sc, wash="AW")
        except Exception:
            # Fall back to BW if AW not available
            reduced, err = reduce_data(daily, sc=sc, wash="BW")
        results.append((d, (reduced, err)))

    x = pd.to_datetime([r[0] for r in results])
    y_means = [r[1][0] for r in results]
    y_errs = [r[1][1] for r in results]

    def mean_segment(data_list, start: int, end: int | None) -> List[float]:
        """Average a slice of each daily spectrum (e.g., first 10 nm for 'Blue')."""
        if end is None:
            return [np.mean(d.iloc[start:]) for d in data_list]
        return [np.mean(d.iloc[start:end]) for d in data_list]

    # Split into three wavelength bands
    B, G, R = (
        mean_segment(y_means, 0, 10),
        mean_segment(y_means, 10, 20),
        mean_segment(y_means, 20, None),
    )
    B_err, G_err, R_err = (
        mean_segment(y_errs, 0, 10),
        mean_segment(y_errs, 10, 20),
        mean_segment(y_errs, 20, None),
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    color_map = {"B": "blue", "G": "green", "R": "red"}
    data_series = {"B": (B, B_err), "G": (G, G_err), "R": (R, R_err)}

    for band, (means, errs) in data_series.items():
        ax.errorbar(
            x, means, yerr=errs, fmt="o",
            mfc=color_map[band], mec="black", ecolor="black"
        )
        if linreg:
            date_range, y_fit, r2 = lin_reg(x, (means, errs))
            ax.plot(date_range, y_fit, color=color_map[band],
                    label=f"{band} fit (R²={r2:.2f})")

    if linreg:
        ax.legend()
    if coating_all:
        coating_dates = [pd.to_datetime(coating_date) for coating_date in coatings[:-1]]
        coating_dates = np.sort(coating_dates)
        for cd in coating_dates:
            ax.axvline(cd, color='k',ls = '--')
        ax.set_xlim(x[0],x[-1])
    ax.set_title(f"{sc} Change Over Time")
    ax.set_ylabel(f"Mean {sc} (%)")
    ax.set_xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig)

    Data = pd.DataFrame(np.array([x, B, B_err, G, G_err, R, R_err]).T, columns = ['Date','B_mean','B_err','G_mean','G_err','R_mean','R_err'])
    return Data

@st.cache_data(show_spinner=False)
def convert_for_download(df: pd.DataFrame) -> bytes:
    """Return CSV bytes for Streamlit download_button."""
    return df.to_csv().encode("utf-8")


# -----------------------------#
# -------- STREAMLIT UI -------#
# -----------------------------#
st.header("Change Over Time Plot")

st.markdown(
    """
    Explore **how the mirror’s reflectivity and scattering evolve over time**.

    * Filter the data by **coating date** to focus on a specific mirror coating cycle.  
    * Choose **start and end dates** to define the time window of interest.  
    * Choose either Reflectivity (SCI) or Scatter (SCE) measurements.
    * Optionally remove known **outlier dates** (comma-separated) to keep the trend clean.  
    * Turn on **Linear Regression** to overlay a best-fit trend line and display an R² value for each color band (B, G, R).

    This page is ideal for spotting gradual degradation, improvements after washes,
    or other long-term patterns in the Mayall primary mirror’s performance.
    """
)


# User selections
coating_choice = st.selectbox("Coating Date", coatings)
dates_range = get_dates_by_filter("Coating Date", coating_choice)

col1, col2 = st.columns(2)
with col1:
    start_date_choice = st.selectbox(
        "Start Date",
        dates_range[::-1],
        index=0,
    )
    end_date_choice = st.selectbox(
        "End Date",
        dates_range[::-1],
        index=len(dates_range) - 1,
    )
with col2:
    sc_choice = st.selectbox(
        "SC",
        ['SCI','SCE'],
        index=0,
    )
    outliers_choice = st.text_input(
        "Outlier Dates to Exclude (comma-separated YYYY-MM-DD)",
        value="2024-10-21",
    )

linreg_choice = st.checkbox("Show Linear Regression Fits")

if st.button("Generate Change Plot"):
    if coating_choice == 'All':
        coating_all = True
    else:
        coating_all = False
    data_ = change_plot(
        start_date_choice,
        end_date_choice,
        sc = sc_choice,
        linreg=linreg_choice,
        outliers_input=outliers_choice,
        coating_all=coating_all
    )
    if not data_.empty:
        time = Time.now()
        st.download_button(
            label="Download CSV",
            data=convert_for_download(data_),
            file_name=f"mayall_multiday_data_{sc_choice}_{time}.csv",
            mime="text/csv",
            icon=":material/download:",
        )
