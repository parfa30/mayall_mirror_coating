import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from datetime import datetime
import matplotlib as mpl

# --- Styling ---
mpl.rcParams['font.family'] = 'sans-serif'
colors = cm.get_cmap('tab20').colors

# --- Load data ---
mayall_data = pd.read_csv(
    'https://docs.google.com/spreadsheets/d/e/2PACX-1vSWF3pE1qBNcatq1Cm-H2z6mAGBPA-EbyUhujoOHXUU9-BfGYkmzP0YaFslu0bm2efn6AB9fNwUEUHj/pub?gid=0&single=true&output=csv',
    skiprows=[1,2])
wavelengths = mayall_data.columns[9:]
mayall_data.index = pd.to_datetime(mayall_data['Date'])
mayall_data.drop('Date', axis=1, inplace=True)

HCL_STAND_SCI = np.array([86.79, 86.88, 86.94, 86.97, 87., 86.99, 86.97, 86.94,
                          86.89, 86.85, 86.8, 86.73, 86.65, 86.57, 86.47, 86.36,
                          86.24, 86.12, 85.98, 85.83, 85.68, 85.51, 85.33, 85.14,
                          84.95, 84.75, 84.54, 84.33, 84.1, 83.86, 83.61])

standard_sci = pd.DataFrame(HCL_STAND_SCI).T
standard_sci.columns = wavelengths

# --- Options ---
wash_types = list(np.unique(mayall_data['Wash Type']))
wash_types.append('All')

coatings = list(np.unique(mayall_data['Coating Date']))
coatings.append('All')

# --- Helper functions ---
def get_dates(wash_type):
    if wash_type == 'All':
        dates = np.unique(mayall_data.index.strftime('%Y-%m-%d'))
    else:
        mask = mayall_data['Wash Type'] == wash_type
        dates = np.unique(mayall_data[mask].index.strftime('%Y-%m-%d'))
    return dates[::-1]

def get_dates2(coating):
    if coating == 'All':
        dates = np.unique(mayall_data.index.strftime('%Y-%m-%d'))
    else:
        mask = mayall_data['Coating Date'] == coating
        dates = np.unique(mayall_data[mask].index.strftime('%Y-%m-%d'))
    return dates[::-1]

def reduce_data(df_, sc='SCI', wash='AW'):
    grouped = df_.groupby(['Type', 'BW/AW', 'Tag'])[wavelengths].agg(['mean', 'sem', 'std', 'min', 'max'])
    cal_group = grouped.loc[('calibration', wash, 'SCI')].xs('mean', level=1)
    meas_group = grouped.loc[('measurement', wash, sc)]
    meas_mean = meas_group.xs('mean', level=1)
    meas_sem = meas_group.xs('sem', level=1)
    scale = cal_group / standard_sci.loc[0]
    reduced_data = meas_mean / scale
    return reduced_data, meas_sem

def lin_reg(x, y_with_err):
    y, y_err = y_with_err
    y = np.array(y, dtype=float)
    y_err = np.array(y_err, dtype=float)
    weights = 1 / (y_err ** 2)
    xx = pd.Series(x).map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(xx, y, sample_weight=weights)
    date_range = pd.date_range(start=min(x), end=max(x), periods=len(y))
    x_range = pd.Series(date_range).map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y_pred = model.predict(x_range)
    ss_res = np.sum(weights * (y - model.predict(xx)) ** 2)
    ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
    r2_weighted = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
    return date_range, y_pred, r2_weighted

# --- Plot functions ---
def daily_plot(date):
    sc_map = {'Reflectance': 'SCI', 'Scatter': 'SCE'}
    colors_dict = {'BW': 'blue', 'AW': 'orange'}
    d_ = mayall_data.loc[mayall_data.index == date]

    if d_.empty:
        st.warning(f"No data available for date {date}")
        return

    wash_type = d_['Wash Type'].iloc[0]

    for label, sc in sc_map.items():
        fig = plt.figure(figsize=(10, 4))
        gs = GridSpec(2, 1, height_ratios=[1, 3], hspace=0.0)
        ax_ratio = fig.add_subplot(gs[0])
        ax_values = fig.add_subplot(gs[1], sharex=ax_ratio)
        has_bw = False
        has_aw = False

        try:
            reduced_bw, bw_err = reduce_data(d_, sc, 'BW')
            reduced_bw.plot(ax=ax_values, yerr=bw_err, alpha=0.6, label='BW', color=colors_dict['BW'])
            has_bw = True
        except Exception as e:
            st.info(f"No data for Before Wash ({label}): {e}")

        try:
            reduced_aw, aw_err = reduce_data(d_, sc, 'AW')
            reduced_aw.plot(ax=ax_values, yerr=aw_err, alpha=0.6, label='AW', color=colors_dict['AW'])
            has_aw = True
        except Exception as e:
            st.info(f"No data for After Wash ({label}): {e}")

        if has_bw and has_aw:
            ratio = reduced_aw / reduced_bw
            ratio.plot(ax=ax_ratio, label='AW/BW', color='k', alpha=0.5)

        ax_ratio.set_title(f'Mean {label} ({sc}) Curves for {date}, {wash_type} wash')
        ax_ratio.set_ylabel('Wash % Increase')
        ax_ratio.legend()
        ax_ratio.tick_params(axis='x', labelbottom=False)
        ax_values.set_xlabel('Wavelength (nm)')
        ax_values.set_ylabel('Mean Value')
        ax_values.legend()
        plt.tight_layout()
        st.pyplot(fig)

def change_plot(start_date, end_date, linreg=False, outliers_input=""):
    this_df = mayall_data.loc[start_date:end_date]
    dates = pd.Series(this_df.index.unique())
    outliers = [o.strip() for o in outliers_input.split(',') if o.strip()]
    if outliers:
        mask = ~dates.astype(str).apply(lambda d: any(o in d for o in outliers))
        dates = dates[mask]

    results = []
    for date in dates:
        d_ = this_df.loc[this_df.index == date]
        try:
            reduced, err = reduce_data(d_)
        except Exception:
            reduced, err = reduce_data(d_, sc='SCI', wash='BW')
        results.append((date, (reduced, err)))

    x = pd.to_datetime([r[0] for r in results])
    y_means = [r[1][0] for r in results]
    y_errs = [r[1][1] for r in results]

    def mean_segment(data_list, start, end):
        if end is None:
            return [np.mean(d.iloc[start:]) for d in data_list]
        else:
            return [np.mean(d.iloc[start:end]) for d in data_list]

    B, G, R = (mean_segment(y_means, 0, 10),
               mean_segment(y_means, 10, 20),
               mean_segment(y_means, 20, None))
    B_err, G_err, R_err = (mean_segment(y_errs, 0, 10),
                           mean_segment(y_errs, 10, 20),
                           mean_segment(y_errs, 20, None))

    fig, ax = plt.subplots(figsize=(10, 4))
    colors_dict = {'B': 'blue', 'G': 'green', 'R': 'red'}
    data_series = {'B': (B, B_err), 'G': (G, G_err), 'R': (R, R_err)}

    for label, (means, errs) in data_series.items():
        ax.errorbar(x, means, yerr=errs, fmt='o', mfc=colors_dict[label], mec='black', ecolor='black')
        if linreg:
            date_range, y_fit, r2 = lin_reg(x, (means, errs))
            ax.plot(date_range, y_fit, color=colors_dict[label], label=f'{label} fit (R²={r2:.2f})')

    plt.tight_layout()
    st.pyplot(fig)

# --- Streamlit UI ---
st.title("Mayall Primary Mirror Wash Data Viewer")

st.subheader("Daily Plot")
wash_choice = st.radio("Wash Type:", wash_types)
date_choice = st.selectbox("Date:", get_dates(wash_choice))
if st.button("Generate Daily Plot"):
    daily_plot(date_choice)

st.subheader("Change Over Time Plot")
coating_choice = st.radio("Coating Date:", coatings)
dates_range = get_dates2(coating_choice)
start_date_choice = st.selectbox("Start Date:", dates_range[::-1])
end_date_choice = st.selectbox("End Date:", dates_range[::-1])
linreg_choice = st.checkbox("Linear Regression")
outliers_choice = st.text_input("Outliers (comma-separated YYYY-MM-DD)", "2024-10-21")

if st.button("Generate Change Plot"):
    change_plot(start_date_choice, end_date_choice, linreg=linreg_choice, outliers_input=outliers_choice)

st.subheader("Upload Data")
new_file = st.file_uploader("Upload a csv")
today = datetime.now()
new_date = st.text_input("Data Taken Date:", today.strftime("%Y-%m-%d"))
telescope = st.text_input("Telescope:","Mayall")
mirror = st.text_input("Mirror:","M1")
zone = st.text_input("Zone:",1)
coating_date = coatings[-1]
wash_type = st.selectbox("Wash Type:",wash_types)
