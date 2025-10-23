import streamlit as st

st.set_page_config(page_title="Mayall Mirror Data Viewer", layout="centered")

st.title("Mayall Mirror Wash Data Viewer")

st.markdown(
    """
    **Welcome to the Mayall Mirror Wash Data Viewer**

    This interactive app helps you explore the mirror's reflectivity and scattering data.

    * **Daily Plot** – Inspect results from a single day of measurements.  
    * **Time Plot** – Examine long-term trends across coating and wash events.
    * **Before/After Plots** - Comparing the Before and After measurements from one date to another.
    * **Upload Data** – Upload new mirror reflectivity/scattering measurements.

    """
)
