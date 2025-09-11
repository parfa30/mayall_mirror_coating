import streamlit as st

st.set_page_config(page_title="Mayall Mirror Data Viewer", layout="centered")

st.title("Mayall Primary Mirror Wash Data Viewer")

st.markdown(
    """
    **Welcome to the Mayall Primary Mirror Wash Data Viewer**

    This interactive app helps you explore the mirror's reflectivity and scattering data.

    * **Daily Plot** – Inspect results from a single day of measurements.  
    * **Time Plot** – Examine long-term trends across coating and wash events.

    Data are pulled directly from the observatory’s live measurement log.
    """
)
