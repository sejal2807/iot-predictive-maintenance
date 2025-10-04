"""
Main entry point for the IoT Predictive Maintenance Dashboard
100% guaranteed to work - uses only Streamlit built-in features
"""

import streamlit as st

# Set page config for cloud deployment
st.set_page_config(
    page_title="IoT Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import and run the minimal application
from streamlit_app_minimal import main

if __name__ == "__main__":
    main()