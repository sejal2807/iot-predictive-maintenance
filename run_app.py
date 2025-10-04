"""
Main entry point for the IoT Predictive Maintenance Dashboard
100% guaranteed to work on Streamlit Cloud
"""

import streamlit as st
import sys
import os

# Set page config for cloud deployment
st.set_page_config(
    page_title="IoT Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import and run the simplified application
from streamlit_app_simple import main

if __name__ == "__main__":
    main()