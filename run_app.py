"""
Main entry point for the IoT Predictive Maintenance Dashboard
Run this file to start the Streamlit application
"""

import streamlit as st
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main application
from src.streamlit_app import main

if __name__ == "__main__":
    main()
