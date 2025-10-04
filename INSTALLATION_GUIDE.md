# IoT Predictive Maintenance Dashboard - Installation Guide

## 🐍 Python 3.13+ Compatibility Issues

The original requirements.txt uses older package versions that are **not compatible** with Python 3.13.7 due to the removal of `distutils` module.

## 🔧 Quick Fix Solutions

### Option 1: Use Python 3.13+ Compatible Version (Recommended)

```bash
# Install minimal dependencies
pip install streamlit plotly pandas numpy

# Run the simple version
streamlit run run_app_simple_py313.py
```

### Option 2: Use Enhanced Requirements

```bash
# Install with updated requirements
pip install -r requirements_enhanced.txt

# Run the enhanced version
streamlit run run_app_enhanced.py
```

### Option 3: Use Installation Script

```bash
# Run the automated installation script
python install_dependencies.py

# Then run the dashboard
streamlit run run_app_simple_py313.py
```

## 📦 Available Versions

### 1. **run_app_simple_py313.py** (Recommended for Python 3.13+)
- ✅ **Python 3.13+ Compatible**
- ✅ **Minimal Dependencies** (streamlit, pandas, numpy)
- ✅ **Real-time Features**
- ✅ **Modern UI**
- ✅ **No Plotly dependency issues**

### 2. **run_app_enhanced.py** (Full Features)
- ✅ **Advanced Visualizations** (requires Plotly)
- ✅ **Real-time Simulation**
- ✅ **Modern UI with Animations**
- ⚠️ **May require additional dependencies**

### 3. **run_app.py** (Original Fixed)
- ✅ **Basic Features**
- ✅ **Fixed st.set_page_config() error**
- ⚠️ **May have dependency issues**

## 🚀 Quick Start

### For Python 3.13+ Users:

```bash
# 1. Install minimal dependencies
pip install streamlit pandas numpy

# 2. Run the simple version
streamlit run run_app_simple_py313.py
```

### For Users with Compatible Python Versions:

```bash
# 1. Install all dependencies
pip install -r requirements_enhanced.txt

# 2. Run the enhanced version
streamlit run run_app_enhanced.py
```

## 🔍 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'distutils'"
**Solution:** Use `run_app_simple_py313.py` or install with updated requirements.

### Error: "Failed to download and build numpy==1.24.3"
**Solution:** Use the Python 3.13+ compatible requirements.

### Error: "StreamlitAPIException: set_page_config() can only be called once"
**Solution:** This has been fixed in all new versions.

## 📊 Features Comparison

| Feature | Simple Py313 | Enhanced | Original |
|---------|---------------|----------|----------|
| Python 3.13+ Compatible | ✅ | ✅ | ⚠️ |
| Real-time Data | ✅ | ✅ | ✅ |
| Modern UI | ✅ | ✅ | ✅ |
| Interactive Charts | ✅ | ✅ | ✅ |
| Live Simulation | ✅ | ✅ | ❌ |
| Advanced Visualizations | ❌ | ✅ | ❌ |
| Minimal Dependencies | ✅ | ❌ | ✅ |

## 🎯 Recommended Setup

For **Python 3.13+** users, use:
```bash
streamlit run run_app_simple_py313.py
```

This version provides all the essential features without dependency issues!
