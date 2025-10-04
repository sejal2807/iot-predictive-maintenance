# Streamlit Cloud Deployment Guide

## 🚀 Recommended Version for Streamlit Cloud

**Use: `run_app_simple_py313.py`**

### Why This Version?

✅ **Python 3.13+ Compatible** - Works with Streamlit Cloud's Python environment  
✅ **Minimal Dependencies** - Only requires streamlit, pandas, numpy  
✅ **No Plotly Issues** - Avoids complex visualization dependencies  
✅ **Real-time Features** - Full IoT monitoring capabilities  
✅ **Modern UI** - Professional dashboard design  
✅ **Error-free** - No st.set_page_config() or dependency conflicts  

## 📋 Deployment Steps

### 1. Update Your Repository

Make sure these files are in your repository:
- `run_app_simple_py313.py` (main app)
- `requirements_minimal.txt` (dependencies)
- `src/` folder with all source files (if needed)

### 2. Create/Update requirements.txt for Streamlit Cloud

```txt
# Streamlit Cloud Requirements (Python 3.13+ Compatible)
streamlit>=1.39.0
pandas>=2.2.0
numpy>=1.26.0
```

### 3. Deploy to Streamlit Cloud

1. **Push to GitHub** (if not already done)
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub repository**
4. **Set Main file path to:** `run_app_simple_py313.py`
5. **Deploy!**

## 🔧 Alternative: Use Enhanced Version (If You Want Advanced Features)

If you want the full-featured version with Plotly charts:

### Update requirements.txt:
```txt
# Enhanced Requirements for Streamlit Cloud
streamlit>=1.39.0
plotly>=5.22.0
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
tensorflow>=2.16.0
pyyaml>=6.0.1
altair>=5.0.0
```

### Set Main file path to: `run_app_enhanced.py`

## ⚠️ Why NOT to Use Original Versions

❌ **`run_app.py`** - May have dependency issues on Streamlit Cloud  
❌ **`run_app_simple.py`** - Missing modern features  
❌ **`src/streamlit_app.py`** - Not a main entry point  

## 🎯 Final Recommendation

**For Streamlit Cloud: Use `run_app_simple_py313.py`**

This version provides:
- ✅ **100% compatibility** with Streamlit Cloud
- ✅ **All essential features** for IoT monitoring
- ✅ **Real-time data simulation**
- ✅ **Modern UI with animations**
- ✅ **Anomaly detection and alerts**
- ✅ **Predictive maintenance recommendations**
- ✅ **No dependency conflicts**

## 🚀 Quick Deployment

1. **Ensure `run_app_simple_py313.py` is in your repo root**
2. **Create/update `requirements.txt` with minimal dependencies**
3. **Deploy to Streamlit Cloud**
4. **Set main file to `run_app_simple_py313.py`**
5. **Enjoy your live IoT dashboard!**

Your dashboard will be available at: `https://your-app-name.streamlit.app`
