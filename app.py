"""
🎯 McKINSEY STİLİ TİCARİ PORTFÖY ANALİZ SİSTEMİ
Professional Analytics with McKinsey Design Language - ENHANCED VERSION

Özellikler:
- 🎨 McKinsey renk paleti ve görsel kimlik (Full Compliance)
- 🗺️ Türkiye il bazlı harita görselleştirme (GELİŞTİRİLMİŞ VERSİYON)
- 🤖 GERÇEK Machine Learning (Linear Regression, Ridge, Random Forest, Gradient Boosting)
- 📊 McKinsey tarzı grafikler ve analizler
- 📈 Aylık/Yıllık görünüm, Trend analizleri
- 📋 Profesyonel raporlama ve insight'lar
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from datetime import datetime, timedelta
import warnings
from io import BytesIO
import json
import base64
from typing import Dict, List, Tuple, Optional, Any
import math

# Machine Learning imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.pipeline import Pipeline

# Geospatial imports
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium

# Statistical imports
from scipy import stats
from scipy.stats import zscore, linregress
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG - McKINSEY STYLE
# =============================================================================
st.set_page_config(
    page_title="McKinsey & Company | Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.mckinsey.com/',
        'Report a bug': None,
        'About': 'McKinsey Portfolio Analytics Dashboard v2.0'
    }
)

# =============================================================================
# McKINSEY COLOR PALETTE (Official Colors from mckinsey.com)
# =============================================================================
MCKINSEY_COLORS = {
    # Primary Brand Colors (From https://www.mckinsey.com)
    "navy": "#002856",           # McKinsey Navy - Primary Brand Color
    "blue": "#0066CC",           # McKinsey Blue - Secondary Color
    "light_blue": "#4A90E2",     # Light Blue for highlights
    "teal": "#00A3A1",           # McKinsey Teal - Data Visualization
    "gold": "#FFB81C",           # McKinsey Gold - Accent Color
    "yellow": "#F7A800",         # Yellow for alerts
    
    # Extended Data Visualization Palette
    "green": "#39B54A",          # Positive/Go Green
    "red": "#E31837",            # Negative/Stop Red (McKinsey Red)
    "orange": "#FF6900",         # Warning/Attention Orange
    "purple": "#663399",         # Special Category Purple
    "pink": "#E6007E",           # Highlight Pink
    
    # Neutrals (McKinsey Corporate)
    "dark_gray": "#333333",      # Dark Gray for text
    "gray": "#666666",           # Medium Gray
    "light_gray": "#999999",     # Light Gray
    "background": "#F5F7FA",     # Background Gray (McKinsey style)
    "white": "#FFFFFF",          # White
    
    # Chart Specific Colors (Sequential)
    "chart_1": "#002856",        # Navy
    "chart_2": "#0066CC",        # Blue
    "chart_3": "#00A3A1",        # Teal
    "chart_4": "#39B54A",        # Green
    "chart_5": "#FFB81C",        # Gold
    "chart_6": "#FF6900",        # Orange
    "chart_7": "#E31837",        # Red
    "chart_8": "#663399",        # Purple
    
    # Gradient Colors for Maps
    "gradient_start": "#002856",
    "gradient_mid": "#0066CC",
    "gradient_end": "#4A90E2",
}

# McKinsey Chart Color Sequence (8-color palette for charts)
MCKINSEY_CHART_SEQUENCE = [
    MCKINSEY_COLORS["navy"],     # 1. Navy
    MCKINSEY_COLORS["blue"],     # 2. Blue
    MCKINSEY_COLORS["teal"],     # 3. Teal
    MCKINSEY_COLORS["green"],    # 4. Green
    MCKINSEY_COLORS["gold"],     # 5. Gold
    MCKINSEY_COLORS["orange"],   # 6. Orange
    MCKINSEY_COLORS["purple"],   # 7. Purple
    MCKINSEY_COLORS["pink"],     # 8. Pink
]

# Performance Rating Colors (McKinsey Style)
PERFORMANCE_COLORS = {
    "excellent": MCKINSEY_COLORS["green"],
    "good": MCKINSEY_COLORS["teal"],
    "average": MCKINSEY_COLORS["gold"],
    "below": MCKINSEY_COLORS["orange"],
    "poor": MCKINSEY_COLORS["red"],
}

# Region Colors for Turkey Map
REGION_COLORS = {
    "MARMARA": MCKINSEY_COLORS["navy"],
    "EGE": MCKINSEY_COLORS["blue"],
    "AKDENİZ": MCKINSEY_COLORS["teal"],
    "İÇ ANADOLU": MCKINSEY_COLORS["gold"],
    "DOĞU ANADOLU": MCKINSEY_COLORS["orange"],
    "GÜNEYDOĞU ANADOLU": MCKINSEY_COLORS["red"],
    "KARADENİZ": MCKINSEY_COLORS["green"],
    "BATI ANADOLU": MCKINSEY_COLORS["purple"],
    "KUZEYDOĞU ANADOLU": MCKINSEY_COLORS["pink"],
    "DİĞER": MCKINSEY_COLORS["light_gray"]
}

# BCG Matrix Colors (McKinsey Interpretation)
BCG_COLORS = {
    "⭐ Star": MCKINSEY_COLORS["green"],
    "🐄 Cash Cow": MCKINSEY_COLORS["navy"],
    "❓ Question Mark": MCKINSEY_COLORS["orange"],
    "🐶 Dog": MCKINSEY_COLORS["red"]
}

# Investment Priority Colors
INVESTMENT_COLORS = {
    "🚀 Aggressive Investment": MCKINSEY_COLORS["red"],
    "📈 Growth Acceleration": MCKINSEY_COLORS["orange"],
    "🛡️ Maintenance": MCKINSEY_COLORS["navy"],
    "💎 High Potential": MCKINSEY_COLORS["teal"],
    "👁️ Monitor": MCKINSEY_COLORS["gray"]
}

# =============================================================================
# McKINSEY CSS STYLING (Complete mckinsey.com styling)
# =============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    /* Base Styles */
    * {{
        font-family: 'Source Sans Pro', 'Lato', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    .stApp {{
        background-color: {MCKINSEY_COLORS["background"]};
    }}
    
    /* McKinsey Header - Official Style */
    .mckinsey-header {{
        background: linear-gradient(135deg, {MCKINSEY_COLORS["navy"]} 0%, {MCKINSEY_COLORS["blue"]} 100%);
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 0;
        margin: -2rem -1rem 2rem -1rem;
        box-shadow: 0 4px 12px rgba(0, 40, 86, 0.15);
        position: relative;
        overflow: hidden;
    }}
    
    .mckinsey-header::before {{
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(255, 184, 28, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }}
    
    .mckinsey-title {{
        font-size: 2.8rem;
        font-weight: 900;
        color: white;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.5px;
        font-family: 'Source Sans Pro', sans-serif;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}
    
    .mckinsey-subtitle {{
        font-size: 1.1rem;
        font-weight: 300;
        color: rgba(255, 255, 255, 0.95);
        margin: 0;
        letter-spacing: 0.5px;
    }}
    
    .mckinsey-badge {{
        display: inline-block;
        background: {MCKINSEY_COLORS["gold"]};
        color: {MCKINSEY_COLORS["navy"]};
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        margin-left: 1rem;
        vertical-align: middle;
    }}
    
    /* McKinsey Metric Cards */
    div[data-testid="metric-container"] {{
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid {MCKINSEY_COLORS["navy"]};
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    
    div[data-testid="metric-container"]::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, {MCKINSEY_COLORS["navy"]}, {MCKINSEY_COLORS["blue"]});
    }}
    
    div[data-testid="metric-container"]:hover {{
        box-shadow: 0 8px 24px rgba(0, 40, 86, 0.15);
        transform: translateY(-4px);
    }}
    
    div[data-testid="stMetricValue"] {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {MCKINSEY_COLORS["navy"]};
        font-family: 'Source Sans Pro', sans-serif;
        line-height: 1.2;
    }}
    
    div[data-testid="stMetricLabel"] {{
        font-size: 0.85rem;
        font-weight: 600;
        color: {MCKINSEY_COLORS["gray"]};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }}
    
    div[data-testid="stMetricDelta"] {{
        font-size: 0.9rem;
        font-weight: 600;
        font-family: 'Source Sans Pro', sans-serif;
    }}
    
    /* McKinsey Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: white;
        border-bottom: 2px solid #E0E6EF;
        padding: 0;
        border-radius: 8px 8px 0 0;
        overflow: hidden;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: {MCKINSEY_COLORS["dark_gray"]};
        font-weight: 600;
        font-size: 0.95rem;
        padding: 1rem 2rem;
        background: transparent;
        border: none;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
        font-family: 'Source Sans Pro', sans-serif;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        color: {MCKINSEY_COLORS["navy"]};
        background: rgba(0, 40, 86, 0.05);
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {MCKINSEY_COLORS["navy"]};
        background: white;
        border-bottom-color: {MCKINSEY_COLORS["gold"]};
        box-shadow: 0 2px 4px rgba(0, 40, 86, 0.1);
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {MCKINSEY_COLORS["navy"]};
        font-weight: 700;
        font-family: 'Source Sans Pro', sans-serif;
        margin-top: 0;
    }}
    
    h1 {{
        font-size: 2.2rem;
        border-bottom: 3px solid {MCKINSEY_COLORS["gold"]};
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem;
        position: relative;
    }}
    
    h1::after {{
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 60px;
        height: 3px;
        background: {MCKINSEY_COLORS["blue"]};
    }}
    
    h2 {{
        font-size: 1.8rem;
        color: {MCKINSEY_COLORS["navy"]};
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-left: 1rem;
        border-left: 4px solid {MCKINSEY_COLORS["blue"]};
    }}
    
    h3 {{
        font-size: 1.4rem;
        color: {MCKINSEY_COLORS["navy"]};
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }}
    
    /* Buttons - McKinsey Style */
    .stButton > button {{
        background: linear-gradient(135deg, {MCKINSEY_COLORS["navy"]} 0%, {MCKINSEY_COLORS["blue"]} 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-family: 'Source Sans Pro', sans-serif;
        box-shadow: 0 2px 4px rgba(0, 40, 86, 0.2);
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(135deg, {MCKINSEY_COLORS["blue"]} 0%, {MCKINSEY_COLORS["teal"]} 100%);
        box-shadow: 0 4px 12px rgba(0, 40, 86, 0.3);
        transform: translateY(-2px);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0, 40, 86, 0.2);
    }}
    
    .stButton > button:disabled {{
        background: {MCKINSEY_COLORS["light_gray"]};
        color: {MCKINSEY_COLORS["gray"]};
    }}
    
    /* Secondary Button */
    .stButton > button.secondary {{
        background: white;
        color: {MCKINSEY_COLORS["navy"]};
        border: 2px solid {MCKINSEY_COLORS["navy"]};
    }}
    
    /* Sidebar - McKinsey Style */
    [data-testid="stSidebar"] {{
        background: white;
        border-right: 1px solid #E0E6EF;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.05);
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 2rem;
    }}
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {MCKINSEY_COLORS["navy"]};
        padding-left: 1rem;
    }}
    
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {{
        color: {MCKINSEY_COLORS["dark_gray"]};
        font-weight: 600;
        font-size: 0.85rem;
    }}
    
    /* Section Dividers */
    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, {MCKINSEY_COLORS["light_gray"]}, transparent);
        margin: 2rem 0;
    }}
    
    .mckinsey-divider {{
        height: 3px;
        background: linear-gradient(90deg, {MCKINSEY_COLORS["navy"]}, {MCKINSEY_COLORS["blue"]}, {MCKINSEY_COLORS["teal"]});
        margin: 2.5rem 0;
        border-radius: 3px;
    }}
    
    /* Alert/Info Boxes */
    .stAlert {{
        border-radius: 8px;
        border-left: 5px solid {MCKINSEY_COLORS["blue"]};
        background: rgba(0, 102, 204, 0.05);
        box-shadow: 0 2px 4px rgba(0, 40, 86, 0.08);
    }}
    
    /* McKinsey Insight Box */
    .mckinsey-insight {{
        background: linear-gradient(135deg, rgba(0, 40, 86, 0.95) 0%, rgba(0, 102, 204, 0.95) 100%);
        color: white;
        padding: 1.75rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 40, 86, 0.15);
        position: relative;
        overflow: hidden;
    }}
    
    .mckinsey-insight::before {{
        content: '💡';
        position: absolute;
        top: -20px;
        right: -20px;
        font-size: 8rem;
        opacity: 0.1;
        transform: rotate(15deg);
    }}
    
    .mckinsey-insight h3 {{
        color: white;
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .mckinsey-insight h3::before {{
        content: '';
        width: 4px;
        height: 24px;
        background: {MCKINSEY_COLORS["gold"]};
        border-radius: 2px;
    }}
    
    .mckinsey-insight p {{
        color: rgba(255, 255, 255, 0.95);
        margin: 0.75rem 0;
        line-height: 1.6;
        font-size: 1rem;
    }}
    
    /* McKinsey Card */
    .mckinsey-card {{
        background: white;
        padding: 1.75rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        border-top: 4px solid {MCKINSEY_COLORS["navy"]};
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .mckinsey-card:hover {{
        box-shadow: 0 8px 24px rgba(0, 40, 86, 0.12);
        transform: translateY(-2px);
    }}
    
    .mckinsey-card h4 {{
        color: {MCKINSEY_COLORS["navy"]};
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 700;
    }}
    
    .mckinsey-card .card-content {{
        color: {MCKINSEY_COLORS["dark_gray"]};
        line-height: 1.6;
    }}
    
    /* Data Tables - McKinsey Style */
    .dataframe {{
        font-size: 0.9rem;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}
    
    .dataframe thead tr th {{
        background: linear-gradient(135deg, {MCKINSEY_COLORS["navy"]} 0%, {MCKINSEY_COLORS["blue"]} 100%);
        color: white !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
        padding: 1rem 0.75rem !important;
        border: none;
        font-family: 'Source Sans Pro', sans-serif;
    }}
    
    .dataframe tbody tr {{
        transition: background-color 0.2s ease;
    }}
    
    .dataframe tbody tr:nth-child(even) {{
        background-color: {MCKINSEY_COLORS["background"]};
    }}
    
    .dataframe tbody tr:hover {{
        background-color: rgba(0, 102, 204, 0.1) !important;
    }}
    
    .dataframe tbody td {{
        padding: 0.75rem;
        border-bottom: 1px solid #E0E6EF;
        color: {MCKINSEY_COLORS["dark_gray"]};
    }}
    
    /* Input Widgets */
    .stSelectbox, .stSlider, .stRadio, .stDateInput, .stNumberInput {{
        margin-bottom: 1rem;
    }}
    
    .stSelectbox > div > div {{
        border: 2px solid #E0E6EF;
        border-radius: 6px;
        transition: all 0.3s ease;
    }}
    
    .stSelectbox > div > div:hover {{
        border-color: {MCKINSEY_COLORS["blue"]};
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
    }}
    
    .stSlider > div {{
        padding-top: 1.5rem;
    }}
    
    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {MCKINSEY_COLORS["navy"]} 0%, {MCKINSEY_COLORS["blue"]} 100%);
        border-radius: 4px;
    }}
    
    /* File Uploader */
    [data-testid="stFileUploader"] {{
        border: 2px dashed {MCKINSEY_COLORS["blue"]};
        border-radius: 8px;
        padding: 2rem;
        background: rgba(0, 102, 204, 0.03);
        transition: all 0.3s ease;
    }}
    
    [data-testid="stFileUploader"]:hover {{
        border-color: {MCKINSEY_COLORS["navy"]};
        background: rgba(0, 102, 204, 0.08);
    }}
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {MCKINSEY_COLORS["background"]};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, {MCKINSEY_COLORS["navy"]} 0%, {MCKINSEY_COLORS["blue"]} 100%);
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, {MCKINSEY_COLORS["blue"]} 0%, {MCKINSEY_COLORS["teal"]} 100%);
    }}
    
    /* McKinsey Badge */
    .mckinsey-badge {{
        display: inline-block;
        background: linear-gradient(135deg, {MCKINSEY_COLORS["gold"]} 0%, {MCKINSEY_COLORS["orange"]} 100%);
        color: {MCKINSEY_COLORS["navy"]};
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        margin-left: 0.5rem;
        vertical-align: middle;
    }}
    
    /* McKinsey Grid */
    .mckinsey-grid {{
        display: grid;
        gap: 1.5rem;
        margin: 1.5rem 0;
    }}
    
    .grid-2 {{
        grid-template-columns: repeat(2, 1fr);
    }}
    
    .grid-3 {{
        grid-template-columns: repeat(3, 1fr);
    }}
    
    .grid-4 {{
        grid-template-columns: repeat(4, 1fr);
    }}
    
    /* Responsive Design */
    @media (max-width: 1200px) {{
        .grid-4 {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}
    
    @media (max-width: 768px) {{
        .grid-2, .grid-3, .grid-4 {{
            grid-template-columns: 1fr;
        }}
        
        .mckinsey-title {{
            font-size: 2rem;
        }}
        
        .mckinsey-subtitle {{
            font-size: 1rem;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS & DATA MAPPINGS
# =============================================================================

FIX_CITY_MAP = {
    "AGRI": "AĞRI",
    "BARTÄ±N": "BARTIN",
    "BINGÃ¶L": "BİNGÖL",
    "DÃ1⁄4ZCE": "DÜZCE",
    "ELAZIG": "ELAZIĞ",
    "ESKISEHIR": "ESKİŞEHİR",
    "GÃ1⁄4MÃ1⁄4SHANE": "GÜMÜŞHANE",
    "HAKKARI": "HAKKARİ",
    "ISTANBUL": "İSTANBUL",
    "IZMIR": "İZMİR",
    "IÄ\x9fDIR": "IĞDIR",
    "KARABÃ1⁄4K": "KARABÜK",
    "KINKKALE": "KIRIKKALE",
    "KIRSEHIR": "KIRŞEHİR",
    "KÃ1⁄4TAHYA": "KÜTAHYA",
    "MUGLA": "MUĞLA",
    "MUS": "MUŞ",
    "NEVSEHIR": "NEVŞEHİR",
    "NIGDE": "NİĞDE",
    "SANLIURFA": "ŞANLIURFA",
    "SIRNAK": "ŞIRNAK",
    "TEKIRDAG": "TEKİRDAĞ",
    "USAK": "UŞAK",
    "ZINGULDAK": "ZONGULDAK",
    "Ã\x87ANAKKALE": "ÇANAKKALE",
    "Ã\x87ANKIRI": "ÇANKIRI",
    "Ã\x87ORUM": "ÇORUM",
    "K. MARAS": "KAHRAMANMARAŞ",
    "CORUM": "ÇORUM",
    "CANKIRI": "ÇANKIRI",
    "ZONGULDAK": "ZONGULDAK",
    "KARABUK": "KARABÜK",
    "GUMUSHANE": "GÜMÜŞHANE",
    "ELÂZıĞ": "ELAZIĞ",
    "KUTAHYA": "KÜTAHYA",
    "CANAKKALE": "ÇANAKKALE"
}

CITY_NORMALIZE_CLEAN = {
    'ADANA': 'Adana',
    'ADIYAMAN': 'Adiyaman',
    'AFYONKARAHISAR': 'Afyonkarahisar',
    'AFYON': 'Afyonkarahisar',
    'AGRI': 'Agri',
    'AĞRI': 'Agri',
    'ANKARA': 'Ankara',
    'ANTALYA': 'Antalya',
    'AYDIN': 'Aydin',
    'BALIKESIR': 'Balikesir',
    'BARTIN': 'Bartin',
    'BATMAN': 'Batman',
    'BILECIK': 'Bilecik',
    'BINGOL': 'Bingol',
    'BITLIS': 'Bitlis',
    'BOLU': 'Bolu',
    'BURDUR': 'Burdur',
    'BURSA': 'Bursa',
    'CANAKKALE': 'Canakkale',
    'ÇANAKKALE': 'Canakkale',
    'CANKIRI': 'Cankiri',
    'ÇANKIRI': 'Cankiri',
    'CORUM': 'Corum',
    'ÇORUM': 'Corum',
    'DENIZLI': 'Denizli',
    'DIYARBAKIR': 'Diyarbakir',
    'DUZCE': 'Duzce',
    'DÜZCE': 'Duzce',
    'EDIRNE': 'Edirne',
    'ELAZIG': 'Elazig',
    'ELAZĞ': 'Elazig',
    'ELAZIĞ': 'Elazig',
    'ERZINCAN': 'Erzincan',
    'ERZURUM': 'Erzurum',
    'ESKISEHIR': 'Eskisehir',
    'ESKİŞEHİR': 'Eskisehir',
    'GAZIANTEP': 'Gaziantep',
    'GIRESUN': 'Giresun',
    'GİRESUN': 'Giresun',
    'GUMUSHANE': 'Gumushane',
    'GÜMÜŞHANE': 'Gumushane',
    'HAKKARI': 'Hakkari',
    'HATAY': 'Hatay',
    'IGDIR': 'Igdir',
    'IĞDIR': 'Igdir',
    'ISPARTA': 'Isparta',
    'ISTANBUL': 'Istanbul',
    'İSTANBUL': 'Istanbul',
    'IZMIR': 'Izmir',
    'İZMİR': 'Izmir',
    'KAHRAMANMARAS': 'K. Maras',
    'KAHRAMANMARAŞ': 'K. Maras',
    'K.MARAS': 'K. Maras',
    'KMARAS': 'K. Maras',
    'KARABUK': 'Karabuk',
    'KARABÜK': 'Karabuk',
    'KARAMAN': 'Karaman',
    'KARS': 'Kars',
    'KASTAMONU': 'Kastamonu',
    'KAYSERI': 'Kayseri',
    'KIRIKKALE': 'Kinkkale',
    'KIRKLARELI': 'Kirklareli',
    'KIRKLARELİ': 'Kirklareli',
    'KIRSEHIR': 'Kirsehir',
    'KIRŞEHİR': 'Kirsehir',
    'KILIS': 'Kilis',
    'KİLİS': 'Kilis',
    'KOCAELI': 'Kocaeli',
    'KONYA': 'Konya',
    'KUTAHYA': 'Kutahya',
    'KÜTAHYA': 'Kutahya',
    'MALATYA': 'Malatya',
    'MANISA': 'Manisa',
    'MANİSA': 'Manisa',
    'MARDIN': 'Mardin',
    'MARDİN': 'Mardin',
    'MERSIN': 'Mersin',
    'MERSİN': 'Mersin',
    'MUGLA': 'Mugla',
    'MUĞLA': 'Mugla',
    'MUS': 'Mus',
    'MUŞ': 'Mus',
    'NEVSEHIR': 'Nevsehir',
    'NEVŞEHİR': 'Nevsehir',
    'NIGDE': 'Nigde',
    'NİĞDE': 'Nigde',
    'ORDU': 'Ordu',
    'OSMANIYE': 'Osmaniye',
    'OSMANİYE': 'Osmaniye',
    'RIZE': 'Rize',
    'RİZE': 'Rize',
    'SAKARYA': 'Sakarya',
    'SAMSUN': 'Samsun',
    'SIIRT': 'Siirt',
    'SİİRT': 'Siirt',
    'SINOP': 'Sinop',
    'SİNOP': 'Sinop',
    'SIVAS': 'Sivas',
    'SİVAS': 'Sivas',
    'SANLIURFA': 'Sanliurfa',
    'ŞANLIURFA': 'Sanliurfa',
    'SIRNAK': 'Sirnak',
    'ŞIRNAK': 'Sirnak',
    'TEKIRDAG': 'Tekirdag',
    'TEKİRDAĞ': 'Tekirdag',
    'TOKAT': 'Tokat',
    'TRABZON': 'Trabzon',
    'TUNCELI': 'Tunceli',
    'TUNCELİ': 'Tunceli',
    'USAK': 'Usak',
    'UŞAK': 'Usak',
    'VAN': 'Van',
    'YALOVA': 'Yalova',
    'YOZGAT': 'Yozgat',
    'ZONGULDAK': 'Zonguldak',
    'ARDAHAN': 'Ardahan'
}

# Turkey City Coordinates (Major Cities)
CITY_COORDINATES = {
    'ADANA': (35.3213, 37.0000),
    'ANKARA': (32.8597, 39.9334),
    'ANTALYA': (30.7133, 36.8969),
    'BURSA': (29.0611, 40.1826),
    'ISTANBUL': (28.9784, 41.0082),
    'IZMIR': (27.1428, 38.4237),
    'KONYA': (32.4846, 37.8667),
    'GAZIANTEP': (37.3826, 37.0662),
    'MERSIN': (34.6415, 36.8121),
    'KAYSERI': (35.4890, 38.7312),
    'ESKISEHIR': (30.5256, 39.7767),
    'DIYARBAKIR': (40.2305, 37.9144),
    'SAMSUN': (36.3361, 41.2867),
    'DENIZLI': (29.0864, 37.7765),
    'ADAPAZARI': (30.4000, 40.7750),
    'MALATYA': (38.3167, 38.3552),
    'ERZURUM': (41.2751, 39.9000),
    'VAN': (43.3800, 38.4942),
    'ELAZIG': (39.2333, 38.6810),
    'BATMAN': (41.1367, 37.8812),
    'SIVAS': (37.0167, 39.7477),
    'TRABZON': (39.7167, 41.0015),
    'MANISA': (27.4265, 38.6191),
    'TEKIRDAG': (27.5117, 40.9781),
    'ISPARTA': (30.5567, 37.7648),
    'BALIKESIR': (27.8865, 39.6484),
    'AYDIN': (27.8456, 37.8560),
    'OSMANIYE': (36.2476, 37.0742),
    'KIRIKKALE': (33.5086, 39.8468),
    'ANTEP': (37.3826, 37.0662),
    'USAK': (29.4058, 38.6823),
    'AKSARAY': (34.0254, 38.3687),
    'CANKIRI': (33.6167, 40.6000),
    'KARABUK': (32.6228, 41.2049),
    'MUGLA': (28.3667, 37.2167),
    'NIGDE': (34.6783, 37.9667),
    'HAKKARI': (43.7333, 37.5833),
    'KARS': (43.0949, 40.5927),
    'KILIS': (37.1150, 36.7161),
    'ARDAHAN': (42.7022, 41.1105),
    'GIRESUN': (38.3894, 40.9128),
    'RİZE': (40.5219, 41.0201),
    'BAYBURT': (40.2276, 40.2552),
    'BARTIN': (32.3375, 41.6356),
    'ARDAHAN': (42.7022, 41.1105),
    'IGDIR': (44.0449, 39.9167),
    'YALOVA': (29.2700, 40.6500),
    'KARAMAN': (33.2222, 37.1811),
    'SIRNAK': (42.4634, 37.5200),
    'BINGOL': (40.4931, 38.8843),
    'BITLIS': (42.1089, 38.4000),
    'TUNCELI': (39.5472, 39.1000),
    'ZONGULDAK': (31.7931, 41.4564),
    'DUZCE': (31.1639, 40.8438),
    'KUTAHYA': (29.9857, 39.4167),
    'CORUM': (34.9533, 40.5506),
    'EDIRNE': (26.5557, 41.6772),
    'YOZGAT': (34.8000, 39.8200),
    'KASTAMONU': (33.7750, 41.3765),
    'NEVSEHIR': (34.7128, 38.6244),
    'SINOP': (35.1500, 42.0267),
    'ORDU': (37.8833, 40.9833),
    'KIRSEHIR': (34.1667, 39.1500),
    'AMASYA': (35.8333, 40.6500),
    'BOLU': (31.6081, 40.7358),
    'CANAKKALE': (26.4144, 40.1553),
    'ERZINCAN': (39.4906, 39.7500),
    'GUMUSHANE': (39.4814, 40.4600),
    'HATAY': (36.3498, 36.4018),
    'MARAS': (36.9261, 37.5833),
    'MARDIN': (40.7376, 37.3122),
    'MUS': (41.4969, 38.7333),
    'TOKAT': (36.5544, 40.3167),
    'AFYON': (30.5433, 38.7636),
    'BILECIK': (29.9793, 40.1501),
    'BURDUR': (30.2833, 37.7167),
    'CESME': (26.3064, 38.3228),
    'FETHIYE': (29.1167, 36.6217),
    'FINIKE': (30.1450, 36.3000),
    'GELIBOLU': (26.6706, 40.4106),
    'INEGOL': (29.5136, 40.0833),
    'ISKENDERUN': (36.1736, 36.5872),
    'IZMIT': (29.9167, 40.7667),
    'KARTAL': (29.1872, 40.9031),
    'KUSADASI': (27.2586, 37.8578),
    'MARMARIS': (28.2742, 36.8547),
    'NIZIP': (37.7944, 37.0094),
    'SALIHLI': (28.1472, 38.4822),
    'SIIRT': (41.9406, 37.9333),
    'SILIFKE': (33.9333, 36.3833),
    'SILIVRI': (28.2469, 41.0739),
    'SOMA': (27.6094, 39.1883),
    'SULEYMANIYE': (45.4333, 35.5617),
    'TAVAS': (29.0700, 37.5739),
    'TEKIR': (27.5117, 40.9781),
    'TURGUTLU': (27.6994, 38.5000),
    'URLA': (26.7647, 38.3222),
    'VAN': (43.3800, 38.4942),
    'YENISEHIR': (29.6539, 40.2644),
    'YUZGAT': (34.8000, 39.8200),
    'ZILE': (35.8861, 40.3031)
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Güvenli bölme işlemi - division by zero koruması"""
    return np.where(b != 0, a / b, 0)

def format_currency(value: float) -> str:
    """Para formatı: K, M, B"""
    if abs(value) >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.0f}"

def format_number(value: float, decimals: int = 1) -> str:
    """Genel sayı formatı"""
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Yüzde formatı"""
    return f"{value:.{decimals}f}%"

def get_product_columns(product: str) -> Dict[str, str]:
    """Ürün kolonlarını döndür"""
    product_map = {
        "TROCMETAM": {"pf": "TROCMETAM", "rakip": "DIGER TROCMETAM"},
        "CORTIPOL": {"pf": "CORTIPOL", "rakip": "DIGER CORTIPOL"},
        "DEKSAMETAZON": {"pf": "DEKSAMETAZON", "rakip": "DIGER DEKSAMETAZON"},
        "PF IZOTONIK": {"pf": "PF IZOTONIK", "rakip": "DIGER IZOTONIK"}
    }
    return product_map.get(product, {"pf": product, "rakip": f"DIGER {product}"})

def normalize_city_name(city_name: str) -> str:
    """Şehir ismini normalize et"""
    if pd.isna(city_name):
        return "UNKNOWN"
    
    city_upper = str(city_name).strip().upper()
    
    # Önce FIX_CITY_MAP'te kontrol et
    if city_upper in FIX_CITY_MAP:
        return FIX_CITY_MAP[city_upper]
    
    # Türkçe karakterleri normalize et
    tr_map = {
        "İ": "I", "Ğ": "G", "Ü": "U", "Ş": "S",
        "Ö": "O", "Ç": "C", "Â": "A", "Î": "I",
        "Û": "U", "ı": "I", "ğ": "G", "ü": "U",
        "ş": "S", "ö": "O", "ç": "C", "â": "A",
        "î": "I", "û": "U"
    }
    
    normalized = city_upper
    for old, new in tr_map.items():
        normalized = normalized.replace(old, new)
    
    # CITY_NORMALIZE_CLEAN'de kontrol et
    return CITY_NORMALIZE_CLEAN.get(normalized, normalized)

def get_city_coordinates(city_name: str) -> Tuple[float, float]:
    """Şehir koordinatlarını getir"""
    normalized = normalize_city_name(city_name)
    return CITY_COORDINATES.get(normalized.upper(), (35.0, 39.0))

def calculate_performance_score(row: pd.Series) -> float:
    """Performans skoru hesapla"""
    weights = {
        'PF_Satis': 0.3,
        'Pazar_Payi_%': 0.4,
        'Buyume_%': 0.3
    }
    
    score = 0
    for metric, weight in weights.items():
        if metric in row.index:
            value = row[metric] if not pd.isna(row[metric]) else 0
            score += value * weight
    
    return score / sum(weights.values()) * 100

def generate_mckinsey_insight(metric: str, value: float, benchmark: float = None, 
                             trend: float = None, context: str = "") -> Dict[str, Any]:
    """McKinsey tarzında insight oluştur"""
    insight = {
        "metric": metric,
        "value": value,
        "status": "neutral",
        "message": "",
        "recommendation": "",
        "icon": "📊"
    }
    
    # Status belirle
    if benchmark:
        if value > benchmark * 1.15:
            insight["status"] = "excellent"
            insight["icon"] = "🎯"
        elif value > benchmark * 1.05:
            insight["status"] = "good"
            insight["icon"] = "📈"
        elif value < benchmark * 0.85:
            insight["status"] = "poor"
            insight["icon"] = "⚠️"
        elif value < benchmark * 0.95:
            insight["status"] = "below"
            insight["icon"] = "📉"
    
    # Trend analizi
    if trend:
        if trend > 10:
            insight["message"] += f"Strong growth momentum (+{trend:.1f}%)"
        elif trend > 0:
            insight["message"] += f"Positive trend (+{trend:.1f}%)"
        elif trend < -10:
            insight["message"] += f"Critical decline ({trend:.1f}%)"
        elif trend < 0:
            insight["message"] += f"Declining trend ({trend:.1f}%)"
    
    # Context ekle
    if context:
        insight["message"] += f" {context}"
    
    # McKinsey tarzı recommendation
    if insight["status"] == "excellent":
        insight["recommendation"] = "Maintain investment levels and replicate success in similar markets"
    elif insight["status"] == "good":
        insight["recommendation"] = "Continue current strategy with focus on optimization"
    elif insight["status"] == "below":
        insight["recommendation"] = "Review strategy and consider tactical adjustments"
    elif insight["status"] == "poor":
        insight["recommendation"] = "Immediate intervention required - conduct deep dive analysis"
    
    return insight

def create_download_link(df: pd.DataFrame, filename: str = "data.csv") -> str:
    """Download link oluştur"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data(show_spinner="Loading data...")
def load_excel_data(file) -> pd.DataFrame:
    """Excel dosyasını yükle ve temizle"""
    try:
        # Dosyayı oku
        df = pd.read_excel(file)
        
        # Temel temizlik
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        # Tarih işlemleri
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df['YEAR'] = df['DATE'].dt.year
            df['MONTH'] = df['DATE'].dt.month
            df['QUARTER'] = df['DATE'].dt.quarter
            df['YEAR_MONTH'] = df['DATE'].dt.strftime('%Y-%m')
            df['WEEK'] = df['DATE'].dt.isocalendar().week
            
        # Şehir normalizasyonu
        if 'CITY' in df.columns:
            df['CITY_CLEAN'] = df['CITY'].apply(normalize_city_name)
        
        # Bölge normalizasyonu
        if 'REGION' in df.columns:
            df['REGION_CLEAN'] = df['REGION'].str.upper().str.strip()
        
        # Manager normalizasyonu
        if 'MANAGER' in df.columns:
            df['MANAGER_CLEAN'] = df['MANAGER'].str.upper().str.strip()
        
        # Territory normalizasyonu
        if 'TERRITORIES' in df.columns:
            df['TERRITORIES_CLEAN'] = df['TERRITORIES'].str.upper().str.strip()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Data loading error: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def load_turkey_geojson() -> Optional[gpd.GeoDataFrame]:
    """Türkiye GeoJSON dosyasını yükle"""
    try:
        # Farklı kaynaklardan deneme
        possible_paths = [
            "turkey.geojson",
            "./turkey.geojson",
            "data/turkey.geojson",
            "https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json"
        ]
        
        for path in possible_paths:
            try:
                if path.startswith("http"):
                    gdf = gpd.read_file(path)
                else:
                    gdf = gpd.read_file(path, encoding='utf-8')
                
                if 'name' in gdf.columns:
                    gdf['name_clean'] = gdf['name'].apply(normalize_city_name)
                    return gdf
                    
            except:
                continue
                
        # Fallback: Basit Türkiye geometrisi oluştur
        st.warning("⚠️ GeoJSON file not found. Using fallback geometry.")
        return create_fallback_geojson()
        
    except Exception as e:
        st.error(f"❌ GeoJSON loading error: {str(e)}")
        return create_fallback_geojson()

def create_fallback_geojson() -> gpd.GeoDataFrame:
    """Fallback GeoJSON oluştur"""
    cities = []
    geometries = []
    
    for city, (lon, lat) in CITY_COORDINATES.items():
        cities.append(city)
        geometries.append(Point(lon, lat))
    
    gdf = gpd.GeoDataFrame(
        {'city': cities, 'geometry': geometries},
        crs="EPSG:4326"
    )
    
    # Buffer ile basit polygonlar oluştur
    gdf['geometry'] = gdf['geometry'].buffer(0.5)
    gdf['name_clean'] = gdf['city'].apply(normalize_city_name)
    
    return gdf

# =============================================================================
# MACHINE LEARNING MODULES
# =============================================================================

class MckinseyMLAnalyzer:
    """McKinsey tarzında ML analiz sınıfı"""
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3),
            'Support Vector': SVR(kernel='rbf', C=100),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
        }
        
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'PF_Satis') -> pd.DataFrame:
        """ML için feature hazırla"""
        df_features = df.copy()
        
        # Lag features
        for lag in [1, 2, 3, 4, 5, 6, 12]:
            df_features[f'lag_{lag}'] = df_features[target_col].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            df_features[f'rolling_mean_{window}'] = df_features[target_col].rolling(window=window).mean()
            df_features[f'rolling_std_{window}'] = df_features[target_col].rolling(window=window).std()
            df_features[f'rolling_min_{window}'] = df_features[target_col].rolling(window=window).min()
            df_features[f'rolling_max_{window}'] = df_features[target_col].rolling(window=window).max()
        
        # Seasonal features
        if 'MONTH' in df_features.columns:
            df_features['month_sin'] = np.sin(2 * np.pi * df_features['MONTH'] / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features['MONTH'] / 12)
        
        if 'QUARTER' in df_features.columns:
            df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['QUARTER'] / 4)
            df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['QUARTER'] / 4)
        
        # Trend
        df_features['trend'] = np.arange(len(df_features))
        
        # Year-over-year growth
        df_features['yoy_growth'] = df_features[target_col].pct_change(periods=12) * 100
        
        # Momentum indicators
        df_features['momentum_3'] = df_features[target_col] / df_features[target_col].shift(3)
        df_features['momentum_6'] = df_features[target_col] / df_features[target_col].shift(6)
        
        # Volatility
        df_features['volatility_3'] = df_features[target_col].rolling(window=3).std()
        df_features['volatility_6'] = df_features[target_col].rolling(window=6).std()
        
        # Fill NaN
        df_features = df_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df_features
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Tüm ML modellerini eğit"""
        
        results = {}
        
        for name, model in self.models.items():
            try:
                # Modeli eğit
                model.fit(X_train, y_train)
                
                # Tahmin yap
                y_pred = model.predict(X_test)
                
                # Metrikleri hesapla
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2,
                    'predictions': y_pred
                }
                
            except Exception as e:
                st.warning(f"⚠️ Model {name} training failed: {str(e)}")
        
        # En iyi modeli seç
        if results:
            self.results = results
            self.best_model = min(results.items(), key=lambda x: x[1]['mape'])
            
        return results
    
    def forecast(self, model_name: str, X_future: pd.DataFrame, 
                periods: int = 12) -> pd.DataFrame:
        """Gelecek tahmini yap"""
        
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.results[model_name]['model']
        predictions = model.predict(X_future)
        
        forecast_df = pd.DataFrame({
            'Period': range(1, periods + 1),
            'Forecast': predictions,
            'Lower_95': predictions * 0.9,  # Basit confidence interval
            'Upper_95': predictions * 1.1
        })
        
        return forecast_df
    
    def feature_importance(self, model_name: str = 'Random Forest') -> pd.DataFrame:
        """Feature importance analizi"""
        
        if model_name not in self.results:
            return pd.DataFrame()
        
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = X_train.columns if 'X_train' in locals() else []
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            return importance_df.head(20)
        
        return pd.DataFrame()

# =============================================================================
# ENHANCED TURKEY MAP VISUALIZATION
# =============================================================================

class EnhancedTurkeyMap:
    """Gelişmiş Türkiye harita görselleştirme sınıfı"""
    
    def __init__(self, geojson_data: gpd.GeoDataFrame):
        self.gdf = geojson_data
        self.fig = None
    
    def create_heatmap(self, city_data: pd.DataFrame, value_column: str = 'PF_Satis',
                      title: str = "Sales Heatmap") -> go.Figure:
        """Heatmap oluştur"""
        
        # Şehir verilerini GeoJSON ile birleştir
        city_data['city_normalized'] = city_data['City'].apply(normalize_city_name)
        
        merged = self.gdf.merge(
            city_data,
            left_on='name_clean',
            right_on='city_normalized',
            how='left'
        )
        
        # NaN değerleri doldur
        merged[value_column] = merged[value_column].fillna(0)
        
        # McKinsey gradient renk skalası
        colorscale = [
            [0, MCKINSEY_COLORS["background"]],
            [0.2, MCKINSEY_COLORS["light_blue"]],
            [0.4, MCKINSEY_COLORS["teal"]],
            [0.6, MCKINSEY_COLORS["blue"]],
            [0.8, MCKINSEY_COLORS["navy"]],
            [1, MCKINSEY_COLORS["dark_gray"]]
        ]
        
        # Choropleth harita oluştur
        fig = go.Figure(go.Choroplethmapbox(
            geojson=json.loads(merged.geometry.to_json()),
            locations=merged.index,
            z=merged[value_column],
            colorscale=colorscale,
            marker_opacity=0.75,
            marker_line_width=1,
            marker_line_color='white',
            colorbar=dict(
                title=dict(
                    text="Value",
                    font=dict(
                        family='Source Sans Pro',
                        color=MCKINSEY_COLORS["navy"],
                        size=12
                    )
                ),
                tickfont=dict(
                    family='Source Sans Pro',
                    color=MCKINSEY_COLORS["dark_gray"],
                    size=10
                ),
                thickness=15,
                len=0.8
            ),
            customdata=np.stack((
                merged['name'] if 'name' in merged.columns else merged.index,
                merged[value_column],
                merged.get('Pazar_Payi_%', np.zeros(len(merged)))
            ), axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{value_column}: %{{customdata[1]:,.0f}}<br>"
                "Market Share: %{customdata[2]:.1f}%<br>"
                "<extra></extra>"
            )
        ))
        
        # Layout - McKinsey style
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox=dict(
                center=dict(lat=39.0, lon=35.0),
                zoom=4.8,
                style="light"
            ),
            height=700,
            margin=dict(l=0, r=0, t=80, b=0),
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(
                    size=20,
                    color=MCKINSEY_COLORS["navy"],
                    family='Source Sans Pro'
                ),
                y=0.95
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Source Sans Pro",
                font_color=MCKINSEY_COLORS["navy"]
            )
        )
        
        self.fig = fig
        return fig
    
    def create_bubble_map(self, city_data: pd.DataFrame, 
                         size_column: str = 'PF_Satis',
                         color_column: str = 'Pazar_Payi_%',
                         title: str = "Market Performance Map") -> go.Figure:
        """Bubble map oluştur"""
        
        # Koordinatları ekle
        city_data['coords'] = city_data['City'].apply(get_city_coordinates)
        city_data[['lon', 'lat']] = pd.DataFrame(
            city_data['coords'].tolist(), 
            index=city_data.index
        )
        
        # Bubble boyutları için scaling
        max_size = city_data[size_column].max()
        if max_size > 0:
            city_data['bubble_size'] = (city_data[size_column] / max_size * 50) + 10
        else:
            city_data['bubble_size'] = 10
        
        # Renk skalası
        colorscale = [
            [0, MCKINSEY_COLORS["red"]],
            [0.3, MCKINSEY_COLORS["orange"]],
            [0.6, MCKINSEY_COLORS["gold"]],
            [0.8, MCKINSEY_COLORS["teal"]],
            [1, MCKINSEY_COLORS["navy"]]
        ]
        
        fig = go.Figure()
        
        # Bubble layer
        fig.add_trace(go.Scattermapbox(
            lat=city_data['lat'],
            lon=city_data['lon'],
            mode='markers',
            marker=dict(
                size=city_data['bubble_size'],
                color=city_data[color_column],
                colorscale=colorscale,
                cmin=0,
                cmax=100,
                opacity=0.8,
                line=dict(width=2, color='white'),
                showscale=True,
                colorbar=dict(
                    title="Market Share %",
                    thickness=15,
                    len=0.8
                )
            ),
            text=city_data.apply(
                lambda row: f"<b>{row['City']}</b><br>"
                          f"Sales: {format_number(row[size_column])}<br>"
                          f"Share: {row[color_column]:.1f}%",
                axis=1
            ),
            hoverinfo='text'
        ))
        
        # Layout
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox=dict(
                center=dict(lat=39.0, lon=35.0),
                zoom=4.8
            ),
            height=700,
            margin=dict(l=0, r=0, t=80, b=0),
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(
                    size=20,
                    color=MCKINSEY_COLORS["navy"],
                    family='Source Sans Pro'
                )
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_region_cluster_map(self, region_data: pd.DataFrame,
                                 folium_map: bool = False):
        """Bölge bazlı cluster haritası"""
        
        if folium_map:
            # Folium haritası oluştur
            m = folium.Map(location=[39.0, 35.0], zoom_start=6,
                          tiles='CartoDB positron')
            
            # Marker cluster
            marker_cluster = MarkerCluster().add_to(m)
            
            # Her şehir için marker ekle
            for _, row in region_data.iterrows():
                coords = get_city_coordinates(row['City'])
                if coords != (35.0, 39.0):  # Default değilse
                    folium.Marker(
                        location=[coords[1], coords[0]],
                        popup=f"<b>{row['City']}</b><br>"
                             f"Sales: {format_number(row['PF_Satis'])}<br>"
                             f"Share: {row.get('Pazar_Payi_%', 0):.1f}%",
                        icon=folium.Icon(
                            color='blue',
                            icon='info-sign',
                            icon_color='white'
                        )
                    ).add_to(marker_cluster)
            
            # Heatmap layer (isteğe bağlı)
            heat_data = []
            for _, row in region_data.iterrows():
                coords = get_city_coordinates(row['City'])
                if coords != (35.0, 39.0):
                    heat_data.append([coords[1], coords[0], row['PF_Satis']])
            
            HeatMap(heat_data, radius=15, blur=10).add_to(m)
            
            return m
        
        else:
            # Plotly ile cluster haritası
            fig = go.Figure()
            
            # Her bölge için ayrı trace
            for region in region_data['Region'].unique():
                region_df = region_data[region_data['Region'] == region]
                
                # Koordinatları al
                region_df['coords'] = region_df['City'].apply(get_city_coordinates)
                region_df[['lon', 'lat']] = pd.DataFrame(
                    region_df['coords'].tolist(),
                    index=region_df.index
                )
                
                color = REGION_COLORS.get(region, MCKINSEY_COLORS["gray"])
                
                fig.add_trace(go.Scattermapbox(
                    lat=region_df['lat'],
                    lon=region_df['lon'],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=color,
                        opacity=0.8
                    ),
                    name=region,
                    text=region_df['City'],
                    hoverinfo='text'
                ))
            
            fig.update_layout(
                mapbox_style="carto-positron",
                mapbox=dict(
                    center=dict(lat=39.0, lon=35.0),
                    zoom=5
                ),
                height=600,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            return fig

# =============================================================================
# McKINSEY STYLE CHARTS
# =============================================================================

def create_mckinsey_bar_chart(data: pd.DataFrame, x_col: str, y_col: str,
                             title: str, color_col: str = None,
                             orientation: str = 'v',
                             group_col: str = None) -> go.Figure:
    """McKinsey tarzında bar chart"""
    
    if color_col:
        # Renk paleti belirle
        unique_vals = data[color_col].unique()
        color_map = {}
        for i, val in enumerate(unique_vals):
            color_map[val] = MCKINSEY_CHART_SEQUENCE[i % len(MCKINSEY_CHART_SEQUENCE)]
        
        colors = [color_map[val] for val in data[color_col]]
    else:
        colors = MCKINSEY_COLORS["navy"]
    
    if group_col and orientation == 'v':
        # Gruplandırılmış bar chart
        fig = go.Figure()
        
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group]
            color = MCKINSEY_CHART_SEQUENCE[list(data[group_col].unique()).index(group) % len(MCKINSEY_CHART_SEQUENCE)]
            
            fig.add_trace(go.Bar(
                x=group_data[x_col],
                y=group_data[y_col],
                name=str(group),
                marker_color=color,
                marker_line_color='white',
                marker_line_width=1.5,
                text=group_data[y_col].apply(lambda v: format_number(v)),
                textposition='outside',
                textfont=dict(
                    size=11,
                    family='Source Sans Pro',
                    color=MCKINSEY_COLORS["dark_gray"]
                )
            ))
        
        fig.update_layout(barmode='group')
        
    else:
        # Basit bar chart
        if orientation == 'v':
            fig = go.Figure(go.Bar(
                x=data[x_col],
                y=data[y_col],
                marker_color=colors,
                marker_line_color='white',
                marker_line_width=1.5,
                text=data[y_col].apply(lambda v: format_number(v)),
                textposition='outside',
                textfont=dict(
                    size=11,
                    family='Source Sans Pro',
                    color=MCKINSEY_COLORS["dark_gray"]
                )
            ))
        else:
            fig = go.Figure(go.Bar(
                x=data[y_col],
                y=data[x_col],
                orientation='h',
                marker_color=colors,
                marker_line_color='white',
                marker_line_width=1.5,
                text=data[y_col].apply(lambda v: format_number(v)),
                textposition='outside',
                textfont=dict(
                    size=11,
                    family='Source Sans Pro',
                    color=MCKINSEY_COLORS["dark_gray"]
                )
            ))
    
    # McKinsey layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(
                size=18,
                color=MCKINSEY_COLORS["navy"],
                family='Source Sans Pro'
            ),
            x=0.5,
            xanchor='center',
            y=0.95
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        font=dict(
            family='Source Sans Pro',
            color=MCKINSEY_COLORS["dark_gray"],
            size=12
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#E0E6EF',
            gridwidth=0.5,
            title=dict(
                text=x_col.replace('_', ' ').title(),
                font=dict(size=13, color=MCKINSEY_COLORS["navy"])
            )
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#E0E6EF',
            gridwidth=0.5,
            title=dict(
                text=y_col.replace('_', ' ').title(),
                font=dict(size=13, color=MCKINSEY_COLORS["navy"])
            )
        ),
        margin=dict(l=80, r=40, t=100, b=80),
        showlegend=group_col is not None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='white',
            bordercolor='#E0E6EF',
            borderwidth=1,
            font=dict(size=11)
        )
    )
    
    return fig

def create_mckinsey_line_chart(data: pd.DataFrame, x_col: str,
                              y_cols: List[str], title: str,
                              y_names: List[str] = None) -> go.Figure:
    """McKinsey tarzında line chart"""
    
    fig = go.Figure()
    
    if y_names is None:
        y_names = y_cols
    
    for idx, (y_col, y_name) in enumerate(zip(y_cols, y_names)):
        color = MCKINSEY_CHART_SEQUENCE[idx % len(MCKINSEY_CHART_SEQUENCE)]
        
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode='lines+markers',
            name=y_name,
            line=dict(color=color, width=3),
            marker=dict(
                size=8,
                color='white',
                line=dict(width=2, color=color)
            ),
            hovertemplate=(
                f"<b>{y_name}</b><br>"
                f"{x_col}: %{{x}}<br>"
                f"Value: %{{y:,.0f}}<br>"
                "<extra></extra>"
            )
        ))
    
    # McKinsey layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(
                size=18,
                color=MCKINSEY_COLORS["navy"],
                family='Source Sans Pro'
            ),
            x=0.5,
            xanchor='center',
            y=0.95
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        font=dict(
            family='Source Sans Pro',
            color=MCKINSEY_COLORS["dark_gray"],
            size=12
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#E0E6EF',
            gridwidth=0.5,
            title=dict(
                text=x_col.replace('_', ' ').title(),
                font=dict(size=13, color=MCKINSEY_COLORS["navy"])
            )
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#E0E6EF',
            gridwidth=0.5,
            title=dict(
                text="Value",
                font=dict(size=13, color=MCKINSEY_COLORS["navy"])
            )
        ),
        margin=dict(l=80, r=40, t=100, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='white',
            bordercolor='#E0E6EF',
            borderwidth=1,
            font=dict(size=11)
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='Source Sans Pro'
        )
    )
    
    return fig

def create_mckinsey_scatter_plot(data: pd.DataFrame, x_col: str, y_col: str,
                                size_col: str = None, color_col: str = None,
                                title: str = "") -> go.Figure:
    """McKinsey tarzında scatter plot"""
    
    if size_col:
        # Bubble boyutları için scaling
        max_size = data[size_col].max()
        if max_size > 0:
            sizes = (data[size_col] / max_size * 40) + 5
        else:
            sizes = 10
    else:
        sizes = 10
    
    if color_col:
        # Renk skalası belirle
        if color_col == 'BCG_Kategori':
            color_map = BCG_COLORS
            colors = data[color_col].map(color_map)
            color_discrete_map = color_map
        elif color_col == 'Region':
            color_map = REGION_COLORS
            colors = data[color_col].map(color_map)
            color_discrete_map = color_map
        else:
            # Continuous color scale
            colors = data[color_col]
            color_discrete_map = None
    else:
        colors = MCKINSEY_COLORS["navy"]
        color_discrete_map = None
    
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col if color_col else None,
        color_discrete_map=color_discrete_map,
        hover_name=data.columns[0] if len(data.columns) > 0 else None,
        size_max=40,
        opacity=0.8,
        custom_data=[data.index] if len(data) > 0 else None
    )
    
    # McKinsey styling
    fig.update_traces(
        marker=dict(
            line=dict(width=1.5, color='white')
        ),
        hovertemplate=(
            f"<b>%{{hovertext}}</b><br>"
            f"{x_col}: %{{x:,.0f}}<br>"
            f"{y_col}: %{{y:.1f}}%<br>"
            f"{size_col if size_col else 'Size'}: %{{marker.size:,.0f}}<br>"
            "<extra></extra>"
        )
    )
    
    # McKinsey layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(
                size=18,
                color=MCKINSEY_COLORS["navy"],
                family='Source Sans Pro'
            ),
            x=0.5,
            xanchor='center',
            y=0.95
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=550,
        font=dict(
            family='Source Sans Pro',
            color=MCKINSEY_COLORS["dark_gray"],
            size=12
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#E0E6EF',
            gridwidth=0.5,
            title=dict(
                text=x_col.replace('_', ' ').title(),
                font=dict(size=13, color=MCKINSEY_COLORS["navy"])
            )
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#E0E6EF',
            gridwidth=0.5,
            title=dict(
                text=y_col.replace('_', ' ').title(),
                font=dict(size=13, color=MCKINSEY_COLORS["navy"])
            )
        ),
        margin=dict(l=80, r=40, t=100, b=80),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='white',
            bordercolor='#E0E6EF',
            borderwidth=1,
            font=dict(size=11)
        )
    )
    
    return fig

def create_mckinsey_heatmap(data: pd.DataFrame, x_col: str, y_col: str,
                           values_col: str, title: str = "") -> go.Figure:
    """McKinsey tarzında heatmap"""
    
    # Pivot table
    pivot = data.pivot_table(
        index=y_col,
        columns=x_col,
        values=values_col,
        aggfunc='sum'
    ).fillna(0)
    
    # McKinsey colorscale
    colorscale = [
        [0, MCKINSEY_COLORS["background"]],
        [0.2, MCKINSEY_COLORS["light_blue"]],
        [0.4, MCKINSEY_COLORS["teal"]],
        [0.6, MCKINSEY_COLORS["blue"]],
        [0.8, MCKINSEY_COLORS["navy"]],
        [1, MCKINSEY_COLORS["dark_gray"]]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=colorscale,
        hoverongaps=False,
        hovertemplate=(
            f"{x_col}: %{{x}}<br>"
            f"{y_col}: %{{y}}<br>"
            f"{values_col}: %{{z:,.0f}}<br>"
            "<extra></extra>"
        )
    ))
    
    # McKinsey layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(
                size=18,
                color=MCKINSEY_COLORS["navy"],
                family='Source Sans Pro'
            ),
            x=0.5,
            xanchor='center',
            y=0.95
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        font=dict(
            family='Source Sans Pro',
            color=MCKINSEY_COLORS["dark_gray"],
            size=12
        ),
        xaxis=dict(
            title=dict(
                text=x_col.replace('_', ' ').title(),
                font=dict(size=13, color=MCKINSEY_COLORS["navy"])
            )
        ),
        yaxis=dict(
            title=dict(
                text=y_col.replace('_', ' ').title(),
                font=dict(size=13, color=MCKINSEY_COLORS["navy"])
            )
        ),
        margin=dict(l=80, r=40, t=100, b=80)
    )
    
    return fig

def create_mckinsey_radar_chart(data: pd.DataFrame, categories: List[str],
                               title: str = "Performance Radar") -> go.Figure:
    """McKinsey tarzında radar chart"""
    
    fig = go.Figure()
    
    for idx, (_, row) in enumerate(data.iterrows()):
        color = MCKINSEY_CHART_SEQUENCE[idx % len(MCKINSEY_CHART_SEQUENCE)]
        
        fig.add_trace(go.Scatterpolar(
            r=row[categories].values,
            theta=categories,
            fill='toself',
            name=row['name'] if 'name' in row.index else f'Category {idx}',
            line=dict(color=color, width=2),
            fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
            hovertemplate=(
                "<b>%{theta}</b><br>"
                "Value: %{r:.1f}<br>"
                "<extra></extra>"
            )
        ))
    
    # McKinsey layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(
                size=18,
                color=MCKINSEY_COLORS["navy"],
                family='Source Sans Pro'
            ),
            x=0.5,
            xanchor='center',
            y=0.95
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, data[categories].max().max() * 1.1],
                gridcolor='#E0E6EF',
                linecolor='#E0E6EF'
            ),
            angularaxis=dict(
                gridcolor='#E0E6EF',
                linecolor='#E0E6EF'
            ),
            bgcolor='white'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        font=dict(
            family='Source Sans Pro',
            color=MCKINSEY_COLORS["dark_gray"],
            size=12
        ),
        margin=dict(l=80, r=40, t=100, b=80),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def calculate_city_performance(df: pd.DataFrame, product: str,
                              date_filter: Tuple = None) -> pd.DataFrame:
    """Şehir bazlı performans analizi"""
    
    cols = get_product_columns(product)
    
    # Date filter
    if date_filter:
        start_date, end_date = date_filter
        df_filtered = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
    else:
        df_filtered = df.copy()
    
    # Gruplama
    city_perf = df_filtered.groupby(['CITY_CLEAN', 'REGION_CLEAN']).agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index()
    
    # Kolon isimlendirme
    city_perf.columns = ['City', 'Region', 'PF_Satis', 'Rakip_Satis']
    
    # Metrikler
    city_perf['Toplam_Pazar'] = city_perf['PF_Satis'] + city_perf['Rakip_Satis']
    city_perf['Pazar_Payi_%'] = safe_divide(city_perf['PF_Satis'], city_perf['Toplam_Pazar']) * 100
    city_perf['Goreceli_Pay'] = safe_divide(city_perf['PF_Satis'], city_perf['Rakip_Satis'])
    
    # Performans skoru
    city_perf['Performans_Skoru'] = city_perf.apply(calculate_performance_score, axis=1)
    
    # Segmentasyon
    city_perf['Segment'] = pd.qcut(city_perf['PF_Satis'], q=4,
                                  labels=['Düşük', 'Orta-Düşük', 'Orta-Yüksek', 'Yüksek'])
    
    return city_perf.sort_values('PF_Satis', ascending=False)

def calculate_territory_performance(df: pd.DataFrame, product: str,
                                   date_filter: Tuple = None) -> pd.DataFrame:
    """Territory bazlı performans analizi"""
    
    cols = get_product_columns(product)
    
    # Date filter
    if date_filter:
        start_date, end_date = date_filter
        df_filtered = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
    else:
        df_filtered = df.copy()
    
    # Gruplama
    terr_perf = df_filtered.groupby(['TERRITORIES_CLEAN', 'REGION_CLEAN',
                                    'CITY_CLEAN', 'MANAGER_CLEAN']).agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index()
    
    # Kolon isimlendirme
    terr_perf.columns = ['Territory', 'Region', 'City', 'Manager',
                        'PF_Satis', 'Rakip_Satis']
    
    # Metrikler
    terr_perf['Toplam_Pazar'] = terr_perf['PF_Satis'] + terr_perf['Rakip_Satis']
    terr_perf['Pazar_Payi_%'] = safe_divide(terr_perf['PF_Satis'], terr_perf['Toplam_Pazar']) * 100
    
    total_pf = terr_perf['PF_Satis'].sum()
    terr_perf['Agirlik_%'] = safe_divide(terr_perf['PF_Satis'], total_pf) * 100
    
    # Performans skoru
    terr_perf['Performans_Skoru'] = terr_perf.apply(calculate_performance_score, axis=1)
    
    return terr_perf.sort_values('PF_Satis', ascending=False)

def calculate_time_series_analysis(df: pd.DataFrame, product: str,
                                 group_by: str = 'YEAR_MONTH',
                                 date_filter: Tuple = None) -> pd.DataFrame:
    """Zaman serisi analizi"""
    
    cols = get_product_columns(product)
    
    # Date filter
    if date_filter:
        start_date, end_date = date_filter
        df_filtered = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
    else:
        df_filtered = df.copy()
    
    # Gruplama
    ts_data = df_filtered.groupby(group_by).agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum',
        'DATE': 'first'
    }).reset_index().sort_values(group_by)
    
    # Kolon isimlendirme
    ts_data.columns = [group_by, 'PF_Satis', 'Rakip_Satis', 'DATE']
    
    # Metrikler
    ts_data['Toplam_Pazar'] = ts_data['PF_Satis'] + ts_data['Rakip_Satis']
    ts_data['Pazar_Payi_%'] = safe_divide(ts_data['PF_Satis'], ts_data['Toplam_Pazar']) * 100
    
    # Büyüme metrikleri
    ts_data['PF_Buyume_%'] = ts_data['PF_Satis'].pct_change() * 100
    ts_data['Rakip_Buyume_%'] = ts_data['Rakip_Satis'].pct_change() * 100
    ts_data['Pazar_Buyume_%'] = ts_data['Toplam_Pazar'].pct_change() * 100
    
    # Trend analizi
    ts_data['PF_Trend'] = ts_data['PF_Satis'].rolling(window=3).mean()
    ts_data['Pay_Trend'] = ts_data['Pazar_Payi_%'].rolling(window=3).mean()
    
    return ts_data

def calculate_competitor_analysis(df: pd.DataFrame, product: str,
                                 date_filter: Tuple = None) -> pd.DataFrame:
    """Rakip analizi"""
    
    cols = get_product_columns(product)
    
    # Date filter
    if date_filter:
        start_date, end_date = date_filter
        df_filtered = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
    else:
        df_filtered = df.copy()
    
    # Aylık analiz
    monthly_comp = df_filtered.groupby('YEAR_MONTH').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index().sort_values('YEAR_MONTH')
    
    # Kolon isimlendirme
    monthly_comp.columns = ['Period', 'PF', 'Competitor']
    
    # Metrikler
    monthly_comp['Total_Market'] = monthly_comp['PF'] + monthly_comp['Competitor']
    monthly_comp['PF_Share_%'] = safe_divide(monthly_comp['PF'], monthly_comp['Total_Market']) * 100
    monthly_comp['Comp_Share_%'] = 100 - monthly_comp['PF_Share_%']
    
    # Büyüme
    monthly_comp['PF_Growth_%'] = monthly_comp['PF'].pct_change() * 100
    monthly_comp['Comp_Growth_%'] = monthly_comp['Competitor'].pct_change() * 100
    
    # Pazar büyümesi
    monthly_comp['Market_Growth_%'] = monthly_comp['Total_Market'].pct_change() * 100
    
    return monthly_comp

def calculate_bcg_matrix(df: pd.DataFrame, product: str,
                        date_filter: Tuple = None) -> pd.DataFrame:
    """BCG Matrix analizi"""
    
    cols = get_product_columns(product)
    
    # Date filter
    if date_filter:
        start_date, end_date = date_filter
        df_filtered = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
    else:
        df_filtered = df.copy()
    
    # Territory bazlı BCG
    terr_bcg = df_filtered.groupby('TERRITORIES_CLEAN').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index()
    
    # Büyüme oranı (son 6 ay vs ilk 6 ay)
    if len(df_filtered) > 6:
        df_sorted = df_filtered.sort_values('DATE')
        mid_idx = len(df_sorted) // 2
        
        first_half = df_sorted.iloc[:mid_idx].groupby('TERRITORIES_CLEAN')[cols['pf']].sum()
        second_half = df_sorted.iloc[mid_idx:].groupby('TERRITORIES_CLEAN')[cols['pf']].sum()
        
        growth_rate = {}
        for terr in first_half.index:
            if terr in second_half.index and first_half[terr] > 0:
                growth_rate[terr] = ((second_half[terr] - first_half[terr]) / first_half[terr]) * 100
            else:
                growth_rate[terr] = 0
        
        terr_bcg['Growth_Rate_%'] = terr_bcg['TERRITORIES_CLEAN'].map(growth_rate).fillna(0)
    
    # Market share
    terr_bcg['Total_Market'] = terr_bcg[cols['pf']] + terr_bcg[cols['rakip']]
    terr_bcg['Market_Share_%'] = safe_divide(terr_bcg[cols['pf']], terr_bcg['Total_Market']) * 100
    
    # BCG kategorisi
    median_growth = terr_bcg['Growth_Rate_%'].median() if 'Growth_Rate_%' in terr_bcg.columns else 0
    median_share = terr_bcg['Market_Share_%'].median()
    
    def assign_bcg(row):
        if row['Market_Share_%'] >= median_share and row['Growth_Rate_%'] >= median_growth:
            return "⭐ Star"
        elif row['Market_Share_%'] >= median_share and row['Growth_Rate_%'] < median_growth:
            return "🐄 Cash Cow"
        elif row['Market_Share_%'] < median_share and row['Growth_Rate_%'] >= median_growth:
            return "❓ Question Mark"
        else:
            return "🐶 Dog"
    
    terr_bcg['BCG_Category'] = terr_bcg.apply(assign_bcg, axis=1)
    
    # Investment priority
    def assign_priority(row):
        if row['BCG_Category'] == "⭐ Star":
            return "🚀 Aggressive Investment"
        elif row['BCG_Category'] == "🐄 Cash Cow":
            return "🛡️ Maintenance"
        elif row['BCG_Category'] == "❓ Question Mark":
            return "💎 High Potential"
        else:
            return "👁️ Monitor"
    
    terr_bcg['Investment_Priority'] = terr_bcg.apply(assign_priority, axis=1)
    
    # Rename columns
    terr_bcg.columns = ['Territory', 'PF_Sales', 'Competitor_Sales', 'Growth_Rate_%',
                       'Total_Market', 'Market_Share_%', 'BCG_Category', 'Investment_Priority']
    
    return terr_bcg

def calculate_seasonality_analysis(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """Sezonallik analizi"""
    
    cols = get_product_columns(product)
    
    # Aylık ortalama
    monthly_avg = df.groupby('MONTH').agg({
        cols['pf']: 'mean',
        cols['rakip']: 'mean'
    }).reset_index()
    
    # Yüzdelik değişim
    monthly_avg['PF_Index'] = (monthly_avg[cols['pf']] / monthly_avg[cols['pf']].mean()) * 100
    monthly_avg['Comp_Index'] = (monthly_avg[cols['rakip']] / monthly_avg[cols['rakip']].mean()) * 100
    
    # Seasonality score
    monthly_avg['Seasonality_Score'] = monthly_avg['PF_Index'].std()
    
    return monthly_avg

def calculate_cluster_analysis(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Kümeleme analizi"""
    
    # Özellikler
    features = ['PF_Satis', 'Pazar_Payi_%', 'Growth_Rate_%']
    cluster_data = df[features].fillna(0)
    
    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # Cluster özellikleri
    cluster_stats = df.groupby('Cluster')[features].agg(['mean', 'std', 'count']).round(2)
    
    return df, cluster_stats

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # McKinsey Header
    st.markdown("""
    <div class="mckinsey-header">
        <h1 class="mckinsey-title">McKinsey & Company</h1>
        <div class="mckinsey-subtitle">Portfolio Analytics Dashboard | Commercial Excellence</div>
        <div style="margin-top: 1rem;">
            <span class="mckinsey-badge">v2.0</span>
            <span class="mckinsey-badge">ENHANCED ML</span>
            <span class="mckinsey-badge">REAL-TIME</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("### 📁 DATA UPLOAD")
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Upload Excel Data File",
            type=['xlsx', 'xls', 'csv'],
            help="Upload your sales data file"
        )
        
        if not uploaded_file:
            st.info("👈 Please upload an Excel file to begin analysis")
            
            # Sample data for demo
            if st.button("Use Sample Data (Demo)"):
                # Create sample data
                np.random.seed(42)
                dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
                cities = list(CITY_COORDINATES.keys())[:20]
                
                sample_data = []
                for date in dates:
                    for city in cities:
                        sample_data.append({
                            'DATE': date,
                            'CITY': city,
                            'TERRITORIES': f"TERR_{np.random.randint(1, 10)}",
                            'REGION': np.random.choice(['MARMARA', 'EGE', 'AKDENİZ', 'İÇ ANADOLU']),
                            'MANAGER': f"MGR_{np.random.randint(1, 5)}",
                            'TROCMETAM': np.random.randint(100, 1000),
                            'DIGER TROCMETAM': np.random.randint(100, 800),
                            'CORTIPOL': np.random.randint(50, 500),
                            'DIGER CORTIPOL': np.random.randint(50, 400)
                        })
                
                df = pd.DataFrame(sample_data)
                st.success("✅ Sample data loaded successfully!")
            else:
                st.stop()
        else:
            # Load uploaded data
            with st.spinner("Loading and processing data..."):
                df = load_excel_data(uploaded_file)
                
            if df.empty:
                st.error("❌ Failed to load data. Please check file format.")
                st.stop()
            
            st.success(f"✅ **{len(df):,}** rows loaded successfully")
        
        st.markdown("---")
        st.markdown("### 💊 PRODUCT SELECTION")
        
        product_options = ["TROCMETAM", "CORTIPOL", "DEKSAMETAZON", "PF IZOTONIK"]
        selected_product = st.selectbox(
            "Select Product",
            product_options,
            help="Choose the product for analysis"
        )
        
        st.markdown("---")
        st.markdown("### 📅 TIME PERIOD")
        
        if 'DATE' in df.columns:
            min_date = df['DATE'].min()
            max_date = df['DATE'].max()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    min_date,
                    min_value=min_date,
                    max_value=max_date
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    max_date,
                    min_value=min_date,
                    max_value=max_date
                )
            
            date_filter = (pd.to_datetime(start_date), pd.to_datetime(end_date))
        else:
            date_filter = None
            st.warning("No date column found in data")
        
        st.markdown("---")
        st.markdown("### 🔍 FILTERS")
        
        # Territory filter
        if 'TERRITORIES_CLEAN' in df.columns:
            territories = ["ALL"] + sorted(df['TERRITORIES_CLEAN'].unique())
            selected_territory = st.selectbox("Territory", territories)
        else:
            selected_territory = "ALL"
        
        # Region filter
        if 'REGION_CLEAN' in df.columns:
            regions = ["ALL"] + sorted(df['REGION_CLEAN'].unique())
            selected_region = st.selectbox("Region", regions)
        else:
            selected_region = "ALL"
        
        # Manager filter
        if 'MANAGER_CLEAN' in df.columns:
            managers = ["ALL"] + sorted(df['MANAGER_CLEAN'].unique())
            selected_manager = st.selectbox("Manager", managers)
        else:
            selected_manager = "ALL"
        
        # Apply filters
        df_filtered = df.copy()
        if selected_territory != "ALL":
            df_filtered = df_filtered[df_filtered['TERRITORIES_CLEAN'] == selected_territory]
        if selected_region != "ALL":
            df_filtered = df_filtered[df_filtered['REGION_CLEAN'] == selected_region]
        if selected_manager != "ALL":
            df_filtered = df_filtered[df_filtered['MANAGER_CLEAN'] == selected_manager]
        
        st.markdown("---")
        st.markdown("### 🗺️ MAP SETTINGS")
        
        map_type = st.radio(
            "Map Type",
            ["Heatmap", "Bubble Map", "Cluster Map"],
            horizontal=True
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ ADVANCED SETTINGS")
        
        # ML settings
        ml_enabled = st.checkbox("Enable Machine Learning", value=True)
        forecast_periods = st.slider("Forecast Periods", 1, 12, 6)
        
        # Analysis depth
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Basic", "Standard", "Advanced", "Expert"],
            value="Standard"
        )
        
    # MAIN CONTENT - TABS
    tab_titles = [
        "📊 Executive Dashboard",
        "🗺️ Geographic Intelligence",
        "🏢 Portfolio Performance",
        "📈 Time Series & Forecasting",
        "🤖 Advanced ML Analytics",
        "⭐ Strategic Insights",
        "📋 Detailed Reports"
    ]
    
    tabs = st.tabs(tab_titles)
    
    # =========================================================================
    # TAB 1: EXECUTIVE DASHBOARD
    # =========================================================================
    with tabs[0]:
        st.markdown("## Executive Dashboard")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pf = df_filtered[get_product_columns(selected_product)['pf']].sum()
            st.metric(
                "Total PF Sales",
                format_currency(total_pf),
                delta=f"{format_number(total_pf)} units"
            )
        
        with col2:
            total_comp = df_filtered[get_product_columns(selected_product)['rakip']].sum()
            total_market = total_pf + total_comp
            market_share = (total_pf / total_market * 100) if total_market > 0 else 0
            st.metric(
                "Market Share",
                f"{market_share:.1f}%",
                delta=f"{100 - market_share:.1f}% competitor"
            )
        
        with col3:
            active_cities = df_filtered['CITY_CLEAN'].nunique()
            st.metric(
                "Active Cities",
                active_cities,
                delta=f"{df_filtered['TERRITORIES_CLEAN'].nunique()} territories"
            )
        
        with col4:
            if 'DATE' in df_filtered.columns:
                avg_daily = total_pf / df_filtered['DATE'].nunique()
                st.metric(
                    "Avg Daily Sales",
                    format_currency(avg_daily),
                    delta=f"{format_number(avg_daily)}/day"
                )
            else:
                st.metric("Avg Monthly", format_currency(total_pf))
        
        st.markdown("---")
        
        # McKinsey Insight Box
        insight = generate_mckinsey_insight(
            "Market Performance",
            market_share,
            benchmark=50,
            trend=market_share - 50 if market_share else None,
            context="Strong positioning in key markets"
        )
        
        st.markdown(f"""
        <div class="mckinsey-insight">
            <h3>{insight['icon']} Key Strategic Insight</h3>
            <p><strong>{insight['metric']}:</strong> {insight['value']:.1f}% market share</p>
            <p>{insight['message']}</p>
            <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Performance Overview Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Top 10 Cities
            city_perf = calculate_city_performance(df_filtered, selected_product, date_filter)
            top_cities = city_perf.head(10)
            
            fig = create_mckinsey_bar_chart(
                top_cities,
                'City',
                'PF_Satis',
                'Top 10 Cities by Sales',
                color_col='Region',
                orientation='h'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            # Monthly Trend
            monthly_data = calculate_time_series_analysis(df_filtered, selected_product, 'YEAR_MONTH', date_filter)
            
            fig = create_mckinsey_line_chart(
                monthly_data,
                'YEAR_MONTH',
                ['PF_Satis', 'Rakip_Satis'],
                'Monthly Sales Trend',
                ['PF Sales', 'Competitor']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional Metrics
        st.markdown("---")
        st.markdown("### 📈 Performance Metrics")
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            # Growth rate
            if len(monthly_data) > 1:
                growth_rate = monthly_data['PF_Buyume_%'].iloc[-1]
                st.metric(
                    "Monthly Growth",
                    f"{growth_rate:+.1f}%",
                    delta="vs previous month"
                )
        
        with col_metric2:
            # Market penetration
            total_cities_all = df['CITY_CLEAN'].nunique()
            penetration_rate = (active_cities / total_cities_all * 100) if total_cities_all > 0 else 0
            st.metric(
                "Market Penetration",
                f"{penetration_rate:.1f}%",
                delta=f"{active_cities}/{total_cities_all} cities"
            )
        
        with col_metric3:
            # Average deal size
            avg_deal_size = total_pf / len(df_filtered) if len(df_filtered) > 0 else 0
            st.metric(
                "Avg Deal Size",
                format_currency(avg_deal_size),
                delta="per transaction"
            )
    
    # =========================================================================
    # TAB 2: GEOGRAPHIC INTELLIGENCE
    # =========================================================================
    with tabs[1]:
        st.markdown("## Geographic Intelligence")
        
        # Load GeoJSON
        with st.spinner("Loading geographic data..."):
            gdf = load_turkey_geojson()
        
        if gdf is not None:
            # Calculate city performance
            city_perf = calculate_city_performance(df_filtered, selected_product, date_filter)
            
            # Map selection
            col_map1, col_map2 = st.columns([3, 1])
            
            with col_map2:
                map_value_col = st.selectbox(
                    "Value Metric",
                    ['PF_Satis', 'Pazar_Payi_%', 'Performans_Skoru'],
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                
                map_color_scheme = st.selectbox(
                    "Color Scheme",
                    ['McKinsey Gradient', 'Region Colors', 'Performance Colors']
                )
            
            with col_map1:
                # Create enhanced map
                map_creator = EnhancedTurkeyMap(gdf)
                
                if map_type == "Heatmap":
                    fig = map_creator.create_heatmap(
                        city_perf,
                        map_value_col,
                        f"{selected_product} - {map_value_col.replace('_', ' ')}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif map_type == "Bubble Map":
                    fig = map_creator.create_bubble_map(
                        city_perf,
                        'PF_Satis',
                        'Pazar_Payi_%',
                        f"{selected_product} - Market Performance"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif map_type == "Cluster Map":
                    fig = map_creator.create_region_cluster_map(city_perf)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Regional Analysis
            st.markdown("---")
            st.markdown("### 🗺️ Regional Performance")
            
            region_stats = city_perf.groupby('Region').agg({
                'PF_Satis': ['sum', 'mean', 'count'],
                'Pazar_Payi_%': 'mean',
                'Performans_Skoru': 'mean'
            }).round(2)
            
            # Flatten column names
            region_stats.columns = ['_'.join(col).strip() for col in region_stats.columns.values]
            region_stats = region_stats.reset_index()
            
            # Display regional metrics
            col_reg1, col_reg2, col_reg3, col_reg4 = st.columns(4)
            
            with col_reg1:
                top_region = region_stats.loc[region_stats['PF_Satis_sum'].idxmax(), 'Region']
                st.metric("Top Region", top_region)
            
            with col_reg2:
                avg_region_share = region_stats['Pazar_Payi_%_mean'].mean()
                st.metric("Avg Region Share", f"{avg_region_share:.1f}%")
            
            with col_reg3:
                region_coverage = len(region_stats)
                st.metric("Regions Covered", region_coverage)
            
            with col_reg4:
                best_perf = region_stats['Performans_Skoru_mean'].max()
                st.metric("Best Performance", f"{best_perf:.0f}")
            
            # Regional charts
            col_reg_chart1, col_reg_chart2 = st.columns(2)
            
            with col_reg_chart1:
                fig = create_mckinsey_bar_chart(
                    region_stats.nlargest(10, 'PF_Satis_sum'),
                    'Region',
                    'PF_Satis_sum',
                    'Sales by Region',
                    orientation='v'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_reg_chart2:
                fig = create_mckinsey_bar_chart(
                    region_stats.nlargest(10, 'Pazar_Payi_%_mean'),
                    'Region',
                    'Pazar_Payi_%_mean',
                    'Market Share by Region',
                    orientation='v'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("❌ Could not load geographic data. Please check GeoJSON file.")
    
    # =========================================================================
    # TAB 3: PORTFOLIO PERFORMANCE
    # =========================================================================
    with tabs[2]:
        st.markdown("## Portfolio Performance Analysis")
        
        # Territory Performance
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        
        # Performance Metrics
        col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
        
        with col_perf1:
            top_terr = terr_perf.iloc[0]['Territory'] if len(terr_perf) > 0 else "N/A"
            st.metric("Top Territory", top_terr)
        
        with col_perf2:
            avg_share = terr_perf['Pazar_Payi_%'].mean()
            st.metric("Avg Territory Share", f"{avg_share:.1f}%")
        
        with col_perf3:
            territories_count = len(terr_perf)
            st.metric("Active Territories", territories_count)
        
        with col_perf4:
            perf_variance = terr_perf['Pazar_Payi_%'].std()
            st.metric("Performance Variance", f"{perf_variance:.1f}")
        
        st.markdown("---")
        
        # Territory Analysis Charts
        col_terr1, col_terr2 = st.columns(2)
        
        with col_terr1:
            # Top 20 Territories
            top_territories = terr_perf.head(20)
            
            fig = create_mckinsey_bar_chart(
                top_territories,
                'Territory',
                'PF_Satis',
                'Top 20 Territories',
                color_col='Region',
                orientation='v'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_terr2:
            # Performance Distribution
            fig = create_mckinsey_scatter_plot(
                terr_perf.head(50),
                'PF_Satis',
                'Pazar_Payi_%',
                'PF_Satis',
                'Region',
                'Territory Performance Matrix'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # BCG Matrix Analysis
        st.markdown("---")
        st.markdown("### 🎯 BCG Matrix Analysis")
        
        bcg_matrix = calculate_bcg_matrix(df_filtered, selected_product, date_filter)
        
        # BCG Distribution
        col_bcg1, col_bcg2, col_bcg3, col_bcg4 = st.columns(4)
        
        with col_bcg1:
            stars = len(bcg_matrix[bcg_matrix['BCG_Category'] == '⭐ Star'])
            st.metric("⭐ Stars", stars)
        
        with col_bcg2:
            cash_cows = len(bcg_matrix[bcg_matrix['BCG_Category'] == '🐄 Cash Cow'])
            st.metric("🐄 Cash Cows", cash_cows)
        
        with col_bcg3:
            questions = len(bcg_matrix[bcg_matrix['BCG_Category'] == '❓ Question Mark'])
            st.metric("❓ Question Marks", questions)
        
        with col_bcg4:
            dogs = len(bcg_matrix[bcg_matrix['BCG_Category'] == '🐶 Dog'])
            st.metric("🐶 Dogs", dogs)
        
        # BCG Scatter Plot
        fig = create_mckinsey_scatter_plot(
            bcg_matrix,
            'Market_Share_%',
            'Growth_Rate_%',
            'PF_Sales',
            'BCG_Category',
            'BCG Matrix Analysis'
        )
        
        # Add quadrant lines
        median_share = bcg_matrix['Market_Share_%'].median()
        median_growth = bcg_matrix['Growth_Rate_%'].median()
        
        fig.add_hline(
            y=median_growth,
            line_dash="dash",
            line_color=MCKINSEY_COLORS["gray"],
            opacity=0.5
        )
        fig.add_vline(
            x=median_share,
            line_dash="dash",
            line_color=MCKINSEY_COLORS["gray"],
            opacity=0.5
        )
        
        # Add quadrant labels
        fig.add_annotation(
            x=median_share * 0.5,
            y=median_growth * 1.5,
            text="⭐ Stars",
            showarrow=False,
            font=dict(color=MCKINSEY_COLORS["green"], size=12)
        )
        fig.add_annotation(
            x=median_share * 1.5,
            y=median_growth * 0.5,
            text="🐄 Cash Cows",
            showarrow=False,
            font=dict(color=MCKINSEY_COLORS["navy"], size=12)
        )
        fig.add_annotation(
            x=median_share * 0.5,
            y=median_growth * 0.5,
            text="❓ Question Marks",
            showarrow=False,
            font=dict(color=MCKINSEY_COLORS["orange"], size=12)
        )
        fig.add_annotation(
            x=median_share * 1.5,
            y=median_growth * 1.5,
            text="🐶 Dogs",
            showarrow=False,
            font=dict(color=MCKINSEY_COLORS["red"], size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # BCG Detailed Table
        st.markdown("#### 📋 BCG Analysis Details")
        st.dataframe(
            bcg_matrix.sort_values('PF_Sales', ascending=False),
            use_container_width=True,
            height=400
        )
    
    # =========================================================================
    # TAB 4: TIME SERIES & FORECASTING
    # =========================================================================
    with tabs[3]:
        st.markdown("## Time Series Analysis & Forecasting")
        
        # Time Series Data
        ts_data = calculate_time_series_analysis(df_filtered, selected_product, 'YEAR_MONTH', date_filter)
        
        if len(ts_data) > 1:
            # Time Series Metrics
            col_ts1, col_ts2, col_ts3, col_ts4 = st.columns(4)
            
            with col_ts1:
                avg_growth = ts_data['PF_Buyume_%'].mean()
                st.metric(
                    "Avg Monthly Growth",
                    f"{avg_growth:+.1f}%",
                    delta="month-over-month"
                )
            
            with col_ts2:
                volatility = ts_data['PF_Buyume_%'].std()
                st.metric(
                    "Growth Volatility",
                    f"{volatility:.1f}",
                    delta="standard deviation"
                )
            
            with col_ts3:
                trend_slope = linregress(range(len(ts_data)), ts_data['PF_Satis']).slope
                st.metric(
                    "Trend Slope",
                    f"{trend_slope:.0f}",
                    delta="units/month"
                )
            
            with col_ts4:
                peak_month = ts_data.loc[ts_data['PF_Satis'].idxmax(), 'YEAR_MONTH']
                st.metric(
                    "Peak Month",
                    peak_month,
                    delta="highest sales"
                )
            
            st.markdown("---")
            
            # Time Series Charts
            col_ts_chart1, col_ts_chart2 = st.columns(2)
            
            with col_ts_chart1:
                # Sales Trend
                fig = create_mckinsey_line_chart(
                    ts_data,
                    'YEAR_MONTH',
                    ['PF_Satis', 'PF_Trend'],
                    'Sales Trend with Moving Average',
                    ['Actual', '3-Month Avg']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_ts_chart2:
                # Market Share Trend
                fig = create_mckinsey_line_chart(
                    ts_data,
                    'YEAR_MONTH',
                    ['Pazar_Payi_%', 'Pay_Trend'],
                    'Market Share Evolution',
                    ['Actual', '3-Month Avg']
                )
                fig.add_hline(
                    y=50,
                    line_dash="dash",
                    line_color=MCKINSEY_COLORS["red"],
                    opacity=0.3,
                    annotation_text="50% Target"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Growth Analysis
            st.markdown("---")
            st.markdown("### 📈 Growth Analysis")
            
            col_growth1, col_growth2 = st.columns(2)
            
            with col_growth1:
                # Growth Rate Chart
                fig = create_mckinsey_line_chart(
                    ts_data,
                    'YEAR_MONTH',
                    ['PF_Buyume_%', 'Rakip_Buyume_%'],
                    'Growth Rate Comparison',
                    ['PF Growth', 'Competitor Growth']
                )
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color=MCKINSEY_COLORS["gray"],
                    opacity=0.5
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_growth2:
                # Seasonality Analysis
                seasonality = calculate_seasonality_analysis(df_filtered, selected_product)
                
                fig = create_mckinsey_bar_chart(
                    seasonality,
                    'MONTH',
                    'PF_Index',
                    'Monthly Seasonality Pattern',
                    orientation='v'
                )
                fig.add_hline(
                    y=100,
                    line_dash="dash",
                    line_color=MCKINSEY_COLORS["gray"],
                    opacity=0.5,
                    annotation_text="Average"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # ML Forecasting
            if ml_enabled and len(ts_data) >= 12:
                st.markdown("---")
                st.markdown("### 🤖 Machine Learning Forecasting")
                
                with st.spinner("Training ML models..."):
                    # Prepare data for ML
                    ml_analyzer = MckinseyMLAnalyzer()
                    
                    # Feature engineering
                    features_df = ml_analyzer.prepare_features(ts_data[['PF_Satis', 'YEAR_MONTH']].copy())
                    
                    # Train/test split
                    split_idx = int(len(features_df) * 0.8)
                    train_data = features_df.iloc[:split_idx]
                    test_data = features_df.iloc[split_idx:]
                    
                    # Feature columns
                    feature_cols = [col for col in features_df.columns if col not in ['PF_Satis', 'YEAR_MONTH']]
                    
                    if len(feature_cols) > 0 and len(train_data) > 0:
                        X_train = train_data[feature_cols]
                        y_train = train_data['PF_Satis']
                        X_test = test_data[feature_cols]
                        y_test = test_data['PF_Satis']
                        
                        # Train models
                        results = ml_analyzer.train_models(X_train, y_train, X_test, y_test)
                        
                        if results:
                            # Display model performance
                            st.markdown("#### Model Performance Comparison")
                            
                            perf_data = []
                            for name, metrics in results.items():
                                perf_data.append({
                                    'Model': name,
                                    'MAE': f"{metrics['mae']:,.0f}",
                                    'RMSE': f"{metrics['rmse']:,.0f}",
                                    'MAPE': f"{metrics['mape']:.2f}%",
                                    'R²': f"{metrics['r2']:.3f}"
                                })
                            
                            perf_df = pd.DataFrame(perf_data)
                            st.dataframe(perf_df, use_container_width=True)
                            
                            # Best model forecast
                            best_model_name, best_metrics = ml_analyzer.best_model
                            
                            st.markdown(f"""
                            <div class="mckinsey-card">
                                <h4>🏆 Best Performing Model</h4>
                                <p><strong>{best_model_name}</strong> with MAPE of {best_metrics['mape']:.2f}%</p>
                                <p><strong>Recommendation:</strong> Use this model for forecasting with high confidence</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Feature Importance
                            if 'Random Forest' in results:
                                st.markdown("#### 🔍 Feature Importance")
                                
                                importance_df = ml_analyzer.feature_importance('Random Forest')
                                if not importance_df.empty:
                                    fig = create_mckinsey_bar_chart(
                                        importance_df,
                                        'Feature',
                                        'Importance',
                                        'Top Feature Importances',
                                        orientation='h'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        
        else:
            st.warning("⚠️ Insufficient time series data for analysis")
    
    # =========================================================================
    # TAB 5: ADVANCED ML ANALYTICS
    # =========================================================================
    with tabs[4]:
        st.markdown("## Advanced Machine Learning Analytics")
        
        if not ml_enabled:
            st.warning("⚠️ Enable ML in sidebar settings to access advanced analytics")
        else:
            # Cluster Analysis
            st.markdown("### 🎯 Customer Segmentation")
            
            city_perf = calculate_city_performance(df_filtered, selected_product, date_filter)
            
            if len(city_perf) >= 10:
                # Prepare data for clustering
                cluster_data = city_perf[['PF_Satis', 'Pazar_Payi_%']].copy()
                cluster_data['Log_Sales'] = np.log1p(cluster_data['PF_Satis'])
                
                # K-means clustering
                n_clusters = st.slider("Number of Clusters", 2, 6, 4)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(cluster_data[['Log_Sales', 'Pazar_Payi_%']])
                
                city_perf['Cluster'] = cluster_labels
                
                # Cluster visualization
                fig = create_mckinsey_scatter_plot(
                    city_perf,
                    'PF_Satis',
                    'Pazar_Payi_%',
                    'PF_Satis',
                    'Cluster',
                    'Customer Segmentation Clusters'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster statistics
                st.markdown("#### 📊 Cluster Characteristics")
                
                cluster_stats = city_perf.groupby('Cluster').agg({
                    'PF_Satis': ['mean', 'sum', 'count'],
                    'Pazar_Payi_%': 'mean',
                    'City': 'count'
                }).round(2)
                
                cluster_stats.columns = ['Avg Sales', 'Total Sales', 'Count', 'Avg Share', 'City Count']
                st.dataframe(cluster_stats, use_container_width=True)
                
                # Cluster recommendations
                st.markdown("#### 💡 Cluster-Based Recommendations")
                
                for cluster_id in range(n_clusters):
                    cluster_data = city_perf[city_perf['Cluster'] == cluster_id]
                    
                    with st.expander(f"Cluster {cluster_id} - {len(cluster_data)} cities"):
                        col_rec1, col_rec2 = st.columns(2)
                        
                        with col_rec1:
                            avg_sales = cluster_data['PF_Satis'].mean()
                            avg_share = cluster_data['Pazar_Payi_%'].mean()
                            
                            st.metric("Avg Sales", format_number(avg_sales))
                            st.metric("Avg Share", f"{avg_share:.1f}%")
                        
                        with col_rec2:
                            total_sales = cluster_data['PF_Satis'].sum()
                            st.metric("Total Sales", format_number(total_sales))
                        
                        # Recommendations based on cluster characteristics
                        if avg_share > 50 and avg_sales > city_perf['PF_Satis'].mean():
                            st.success("**Strategy:** Maintain dominance, optimize pricing")
                        elif avg_share > 50 and avg_sales < city_perf['PF_Satis'].mean():
                            st.warning("**Strategy:** Increase volume, defend position")
                        elif avg_share < 50 and avg_sales > city_perf['PF_Satis'].mean():
                            st.info("**Strategy:** Grow share, competitive positioning")
                        else:
                            st.error("**Strategy:** Evaluate market viability")
            
            # Predictive Analytics
            st.markdown("---")
            st.markdown("### 🔮 Predictive Analytics")
            
            if 'DATE' in df_filtered.columns:
                # Prepare time series data
                daily_data = df_filtered.groupby('DATE').agg({
                    get_product_columns(selected_product)['pf']: 'sum'
                }).reset_index()
                daily_data.columns = ['DATE', 'PF_Satis']
                
                if len(daily_data) >= 30:
                    # Prophet model for forecasting
                    with st.spinner("Training Prophet model..."):
                        prophet_data = daily_data[['DATE', 'PF_Satis']].copy()
                        prophet_data.columns = ['ds', 'y']
                        
                        # Train Prophet model
                        model = Prophet(
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False,
                            seasonality_mode='multiplicative'
                        )
                        model.fit(prophet_data)
                        
                        # Make future dataframe
                        future = model.make_future_dataframe(periods=forecast_periods * 30)  # Daily forecast
                        forecast = model.predict(future)
                        
                        # Plot forecast
                        fig1 = model.plot(forecast)
                        st.pyplot(fig1)
                        
                        # Plot components
                        fig2 = model.plot_components(forecast)
                        st.pyplot(fig2)
                else:
                    st.warning("Need at least 30 days of data for Prophet forecasting")
    
    # =========================================================================
    # TAB 6: STRATEGIC INSIGHTS
    # =========================================================================
    with tabs[5]:
        st.markdown("## Strategic Insights & Recommendations")
        
        # Generate comprehensive insights
        city_perf = calculate_city_performance(df_filtered, selected_product, date_filter)
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        bcg_matrix = calculate_bcg_matrix(df_filtered, selected_product, date_filter)
        
        # McKinsey-style insights
        insights = []
        
        # Insight 1: Market Share Analysis
        avg_share = city_perf['Pazar_Payi_%'].mean()
        if avg_share > 60:
            insights.append({
                "title": "🎯 Market Dominance",
                "content": f"Strong market position with {avg_share:.1f}% average share across all cities",
                "recommendation": "Leverage dominant position to increase prices and margins",
                "priority": "High"
            })
        elif avg_share > 40:
            insights.append({
                "title": "📈 Competitive Position",
                "content": f"Solid competitive position with {avg_share:.1f}% market share",
                "recommendation": "Focus on gaining share in underperforming territories",
                "priority": "Medium"
            })
        else:
            insights.append({
                "title": "⚠️ Market Challenge",
                "content": f"Below average market share of {avg_share:.1f}%",
                "recommendation": "Aggressive market penetration strategy required",
                "priority": "Critical"
            })
        
        # Insight 2: Geographic Coverage
        total_cities = df['CITY_CLEAN'].nunique()
        covered_cities = len(city_perf)
        coverage_rate = (covered_cities / total_cities * 100) if total_cities > 0 else 0
        
        if coverage_rate > 80:
            insights.append({
                "title": "🗺️ Extensive Coverage",
                "content": f"Excellent geographic coverage: {covered_cities}/{total_cities} cities ({coverage_rate:.1f}%)",
                "recommendation": "Focus on deepening penetration in existing markets",
                "priority": "Medium"
            })
        elif coverage_rate > 50:
            insights.append({
                "title": "📍 Moderate Coverage",
                "content": f"Moderate geographic coverage: {covered_cities}/{total_cities} cities ({coverage_rate:.1f}%)",
                "recommendation": "Expand to adjacent territories with similar characteristics",
                "priority": "High"
            })
        else:
            insights.append({
                "title": "🚀 Expansion Opportunity",
                "content": f"Significant expansion opportunity: only {covered_cities}/{total_cities} cities covered ({coverage_rate:.1f}%)",
                "recommendation": "Aggressive geographic expansion strategy",
                "priority": "Critical"
            })
        
        # Insight 3: Portfolio Health
        stars_ratio = len(bcg_matrix[bcg_matrix['BCG_Category'] == '⭐ Star']) / len(bcg_matrix) * 100
        dogs_ratio = len(bcg_matrix[bcg_matrix['BCG_Category'] == '🐶 Dog']) / len(bcg_matrix) * 100
        
        if stars_ratio > 30:
            insights.append({
                "title": "⭐ Healthy Portfolio",
                "content": f"Strong portfolio with {stars_ratio:.1f}% Stars and {dogs_ratio:.1f}% Dogs",
                "recommendation": "Maintain investment in Stars, harvest Cash Cows",
                "priority": "Low"
            })
        elif dogs_ratio > 30:
            insights.append({
                "title": "🔧 Portfolio Optimization Needed",
                "content": f"Portfolio requires optimization: {dogs_ratio:.1f}% Dogs need attention",
                "recommendation": "Divest Dogs, reallocate resources to Question Marks",
                "priority": "High"
            })
        
        # Display insights
        for insight in insights:
            priority_color = {
                "Critical": MCKINSEY_COLORS["red"],
                "High": MCKINSEY_COLORS["orange"],
                "Medium": MCKINSEY_COLORS["teal"],
                "Low": MCKINSEY_COLORS["green"]
            }.get(insight["priority"], MCKINSEY_COLORS["gray"])
            
            st.markdown(f"""
            <div class="mckinsey-card">
                <h4>{insight['title']} <span style="color: {priority_color}; font-size: 0.8em;">[{insight['priority']}]</span></h4>
                <div class="card-content">
                    <p><strong>Observation:</strong> {insight['content']}</p>
                    <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Strategic Recommendations Summary
        st.markdown("---")
        st.markdown("### 🎯 Strategic Action Plan")
        
        col_action1, col_action2, col_action3 = st.columns(3)
        
        with col_action1:
            st.markdown("""
            <div class="mckinsey-card">
                <h4>🚀 Growth Initiatives</h4>
                <div class="card-content">
                    <p>• Increase investment in Star territories</p>
                    <p>• Expand geographic coverage by 20%</p>
                    <p>• Launch 3 new market initiatives</p>
                    <p>• Target 15% market share growth</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_action2:
            st.markdown("""
            <div class="mckinsey-card">
                <h4>🛡️ Defensive Actions</h4>
                <div class="card-content">
                    <p>• Protect market share in Cash Cows</p>
                    <p>• Improve retention in key territories</p>
                    <p>• Competitive pricing analysis</p>
                    <p>• Customer loyalty programs</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_action3:
            st.markdown("""
            <div class="mckinsey-card">
                <h4>💡 Optimization</h4>
                <div class="card-content">
                    <p>• Divest underperforming Dogs</p>
                    <p>• Reallocate resources to high-potential markets</p>
                    <p>• Process optimization initiatives</p>
                    <p>• Cost reduction programs</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 7: DETAILED REPORTS
    # =========================================================================
    with tabs[6]:
        st.markdown("## Detailed Reports & Export")
        
        # Report Generation
        col_report1, col_report2, col_report3 = st.columns(3)
        
        with col_report1:
            if st.button("📊 Generate Performance Report", use_container_width=True):
                with st.spinner("Generating report..."):
                    # Create comprehensive report
                    report_data = []
                    
                    # Executive Summary
                    report_data.append({
                        'Section': 'Executive Summary',
                        'Metric': 'Total PF Sales',
                        'Value': format_currency(total_pf),
                        'Insight': f"{market_share:.1f}% market share"
                    })
                    
                    # Geographic Analysis
                    top_cities = city_perf.head(5)
                    for _, row in top_cities.iterrows():
                        report_data.append({
                            'Section': 'Top Cities',
                            'Metric': row['City'],
                            'Value': format_number(row['PF_Satis']),
                            'Insight': f"{row['Pazar_Payi_%']:.1f}% share"
                        })
                    
                    # BCG Analysis
                    for category in ['⭐ Star', '🐄 Cash Cow', '❓ Question Mark', '🐶 Dog']:
                        count = len(bcg_matrix[bcg_matrix['BCG_Category'] == category])
                        report_data.append({
                            'Section': 'BCG Analysis',
                            'Metric': category,
                            'Value': str(count),
                            'Insight': f"{count/len(bcg_matrix)*100:.1f}% of portfolio"
                        })
                    
                    report_df = pd.DataFrame(report_data)
                    st.dataframe(report_df, use_container_width=True)
                    
                    # Download button
                    csv = report_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Report as CSV",
                        data=csv,
                        file_name="mckinsey_report.csv",
                        mime="text/csv"
                    )
        
        with col_report2:
            if st.button("🗺️ Export Geographic Data", use_container_width=True):
                st.download_button(
                    label="📥 Download Geographic Data",
                    data=city_perf.to_csv(index=False),
                    file_name="geographic_analysis.csv",
                    mime="text/csv"
                )
        
        with col_report3:
            if st.button("📈 Export Time Series Data", use_container_width=True):
                ts_data = calculate_time_series_analysis(df_filtered, selected_product, 'YEAR_MONTH', date_filter)
                st.download_button(
                    label="📥 Download Time Series Data",
                    data=ts_data.to_csv(index=False),
                    file_name="time_series_data.csv",
                    mime="text/csv"
                )
        
        # Data Explorer
        st.markdown("---")
        st.markdown("### 🔍 Data Explorer")
        
        # Select data to view
        data_option = st.selectbox(
            "Select Dataset",
            [
                "City Performance",
                "Territory Performance", 
                "BCG Matrix",
                "Time Series",
                "Competitor Analysis"
            ]
        )
        
        if data_option == "City Performance":
            display_data = city_perf
        elif data_option == "Territory Performance":
            display_data = terr_perf
        elif data_option == "BCG Matrix":
            display_data = bcg_matrix
        elif data_option == "Time Series":
            display_data = calculate_time_series_analysis(df_filtered, selected_product, 'YEAR_MONTH', date_filter)
        elif data_option == "Competitor Analysis":
            display_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
        
        # Data filters
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            sort_by = st.selectbox(
                "Sort By",
                display_data.columns.tolist(),
                index=0 if len(display_data.columns) > 0 else 0
            )
        
        with col_filter2:
            sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
        
        # Sort data
        display_data = display_data.sort_values(
            sort_by,
            ascending=(sort_order == "Ascending")
        )
        
        # Display data
        st.dataframe(
            display_data,
            use_container_width=True,
            height=500
        )
        
        # Data summary
        st.markdown("#### 📊 Data Summary")
        
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        
        with col_sum1:
            st.metric("Rows", len(display_data))
        
        with col_sum2:
            numeric_cols = display_data.select_dtypes(include=[np.number]).columns.tolist()
            st.metric("Numeric Columns", len(numeric_cols))
        
        with col_sum3:
            if 'PF_Satis' in display_data.columns:
                total_value = display_data['PF_Satis'].sum()
                st.metric("Total PF Sales", format_currency(total_value))
        
        with col_sum4:
            if 'Pazar_Payi_%' in display_data.columns:
                avg_share = display_data['Pazar_Payi_%'].mean()
                st.metric("Avg Market Share", f"{avg_share:.1f}%")

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
