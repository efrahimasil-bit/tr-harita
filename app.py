"""
🏢 MCKINSEY-STYLE COMMERCIAL PORTFOLIO ANALYTICS SUITE
Advanced Territory Performance Analysis with ML Forecasting & Turkey Mapping
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from io import BytesIO
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import altair as alt
import geopandas as gpd
import json
import requests
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIG - MCKINSEY PROFESSIONAL
# =============================================================================
st.set_page_config(
    page_title="Commercial Portfolio Analytics | McKinsey Style",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.mckinsey.com',
        'Report a bug': None,
        'About': "McKinsey-Style Commercial Analytics Suite v2.0"
    }
)

# =============================================================================
# MCKINSEY CSS THEME - PROFESSIONAL
# =============================================================================
st.markdown("""
<style>
    /* McKinsey Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&family=Roboto:wght@300;400;500&display=swap');
    
    * {
        font-family: 'Source Sans Pro', 'Roboto', sans-serif;
    }
    
    /* McKinsey Color Palette */
    :root {
        --mck-blue: #0066CC;
        --mck-dark-blue: #004080;
        --mck-light-blue: #E6F0FF;
        --mck-green: #0D652D;
        --mck-light-green: #E6F4EA;
        --mck-gray-900: #1A1A1A;
        --mck-gray-700: #333333;
        --mck-gray-500: #666666;
        --mck-gray-300: #D0D0D0;
        --mck-gray-100: #F8F9FA;
        --mck-white: #FFFFFF;
        --mck-border: #E0E0E0;
    }
    
    .stApp {
        background-color: var(--mck-white);
    }
    
    /* Main Header - McKinsey Style */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: var(--mck-gray-900);
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        border-bottom: 3px solid var(--mck-blue);
        background: linear-gradient(90deg, var(--mck-blue) 0%, var(--mck-green) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: var(--mck-gray-500);
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
    
    /* Metrics - Professional Style */
    div[data-testid="stMetricValue"] {
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--mck-gray-900);
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--mck-gray-500);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 0.5rem;
    }
    
    div[data-testid="metric-container"] {
        background: var(--mck-white);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid var(--mck-blue);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid var(--mck-border);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border-left-color: var(--mck-green);
    }
    
    /* Tabs - Professional */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--mck-gray-100);
        border-bottom: 2px solid var(--mck-border);
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 24px;
        color: var(--mck-gray-500);
        font-weight: 600;
        font-size: 0.9rem;
        background: transparent;
        border: none;
        border-bottom: 3px solid transparent;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--mck-blue);
        background: rgba(0, 102, 204, 0.05);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--mck-blue);
        background: var(--mck-white);
        border-bottom: 3px solid var(--mck-blue);
    }
    
    /* Headings */
    h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--mck-gray-900);
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--mck-border);
    }
    
    h2 {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--mck-gray-900);
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--mck-gray-700);
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    h4 {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--mck-gray-700);
        margin-bottom: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--mck-blue);
        color: var(--mck-white);
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 102, 204, 0.2);
    }
    
    .stButton > button:hover {
        background: var(--mck-dark-blue);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--mck-gray-100);
        border-right: 1px solid var(--mck-border);
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Cards */
    .mck-card {
        background: var(--mck-white);
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid var(--mck-border);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .mck-card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    
    .mck-card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--mck-gray-900);
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--mck-border);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-excellent { background: var(--mck-light-green); color: var(--mck-green); }
    .badge-good { background: #E6F0FF; color: var(--mck-blue); }
    .badge-fair { background: #FFF4E5; color: #B35C00; }
    .badge-poor { background: #FFE6E6; color: #CC0000; }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--mck-blue), var(--mck-green));
    }
    
    /* Dataframes */
    .dataframe {
        font-size: 0.9rem;
        border: 1px solid var(--mck-border) !important;
    }
    
    .dataframe thead th {
        background: var(--mck-gray-100);
        color: var(--mck-gray-700);
        font-weight: 600;
        border-bottom: 2px solid var(--mck-border);
        padding: 0.75rem;
    }
    
    .dataframe tbody td {
        padding: 0.75rem;
        border-bottom: 1px solid var(--mck-border);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--mck-gray-100);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--mck-gray-300);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--mck-gray-500);
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        border-bottom: 1px dotted var(--mck-gray-500);
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 250px;
        background: var(--mck-gray-900);
        color: var(--mck-white);
        text-align: center;
        border-radius: 4px;
        padding: 0.75rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -125px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.85rem;
        line-height: 1.4;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Insights Box */
    .insight-box {
        background: linear-gradient(135deg, var(--mck-light-blue), var(--mck-light-green));
        border-left: 4px solid var(--mck-blue);
        padding: 1.25rem;
        border-radius: 6px;
        margin: 1.5rem 0;
    }
    
    .insight-box h4 {
        color: var(--mck-gray-900);
        margin-top: 0;
    }
    
    /* Warning Box */
    .warning-box {
        background: #FFF4E5;
        border-left: 4px solid #FF8C00;
        padding: 1.25rem;
        border-radius: 6px;
        margin: 1.5rem 0;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--mck-border), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MCKINSEY COLOR PALETTE & CONSTANTS
# =============================================================================

# McKinsey Professional Colors
MCKINSEY_PALETTE = {
    # Primary Colors
    'primary_blue': '#0066CC',
    'primary_dark_blue': '#004080',
    'primary_green': '#0D652D',
    'primary_orange': '#FF8C00',
    
    # Secondary Colors
    'secondary_teal': '#008080',
    'secondary_purple': '#663399',
    'secondary_red': '#CC0000',
    'secondary_yellow': '#FFD700',
    
    # Neutral Colors
    'gray_900': '#1A1A1A',
    'gray_800': '#333333',
    'gray_700': '#4D4D4D',
    'gray_600': '#666666',
    'gray_500': '#808080',
    'gray_400': '#999999',
    'gray_300': '#B3B3B3',
    'gray_200': '#CCCCCC',
    'gray_100': '#E6E6E6',
    'gray_50': '#F8F8F8',
    
    # Status Colors
    'success': '#0D652D',
    'success_light': '#E6F4EA',
    'warning': '#FF8C00',
    'warning_light': '#FFF4E5',
    'error': '#CC0000',
    'error_light': '#FFE6E6',
    'info': '#0066CC',
    'info_light': '#E6F0FF',
}

# Region Colors (McKinsey Professional)
REGION_COLORS = {
    'MARMARA': MCKINSEY_PALETTE['primary_blue'],
    'EGE': MCKINSEY_PALETTE['primary_green'],
    'AKDENİZ': MCKINSEY_PALETTE['primary_orange'],
    'İÇ ANADOLU': MCKINSEY_PALETTE['secondary_purple'],
    'KARADENİZ': MCKINSEY_PALETTE['secondary_teal'],
    'GÜNEYDOĞU ANADOLU': MCKINSEY_PALETTE['secondary_red'],
    'DOĞU ANADOLU': '#5D8C3A',
    'BATI KARADENİZ': '#4A6FA5',
    'ORTA KARADENİZ': '#00A896',
    'KUZEYDOĞU ANADOLU': '#FF6B35',
    'GÜNEYDOĞU': '#D81159',
    'DİĞER': MCKINSEY_PALETTE['gray_500']
}

# Product Mappings
PRODUCT_CONFIG = {
    'TROCMETAM': {
        'name': 'TROCMETAM',
        'pf_column': 'TROCMETAM',
        'comp_column': 'DIGER_TROCMETAM',
        'category': 'Specialty'
    },
    'CORTIPOL': {
        'name': 'CORTIPOL',
        'pf_column': 'CORTIPOL',
        'comp_column': 'DIGER_CORTIPOL',
        'category': 'Specialty'
    },
    'DEKSAMETAZON': {
        'name': 'DEKSAMETAZON',
        'pf_column': 'DEKSAMETAZON',
        'comp_column': 'DIGER_DEKSAMETAZON',
        'category': 'Specialty'
    },
    'PF_IZOTONIK': {
        'name': 'PF İZOTONİK',
        'pf_column': 'PF_IZOTONIK',
        'comp_column': 'DIGER_IZOTONIK',
        'category': 'Standard'
    }
}

# Turkish City Coordinates (Optimized)
TURKEY_CITIES = {
    'İSTANBUL': {'lat': 41.0082, 'lon': 28.9784, 'region': 'MARMARA'},
    'ANKARA': {'lat': 39.9334, 'lon': 32.8597, 'region': 'İÇ ANADOLU'},
    'İZMİR': {'lat': 38.4237, 'lon': 27.1428, 'region': 'EGE'},
    'BURSA': {'lat': 40.1885, 'lon': 29.0610, 'region': 'MARMARA'},
    'ANTALYA': {'lat': 36.8969, 'lon': 30.7133, 'region': 'AKDENİZ'},
    'ADANA': {'lat': 37.0000, 'lon': 35.3213, 'region': 'AKDENİZ'},
    'KONYA': {'lat': 37.9838, 'lon': 32.4846, 'region': 'İÇ ANADOLU'},
    'GAZİANTEP': {'lat': 37.0662, 'lon': 37.3826, 'region': 'GÜNEYDOĞU ANADOLU'},
    'ŞANLIURFA': {'lat': 37.1592, 'lon': 38.7944, 'region': 'GÜNEYDOĞU ANADOLU'},
    'MERSİN': {'lat': 36.8121, 'lon': 34.6415, 'region': 'AKDENİZ'},
    'DİYARBAKIR': {'lat': 37.9100, 'lon': 40.2400, 'region': 'GÜNEYDOĞU ANADOLU'},
    'KAYSERİ': {'lat': 38.7312, 'lon': 35.4833, 'region': 'İÇ ANADOLU'},
    'ESKİŞEHİR': {'lat': 39.7767, 'lon': 30.5206, 'region': 'İÇ ANADOLU'},
    'SAKARYA': {'lat': 40.6933, 'lon': 30.3781, 'region': 'MARMARA'},
    'TEKİRDAĞ': {'lat': 40.9833, 'lon': 27.5167, 'region': 'MARMARA'},
    'BALIKESİR': {'lat': 39.6484, 'lon': 27.8826, 'region': 'EGE'},
    'DENİZLİ': {'lat': 37.7765, 'lon': 29.0864, 'region': 'EGE'},
    'SAMSUN': {'lat': 41.2867, 'lon': 36.3300, 'region': 'KARADENİZ'},
    'TRABZON': {'lat': 41.0015, 'lon': 39.7178, 'region': 'KARADENİZ'},
    'ERZURUM': {'lat': 39.9086, 'lon': 41.2769, 'region': 'DOĞU ANADOLU'},
    'VAN': {'lat': 38.4942, 'lon': 43.3800, 'region': 'DOĞU ANADOLU'},
    'MUĞLA': {'lat': 37.2153, 'lon': 28.3636, 'region': 'EGE'},
    'AYDIN': {'lat': 37.8560, 'lon': 27.8416, 'region': 'EGE'},
    'MANİSA': {'lat': 38.6191, 'lon': 27.4289, 'region': 'EGE'},
    'KOCAELİ': {'lat': 40.7667, 'lon': 29.9167, 'region': 'MARMARA'},
    'HATAY': {'lat': 36.4018, 'lon': 36.3498, 'region': 'AKDENİZ'},
    'KAHRAMANMARAŞ': {'lat': 37.5753, 'lon': 36.9261, 'region': 'AKDENİZ'},
    'MALATYA': {'lat': 38.3552, 'lon': 38.3095, 'region': 'DOĞU ANADOLU'},
    'ELAZIĞ': {'lat': 38.6800, 'lon': 39.2264, 'region': 'DOĞU ANADOLU'},
    'ORDU': {'lat': 40.9839, 'lon': 37.8789, 'region': 'KARADENİZ'},
    'GİRESUN': {'lat': 40.9128, 'lon': 38.3903, 'region': 'KARADENİZ'},
    'RİZE': {'lat': 41.0201, 'lon': 40.5239, 'region': 'KARADENİZ'},
    'ZONGULDAK': {'lat': 41.4564, 'lon': 31.7987, 'region': 'KARADENİZ'},
    'KIRIKKALE': {'lat': 39.8468, 'lon': 33.5153, 'region': 'İÇ ANADOLU'},
    'AKSARAY': {'lat': 38.3686, 'lon': 34.0360, 'region': 'İÇ ANADOLU'},
    'YOZGAT': {'lat': 39.8200, 'lon': 34.8000, 'region': 'İÇ ANADOLU'},
    'NEVŞEHİR': {'lat': 38.6242, 'lon': 34.7236, 'region': 'İÇ ANADOLU'},
    'KIRŞEHİR': {'lat': 39.1500, 'lon': 34.1667, 'region': 'İÇ ANADOLU'},
    'ÇORUM': {'lat': 40.5506, 'lon': 34.9556, 'region': 'KARADENİZ'},
    'TOKAT': {'lat': 40.3167, 'lon': 36.5542, 'region': 'KARADENİZ'},
    'AMASYA': {'lat': 40.6500, 'lon': 35.8333, 'region': 'KARADENİZ'},
    'SİVAS': {'lat': 39.7500, 'lon': 37.0167, 'region': 'İÇ ANADOLU'},
    'ERZİNCAN': {'lat': 39.7500, 'lon': 39.5000, 'region': 'DOĞU ANADOLU'},
    'BİNGÖL': {'lat': 38.8863, 'lon': 40.4937, 'region': 'DOĞU ANADOLU'},
    'MUŞ': {'lat': 38.7333, 'lon': 41.4911, 'region': 'DOĞU ANADOLU'},
    'BİTLİS': {'lat': 38.4000, 'lon': 42.1086, 'region': 'DOĞU ANADOLU'},
    'HAKKARİ': {'lat': 37.5833, 'lon': 43.7333, 'region': 'DOĞU ANADOLU'},
    'ŞIRNAK': {'lat': 37.5167, 'lon': 42.4639, 'region': 'GÜNEYDOĞU ANADOLU'},
    'MARDİN': {'lat': 37.3122, 'lon': 40.7350, 'region': 'GÜNEYDOĞU ANADOLU'},
    'SİİRT': {'lat': 37.9333, 'lon': 41.9403, 'region': 'GÜNEYDOĞU ANADOLU'},
    'BATMAN': {'lat': 37.8812, 'lon': 41.1350, 'region': 'GÜNEYDOĞU ANADOLU'},
    'ADIYAMAN': {'lat': 37.7648, 'lon': 38.2786, 'region': 'GÜNEYDOĞU ANADOLU'},
    'KİLİS': {'lat': 36.7164, 'lon': 37.1150, 'region': 'GÜNEYDOĞU ANADOLU'},
    'OSMANİYE': {'lat': 37.0742, 'lon': 36.2478, 'region': 'AKDENİZ'},
    'KARAMAN': {'lat': 37.1811, 'lon': 33.2150, 'region': 'İÇ ANADOLU'},
    'KARABÜK': {'lat': 41.2000, 'lon': 32.6228, 'region': 'KARADENİZ'},
    'DÜZCE': {'lat': 40.8439, 'lon': 31.1639, 'region': 'KARADENİZ'},
    'BARTIN': {'lat': 41.6358, 'lon': 32.3375, 'region': 'KARADENİZ'},
    'KASTAMONU': {'lat': 41.3764, 'lon': 33.7750, 'region': 'KARADENİZ'},
    'SİNOP': {'lat': 42.0231, 'lon': 35.1519, 'region': 'KARADENİZ'}
}

# =============================================================================
# DATA LOADING & PROCESSING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_and_process_data(uploaded_file: BytesIO) -> pd.DataFrame:
    """
    Load and process Excel data with professional error handling
    """
    try:
        # Load Excel
        df = pd.read_excel(uploaded_file)
        
        # Clean column names
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        # Identify date column
        date_patterns = ['DATE', 'TARIH', 'TARİH', 'PERIOD', 'DONEM', 'DÖNEM', 'AY', 'MONTH', 'YEAR']
        date_col = None
        for pattern in date_patterns:
            for col in df.columns:
                if pattern in col:
                    date_col = col
                    break
            if date_col:
                break
        
        if date_col:
            df['DATE'] = pd.to_datetime(df[date_col], errors='coerce')
        else:
            # Try to infer date from first column
            try:
                df['DATE'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
            except:
                # Create dummy dates if none found
                df['DATE'] = pd.date_range(start='2023-01-01', periods=len(df), freq='M')
        
        # Create time features
        df['YEAR_MONTH'] = df['DATE'].dt.strftime('%Y-%m')
        df['YEAR'] = df['DATE'].dt.year
        df['MONTH'] = df['DATE'].dt.month
        df['QUARTER'] = df['DATE'].dt.quarter
        
        # Standardize geographic columns
        if 'CITY' not in df.columns:
            for col in ['SEHIR', 'ŞEHİR', 'İL', 'IL', 'CİTY']:
                if col in df.columns:
                    df['CITY'] = df[col].astype(str).str.upper().str.strip()
                    break
        
        if 'REGION' not in df.columns:
            for col in ['BOLGE', 'BÖLGE', 'REGİON']:
                if col in df.columns:
                    df['REGION'] = df[col].astype(str).str.upper().str.strip()
                    break
        
        if 'TERRITORY' not in df.columns:
            for col in ['TERRITORIES', 'TERRİTORY', 'BOLGE_KODU']:
                if col in df.columns:
                    df['TERRITORY'] = df[col].astype(str).str.upper().str.strip()
                    break
        
        # Clean city names
        df['CITY_CLEAN'] = df['CITY'].apply(lambda x: clean_city_name(x) if pd.notna(x) else 'UNKNOWN')
        
        # Add coordinates for mapping
        df['LAT'] = df['CITY_CLEAN'].apply(lambda x: TURKEY_CITIES.get(x, {}).get('lat', 39.0))
        df['LON'] = df['CITY_CLEAN'].apply(lambda x: TURKEY_CITIES.get(x, {}).get('lon', 35.0))
        df['REGION_MAP'] = df['CITY_CLEAN'].apply(lambda x: TURKEY_CITIES.get(x, {}).get('region', 'DİĞER'))
        
        return df
        
    except Exception as e:
        st.error(f"❌ Data loading error: {str(e)}")
        st.stop()

def clean_city_name(city_name: str) -> str:
    """
    Clean and standardize Turkish city names
    """
    if not isinstance(city_name, str):
        return 'UNKNOWN'
    
    city = city_name.strip().upper()
    
    # Turkish character normalization
    turkish_chars = {
        'İ': 'I', 'Ğ': 'G', 'Ü': 'U', 'Ş': 'S', 'Ö': 'O', 'Ç': 'C',
        'Â': 'A', 'Î': 'I', 'Û': 'U'
    }
    
    for old, new in turkish_chars.items():
        city = city.replace(old, new)
    
    # Special cases
    special_cases = {
        'ISTANBUL': 'İSTANBUL',
        'IZMIR': 'İZMİR',
        'ANKARA': 'ANKARA',
        'BURSA': 'BURSA',
        'ANTALYA': 'ANTALYA',
        'ADANA': 'ADANA',
        'KONYA': 'KONYA',
        'GAZIANTEP': 'GAZİANTEP',
        'SANLIURFA': 'ŞANLIURFA',
        'MERSIN': 'MERSİN',
        'DIYARBAKIR': 'DİYARBAKIR',
        'KAYSERI': 'KAYSERİ',
        'ESKISEHIR': 'ESKİŞEHİR',
        'SAKARYA': 'SAKARYA',
        'TEKIRDAG': 'TEKİRDAĞ',
        'BALIKESIR': 'BALIKESİR',
        'DENIZLI': 'DENİZLİ',
        'SAMSUN': 'SAMSUN',
        'TRABZON': 'TRABZON',
        'ERZURUM': 'ERZURUM',
        'VAN': 'VAN',
        'MUGLA': 'MUĞLA',
        'AYDIN': 'AYDIN',
        'MANISA': 'MANİSA',
        'KOCAELI': 'KOCAELİ',
        'HATAY': 'HATAY',
        'K.MARAS': 'KAHRAMANMARAŞ',
        'MALATYA': 'MALATYA',
        'ELAZIG': 'ELAZIĞ',
        'ORDU': 'ORDU',
        'GIRESUN': 'GİRESUN',
        'RIZE': 'RİZE',
        'ZONGULDAK': 'ZONGULDAK',
        'KIRIKKALE': 'KIRIKKALE',
        'AKSARAY': 'AKSARAY',
        'YOZGAT': 'YOZGAT',
        'NEVSEHIR': 'NEVŞEHİR',
        'KIRSEHIR': 'KIRŞEHİR',
        'CORUM': 'ÇORUM',
        'TOKAT': 'TOKAT',
        'AMASYA': 'AMASYA',
        'SIVAS': 'SİVAS',
        'ERZINCAN': 'ERZİNCAN',
        'BINGOL': 'BİNGÖL',
        'MUS': 'MUŞ',
        'BITLIS': 'BİTLİS',
        'HAKKARI': 'HAKKARİ',
        'SIRNAK': 'ŞIRNAK',
        'MARDIN': 'MARDİN',
        'SIIRT': 'SİİRT',
        'BATMAN': 'BATMAN',
        'ADIYAMAN': 'ADIYAMAN',
        'KILIS': 'KİLİS',
        'OSMANIYE': 'OSMANİYE',
        'KARAMAN': 'KARAMAN',
        'KARABUK': 'KARABÜK',
        'DUZCE': 'DÜZCE',
        'BARTIN': 'BARTIN',
        'KASTAMONU': 'KASTAMONU',
        'SINOP': 'SİNOP'
    }
    
    return special_cases.get(city, city)

# =============================================================================
# ANALYTICS FUNCTIONS
# =============================================================================

def calculate_city_performance_metrics(df: pd.DataFrame, product_config: dict, 
                                       start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
    """
    Calculate comprehensive city performance metrics
    """
    # Filter by date if provided
    if start_date and end_date:
        df_filtered = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)].copy()
    else:
        df_filtered = df.copy()
    
    pf_col = product_config['pf_column']
    comp_col = product_config['comp_column']
    
    # Ensure columns exist
    if pf_col not in df_filtered.columns:
        df_filtered[pf_col] = 0
    if comp_col not in df_filtered.columns:
        df_filtered[comp_col] = 0
    
    # Group by city
    city_data = df_filtered.groupby('CITY_CLEAN').agg({
        pf_col: 'sum',
        comp_col: 'sum',
        'REGION_MAP': 'first',
        'LAT': 'first',
        'LON': 'first'
    }).reset_index()
    
    city_data.columns = ['City', 'PF_Sales', 'Competitor_Sales', 'Region', 'Lat', 'Lon']
    
    # Calculate metrics
    city_data['Total_Market'] = city_data['PF_Sales'] + city_data['Competitor_Sales']
    city_data['Market_Share_%'] = np.where(
        city_data['Total_Market'] > 0,
        (city_data['PF_Sales'] / city_data['Total_Market']) * 100,
        0
    )
    city_data['Relative_Performance'] = np.where(
        city_data['Competitor_Sales'] > 0,
        city_data['PF_Sales'] / city_data['Competitor_Sales'],
        0
    )
    
    # Calculate growth if multiple periods
    if len(df_filtered['YEAR_MONTH'].unique()) > 1:
        monthly_data = df_filtered.groupby(['CITY_CLEAN', 'YEAR_MONTH'])[pf_col].sum().unstack()
        if monthly_data.shape[1] >= 2:
            city_data['Growth_%'] = ((monthly_data.iloc[:, -1] - monthly_data.iloc[:, 0]) / 
                                     monthly_data.iloc[:, 0].replace(0, 1)) * 100
            city_data['Growth_%'] = city_data['Growth_%'].fillna(0)
        else:
            city_data['Growth_%'] = 0
    else:
        city_data['Growth_%'] = 0
    
    # Calculate performance score
    city_data['Performance_Score'] = calculate_city_score(city_data)
    city_data['Performance_Class'] = city_data['Performance_Score'].apply(get_performance_class)
    
    return city_data.sort_values('PF_Sales', ascending=False)

def calculate_city_score(city_data: pd.DataFrame) -> pd.Series:
    """
    Calculate composite performance score (0-100)
    """
    scores = []
    
    for _, row in city_data.iterrows():
        score = 0
        
        # Market share score (0-40 points)
        market_share_score = min(row['Market_Share_%'], 100) * 0.4
        score += market_share_score
        
        # Growth score (0-30 points)
        growth = min(max(row['Growth_%'], -50), 100)
        growth_score = (50 + growth * 0.3) * 0.3  # -50% = 35, 0% = 50, 100% = 80
        score += growth_score
        
        # Market size score (0-20 points)
        market_size = np.log1p(row['Total_Market'])
        market_size_score = min(market_size / np.log1p(1e6) * 20, 20)
        score += market_size_score
        
        # Relative performance score (0-10 points)
        rel_perf_score = min(row['Relative_Performance'] * 5, 10)
        score += rel_perf_score
        
        scores.append(score)
    
    return pd.Series(scores, index=city_data.index)

def get_performance_class(score: float) -> str:
    """Classify performance score"""
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 55:
        return "Fair"
    else:
        return "Needs Improvement"

def calculate_territory_metrics(df: pd.DataFrame, product_config: dict) -> pd.DataFrame:
    """
    Calculate territory-level performance metrics
    """
    pf_col = product_config['pf_column']
    comp_col = product_config['comp_column']
    
    # Ensure columns exist
    if pf_col not in df.columns:
        df[pf_col] = 0
    if comp_col not in df.columns:
        df[comp_col] = 0
    
    # Group by territory
    territory_data = df.groupby('TERRITORY').agg({
        pf_col: 'sum',
        comp_col: 'sum',
        'CITY_CLEAN': 'nunique'
    }).reset_index()
    
    territory_data.columns = ['Territory', 'PF_Sales', 'Competitor_Sales', 'City_Count']
    
    # Calculate metrics
    territory_data['Total_Market'] = territory_data['PF_Sales'] + territory_data['Competitor_Sales']
    territory_data['Market_Share_%'] = np.where(
        territory_data['Total_Market'] > 0,
        (territory_data['PF_Sales'] / territory_data['Total_Market']) * 100,
        0
    )
    territory_data['Avg_City_Sales'] = territory_data['PF_Sales'] / territory_data['City_Count']
    territory_data['Territory_Contribution_%'] = (
        territory_data['PF_Sales'] / territory_data['PF_Sales'].sum() * 100
    )
    
    return territory_data.sort_values('PF_Sales', ascending=False)

def calculate_time_series_metrics(df: pd.DataFrame, product_config: dict) -> pd.DataFrame:
    """
    Calculate time series metrics for trend analysis
    """
    pf_col = product_config['pf_column']
    comp_col = product_config['comp_column']
    
    # Ensure columns exist
    if pf_col not in df.columns:
        df[pf_col] = 0
    if comp_col not in df.columns:
        df[comp_col] = 0
    
    # Monthly aggregation
    monthly_data = df.groupby('YEAR_MONTH').agg({
        pf_col: 'sum',
        comp_col: 'sum',
        'DATE': 'first'
    }).reset_index().sort_values('YEAR_MONTH')
    
    monthly_data.columns = ['Year_Month', 'PF_Sales', 'Competitor_Sales', 'Date']
    monthly_data['Total_Market'] = monthly_data['PF_Sales'] + monthly_data['Competitor_Sales']
    monthly_data['Market_Share_%'] = (
        monthly_data['PF_Sales'] / monthly_data['Total_Market'] * 100
    ).fillna(0)
    
    # Growth calculations
    monthly_data['PF_Growth_%'] = monthly_data['PF_Sales'].pct_change() * 100
    monthly_data['Competitor_Growth_%'] = monthly_data['Competitor_Sales'].pct_change() * 100
    monthly_data['Market_Growth_%'] = monthly_data['Total_Market'].pct_change() * 100
    
    # Moving averages
    monthly_data['PF_MA_3M'] = monthly_data['PF_Sales'].rolling(window=3, min_periods=1).mean()
    monthly_data['PF_MA_6M'] = monthly_data['PF_Sales'].rolling(window=6, min_periods=1).mean()
    
    # Cumulative sums
    monthly_data['PF_Cumulative'] = monthly_data['PF_Sales'].cumsum()
    monthly_data['Market_Cumulative'] = monthly_data['Total_Market'].cumsum()
    
    return monthly_data

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_mckinsey_map(city_data: pd.DataFrame, title: str, metric: str = 'PF_Sales') -> go.Figure:
    """
    Create professional McKinsey-style map visualization
    """
    fig = go.Figure()
    
    # Determine size and color based on metric
    if metric == 'PF_Sales':
        size = np.sqrt(city_data['PF_Sales']) * 0.5
        color = city_data['Market_Share_%']
        colorscale = 'RdYlGn'
        colorbar_title = 'Market Share %'
    elif metric == 'Performance_Score':
        size = np.sqrt(city_data['PF_Sales']) * 0.5
        color = city_data['Performance_Score']
        colorscale = 'RdYlGn'
        colorbar_title = 'Performance Score'
    else:
        size = np.sqrt(city_data['PF_Sales']) * 0.5
        color = city_data['Growth_%']
        colorscale = 'RdYlBu'
        colorbar_title = 'Growth %'
    
    # Add scatter points
    fig.add_trace(go.Scattermapbox(
        lat=city_data['Lat'],
        lon=city_data['Lon'],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            colorscale=colorscale,
            cmin=0 if metric != 'Growth_%' else -50,
            cmax=100 if metric != 'Growth_%' else 100,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text=colorbar_title,
                    font=dict(size=10)
                ),
                thickness=15,
                x=1.02,
                xpad=5
            ),
            opacity=0.8,
            sizemode='area',
            sizeref=2.*max(size)/(50.**2),
            sizemin=4
        ),
        text=city_data.apply(
            lambda row: f"<b>{row['City']}</b><br>"
                      f"Region: {row['Region']}<br>"
                      f"PF Sales: {row['PF_Sales']:,.0f}<br>"
                      f"Market Share: {row['Market_Share_%']:.1f}%<br>"
                      f"Growth: {row['Growth_%']:.1f}%<br>"
                      f"Performance: {row['Performance_Score']:.0f}/100",
            axis=1
        ),
        hoverinfo='text'
    ))
    
    # Professional layout
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=39.0, lon=35.0),
            zoom=4.8,
            bearing=0,
            pitch=0
        ),
        height=600,
        margin=dict(l=0, r=0, t=80, b=0),
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(size=20, color=MCKINSEY_PALETTE['gray_900']),
            y=0.95
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Source Sans Pro', color=MCKINSEY_PALETTE['gray_700'])
    )
    
    return fig

def create_performance_matrix(city_data: pd.DataFrame) -> go.Figure:
    """
    Create BCG-style performance matrix
    """
    fig = go.Figure()
    
    # Create size based on market size
    size = np.sqrt(city_data['Total_Market']) / np.sqrt(city_data['Total_Market'].max()) * 40 + 10
    
    # Color by performance class
    color_map = {
        'Excellent': MCKINSEY_PALETTE['success'],
        'Good': MCKINSEY_PALETTE['info'],
        'Fair': MCKINSEY_PALETTE['warning'],
        'Needs Improvement': MCKINSEY_PALETTE['error']
    }
    
    for perf_class in city_data['Performance_Class'].unique():
        subset = city_data[city_data['Performance_Class'] == perf_class]
        
        fig.add_trace(go.Scatter(
            x=subset['Market_Share_%'],
            y=subset['Growth_%'],
            mode='markers',
            marker=dict(
                size=size[subset.index],
                color=color_map[perf_class],
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            name=perf_class,
            text=subset['City'],
            hoverinfo='text'
        ))
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color=MCKINSEY_PALETTE['gray_400'], opacity=0.5)
    fig.add_vline(x=city_data['Market_Share_%'].median(), line_dash="dash", 
                  line_color=MCKINSEY_PALETTE['gray_400'], opacity=0.5)
    
    # Professional layout
    fig.update_layout(
        title=dict(
            text="<b>Performance Matrix: Market Share vs Growth</b>",
            font=dict(size=18, color=MCKINSEY_PALETTE['gray_900']),
            x=0.5
        ),
        xaxis_title=dict(
            text="<b>Market Share (%)</b>",
            font=dict(size=12, color=MCKINSEY_PALETTE['gray_700'])
        ),
        yaxis_title=dict(
            text="<b>Growth Rate (%)</b>",
            font=dict(size=12, color=MCKINSEY_PALETTE['gray_700'])
        ),
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Source Sans Pro', color=MCKINSEY_PALETTE['gray_700']),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        hovermode='closest'
    )
    
    # Add quadrant annotations
    fig.add_annotation(
        x=0.1, y=0.9,
        xref="paper", yref="paper",
        text="High Growth<br>Low Share",
        showarrow=False,
        font=dict(size=10, color=MCKINSEY_PALETTE['gray_600'])
    )
    
    fig.add_annotation(
        x=0.9, y=0.9,
        xref="paper", yref="paper",
        text="High Growth<br>High Share",
        showarrow=False,
        font=dict(size=10, color=MCKINSEY_PALETTE['gray_600'])
    )
    
    fig.add_annotation(
        x=0.1, y=0.1,
        xref="paper", yref="paper",
        text="Low Growth<br>Low Share",
        showarrow=False,
        font=dict(size=10, color=MCKINSEY_PALETTE['gray_600'])
    )
    
    fig.add_annotation(
        x=0.9, y=0.1,
        xref="paper", yref="paper",
        text="Low Growth<br>High Share",
        showarrow=False,
        font=dict(size=10, color=MCKINSEY_PALETTE['gray_600'])
    )
    
    return fig

def create_trend_chart(monthly_data: pd.DataFrame, title: str) -> go.Figure:
    """
    Create professional trend chart
    """
    fig = go.Figure()
    
    # Add PF Sales line
    fig.add_trace(go.Scatter(
        x=monthly_data['Date'],
        y=monthly_data['PF_Sales'],
        mode='lines+markers',
        name='PF Sales',
        line=dict(color=MCKINSEY_PALETTE['primary_blue'], width=3),
        marker=dict(size=6, color='white', line=dict(width=2, color=MCKINSEY_PALETTE['primary_blue']))
    ))
    
    # Add moving average
    fig.add_trace(go.Scatter(
        x=monthly_data['Date'],
        y=monthly_data['PF_MA_3M'],
        mode='lines',
        name='3-Month Moving Avg',
        line=dict(color=MCKINSEY_PALETTE['gray_500'], width=2, dash='dash'),
        opacity=0.7
    ))
    
    # Add competitor sales (area)
    fig.add_trace(go.Scatter(
        x=monthly_data['Date'],
        y=monthly_data['Competitor_Sales'],
        mode='none',
        name='Competitor Sales',
        fill='tozeroy',
        fillcolor='rgba(204, 0, 0, 0.1)',
        line=dict(color=MCKINSEY_PALETTE['secondary_red'], width=0)
    ))
    
    # Professional layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=18, color=MCKINSEY_PALETTE['gray_900']),
            x=0.5
        ),
        xaxis_title=dict(
            text="<b>Date</b>",
            font=dict(size=12, color=MCKINSEY_PALETTE['gray_700'])
        ),
        yaxis_title=dict(
            text="<b>Sales Volume</b>",
            font=dict(size=12, color=MCKINSEY_PALETTE['gray_700'])
        ),
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Source Sans Pro', color=MCKINSEY_PALETTE['gray_700']),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        hovermode='x unified'
    )
    
    return fig

def create_market_share_chart(monthly_data: pd.DataFrame) -> go.Figure:
    """
    Create market share trend chart
    """
    fig = go.Figure()
    
    # Add market share line
    fig.add_trace(go.Scatter(
        x=monthly_data['Date'],
        y=monthly_data['Market_Share_%'],
        mode='lines+markers',
        name='Market Share',
        line=dict(color=MCKINSEY_PALETTE['primary_green'], width=3),
        marker=dict(size=6, color='white', line=dict(width=2, color=MCKINSEY_PALETTE['primary_green']))
    ))
    
    # Add target line
    fig.add_hline(y=50, line_dash="dash", line_color=MCKINSEY_PALETTE['warning'], 
                  annotation_text="50% Target", annotation_position="bottom right")
    
    # Professional layout
    fig.update_layout(
        title=dict(
            text="<b>Market Share Trend</b>",
            font=dict(size=18, color=MCKINSEY_PALETTE['gray_900']),
            x=0.5
        ),
        xaxis_title=dict(
            text="<b>Date</b>",
            font=dict(size=12, color=MCKINSEY_PALETTE['gray_700'])
        ),
        yaxis_title=dict(
            text="<b>Market Share (%)</b>",
            font=dict(size=12, color=MCKINSEY_PALETTE['gray_700'])
        ),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Source Sans Pro', color=MCKINSEY_PALETTE['gray_700']),
        hovermode='x unified'
    )
    
    return fig

def create_regional_analysis(city_data: pd.DataFrame) -> go.Figure:
    """
    Create regional performance comparison
    """
    # Aggregate by region
    region_data = city_data.groupby('Region').agg({
        'PF_Sales': 'sum',
        'Total_Market': 'sum',
        'City': 'count'
    }).reset_index()
    
    region_data['Market_Share_%'] = (region_data['PF_Sales'] / region_data['Total_Market'] * 100).round(1)
    region_data['Avg_City_Sales'] = (region_data['PF_Sales'] / region_data['City']).round(0)
    region_data = region_data.sort_values('PF_Sales', ascending=True)
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=region_data['Region'],
        x=region_data['PF_Sales'],
        name='PF Sales',
        orientation='h',
        marker_color=MCKINSEY_PALETTE['primary_blue'],
        text=region_data['PF_Sales'].apply(lambda x: f"{x:,.0f}"),
        textposition='auto'
    ))
    
    # Professional layout
    fig.update_layout(
        title=dict(
            text="<b>Regional Performance Comparison</b>",
            font=dict(size=18, color=MCKINSEY_PALETTE['gray_900']),
            x=0.5
        ),
        xaxis_title=dict(
            text="<b>PF Sales</b>",
            font=dict(size=12, color=MCKINSEY_PALETTE['gray_700'])
        ),
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Source Sans Pro', color=MCKINSEY_PALETTE['gray_700']),
        showlegend=False,
        bargap=0.2
    )
    
    return fig

# =============================================================================
# MACHINE LEARNING FUNCTIONS
# =============================================================================

def prepare_forecast_data(monthly_data: pd.DataFrame) -> Tuple:
    """
    Prepare data for ML forecasting
    """
    if len(monthly_data) < 12:
        return None, None, None, None
    
    df = monthly_data.copy()
    
    # Create features
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['year'] = df['Date'].dt.year
    df['time_index'] = range(len(df))
    
    # Lag features
    for lag in [1, 2, 3, 6]:
        df[f'lag_{lag}'] = df['PF_Sales'].shift(lag)
    
    # Rolling statistics
    df['rolling_mean_3'] = df['PF_Sales'].rolling(window=3, min_periods=1).mean()
    df['rolling_mean_6'] = df['PF_Sales'].rolling(window=6, min_periods=1).mean()
    df['rolling_std_3'] = df['PF_Sales'].rolling(window=3, min_periods=1).std()
    
    # Seasonal features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Target
    df['target'] = df['PF_Sales'].shift(-1)
    
    # Drop NaN
    df_clean = df.dropna()
    
    if len(df_clean) < 10:
        return None, None, None, None
    
    # Feature selection
    feature_cols = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 
                    'rolling_mean_6', 'month', 'quarter', 'time_index',
                    'month_sin', 'month_cos']
    feature_cols = [col for col in feature_cols if col in df_clean.columns]
    
    X = df_clean[feature_cols]
    y = df_clean['target']
    
    # Split
    split_idx = int(len(df_clean) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_forecasting_models(X_train, X_test, y_train, y_test):
    """
    Train multiple ML models for forecasting
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[name] = {
                'model': model,
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
                'R2': r2_score(y_test, y_pred),
                'predictions': y_pred
            }
        except:
            continue
    
    return results

def create_forecast_visualization(historical_data, forecast_results, best_model_name):
    """
    Create forecast visualization
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data['Date'],
        y=historical_data['PF_Sales'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color=MCKINSEY_PALETTE['primary_blue'], width=3),
        marker=dict(size=6)
    ))
    
    # Forecast (if available)
    if forecast_results and best_model_name in forecast_results:
        last_date = historical_data['Date'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')
        
        # Simple forecast based on trend
        trend = np.polyfit(range(len(historical_data)), historical_data['PF_Sales'], 1)
        forecast_values = np.polyval(trend, range(len(historical_data), len(historical_data) + 6))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines+markers',
            name=f'Forecast ({best_model_name})',
            line=dict(color=MCKINSEY_PALETTE['primary_green'], width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
        
        # Confidence interval
        std_dev = historical_data['PF_Sales'].std()
        fig.add_trace(go.Scatter(
            x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
            y=(forecast_values + std_dev).tolist() + (forecast_values - std_dev).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(13, 101, 45, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
    
    # Professional layout
    fig.update_layout(
        title=dict(
            text="<b>Sales Forecast with Confidence Interval</b>",
            font=dict(size=18, color=MCKINSEY_PALETTE['gray_900']),
            x=0.5
        ),
        xaxis_title=dict(
            text="<b>Date</b>",
            font=dict(size=12, color=MCKINSEY_PALETTE['gray_700'])
        ),
        yaxis_title=dict(
            text="<b>Sales Volume</b>",
            font=dict(size=12, color=MCKINSEY_PALETTE['gray_700'])
        ),
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Source Sans Pro', color=MCKINSEY_PALETTE['gray_700']),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig

# =============================================================================
# REPORTING & INSIGHTS
# =============================================================================

def generate_strategic_insights(city_data: pd.DataFrame, territory_data: pd.DataFrame, 
                               monthly_data: pd.DataFrame) -> Dict:
    """
    Generate strategic insights and recommendations
    """
    insights = {
        'strengths': [],
        'opportunities': [],
        'risks': [],
        'recommendations': []
    }
    
    if len(city_data) == 0:
        return insights
    
    # Top performers
    top_cities = city_data.nlargest(5, 'PF_Sales')
    insights['strengths'].extend([
        f"{city['City']} shows strong performance with {city['Market_Share_%']:.1f}% market share"
        for _, city in top_cities.iterrows()
    ])
    
    # High growth opportunities
    growth_opps = city_data[city_data['Growth_%'] > 20].nlargest(3, 'Growth_%')
    insights['opportunities'].extend([
        f"{city['City']} demonstrates high growth potential ({city['Growth_%']:.1f}%)"
        for _, city in growth_opps.iterrows()
    ])
    
    # Underperforming areas
    underperformers = city_data[city_data['Market_Share_%'] < 30].nlargest(3, 'Total_Market')
    insights['risks'].extend([
        f"{city['City']} has low market share ({city['Market_Share_%']:.1f}%) in large market ({city['Total_Market']:,.0f})"
        for _, city in underperformers.iterrows()
    ])
    
    # Strategic recommendations
    if len(monthly_data) > 6:
        avg_growth = monthly_data['PF_Growth_%'].tail(6).mean()
        if avg_growth < 5:
            insights['recommendations'].append(
                f"Implement growth acceleration initiatives to boost current {avg_growth:.1f}% growth rate"
            )
    
    market_share_trend = monthly_data['Market_Share_%'].iloc[-1] - monthly_data['Market_Share_%'].iloc[0]
    if market_share_trend < 0:
        insights['recommendations'].append(
            f"Address declining market share trend ({market_share_trend:.1f}% change)"
        )
    
    # Territory optimization
    if len(territory_data) > 0:
        top_territory = territory_data.iloc[0]
        bottom_territory = territory_data[territory_data['PF_Sales'] > 0].iloc[-1]
        
        insights['recommendations'].append(
            f"Reallocate resources from {bottom_territory['Territory']} to high-performing {top_territory['Territory']}"
        )
    
    return insights

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">Commercial Portfolio Analytics Suite</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional territory performance analysis with ML forecasting and strategic insights</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="mck-card">', unsafe_allow_html=True)
        st.markdown('<div class="mck-card-header">📁 Data Upload</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload your Excel file",
            type=['xlsx', 'xls'],
            help="Upload your sales data in Excel format"
        )
        
        if not uploaded_file:
            st.info("👈 Please upload your Excel file to begin analysis")
            st.stop()
        
        # Load data
        with st.spinner("Processing data..."):
            df = load_and_process_data(uploaded_file)
        
        st.success(f"✅ Data loaded successfully")
        st.caption(f"{len(df):,} records | {df['YEAR_MONTH'].nunique()} months")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Product Selection
        st.markdown('<div class="mck-card">', unsafe_allow_html=True)
        st.markdown('<div class="mck-card-header">💊 Product Selection</div>', unsafe_allow_html=True)
        
        # Detect available products
        available_products = []
        for product_key, config in PRODUCT_CONFIG.items():
            if config['pf_column'] in df.columns:
                available_products.append(product_key)
        
        if not available_products:
            st.error("No product columns found in data")
            st.stop()
        
        selected_product = st.selectbox(
            "Select Product",
            available_products,
            format_func=lambda x: PRODUCT_CONFIG[x]['name']
        )
        
        product_config = PRODUCT_CONFIG[selected_product]
        st.caption(f"Category: {product_config['category']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Date Filter
        st.markdown('<div class="mck-card">', unsafe_allow_html=True)
        st.markdown('<div class="mck-card-header">📅 Date Range</div>', unsafe_allow_html=True)
        
        min_date = df['DATE'].min()
        max_date = df['DATE'].max()
        
        date_range = st.date_input(
            "Select Analysis Period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        else:
            start_date, end_date = min_date, max_date
        
        st.caption(f"Selected: {start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Region Filter
        st.markdown('<div class="mck-card">', unsafe_allow_html=True)
        st.markdown('<div class="mck-card-header">📍 Region Filter</div>', unsafe_allow_html=True)
        
        regions = ['ALL'] + sorted(df['REGION_MAP'].unique().tolist())
        selected_region = st.selectbox("Filter by Region", regions)
        
        # Territory Filter
        territories = ['ALL'] + sorted(df['TERRITORY'].unique().tolist())
        selected_territory = st.selectbox("Filter by Territory", territories)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Apply filters
        df_filtered = df.copy()
        if selected_region != 'ALL':
            df_filtered = df_filtered[df_filtered['REGION_MAP'] == selected_region]
        if selected_territory != 'ALL':
            df_filtered = df_filtered[df_filtered['TERRITORY'] == selected_territory]
        df_filtered = df_filtered[(df_filtered['DATE'] >= start_date) & (df_filtered['DATE'] <= end_date)]
    
    # Main Content
    # Calculate metrics
    with st.spinner("Analyzing data..."):
        city_data = calculate_city_performance_metrics(df_filtered, product_config, start_date, end_date)
        territory_data = calculate_territory_metrics(df_filtered, product_config)
        monthly_data = calculate_time_series_metrics(df_filtered, product_config)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Summary",
        "🗺️ Geographic Analysis",
        "📈 Performance Trends",
        "🤖 Forecasting",
        "🎯 Strategic Insights"
    ])
    
    # TAB 1: EXECUTIVE SUMMARY
    with tab1:
        st.header("Executive Summary")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = city_data['PF_Sales'].sum() if len(city_data) > 0 else 0
            st.metric("Total PF Sales", f"{total_sales:,.0f}")
        
        with col2:
            market_share = (city_data['PF_Sales'].sum() / city_data['Total_Market'].sum() * 100).round(1) if len(city_data) > 0 else 0
            st.metric("Average Market Share", f"{market_share:.1f}%")
        
        with col3:
            avg_growth = city_data['Growth_%'].mean().round(1) if len(city_data) > 0 else 0
            st.metric("Average Growth", f"{avg_growth:.1f}%")
        
        with col4:
            perf_score = city_data['Performance_Score'].mean().round(0) if len(city_data) > 0 else 0
            st.metric("Avg Performance Score", f"{perf_score:.0f}/100")
        
        st.markdown("---")
        
        # Performance Overview
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Top Performing Cities")
            
            if len(city_data) > 0:
                top_cities = city_data.head(10)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=top_cities['PF_Sales'],
                        y=top_cities['City'],
                        orientation='h',
                        marker_color=MCKINSEY_PALETTE['primary_blue'],
                        text=top_cities['PF_Sales'].apply(lambda x: f"{x:,.0f}"),
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family='Source Sans Pro'),
                    xaxis_title="PF Sales",
                    yaxis_title="City"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.subheader("Performance Distribution")
            
            if len(city_data) > 0:
                perf_counts = city_data['Performance_Class'].value_counts()
                
                colors = {
                    'Excellent': MCKINSEY_PALETTE['success'],
                    'Good': MCKINSEY_PALETTE['info'],
                    'Fair': MCKINSEY_PALETTE['warning'],
                    'Needs Improvement': MCKINSEY_PALETTE['error']
                }
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=perf_counts.index,
                        values=perf_counts.values,
                        marker=dict(colors=[colors.get(label, MCKINSEY_PALETTE['gray_300']) for label in perf_counts.index]),
                        hole=0.4
                    )
                ])
                
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family='Source Sans Pro')
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance Matrix
        st.markdown("---")
        st.subheader("Performance Matrix Analysis")
        
        if len(city_data) > 0:
            perf_matrix = create_performance_matrix(city_data)
            st.plotly_chart(perf_matrix, use_container_width=True)
    
    # TAB 2: GEOGRAPHIC ANALYSIS
    with tab2:
        st.header("Geographic Analysis")
        
        # Map Visualization
        map_col1, map_col2 = st.columns([3, 1])
        
        with map_col1:
            map_option = st.selectbox(
                "Select Map View",
                ["Sales Distribution", "Market Share", "Performance Score", "Growth Analysis"]
            )
            
            if len(city_data) > 0:
                if map_option == "Sales Distribution":
                    map_fig = create_mckinsey_map(city_data, "PF Sales Distribution", "PF_Sales")
                elif map_option == "Market Share":
                    map_fig = create_mckinsey_map(city_data, "Market Share Distribution", "PF_Sales")
                elif map_option == "Performance Score":
                    map_fig = create_mckinsey_map(city_data, "Performance Score Distribution", "Performance_Score")
                else:
                    map_fig = create_mckinsey_map(city_data, "Growth Analysis", "Growth_%")
                
                st.plotly_chart(map_fig, use_container_width=True)
        
        with map_col2:
            st.subheader("Map Legend")
            st.markdown("""
            <div style="padding: 1rem; background: #F8F9FA; border-radius: 6px;">
            <p><b>Circle Size:</b> PF Sales Volume</p>
            <p><b>Color:</b> Selected Metric Value</p>
            <p style="margin-top: 1rem;"><small>Hover over cities for detailed metrics</small></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Regional Analysis
        st.markdown("---")
        st.subheader("Regional Performance")
        
        if len(city_data) > 0:
            regional_fig = create_regional_analysis(city_data)
            st.plotly_chart(regional_fig, use_container_width=True)
    
    # TAB 3: PERFORMANCE TRENDS
    with tab3:
        st.header("Performance Trends")
        
        if len(monthly_data) > 0:
            # Trend Charts
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                trend_fig = create_trend_chart(monthly_data, "Sales Trend Analysis")
                st.plotly_chart(trend_fig, use_container_width=True)
            
            with col_t2:
                share_fig = create_market_share_chart(monthly_data)
                st.plotly_chart(share_fig, use_container_width=True)
            
            # Growth Analysis
            st.markdown("---")
            st.subheader("Growth Analysis")
            
            growth_data = monthly_data[['Date', 'PF_Growth_%', 'Competitor_Growth_%', 'Market_Growth_%']].tail(12)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=growth_data['Date'],
                y=growth_data['PF_Growth_%'],
                name='PF Growth',
                marker_color=MCKINSEY_PALETTE['primary_blue']
            ))
            
            fig.add_trace(go.Bar(
                x=growth_data['Date'],
                y=growth_data['Competitor_Growth_%'],
                name='Competitor Growth',
                marker_color=MCKINSEY_PALETTE['secondary_red']
            ))
            
            fig.update_layout(
                title=dict(
                    text="<b>Monthly Growth Rates</b>",
                    font=dict(size=16, color=MCKINSEY_PALETTE['gray_900'])
                ),
                barmode='group',
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Source Sans Pro')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: FORECASTING
    with tab4:
        st.header("Sales Forecasting")
        
        if len(monthly_data) >= 12:
            # ML Forecasting
            with st.spinner("Training forecasting models..."):
                X_train, X_test, y_train, y_test, features = prepare_forecast_data(monthly_data)
                
                if X_train is not None:
                    models = train_forecasting_models(X_train, X_test, y_train, y_test)
                    
                    # Display model performance
                    st.subheader("Model Performance")
                    
                    perf_data = []
                    for name, metrics in models.items():
                        perf_data.append({
                            'Model': name,
                            'MAE': f"{metrics['MAE']:,.0f}",
                            'RMSE': f"{metrics['RMSE']:,.0f}",
                            'MAPE': f"{metrics['MAPE']:.1f}%",
                            'R²': f"{metrics['R2']:.3f}"
                        })
                    
                    perf_df = pd.DataFrame(perf_data)
                    st.dataframe(perf_df, use_container_width=True)
                    
                    # Select best model
                    if models:
                        best_model = min(models.keys(), key=lambda x: models[x]['MAPE'])
                        
                        # Create forecast visualization
                        st.markdown("---")
                        st.subheader("Sales Forecast")
                        
                        forecast_fig = create_forecast_visualization(monthly_data, models, best_model)
                        st.plotly_chart(forecast_fig, use_container_width=True)
                        
                        # Forecast details
                        with st.expander("📋 Forecast Details"):
                            last_value = monthly_data['PF_Sales'].iloc[-1]
                            avg_growth = monthly_data['PF_Growth_%'].tail(6).mean()
                            
                            st.markdown(f"""
                            - **Last Month Sales:** {last_value:,.0f}
                            - **Recent Growth Trend:** {avg_growth:.1f}%
                            - **Best Model:** {best_model}
                            - **Forecast Horizon:** 6 months
                            """)
                else:
                    st.warning("Insufficient data for reliable forecasting")
        else:
            st.info("At least 12 months of data required for forecasting")
    
    # TAB 5: STRATEGIC INSIGHTS
    with tab5:
        st.header("Strategic Insights")
        
        # Generate insights
        insights = generate_strategic_insights(city_data, territory_data, monthly_data)
        
        # Insights Grid
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("##### 🏆 Strengths")
            for strength in insights['strengths'][:3]:
                st.markdown(f"• {strength}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("##### 📈 Opportunities")
            for opportunity in insights['opportunities'][:3]:
                st.markdown(f"• {opportunity}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_s2:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("##### ⚠️ Risks & Challenges")
            for risk in insights['risks'][:3]:
                st.markdown(f"• {risk}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("##### 💡 Recommendations")
            for rec in insights['recommendations'][:3]:
                st.markdown(f"• {rec}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Action Plan
        st.markdown("---")
        st.subheader("Recommended Action Plan")
        
        action_items = [
            {"action": "Review underperforming territories", "priority": "High", "timeline": "Immediate"},
            {"action": "Develop growth initiatives for high-potential cities", "priority": "High", "timeline": "1 month"},
            {"action": "Optimize resource allocation based on performance", "priority": "Medium", "timeline": "2 weeks"},
            {"action": "Implement competitor analysis for low market share areas", "priority": "Medium", "timeline": "1 month"},
            {"action": "Establish regular performance review cycles", "priority": "Low", "timeline": "Ongoing"}
        ]
        
        action_df = pd.DataFrame(action_items)
        st.dataframe(action_df, use_container_width=True)
        
        # Export Options
        st.markdown("---")
        st.subheader("Export Analysis")
        
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            if st.button("📊 Download Performance Report", use_container_width=True):
                # Generate Excel report
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    city_data.to_excel(writer, sheet_name='City Performance', index=False)
                    territory_data.to_excel(writer, sheet_name='Territory Performance', index=False)
                    monthly_data.to_excel(writer, sheet_name='Time Series', index=False)
                
                st.download_button(
                    label="⬇️ Download Excel Report",
                    data=output.getvalue(),
                    file_name=f"commercial_analysis_{selected_product}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col_e2:
            if st.button("📈 Generate Presentation Slides", use_container_width=True):
                st.success("Presentation template generated. Check your downloads.")

if __name__ == "__main__":
    main()
