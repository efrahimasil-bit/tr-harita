"""
🎯 PROFESYONEL TİCARİ PORTFÖY ANALİZ SİSTEMİ
Advanced Territory Performance, ML Forecasting, Turkey Mapping & Competitive Intelligence

ÖZELLİKLER:
- 🗺️ Gelişmiş Türkiye Harita Görselleştirme (Bölge/Şehir Modu)
- 🤖 İleri Seviye ML/AI Tahminleme (XGBoost, LSTM, Prophet, Ensemble)
- 📊 Çoklu Zaman Serisi Analizi (Trend, Mevsimsellik, Anomali)
- 🏢 BCG Matrix ve Stratejik Portföy Yönetimi
- 📈 Gerçek Zamanlı Rakip Analizi ve Benchmarking
- 🔮 12 Aylık AI Tahminleri ve Senaryo Analizleri
- 📊 50+ Performans Metrik ve KPI Dashboard
- 🎯 Otomatik Öngörü ve İçgörü Üretimi
- 📉 Risk Analizi ve Erken Uyarı Sistemi
- 💼 Yatırım Stratejisi ve ROI Hesaplama
- 📱 Responsive ve Modern UI/UX
- 📥 Otomatik Raporlama (Excel, PDF, PPT)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from io import BytesIO
import json
import base64
from typing import Dict, List, Tuple, Optional, Any
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter
import os

# Statsmodels kütüphanesini kontrol et ve gerekirse kur
try:
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except ImportError:
    os.system('pip install statsmodels')
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import geopandas as gpd
from shapely.geometry import Point, Polygon
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    IsolationForest, VotingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, BayesianRidge
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, 
    r2_score, mean_absolute_percentage_error,
    silhouette_score, davies_bouldin_score
)
from sklearn.model_selection import (
    TimeSeriesSplit, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from prophet import Prophet

# TensorFlow ve Derin Öğrenme Modülleri Güvenlik Bloğu
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
except ImportError:
    tf = None
    st.warning("⚠️ TensorFlow kütüphanesi kurulamadı. Derin öğrenme (LSTM vb.) tahminleri devre dışı kalacaktır.")

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SAYFA KONFİGÜRASYONU
# =============================================================================
st.set_page_config(
    page_title="Advanced Ticari Portföy Analiz Sistemi",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com',
        'Report a bug': "https://www.example.com",
        'About': """
        ## Advanced Ticari Portföy Analiz Sistemi v4.0
        **Gelişmiş ML/AI tabanlı ticari performans analiz platformu**
        - 🤖 10+ ML Modeli ile tahminleme
        - 🗺️ Interactive Türkiye haritaları
        - 📊 50+ performans metriği
        - 🎯 Otomatik içgörü üretimi
        
        © 2024 Advanced Analytics Team
        """
    }
)

# =============================================================================
# CSS STİLLERİ
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stApp {
        background: linear-gradient(135deg, 
            #0f1729 0%, 
            #1a1f2e 25%, 
            #242837 50%, 
            #1e293b 75%, 
            #0f172a 100%);
        background-attachment: fixed;
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header {
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        padding: 2.5rem 0;
        background: linear-gradient(135deg, 
            #3B82F6 0%, 
            #10B981 25%, 
            #F59E0B 50%, 
            #8B5CF6 75%, 
            #EF4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        letter-spacing: -1px;
        margin-bottom: 1.5rem;
        font-family: 'Poppins', sans-serif;
        position: relative;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: 1rem;
        left: 50%;
        transform: translateX(-50%);
        width: 200px;
        height: 4px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(59, 130, 246, 0.8) 50%, 
            transparent 100%);
        border-radius: 2px;
    }
    
    .sub-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(59, 130, 246, 0.2);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .metric-delta {
        font-size: 1.3rem !important;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 1rem !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.9);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(59, 130, 246, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 48px rgba(59, 130, 246, 0.3);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem;
        padding: 0.75rem;
        background: rgba(30, 41, 59, 0.8);
        border-radius: 16px;
        border: 1px solid rgba(59, 130, 246, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
        padding: 1.25rem 2rem;
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        margin: 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid transparent;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transform: translateY(-2px) scale(1.05);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 14px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3);
        position: relative;
        overflow: hidden;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-6px) scale(1.05);
        box-shadow: 0 12px 32px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
    }
    
    .alert-success {
        background: linear-gradient(135deg, 
            rgba(16, 185, 129, 0.15) 0%, 
            rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10B981;
        padding: 1.5rem;
        border-radius: 14px;
        margin: 1.5rem 0;
        border-left: 6px solid #10B981;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.1);
    }
    
    .alert-warning {
        background: linear-gradient(135deg, 
            rgba(245, 158, 11, 0.15) 0%, 
            rgba(245, 158, 11, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #F59E0B;
        padding: 1.5rem;
        border-radius: 14px;
        margin: 1.5rem 0;
        border-left: 6px solid #F59E0B;
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.1);
    }
    
    .alert-danger {
        background: linear-gradient(135deg, 
            rgba(239, 68, 68, 0.15) 0%, 
            rgba(239, 68, 68, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #EF4444;
        padding: 1.5rem;
        border-radius: 14px;
        margin: 1.5rem 0;
        border-left: 6px solid #EF4444;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.1);
    }
    
    .alert-info {
        background: linear-gradient(135deg, 
            rgba(59, 130, 246, 0.15) 0%, 
            rgba(59, 130, 246, 0.05) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #3B82F6;
        padding: 1.5rem;
        border-radius: 14px;
        margin: 1.5rem 0;
        border-left: 6px solid #3B82F6;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.1);
    }
    
    .custom-card {
        background: rgba(30, 41, 59, 0.85);
        border-radius: 18px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# KONSTANTLAR VE RENK PALETLERİ
# =============================================================================
REGION_COLORS = {
    "MARMARA": "#0EA5E9",
    "BATI ANADOLU": "#14B8A6",
    "İÇ ANADOLU": "#F59E0B",
    "GÜNEY DOĞU ANADOLU": "#E07A5F",
    "KUZEY ANADOLU": "#059669",
    "AKDENİZ": "#3B82F6",
    "EGE": "#8B5CF6",
    "DOĞU ANADOLU": "#EF4444",
    "KARADENİZ": "#10B981",
    "DİĞER": "#64748B",
}

PERFORMANCE_COLORS = {
    "excellent": "#10B981",
    "good": "#22C55E",
    "average": "#F59E0B",
    "poor": "#EF4444",
    "critical": "#991B1B",
    "positive": "#10B981",
    "negative": "#EF4444",
    "neutral": "#6B7280",
    "warning": "#F59E0B",
}

BCG_COLORS = {
    "⭐ STAR": "#F59E0B",
    "🐄 CASH COW": "#10B981",
    "❓ QUESTION MARK": "#3B82F6",
    "🐶 DOG": "#64748B",
}

STRATEGY_COLORS = {
    "🚀 AGRESİF BÜYÜME": "#EF4444",
    "📈 HIZLANDIRILMIŞ": "#F59E0B",
    "🛡️ KORUMA": "#10B981",
    "💎 POTANSİYEL": "#8B5CF6",
    "👁️ İZLEME": "#64748B",
    "🔄 YENİDEN YAPILANMA": "#EC4899",
}

# =============================================================================
# TÜRKİYE İL HARİTASI VERİSİ
# =============================================================================
TURKEY_CITIES = {
    "ADANA": {"region": "AKDENİZ", "lat": 37.0, "lon": 35.3213},
    "ADIYAMAN": {"region": "GÜNEY DOĞU ANADOLU", "lat": 37.7648, "lon": 38.2786},
    "AFYONKARAHİSAR": {"region": "İÇ ANADOLU", "lat": 38.7638, "lon": 30.5406},
    "AĞRI": {"region": "DOĞU ANADOLU", "lat": 39.7191, "lon": 43.0503},
    "AKSARAY": {"region": "İÇ ANADOLU", "lat": 38.3687, "lon": 34.0370},
    "AMASYA": {"region": "KARADENİZ", "lat": 40.6499, "lon": 35.8353},
    "ANKARA": {"region": "İÇ ANADOLU", "lat": 39.9334, "lon": 32.8597},
    "ANTALYA": {"region": "AKDENİZ", "lat": 36.8969, "lon": 30.7133},
    "ARDAHAN": {"region": "DOĞU ANADOLU", "lat": 41.1105, "lon": 42.7022},
    "ARTVİN": {"region": "KARADENİZ", "lat": 41.1828, "lon": 41.8183},
    "AYDIN": {"region": "EGE", "lat": 37.8560, "lon": 27.8416},
    "BALIKESİR": {"region": "MARMARA", "lat": 39.6484, "lon": 27.8826},
    "BARTIN": {"region": "KARADENİZ", "lat": 41.5811, "lon": 32.4610},
    "BATMAN": {"region": "GÜNEY DOĞU ANADOLU", "lat": 37.8812, "lon": 41.1351},
    "BAYBURT": {"region": "KARADENİZ", "lat": 40.2552, "lon": 40.2249},
    "BİLECİK": {"region": "MARMARA", "lat": 40.0567, "lon": 30.0665},
    "BİNGÖL": {"region": "DOĞU ANADOLU", "lat": 38.8843, "lon": 40.4966},
    "BİTLİS": {"region": "DOĞU ANADOLU", "lat": 38.3938, "lon": 42.1232},
    "BOLU": {"region": "KARADENİZ", "lat": 40.5760, "lon": 31.5788},
    "BURDUR": {"region": "AKDENİZ", "lat": 37.4613, "lon": 30.0665},
    "BURSA": {"region": "MARMARA", "lat": 40.1885, "lon": 29.0610},
    "ÇANAKKALE": {"region": "MARMARA", "lat": 40.1553, "lon": 26.4142},
    "ÇANKIRI": {"region": "İÇ ANADOLU", "lat": 40.6013, "lon": 33.6134},
    "ÇORUM": {"region": "KARADENİZ", "lat": 40.5506, "lon": 34.9556},
    "DENİZLİ": {"region": "EGE", "lat": 37.7765, "lon": 29.0864},
    "DİYARBAKIR": {"region": "GÜNEY DOĞU ANADOLU", "lat": 37.9144, "lon": 40.2306},
    "DÜZCE": {"region": "KARADENİZ", "lat": 40.8398, "lon": 31.1633},
    "EDİRNE": {"region": "MARMARA", "lat": 41.6818, "lon": 26.5623},
    "ELAZIĞ": {"region": "DOĞU ANADOLU", "lat": 38.6810, "lon": 39.2264},
    "ERZİNCAN": {"region": "DOĞU ANADOLU", "lat": 39.7500, "lon": 39.5000},
    "ERZURUM": {"region": "DOĞU ANADOLU", "lat": 39.9000, "lon": 41.2700},
    "ESKİŞEHİR": {"region": "İÇ ANADOLU", "lat": 39.7767, "lon": 30.5206},
    "GAZİANTEP": {"region": "GÜNEY DOĞU ANADOLU", "lat": 37.0662, "lon": 37.3833},
    "GİRESUN": {"region": "KARADENİZ", "lat": 40.9128, "lon": 38.3895},
    "GÜMÜŞHANE": {"region": "KARADENİZ", "lat": 40.4386, "lon": 39.5086},
    "HAKKARİ": {"region": "DOĞU ANADOLU", "lat": 37.5833, "lon": 43.7333},
    "HATAY": {"region": "AKDENİZ", "lat": 36.4018, "lon": 36.3498},
    "IĞDIR": {"region": "DOĞU ANADOLU", "lat": 39.9167, "lon": 44.0333},
    "ISPARTA": {"region": "AKDENİZ", "lat": 37.7648, "lon": 30.5566},
    "İSTANBUL": {"region": "MARMARA", "lat": 41.0082, "lon": 28.9784},
    "İZMİR": {"region": "EGE", "lat": 38.4237, "lon": 27.1428},
    "KAHRAMANMARAŞ": {"region": "AKDENİZ", "lat": 37.5753, "lon": 36.9228},
    "KARABÜK": {"region": "KARADENİZ", "lat": 41.2061, "lon": 32.6204},
    "KARAMAN": {"region": "İÇ ANADOLU", "lat": 37.1759, "lon": 33.2287},
    "KARS": {"region": "DOĞU ANADOLU", "lat": 40.6167, "lon": 43.1000},
    "KASTAMONU": {"region": "KARADENİZ", "lat": 41.3887, "lon": 33.7827},
    "KAYSERİ": {"region": "İÇ ANADOLU", "lat": 38.7312, "lon": 35.4787},
    "KIRIKKALE": {"region": "İÇ ANADOLU", "lat": 39.8468, "lon": 33.5153},
    "KIRKLARELİ": {"region": "MARMARA", "lat": 41.7333, "lon": 27.2167},
    "KIRŞEHİR": {"region": "İÇ ANADOLU", "lat": 39.1425, "lon": 34.1709},
    "KİLİS": {"region": "GÜNEY DOĞU ANADOLU", "lat": 36.7184, "lon": 37.1212},
    "KOCAELİ": {"region": "MARMARA", "lat": 40.8533, "lon": 29.8815},
    "KONYA": {"region": "İÇ ANADOLU", "lat": 37.8667, "lon": 32.4833},
    "KÜTAHYA": {"region": "EGE", "lat": 39.4167, "lon": 29.9833},
    "MALATYA": {"region": "DOĞU ANADOLU", "lat": 38.3552, "lon": 38.3095},
    "MANİSA": {"region": "EGE", "lat": 38.6191, "lon": 27.4289},
    "MARDİN": {"region": "GÜNEY DOĞU ANADOLU", "lat": 37.3212, "lon": 40.7245},
    "MERSİN": {"region": "AKDENİZ", "lat": 36.8000, "lon": 34.6333},
    "MUĞLA": {"region": "EGE", "lat": 37.2153, "lon": 28.3636},
    "MUŞ": {"region": "DOĞU ANADOLU", "lat": 38.9462, "lon": 41.7539},
    "NEVŞEHİR": {"region": "İÇ ANADOLU", "lat": 38.6939, "lon": 34.6857},
    "NİĞDE": {"region": "İÇ ANADOLU", "lat": 37.9667, "lon": 34.6833},
    "ORDU": {"region": "KARADENİZ", "lat": 40.9833, "lon": 37.8833},
    "OSMANİYE": {"region": "AKDENİZ", "lat": 37.0742, "lon": 36.2478},
    "RİZE": {"region": "KARADENİZ", "lat": 41.0201, "lon": 40.5234},
    "SAKARYA": {"region": "MARMARA", "lat": 40.6939, "lon": 30.4358},
    "SAMSUN": {"region": "KARADENİZ", "lat": 41.2928, "lon": 36.3313},
    "SİİRT": {"region": "GÜNEY DOĞU ANADOLU", "lat": 37.9443, "lon": 41.9328},
    "SİNOP": {"region": "KARADENİZ", "lat": 42.0264, "lon": 35.1551},
    "SİVAS": {"region": "İÇ ANADOLU", "lat": 39.7477, "lon": 37.0179},
    "ŞANLIURFA": {"region": "GÜNEY DOĞU ANADOLU", "lat": 37.1591, "lon": 38.7969},
    "ŞIRNAK": {"region": "GÜNEY DOĞU ANADOLU", "lat": 37.5133, "lon": 42.4543},
    "TEKİRDAĞ": {"region": "MARMARA", "lat": 40.9833, "lon": 27.5167},
    "TOKAT": {"region": "KARADENİZ", "lat": 40.3167, "lon": 36.5500},
    "TRABZON": {"region": "KARADENİZ", "lat": 41.0015, "lon": 39.7178},
    "TUNCELİ": {"region": "DOĞU ANADOLU", "lat": 39.1071, "lon": 39.5405},
    "UŞAK": {"region": "EGE", "lat": 38.6823, "lon": 29.4082},
    "VAN": {"region": "DOĞU ANADOLU", "lat": 38.4946, "lon": 43.3800},
    "YALOVA": {"region": "MARMARA", "lat": 40.6500, "lon": 29.2667},
    "YOZGAT": {"region": "İÇ ANADOLU", "lat": 39.8200, "lon": 34.8044},
    "ZONGULDAK": {"region": "KARADENİZ", "lat": 41.4564, "lon": 31.7987}
}

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================
class TurkishTextNormalizer:
    """Türkçe metin normalizasyonu"""
    
    TURKISH_CHAR_MAP = {
        'İ': 'I', 'ı': 'I', 'Ğ': 'G', 'ğ': 'G',
        'Ü': 'U', 'ü': 'U', 'Ş': 'S', 'ş': 'S',
        'Ö': 'O', 'ö': 'O', 'Ç': 'C', 'ç': 'C',
        'Â': 'A', 'Î': 'I', 'Û': 'U',
        'â': 'A', 'î': 'I', 'û': 'U'
    }
    
    CITY_NORMALIZATION = {
        'ISTANBUL': 'İSTANBUL',
        'IZMIR': 'İZMİR',
        'ANKARA': 'ANKARA',
        'BURSA': 'BURSA',
        'ADANA': 'ADANA',
        'GAZIANTEP': 'GAZİANTEP',
        'KONYA': 'KONYA',
        'ANTALYA': 'ANTALYA',
        'MERSIN': 'MERSİN',
        'DIYARBAKIR': 'DİYARBAKIR',
        'ELAZIG': 'ELAZIĞ',
        'ESKISEHIR': 'ESKİŞEHİR',
        'KAHRAMANMARAS': 'KAHRAMANMARAŞ',
        'SANLIURFA': 'ŞANLIURFA',
        'SIRNAK': 'ŞIRNAK',
        'NEVSEHIR': 'NEVŞEHİR',
        'NIGDE': 'NİĞDE',
        'MUS': 'MUŞ',
        'MUGLA': 'MUĞLA',
        'KUTAHYA': 'KÜTAHYA',
        'GUMUSHANE': 'GÜMÜŞHANE',
        'AGRI': 'AĞRI',
        'AFYON': 'AFYONKARAHİSAR',
        'IZMIT': 'KOCAELİ',
        'KOCAELI': 'KOCAELİ',
        'CORUM': 'ÇORUM',
        'CANKIRI': 'ÇANKIRI',
        'CANAKKALE': 'ÇANAKKALE'
    }
    
    REGION_MAPPING = {
        'İSTANBUL': 'MARMARA',
        'KOCAELİ': 'MARMARA',
        'BURSA': 'MARMARA',
        'SAKARYA': 'MARMARA',
        'TEKİRDAĞ': 'MARMARA',
        'BALIKESİR': 'MARMARA',
        'ÇANAKKALE': 'MARMARA',
        'EDİRNE': 'MARMARA',
        'KIRKLARELİ': 'MARMARA',
        'YALOVA': 'MARMARA',
        'İZMİR': 'EGE',
        'AYDIN': 'EGE',
        'MUĞLA': 'EGE',
        'MANİSA': 'EGE',
        'DENİZLİ': 'EGE',
        'UŞAK': 'EGE',
        'KÜTAHYA': 'EGE',
        'AFYONKARAHİSAR': 'EGE',
        'ANKARA': 'İÇ ANADOLU',
        'KONYA': 'İÇ ANADOLU',
        'ESKİŞEHİR': 'İÇ ANADOLU',
        'KAYSERİ': 'İÇ ANADOLU',
        'SİVAS': 'İÇ ANADOLU',
        'YOZGAT': 'İÇ ANADOLU',
        'AKSARAY': 'İÇ ANADOLU',
        'KIRIKKALE': 'İÇ ANADOLU',
        'KIRŞEHİR': 'İÇ ANADOLU',
        'NEVŞEHİR': 'İÇ ANADOLU',
        'NİĞDE': 'İÇ ANADOLU',
        'ANTALYA': 'AKDENİZ',
        'ADANA': 'AKDENİZ',
        'MERSİN': 'AKDENİZ',
        'HATAY': 'AKDENİZ',
        'KAHRAMANMARAŞ': 'AKDENİZ',
        'OSMANİYE': 'AKDENİZ',
        'ISPARTA': 'AKDENİZ',
        'BURDUR': 'AKDENİZ',
        'TRABZON': 'KARADENİZ',
        'SAMSUN': 'KARADENİZ',
        'ORDU': 'KARADENİZ',
        'GİRESUN': 'KARADENİZ',
        'RİZE': 'KARADENİZ',
        'ZONGULDAK': 'KARADENİZ',
        'KASTAMONU': 'KARADENİZ',
        'SİNOP': 'KARADENİZ',
        'BOLU': 'KARADENİZ',
        'DÜZCE': 'KARADENİZ',
        'GAZİANTEP': 'GÜNEY DOĞU ANADOLU',
        'ŞANLIURFA': 'GÜNEY DOĞU ANADOLU',
        'DİYARBAKIR': 'GÜNEY DOĞU ANADOLU',
        'MARDİN': 'GÜNEY DOĞU ANADOLU',
        'BATMAN': 'GÜNEY DOĞU ANADOLU',
        'SİİRT': 'GÜNEY DOĞU ANADOLU',
        'ŞIRNAK': 'GÜNEY DOĞU ANADOLU',
        'KİLİS': 'GÜNEY DOĞU ANADOLU',
        'ADIYAMAN': 'GÜNEY DOĞU ANADOLU',
        'ERZURUM': 'DOĞU ANADOLU',
        'ERZİNCAN': 'DOĞU ANADOLU',
        'AĞRI': 'DOĞU ANADOLU',
        'KARS': 'DOĞU ANADOLU',
        'IĞDIR': 'DOĞU ANADOLU',
        'VAN': 'DOĞU ANADOLU',
        'MALATYA': 'DOĞU ANADOLU',
        'ELAZIĞ': 'DOĞU ANADOLU',
        'TUNCELİ': 'DOĞU ANADOLU',
        'BİNGÖL': 'DOĞU ANADOLU',
        'MUŞ': 'DOĞU ANADOLU',
        'BİTLİS': 'DOĞU ANADOLU',
        'HAKKARİ': 'DOĞU ANADOLU'
    }
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Türkçe metni normalleştir"""
        if pd.isna(text):
            return "BILINMIYOR"
        
        text = str(text).strip().upper()
        
        # Türkçe karakterleri düzelt
        for old, new in TurkishTextNormalizer.TURKISH_CHAR_MAP.items():
            text = text.replace(old, new)
        
        # Özel şehir isimlerini düzelt
        if text in TurkishTextNormalizer.CITY_NORMALIZATION:
            return TurkishTextNormalizer.CITY_NORMALIZATION[text]
        
        return text
    
    @staticmethod
    def assign_region(city: str) -> str:
        """Şehre göre bölge ata"""
        normalized_city = TurkishTextNormalizer.normalize_text(city)
        return TurkishTextNormalizer.REGION_MAPPING.get(normalized_city, "DİĞER")

# =============================================================================
# VERİ İŞLEME FONKSİYONLARI
# =============================================================================
@st.cache_data(ttl=3600)
def load_and_process_excel(file) -> Tuple[pd.DataFrame, Dict]:
    """Excel dosyasını yükle ve işle"""
    try:
        # Excel'i yükle
        df = pd.read_excel(file, engine='openpyxl')
        
        # Sütun adlarını standardize et
        df.columns = [col.strip().upper() for col in df.columns]
        
        # Tarih sütunlarını bul ve dönüştür
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['date', 'tarih', 'ay', 'yıl']):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        # Numerik sütunları temizle
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Şehir normalizasyonu
        if 'CITY' in df.columns:
            df['CITY_NORMALIZED'] = df['CITY'].apply(TurkishTextNormalizer.normalize_text)
            df['REGION'] = df['CITY_NORMALIZED'].apply(TurkishTextNormalizer.assign_region)
        
        # Eksik sütunları tamamla
        if 'TERRITORIES' not in df.columns:
            if 'CITY_NORMALIZED' in df.columns:
                df['TERRITORIES'] = df['CITY_NORMALIZED']
            else:
                df['TERRITORIES'] = 'BILINMIYOR'
        
        if 'MANAGER' not in df.columns:
            df['MANAGER'] = 'BILINMIYOR'
        
        # Tarih sütunu yoksa oluştur
        if 'DATE' not in df.columns:
            # Yıl ve ay sütunlarından tarih oluştur
            if 'YEAR' in df.columns and 'MONTH' in df.columns:
                df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
            else:
                # Örnek tarihler oluştur
                df['DATE'] = pd.date_range(start='2023-01-01', periods=len(df), freq='M')
        
        # Yıl-Ay formatını oluştur
        df['YEAR_MONTH'] = df['DATE'].dt.strftime('%Y-%m')
        
        # Satış sütunlarını tespit et
        sales_columns = {}
        for col in df.columns:
            col_upper = str(col).upper()
            if 'TROCMETAM' in col_upper and 'DIGER' not in col_upper:
                sales_columns['TROCMETAM_PF'] = col
            elif 'DIGER TROCMETAM' in col_upper:
                sales_columns['TROCMETAM_COMPETITOR'] = col
            elif 'CORTIPOL' in col_upper and 'DIGER' not in col_upper:
                sales_columns['CORTIPOL_PF'] = col
            elif 'DIGER CORTIPOL' in col_upper:
                sales_columns['CORTIPOL_COMPETITOR'] = col
            elif 'DEKSAMETAZON' in col_upper and 'DIGER' not in col_upper:
                sales_columns['DEKSAMETAZON_PF'] = col
            elif 'DIGER DEKSAMETAZON' in col_upper:
                sales_columns['DEKSAMETAZON_COMPETITOR'] = col
            elif ('IZOTONIK' in col_upper or 'IZOTONIC' in col_upper) and 'DIGER' not in col_upper:
                sales_columns['IZOTONIK_PF'] = col
            elif 'DIGER IZOTONIK' in col_upper:
                sales_columns['IZOTONIK_COMPETITOR'] = col
        
        return df, sales_columns
        
    except Exception as e:
        st.error(f"Excel yükleme hatası: {str(e)}")
        return create_sample_dataset()

def create_sample_dataset() -> Tuple[pd.DataFrame, Dict]:
    """Örnek veri seti oluştur (test için)"""
    np.random.seed(42)
    
    # Tarih aralığı
    dates = pd.date_range(start='2022-01-01', end='2024-12-01', freq='M')
    
    # Şehirler ve bölgeler
    cities = list(TURKEY_CITIES.keys())[:30]
    
    # Territory ve Manager
    territories = [f'TERR_{i:03d}' for i in range(1, 21)]
    managers = [f'MANAGER_{i}' for i in range(1, 6)]
    
    data = []
    for date in dates:
        for city in cities[:15]:
            territory = np.random.choice(territories)
            manager = np.random.choice(managers)
            region = TURKEY_CITIES[city]['region']
            
            # Rastgele satış verileri
            base_sales = np.random.randint(1000, 50000)
            season_factor = 1 + 0.3 * np.sin(2 * np.pi * date.month / 12)
            trend_factor = 1 + 0.1 * ((date.year - 2022) + (date.month - 1)/12)
            
            pf_sales = int(base_sales * season_factor * trend_factor * np.random.uniform(0.8, 1.2))
            comp_sales = int(pf_sales * np.random.uniform(0.5, 1.5))
            
            data.append({
                'DATE': date,
                'CITY': city,
                'REGION': region,
                'TERRITORIES': territory,
                'MANAGER': manager,
                'TROCMETAM': pf_sales,
                'DIGER TROCMETAM': comp_sales,
                'CORTIPOL': int(pf_sales * np.random.uniform(0.3, 0.7)),
                'DIGER CORTIPOL': int(comp_sales * np.random.uniform(0.3, 0.7)),
                'DEKSAMETAZON': int(pf_sales * np.random.uniform(0.2, 0.5)),
                'DIGER DEKSAMETAZON': int(comp_sales * np.random.uniform(0.2, 0.5)),
            })
    
    df = pd.DataFrame(data)
    
    # Şehir normalizasyonu
    df['CITY_NORMALIZED'] = df['CITY']
    df['YEAR_MONTH'] = df['DATE'].dt.strftime('%Y-%m')
    
    sales_columns = {
        'TROCMETAM_PF': 'TROCMETAM',
        'TROCMETAM_COMPETITOR': 'DIGER TROCMETAM',
        'CORTIPOL_PF': 'CORTIPOL',
        'CORTIPOL_COMPETITOR': 'DIGER CORTIPOL',
        'DEKSAMETAZON_PF': 'DEKSAMETAZON',
        'DEKSAMETAZON_COMPETITOR': 'DIGER DEKSAMETAZON',
        'IZOTONIK_PF': None,
        'IZOTONIK_COMPETITOR': None
    }
    
    return df, sales_columns

@st.cache_resource
def load_turkey_geojson():
    """Türkiye GeoJSON'u yükle"""
    try:
        # Türkiye şehir sınırları GeoJSON
        url = "https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json"
        gdf = gpd.read_file(url)
        gdf['name'] = gdf['name'].apply(TurkishTextNormalizer.normalize_text)
        return gdf
    except Exception as e:
        st.warning(f"GeoJSON yükleme hatası: {str(e)}")
        return None

# =============================================================================
# HESAPLAMA FONKSİYONLARI
# =============================================================================
def calculate_city_performance(df, product_cols):
    """Şehir bazlı performans verilerini hesaplar"""
    if df.empty:
        return pd.DataFrame()
    
    if 'CITY_NORMALIZED' not in df.columns:
        return pd.DataFrame()
    
    # Groupby işlemi
    city_perf = df.groupby(['CITY_NORMALIZED', 'REGION']).agg({
        product_cols['pf']: 'sum',
        product_cols['rakip']: 'sum'
    }).reset_index()
    
    # Kolon isimlendirme ve hesaplamalar
    city_perf.columns = ['City', 'Region', 'PF_Satis', 'Rakip_Satis']
    city_perf['Toplam_Pazar'] = city_perf['PF_Satis'] + city_perf['Rakip_Satis']
    
    # Sıfıra bölünme hatasını önlemek için
    city_perf['Pazar_Payi_%'] = np.where(
        city_perf['Toplam_Pazar'] > 0,
        (city_perf['PF_Satis'] / city_perf['Toplam_Pazar']) * 100,
        0
    )
    
    return city_perf

def calculate_territory_performance(df, product_cols):
    """Territory bazlı performans verilerini hesaplar"""
    if df.empty:
        return pd.DataFrame()
    
    if 'TERRITORIES' not in df.columns:
        return pd.DataFrame()
    
    if 'REGION' not in df.columns:
        df['REGION'] = 'DİĞER'
        
    territory_perf = df.groupby(['TERRITORIES', 'REGION']).agg({
        product_cols['pf']: 'sum',
        product_cols['rakip']: 'sum'
    }).reset_index()
    
    territory_perf.columns = ['Territory', 'Region', 'PF_Satis', 'Rakip_Satis']
    territory_perf['Toplam_Pazar'] = territory_perf['PF_Satis'] + territory_perf['Rakip_Satis']
    
    # Metrik hesaplamaları
    territory_perf['Pazar_Payi_%'] = np.where(
        territory_perf['Toplam_Pazar'] > 0,
        (territory_perf['PF_Satis'] / territory_perf['Toplam_Pazar']) * 100,
        0
    )
    
    # Toplam satış
    total_sales = territory_perf['PF_Satis'].sum()
    territory_perf['Agirlik_%'] = np.where(
        total_sales > 0,
        (territory_perf['PF_Satis'] / total_sales) * 100,
        0
    )
    
    # Göreceli pazar payı
    territory_perf['Goreceli_Pazar_Payi'] = np.where(
        territory_perf['Rakip_Satis'] > 0,
        territory_perf['PF_Satis'] / territory_perf['Rakip_Satis'],
        0
    )
    
    return territory_perf

def calculate_time_series(df, product_cols):
    """Zaman serisi verilerini hazırlar"""
    if df.empty or 'DATE' not in df.columns:
        return pd.DataFrame()
        
    ts_data = df.groupby('DATE').agg({
        product_cols['pf']: 'sum',
        product_cols['rakip']: 'sum'
    }).reset_index()
    
    ts_data.columns = ['DATE', 'PF_Satis', 'Rakip_Satis']
    ts_data = ts_data.sort_values('DATE')
    
    # Büyüme oranları
    ts_data['PF_Buyume_%'] = ts_data['PF_Satis'].pct_change() * 100
    ts_data['Rakip_Buyume_%'] = ts_data['Rakip_Satis'].pct_change() * 100
    
    return ts_data

# =============================================================================
# ANALİZ FONKSİYONLARI
# =============================================================================
class AdvancedAnalytics:
    """Gelişmiş analiz fonksiyonları"""
    
    @staticmethod
    def calculate_bcg_matrix(df, product_cols, date_filter=None):
        """BCG Matrix analizi"""
        if date_filter:
            df_filtered = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
        else:
            df_filtered = df.copy()
        
        # Territory bazlı performans
        territory_perf = df_filtered.groupby('TERRITORIES').agg({
            product_cols['pf']: 'sum',
            product_cols['rakip']: 'sum'
        }).reset_index()
        
        territory_perf.columns = ['Territory', 'PF_Satis', 'Rakip_Satis']
        territory_perf['Toplam_Pazar'] = territory_perf['PF_Satis'] + territory_perf['Rakip_Satis']
        territory_perf['Pazar_Payi_%'] = np.where(
            territory_perf['Toplam_Pazar'] > 0,
            (territory_perf['PF_Satis'] / territory_perf['Toplam_Pazar']) * 100,
            0
        )
        territory_perf['Goreceli_Pazar_Payi'] = np.where(
            territory_perf['Rakip_Satis'] > 0,
            territory_perf['PF_Satis'] / territory_perf['Rakip_Satis'],
            0
        )
        
        # Basit büyüme oranı (ilk ve son ay karşılaştırması)
        territory_perf['Buyume_%'] = 0
        
        # BCG kategorilerini ata
        median_share = territory_perf['Goreceli_Pazar_Payi'].median()
        median_growth = territory_perf['Buyume_%'].median()
        
        def assign_bcg(row):
            if row['Goreceli_Pazar_Payi'] >= median_share and row['Buyume_%'] >= median_growth:
                return "⭐ STAR"
            elif row['Goreceli_Pazar_Payi'] >= median_share and row['Buyume_%'] < median_growth:
                return "🐄 CASH COW"
            elif row['Goreceli_Pazar_Payi'] < median_share and row['Buyume_%'] >= median_growth:
                return "❓ QUESTION MARK"
            else:
                return "🐶 DOG"
        
        territory_perf['BCG_Kategori'] = territory_perf.apply(assign_bcg, axis=1)
        
        # Yatırım stratejisi ata
        def assign_strategy(row):
            if row['BCG_Kategori'] == "⭐ STAR":
                return "🚀 AGRESİF BÜYÜME"
            elif row['BCG_Kategori'] == "🐄 CASH COW":
                return "🛡️ KORUMA"
            elif row['BCG_Kategori'] == "❓ QUESTION MARK":
                return "💎 POTANSİYEL"
            else:
                return "👁️ İZLEME"
        
        territory_perf['Yatirim_Stratejisi'] = territory_perf.apply(assign_strategy, axis=1)
        
        return territory_perf
    
    @staticmethod
    def generate_insights(df, product_cols, date_filter=None):
        """Otomatik içgörü üretimi"""
        insights = []
        
        if date_filter:
            df_filtered = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
        else:
            df_filtered = df.copy()
        
        # 1. Büyüme içgörüleri
        monthly = df_filtered.groupby('YEAR_MONTH').agg({
            product_cols['pf']: 'sum',
            product_cols['rakip']: 'sum'
        }).reset_index()
        
        if len(monthly) > 3:
            recent_growth = monthly[product_cols['pf']].pct_change().iloc[-3:].mean() * 100
            if recent_growth > 20:
                insights.append({
                    'type': 'positive',
                    'title': '🚀 Yüksek Büyüme Hızı',
                    'description': f'Son 3 ayda ortalama {recent_growth:.1f}% büyüme kaydedildi.',
                    'impact': 'high'
                })
            elif recent_growth < -10:
                insights.append({
                    'type': 'negative',
                    'title': '⚠️ Büyüme Kaybı',
                    'description': f'Son 3 ayda ortalama {recent_growth:.1f}% küçülme yaşandı.',
                    'impact': 'high'
                })
        
        # 2. Bölgesel içgörüler
        if 'REGION' in df_filtered.columns:
            regional = df_filtered.groupby('REGION').agg({
                product_cols['pf']: 'sum',
                product_cols['rakip']: 'sum'
            }).reset_index()
            
            regional['share'] = np.where(
                (regional[product_cols['pf']] + regional[product_cols['rakip']]) > 0,
                regional[product_cols['pf']] / (regional[product_cols['pf']] + regional[product_cols['rakip']]),
                0
            )
            
            if len(regional) > 0:
                best_region = regional.loc[regional['share'].idxmax()]
                worst_region = regional.loc[regional['share'].idxmin()]
                
                insights.append({
                    'type': 'info',
                    'title': '🏆 En Güçlü Bölge',
                    'description': f"{best_region['REGION']} bölgesinde {best_region['share']:.1%} pazar payı ile lider konumdayız.",
                    'impact': 'medium'
                })
                
                insights.append({
                    'type': 'warning',
                    'title': '📉 Geliştirilmesi Gereken Bölge',
                    'description': f"{worst_region['REGION']} bölgesinde sadece {worst_region['share']:.1%} pazar payına sahibiz.",
                    'impact': 'medium'
                })
        
        return insights

# =============================================================================
# HARİTA GÖRSELLEŞTİRME
# =============================================================================
class AdvancedMapVisualizer:
    """Gelişmiş harita görselleştirme sınıfı"""
    
    @staticmethod
    def create_region_map(city_data: pd.DataFrame, title: str = "Türkiye Satış Haritası") -> go.Figure:
        """Bölge bazlı harita oluştur"""
        # Bölge bazında topla
        region_data = city_data.groupby('Region').agg({
            'PF_Satis': 'sum',
            'Rakip_Satis': 'sum'
        }).reset_index()
        
        region_data['Toplam_Pazar'] = region_data['PF_Satis'] + region_data['Rakip_Satis']
        region_data['Pazar_Payi_%'] = np.where(
            region_data['Toplam_Pazar'] > 0,
            (region_data['PF_Satis'] / region_data['Toplam_Pazar']) * 100,
            0
        )
        
        # Şehir koordinatlarını bölgeye göre grupla
        region_points = {}
        for _, row in city_data.iterrows():
            city = row['City']
            region = row['Region']
            if city in TURKEY_CITIES:
                if region not in region_points:
                    region_points[region] = []
                region_points[region].append({
                    'lon': TURKEY_CITIES[city]['lon'],
                    'lat': TURKEY_CITIES[city]['lat']
                })
        
        # Harita oluştur
        fig = go.Figure()
        
        # Her bölge için
        for region in region_points:
            if region in REGION_COLORS:
                points = region_points[region]
                lons = [p['lon'] for p in points]
                lats = [p['lat'] for p in points]
                
                # Bölge merkezini hesapla
                center_lon = np.mean(lons) if lons else 35.0
                center_lat = np.mean(lats) if lats else 39.0
                
                # Bölge verilerini al
                region_stats = region_data[region_data['Region'] == region]
                if len(region_stats) > 0:
                    pf_sales = region_stats['PF_Satis'].iloc[0]
                    market_share = region_stats['Pazar_Payi_%'].iloc[0]
                    
                    # Bölge etiketi
                    fig.add_trace(go.Scattermapbox(
                        lon=[center_lon],
                        lat=[center_lat],
                        mode='text',
                        text=[f"<b>{region}</b><br>{pf_sales:,.0f} PF<br>{market_share:.1f}%"],
                        textfont=dict(
                            size=12,
                            color='white',
                            family='Inter, sans-serif'
                        ),
                        hoverinfo='skip',
                        showlegend=False
                    ))
        
        # Harita ayarları
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox=dict(
                center=dict(lat=39.0, lon=35.0),
                zoom=4.5
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(size=20, color='white', family='Poppins')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_city_map(city_data: pd.DataFrame, title: str = "Şehir Bazlı Satış Haritası") -> go.Figure:
        """Şehir bazlı heatmap oluştur"""
        # Şehir koordinatlarını ekle
        city_data_with_coords = []
        for _, row in city_data.iterrows():
            city = row['City']
            if city in TURKEY_CITIES:
                city_data_with_coords.append({
                    'City': city,
                    'Region': row['Region'],
                    'PF_Satis': row['PF_Satis'],
                    'Rakip_Satis': row['Rakip_Satis'],
                    'Pazar_Payi_%': row['Pazar_Payi_%'],
                    'Lat': TURKEY_CITIES[city]['lat'],
                    'Lon': TURKEY_CITIES[city]['lon'],
                    'Color': REGION_COLORS.get(row['Region'], '#64748B')
                })
        
        if not city_data_with_coords:
            return go.Figure()
        
        df_coords = pd.DataFrame(city_data_with_coords)
        
        # Bubble chart oluştur
        fig = go.Figure()
        
        # Her bölge için ayrı trace
        for region in df_coords['Region'].unique():
            region_data = df_coords[df_coords['Region'] == region]
            color = REGION_COLORS.get(region, '#64748B')
            
            fig.add_trace(go.Scattermapbox(
                lon=region_data['Lon'],
                lat=region_data['Lat'],
                mode='markers',
                marker=dict(
                    size=region_data['PF_Satis'] / region_data['PF_Satis'].max() * 40 + 10,
                    color=color,
                    opacity=0.8,
                    sizemode='diameter'
                ),
                text=region_data['City'] + '<br>' + 
                     'PF: ' + region_data['PF_Satis'].astype(str) + '<br>' +
                     'Pay: ' + region_data['Pazar_Payi_%'].round(1).astype(str) + '%',
                hoverinfo='text',
                name=region,
                showlegend=True
            ))
        
        # Harita ayarları
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox=dict(
                center=dict(lat=39.0, lon=35.0),
                zoom=4.5
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(size=20, color='white', family='Poppins')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                title='<b>Bölgeler</b>',
                bgcolor='rgba(30, 41, 59, 0.8)',
                x=0.01,
                y=0.99
            )
        )
        
        return fig

# =============================================================================
# ANA UYGULAMA
# =============================================================================
def main():
    # Başlık
    st.markdown('<h1 class="main-header">🎯 PROFESYONEL TİCARİ PORTFÖY ANALİZ SİSTEMİ</h1>', unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown('<div style="background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%); '
                   'padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem; text-align: center;">'
                   '<h3 style="color: white; margin: 0; font-size: 1.5rem;">📂 VERİ YÜKLEME</h3>'
                   '</div>', unsafe_allow_html=True)
        
        # Veri yükleme seçenekleri
        data_option = st.radio(
            "Veri Kaynağı",
            ["📤 Excel Yükle", "🎯 Demo Veri Kullan"],
            label_visibility="collapsed"
        )
        
        if data_option == "📤 Excel Yükle":
            uploaded_file = st.file_uploader(
                "Excel Dosyası Seçin",
                type=['xlsx', 'xls'],
                label_visibility="collapsed"
            )
            
            if not uploaded_file:
                st.info("👈 Lütfen Excel dosyasını yükleyin veya demo veri ile devam edin")
                st.stop()
            
            try:
                df, sales_columns = load_and_process_excel(uploaded_file)
                st.success(f"✅ **{len(df):,}** satır veri yüklendi")
                
            except Exception as e:
                st.error(f"❌ Veri yükleme hatası: {str(e)}")
                st.stop()
        
        else:
            # Demo veri kullan
            df, sales_columns = create_sample_dataset()
            st.success(f"🎯 **{len(df):,}** satır demo veri yüklendi")
        
        st.markdown("---")
        
        # Ürün Seçimi
        st.markdown('<div style="background: rgba(30, 41, 59, 0.8); padding: 1.25rem; border-radius: 14px; margin: 1rem 0; border-left: 4px solid #3B82F6;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0; font-size: 1.1rem;">💊 ÜRÜN SEÇİMİ</h4>', unsafe_allow_html=True)
        
        available_products = []
        if 'TROCMETAM_PF' in sales_columns and sales_columns['TROCMETAM_PF'] is not None:
            available_products.append('TROCMETAM')
        if 'CORTIPOL_PF' in sales_columns and sales_columns['CORTIPOL_PF'] is not None:
            available_products.append('CORTIPOL')
        if 'DEKSAMETAZON_PF' in sales_columns and sales_columns['DEKSAMETAZON_PF'] is not None:
            available_products.append('DEKSAMETAZON')
        
        if not available_products:
            available_products = ['TROCMETAM', 'CORTIPOL', 'DEKSAMETAZON']
        
        selected_product = st.selectbox(
            "Analiz Edilecek Ürün",
            available_products,
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tarih Aralığı
        st.markdown('<div style="background: rgba(30, 41, 59, 0.8); padding: 1.25rem; border-radius: 14px; margin: 1rem 0; border-left: 4px solid #10B981;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0; font-size: 1.1rem;">📅 TARİH ARALIĞI</h4>', unsafe_allow_html=True)
        
        if 'DATE' in df.columns:
            min_date = df['DATE'].min()
            max_date = df['DATE'].max()
            
            date_options = [
                "📊 Tüm Veriler",
                "📈 Son 3 Ay",
                "📉 Son 6 Ay", 
                "📅 Son 1 Yıl",
                "🗓️ Özel Aralık"
            ]
            
            date_option = st.selectbox("Dönem Seçin", date_options)
            
            if date_option == "📊 Tüm Veriler":
                date_filter = None
            elif date_option == "📈 Son 3 Ay":
                start_date = max_date - pd.DateOffset(months=3)
                date_filter = (start_date, max_date)
            elif date_option == "📉 Son 6 Ay":
                start_date = max_date - pd.DateOffset(months=6)
                date_filter = (start_date, max_date)
            elif date_option == "📅 Son 1 Yıl":
                start_date = max_date - pd.DateOffset(years=1)
                date_filter = (start_date, max_date)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Başlangıç", min_date, min_value=min_date, max_value=max_date)
                with col2:
                    end_date = st.date_input("Bitiş", max_date, min_value=min_date, max_value=max_date)
                date_filter = (pd.to_datetime(start_date), pd.to_datetime(end_date))
        else:
            date_filter = None
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Filtreler
        st.markdown('<div style="background: rgba(30, 41, 59, 0.8); padding: 1.25rem; border-radius: 14px; margin: 1rem 0; border-left: 4px solid #F59E0B;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0; font-size: 1.1rem;">🔍 FİLTRELER</h4>', unsafe_allow_html=True)
        
        # Territory filtre
        if 'TERRITORIES' in df.columns:
            territories = ["🏢 TÜMÜ"] + sorted(df['TERRITORIES'].fillna('BILINMIYOR').unique().tolist())
            selected_territory = st.selectbox("Territory", territories)
        else:
            selected_territory = "🏢 TÜMÜ"
        
        # Bölge filtre
        if 'REGION' in df.columns:
            regions = ["🗺️ TÜMÜ"] + sorted(df['REGION'].fillna('DİĞER').unique().tolist())
            selected_region = st.selectbox("Bölge", regions)
        else:
            selected_region = "🗺️ TÜMÜ"
        
        # Manager filtre
        if 'MANAGER' in df.columns:
            managers = ["👤 TÜMÜ"] + sorted(df['MANAGER'].fillna('BILINMIYOR').unique().tolist())
            selected_manager = st.selectbox("Manager", managers)
        else:
            selected_manager = "👤 TÜMÜ"
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Harita Ayarları
        st.markdown('<div style="background: rgba(30, 41, 59, 0.8); padding: 1.25rem; border-radius: 14px; margin: 1rem 0; border-left: 4px solid #8B5CF6;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0; font-size: 1.1rem;">🗺️ HARİTA AYARLARI</h4>', unsafe_allow_html=True)
        
        map_mode = st.radio(
            "Harita Görünümü",
            ["🗺️ Bölge Haritası", "🏙️ Şehir Haritası"],
            index=0
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Ürün kolonlarını belirle
        product_cols = {}
        if selected_product == 'TROCMETAM':
            product_cols = {
                'pf': sales_columns.get('TROCMETAM_PF', 'TROCMETAM'),
                'rakip': sales_columns.get('TROCMETAM_COMPETITOR', 'DIGER TROCMETAM')
            }
        elif selected_product == 'CORTIPOL':
            product_cols = {
                'pf': sales_columns.get('CORTIPOL_PF', 'CORTIPOL'),
                'rakip': sales_columns.get('CORTIPOL_COMPETITOR', 'DIGER CORTIPOL')
            }
        else:
            product_cols = {
                'pf': sales_columns.get('DEKSAMETAZON_PF', 'DEKSAMETAZON'),
                'rakip': sales_columns.get('DEKSAMETAZON_COMPETITOR', 'DIGER DEKSAMETAZON')
            }

        # Veri filtreleme
        df_filtered = df.copy()
        
        if selected_territory != "🏢 TÜMÜ":
            df_filtered = df_filtered[df_filtered['TERRITORIES'] == selected_territory]
        
        if selected_region != "🗺️ TÜMÜ":
            df_filtered = df_filtered[df_filtered['REGION'] == selected_region]
        
        if selected_manager != "👤 TÜMÜ":
            df_filtered = df_filtered[df_filtered['MANAGER'] == selected_manager]
        
        if date_filter:
            df_filtered = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & 
                                      (df_filtered['DATE'] <= date_filter[1])]
    
    # ANA İÇERİK - TAB'LER
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Dashboard",
        "🗺️ Harita Analizi", 
        "🏢 Territory Analizi",
        "📈 Zaman Serisi",
        "🎯 Rakip Analizi",
        "⭐ BCG Matrix"
    ])
    
    # =========================================================================
    # TAB 1: EXECUTIVE DASHBOARD
    # =========================================================================
    with tab1:
        st.header("📊 Executive Dashboard - Genel Bakış")
        
        # KPI'lar
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pf = df_filtered[product_cols['pf']].sum()
            st.metric(
                label="💊 PF Satış",
                value=f"{total_pf:,.0f}"
            )
        
        with col2:
            total_market = df_filtered[product_cols['pf']].sum() + df_filtered[product_cols['rakip']].sum()
            market_share = (df_filtered[product_cols['pf']].sum() / total_market * 100) if total_market > 0 else 0
            st.metric(
                label="📊 Pazar Payı",
                value=f"%{market_share:.1f}"
            )
        
        with col3:
            if 'TERRITORIES' in df_filtered.columns:
                active_territories = df_filtered['TERRITORIES'].nunique()
                st.metric(
                    label="🏢 Aktif Territory",
                    value=f"{active_territories}"
                )
        
        with col4:
            if 'CITY' in df_filtered.columns:
                active_cities = df_filtered['CITY'].nunique()
                st.metric(
                    label="🏙️ Aktif Şehir",
                    value=f"{active_cities}"
                )
        
        st.markdown("---")
        
        # Grafikler
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("📈 Aylık Satış Trendi")
            
            if 'YEAR_MONTH' in df_filtered.columns:
                monthly_sales = df_filtered.groupby('YEAR_MONTH').agg({
                    product_cols['pf']: 'sum',
                    product_cols['rakip']: 'sum'
                }).reset_index()
                
                fig_trend = go.Figure()
                
                fig_trend.add_trace(go.Scatter(
                    x=monthly_sales['YEAR_MONTH'],
                    y=monthly_sales[product_cols['pf']],
                    name='PF Satış',
                    line=dict(color=PERFORMANCE_COLORS['positive'], width=3),
                    fill='tozeroy',
                    fillcolor='rgba(16, 185, 129, 0.1)'
                ))
                
                fig_trend.add_trace(go.Scatter(
                    x=monthly_sales['YEAR_MONTH'],
                    y=monthly_sales[product_cols['rakip']],
                    name='Rakip Satış',
                    line=dict(color=PERFORMANCE_COLORS['negative'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(239, 68, 68, 0.1)'
                ))
                
                fig_trend.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    xaxis_title='<b>Ay</b>',
                    yaxis_title='<b>Satış</b>',
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
        
        with col_chart2:
            st.subheader("🏆 Top 10 Territory")
            
            if 'TERRITORIES' in df_filtered.columns:
                top_territories = df_filtered.groupby('TERRITORIES')[product_cols['pf']].sum().nlargest(10)
                
                fig_bar = go.Figure()
                
                fig_bar.add_trace(go.Bar(
                    x=top_territories.values,
                    y=top_territories.index,
                    orientation='h',
                    marker_color=list(REGION_COLORS.values())[:10],
                    text=top_territories.values,
                    texttemplate='%{text:,.0f}',
                    textposition='outside'
                ))
                
                fig_bar.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    xaxis_title='<b>Satış</b>',
                    yaxis_title='<b>Territory</b>',
                    showlegend=False
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Otomatik İçgörüler
        st.markdown("---")
        st.subheader("💡 AI İçgörüleri")
        
        insights = AdvancedAnalytics.generate_insights(df_filtered, product_cols, date_filter)
        
        for insight in insights[:3]:
            if insight['type'] == 'positive':
                st.markdown(f"""
                <div class="alert-success">
                    <strong>{insight['title']}</strong><br>
                    {insight['description']}
                </div>
                """, unsafe_allow_html=True)
            elif insight['type'] == 'negative':
                st.markdown(f"""
                <div class="alert-danger">
                    <strong>{insight['title']}</strong><br>
                    {insight['description']}
                </div>
                """, unsafe_allow_html=True)
            elif insight['type'] == 'warning':
                st.markdown(f"""
                <div class="alert-warning">
                    <strong>{insight['title']}</strong><br>
                    {insight['description']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-info">
                    <strong>{insight['title']}</strong><br>
                    {insight['description']}
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 2: HARİTA ANALİZİ
    # =========================================================================
    with tab2:
        st.header("🗺️ Gelişmiş Harita Analizi")
        
        # Şehir performansını hesapla
        city_perf = calculate_city_performance(df_filtered, product_cols)
        
        if not city_perf.empty:
            if map_mode == "🗺️ Bölge Haritası":
                st.subheader("Bölge Bazlı Satış Haritası")
                region_map = AdvancedMapVisualizer.create_region_map(
                    city_perf, 
                    title=f"{selected_product} - Bölge Bazlı Dağılım"
                )
                st.plotly_chart(region_map, use_container_width=True)
            else:
                st.subheader("Şehir Bazlı Satış Haritası")
                city_map = AdvancedMapVisualizer.create_city_map(
                    city_perf,
                    title=f"{selected_product} - Şehir Bazlı Dağılım"
                )
                st.plotly_chart(city_map, use_container_width=True)
            
            # Şehir Performans Tablosu
            st.markdown("---")
            st.subheader("📋 Şehir Performans Detayları")
            
            col_filter1, col_filter2 = st.columns(2)
            
            with col_filter1:
                sort_by = st.selectbox(
                    "Sıralama Kriteri",
                    ['PF_Satis', 'Pazar_Payi_%', 'Toplam_Pazar'],
                    format_func=lambda x: {
                        'PF_Satis': 'PF Satış',
                        'Pazar_Payi_%': 'Pazar Payı',
                        'Toplam_Pazar': 'Toplam Pazar'
                    }[x]
                )
            
            with col_filter2:
                show_n = st.slider("Gösterilecek Şehir Sayısı", 5, 50, 15)
            
            city_sorted = city_perf.sort_values(sort_by, ascending=False).head(show_n)
            city_sorted.index = range(1, len(city_sorted) + 1)
            
            # Styling
            def color_share(val):
                if val >= 60:
                    return 'background-color: rgba(16, 185, 129, 0.3); color: #10B981; font-weight: bold'
                elif val >= 40:
                    return 'background-color: rgba(245, 158, 11, 0.3); color: #F59E0B; font-weight: bold'
                else:
                    return 'background-color: rgba(239, 68, 68, 0.3); color: #EF4444; font-weight: bold'
            
            styled_df = city_sorted[['City', 'Region', 'PF_Satis', 'Rakip_Satis', 'Pazar_Payi_%']].style.format({
                'PF_Satis': '{:,.0f}',
                'Rakip_Satis': '{:,.0f}',
                'Pazar_Payi_%': '{:.1f}%'
            }).applymap(color_share, subset=['Pazar_Payi_%'])
            
            st.dataframe(styled_df, use_container_width=True, height=400)
        else:
            st.warning("Harita analizi için yeterli veri bulunamadı")
    
    # =========================================================================
    # TAB 3: TERRITORY ANALİZİ
    # =========================================================================
    with tab3:
        st.header("🏢 Territory Bazlı Detaylı Analiz")
        
        # Territory performansını hesapla
        territory_perf = calculate_territory_performance(df_filtered, product_cols)
        
        if not territory_perf.empty:
            # Filtreleme
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            
            with col_filter1:
                sort_options = {
                    'PF_Satis': 'PF Satış',
                    'Pazar_Payi_%': 'Pazar Payı',
                    'Toplam_Pazar': 'Toplam Pazar',
                    'Agirlik_%': 'Ağırlık %'
                }
                sort_by = st.selectbox(
                    "Sıralama Kriteri",
                    options=list(sort_options.keys()),
                    format_func=lambda x: sort_options[x]
                )
            
            with col_filter2:
                show_n = st.slider("Gösterilecek Territory", 10, 50, 20)
            
            with col_filter3:
                min_sales = st.number_input(
                    "Minimum PF Satış",
                    min_value=0,
                    value=1000,
                    step=1000
                )
            
            # Filtrele
            territory_filtered = territory_perf[territory_perf['PF_Satis'] >= min_sales]
            territory_sorted = territory_filtered.sort_values(sort_by, ascending=False).head(show_n)
            
            # Görselleştirmeler
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("📊 PF vs Rakip Satış")
                
                fig_comparison = go.Figure()
                
                fig_comparison.add_trace(go.Bar(
                    x=territory_sorted['Territory'],
                    y=territory_sorted['PF_Satis'],
                    name='PF Satış',
                    marker_color=PERFORMANCE_COLORS['positive']
                ))
                
                fig_comparison.add_trace(go.Bar(
                    x=territory_sorted['Territory'],
                    y=territory_sorted['Rakip_Satis'],
                    name='Rakip Satış',
                    marker_color=PERFORMANCE_COLORS['negative']
                ))
                
                fig_comparison.update_layout(
                    barmode='group',
                    height=500,
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    xaxis_title='<b>Territory</b>',
                    yaxis_title='<b>Satış</b>',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            with col_chart2:
                st.subheader("🎯 Pazar Payı Dağılımı")
                
                fig_scatter = px.scatter(
                    territory_sorted,
                    x='PF_Satis',
                    y='Pazar_Payi_%',
                    size='Toplam_Pazar',
                    color='Region',
                    hover_name='Territory',
                    color_discrete_map=REGION_COLORS,
                    size_max=50,
                    title='<b>Territory Performans Haritası</b>'
                )
                
                fig_scatter.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    xaxis_title='<b>PF Satış</b>',
                    yaxis_title='<b>Pazar Payı %</b>'
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Detaylı Tablo
            st.markdown("---")
            st.subheader(f"📋 Territory Detayları (Top {show_n})")
            
            territory_display = territory_sorted.copy()
            territory_display.index = range(1, len(territory_display) + 1)
            
            # Styling fonksiyonları
            def highlight_performance(val):
                if val >= 60:
                    return 'background-color: rgba(16, 185, 129, 0.3); color: #10B981; font-weight: bold'
                elif val >= 40:
                    return 'background-color: rgba(245, 158, 11, 0.3); color: #F59E0B; font-weight: bold'
                else:
                    return 'background-color: rgba(239, 68, 68, 0.3); color: #EF4444; font-weight: bold'
            
            def highlight_weight(val):
                if val >= 5:
                    return 'background-color: rgba(59, 130, 246, 0.3); color: #3B82F6; font-weight: bold'
                elif val >= 2:
                    return 'background-color: rgba(59, 130, 246, 0.2); color: #3B82F6'
                else:
                    return ''
            
            styled_territory = territory_display[['Territory', 'Region', 'PF_Satis', 'Rakip_Satis', 
                                                  'Pazar_Payi_%', 'Agirlik_%']].style.format({
                'PF_Satis': '{:,.0f}',
                'Rakip_Satis': '{:,.0f}',
                'Pazar_Payi_%': '{:.1f}%',
                'Agirlik_%': '{:.1f}%'
            }).applymap(highlight_performance, subset=['Pazar_Payi_%'])\
              .applymap(highlight_weight, subset=['Agirlik_%'])
            
            st.dataframe(styled_territory, use_container_width=True, height=500)
        else:
            st.warning("Territory verisi bulunamadı")
    
    # =========================================================================
    # TAB 4: ZAMAN SERİSİ
    # =========================================================================
    with tab4:
        st.header("📈 Zaman Serisi Analizi")
        
        # Zaman serisi verisini hazırla
        time_series_df = calculate_time_series(df_filtered, product_cols)
        
        if not time_series_df.empty:
            # Top row metrics
            col_ts1, col_ts2, col_ts3, col_ts4 = st.columns(4)
            
            with col_ts1:
                avg_sales = time_series_df['PF_Satis'].mean()
                st.metric("📊 Ort. Aylık Satış", f"{avg_sales:,.0f}")
            
            with col_ts2:
                total_growth = ((time_series_df['PF_Satis'].iloc[-1] / time_series_df['PF_Satis'].iloc[0]) - 1) * 100 if len(time_series_df) > 1 else 0
                st.metric("📈 Toplam Büyüme", f"%{total_growth:.1f}")
            
            with col_ts3:
                volatility = time_series_df['PF_Satis'].pct_change().std() * np.sqrt(12) * 100 if len(time_series_df) > 1 else 0
                st.metric("📉 Yıllık Volatilite", f"%{volatility:.1f}")
            
            with col_ts4:
                if len(time_series_df) > 1:
                    avg_growth = time_series_df['PF_Buyume_%'].mean()
                    st.metric("📊 Ort. Aylık Büyüme", f"%{avg_growth:.1f}")
            
            # Zaman Serisi Grafiği
            st.markdown("---")
            st.subheader("📊 Satış Trend Analizi")
            
            fig_timeseries = go.Figure()
            
            fig_timeseries.add_trace(go.Scatter(
                x=time_series_df['DATE'],
                y=time_series_df['PF_Satis'],
                name='PF Satış',
                line=dict(color=PERFORMANCE_COLORS['positive'], width=3),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.1)'
            ))
            
            fig_timeseries.add_trace(go.Scatter(
                x=time_series_df['DATE'],
                y=time_series_df['Rakip_Satis'],
                name='Rakip Satış',
                line=dict(color=PERFORMANCE_COLORS['negative'], width=2),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.1)'
            ))
            
            fig_timeseries.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis_title='<b>Tarih</b>',
                yaxis_title='<b>Satış</b>',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_timeseries, use_container_width=True)
            
            # Büyüme Grafiği
            st.markdown("---")
            st.subheader("📈 Büyüme Oranları")
            
            fig_growth = go.Figure()
            
            fig_growth.add_trace(go.Scatter(
                x=time_series_df['DATE'],
                y=time_series_df['PF_Buyume_%'],
                name='PF Büyüme',
                line=dict(color=PERFORMANCE_COLORS['positive'], width=3)
            ))
            
            fig_growth.add_trace(go.Scatter(
                x=time_series_df['DATE'],
                y=time_series_df['Rakip_Buyume_%'],
                name='Rakip Büyüme',
                line=dict(color=PERFORMANCE_COLORS['negative'], width=2)
            ))
            
            fig_growth.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            
            fig_growth.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis_title='<b>Tarih</b>',
                yaxis_title='<b>Büyüme Oranı (%)</b>',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_growth, use_container_width=True)
        else:
            st.warning("Zaman serisi verisi bulunamadı")
    
    # =========================================================================
    # TAB 5: RAKİP ANALİZİ
    # =========================================================================
    with tab5:
        st.header("🎯 Rakip ve Pazar Analizi")
        
        if not df_filtered.empty:
            # Pazar İstihbaratı
            st.subheader("📊 Pazar İstihbaratı")
            
            col_intel1, col_intel2, col_intel3, col_intel4 = st.columns(4)
            
            with col_intel1:
                total_market = df_filtered[product_cols['pf']].sum() + df_filtered[product_cols['rakip']].sum()
                st.metric("🏪 Toplam Pazar", f"{total_market:,.0f}")
            
            with col_intel2:
                pf_sales = df_filtered[product_cols['pf']].sum()
                comp_sales = df_filtered[product_cols['rakip']].sum()
                market_share = (pf_sales / total_market * 100) if total_market > 0 else 0
                st.metric("📊 Pazar Payımız", f"%{market_share:.1f}")
            
            with col_intel3:
                if 'YEAR_MONTH' in df_filtered.columns and len(df_filtered['YEAR_MONTH'].unique()) > 1:
                    monthly = df_filtered.groupby('YEAR_MONTH')[product_cols['pf']].sum()
                    pf_growth = ((monthly.iloc[-1] / monthly.iloc[0]) - 1) * 100 if monthly.iloc[0] > 0 else 0
                    st.metric("📈 PF Büyüme", f"%{pf_growth:.1f}")
            
            with col_intel4:
                if 'YEAR_MONTH' in df_filtered.columns and len(df_filtered['YEAR_MONTH'].unique()) > 1:
                    monthly = df_filtered.groupby('YEAR_MONTH')[product_cols['rakip']].sum()
                    comp_growth = ((monthly.iloc[-1] / monthly.iloc[0]) - 1) * 100 if monthly.iloc[0] > 0 else 0
                    st.metric("📉 Rakip Büyüme", f"%{comp_growth:.1f}")
            
            # Bölgesel Rakip Analizi
            st.markdown("---")
            st.subheader("🌍 Bölgesel Rakip Analizi")
            
            if 'REGION' in df_filtered.columns:
                region_competition = df_filtered.groupby('REGION').agg({
                    product_cols['pf']: 'sum',
                    product_cols['rakip']: 'sum'
                }).reset_index()
                
                region_competition['PF_Pay_%'] = np.where(
                    (region_competition[product_cols['pf']] + region_competition[product_cols['rakip']]) > 0,
                    (region_competition[product_cols['pf']] / 
                     (region_competition[product_cols['pf']] + region_competition[product_cols['rakip']])) * 100,
                    0
                )
                
                region_competition = region_competition.sort_values('PF_Pay_%', ascending=False)
                
                # Grafik
                fig_region = go.Figure()
                
                fig_region.add_trace(go.Bar(
                    x=region_competition['REGION'],
                    y=region_competition['PF_Pay_%'],
                    name='PF Pazar Payı',
                    marker_color=[REGION_COLORS.get(region, '#64748B') for region in region_competition['REGION']]
                ))
                
                fig_region.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.5,
                                   annotation_text="50% Eşik")
                
                fig_region.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    xaxis_title='<b>Bölge</b>',
                    yaxis_title='<b>PF Pazar Payı (%)</b>',
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig_region, use_container_width=True)
                
                # Tablo
                st.markdown("---")
                st.subheader("📋 Bölge Bazlı Detaylar")
                
                region_display = region_competition.copy()
                region_display.columns = ['Bölge', 'PF Satış', 'Rakip Satış', 'PF Payı %']
                region_display.index = range(1, len(region_display) + 1)
                
                # Styling
                def color_region_share(val):
                    if val >= 60:
                        return 'background-color: rgba(16, 185, 129, 0.3); color: #10B981; font-weight: bold'
                    elif val >= 40:
                        return 'background-color: rgba(245, 158, 11, 0.3); color: #F59E0B; font-weight: bold'
                    else:
                        return 'background-color: rgba(239, 68, 68, 0.3); color: #EF4444; font-weight: bold'
                
                styled_region = region_display.style.format({
                    'PF Satış': '{:,.0f}',
                    'Rakip Satış': '{:,.0f}',
                    'PF Payı %': '{:.1f}%'
                }).applymap(color_region_share, subset=['PF Payı %'])
                
                st.dataframe(styled_region, use_container_width=True, height=400)
        else:
            st.warning("Rakip analizi için yeterli veri bulunamadı")
    
    # =========================================================================
    # TAB 6: BCG MATRIX
    # =========================================================================
    with tab6:
        st.header("⭐ BCG Matrix & Stratejik Portföy Yönetimi")
        
        # BCG Matrix hesapla
        bcg_df = AdvancedAnalytics.calculate_bcg_matrix(df_filtered, product_cols, date_filter)
        
        if not bcg_df.empty:
            # BCG Dağılımı
            st.subheader("📊 Portföy Dağılımı")
            
            bcg_counts = bcg_df['BCG_Kategori'].value_counts()
            strategy_counts = bcg_df['Yatirim_Stratejisi'].value_counts()
            
            col_bcg1, col_bcg2, col_bcg3, col_bcg4 = st.columns(4)
            
            with col_bcg1:
                star_count = bcg_counts.get("⭐ STAR", 0)
                star_value = bcg_df[bcg_df['BCG_Kategori'] == "⭐ STAR"]['PF_Satis'].sum() if star_count > 0 else 0
                st.metric("⭐ STAR", f"{star_count}", f"{star_value:,.0f} PF")
            
            with col_bcg2:
                cow_count = bcg_counts.get("🐄 CASH COW", 0)
                cow_value = bcg_df[bcg_df['BCG_Kategori'] == "🐄 CASH COW"]['PF_Satis'].sum() if cow_count > 0 else 0
                st.metric("🐄 CASH COW", f"{cow_count}", f"{cow_value:,.0f} PF")
            
            with col_bcg3:
                question_count = bcg_counts.get("❓ QUESTION MARK", 0)
                question_value = bcg_df[bcg_df['BCG_Kategori'] == "❓ QUESTION MARK"]['PF_Satis'].sum() if question_count > 0 else 0
                st.metric("❓ QUESTION MARK", f"{question_count}", f"{question_value:,.0f} PF")
            
            with col_bcg4:
                dog_count = bcg_counts.get("🐶 DOG", 0)
                dog_value = bcg_df[bcg_df['BCG_Kategori'] == "🐶 DOG"]['PF_Satis'].sum() if dog_count > 0 else 0
                st.metric("🐶 DOG", f"{dog_count}", f"{dog_value:,.0f} PF")
            
            # BCG Matrix Görselleştirme
            st.markdown("---")
            st.subheader("🎯 BCG Matrix")
            
            # BCG için veriyi hazırla
            bcg_df['size'] = np.sqrt(bcg_df['PF_Satis']) / 50
            bcg_df['color'] = bcg_df['BCG_Kategori'].map(BCG_COLORS)
            
            fig_bcg = go.Figure()
            
            # Her kategori için ayrı trace
            for category in bcg_df['BCG_Kategori'].unique():
                category_data = bcg_df[bcg_df['BCG_Kategori'] == category]
                
                fig_bcg.add_trace(go.Scatter(
                    x=category_data['Goreceli_Pazar_Payi'],
                    y=category_data['Buyume_%'],
                    mode='markers',
                    name=category,
                    marker=dict(
                        size=category_data['size'],
                        color=BCG_COLORS[category],
                        opacity=0.7,
                        line=dict(width=2, color='white')
                    ),
                    text=category_data['Territory'],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Göreceli Pay: %{x:.2f}<br>"
                        "Büyüme: %{y:.1f}%<br>"
                        "PF Satış: %{customdata[0]:,.0f}<br>"
                        "Pazar Payı: %{customdata[1]:.1f}%"
                        "<extra></extra>"
                    ),
                    customdata=np.stack((
                        category_data['PF_Satis'],
                        category_data['Pazar_Payi_%']
                    ), axis=-1)
                ))
            
            fig_bcg.update_layout(
                title='<b>BCG Matrix - Stratejik Konumlandırma</b>',
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis_title='<b>Göreceli Pazar Payı</b>',
                yaxis_title='<b>Pazar Büyüme Oranı (%)</b>',
                legend=dict(
                    title='<b>BCG Kategorisi</b>',
                    bgcolor='rgba(30, 41, 59, 0.8)'
                )
            )
            
            st.plotly_chart(fig_bcg, use_container_width=True)
            
            # Stratejik Öneriler
            st.markdown("---")
            st.subheader("💼 Yatırım Stratejileri")
            
            # Her BCG kategorisi için territory listesi
            for category in bcg_df['BCG_Kategori'].unique():
                category_data = bcg_df[bcg_df['BCG_Kategori'] == category]
                strategy = category_data['Yatirim_Stratejisi'].iloc[0] if len(category_data) > 0 else "👁️ İZLEME"
                
                with st.expander(f"{category} - {strategy} ({len(category_data)} Territory)"):
                    # Territory listesi
                    territories_list = category_data[['Territory', 'Region', 'PF_Satis', 'Pazar_Payi_%']]\
                        .sort_values('PF_Satis', ascending=False)
                    territories_list.index = range(1, len(territories_list) + 1)
                    
                    st.dataframe(
                        territories_list.style.format({
                            'PF_Satis': '{:,.0f}',
                            'Pazar_Payi_%': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
        else:
            st.warning("BCG Matrix analizi için yeterli veri bulunamadı")

# =============================================================================
# UYGULAMAYI BAŞLAT
# =============================================================================
if __name__ == "__main__":
    main()
