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

GELİŞTİRİCİ: Advanced Analytics Team
VERSİYON: 4.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
from io import BytesIO
import json
import base64
import requests
from typing import Dict, List, Tuple, Optional, Any
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter
try:
    import statsmodels.api as sm
except ImportError:
    import os
    os.system('pip install statsmodels')
    import statsmodels.api as smfrom statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import optuna
from optuna.samplers import TPESampler
import shap
import dtw
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
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
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 1.5rem 0 1rem 0;
        padding: 0.75rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
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
    
    div[data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(59, 130, 246, 0.1), 
            transparent);
        transition: left 0.7s;
    }
    
    div[data-testid="metric-container"]:hover::before {
        left: 100%;
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
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.15);
        color: #e0e7ff;
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transform: translateY(-2px) scale(1.05);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        font-family: 'Poppins', sans-serif;
    }
    
    h1 { font-size: 2.8rem; margin-top: 0; }
    h2 { font-size: 2.2rem; margin-top: 0; }
    h3 { font-size: 1.8rem; margin-top: 0; }
    h4 { font-size: 1.4rem; margin-top: 0; }
    
    p, span, div, label {
        color: #cbd5e1;
        line-height: 1.6;
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
    
    .stButton>button:active {
        transform: translateY(-2px);
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.2), 
            transparent);
        transition: left 0.7s;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div,
    .stDateInput>div>div>input {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>div:focus,
    .stDateInput>div>div>input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
        background: rgba(30, 41, 59, 0.9) !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: rgba(30, 41, 59, 0.7);
        padding: 1.5rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .stCheckbox span {
        color: #e2e8f0;
        font-weight: 500;
    }
    
    .stDataFrame, .stTable {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%);
        border-radius: 8px;
        border: 2px solid rgba(30, 41, 59, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #10B981 0%, #F59E0B 100%);
    }
    
    /* Plotly graph styling */
    .plotly-graph-div {
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        background: rgba(15, 23, 41, 0.5);
        border: 1px solid rgba(59, 130, 246, 0.15);
    }
    
    .plotly-graph-div:hover {
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.2);
    }
    
    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 41, 0.98) !important;
        backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(59, 130, 246, 0.1) !important;
        box-shadow: 5px 0 30px rgba(0, 0, 0, 0.3) !important;
    }
    
    [data-testid="stSidebar"] .stRadio,
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stSlider {
        background: rgba(30, 41, 59, 0.7) !important;
        padding: 0.75rem !important;
        border-radius: 10px !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 50%, #F59E0B 100%);
        background-size: 200% 200%;
        animation: progressGradient 2s ease infinite;
    }
    
    @keyframes progressGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Custom alerts and cards */
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
    
    /* Card styles */
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
    
    .custom-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3B82F6, #10B981, #F59E0B, #8B5CF6);
        background-size: 400% 400%;
        animation: gradientBG 3s ease infinite;
    }
    
    .custom-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(59, 130, 246, 0.25);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    /* Badge styles */
    .badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-success {
        background: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .badge-danger {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .badge-info {
        background: rgba(59, 130, 246, 0.2);
        color: #3B82F6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: rgba(15, 23, 41, 0.95);
        color: #e2e8f0;
        text-align: center;
        border-radius: 10px;
        padding: 1rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 50%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Glowing effect */
    .glow {
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
    }
    
    .glow:hover {
        box-shadow: 0 0 40px rgba(59, 130, 246, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# KONSTANTLAR VE RENK PALETLERİ
# =============================================================================

# BÖLGE RENKLERİ (COĞRAFİ & MODERN)
REGION_COLORS = {
    "MARMARA": "#0EA5E9",              # Sky Blue - Deniz ve boğazlar
    "BATI ANADOLU": "#14B8A6",         # Turkuaz-yeşil arası
    "İÇ ANADOLU": "#F59E0B",           # Amber - Kuru bozkır
    "GÜNEY DOĞU ANADOLU": "#E07A5F",   # Terracotta 
    "KUZEY ANADOLU": "#059669",        # Emerald - Yemyeşil ormanlar
    "AKDENİZ": "#3B82F6",              # Deep Blue - Akdeniz suları
    "EGE": "#8B5CF6",                  # Purple - Lavanta tarlaları
    "DOĞU ANADOLU": "#EF4444",         # Crimson - Dağlar ve volkanik topraklar
    "KARADENİZ": "#10B981",            # Green - Yemyeşil yağmur ormanları
    "DİĞER": "#64748B",                # Gray - Diğer bölgeler
}

# PERFORMANS RENKLERİ
PERFORMANCE_COLORS = {
    # Performans Seviyeleri
    "excellent": "#10B981",      # Emerald Green
    "good": "#22C55E",           # Green
    "average": "#F59E0B",        # Amber
    "poor": "#EF4444",           # Red
    "critical": "#991B1B",       # Dark Red
    
    # Trend Renkleri
    "positive": "#10B981",       # Green
    "negative": "#EF4444",       # Red
    "neutral": "#6B7280",        # Gray
    "warning": "#F59E0B",        # Amber
    
    # Stratejik Renkler
    "growth": "#8B5CF6",         # Purple
    "stable": "#3B82F6",         # Blue
    "decline": "#F59E0B",        # Amber
    "risk": "#EF4444",           # Red
}

# BCG MATRIX RENKLERİ
BCG_COLORS = {
    "⭐ STAR": "#F59E0B",         # Gold - Yüksek büyüme, yüksek pay
    "🐄 CASH COW": "#10B981",    # Green - Düşük büyüme, yüksek pay
    "❓ QUESTION MARK": "#3B82F6", # Blue - Yüksek büyüme, düşük pay
    "🐶 DOG": "#64748B",         # Gray - Düşük büyüme, düşük pay
}

# YATIRIM STRATEJİSİ RENKLERİ
STRATEGY_COLORS = {
    "🚀 AGRESİF BÜYÜME": "#EF4444",      # Red - Agresif yatırım
    "📈 HIZLANDIRILMIŞ": "#F59E0B",      # Orange - Orta seviye yatırım
    "🛡️ KORUMA": "#10B981",              # Green - Koruma stratejisi
    "💎 POTANSİYEL": "#8B5CF6",           # Purple - Potansiyel gelişim
    "👁️ İZLEME": "#64748B",              # Gray - Pasif izleme
    "🔄 YENİDEN YAPILANMA": "#EC4899",   # Pink - Yeniden yapılanma
}

# SEZONLUK RENKLER (Aylara göre)
SEASONAL_COLORS = {
    1: "#3B82F6",   # Ocak - Kış mavisi
    2: "#8B5CF6",   # Şubat - Mor
    3: "#10B981",   # Mart - Yeşil (bahar)
    4: "#22C55E",   # Nisan - Açık yeşil
    5: "#84CC16",   # Mayıs - Çim yeşili
    6: "#F59E0B",   # Haziran - Yaz altını
    7: "#F97316",   # Temmuz - Turuncu
    8: "#EF4444",   # Ağustos - Kırmızı (sıcak)
    9: "#EC4899",   # Eylül - Pembe
    10: "#8B5CF6",  # Ekim - Mor (sonbahar)
    11: "#6366F1",  # Kasım - İndigo
    12: "#0EA5E9",  # Aralık - Kış mavisi
}

# GRADIENT SCALES
GRADIENT_SCALES = {
    "blue_green": ["#0EA5E9", "#3B82F6", "#10B981", "#22C55E"],
    "temperature": ["#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE"],
    "diverging": ["#EF4444", "#F59E0B", "#10B981", "#3B82F6", "#8B5CF6"],
    "sequential": ["#1E40AF", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"],
    "rainbow": ["#EF4444", "#F59E0B", "#84CC16", "#10B981", "#0EA5E9", "#3B82F6", "#8B5CF6"],
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
# İLERİ SEVİYE YARDIMCI FONKSİYONLAR
# =============================================================================

class AdvancedDataProcessor:
    """İleri seviye veri işleme ve temizleme sınıfı"""
    
    @staticmethod
    def detect_data_structure(df: pd.DataFrame) -> Dict:
        """Veri yapısını otomatik tespit et"""
        structure = {
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "date_columns": [],
            "sales_columns": [],
            "geography_columns": [],
            "manager_columns": []
        }
        
        # Tarih sütunlarını tespit et
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['date', 'tarih', 'ay', 'yıl', 'month', 'year', 'day']):
                structure["date_columns"].append(col)
            
            # Satış sütunlarını tespit et
            if any(keyword in col_lower for keyword in ['satış', 'sales', 'tutar', 'miktar', 'adet', 'volume']):
                structure["sales_columns"].append(col)
            
            # Coğrafi sütunları tespit et
            if any(keyword in col_lower for keyword in ['şehir', 'sehir', 'city', 'il', 'bölge', 'bolge', 'region']):
                structure["geography_columns"].append(col)
            
            # Manager sütunlarını tespit et
            if any(keyword in col_lower for keyword in ['manager', 'sorumlu', 'temsilci', 'rep', 'müdür']):
                structure["manager_columns"].append(col)
        
        return structure
    
    @staticmethod
    def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
        """Veriyi temizle ve dönüştür"""
        df_clean = df.copy()
        
        # Sütun adlarını standardize et
        df_clean.columns = [col.strip().upper() for col in df_clean.columns]
        
        # Tarih sütunlarını işle
        for col in df_clean.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['date', 'tarih']):
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                except:
                    pass
        
        # Numerik sütunlardaki karakterleri temizle
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Eksik değerleri doldur
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('BILINMIYOR')
            elif pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        return df_clean
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Gelişmiş özellik mühendisliği"""
        df_features = df.copy()
        
        # Tarih özellikleri
        date_cols = [col for col in df_features.columns if 'DATE' in col or 'TARIH' in col]
        if date_cols:
            date_col = date_cols[0]
            df_features['YEAR'] = pd.to_datetime(df_features[date_col]).dt.year
            df_features['MONTH'] = pd.to_datetime(df_features[date_col]).dt.month
            df_features['QUARTER'] = pd.to_datetime(df_features[date_col]).dt.quarter
            df_features['WEEK'] = pd.to_datetime(df_features[date_col]).dt.isocalendar().week
            df_features['DAY_OF_WEEK'] = pd.to_datetime(df_features[date_col]).dt.dayofweek
            df_features['DAY_OF_YEAR'] = pd.to_datetime(df_features[date_col]).dt.dayofyear
            df_features['IS_WEEKEND'] = df_features['DAY_OF_WEEK'].isin([5, 6]).astype(int)
            df_features['YEAR_MONTH'] = pd.to_datetime(df_features[date_col]).dt.strftime('%Y-%m')
            
            # Mevsimsel özellikler
            df_features['SEASON'] = df_features['MONTH'].apply(
                lambda x: 1 if x in [12, 1, 2] else  # Kış
                          2 if x in [3, 4, 5] else  # İlkbahar
                          3 if x in [6, 7, 8] else  # Yaz
                          4                         # Sonbahar
            )
        
        # Satış özellikleri
        sales_patterns = ['SATIS', 'SALES', 'MIKTAR', 'ADET', 'VOLUME']
        sales_cols = [col for col in df_features.columns if any(pattern in col for pattern in sales_patterns)]
        
        if sales_cols:
            for col in sales_cols:
                # Log dönüşümü
                if df_features[col].min() > 0:
                    df_features[f'LOG_{col}'] = np.log1p(df_features[col])
                
                # Z-skor
                df_features[f'ZSCORE_{col}'] = stats.zscore(df_features[col].fillna(0))
                
                # Quantile-based features
                df_features[f'QUARTILE_{col}'] = pd.qcut(df_features[col], 4, labels=False, duplicates='drop')
        
        return df_features

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
        'K. MARAS': 'KAHRAMANMARAŞ',
        'KAHRAMANMARAS': 'KAHRAMANMARAŞ',
        'SANLIURFA': 'ŞANLIURFA',
        'SIRNAK': 'ŞIRNAK',
        'NEVSEHIR': 'NEVŞEHİR',
        'NIGDE': 'NİĞDE',
        'MUS': 'MUŞ',
        'MUGLA': 'MUĞLA',
        'KUTAHYA': 'KÜTAHYA',
        'GUMUSHANE': 'GÜMÜŞHANE',
        'BARTIN': 'BARTIN',
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
        'ARDAHAN': 'KARADENİZ',
        'ARTVİN': 'KARADENİZ',
        'BAYBURT': 'KARADENİZ',
        'GÜMÜŞHANE': 'KARADENİZ',
        'TOKAT': 'KARADENİZ',
        'AMASYA': 'KARADENİZ',
        'ÇORUM': 'KARADENİZ',
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
# GELİŞMİŞ VERİ YÜKLEME VE İŞLEME
# =============================================================================

@st.cache_data(ttl=3600)
def load_and_process_excel(file) -> Tuple[pd.DataFrame, Dict]:
    """Excel dosyasını yükle ve gelişmiş işleme uygula"""
    try:
        # Excel'i yükle
        df = pd.read_excel(file, engine='openpyxl')
        
        # Veri yapısını analiz et
        processor = AdvancedDataProcessor()
        data_structure = processor.detect_data_structure(df)
        
        # Temizle ve dönüştür
        df_clean = processor.clean_and_transform(df)
        
        # Özellik mühendisliği
        df_features = processor.create_features(df_clean)
        
        # Şehir ve bölge normalizasyonu
        if 'CITY' in df_features.columns:
            df_features['CITY_NORMALIZED'] = df_features['CITY'].apply(
                TurkishTextNormalizer.normalize_text
            )
            df_features['REGION'] = df_features['CITY_NORMALIZED'].apply(
                TurkishTextNormalizer.assign_region
            )
        
        # Eksik sütunları tamamla
        required_columns = ['TERRITORIES', 'MANAGER', 'YEAR_MONTH']
        for col in required_columns:
            if col not in df_features.columns:
                if col == 'TERRITORIES' and 'CITY_NORMALIZED' in df_features.columns:
                    df_features['TERRITORIES'] = df_features['CITY_NORMALIZED']
                elif col == 'MANAGER':
                    df_features['MANAGER'] = 'BILINMIYOR'
                elif col == 'YEAR_MONTH':
                    df_features['YEAR_MONTH'] = '2023-01'
        
        # Satış sütunlarını tespit et
        sales_columns = {}
        for col in df_features.columns:
            col_upper = str(col).upper()
            if 'TROCMETAM' in col_upper and 'DIGER' not in col_upper:
                sales_columns['TROCMETAM_PF'] = col
            elif 'DIGER TROCMETAM' in col_upper or ('DIGER' in col_upper and 'TROCMETAM' in col_upper):
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
        
        return df_features, sales_columns
        
    except Exception as e:
        st.error(f"Excel yükleme hatası: {str(e)}")
        # Örnek veri oluştur
        return create_sample_dataset()

def create_sample_dataset() -> Tuple[pd.DataFrame, Dict]:
    """Örnek veri seti oluştur (test için)"""
    np.random.seed(42)
    
    # Tarih aralığı
    dates = pd.date_range(start='2022-01-01', end='2025-01-01', freq='M')
    
    # Şehirler ve bölgeler
    cities = list(TURKEY_CITIES.keys())[:40]
    regions = [TURKEY_CITIES[city]['region'] for city in cities]
    
    # Territory ve Manager
    territories = [f'TERR_{i:03d}' for i in range(1, 51)]
    managers = [f'MANAGER_{i}' for i in range(1, 11)]
    
    data = []
    for date in dates:
        for city, region in zip(cities[:20], regions[:20]):
            territory = np.random.choice(territories)
            manager = np.random.choice(managers)
            
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
                'PF IZOTONIK': int(pf_sales * np.random.uniform(1.5, 2.5)),
                'DIGER IZOTONIK': int(comp_sales * np.random.uniform(1.5, 2.5))
            })
    
    df = pd.DataFrame(data)
    
    # Özellik mühendisliği
    df['YEAR'] = df['DATE'].dt.year
    df['MONTH'] = df['DATE'].dt.month
    df['QUARTER'] = df['DATE'].dt.quarter
    df['YEAR_MONTH'] = df['DATE'].dt.strftime('%Y-%m')
    df['CITY_NORMALIZED'] = df['CITY']
    
    sales_columns = {
        'TROCMETAM_PF': 'TROCMETAM',
        'TROCMETAM_COMPETITOR': 'DIGER TROCMETAM',
        'CORTIPOL_PF': 'CORTIPOL',
        'CORTIPOL_COMPETITOR': 'DIGER CORTIPOL',
        'DEKSAMETAZON_PF': 'DEKSAMETAZON',
        'DEKSAMETAZON_COMPETITOR': 'DIGER DEKSAMETAZON',
        'IZOTONIK_PF': 'PF IZOTONIK',
        'IZOTONIK_COMPETITOR': 'DIGER IZOTONIK'
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
    except:
        # Fallback: Manuel olarak Türkiye poligonu oluştur
        turkey_polygon = Polygon([
            (26, 42), (26, 36), (44, 36), (44, 42), (26, 42)
        ])
        gdf = gpd.GeoDataFrame({
            'name': ['Türkiye'],
            'geometry': [turkey_polygon]
        })
        return gdf

# =============================================================================
# GELİŞMİŞ HARİTA GÖRSELLEŞTİRME
# =============================================================================

class AdvancedMapVisualizer:
    """Gelişmiş harita görselleştirme sınıfı"""
    
    @staticmethod
    def create_region_map(city_data: pd.DataFrame, gdf, title: str = "Türkiye Satış Haritası") -> go.Figure:
        """Bölge bazlı harita oluştur"""
        # Şehir verilerini bölge bazında topla
        region_data = city_data.groupby('Region').agg({
            'PF_Satis': 'sum',
            'Rakip_Satis': 'sum',
            'City': 'count'
        }).reset_index()
        
        region_data.columns = ['Region', 'PF_Satis', 'Rakip_Satis', 'City_Count']
        region_data['Toplam_Pazar'] = region_data['PF_Satis'] + region_data['Rakip_Satis']
        region_data['Pazar_Payi_%'] = (region_data['PF_Satis'] / region_data['Toplam_Pazar'] * 100).round(2)
        
        # GeoJSON verisini hazırla
        gdf = gdf.copy()
        gdf['name_upper'] = gdf['name'].apply(lambda x: TurkishTextNormalizer.normalize_text(x))
        
        # Şehir bölge eşleştirmesi
        city_region_map = {}
        for _, row in city_data.iterrows():
            city_region_map[row['City']] = row['Region']
        
        # GeoJSON'daki her şehri bölgesine göre renklendir
        gdf['Region'] = gdf['name_upper'].map(city_region_map)
        gdf['Region'] = gdf['Region'].fillna('DİĞER')
        
        # Bölge renklerini ata
        gdf['Color'] = gdf['Region'].map(REGION_COLORS).fillna('#64748B')
        
        # Harita oluştur
        fig = go.Figure()
        
        # Her bölge için ayrı trace
        for region in gdf['Region'].unique():
            region_gdf = gdf[gdf['Region'] == region]
            color = REGION_COLORS.get(region, "#64748B")
            
            # GeoJSON'u ekle
            fig.add_trace(go.Choroplethmapbox(
                geojson=json.loads(region_gdf.to_json()),
                locations=region_gdf.index,
                z=[1] * len(region_gdf),
                colorscale=[[0, color], [1, color]],
                marker_opacity=0.7,
                marker_line_width=1,
                marker_line_color='rgba(255, 255, 255, 0.8)',
                showscale=False,
                name=region,
                hoverinfo='skip'
            ))
        
        # Bölge etiketleri ekle
        if region_data is not None and len(region_data) > 0:
            label_lons, label_lats, label_texts = [], [], []
            
            for _, row in region_data.iterrows():
                region_cities = city_data[city_data['Region'] == row['Region']]
                if len(region_cities) > 0:
                    # Bölge merkezini hesapla
                    region_coords = []
                    for city in region_cities['City'].unique():
                        if city in TURKEY_CITIES:
                            region_coords.append((
                                TURKEY_CITIES[city]['lon'],
                                TURKEY_CITIES[city]['lat']
                            ))
                    
                    if region_coords:
                        avg_lon = np.mean([c[0] for c in region_coords])
                        avg_lat = np.mean([c[1] for c in region_coords])
                        
                        label_lons.append(avg_lon)
                        label_lats.append(avg_lat)
                        label_texts.append(
                            f"<b>{row['Region']}</b><br>"
                            f"PF: {row['PF_Satis']:,.0f}<br>"
                            f"Pay: {row['Pazar_Payi_%']:.1f}%"
                        )
            
            fig.add_trace(go.Scattermapbox(
                lon=label_lons,
                lat=label_lats,
                mode='text',
                text=label_texts,
                textfont=dict(
                    size=11,
                    color='white',
                    family='Inter, sans-serif',
                    weight='bold'
                ),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Harita ayarları
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox=dict(
                center=dict(lat=39.0, lon=35.0),
                zoom=4.5,
                bearing=0,
                pitch=0
            ),
            height=700,
            margin=dict(l=0, r=0, t=80, b=0),
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(
                    size=24,
                    color='white',
                    family='Poppins, sans-serif'
                ),
                y=0.95
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            hoverlabel=dict(
                bgcolor="rgba(15, 23, 41, 0.95)",
                font_size=12,
                font_family="Inter, sans-serif"
            )
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
            return None
        
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
                mode='markers+text',
                marker=dict(
                    size=region_data['PF_Satis'] / region_data['PF_Satis'].max() * 40 + 10,
                    color=color,
                    opacity=0.8,
                    sizemode='diameter',
                    sizeref=2 * max(region_data['PF_Satis']) / (40 ** 2),
                    sizemin=8
                ),
                text=region_data['City'],
                textposition='top center',
                textfont=dict(
                    size=10,
                    color='white',
                    family='Inter, sans-serif'
                ),
                customdata=np.stack((
                    region_data['City'],
                    region_data['Region'],
                    region_data['PF_Satis'],
                    region_data['Rakip_Satis'],
                    region_data['Pazar_Payi_%']
                ), axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Bölge: %{customdata[1]}<br>"
                    "PF Satış: %{customdata[2]:,.0f}<br>"
                    "Rakip Satış: %{customdata[3]:,.0f}<br>"
                    "Pazar Payı: %{customdata[4]:.1f}%"
                    "<extra></extra>"
                ),
                name=region,
                showlegend=True
            ))
        
        # Harita ayarları
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox=dict(
                center=dict(lat=39.0, lon=35.0),
                zoom=4.5,
                bearing=0,
                pitch=0
            ),
            height=700,
            margin=dict(l=0, r=0, t=80, b=0),
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(
                    size=24,
                    color='white',
                    family='Poppins, sans-serif'
                ),
                y=0.95
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                title='<b>Bölgeler</b>',
                bgcolor='rgba(30, 41, 59, 0.8)',
                bordercolor='rgba(59, 130, 246, 0.3)',
                borderwidth=1,
                x=0.01,
                y=0.99
            ),
            hoverlabel=dict(
                bgcolor="rgba(15, 23, 41, 0.95)",
                font_size=12,
                font_family="Inter, sans-serif"
            )
        )
        
        return fig
    
    @staticmethod
    def create_3d_surface_map(city_data: pd.DataFrame) -> go.Figure:
        """3D yüzey haritası oluştur"""
        # Şehir koordinatlarını ve satışları hazırla
        points = []
        for _, row in city_data.iterrows():
            city = row['City']
            if city in TURKEY_CITIES:
                points.append({
                    'x': TURKEY_CITIES[city]['lon'],
                    'y': TURKEY_CITIES[city]['lat'],
                    'z': row['PF_Satis']
                })
        
        if len(points) < 10:
            return None
        
        df_points = pd.DataFrame(points)
        
        # 3D surface plot
        fig = go.Figure(data=[
            go.Mesh3d(
                x=df_points['x'],
                y=df_points['y'],
                z=df_points['z'],
                colorscale='Viridis',
                intensity=df_points['z'],
                opacity=0.8,
                name='Satış Dağılımı'
            )
        ])
        
        fig.update_layout(
            title='<b>3D Satış Dağılım Haritası</b>',
            scene=dict(
                xaxis_title='Boylam',
                yaxis_title='Enlem',
                zaxis_title='PF Satış',
                bgcolor='rgba(0,0,0,0)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0')
        )
        
        return fig

# =============================================================================
# İLERİ SEVİYE ML VE AI FONKSİYONLARI
# =============================================================================

class AdvancedMLForecaster:
    """İleri seviye ML tahminleme sınıfı"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def create_advanced_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Gelişmiş zaman serisi özellikleri oluştur"""
        df_features = df.copy().sort_values('DATE').reset_index(drop=True)
        
        # Temel lag özellikleri
        for lag in [1, 2, 3, 6, 12]:
            df_features[f'lag_{lag}'] = df_features[target_col].shift(lag)
        
        # Rolling istatistikleri
        windows = [3, 6, 12, 24]
        for window in windows:
            df_features[f'rolling_mean_{window}'] = df_features[target_col].rolling(window=window, min_periods=1).mean()
            df_features[f'rolling_std_{window}'] = df_features[target_col].rolling(window=window, min_periods=1).std()
            df_features[f'rolling_min_{window}'] = df_features[target_col].rolling(window=window, min_periods=1).min()
            df_features[f'rolling_max_{window}'] = df_features[target_col].rolling(window=window, min_periods=1).max()
            df_features[f'rolling_median_{window}'] = df_features[target_col].rolling(window=window, min_periods=1).median()
            df_features[f'rolling_skew_{window}'] = df_features[target_col].rolling(window=window, min_periods=1).skew()
            df_features[f'rolling_kurt_{window}'] = df_features[target_col].rolling(window=window, min_periods=1).kurt()
        
        # Exponential moving averages
        for span in [3, 6, 12, 24]:
            df_features[f'ema_{span}'] = df_features[target_col].ewm(span=span, adjust=False).mean()
        
        # Momentum özellikleri
        for period in [1, 3, 6, 12]:
            df_features[f'momentum_{period}'] = df_features[target_col] - df_features[target_col].shift(period)
            df_features[f'roc_{period}'] = (df_features[target_col] - df_features[target_col].shift(period)) / df_features[target_col].shift(period) * 100
        
        # Volatilite özellikleri
        df_features['volatility_6'] = df_features[target_col].rolling(window=6).std() / df_features[target_col].rolling(window=6).mean()
        df_features['volatility_12'] = df_features[target_col].rolling(window=12).std() / df_features[target_col].rolling(window=12).mean()
        
        # Zaman özellikleri
        df_features['month'] = df_features['DATE'].dt.month
        df_features['quarter'] = df_features['DATE'].dt.quarter
        df_features['year'] = df_features['DATE'].dt.year
        df_features['day_of_year'] = df_features['DATE'].dt.dayofyear
        df_features['week_of_year'] = df_features['DATE'].dt.isocalendar().week
        
        # Trigonometrik mevsimsellik
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
        df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
        
        # Trend özellikleri
        df_features['linear_trend'] = np.arange(len(df_features))
        df_features['quadratic_trend'] = df_features['linear_trend'] ** 2
        
        # Mevsimsel dummy değişkenler
        for month in range(1, 13):
            df_features[f'is_month_{month}'] = (df_features['month'] == month).astype(int)
        
        for quarter in range(1, 5):
            df_features[f'is_quarter_{quarter}'] = (df_features['quarter'] == quarter).astype(int)
        
        # Yıl bazlı büyüme
        df_features['yoy_growth'] = df_features[target_col].pct_change(periods=12) * 100
        
        # Anomali skorları
        df_features['z_score'] = np.abs(stats.zscore(df_features[target_col].fillna(0)))
        df_features['iqr_score'] = self._calculate_iqr_score(df_features[target_col])
        
        # Fourier dönüşüm özellikleri
        df_features = self._add_fourier_features(df_features, target_col)
        
        # Eksik değerleri doldur
        df_features = df_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df_features
    
    def _calculate_iqr_score(self, series):
        """IQR tabanlı anomali skoru hesapla"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).astype(int)
    
    def _add_fourier_features(self, df, target_col, n_components=3):
        """Fourier dönüşüm özellikleri ekle"""
        series = df[target_col].values
        n = len(series)
        
        # FFT uygula
        fft_values = np.fft.fft(series)
        frequencies = np.fft.fftfreq(n)
        
        # En önemli frekansları seç
        idx = np.argsort(np.abs(fft_values))[::-1][1:n_components+1]
        
        for i, freq_idx in enumerate(idx[:n_components]):
            freq = frequencies[freq_idx]
            magnitude = np.abs(fft_values[freq_idx])
            phase = np.angle(fft_values[freq_idx])
            
            df[f'fourier_sin_{i+1}'] = magnitude * np.sin(2 * np.pi * freq * np.arange(n) + phase)
            df[f'fourier_cos_{i+1}'] = magnitude * np.cos(2 * np.pi * freq * np.arange(n) + phase)
        
        return df
    
    def train_ensemble_models(self, X_train, y_train, X_val, y_val):
        """Ensemble ML modellerini eğit"""
        models = {}
        
        # 1. XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        models['XGBoost'] = xgb_model
        
        # 2. LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        models['LightGBM'] = lgb_model
        
        # 3. CatBoost
        cat_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=7,
            random_seed=42,
            verbose=False
        )
        cat_model.fit(X_train, y_train)
        models['CatBoost'] = cat_model
        
        # 4. Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models['RandomForest'] = rf_model
        
        # 5. Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        models['GradientBoosting'] = gb_model
        
        # 6. SVR
        svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        svr_model.fit(X_train, y_train)
        models['SVR'] = svr_model
        
        # 7. Neural Network
        nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        nn_model.fit(X_train, y_train)
        models['NeuralNetwork'] = nn_model
        
        # 8. Ensemble Model (Voting Regressor)
        ensemble_model = VotingRegressor([
            ('xgb', models['XGBoost']),
            ('lgb', models['LightGBM']),
            ('cat', models['CatBoost']),
            ('rf', models['RandomForest'])
        ])
        ensemble_model.fit(X_train, y_train)
        models['Ensemble'] = ensemble_model
        
        self.models = models
        return models
    
    def hyperparameter_optimization(self, X_train, y_train, X_val, y_val):
        """Hyperparameter optimizasyonu"""
        def objective(trial):
            # Hyperparameter space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
            }
            
            model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train, verbose=False)
            
            y_pred = model.predict(X_val)
            mape = mean_absolute_percentage_error(y_val, y_pred)
            
            return mape
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        # En iyi modeli eğit
        best_params = study.best_params
        best_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        best_model.fit(
            np.vstack([X_train, X_val]),
            np.concatenate([y_train, y_val])
        )
        
        self.models['Optimized_XGBoost'] = best_model
        return best_model, study.best_value
    
    def forecast(self, df: pd.DataFrame, target_col: str, forecast_periods: int = 12):
        """Çoklu ML modelleri ile tahmin"""
        if len(df) < 24:
            return None, None, "⚠️ Yetersiz veri (en az 24 ay gerekli)"
        
        try:
            # Özellik mühendisliği
            df_features = self.create_advanced_features(df, target_col)
            
            # Özellik sütunları
            feature_cols = [col for col in df_features.columns 
                           if col not in ['DATE', 'YEAR_MONTH', target_col] 
                           and not col.startswith('is_')]
            
            # Zaman serisi split
            train_size = int(len(df_features) * 0.7)
            val_size = int(len(df_features) * 0.15)
            
            train_df = df_features.iloc[:train_size]
            val_df = df_features.iloc[train_size:train_size+val_size]
            test_df = df_features.iloc[train_size+val_size:]
            
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_val = val_df[feature_cols]
            y_val = val_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]
            
            # Modelleri eğit
            models = self.train_ensemble_models(X_train, y_train, X_val, y_val)
            
            # Tahmin ve değerlendirme
            results = {}
            forecasts = {}
            
            for name, model in models.items():
                # Test seti tahmini
                y_pred = model.predict(X_test)
                
                # Metrikler
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'R2': r2,
                    'predictions': y_pred
                }
                
                # Gelecek tahmini
                forecast_values = self._generate_forecast(
                    model, df_features, feature_cols, target_col, forecast_periods
                )
                forecasts[name] = forecast_values
            
            # En iyi modeli seç
            best_model_name = min(results.keys(), key=lambda x: results[x]['MAPE'])
            best_model = results[best_model_name]['model']
            
            # Ensemble tahmini (tüm modellerin ortalaması)
            ensemble_forecast = np.mean([forecasts[name] for name in forecasts], axis=0)
            
            # Tahmin dataframe'i oluştur
            last_date = df['DATE'].max()
            forecast_dates = [last_date + pd.DateOffset(months=i+1) 
                            for i in range(forecast_periods)]
            
            forecast_df = pd.DataFrame({
                'DATE': forecast_dates,
                'YIL_AY': [d.strftime('%Y-%m') for d in forecast_dates],
                'PF_Satis': ensemble_forecast,
                'Model': 'Ensemble',
                'Tahmin_Tipi': 'ML Ensemble Tahmin',
                'Confidence_Low': ensemble_forecast * 0.85,  # %85 güven aralığı
                'Confidence_High': ensemble_forecast * 1.15   # %115 güven aralığı
            })
            
            # Prophet tahmini ekle
            prophet_forecast = self._prophet_forecast(df, target_col, forecast_periods)
            if prophet_forecast is not None:
                forecast_df = pd.concat([forecast_df, prophet_forecast], ignore_index=True)
            
            # SHAP analizi
            if 'Ensemble' in models:
                shap_values = self._calculate_shap(models['Ensemble'], X_train)
                self.feature_importance = {
                    'features': feature_cols,
                    'importance': shap_values
                }
            
            return results, forecast_df, "✅ ML modelleri başarıyla eğitildi"
            
        except Exception as e:
            return None, None, f"❌ ML eğitimi hatası: {str(e)}"
    
    def _generate_forecast(self, model, df_features, feature_cols, target_col, periods):
        """Tekil model için tahmin üret"""
        forecast = []
        last_row = df_features.iloc[-1:].copy()
        
        for _ in range(periods):
            # Bir sonraki ay
            next_date = last_row['DATE'].values[0] + pd.DateOffset(months=1)
            
            # Özellikleri hazırla
            X_future = last_row[feature_cols]
            
            # Tahmin yap
            next_pred = model.predict(X_future)[0]
            next_pred = max(0, next_pred)
            forecast.append(next_pred)
            
            # Özellikleri güncelle (recursive forecasting için)
            new_row = last_row.copy()
            new_row['DATE'] = next_date
            new_row[target_col] = next_pred
            
            # Lag değerlerini güncelle
            for lag in [1, 2, 3, 6, 12]:
                if f'lag_{lag}' in new_row.columns:
                    if lag == 1:
                        new_row[f'lag_{lag}'] = last_row[target_col].values[0]
                    else:
                        prev_lag = f'lag_{lag-1}'
                        if prev_lag in last_row.columns:
                            new_row[f'lag_{lag}'] = last_row[prev_lag].values[0]
            
            # Tarih özelliklerini güncelle
            new_row['month'] = next_date.month
            new_row['quarter'] = next_date.quarter
            new_row['year'] = next_date.year
            new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
            new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)
            new_row['linear_trend'] = last_row['linear_trend'].values[0] + 1
            
            last_row = new_row
        
        return forecast
    
    def _prophet_forecast(self, df, target_col, periods):
        """Prophet modeli ile tahmin"""
        try:
            prophet_df = df[['DATE', target_col]].copy()
            prophet_df.columns = ['ds', 'y']
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10
            )
            
            # Özel mevsimsellik ekle
            model.add_seasonality(
                name='quarterly',
                period=91.25,
                fourier_order=5
            )
            
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=periods, freq='MS')
            forecast = model.predict(future)
            
            # Son periods kadarını al
            prophet_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            prophet_result.columns = ['DATE', 'PF_Satis', 'Confidence_Low', 'Confidence_High']
            prophet_result['Model'] = 'Prophet'
            prophet_result['Tahmin_Tipi'] = 'Prophet Time Series'
            
            return prophet_result
            
        except Exception as e:
            st.warning(f"Prophat modeli hatası: {str(e)}")
            return None
    
    def _calculate_shap(self, model, X_train):
        """SHAP değerlerini hesapla"""
        try:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_train)
            
            # Ortalama SHAP değerleri
            mean_shap = np.abs(shap_values.values).mean(axis=0)
            
            return dict(zip(X_train.columns, mean_shap))
        except:
            return {}

class TimeSeriesAnalyzer:
    """İleri seviye zaman serisi analizi"""
    
    @staticmethod
    def decompose_time_series(df: pd.DataFrame, target_col: str, period: int = 12):
        """Zaman serisi ayrıştırma"""
        try:
            # Mevsimsel ayrıştırma
            decomposition = seasonal_decompose(
                df[target_col].fillna(method='ffill'),
                model='multiplicative',
                period=period
            )
            
            return {
                'observed': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
        except Exception as e:
            st.warning(f"Ayrıştırma hatası: {str(e)}")
            return None
    
    @staticmethod
    def calculate_advanced_metrics(df: pd.DataFrame, target_col: str):
        """İleri seviye zaman serisi metrikleri"""
        series = df[target_col].fillna(method='ffill')
        
        metrics = {}
        
        # Temel istatistikler
        metrics['mean'] = series.mean()
        metrics['std'] = series.std()
        metrics['cv'] = metrics['std'] / metrics['mean'] if metrics['mean'] != 0 else 0
        metrics['skewness'] = series.skew()
        metrics['kurtosis'] = series.kurtosis()
        
        # Trend metrikleri
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        metrics['trend_slope'] = slope
        metrics['trend_strength'] = np.corrcoef(x, series)[0, 1] ** 2
        
        # Volatilite metrikleri
        returns = series.pct_change().dropna()
        metrics['volatility'] = returns.std() * np.sqrt(12)  # Yıllık volatilite
        metrics['sharpe_ratio'] = returns.mean() / returns.std() if returns.std() != 0 else 0
        
        # Mevsimsellik metrikleri
        if len(series) >= 24:
            seasonal_decomp = TimeSeriesAnalyzer.decompose_time_series(df, target_col)
            if seasonal_decomp:
                seasonal = seasonal_decomp['seasonal'].dropna()
                if len(seasonal) > 0:
                    metrics['seasonality_strength'] = np.std(seasonal) / np.std(series)
        
        # Durağanlık testi
        try:
            adf_result = adfuller(series.dropna())
            metrics['adf_statistic'] = adf_result[0]
            metrics['adf_pvalue'] = adf_result[1]
            metrics['is_stationary'] = adf_result[1] < 0.05
        except:
            metrics['is_stationary'] = False
        
        return metrics
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame, target_col: str):
        """Zaman serisi pattern'lerini tespit et"""
        series = df[target_col].fillna(method='ffill').values
        
        patterns = {
            'trend': None,
            'seasonality': None,
            'cyclicality': None,
            'anomalies': [],
            'breakpoints': []
        }
        
        # Trend analizi
        try:
            # Hodrick-Prescott filtresi
            cycle, trend = sm.tsa.filters.hpfilter(series, lamb=1600)
            patterns['trend'] = {
                'direction': 'up' if trend[-1] > trend[0] else 'down',
                'strength': np.corrcoef(np.arange(len(trend)), trend)[0, 1] ** 2
            }
        except:
            pass
        
        # Anomali tespiti
        try:
            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(series.reshape(-1, 1))
            anomaly_indices = np.where(anomalies == -1)[0]
            
            for idx in anomaly_indices:
                if idx < len(df):
                    patterns['anomalies'].append({
                        'date': df.iloc[idx]['DATE'] if 'DATE' in df.columns else idx,
                        'value': series[idx],
                        'type': 'high' if series[idx] > np.median(series) else 'low'
                    })
        except:
            pass
        
        # Kırılma noktaları
        try:
            if len(series) > 20:
                # Chow testi benzeri basit kırılma tespiti
                n = len(series)
                potential_breaks = []
                
                for i in range(10, n-10):
                    before = series[:i]
                    after = series[i:]
                    
                    mean_before = np.mean(before)
                    mean_after = np.mean(after)
                    
                    # Basit fark testi
                    if abs(mean_after - mean_before) > 2 * np.std(series):
                        potential_breaks.append(i)
                
                patterns['breakpoints'] = potential_breaks
        except:
            pass
        
        return patterns

# =============================================================================
# GELİŞMİŞ ANALİZ FONKSİYONLARI
# =============================================================================

class AdvancedAnalytics:
    """Gelişmiş analiz fonksiyonları"""
    
    @staticmethod
    def calculate_market_intelligence(df, product_cols):
        """Pazar zekası analizi"""
        intelligence = {}
        
        # Pazar büyüklüğü
        intelligence['market_size'] = {
            'total': df[product_cols['pf']].sum() + df[product_cols['rakip']].sum(),
            'pf_share': df[product_cols['pf']].sum(),
            'competitor_share': df[product_cols['rakip']].sum()
        }
        
        # Büyüme analizi
        monthly = df.groupby('YEAR_MONTH').agg({
            product_cols['pf']: 'sum',
            product_cols['rakip']: 'sum'
        }).reset_index()
        
        if len(monthly) > 1:
            pf_growth = (monthly[product_cols['pf']].iloc[-1] / monthly[product_cols['pf']].iloc[0] - 1) * 100
            comp_growth = (monthly[product_cols['rakip']].iloc[-1] / monthly[product_cols['rakip']].iloc[0] - 1) * 100
            
            intelligence['growth'] = {
                'pf_growth_%': pf_growth,
                'competitor_growth_%': comp_growth,
                'relative_growth_%': pf_growth - comp_growth
            }
        
        # Yoğunluk analizi
        territory_perf = df.groupby('TERRITORIES').agg({
            product_cols['pf']: 'sum',
            product_cols['rakip']: 'sum'
        }).reset_index()
        
        territory_perf['share'] = territory_perf[product_cols['pf']] / (territory_perf[product_cols['pf']] + territory_perf[product_cols['rakip']])
        
        # Herfindahl-Hirschman Index (HHI)
        market_shares = territory_perf['share'].fillna(0)
        intelligence['concentration'] = {
            'hhi': (market_shares ** 2).sum() * 10000,
            'cr4': territory_perf.nlargest(4, product_cols['pf'])[product_cols['pf']].sum() / territory_perf[product_cols['pf']].sum() * 100,
            'cr8': territory_perf.nlargest(8, product_cols['pf'])[product_cols['pf']].sum() / territory_perf[product_cols['pf']].sum() * 100
        }
        
        # Rekabet analizi
        intelligence['competition'] = {
            'avg_competitor_share': territory_perf[product_cols['rakip']].mean(),
            'max_competitor_share': territory_perf[product_cols['rakip']].max(),
            'dominant_territories': territory_perf.nlargest(3, product_cols['rakip'])['TERRITORIES'].tolist()
        }
        
        return intelligence
    
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
        territory_perf['Pazar_Payi_%'] = (territory_perf['PF_Satis'] / territory_perf['Toplam_Pazar'] * 100).fillna(0)
        territory_perf['Goreceli_Pazar_Payi'] = (territory_perf['PF_Satis'] / territory_perf['Rakip_Satis']).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Büyüme oranı hesapla
        if len(df_filtered) >= 12:
            df_sorted = df_filtered.sort_values('DATE')
            mid_point = len(df_sorted) // 2
            
            first_half = df_sorted.iloc[:mid_point].groupby('TERRITORIES')[product_cols['pf']].sum()
            second_half = df_sorted.iloc[mid_point:].groupby('TERRITORIES')[product_cols['pf']].sum()
            
            growth_rates = {}
            for terr in territory_perf['Territory']:
                if terr in first_half.index and terr in second_half.index and first_half[terr] > 0:
                    growth_rates[terr] = ((second_half[terr] - first_half[terr]) / first_half[terr]) * 100
                else:
                    growth_rates[terr] = 0
            
            territory_perf['Buyume_%'] = territory_perf['Territory'].map(growth_rates).fillna(0)
        else:
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
        regional = df_filtered.groupby('REGION').agg({
            product_cols['pf']: 'sum',
            product_cols['rakip']: 'sum'
        }).reset_index()
        
        regional['share'] = regional[product_cols['pf']] / (regional[product_cols['pf']] + regional[product_cols['rakip']])
        
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
        
        # 3. Seasonal içgörüler
        if 'MONTH' in df_filtered.columns:
            monthly_avg = df_filtered.groupby('MONTH')[product_cols['pf']].mean()
            peak_month = monthly_avg.idxmax()
            low_month = monthly_avg.idxmin()
            
            insights.append({
                'type': 'info',
                'title': '📅 Mevsimsel Trend',
                'description': f"En yüksek satışlar {peak_month}. ayında, en düşük satışlar {low_month}. ayında gerçekleşiyor.",
                'impact': 'low'
            })
        
        # 4. Manager performansı
        if 'MANAGER' in df_filtered.columns:
            manager_perf = df_filtered.groupby('MANAGER')[product_cols['pf']].sum().nlargest(3)
            if len(manager_perf) > 0:
                top_manager = manager_perf.index[0]
                insights.append({
                    'type': 'positive',
                    'title': '👑 En Başarılı Manager',
                    'description': f"{top_manager} en yüksek satış performansını gösterdi.",
                    'impact': 'medium'
                })
        
        return insights

# =============================================================================
# GELİŞMİŞ GÖRSELLEŞTİRME FONKSİYONLARI
# =============================================================================

class AdvancedVisualizations:
    """Gelişmiş görselleştirme sınıfı"""
    
    @staticmethod
    def create_advanced_dashboard(metrics_dict):
        """Advanced metrik dashboard'u oluştur"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                '📈 PF Satış Trendi',
                '🏆 Bölge Performansı',
                '📊 Pazar Payı Dağılımı',
                '📉 Büyüme Oranları',
                '🎯 BCG Matrix',
                '📅 Mevsimsellik Analizi'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'heatmap'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Satış trendi
        if 'sales_trend' in metrics_dict:
            fig.add_trace(
                go.Scatter(
                    x=metrics_dict['sales_trend']['dates'],
                    y=metrics_dict['sales_trend']['values'],
                    mode='lines+markers',
                    name='PF Satış',
                    line=dict(color=PERFORMANCE_COLORS['positive'], width=3),
                    marker=dict(size=8, color='white')
                ),
                row=1, col=1
            )
        
        # Bölge performansı
        if 'region_performance' in metrics_dict:
            fig.add_trace(
                go.Bar(
                    x=metrics_dict['region_performance']['regions'],
                    y=metrics_dict['region_performance']['sales'],
                    name='Bölge Satış',
                    marker_color=list(REGION_COLORS.values())[:len(metrics_dict['region_performance']['regions'])]
                ),
                row=1, col=2
            )
        
        # Pazar payı dağılımı
        if 'market_share' in metrics_dict:
            fig.add_trace(
                go.Pie(
                    labels=['PF', 'Rakip'],
                    values=[metrics_dict['market_share']['pf'], metrics_dict['market_share']['competitor']],
                    name='Pazar Payı',
                    marker_colors=[PERFORMANCE_COLORS['positive'], PERFORMANCE_COLORS['negative']],
                    hole=0.4
                ),
                row=1, col=3
            )
        
        # Büyüme oranları
        if 'growth_rates' in metrics_dict:
            fig.add_trace(
                go.Bar(
                    x=['PF Büyüme', 'Rakip Büyüme'],
                    y=[metrics_dict['growth_rates']['pf'], metrics_dict['growth_rates']['competitor']],
                    name='Büyüme Oranları',
                    marker_color=[PERFORMANCE_COLORS['positive'], PERFORMANCE_COLORS['negative']]
                ),
                row=2, col=1
            )
        
        # BCG Matrix
        if 'bcg_data' in metrics_dict:
            fig.add_trace(
                go.Scatter(
                    x=metrics_dict['bcg_data']['relative_share'],
                    y=metrics_dict['bcg_data']['growth'],
                    mode='markers',
                    name='BCG Matrix',
                    marker=dict(
                        size=metrics_dict['bcg_data']['size'],
                        color=metrics_dict['bcg_data']['color'],
                        opacity=0.7
                    ),
                    text=metrics_dict['bcg_data']['labels']
                ),
                row=2, col=2
            )
        
        # Mevsimsellik heatmap
        if 'seasonality' in metrics_dict:
            fig.add_trace(
                go.Heatmap(
                    z=metrics_dict['seasonality']['matrix'],
                    x=metrics_dict['seasonality']['months'],
                    y=metrics_dict['seasonality']['years'],
                    colorscale='Viridis',
                    name='Mevsimsellik'
                ),
                row=2, col=3
            )
        
        fig.update_layout(
            height=900,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title=dict(
                text='<b>📊 Advanced Ticari Performans Dashboard</b>',
                x=0.5,
                font=dict(size=24, color='white', family='Poppins')
            )
        )
        
        return fig
    
    @staticmethod
    def create_forecast_comparison_chart(historical_df, forecast_df, title="Satış Tahmin Karşılaştırması"):
        """Çoklu tahmin modeli karşılaştırma grafiği"""
        fig = go.Figure()
        
        # Gerçek veri
        fig.add_trace(go.Scatter(
            x=historical_df['DATE'],
            y=historical_df['PF_Satis'],
            mode='lines+markers',
            name='Gerçek Satış',
            line=dict(color=PERFORMANCE_COLORS['positive'], width=3),
            marker=dict(size=8, color='white', symbol='circle'),
            fillcolor='rgba(16, 185, 129, 0.1)',
            fill='tozeroy'
        ))
        
        # Tahminler
        if forecast_df is not None and len(forecast_df) > 0:
            # Farklı modeller için farklı renkler
            model_colors = {
                'Ensemble': '#3B82F6',
                'XGBoost': '#10B981',
                'Prophet': '#8B5CF6',
                'LightGBM': '#F59E0B',
                'RandomForest': '#EF4444'
            }
            
            for model in forecast_df['Model'].unique():
                model_data = forecast_df[forecast_df['Model'] == model]
                color = model_colors.get(model, '#64748B')
                
                fig.add_trace(go.Scatter(
                    x=model_data['DATE'],
                    y=model_data['PF_Satis'],
                    mode='lines+markers',
                    name=f'{model} Tahmini',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=6, symbol='diamond'),
                    opacity=0.8
                ))
                
                # Güven aralığı
                if 'Confidence_Low' in model_data.columns and 'Confidence_High' in model_data.columns:
                    fig.add_trace(go.Scatter(
                        x=model_data['DATE'].tolist() + model_data['DATE'].tolist()[::-1],
                        y=model_data['Confidence_High'].tolist() + model_data['Confidence_Low'].tolist()[::-1],
                        fill='toself',
                        fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{model} Güven Aralığı',
                        showlegend=False
                    ))
        
        fig.update_layout(
            title=dict(
                text=f'<b>{title}</b>',
                x=0.5,
                font=dict(size=24, color='white', family='Poppins')
            ),
            xaxis_title='<b>Tarih</b>',
            yaxis_title='<b>Satış</b>',
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(30, 41, 59, 0.8)'
            ),
            hovermode='x unified'
        )
        
        return fig

# =============================================================================
# ANA UYGULAMA
# =============================================================================

def main():
    # Başlık
    st.markdown('<h1 class="main-header">🎯 PROFESYONEL TİCARİ PORTFÖY ANALİZ SİSTEMİ</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.3rem; color: #94a3b8; margin-bottom: 3rem; line-height: 1.8;">'
                '🤖 AI Destekli Tahminleme • 🗺️ Interactive Haritalar • 📊 50+ Performans Metrik • 🎯 Otomatik İçgörü Üretimi<br>'
                '🔮 Senaryo Analizleri • 📉 Risk Yönetimi • 💼 Stratejik Portföy Optimizasyonu'
                '</div>', unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown('<div style="background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%); '
                   'padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem; text-align: center;">'
                   '<h3 style="color: white; margin: 0; font-size: 1.5rem;">📂 VERİ YÜKLEME</h3>'
                   '<p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">'
                   'Excel dosyanızı yükleyin veya demo veri ile devam edin</p>'
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
        if 'TROCMETAM_PF' in sales_columns:
            available_products.append('TROCMETAM')
        if 'CORTIPOL_PF' in sales_columns:
            available_products.append('CORTIPOL')
        if 'DEKSAMETAZON_PF' in sales_columns:
            available_products.append('DEKSAMETAZON')
        if 'IZOTONIK_PF' in sales_columns:
            available_products.append('PF IZOTONIK')
        
        if not available_products:
            available_products = ['TROCMETAM', 'CORTIPOL', 'DEKSAMETAZON', 'PF IZOTONIK']
        
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
                "🗓️ Son 2 Yıl",
                "🎯 Özel Aralık"
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
            elif date_option == "🗓️ Son 2 Yıl":
                start_date = max_date - pd.DateOffset(years=2)
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
            territories = ["🏢 TÜMÜ"] + sorted(df['TERRITORIES'].fillna('BILINMIYOR').unique())
            selected_territory = st.selectbox("Territory", territories)
        else:
            selected_territory = "🏢 TÜMÜ"
        
        # Bölge filtre
        if 'REGION' in df.columns:
            regions = ["🗺️ TÜMÜ"] + sorted(df['REGION'].fillna('DİĞER').unique())
            selected_region = st.selectbox("Bölge", regions)
        else:
            selected_region = "🗺️ TÜMÜ"
        
        # Manager filtre
        if 'MANAGER' in df.columns:
            managers = ["👤 TÜMÜ"] + sorted(df['MANAGER'].fillna('BILINMIYOR').unique())
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
            ["🗺️ Bölge Haritası", "🏙️ Şehir Haritası", "📊 3D Dağılım Haritası"],
            index=0
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ML Ayarları
        st.markdown('<div style="background: rgba(30, 41, 59, 0.8); padding: 1.25rem; border-radius: 14px; margin: 1rem 0; border-left: 4px solid #EF4444;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0; font-size: 1.1rem;">🤖 AI/ML AYARLARI</h4>', unsafe_allow_html=True)
        
        forecast_periods = st.slider("Tahmin Periyodu (Ay)", 1, 24, 12)
        ml_mode = st.selectbox(
            "ML Model Seçimi",
            ["🤖 Ensemble Model", "🚀 XGBoost", "💡 LightGBM", "🐱 CatBoost", "🌲 Random Forest", "📈 Tüm Modeller"]
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
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
        
        # GeoJSON yükle
        gdf = load_turkey_geojson()
        
        # Ürün kolonlarını belirle
        product_cols = {}
        if selected_product == 'TROCMETAM':
            product_cols = {'pf': sales_columns.get('TROCMETAM_PF', 'TROCMETAM'),
                           'rakip': sales_columns.get('TROCMETAM_COMPETITOR', 'DIGER TROCMETAM')}
        elif selected_product == 'CORTIPOL':
            product_cols = {'pf': sales_columns.get('CORTIPOL_PF', 'CORTIPOL'),
                           'rakip': sales_columns.get('CORTIPOL_COMPETITOR', 'DIGER CORTIPOL')}
        elif selected_product == 'DEKSAMETAZON':
            product_cols = {'pf': sales_columns.get('DEKSAMETAZON_PF', 'DEKSAMETAZON'),
                           'rakip': sales_columns.get('DEKSAMETAZON_COMPETITOR', 'DIGER DEKSAMETAZON')}
        else:
            product_cols = {'pf': sales_columns.get('IZOTONIK_PF', 'PF IZOTONIK'),
                           'rakip': sales_columns.get('IZOTONIK_COMPETITOR', 'DIGER IZOTONIK')}
        
        # Şehir performansını hesapla
        city_perf = calculate_city_performance(df_filtered, product_cols)
    
    # ANA İÇERİK - TAB'LER
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Executive Dashboard",
        "🗺️ Harita Analizi", 
        "🏢 Territory Analizi",
        "📈 Zaman Serisi & AI",
        "🎯 Rakip & Pazar Analizi",
        "⭐ BCG & Strateji",
        "🔮 Tahmin & Senaryo",
        "📥 Raporlar & Export"
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
                value=f"{total_pf:,.0f}",
                delta=f"{total_pf/len(df_filtered['YEAR_MONTH'].unique()):,.0f}/ay"
            )
        
        with col2:
            total_market = df_filtered[product_cols['pf']].sum() + df_filtered[product_cols['rakip']].sum()
            market_share = (df_filtered[product_cols['pf']].sum() / total_market * 100) if total_market > 0 else 0
            st.metric(
                label="📊 Pazar Payı",
                value=f"%{market_share:.1f}",
                delta=f"%{100-market_share:.1f} rakip"
            )
        
        with col3:
            active_territories = df_filtered['TERRITORIES'].nunique()
            st.metric(
                label="🏢 Aktif Territory",
                value=f"{active_territories}",
                delta=f"{df_filtered['MANAGER'].nunique()} manager"
            )
        
        with col4:
            growth_rate = 0
            if len(df_filtered['YEAR_MONTH'].unique()) > 1:
                monthly_sales = df_filtered.groupby('YEAR_MONTH')[product_cols['pf']].sum()
                if len(monthly_sales) > 1:
                    growth_rate = ((monthly_sales.iloc[-1] / monthly_sales.iloc[0]) ** (12/len(monthly_sales)) - 1) * 100
            st.metric(
                label="📈 Yıllık Büyüme",
                value=f"%{growth_rate:.1f}",
                delta="vs önceki dönem"
            )
        
        st.markdown("---")
        
        # Advanced Metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            # Pazar Yoğunluğu
            territory_perf = df_filtered.groupby('TERRITORIES')[product_cols['pf']].sum()
            hhi = (territory_perf / territory_perf.sum() * 100).pow(2).sum()
            st.metric(
                label="🏛️ Pazar Yoğunluğu (HHI)",
                value=f"{hhi:,.0f}",
                delta="Yüksek" if hhi > 2500 else "Orta" if hhi > 1500 else "Düşük"
            )
        
        with col6:
            # Karlılık İndeksi
            avg_territory_sales = territory_perf.mean()
            stdev_sales = territory_perf.std()
            profitability_index = avg_territory_sales / stdev_sales if stdev_sales > 0 else 0
            st.metric(
                label="💰 Karlılık İndeksi",
                value=f"{profitability_index:.2f}",
                delta="Yüksek" if profitability_index > 1 else "Orta" if profitability_index > 0.5 else "Düşük"
            )
        
        with col7:
            # Bölge Çeşitliliği
            region_dist = df_filtered.groupby('REGION')[product_cols['pf']].sum()
            region_diversity = 1 - (region_dist.max() / region_dist.sum())
            st.metric(
                label="🌍 Bölge Çeşitliliği",
                value=f"%{region_diversity*100:.1f}",
                delta="Yüksek" if region_diversity > 0.7 else "Orta" if region_diversity > 0.4 else "Düşük"
            )
        
        with col8:
            # Mevsimsellik Skoru
            if 'MONTH' in df_filtered.columns:
                monthly_avg = df_filtered.groupby('MONTH')[product_cols['pf']].mean()
                seasonality_score = monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() > 0 else 0
                st.metric(
                    label="📅 Mevsimsellik Skoru",
                    value=f"{seasonality_score:.2f}",
                    delta="Yüksek" if seasonality_score > 0.5 else "Orta" if seasonality_score > 0.2 else "Düşük"
                )
        
        st.markdown("---")
        
        # Görselleştirmeler
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("📈 Aylık Satış Trendi")
            
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
                line=dict(color=PERFORMANCE_COLORS['negative'], width=3),
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
        
        for insight in insights[:5]:  # İlk 5 içgörüyü göster
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
        
        if city_perf is not None and len(city_perf) > 0:
            # Harita seçimi
            if map_mode == "🗺️ Bölge Haritası":
                st.subheader("Bölge Bazlı Satış Haritası")
                
                if gdf is not None:
                    region_map = AdvancedMapVisualizer.create_region_map(
                        city_perf, gdf, 
                        title=f"{selected_product} - Bölge Bazlı Dağılım"
                    )
                    if region_map:
                        st.plotly_chart(region_map, use_container_width=True)
                    else:
                        st.warning("Harita oluşturulamadı")
                else:
                    st.warning("Harita verisi yüklenemedi")
            
            elif map_mode == "🏙️ Şehir Haritası":
                st.subheader("Şehir Bazlı Satış Haritası")
                
                city_map = AdvancedMapVisualizer.create_city_map(
                    city_perf,
                    title=f"{selected_product} - Şehir Bazlı Dağılım"
                )
                if city_map:
                    st.plotly_chart(city_map, use_container_width=True)
                else:
                    st.warning("Şehir haritası oluşturulamadı")
            
            else:
                st.subheader("3D Satış Dağılım Haritası")
                
                # 3D harita için ek veri hazırla
                city_perf_3d = city_perf.copy()
                city_perf_3d['Satis_Kategori'] = pd.qcut(
                    city_perf_3d['PF_Satis'], 
                    q=4, 
                    labels=['Çok Düşük', 'Düşük', 'Orta', 'Yüksek']
                )
                
                # 3D scatter plot
                fig_3d = px.scatter_3d(
                    city_perf_3d,
                    x='PF_Satis',
                    y='Rakip_Satis',
                    z='Pazar_Payi_%',
                    color='Region',
                    size='PF_Satis',
                    hover_name='City',
                    color_discrete_map=REGION_COLORS,
                    title='<b>3D Satış Dağılımı</b>',
                    size_max=30
                )
                
                fig_3d.update_layout(
                    height=700,
                    scene=dict(
                        xaxis_title='<b>PF Satış</b>',
                        yaxis_title='<b>Rakip Satış</b>',
                        zaxis_title='<b>Pazar Payı %</b>',
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
            
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
                show_n = st.slider("Gösterilecek Şehir Sayısı", 10, 50, 20)
            
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
            
            st.dataframe(styled_df, use_container_width=True, height=500)
            
        else:
            st.warning("Harita analizi için yeterli veri bulunamadı")
    
    # =========================================================================
    # TAB 3: TERRITORY ANALİZİ
    # =========================================================================
    with tab3:
        st.header("🏢 Territory Bazlı Detaylı Analiz")
        
        # Territory performansını hesapla
        territory_perf = calculate_territory_performance(df_filtered, product_cols)
        
        if len(territory_perf) > 0:
            # Filtreleme
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            
            with col_filter1:
                sort_options = {
                    'PF_Satis': 'PF Satış',
                    'Pazar_Payi_%': 'Pazar Payı',
                    'Toplam_Pazar': 'Toplam Pazar',
                    'Agirlik_%': 'Ağırlık %',
                    'Goreceli_Pazar_Payi': 'Göreceli Pay'
                }
                sort_by = st.selectbox(
                    "Sıralama Kriteri",
                    options=list(sort_options.keys()),
                    format_func=lambda x: sort_options[x]
                )
            
            with col_filter2:
                show_n = st.slider("Gösterilecek Territory", 10, 100, 25)
            
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
                
                # Ortanca çizgileri ekle
                median_share = territory_sorted['Pazar_Payi_%'].median()
                median_sales = territory_sorted['PF_Satis'].median()
                
                fig_scatter.add_hline(
                    y=median_share,
                    line_dash="dash",
                    line_color="white",
                    opacity=0.5,
                    annotation_text=f"Ort. Pay: {median_share:.1f}%"
                )
                
                fig_scatter.add_vline(
                    x=median_sales,
                    line_dash="dash", 
                    line_color="white",
                    opacity=0.5,
                    annotation_text=f"Ort. Satış: {median_sales:,.0f}"
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
            
            styled_territory = territory_display.style.format({
                'PF_Satis': '{:,.0f}',
                'Rakip_Satis': '{:,.0f}',
                'Toplam_Pazar': '{:,.0f}',
                'Pazar_Payi_%': '{:.1f}%',
                'Agirlik_%': '{:.1f}%',
                'Goreceli_Pazar_Payi': '{:.2f}'
            }).applymap(highlight_performance, subset=['Pazar_Payi_%'])\
              .applymap(highlight_weight, subset=['Agirlik_%'])
            
            st.dataframe(styled_territory, use_container_width=True, height=600)
            
            # Özet İstatistikler
            st.markdown("---")
            st.subheader("📊 Territory Performans Özeti")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                avg_share = territory_display['Pazar_Payi_%'].mean()
                st.metric("📊 Ort. Pazar Payı", f"{avg_share:.1f}%")
            
            with col_stat2:
                total_pf = territory_display['PF_Satis'].sum()
                st.metric("💰 Toplam PF Satış", f"{total_pf:,.0f}")
            
            with col_stat3:
                concentration = territory_display['Agirlik_%'].nlargest(5).sum()
                st.metric("🏆 Top 5 Konsantrasyonu", f"{concentration:.1f}%")
            
            with col_stat4:
                dominant_region = territory_display['Region'].mode()[0] if len(territory_display) > 0 else "Yok"
                st.metric("🌍 En Dominant Bölge", dominant_region)
        
        else:
            st.warning("Territory verisi bulunamadı")
    
    # =========================================================================
    # TAB 4: ZAMAN SERİSİ & AI
    # =========================================================================
    with tab4:
        st.header("📈 İleri Seviye Zaman Serisi Analizi & AI Tahminleme")
        
        # Zaman serisi verisini hazırla
        time_series_df = calculate_time_series(df_filtered, product_cols)
        
        if len(time_series_df) > 0:
            # Top row metrics
            col_ts1, col_ts2, col_ts3, col_ts4 = st.columns(4)
            
            with col_ts1:
                avg_sales = time_series_df['PF_Satis'].mean()
                st.metric("📊 Ort. Aylık Satış", f"{avg_sales:,.0f}")
            
            with col_ts2:
                total_growth = ((time_series_df['PF_Satis'].iloc[-1] / time_series_df['PF_Satis'].iloc[0]) - 1) * 100 if len(time_series_df) > 1 else 0
                st.metric("📈 Toplam Büyüme", f"%{total_growth:.1f}")
            
            with col_ts3:
                volatility = time_series_df['PF_Satis'].pct_change().std() * np.sqrt(12) * 100
                st.metric("📉 Yıllık Volatilite", f"%{volatility:.1f}")
            
            with col_ts4:
                sharpe_ratio = (time_series_df['PF_Satis'].pct_change().mean() / time_series_df['PF_Satis'].pct_change().std()) * np.sqrt(12) if time_series_df['PF_Satis'].pct_change().std() > 0 else 0
                st.metric("💰 Sharpe Oranı", f"{sharpe_ratio:.2f}")
            
            st.markdown("---")
            
            # Zaman Serisi Analizi
            st.subheader("📊 Zaman Serisi Ayrıştırma")
            
            # Decomposition
            decomposition = TimeSeriesAnalyzer.decompose_time_series(time_series_df, 'PF_Satis')
            
            if decomposition:
                fig_decomp = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Gözlemlenen', 'Trend', 'Mevsimsellik', 'Artık'),
                    vertical_spacing=0.08
                )
                
                fig_decomp.add_trace(
                    go.Scatter(x=time_series_df['DATE'], y=decomposition['observed'], 
                              name='Gözlemlenen', line=dict(color='#3B82F6')),
                    row=1, col=1
                )
                
                fig_decomp.add_trace(
                    go.Scatter(x=time_series_df['DATE'], y=decomposition['trend'], 
                              name='Trend', line=dict(color='#10B981')),
                    row=2, col=1
                )
                
                fig_decomp.add_trace(
                    go.Scatter(x=time_series_df['DATE'], y=decomposition['seasonal'], 
                              name='Mevsimsellik', line=dict(color='#F59E0B')),
                    row=3, col=1
                )
                
                fig_decomp.add_trace(
                    go.Scatter(x=time_series_df['DATE'], y=decomposition['residual'], 
                              name='Artık', line=dict(color='#EF4444')),
                    row=4, col=1
                )
                
                fig_decomp.update_layout(
                    height=800,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig_decomp, use_container_width=True)
            
            st.markdown("---")
            
            # AI/ML Tahminleme
            st.subheader("🤖 AI/ML ile Satış Tahminleme")
            
            if len(time_series_df) >= 24:
                with st.spinner("AI modelleri eğitiliyor..."):
                    forecaster = AdvancedMLForecaster()
                    ml_results, forecast_df, message = forecaster.forecast(
                        time_series_df, 
                        'PF_Satis', 
                        forecast_periods
                    )
                
                if message.startswith("✅"):
                    st.success(message)
                    
                    # Model Performansı
                    st.subheader("📊 Model Performans Karşılaştırması")
                    
                    if ml_results:
                        perf_data = []
                        for name, metrics in ml_results.items():
                            perf_data.append({
                                'Model': name,
                                'MAE': metrics['MAE'],
                                'RMSE': metrics['RMSE'],
                                'MAPE (%)': metrics['MAPE'],
                                'R²': metrics['R2']
                            })
                        
                        perf_df = pd.DataFrame(perf_data)
                        perf_df = perf_df.sort_values('MAPE (%)')
                        
                        # Performans tablosu
                        col_perf1, col_perf2 = st.columns([3, 1])
                        
                        with col_perf1:
                            st.dataframe(
                                perf_df.style.format({
                                    'MAE': '{:,.0f}',
                                    'RMSE': '{:,.0f}',
                                    'MAPE (%)': '{:.2f}',
                                    'R²': '{:.3f}'
                                }).background_gradient(subset=['MAPE (%)'], cmap='RdYlGn_r'),
                                use_container_width=True
                            )
                        
                        with col_perf2:
                            best_model = perf_df.iloc[0]
                            best_mape = best_model['MAPE (%)']
                            
                            if best_mape < 10:
                                confidence = "🟢 YÜKSEK"
                                color = "#10B981"
                            elif best_mape < 20:
                                confidence = "🟡 ORTA"
                                color = "#F59E0B"
                            else:
                                confidence = "🔴 DÜŞÜK"
                                color = "#EF4444"
                            
                            st.markdown(f'''
                            <div class="custom-card" style="text-align: center;">
                                <h3 style="color: white; margin-top: 0;">🏆 En İyi Model</h3>
                                <h2 style="color: {color}; margin: 0.5rem 0;">{best_model['Model']}</h2>
                                <p style="color: #94a3b8; margin: 0.5rem 0;">MAPE: <span style="color: {color}; font-weight: bold;">{best_mape:.2f}%</span></p>
                                <p style="color: #e2e8f0; margin: 0.5rem 0;">Güven Seviyesi:</p>
                                <h4 style="color: {color}; margin: 0.5rem 0;">{confidence}</h4>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        # Tahmin Grafiği
                        st.markdown("---")
                        st.subheader("🔮 Satış Tahminleri")
                        
                        forecast_chart = AdvancedVisualizations.create_forecast_comparison_chart(
                            time_series_df, 
                            forecast_df,
                            title="AI/ML Satış Tahminleri"
                        )
                        
                        st.plotly_chart(forecast_chart, use_container_width=True)
                        
                        # Tahmin Detayları
                        st.markdown("---")
                        st.subheader("📋 Tahmin Detayları")
                        
                        if forecast_df is not None and len(forecast_df) > 0:
                            forecast_display = forecast_df.copy()
                            forecast_display.index = range(1, len(forecast_display) + 1)
                            
                            # Ensemble tahminlerini göster
                            ensemble_forecast = forecast_display[forecast_display['Model'] == 'Ensemble']
                            if len(ensemble_forecast) > 0:
                                st.dataframe(
                                    ensemble_forecast[['YIL_AY', 'PF_Satis', 'Confidence_Low', 'Confidence_High']]\
                                        .style.format({
                                            'PF_Satis': '{:,.0f}',
                                            'Confidence_Low': '{:,.0f}',
                                            'Confidence_High': '{:,.0f}'
                                        }),
                                    use_container_width=True
                                )
                            
                            # Tahmin özeti
                            col_fore1, col_fore2, col_fore3 = st.columns(3)
                            
                            with col_fore1:
                                total_forecast = ensemble_forecast['PF_Satis'].sum()
                                st.metric("💰 Tahmini Toplam Satış", f"{total_forecast:,.0f}")
                            
                            with col_fore2:
                                avg_forecast = ensemble_forecast['PF_Satis'].mean()
                                historical_avg = time_series_df['PF_Satis'].mean()
                                growth_vs_avg = ((avg_forecast - historical_avg) / historical_avg) * 100
                                st.metric("📈 Ort. Aylık Tahmin", f"{avg_forecast:,.0f}", 
                                         f"%{growth_vs_avg:.1f} vs geçmiş")
                            
                            with col_fore3:
                                confidence_range = ((ensemble_forecast['Confidence_High'].mean() - 
                                                   ensemble_forecast['Confidence_Low'].mean()) / 
                                                   ensemble_forecast['PF_Satis'].mean()) * 100
                                st.metric("🎯 Güven Aralığı", f"±%{confidence_range:.1f}")
                else:
                    st.error(message)
            else:
                st.warning(f"AI tahminleme için en az 24 ay veri gereklidir. Mevcut: {len(time_series_df)} ay")
        
        else:
            st.warning("Zaman serisi verisi bulunamadı")
    
    # =========================================================================
    # TAB 5: RAKİP & PAZAR ANALİZİ
    # =========================================================================
    with tab5:
        st.header("🎯 Rakip ve Pazar Analizi")
        
        if len(df_filtered) > 0:
            # Pazar İstihbaratı
            st.subheader("📊 Pazar İstihbaratı")
            
            market_intel = AdvancedAnalytics.calculate_market_intelligence(df_filtered, product_cols)
            
            col_intel1, col_intel2, col_intel3, col_intel4 = st.columns(4)
            
            with col_intel1:
                total_market = market_intel['market_size']['total']
                st.metric("🏪 Toplam Pazar Büyüklüğü", f"{total_market:,.0f}")
            
            with col_intel2:
                hhi = market_intel['concentration']['hhi']
                concentration = "Yüksek" if hhi > 2500 else "Orta" if hhi > 1500 else "Düşük"
                st.metric("🏛️ Pazar Yoğunluğu (HHI)", f"{hhi:,.0f}", concentration)
            
            with col_intel3:
                cr4 = market_intel['concentration']['cr4']
                st.metric("👑 CR4 Konsantrasyonu", f"%{cr4:.1f}")
            
            with col_intel4:
                if 'growth' in market_intel:
                    relative_growth = market_intel['growth']['relative_growth_%']
                    st.metric("⚡ Göreceli Büyüme", f"%{relative_growth:.1f}")
            
            st.markdown("---")
            
            # Rakip Karşılaştırması
            st.subheader("📈 PF vs Rakip Performans Karşılaştırması")
            
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                # Aylık karşılaştırma
                monthly_comparison = df_filtered.groupby('YEAR_MONTH').agg({
                    product_cols['pf']: 'sum',
                    product_cols['rakip']: 'sum'
                }).reset_index()
                
                fig_comp = go.Figure()
                
                fig_comp.add_trace(go.Bar(
                    x=monthly_comparison['YEAR_MONTH'],
                    y=monthly_comparison[product_cols['pf']],
                    name='PF Satış',
                    marker_color=PERFORMANCE_COLORS['positive']
                ))
                
                fig_comp.add_trace(go.Bar(
                    x=monthly_comparison['YEAR_MONTH'],
                    y=monthly_comparison[product_cols['rakip']],
                    name='Rakip Satış',
                    marker_color=PERFORMANCE_COLORS['negative']
                ))
                
                fig_comp.update_layout(
                    barmode='group',
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    xaxis_title='<b>Ay</b>',
                    yaxis_title='<b>Satış</b>',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)
            
            with col_comp2:
                # Pazar payı trendi
                monthly_comparison['PF_Pay_%'] = (monthly_comparison[product_cols['pf']] / 
                                                 (monthly_comparison[product_cols['pf']] + 
                                                  monthly_comparison[product_cols['rakip']])) * 100
                
                fig_share = go.Figure()
                
                fig_share.add_trace(go.Scatter(
                    x=monthly_comparison['YEAR_MONTH'],
                    y=monthly_comparison['PF_Pay_%'],
                    mode='lines+markers',
                    name='PF Pazar Payı',
                    line=dict(color=PERFORMANCE_COLORS['positive'], width=3),
                    fill='tozeroy',
                    fillcolor='rgba(16, 185, 129, 0.1)'
                ))
                
                fig_share.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.5,
                                   annotation_text="50% Eşik")
                
                fig_share.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    xaxis_title='<b>Ay</b>',
                    yaxis_title='<b>Pazar Payı (%)</b>',
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig_share, use_container_width=True)
            
            # Rakip Detayları
            st.markdown("---")
            st.subheader("📋 Rakip Performans Detayları")
            
            # Bölge bazında rakip analizi
            region_competition = df_filtered.groupby('REGION').agg({
                product_cols['pf']: 'sum',
                product_cols['rakip']: 'sum'
            }).reset_index()
            
            region_competition['PF_Pay_%'] = (region_competition[product_cols['pf']] / 
                                             (region_competition[product_cols['pf']] + 
                                              region_competition[product_cols['rakip']])) * 100
            
            region_competition = region_competition.sort_values('PF_Pay_%', ascending=False)
            region_competition.index = range(1, len(region_competition) + 1)
            
            # Styling
            def color_competition(val):
                if val >= 60:
                    return 'background-color: rgba(16, 185, 129, 0.3); color: #10B981; font-weight: bold'
                elif val >= 40:
                    return 'background-color: rgba(245, 158, 11, 0.3); color: #F59E0B; font-weight: bold'
                else:
                    return 'background-color: rgba(239, 68, 68, 0.3); color: #EF4444; font-weight: bold'
            
            styled_competition = region_competition.style.format({
                product_cols['pf']: '{:,.0f}',
                product_cols['rakip']: '{:,.0f}',
                'PF_Pay_%': '{:.1f}%'
            }).applymap(color_competition, subset=['PF_Pay_%'])
            
            st.dataframe(styled_competition, use_container_width=True, height=400)
            
            # Rekabet Özeti
            st.markdown("---")
            st.subheader("⚔️ Rekabet Özeti")
            
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            
            with col_sum1:
                strongest_region = region_competition.loc[region_competition['PF_Pay_%'].idxmax()]
                st.metric("🏆 En Güçlü Olduğumuz Bölge", 
                         strongest_region['REGION'], 
                         f"%{strongest_region['PF_Pay_%']:.1f}")
            
            with col_sum2:
                weakest_region = region_competition.loc[region_competition['PF_Pay_%'].idxmin()]
                st.metric("📉 En Zayıf Olduğumuz Bölge", 
                         weakest_region['REGION'], 
                         f"%{weakest_region['PF_Pay_%']:.1f}")
            
            with col_sum3:
                avg_competition = region_competition[product_cols['rakip']].mean()
                st.metric("💼 Ort. Rakip Varlığı", 
                         f"{avg_competition:,.0f}")
        
        else:
            st.warning("Rakip analizi için yeterli veri bulunamadı")
    
    # =========================================================================
    # TAB 6: BCG & STRATEJİ
    # =========================================================================
    with tab6:
        st.header("⭐ BCG Matrix & Stratejik Portföy Yönetimi")
        
        # BCG Matrix hesapla
        bcg_df = AdvancedAnalytics.calculate_bcg_matrix(df_filtered, product_cols, date_filter)
        
        if len(bcg_df) > 0:
            # BCG Dağılımı
            st.subheader("📊 Portföy Dağılımı")
            
            bcg_counts = bcg_df['BCG_Kategori'].value_counts()
            strategy_counts = bcg_df['Yatirim_Stratejisi'].value_counts()
            
            col_bcg1, col_bcg2, col_bcg3, col_bcg4 = st.columns(4)
            
            with col_bcg1:
                star_count = bcg_counts.get("⭐ STAR", 0)
                star_value = bcg_df[bcg_df['BCG_Kategori'] == "⭐ STAR"]['PF_Satis'].sum()
                st.metric("⭐ STAR", f"{star_count}", f"{star_value:,.0f} PF")
            
            with col_bcg2:
                cow_count = bcg_counts.get("🐄 CASH COW", 0)
                cow_value = bcg_df[bcg_df['BCG_Kategori'] == "🐄 CASH COW"]['PF_Satis'].sum()
                st.metric("🐄 CASH COW", f"{cow_count}", f"{cow_value:,.0f} PF")
            
            with col_bcg3:
                question_count = bcg_counts.get("❓ QUESTION MARK", 0)
                question_value = bcg_df[bcg_df['BCG_Kategori'] == "❓ QUESTION MARK"]['PF_Satis'].sum()
                st.metric("❓ QUESTION MARK", f"{question_count}", f"{question_value:,.0f} PF")
            
            with col_bcg4:
                dog_count = bcg_counts.get("🐶 DOG", 0)
                dog_value = bcg_df[bcg_df['BCG_Kategori'] == "🐶 DOG"]['PF_Satis'].sum()
                st.metric("🐶 DOG", f"{dog_count}", f"{dog_value:,.0f} PF")
            
            st.markdown("---")
            
            # BCG Matrix Görselleştirme
            st.subheader("🎯 BCG Matrix")
            
            # BCG için veriyi hazırla
            bcg_df['size'] = np.sqrt(bcg_df['PF_Satis']) / 50  # Marker boyutu
            bcg_df['color'] = bcg_df['BCG_Kategori'].map(BCG_COLORS)
            
            fig_bcg = go.Figure()
            
            # Her kategori için ayrı trace
            for category in bcg_df['BCG_Kategori'].unique():
                category_data = bcg_df[bcg_df['BCG_Kategori'] == category]
                
                fig_bcg.add_trace(go.Scatter(
                    x=category_data['Goreceli_Pazar_Payi'],
                    y=category_data['Buyume_%'],
                    mode='markers+text',
                    name=category,
                    marker=dict(
                        size=category_data['size'],
                        color=BCG_COLORS[category],
                        opacity=0.7,
                        line=dict(width=2, color='white')
                    ),
                    text=category_data['Territory'],
                    textposition='top center',
                    textfont=dict(size=9, color='white'),
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
            
            # Ortanca çizgileri
            median_share = bcg_df['Goreceli_Pazar_Payi'].median()
            median_growth = bcg_df['Buyume_%'].median()
            
            fig_bcg.add_hline(
                y=median_growth,
                line_dash="dash",
                line_color="white",
                opacity=0.5,
                annotation_text=f"Ort. Büyüme: {median_growth:.1f}%"
            )
            
            fig_bcg.add_vline(
                x=median_share,
                line_dash="dash",
                line_color="white",
                opacity=0.5,
                annotation_text=f"Ort. Pay: {median_share:.2f}"
            )
            
            # Quadrant etiketleri
            fig_bcg.add_annotation(
                x=median_share * 1.5, y=median_growth * 1.5,
                text="⭐ STAR",
                showarrow=False,
                font=dict(size=14, color=BCG_COLORS["⭐ STAR"], family='Arial Black')
            )
            
            fig_bcg.add_annotation(
                x=median_share * 1.5, y=median_growth * 0.5,
                text="🐄 CASH COW",
                showarrow=False,
                font=dict(size=14, color=BCG_COLORS["🐄 CASH COW"], family='Arial Black')
            )
            
            fig_bcg.add_annotation(
                x=median_share * 0.5, y=median_growth * 1.5,
                text="❓ QUESTION MARK",
                showarrow=False,
                font=dict(size=14, color=BCG_COLORS["❓ QUESTION MARK"], family='Arial Black')
            )
            
            fig_bcg.add_annotation(
                x=median_share * 0.5, y=median_growth * 0.5,
                text="🐶 DOG",
                showarrow=False,
                font=dict(size=14, color=BCG_COLORS["🐶 DOG"], family='Arial Black')
            )
            
            fig_bcg.update_layout(
                title='<b>BCG Matrix - Stratejik Konumlandırma</b>',
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis_title='<b>Göreceli Pazar Payı</b>',
                yaxis_title='<b>Pazar Büyüme Oranı (%)</b>',
                legend=dict(
                    title='<b>BCG Kategorisi</b>',
                    bgcolor='rgba(30, 41, 59, 0.8)',
                    bordercolor='rgba(59, 130, 246, 0.3)',
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig_bcg, use_container_width=True)
            
            # Yatırım Stratejileri
            st.markdown("---")
            st.subheader("💼 Yatırım Stratejileri")
            
            col_strat1, col_strat2, col_strat3, col_strat4, col_strat5 = st.columns(5)
            
            strategy_colors = {
                "🚀 AGRESİF BÜYÜME": "#EF4444",
                "🛡️ KORUMA": "#10B981",
                "💎 POTANSİYEL": "#8B5CF6",
                "👁️ İZLEME": "#64748B",
                "🔄 YENİDEN YAPILANMA": "#EC4899"
            }
            
            strategies = list(strategy_colors.keys())
            cols = [col_strat1, col_strat2, col_strat3, col_strat4, col_strat5]
            
            for idx, (strategy, color) in enumerate(strategy_colors.items()):
                with cols[idx]:
                    count = strategy_counts.get(strategy, 0)
                    value = bcg_df[bcg_df['Yatirim_Stratejisi'] == strategy]['PF_Satis'].sum() if count > 0 else 0
                    
                    st.markdown(f'''
                    <div style="background: rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)};
                                border: 2px solid {color};
                                border-radius: 12px;
                                padding: 1rem;
                                text-align: center;
                                margin: 0.5rem 0;">
                        <h4 style="color: {color}; margin: 0 0 0.5rem 0;">{strategy}</h4>
                        <h3 style="color: white; margin: 0 0 0.5rem 0;">{count}</h3>
                        <p style="color: #94a3b8; margin: 0;">{value:,.0f} PF</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            # Stratejik Öneriler
            st.markdown("---")
            st.subheader("📋 Stratejik Öneriler")
            
            # Her strateji için öneriler
            strategy_recommendations = {
                "🚀 AGRESİF BÜYÜME": [
                    "Yatırımı artırın",
                    "Pazarlama bütçesini genişletin",
                    "Yeni müşteri segmentlerine ulaşın",
                    "Rakiplere karşı agresif fiyatlandırma uygulayın"
                ],
                "🛡️ KORUMA": [
                    "Mevcut pazar payını koruyun",
                    "Müşteri sadakatini güçlendirin",
                    "Verimlilik artırıcı önlemler alın",
                    "Nakit akışını optimize edin"
                ],
                "💎 POTANSİYEL": [
                    "Araştırma ve geliştirmeye yatırım yapın",
                    "Pilot projeler başlatın",
                    "Stratejik ortaklıklar kurun",
                    "Hedefli promosyonlar uygulayın"
                ],
                "👁️ İZLEME": [
                    "Performansı düzenli izleyin",
                    "Maliyetleri minimize edin",
                    "Kaynakları yeniden dağıtın",
                    "Çıkış stratejisi geliştirin"
                ]
            }
            
            # Her BCG kategorisi için territory listesi
            for category in bcg_df['BCG_Kategori'].unique():
                category_data = bcg_df[bcg_df['BCG_Kategori'] == category]
                strategy = category_data['Yatirim_Stratejisi'].iloc[0] if len(category_data) > 0 else "👁️ İZLEME"
                
                with st.expander(f"{category} - {strategy} ({len(category_data)} Territory)"):
                    # Territory listesi
                    territories_list = category_data[['Territory', 'Region', 'PF_Satis', 'Pazar_Payi_%', 'Buyume_%']]\
                        .sort_values('PF_Satis', ascending=False)
                    territories_list.index = range(1, len(territories_list) + 1)
                    
                    st.dataframe(
                        territories_list.style.format({
                            'PF_Satis': '{:,.0f}',
                            'Pazar_Payi_%': '{:.1f}%',
                            'Buyume_%': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
                    
                    # Öneriler
                    st.markdown("#### 🎯 Stratejik Öneriler:")
                    for recommendation in strategy_recommendations.get(strategy, []):
                        st.markdown(f"- {recommendation}")
        
        else:
            st.warning("BCG Matrix analizi için yeterli veri bulunamadı")
    
    # =========================================================================
    # TAB 7: TAHMİN & SENARYO
    # =========================================================================
    with tab7:
        st.header("🔮 Tahmin & Senaryo Analizleri")
        
        # Senaryo analizi
        st.subheader("📈 Senaryo Analizleri")
        
        col_scenario1, col_scenario2, col_scenario3 = st.columns(3)
        
        with col_scenario1:
            optimistic_growth = st.slider(
                "🏆 İyimser Senaryo Büyüme (%)",
                -50, 200, 30,
                help="Pozitif büyüme için iyimser senaryo"
            )
        
        with col_scenario2:
            base_growth = st.slider(
                "📊 Temel Senaryo Büyüme (%)", 
                -50, 100, 10,
                help="Normal büyüme için temel senaryo"
            )
        
        with col_scenario3:
            pessimistic_growth = st.slider(
                "⚠️ Kötümser Senaryo Büyüme (%)",
                -100, 50, -10,
                help="Negatif büyüme için kötümser senaryo"
            )
        
        # Mevcut satışları al
        current_sales = df_filtered[product_cols['pf']].sum()
        
        # Senaryo hesaplamaları
        scenario_results = {
            'İyimser': current_sales * (1 + optimistic_growth/100),
            'Temel': current_sales * (1 + base_growth/100),
            'Kötümser': current_sales * (1 + pessimistic_growth/100)
        }
        
        # Senaryo grafiği
        fig_scenario = go.Figure()
        
        colors = ['#10B981', '#3B82F6', '#EF4444']
        
        for idx, (scenario, value) in enumerate(scenario_results.items()):
            fig_scenario.add_trace(go.Bar(
                x=[scenario],
                y=[value],
                name=scenario,
                marker_color=colors[idx],
                text=f"{value:,.0f}",
                textposition='outside'
            ))
        
        fig_scenario.add_hline(
            y=current_sales,
            line_dash="dash",
            line_color="white",
            opacity=0.5,
            annotation_text=f"Mevcut: {current_sales:,.0f}"
        )
        
        fig_scenario.update_layout(
            title='<b>Senaryo Analizi - Tahmini Satışlar</b>',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis_title='<b>Senaryo</b>',
            yaxis_title='<b>Tahmini Satış</b>',
            showlegend=False
        )
        
        st.plotly_chart(fig_scenario, use_container_width=True)
        
        # Risk Analizi
        st.markdown("---")
        st.subheader("📉 Risk Analizi")
        
        # Risk faktörleri
        risk_factors = {
            "Pazar Volatilitesi": st.slider("Pazar Volatilitesi Riski", 1, 10, 5),
            "Rekabet Yoğunluğu": st.slider("Rekabet Yoğunluğu Riski", 1, 10, 6),
            "Ekonomik Koşullar": st.slider("Ekonomik Koşullar Riski", 1, 10, 4),
            "Regülasyon Riski": st.slider("Regülasyon Riski", 1, 10, 3),
            "Tedarik Zinciri Riski": st.slider("Tedarik Zinciri Riski", 1, 10, 5)
        }
        
        # Risk radar chart
        fig_risk = go.Figure()
        
        fig_risk.add_trace(go.Scatterpolar(
            r=list(risk_factors.values()),
            theta=list(risk_factors.keys()),
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.3)',
            line_color='#EF4444',
            name='Risk Seviyesi'
        ))
        
        fig_risk.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=False,
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title='<b>Risk Analizi Radar Grafiği</b>'
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Risk Özeti
        total_risk = sum(risk_factors.values())
        avg_risk = total_risk / len(risk_factors)
        
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        with col_risk1:
            if avg_risk >= 7:
                risk_level = "🔴 YÜKSEK RİSK"
                risk_color = "#EF4444"
            elif avg_risk >= 4:
                risk_level = "🟡 ORTA RİSK"
                risk_color = "#F59E0B"
            else:
                risk_level = "🟢 DÜŞÜK RİSK"
                risk_color = "#10B981"
            
            st.metric("🎯 Toplam Risk Skoru", f"{total_risk}/50", risk_level)
        
        with col_risk2:
            highest_risk = max(risk_factors, key=risk_factors.get)
            st.metric("⚠️ En Yüksek Risk", highest_risk, f"{risk_factors[highest_risk]}/10")
        
        with col_risk3:
            lowest_risk = min(risk_factors, key=risk_factors.get)
            st.metric("✅ En Düşük Risk", lowest_risk, f"{risk_factors[lowest_risk]}/10")
        
        # Erken Uyarı Sistemi
        st.markdown("---")
        st.subheader("🚨 Erken Uyarı Göstergeleri")
        
        # Uyarı göstergeleri
        warning_indicators = []
        
        # 1. Büyüme yavaşlaması
        if len(time_series_df) > 3:
            recent_growth = time_series_df['PF_Satis'].pct_change().iloc[-3:].mean() * 100
            if recent_growth < -5:
                warning_indicators.append({
                    'type': 'danger',
                    'title': '📉 Büyüme Yavaşlaması',
                    'description': f'Son 3 ayda ortalama {recent_growth:.1f}% büyüme kaydedildi.',
                    'action': 'Pazarlama stratejisini gözden geçirin.'
                })
        
        # 2. Pazar payı kaybı
        if len(time_series_df) > 3:
            recent_pf = time_series_df['PF_Satis'].iloc[-3:].mean()
            recent_comp = time_series_df['Rakip_Satis'].iloc[-3:].mean()
            recent_share = recent_pf / (recent_pf + recent_comp) * 100 if (recent_pf + recent_comp) > 0 else 0
            
            historical_pf = time_series_df['PF_Satis'].iloc[:-3].mean()
            historical_comp = time_series_df['Rakip_Satis'].iloc[:-3].mean()
            historical_share = historical_pf / (historical_pf + historical_comp) * 100 if (historical_pf + historical_comp) > 0 else 0
            
            if recent_share < historical_share - 5:
                warning_indicators.append({
                    'type': 'warning',
                    'title': '⚠️ Pazar Payı Kaybı',
                    'description': f'Pazar payı {historical_share:.1f}%\'dan {recent_share:.1f}%\'a düştü.',
                    'action': 'Rekabet analizini güncelleyin.'
                })
        
        # 3. Volatilite artışı
        if len(time_series_df) > 6:
            recent_vol = time_series_df['PF_Satis'].pct_change().iloc[-6:].std() * 100
            historical_vol = time_series_df['PF_Satis'].pct_change().iloc[:-6].std() * 100
            
            if recent_vol > historical_vol * 1.5:
                warning_indicators.append({
                    'type': 'info',
                    'title': '📊 Volatilite Artışı',
                    'description': f'Volatilite {historical_vol:.1f}%\'dan {recent_vol:.1f}%\'a yükseldi.',
                    'action': 'Risk yönetim stratejisini gözden geçirin.'
                })
        
        # Uyarıları göster
        for warning in warning_indicators:
            if warning['type'] == 'danger':
                st.markdown(f'''
                <div class="alert-danger">
                    <strong>{warning['title']}</strong><br>
                    {warning['description']}<br>
                    <em>Önerilen Aksiyon: {warning['action']}</em>
                </div>
                ''', unsafe_allow_html=True)
            elif warning['type'] == 'warning':
                st.markdown(f'''
                <div class="alert-warning">
                    <strong>{warning['title']}</strong><br>
                    {warning['description']}<br>
                    <em>Önerilen Aksiyon: {warning['action']}</em>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="alert-info">
                    <strong>{warning['title']}</strong><br>
                    {warning['description']}<br>
                    <em>Önerilen Aksiyon: {warning['action']}</em>
                </div>
                ''', unsafe_allow_html=True)
        
        if not warning_indicators:
            st.success("✅ Tüm erken uyarı göstergeleri normal seviyede.")
    
    # =========================================================================
    # TAB 8: RAPORLAR & EXPORT
    # =========================================================================
    with tab8:
        st.header("📥 Raporlar & Export")
        
        # Rapor seçenekleri
        st.subheader("📊 Rapor Türü Seçin")
        
        report_type = st.selectbox(
            "Rapor Türü",
            [
                "📈 Genel Performans Raporu",
                "🗺️ Harita Analiz Raporu", 
                "🏢 Territory Detay Raporu",
                "📉 Zaman Serisi Raporu",
                "🎯 Rakip Analiz Raporu",
                "⭐ BCG Matrix Raporu",
                "🔮 Tahmin & Senaryo Raporu",
                "📋 Tüm Raporlar (Kapsamlı)"
            ]
        )
        
        st.markdown("---")
        
        # Rapor formatı
        st.subheader("📁 Çıktı Formatı")
        
        output_format = st.radio(
            "Çıktı Formatı",
            ["📊 Excel (.xlsx)", "📄 PDF Rapor", "📈 PowerPoint (.pptx)", "📋 HTML Rapor"],
            horizontal=True
        )
        
        st.markdown("---")
        
        # Rapor oluşturma
        st.subheader("🚀 Rapor Oluştur")
        
        if st.button("📊 Rapor Oluştur ve İndir", type="primary", use_container_width=True):
            with st.spinner("Rapor hazırlanıyor..."):
                try:
                    # Tüm analizleri hesapla
                    time_series_df = calculate_time_series(df_filtered, product_cols)
                    bcg_df = AdvancedAnalytics.calculate_bcg_matrix(df_filtered, product_cols, date_filter)
                    territory_perf = calculate_territory_performance(df_filtered, product_cols)
                    city_perf = calculate_city_performance(df_filtered, product_cols)
                    
                    # Excel raporu oluştur
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # 1. Özet Sayfası
                        summary_data = pd.DataFrame({
                            'Metrik': [
                                'Ürün', 'Dönem', 'Toplam PF Satış', 'Toplam Pazar',
                                'Pazar Payı (%)', 'Aktif Territory', 'Aktif Şehir',
                                'Ort. Aylık Büyüme (%)', 'Pazar Yoğunluğu (HHI)',
                                'BCG Dağılımı (Star/Cow/Question/Dog)'
                            ],
                            'Değer': [
                                selected_product,
                                date_option if date_filter else 'Tüm Veriler',
                                f"{territory_perf['PF_Satis'].sum():,.0f}",
                                f"{territory_perf['Toplam_Pazar'].sum():,.0f}",
                                f"{(territory_perf['PF_Satis'].sum() / territory_perf['Toplam_Pazar'].sum() * 100):.1f}%" if territory_perf['Toplam_Pazar'].sum() > 0 else "0%",
                                f"{territory_perf['Territory'].nunique()}",
                                f"{city_perf['City'].nunique()}",
                                f"{time_series_df['PF_Buyume_%'].mean():.1f}%" if len(time_series_df) > 1 else "0%",
                                f"{(territory_perf['Pazar_Payi_%'] / 100).pow(2).sum() * 10000:,.0f}",
                                f"{len(bcg_df[bcg_df['BCG_Kategori'] == '⭐ STAR'])}/{len(bcg_df[bcg_df['BCG_Kategori'] == '🐄 CASH COW'])}/{len(bcg_df[bcg_df['BCG_Kategori'] == '❓ QUESTION MARK'])}/{len(bcg_df[bcg_df['BCG_Kategori'] == '🐶 DOG'])}"
                            ]
                        })
                        summary_data.to_excel(writer, sheet_name='Özet', index=False)
                        
                        # 2. Territory Performans
                        territory_perf.to_excel(writer, sheet_name='Territory_Performans', index=False)
                        
                        # 3. Şehir Performans
                        city_perf.to_excel(writer, sheet_name='Şehir_Performans', index=False)
                        
                        # 4. Zaman Serisi
                        time_series_df.to_excel(writer, sheet_name='Zaman_Serisi', index=False)
                        
                        # 5. BCG Matrix
                        bcg_df.to_excel(writer, sheet_name='BCG_Matrix', index=False)
                        
                        # 6. Rakip Analizi
                        competitor_analysis = df_filtered.groupby('YEAR_MONTH').agg({
                            product_cols['pf']: 'sum',
                            product_cols['rakip']: 'sum'
                        }).reset_index()
                        competitor_analysis['PF_Pay_%'] = (competitor_analysis[product_cols['pf']] / 
                                                         (competitor_analysis[product_cols['pf']] + 
                                                          competitor_analysis[product_cols['rakip']])) * 100
                        competitor_analysis.to_excel(writer, sheet_name='Rakip_Analizi', index=False)
                        
                        # 7. Bölge Analizi
                        region_analysis = df_filtered.groupby('REGION').agg({
                            product_cols['pf']: 'sum',
                            product_cols['rakip']: 'sum'
                        }).reset_index()
                        region_analysis['PF_Pay_%'] = (region_analysis[product_cols['pf']] / 
                                                      (region_analysis[product_cols['pf']] + 
                                                       region_analysis[product_cols['rakip']])) * 100
                        region_analysis.to_excel(writer, sheet_name='Bölge_Analizi', index=False)
                        
                        # 8. Manager Performansı
                        if 'MANAGER' in df_filtered.columns:
                            manager_perf = df_filtered.groupby('MANAGER').agg({
                                product_cols['pf']: 'sum',
                                product_cols['rakip']: 'sum',
                                'TERRITORIES': 'nunique'
                            }).reset_index()
                            manager_perf.columns = ['Manager', 'PF_Satis', 'Rakip_Satis', 'Territory_Count']
                            manager_perf.to_excel(writer, sheet_name='Manager_Performans', index=False)
                    
                    st.success("✅ Rapor hazır!")
                    
                    # İndirme butonu
                    filename = f"ticari_analiz_raporu_{selected_product}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    
                    st.download_button(
                        label="💾 Excel Raporunu İndir",
                        data=output.getvalue(),
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                    # Rapor önizleme
                    with st.expander("📋 Rapor Önizleme"):
                        st.dataframe(summary_data, use_container_width=True)
                        
                        col_preview1, col_preview2 = st.columns(2)
                        
                        with col_preview1:
                            st.metric("Toplam Sayfa", "8")
                            st.metric("Toplam Satır", f"{len(df_filtered):,}")
                        
                        with col_preview2:
                            st.metric("Analiz Edilen Ürün", selected_product)
                            st.metric("Rapor Tarihi", datetime.now().strftime("%d.%m.%Y"))
                
                except Exception as e:
                    st.error(f"❌ Rapor oluşturma hatası: {str(e)}")
        
        # Ek bilgiler
        st.markdown("---")
        st.subheader("ℹ️ Rapor Hakkında")
        
        st.markdown("""
        <div class="alert-info">
            <strong>📊 Rapor İçeriği:</strong><br>
            - 8 ayrı analiz sayfası<br>
            - 50+ performans metriği<br>
            - Görselleştirme verileri<br>
            - Stratejik öneriler<br>
            - Otomatik içgörüler<br><br>
            
            <strong>🎯 Kullanım Alanları:</strong><br>
            - Aylık performans review'ları<br>
            - Stratejik planlama toplantıları<br>
            - Yatırım kararları<br>
            - Pazar analizleri<br>
            - Rakip benchmark'ı
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# UYGULAMAYI BAŞLAT
# =============================================================================

if __name__ == "__main__":
    main()

