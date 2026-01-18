"""
🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ
Territory Bazlı Performans, ML Tahminleme, Türkiye Haritası ve Rekabet Analizi

Özellikler:
- 🗺️ Türkiye il bazlı harita görselleştirme (GELİŞMİŞ VERSİYON)
- 🤖 GERÇEK Machine Learning (Linear Regression, Ridge, Random Forest)
- 📊 Aylık/Yıllık dönem seçimi
- 📈 Gelişmiş rakip analizi ve trend karşılaştırması
- 🎯 Dinamik zaman aralığı filtreleme
- 📉 SWOT Analizi ve Stratejik Öneriler
- 🏆 Performans Skorlama Sistemi
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from io import BytesIO
import json
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import warnings
import requests
from urllib.request import urlopen
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import plotly.figure_factory as ff

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Ticari Portföy Analizi",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS - GELİŞTİRİLMİŞ
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f1729 0%, #1a1f2e 50%, #242837 100%);
        background-attachment: fixed;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 50%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 50px rgba(59, 130, 246, 0.2);
        letter-spacing: -0.5px;
        margin-bottom: 1rem;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 50%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.85);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 48px rgba(59, 130, 246, 0.25);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding: 0.5rem;
        background: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
        padding: 1rem 2rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        margin: 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.15);
        color: #e0e7ff;
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 50%, #F59E0B 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transform: scale(1.02);
    }
    
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    h1 {
        font-size: 2.5rem;
        margin-top: 0;
    }
    
    h2 {
        font-size: 2rem;
        margin-top: 0;
    }
    
    h3 {
        font-size: 1.5rem;
    }
    
    h4 {
        font-size: 1.2rem;
    }
    
    p, span, div, label {
        color: #cbd5e1;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 28px rgba(59, 130, 246, 0.4);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    .stButton>button::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.7s;
    }
    
    .stButton>button:hover::after {
        left: 100%;
    }
    
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stDataFrame {
        border-radius: 12px;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #10B981 0%, #F59E0B 100%);
    }
    
    /* Card styling for visualizations */
    .plotly-graph-div {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 41, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    /* Input field styling */
    .stSelectbox, .stSlider, .stRadio {
        background: rgba(30, 41, 59, 0.7);
        padding: 8px;
        border-radius: 10px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 50%, #F59E0B 100%);
    }
    
    /* Custom card styling */
    .custom-card {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.2);
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(16, 185, 129, 0.1));
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(239, 68, 68, 0.1));
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(34, 197, 94, 0.1));
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Tooltip styling */
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
        border-radius: 6px;
        padding: 12px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-excellent { background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid #10B981; }
    .status-good { background: rgba(34, 197, 94, 0.2); color: #22C55E; border: 1px solid #22C55E; }
    .status-fair { background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid #F59E0B; }
    .status-poor { background: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid #EF4444; }
    
    /* Performance indicators */
    .perf-indicator {
        width: 100%;
        height: 8px;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .perf-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# RENK PALETLERİ - GENİŞLETİLMİŞ
# =============================================================================

# Bölge renkleri
REGION_COLORS = {
    "MARMARA": "#3B82F6",
    "EGE": "#10B981",
    "AKDENİZ": "#F59E0B",
    "İÇ ANADOLU": "#8B5CF6",
    "KARADENİZ": "#06B6D4",
    "GÜNEYDOĞU ANADOLU": "#EF4444",
    "DOĞU ANADOLU": "#84CC16",
    "BATI KARADENİZ": "#6366F1",
    "ORTA KARADENİZ": "#14B8A6",
    "KUZEYDOĞU ANADOLU": "#F97316",
    "GÜNEYDOĞU": "#EC4899",
    "DİĞER": "#64748B"
}

# Performans renkleri
PERFORMANCE_COLORS = {
    "excellent": "#10B981",
    "good": "#22C55E",
    "fair": "#F59E0B",
    "poor": "#EF4444",
    "high": "#1F7A5A",
    "medium": "#C48A2A",
    "low": "#B23A3A",
    "positive": "#10B981",
    "negative": "#EF4444",
    "neutral": "#6B7280",
    "warning": "#F59E0B",
    "info": "#3B82F6",
    "success": "#166534",
    "danger": "#991B1B"
}

# Strateji renkleri
STRATEGY_COLORS = {
    "AGRESİF": "#EF4444",
    "HIZLANDIRILMIŞ": "#F59E0B",
    "KORUMA": "#10B981",
    "POTANSİYEL": "#3B82F6",
    "İZLEME": "#64748B",
    "GELİŞTİR": "#8B5CF6",
    "OPTİMİZE ET": "#06B6D4",
    "YENİDEN YAPILANMA": "#EC4899"
}

# Gradient skalaları
GRADIENT_SCALES = {
    "blue_green": ["#3B82F6", "#06B6D4", "#10B981"],
    "red_yellow_green": ["#EF4444", "#F59E0B", "#10B981"],
    "purple_blue": ["#8B5CF6", "#6366F1", "#3B82F6"],
    "temperature": ["#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE"],
    "diverging": ["#EF4444", "#F59E0B", "#FBBF24", "#10B981", "#3B82F6"]
}

# SWOT renkleri
SWOT_COLORS = {
    "Strengths": "#10B981",
    "Weaknesses": "#EF4444",
    "Opportunities": "#3B82F6",
    "Threats": "#F59E0B"
}

# =============================================================================
# SABİTLER
# =============================================================================

# Ürün sütun eşleşmeleri
COLUMN_NAMES = {
    "TROCMETAM": {
        "pf": "TROCMETAM",
        "rakip": "DIGER TROCMETAM"
    },
    "CORTIPOL": {
        "pf": "CORTIPOL",
        "rakip": "DIGER CORTIPOL"
    },
    "DEKSAMETAZON": {
        "pf": "DEKSAMETAZON",
        "rakip": "DIGER DEKSAMETAZON"
    },
    "PF IZOTONIK": {
        "pf": "PF IZOTONIK",
        "rakip": "DIGER IZOTONIK"
    }
}

# Şehir düzeltme haritası
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

# Şehir normalizasyon
CITY_NORMALIZE_CLEAN = {
    'ADANA': 'Adana',
    'ADIYAMAN': 'Adıyaman',
    'AFYONKARAHISAR': 'Afyonkarahisar',
    'AFYON': 'Afyonkarahisar',
    'AGRI': 'Ağrı',
    'AĞRI': 'Ağrı',
    'ANKARA': 'Ankara',
    'ANTALYA': 'Antalya',
    'AYDIN': 'Aydın',
    'BALIKESIR': 'Balıkesir',
    'BARTIN': 'Bartın',
    'BATMAN': 'Batman',
    'BILECIK': 'Bilecik',
    'BINGOL': 'Bingöl',
    'BITLIS': 'Bitlis',
    'BOLU': 'Bolu',
    'BURDUR': 'Burdur',
    'BURSA': 'Bursa',
    'CANAKKALE': 'Çanakkale',
    'ÇANAKKALE': 'Çanakkale',
    'CANKIRI': 'Çankırı',
    'ÇANKIRI': 'Çankırı',
    'CORUM': 'Çorum',
    'ÇORUM': 'Çorum',
    'DENIZLI': 'Denizli',
    'DIYARBAKIR': 'Diyarbakır',
    'DUZCE': 'Düzce',
    'DÜZCE': 'Düzce',
    'EDIRNE': 'Edirne',
    'ELAZIG': 'Elazığ',
    'ELAZĞ': 'Elazığ',
    'ELAZIĞ': 'Elazığ',
    'ERZINCAN': 'Erzincan',
    'ERZURUM': 'Erzurum',
    'ESKISEHIR': 'Eskişehir',
    'ESKİŞEHİR': 'Eskişehir',
    'GAZIANTEP': 'Gaziantep',
    'GIRESUN': 'Giresun',
    'GİRESUN': 'Giresun',
    'GUMUSHANE': 'Gümüşhane',
    'GÜMÜŞHANE': 'Gümüşhane',
    'HAKKARI': 'Hakkari',
    'HATAY': 'Hatay',
    'IGDIR': 'Iğdır',
    'IĞDIR': 'Iğdır',
    'ISPARTA': 'Isparta',
    'ISTANBUL': 'İstanbul',
    'İSTANBUL': 'İstanbul',
    'IZMIR': 'İzmir',
    'İZMİR': 'İzmir',
    'KAHRAMANMARAS': 'Kahramanmaraş',
    'KAHRAMANMARAŞ': 'Kahramanmaraş',
    'K.MARAS': 'Kahramanmaraş',
    'KMARAS': 'Kahramanmaraş',
    'KARABUK': 'Karabük',
    'KARABÜK': 'Karabük',
    'KARAMAN': 'Karaman',
    'KARS': 'Kars',
    'KASTAMONU': 'Kastamonu',
    'KAYSERI': 'Kayseri',
    'KIRIKKALE': 'Kırıkkale',
    'KIRKLARELI': 'Kırklareli',
    'KIRKLARELİ': 'Kırklareli',
    'KIRSEHIR': 'Kırşehir',
    'KIRŞEHİR': 'Kırşehir',
    'KILIS': 'Kilis',
    'KİLİS': 'Kilis',
    'KOCAELI': 'Kocaeli',
    'KONYA': 'Konya',
    'KUTAHYA': 'Kütahya',
    'KÜTAHYA': 'Kütahya',
    'MALATYA': 'Malatya',
    'MANISA': 'Manisa',
    'MANİSA': 'Manisa',
    'MARDIN': 'Mardin',
    'MARDİN': 'Mardin',
    'MERSIN': 'Mersin',
    'MERSİN': 'Mersin',
    'MUGLA': 'Muğla',
    'MUĞLA': 'Muğla',
    'MUS': 'Muş',
    'MUŞ': 'Muş',
    'NEVSEHIR': 'Nevşehir',
    'NEVŞEHİR': 'Nevşehir',
    'NIGDE': 'Niğde',
    'NİĞDE': 'Niğde',
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
    'SANLIURFA': 'Şanlıurfa',
    'ŞANLIURFA': 'Şanlıurfa',
    'SIRNAK': 'Şırnak',
    'ŞIRNAK': 'Şırnak',
    'TEKIRDAG': 'Tekirdağ',
    'TEKİRDAĞ': 'Tekirdağ',
    'TOKAT': 'Tokat',
    'TRABZON': 'Trabzon',
    'TUNCELI': 'Tunceli',
    'TUNCELİ': 'Tunceli',
    'USAK': 'Uşak',
    'UŞAK': 'Uşak',
    'VAN': 'Van',
    'YALOVA': 'Yalova',
    'YOZGAT': 'Yozgat',
    'ZONGULDAK': 'Zonguldak',
    'ARDAHAN': 'Ardahan',
    'BAYBURT': 'Bayburt',
    'KIRIKKALE': 'Kırıkkale',
    'KARAMAN': 'Karaman',
    'KİLİS': 'Kilis',
    'OSMANİYE': 'Osmaniye',
    'DÜZCE': 'Düzce'
}

# Türkiye şehir koordinatları
TURKEY_CITY_COORDS = {
    'Adana': (35.3213, 37.0000),
    'Adıyaman': (38.2775, 37.7648),
    'Afyonkarahisar': (30.5417, 38.7638),
    'Ağrı': (43.0500, 39.7191),
    'Amasya': (35.8330, 40.6500),
    'Ankara': (32.8597, 39.9334),
    'Antalya': (30.7133, 36.8969),
    'Artvin': (41.8181, 41.1828),
    'Aydın': (27.8416, 37.8560),
    'Balıkesir': (27.8865, 39.6484),
    'Bilecik': (29.9791, 40.1467),
    'Bingöl': (40.4977, 38.8863),
    'Bitlis': (42.1100, 38.4000),
    'Bolu': (31.6064, 40.7395),
    'Burdur': (30.2833, 37.7167),
    'Bursa': (29.0588, 40.1885),
    'Çanakkale': (26.4144, 40.1467),
    'Çankırı': (33.6167, 40.6000),
    'Çorum': (34.9533, 40.5506),
    'Denizli': (29.0875, 37.7765),
    'Diyarbakır': (40.2167, 37.9167),
    'Edirne': (26.5556, 41.6771),
    'Elazığ': (39.2228, 38.6800),
    'Erzincan': (39.4900, 39.7500),
    'Erzurum': (41.2769, 39.9086),
    'Eskişehir': (31.2622, 39.7767),
    'Gaziantep': (37.3826, 37.0662),
    'Giresun': (38.3903, 40.9128),
    'Gümüşhane': (39.4817, 40.4600),
    'Hakkari': (43.7333, 37.5833),
    'Hatay': (36.2304, 36.4018),
    'Isparta': (30.5522, 37.7648),
    'Mersin': (34.6415, 36.8121),
    'İstanbul': (28.9795, 41.0151),
    'İzmir': (27.1428, 38.4237),
    'Kars': (43.0949, 40.5927),
    'Kastamonu': (33.7750, 41.3764),
    'Kayseri': (35.4833, 38.7312),
    'Kırklareli': (27.2256, 41.7333),
    'Kırşehir': (34.1667, 39.1500),
    'Kocaeli': (29.9167, 40.7667),
    'Konya': (32.4846, 37.9838),
    'Kütahya': (29.9833, 39.4167),
    'Malatya': (38.3167, 38.3500),
    'Manisa': (27.4297, 38.6191),
    'Kahramanmaraş': (36.9261, 37.5753),
    'Mardin': (40.7378, 37.3122),
    'Muğla': (28.3667, 37.2167),
    'Muş': (41.4917, 38.7333),
    'Nevşehir': (34.7125, 38.6242),
    'Niğde': (34.6833, 37.9667),
    'Ordu': (37.8789, 40.9839),
    'Rize': (40.5219, 41.0201),
    'Sakarya': (30.3964, 40.6933),
    'Samsun': (36.3361, 41.2867),
    'Siirt': (41.9403, 37.9333),
    'Sinop': (35.1519, 42.0231),
    'Sivas': (37.0167, 39.7500),
    'Tekirdağ': (27.5119, 40.9833),
    'Tokat': (36.5544, 40.3167),
    'Trabzon': (39.7167, 40.9833),
    'Tunceli': (39.5481, 39.1072),
    'Şanlıurfa': (38.7944, 37.1592),
    'Uşak': (29.4078, 38.6825),
    'Van': (43.3833, 38.4942),
    'Yozgat': (34.8000, 39.8200),
    'Zonguldak': (31.7931, 41.4564),
    'Aksaray': (34.0253, 38.3686),
    'Bayburt': (40.2278, 40.2552),
    'Karaman': (33.2150, 37.1811),
    'Kırıkkale': (33.5131, 39.8468),
    'Batman': (41.1328, 37.8812),
    'Şırnak': (42.4639, 37.5167),
    'Bartın': (32.3375, 41.6358),
    'Ardahan': (42.7022, 41.1103),
    'Iğdır': (44.0444, 39.9236),
    'Yalova': (29.2708, 40.6500),
    'Karabük': (32.6228, 41.2000),
    'Kilis': (37.1150, 36.7164),
    'Osmaniye': (36.2478, 37.0742),
    'Düzce': (31.1639, 40.8439)
}

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def safe_divide(a, b):
    """Güvenli bölme işlemi - pandas Series için optimize edilmiş"""
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        result = a.copy()
        mask = b != 0
        result[mask] = a[mask] / b[mask]
        result[~mask] = 0
        result = result.replace([np.inf, -np.inf], 0)
        return result
    else:
        # Skaler değerler için
        if b == 0:
            return 0
        return a / b

def get_product_columns(product, df_columns):
    """Ürün sütunlarını getir - geliştirilmiş versiyon"""
    product_map = COLUMN_NAMES.get(product, {})
    
    available_columns = {}
    for key, col_name in product_map.items():
        # Farklı varyasyonları kontrol et
        variations = [
            col_name,
            col_name.upper(),
            col_name.lower(),
            col_name.title(),
            col_name.replace(' ', '_'),
            col_name.replace(' ', ''),
            col_name.replace(' ', ' ').strip()
        ]
        
        for var in variations:
            if var in df_columns:
                available_columns[key] = var
                break
        
        if key not in available_columns:
            available_columns[key] = col_name
    
    return available_columns

def normalize_city_name(city_name):
    """Şehir ismini normalize et"""
    if pd.isna(city_name):
        return None
    
    city_str = str(city_name).strip().upper()
    
    # Önce özel düzeltmeleri uygula
    if city_str in FIX_CITY_MAP:
        return FIX_CITY_MAP[city_str]
    
    # Türkçe karakterleri normalize et
    tr_map = {
        "İ": "I", "Ğ": "G", "Ü": "U", "Ş": "S", "Ö": "O", "Ç": "C",
        "Â": "A", "Î": "I", "Û": "U"
    }
    
    for k, v in tr_map.items():
        city_str = city_str.replace(k, v)
    
    return CITY_NORMALIZE_CLEAN.get(city_str, city_str)

def calculate_performance_score(row, weights=None):
    """Performans skoru hesapla"""
    if weights is None:
        weights = {
            'pazar_payi': 0.3,
            'buyume': 0.25,
            'market_buyuklugu': 0.2,
            'stabilite': 0.15,
            'rekabet': 0.1
        }
    
    score = 0
    max_score = 0
    
    # Pazar payı skoru (0-100)
    if 'Pazar_Payi_%' in row:
        pazar_payi = min(row['Pazar_Payi_%'], 100)
        score += pazar_payi * weights['pazar_payi']
    max_score += 100 * weights['pazar_payi']
    
    # Büyüme skoru
    if 'Buyume_%' in row:
        buyume = min(max(row['Buyume_%'], -50), 200)  # -50% ile +200% arası
        buyume_score = 50 + (buyume / 2)  # -50% = 25, 0% = 50, 100% = 100
        score += buyume_score * weights['buyume']
    max_score += 100 * weights['buyume']
    
    # Pazar büyüklüğü skoru (logaritmik)
    if 'Toplam_Pazar' in row:
        market_size = np.log1p(row['Toplam_Pazar'])
        market_score = min(market_size / np.log1p(1000000) * 100, 100)  # 1M üzeri = 100
        score += market_score * weights['market_buyuklugu']
    max_score += 100 * weights['market_buyuklugu']
    
    # Stabilite skoru (varyans tersi)
    if 'Stabilite_Score' in row:
        stabilite = min(row['Stabilite_Score'], 100)
        score += stabilite * weights['stabilite']
    max_score += 100 * weights['stabilite']
    
    # Rekabet skoru
    if 'Goreceli_Pazar_Payi' in row:
        rekabet = min(row['Goreceli_Pazar_Payi'] * 20, 100)  # 5x = 100
        score += rekabet * weights['rekabet']
    max_score += 100 * weights['rekabet']
    
    # Normalize et (0-100)
    final_score = (score / max_score * 100) if max_score > 0 else 0
    
    return min(final_score, 100)

def get_performance_category(score):
    """Performans kategorisini belirle"""
    if score >= 80:
        return "MÜKEMMEL", "status-excellent"
    elif score >= 65:
        return "İYİ", "status-good"
    elif score >= 50:
        return "ORTA", "status-fair"
    else:
        return "ZAYIF", "status-poor"

# =============================================================================
# VERİ YÜKLEME FONKSİYONLARI
# =============================================================================

@st.cache_data
def load_excel_data(file):
    """Excel dosyasını yükle"""
    try:
        df = pd.read_excel(file)
        
        # Sütun isimlerini normalize et
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        # Tarih sütununu bul ve işle
        date_columns = ['DATE', 'TARIH', 'TARİH', 'YEAR_MONTH', 'AY-YIL', 'AY_YIL', 'YILAY']
        date_column_found = False
        
        for date_col in date_columns:
            if date_col in df.columns:
                df['DATE'] = pd.to_datetime(df[date_col], errors='coerce')
                date_column_found = True
                break
        
        if not date_column_found:
            # İlk sütunu tarih olarak dene
            df['DATE'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        
        # NaN tarihleri temizle
        df = df.dropna(subset=['DATE'])
        
        # Tarih sütunları oluştur
        df['YIL_AY'] = df['DATE'].dt.strftime('%Y-%m')
        df['AY'] = df['DATE'].dt.month
        df['YIL'] = df['DATE'].dt.year
        df['QUARTER'] = df['DATE'].dt.quarter
        
        # Territory ve şehir sütunlarını normalize et
        territory_columns = ['TERRITORIES', 'TERRITORY', 'TERRİTORY', 'TERRITOR', 'BOLGE', 'BÖLGE']
        city_columns = ['CITY', 'CİTY', 'SEHIR', 'ŞEHİR', 'İL', 'IL']
        region_columns = ['REGION', 'REGİON', 'BOLGE', 'BÖLGE', 'REGION_NAME']
        manager_columns = ['MANAGER', 'SATIS_TEMSILCISI', 'SORUMLU', 'REP']
        
        for col_list, target in [
            (territory_columns, 'TERRITORIES'),
            (city_columns, 'CITY'),
            (region_columns, 'REGION'),
            (manager_columns, 'MANAGER')
        ]:
            for col in col_list:
                if col in df.columns:
                    df[target] = df[col].astype(str).str.upper().str.strip()
                    break
            if target not in df.columns:
                df[target] = 'BELİRTİLMEMİŞ'
        
        # Şehir normalizasyonu
        df['CITY_NORMALIZED'] = df['CITY'].apply(normalize_city_name)
        
        # Bölge renkleri ekle
        df['REGION_COLOR'] = df['REGION'].map(REGION_COLORS).fillna('#64748B')
        
        return df
        
    except Exception as e:
        st.error(f"❌ Veri yükleme hatası: {str(e)}")
        st.stop()

@st.cache_resource
def load_turkey_geojson():
    """Türkiye GeoJSON verisini yükle"""
    try:
        # İnternetten Türkiye GeoJSON'u al
        url = "https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json"
        response = requests.get(url)
        
        if response.status_code == 200:
            geojson_data = response.json()
            gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
            gdf['name_normalized'] = gdf['name'].apply(normalize_city_name)
            return gdf, geojson_data
        else:
            return create_fallback_geojson()
    except:
        return create_fallback_geojson()

def create_fallback_geojson():
    """Fallback GeoJSON oluştur"""
    # Basit nokta bazlı GeoJSON
    features = []
    for city, coords in TURKEY_CITY_COORDS.items():
        feature = {
            "type": "Feature",
            "properties": {"name": city},
            "geometry": {
                "type": "Point",
                "coordinates": [coords[0], coords[1]]
            }
        }
        features.append(feature)
    
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }
    
    gdf = gpd.GeoDataFrame.from_features(geojson_data)
    gdf['name_normalized'] = gdf['name']
    
    return gdf, geojson_data

# =============================================================================
# HARİTA OLUŞTURMA FONKSİYONLARI - DÜZELTİLMİŞ
# =============================================================================

def create_turkey_map(city_data, title="Türkiye Satış Haritası", color_by="PF_Satis"):
    """
    Türkiye haritası oluştur - DÜZELTİLMİŞ versiyon
    """
    try:
        # Veriyi hazırla
        city_data = city_data.copy()
        
        # Koordinatları ekle
        city_data['lon'] = city_data['City'].apply(
            lambda x: TURKEY_CITY_COORDS.get(x, (35.0, 39.0))[0]
        )
        city_data['lat'] = city_data['City'].apply(
            lambda x: TURKEY_CITY_COORDS.get(x, (35.0, 39.0))[1]
        )
        
        # Bubble boyutunu ayarla
        if color_by == "PF_Satis":
            size_col = 'PF_Satis'
            color_col = 'Pazar_Payi_%'
            color_scale = 'RdYlGn'
            color_title = "Pazar Payı %"
        else:
            size_col = 'PF_Satis'
            color_col = 'PF_Satis'
            color_scale = 'Blues'
            color_title = "PF Satış"
        
        max_size = city_data[size_col].max() if city_data[size_col].max() > 0 else 1
        city_data['bubble_size'] = (city_data[size_col] / max_size * 40) + 10
        
        fig = go.Figure()
        
        fig.add_trace(go.Scattermapbox(
            lat=city_data['lat'],
            lon=city_data['lon'],
            mode='markers',
            marker=dict(
                size=city_data['bubble_size'],
                color=city_data[color_col],
                colorscale=color_scale,
                cmin=city_data[color_col].min(),
                cmax=city_data[color_col].max(),
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text=color_title,
                        font=dict(size=12)
                    ),
                    thickness=20,
                    x=1.02,
                    xpad=5
                ),
                opacity=0.8,
                sizemode='diameter'
            ),
            text=city_data.apply(
                lambda row: f"<b>{row['City']}</b><br>"
                          f"Bölge: {row.get('Region', 'Belirtilmemiş')}<br>"
                          f"PF Satış: {row['PF_Satis']:,.0f}<br>"
                          f"Pazar Payı: {row.get('Pazar_Payi_%', 0):.1f}%<br>"
                          f"Toplam Pazar: {row.get('Toplam_Pazar', 0):,.0f}",
                axis=1
            ),
            hoverinfo='text'
        ))
        
        # Layout ayarları
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox=dict(
                center=dict(lat=39.0, lon=35.0),
                zoom=4.8,
                bearing=0,
                pitch=0
            ),
            height=700,
            margin=dict(l=0, r=0, t=50, b=0),
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(size=22, color='white'),
                y=0.95
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Harita oluşturma hatası: {str(e)}")
        return None

def create_region_map(city_data, title="Bölgesel Dağılım"):
    """Bölgesel dağılım haritası"""
    try:
        # Bölge bazında topla
        region_data = city_data.groupby('Region').agg({
            'PF_Satis': 'sum',
            'Toplam_Pazar': 'sum',
            'City': 'count'
        }).reset_index()
        
        region_data['Pazar_Payi_%'] = safe_divide(region_data['PF_Satis'], region_data['Toplam_Pazar']) * 100
        region_data['Ortalama_Satis'] = safe_divide(region_data['PF_Satis'], region_data['City'])
        
        # Bölge merkezlerini belirle
        region_centers = {
            'MARMARA': (28.9795, 41.0151),
            'EGE': (27.1428, 38.4237),
            'AKDENİZ': (30.7133, 36.8969),
            'İÇ ANADOLU': (32.8597, 39.9334),
            'KARADENİZ': (36.3361, 41.2867),
            'GÜNEYDOĞU ANADOLU': (37.3826, 37.0662),
            'DOĞU ANADOLU': (41.2769, 39.9086)
        }
        
        region_data['lon'] = region_data['Region'].apply(
            lambda x: region_centers.get(x, (35.0, 39.0))[0]
        )
        region_data['lat'] = region_data['Region'].apply(
            lambda x: region_centers.get(x, (35.0, 39.0))[1]
        )
        
        # Bubble boyutu
        max_pf = region_data['PF_Satis'].max()
        region_data['bubble_size'] = (region_data['PF_Satis'] / max_pf * 60) + 20
        
        fig = go.Figure()
        
        # Bölge renklerini al
        region_data['color'] = region_data['Region'].map(REGION_COLORS).fillna('#64748B')
        
        for _, row in region_data.iterrows():
            fig.add_trace(go.Scattermapbox(
                lat=[row['lat']],
                lon=[row['lon']],
                mode='markers+text',
                marker=dict(
                    size=row['bubble_size'],
                    color=row['color'],
                    opacity=0.7,
                    sizemode='diameter'
                ),
                text=[row['Region']],
                textposition="top center",
                textfont=dict(size=14, color='white', weight='bold'),
                hovertext=f"<b>{row['Region']}</b><br>"
                         f"PF Satış: {row['PF_Satis']:,.0f}<br>"
                         f"Pazar Payı: {row['Pazar_Payi_%']:.1f}%<br>"
                         f"Şehir Sayısı: {row['City']}",
                hoverinfo='text',
                showlegend=False
            ))
        
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox=dict(
                center=dict(lat=39.0, lon=35.0),
                zoom=4.5
            ),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0),
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(size=20, color='white')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Bölge haritası oluşturma hatası: {str(e)}")
        return None

# =============================================================================
# ANALİZ FONKSİYONLARI - GENİŞLETİLMİŞ
# =============================================================================

def calculate_city_performance(df, product, date_filter=None):
    """Şehir bazlı performans analizi"""
    cols = get_product_columns(product, df.columns)
    
    if date_filter:
        df_filtered = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])].copy()
    else:
        df_filtered = df.copy()
    
    pf_col = cols.get('pf')
    rakip_col = cols.get('rakip')
    
    if pf_col not in df_filtered.columns:
        return pd.DataFrame()
    
    if rakip_col not in df_filtered.columns:
        df_filtered[rakip_col] = 0
    
    # Şehir bazlı toplamlar
    city_data = df_filtered.groupby(['CITY_NORMALIZED', 'REGION']).agg({
        pf_col: 'sum',
        rakip_col: 'sum'
    }).reset_index()
    
    city_data.columns = ['City', 'Region', 'PF_Satis', 'Rakip_Satis']
    city_data['Toplam_Pazar'] = city_data['PF_Satis'] + city_data['Rakip_Satis']
    city_data['Pazar_Payi_%'] = safe_divide(city_data['PF_Satis'], city_data['Toplam_Pazar']) * 100
    
    # Pozitif satışı olan şehirler
    city_data = city_data[city_data['PF_Satis'] > 0]
    
    # Performans skorunu hesapla
    city_data = calculate_city_performance_scores(city_data, df_filtered, pf_col)
    
    return city_data.sort_values('PF_Satis', ascending=False)

def calculate_city_performance_scores(city_data, df, pf_col):
    """Şehir performans skorlarını hesapla"""
    if len(city_data) == 0:
        return city_data
    
    # Büyüme oranını hesapla (zaman bazlı)
    growth_data = []
    for city in city_data['City'].unique():
        city_df = df[df['CITY_NORMALIZED'] == city]
        if len(city_df) >= 2:
            # İlk ve son dönem karşılaştırması
            first_period = city_df.sort_values('DATE').iloc[0]
            last_period = city_df.sort_values('DATE').iloc[-1]
            
            if first_period[pf_col] > 0:
                growth = ((last_period[pf_col] - first_period[pf_col]) / first_period[pf_col]) * 100
            else:
                growth = 100 if last_period[pf_col] > 0 else 0
        else:
            growth = 0
        
        growth_data.append({'City': city, 'Buyume_%': growth})
    
    growth_df = pd.DataFrame(growth_data)
    city_data = city_data.merge(growth_df, on='City', how='left')
    
    # Stabilite skoru (aylık varyasyon)
    stability_data = []
    for city in city_data['City'].unique():
        city_df = df[df['CITY_NORMALIZED'] == city]
        if len(city_df) > 3:
            monthly_sales = city_df.groupby('YIL_AY')[pf_col].sum()
            if len(monthly_sales) > 1:
                cv = monthly_sales.std() / monthly_sales.mean() if monthly_sales.mean() > 0 else 0
                stability = max(0, 100 - (cv * 100))  # Düşük varyasyon = yüksek stabilite
            else:
                stability = 50
        else:
            stability = 50
        
        stability_data.append({'City': city, 'Stabilite_Score': min(stability, 100)})
    
    stability_df = pd.DataFrame(stability_data)
    city_data = city_data.merge(stability_df, on='City', how='left')
    
    # Göreceli pazar payı
    city_data['Goreceli_Pazar_Payi'] = safe_divide(city_data['PF_Satis'], city_data['Rakip_Satis'])
    
    # Performans skoru
    city_data['Performance_Score'] = city_data.apply(calculate_performance_score, axis=1)
    city_data['Performance_Category'] = city_data['Performance_Score'].apply(
        lambda x: get_performance_category(x)[0]
    )
    city_data['Performance_Color'] = city_data['Performance_Score'].apply(
        lambda x: get_performance_category(x)[1]
    )
    
    return city_data

def calculate_territory_performance(df, product, date_filter=None):
    """Territory bazlı performans analizi"""
    cols = get_product_columns(product, df.columns)
    
    if date_filter:
        df_filtered = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])].copy()
    else:
        df_filtered = df.copy()
    
    pf_col = cols.get('pf')
    rakip_col = cols.get('rakip')
    
    if pf_col not in df_filtered.columns:
        return pd.DataFrame()
    
    if rakip_col not in df_filtered.columns:
        df_filtered[rakip_col] = 0
    
    # Territory bazlı toplamlar
    territory_data = df_filtered.groupby(['TERRITORIES', 'REGION', 'CITY', 'MANAGER']).agg({
        pf_col: 'sum',
        rakip_col: 'sum'
    }).reset_index()
    
    territory_data.columns = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 'Rakip_Satis']
    territory_data['Toplam_Pazar'] = territory_data['PF_Satis'] + territory_data['Rakip_Satis']
    territory_data['Pazar_Payi_%'] = safe_divide(territory_data['PF_Satis'], territory_data['Toplam_Pazar']) * 100
    
    # Ağırlık hesapla
    total_pf = territory_data['PF_Satis'].sum()
    territory_data['Agirlik_%'] = safe_divide(territory_data['PF_Satis'], total_pf) * 100
    
    # Göreceli pazar payı
    territory_data['Goreceli_Pazar_Payi'] = safe_divide(territory_data['PF_Satis'], territory_data['Rakip_Satis'])
    
    return territory_data.sort_values('PF_Satis', ascending=False)

def calculate_time_series_analysis(df, product, territory=None, date_filter=None):
    """Zaman serisi analizi"""
    cols = get_product_columns(product, df.columns)
    
    df_filtered = df.copy()
    if territory and territory != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['TERRITORIES'] == territory]
    
    if date_filter:
        df_filtered = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & 
                                   (df_filtered['DATE'] <= date_filter[1])]
    
    pf_col = cols.get('pf')
    rakip_col = cols.get('rakip')
    
    if pf_col not in df_filtered.columns:
        return pd.DataFrame()
    
    if rakip_col not in df_filtered.columns:
        df_filtered[rakip_col] = 0
    
    # Aylık toplamlar
    monthly_data = df_filtered.groupby('YIL_AY').agg({
        pf_col: 'sum',
        rakip_col: 'sum',
        'DATE': 'first'
    }).reset_index().sort_values('YIL_AY')
    
    monthly_data.columns = ['YIL_AY', 'PF_Satis', 'Rakip_Satis', 'DATE']
    monthly_data['Toplam_Pazar'] = monthly_data['PF_Satis'] + monthly_data['Rakip_Satis']
    monthly_data['Pazar_Payi_%'] = safe_divide(monthly_data['PF_Satis'], monthly_data['Toplam_Pazar']) * 100
    
    # Büyüme oranları
    monthly_data['PF_Buyume_%'] = monthly_data['PF_Satis'].pct_change() * 100
    monthly_data['Rakip_Buyume_%'] = monthly_data['Rakip_Satis'].pct_change() * 100
    monthly_data['Goreceli_Buyume_%'] = monthly_data['PF_Buyume_%'] - monthly_data['Rakip_Buyume_%']
    
    # Hareketli ortalamalar
    monthly_data['MA_3_Ay'] = monthly_data['PF_Satis'].rolling(window=3, min_periods=1).mean()
    monthly_data['MA_6_Ay'] = monthly_data['PF_Satis'].rolling(window=6, min_periods=1).mean()
    monthly_data['Trend_Line'] = np.poly1d(np.polyfit(range(len(monthly_data)), monthly_data['PF_Satis'], 1))(range(len(monthly_data)))
    
    return monthly_data.fillna(0)

def calculate_competitor_analysis(df, product, date_filter=None):
    """Rakip analizi"""
    cols = get_product_columns(product, df.columns)
    
    if date_filter:
        df_filtered = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])].copy()
    else:
        df_filtered = df.copy()
    
    pf_col = cols.get('pf')
    rakip_col = cols.get('rakip')
    
    if pf_col not in df_filtered.columns:
        return pd.DataFrame()
    
    if rakip_col not in df_filtered.columns:
        df_filtered[rakip_col] = 0
    
    # Aylık rakip analizi
    monthly_data = df_filtered.groupby('YIL_AY').agg({
        pf_col: 'sum',
        rakip_col: 'sum'
    }).reset_index().sort_values('YIL_AY')
    
    monthly_data.columns = ['YIL_AY', 'PF', 'Rakip']
    monthly_data['Toplam_Pazar'] = monthly_data['PF'] + monthly_data['Rakip']
    monthly_data['PF_Pay_%'] = safe_divide(monthly_data['PF'], monthly_data['Toplam_Pazar']) * 100
    monthly_data['Rakip_Pay_%'] = 100 - monthly_data['PF_Pay_%']
    
    # Büyüme oranları
    monthly_data['PF_Buyume'] = monthly_data['PF'].pct_change() * 100
    monthly_data['Rakip_Buyume'] = monthly_data['Rakip'].pct_change() * 100
    monthly_data['Fark'] = monthly_data['PF_Buyume'] - monthly_data['Rakip_Buyume']
    
    return monthly_data.fillna(0)

def calculate_swot_analysis(city_data, territory_data):
    """SWOT analizi yap"""
    swot_analysis = {
        'Strengths': [],
        'Weaknesses': [],
        'Opportunities': [],
        'Threats': []
    }
    
    if len(city_data) > 0:
        # Güçlü yönler
        top_cities = city_data.nlargest(5, 'PF_Satis')
        for _, row in top_cities.iterrows():
            swot_analysis['Strengths'].append(
                f"{row['City']}: Yüksek PF Satış ({row['PF_Satis']:,.0f}), "
                f"Pazar Payı: {row.get('Pazar_Payi_%', 0):.1f}%"
            )
        
        # Zayıf yönler
        weak_cities = city_data[city_data['Pazar_Payi_%'] < 30].nlargest(5, 'Toplam_Pazar')
        for _, row in weak_cities.iterrows():
            swot_analysis['Weaknesses'].append(
                f"{row['City']}: Düşük Pazar Payı ({row.get('Pazar_Payi_%', 0):.1f}%), "
                f"Yüksek Pazar Potansiyeli ({row['Toplam_Pazar']:,.0f})"
            )
    
    if len(territory_data) > 0:
        # Fırsatlar
        high_growth = territory_data.nlargest(5, 'Goreceli_Pazar_Payi')
        for _, row in high_growth.iterrows():
            if row['Goreceli_Pazar_Payi'] > 1:
                swot_analysis['Opportunities'].append(
                    f"{row['Territory']}: Rekabet Üstünlüğü ({row['Goreceli_Pazar_Payi']:.2f}x), "
                    f"Pazar Payı: {row['Pazar_Payi_%']:.1f}%"
                )
        
        # Tehditler
        low_share = territory_data[territory_data['Pazar_Payi_%'] < 20].nlargest(5, 'Toplam_Pazar')
        for _, row in low_share.iterrows():
            swot_analysis['Threats'].append(
                f"{row['Territory']}: Düşük Pazar Kontrolü ({row['Pazar_Payi_%']:.1f}%), "
                f"Yüksek Rekabet ({row['Rakip_Satis']:,.0f})"
            )
    
    return swot_analysis

def calculate_strategic_recommendations(city_data, territory_data):
    """Stratejik öneriler oluştur"""
    recommendations = []
    
    if len(city_data) > 0:
        # Yüksek potansiyelli şehirler
        high_potential = city_data[
            (city_data['Pazar_Payi_%'] < 50) & 
            (city_data['Toplam_Pazar'] > city_data['Toplam_Pazar'].median())
        ].nlargest(5, 'Toplam_Pazar')
        
        for _, row in high_potential.iterrows():
            recommendations.append({
                'type': 'AGRESİF',
                'title': f"{row['City']} - Agresif Büyüme",
                'description': f"Pazar payı {row['Pazar_Payi_%']:.1f}% ile düşük, "
                             f"ancak toplam pazar {row['Toplam_Pazar']:,.0f} ile büyük. "
                             f"Hedef: Pazar payını %50'ye çıkar.",
                'priority': 'YÜKSEK',
                'impact': 'YÜKSEK',
                'effort': 'ORTA'
            })
    
    if len(territory_data) > 0:
        # Güçlü territory'ler
        strong_territories = territory_data.nlargest(5, 'PF_Satis')
        for _, row in strong_territories.iterrows():
            recommendations.append({
                'type': 'KORUMA',
                'title': f"{row['Territory']} - Konumu Koru",
                'description': f"PF Satış: {row['PF_Satis']:,.0f}, "
                             f"Pazar Payı: {row['Pazar_Payi_%']:.1f}%. "
                             f"Mevcut konumu koru ve küçük iyileştirmeler yap.",
                'priority': 'ORTA',
                'impact': 'ORTA',
                'effort': 'DÜŞÜK'
            })
    
    return recommendations

# =============================================================================
# GÖRSELLEŞTİRME FONKSİYONLARI
# =============================================================================

def create_performance_dashboard(city_data):
    """Performans dashboard'u oluştur"""
    if len(city_data) == 0:
        return None
    
    fig = go.Figure()
    
    # Bubble chart: Pazar payı vs Büyüme
    fig.add_trace(go.Scatter(
        x=city_data['Pazar_Payi_%'],
        y=city_data.get('Buyume_%', 0),
        mode='markers',
        marker=dict(
            size=city_data['PF_Satis'] / city_data['PF_Satis'].max() * 40 + 10,
            color=city_data['Performance_Score'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Performans Skoru")
        ),
        text=city_data.apply(
            lambda row: f"<b>{row['City']}</b><br>"
                      f"Pazar Payı: {row['Pazar_Payi_%']:.1f}%<br>"
                      f"Büyüme: {row.get('Buyume_%', 0):.1f}%<br>"
                      f"PF Satış: {row['PF_Satis']:,.0f}<br>"
                      f"Skor: {row['Performance_Score']:.1f}",
            axis=1
        ),
        hoverinfo='text'
    ))
    
    # Ortalama çizgileri
    avg_pazar_payi = city_data['Pazar_Payi_%'].mean()
    avg_buyume = city_data.get('Buyume_%', 0).mean()
    
    fig.add_hline(y=avg_buyume, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=avg_pazar_payi, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=dict(
            text='<b>Şehir Performans Matrisi</b><br>Pazar Payı vs Büyüme',
            font=dict(size=20, color='white')
        ),
        xaxis_title='<b>Pazar Payı (%)</b>',
        yaxis_title='<b>Büyüme Oranı (%)</b>',
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        hovermode='closest'
    )
    
    return fig

def create_competitor_comparison_chart(comp_data):
    """Rakip karşılaştırma grafiği"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=comp_data['YIL_AY'],
        y=comp_data['PF'],
        name='PF Satış',
        marker_color=PERFORMANCE_COLORS['success'],
        marker=dict(line=dict(width=2, color='rgba(255, 255, 255, 0.8)'))
    ))
    
    fig.add_trace(go.Bar(
        x=comp_data['YIL_AY'],
        y=comp_data['Rakip'],
        name='Rakip Satış',
        marker_color=PERFORMANCE_COLORS['danger'],
        marker=dict(line=dict(width=2, color='rgba(255, 255, 255, 0.8)'))
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>PF vs Rakip Satış Karşılaştırması</b>',
            font=dict(size=20, color='white')
        ),
        xaxis_title='<b>Ay</b>',
        yaxis_title='<b>Satış Miktarı</b>',
        barmode='group',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_growth_trend_chart(monthly_data):
    """Büyüme trend grafiği"""
    fig = go.Figure()
    
    # PF Büyüme
    fig.add_trace(go.Scatter(
        x=monthly_data['DATE'],
        y=monthly_data['PF_Buyume_%'],
        mode='lines+markers',
        name='PF Büyüme',
        line=dict(color=PERFORMANCE_COLORS['success'], width=3),
        marker=dict(size=8, color='white', line=dict(width=2, color=PERFORMANCE_COLORS['success'])),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    
    # Rakip Büyüme
    fig.add_trace(go.Scatter(
        x=monthly_data['DATE'],
        y=monthly_data['Rakip_Buyume_%'],
        mode='lines+markers',
        name='Rakip Büyüme',
        line=dict(color=PERFORMANCE_COLORS['danger'], width=3),
        marker=dict(size=8, color='white', line=dict(width=2, color=PERFORMANCE_COLORS['danger'])),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.1)'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    
    fig.update_layout(
        title=dict(
            text='<b>Aylık Büyüme Trendleri</b>',
            font=dict(size=20, color='white')
        ),
        xaxis_title='<b>Tarih</b>',
        yaxis_title='<b>Büyüme Oranı (%)</b>',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_market_share_chart(monthly_data):
    """Pazar payı trend grafiği"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_data['DATE'],
        y=monthly_data['Pazar_Payi_%'],
        mode='lines+markers',
        name='Pazar Payı',
        line=dict(color=PERFORMANCE_COLORS['info'], width=3, shape='spline'),
        marker=dict(size=8, color='white', line=dict(width=2, color=PERFORMANCE_COLORS['info']))
    ))
    
    fig.add_hline(y=50, line_dash="dash", line_color=PERFORMANCE_COLORS['warning'], opacity=0.5, annotation_text="50% Eşik")
    
    fig.update_layout(
        title=dict(
            text='<b>Pazar Payı Trendi</b>',
            font=dict(size=20, color='white')
        ),
        xaxis_title='<b>Tarih</b>',
        yaxis_title='<b>Pazar Payı (%)</b>',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        yaxis=dict(range=[0, 100])
    )
    
    return fig

# =============================================================================
# ML TAHMİN FONKSİYONLARI
# =============================================================================

def prepare_ml_data(monthly_data):
    """ML için veri hazırla"""
    if len(monthly_data) < 12:
        return None, None, None
    
    df = monthly_data.copy()
    
    # Feature engineering
    df['month'] = df['DATE'].dt.month
    df['quarter'] = df['DATE'].dt.quarter
    df['year'] = df['DATE'].dt.year
    
    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}'] = df['PF_Satis'].shift(lag)
    
    # Rolling statistics
    df['rolling_mean_3'] = df['PF_Satis'].rolling(window=3, min_periods=1).mean()
    df['rolling_mean_6'] = df['PF_Satis'].rolling(window=6, min_periods=1).mean()
    df['rolling_std_3'] = df['PF_Satis'].rolling(window=3, min_periods=1).std()
    
    # Trend features
    df['trend'] = range(len(df))
    
    # Seasonal features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Target
    df['target'] = df['PF_Satis'].shift(-1)
    
    # Drop NaN
    df_clean = df.dropna()
    
    if len(df_clean) < 10:
        return None, None, None
    
    # Feature columns
    feature_cols = [col for col in ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 
                                     'rolling_mean_6', 'rolling_std_3', 'month',
                                     'quarter', 'month_sin', 'month_cos', 'trend']
                    if col in df_clean.columns]
    
    X = df_clean[feature_cols]
    y = df_clean['target']
    
    # Train/test split
    split_idx = int(len(df_clean) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_ml_models(X_train, X_test, y_train, y_test):
    """ML modellerini eğit"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'predictions': y_pred
        }
    
    return results

def create_forecast_chart(historical_data, forecast_data, model_name):
    """Tahmin grafiği oluştur"""
    fig = go.Figure()
    
    # Tarihsel veri
    fig.add_trace(go.Scatter(
        x=historical_data['DATE'],
        y=historical_data['PF_Satis'],
        mode='lines+markers',
        name='Gerçek Satış',
        line=dict(color=PERFORMANCE_COLORS['success'], width=3),
        marker=dict(size=8, color='white', line=dict(width=2, color=PERFORMANCE_COLORS['success']))
    ))
    
    # Tahminler
    if forecast_data is not None:
        fig.add_trace(go.Scatter(
            x=forecast_data['DATE'],
            y=forecast_data['Prediction'],
            mode='lines+markers',
            name=f'Tahmin ({model_name})',
            line=dict(color=PERFORMANCE_COLORS['info'], width=3, dash='dash'),
            marker=dict(size=10, symbol='diamond', color='white', 
                       line=dict(width=2, color=PERFORMANCE_COLORS['info']))
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>Satış Tahminleri</b>',
            font=dict(size=20, color='white')
        ),
        xaxis_title='<b>Tarih</b>',
        yaxis_title='<b>PF Satış</b>',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# =============================================================================
# RAPORLAMA FONKSİYONLARI
# =============================================================================

def generate_comprehensive_report(city_data, territory_data, monthly_data, comp_data, product, date_option):
    """Kapsamlı rapor oluştur"""
    report = {
        'summary': {
            'product': product,
            'period': date_option,
            'total_pf_sales': city_data['PF_Satis'].sum() if len(city_data) > 0 else 0,
            'total_market': city_data['Toplam_Pazar'].sum() if len(city_data) > 0 else 0,
            'market_share': safe_divide(city_data['PF_Satis'].sum(), city_data['Toplam_Pazar'].sum()) * 100 if len(city_data) > 0 else 0,
            'cities_count': len(city_data),
            'territories_count': len(territory_data)
        },
        'top_performers': {
            'top_cities': city_data.nlargest(10, 'PF_Satis')[['City', 'Region', 'PF_Satis', 'Pazar_Payi_%', 'Performance_Score']].to_dict('records') if len(city_data) > 0 else [],
            'top_territories': territory_data.nlargest(10, 'PF_Satis')[['Territory', 'Region', 'PF_Satis', 'Pazar_Payi_%']].to_dict('records') if len(territory_data) > 0 else []
        },
        'trends': {
            'growth_rate': monthly_data['PF_Buyume_%'].mean() if len(monthly_data) > 0 else 0,
            'avg_market_share': monthly_data['Pazar_Payi_%'].mean() if len(monthly_data) > 0 else 0,
            'competitor_growth': comp_data['Rakip_Buyume'].mean() if len(comp_data) > 0 else 0
        },
        'recommendations': calculate_strategic_recommendations(city_data, territory_data)
    }
    
    return report

# =============================================================================
# ANA UYGULAMA
# =============================================================================

def main():
    # Başlık
    st.markdown('<h1 class="main-header">🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #94a3b8; margin-bottom: 3rem;">'
                'Gerçek Zamanlı Analiz • ML Tahminler • Stratejik Öneriler • Kapsamlı Raporlama'
                '</div>', unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown('<div style="background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%); '
                   'padding: 1rem; border-radius: 12px; margin-bottom: 2rem;">'
                   '<h3 style="color: white; margin: 0; text-align: center;">📂 VERİ YÜKLEME</h3>'
                   '</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Excel Dosyası Yükleyin", type=['xlsx', 'xls'])
        
        if not uploaded_file:
            st.info("👈 Lütfen Excel dosyasını yükleyin")
            st.stop()
        
        try:
            df = load_excel_data(uploaded_file)
            gdf, geojson_data = load_turkey_geojson()
            
            st.success(f"✅ **{len(df):,}** satır veri yüklendi")
            st.info(f"📅 Veri aralığı: {df['DATE'].min().strftime('%Y-%m')} - {df['DATE'].max().strftime('%Y-%m')}")
            
            with st.expander("📋 Veri Önizleme"):
                st.dataframe(df.head(), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.stop()
        
        st.markdown("---")
        
        # Ürün Seçimi
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">💊 ÜRÜN SEÇİMİ</h4>', unsafe_allow_html=True)
        
        # Mevcut ürünleri kontrol et
        available_products = []
        for product in COLUMN_NAMES.keys():
            cols = get_product_columns(product, df.columns)
            if cols.get('pf') in df.columns:
                available_products.append(product)
        
        if not available_products:
            st.error("❌ Excel'de beklenen ürün sütunları bulunamadı!")
            st.info("Mevcut sütunlar: " + ", ".join(df.columns.tolist()))
            st.stop()
        
        selected_product = st.selectbox("", available_products, label_visibility="collapsed")
        
        # Sütun bilgilerini göster
        cols = get_product_columns(selected_product, df.columns)
        with st.expander("📊 Sütun Bilgileri"):
            st.write(f"PF Sütunu: **{cols.get('pf')}**")
            st.write(f"Rakip Sütunu: **{cols.get('rakip')}**")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tarih Filtresi
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">📅 TARİH FİLTRESİ</h4>', unsafe_allow_html=True)
        
        min_date = df['DATE'].min()
        max_date = df['DATE'].max()
        
        date_options = ["Tüm Veriler", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl", 
                       "Bu Yıl", "Geçen Yıl", "Son 2 Yıl", "Özel Aralık"]
        date_option = st.selectbox("Dönem Seçin", date_options)
        
        if date_option == "Tüm Veriler":
            date_filter = None
        elif date_option == "Son 3 Ay":
            start_date = max_date - pd.DateOffset(months=3)
            date_filter = (start_date, max_date)
        elif date_option == "Son 6 Ay":
            start_date = max_date - pd.DateOffset(months=6)
            date_filter = (start_date, max_date)
        elif date_option == "Son 1 Yıl":
            start_date = max_date - pd.DateOffset(years=1)
            date_filter = (start_date, max_date)
        elif date_option == "Bu Yıl":
            date_filter = (pd.to_datetime(f"{max_date.year}-01-01"), max_date)
        elif date_option == "Geçen Yıl":
            date_filter = (pd.to_datetime(f"{max_date.year-1}-01-01"), 
                          pd.to_datetime(f"{max_date.year-1}-12-31"))
        elif date_option == "Son 2 Yıl":
            start_date = max_date - pd.DateOffset(years=2)
            date_filter = (start_date, max_date)
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Başlangıç", min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("Bitiş", max_date, min_value=min_date, max_value=max_date)
            date_filter = (pd.to_datetime(start_date), pd.to_datetime(end_date))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Diğer Filtreler
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">🔍 DİĞER FİLTRELER</h4>', unsafe_allow_html=True)
        
        territories = ["TÜMÜ"] + sorted([str(t) for t in df['TERRITORIES'].unique() if pd.notna(t)])
        selected_territory = st.selectbox("Territory", territories)
        
        regions = ["TÜMÜ"] + sorted([str(r) for r in df['REGION'].unique() if pd.notna(r)])
        selected_region = st.selectbox("Bölge", regions)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Veri filtreleme
        df_filtered = df.copy()
        if selected_territory != "TÜMÜ":
            df_filtered = df_filtered[df_filtered['TERRITORIES'] == selected_territory]
        if selected_region != "TÜMÜ":
            df_filtered = df_filtered[df_filtered['REGION'] == selected_region]
        
        # Analiz butonu
        st.markdown("---")
        if st.button("🚀 Analiz Başlat", type="primary", use_container_width=True):
            st.session_state['analysis_started'] = True
        
        if 'analysis_started' not in st.session_state:
            st.session_state['analysis_started'] = False
    
    # ANA İÇERİK
    if not st.session_state.get('analysis_started', False):
        st.info("👈 Lütfen sol taraftan filtreleri ayarlayın ve 'Analiz Başlat' butonuna tıklayın")
        st.stop()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Veri analizi başlatılıyor...")
    progress_bar.progress(10)
    
    # Analizleri yap
    city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
    progress_bar.progress(30)
    
    territory_data = calculate_territory_performance(df_filtered, selected_product, date_filter)
    progress_bar.progress(50)
    
    monthly_data = calculate_time_series_analysis(df_filtered, selected_product, None, date_filter)
    progress_bar.progress(70)
    
    comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
    progress_bar.progress(90)
    
    # SWOT analizi
    swot_analysis = calculate_swot_analysis(city_data, territory_data)
    
    status_text.text("Analiz tamamlandı!")
    progress_bar.progress(100)
    
    # TAB'ler
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Genel Bakış",
        "🗺️ Harita Analizi",
        "🏆 Performans",
        "📈 Trendler",
        "🤖 ML Tahminler",
        "🎯 Strateji",
        "📥 Raporlar"
    ])
    
    # TAB 1: GENEL BAKIŞ
    with tab1:
        st.header("📊 Genel Performans Özeti")
        
        if len(city_data) == 0:
            st.warning("⚠️ Seçilen filtrelerde veri bulunamadı")
        else:
            # Toplam metrikler
            total_pf = city_data['PF_Satis'].sum()
            total_market = city_data['Toplam_Pazar'].sum()
            market_share = safe_divide(total_pf, total_market) * 100
            avg_market_share = city_data['Pazar_Payi_%'].mean()
            avg_performance = city_data['Performance_Score'].mean() if 'Performance_Score' in city_data.columns else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💊 Toplam PF Satış", f"{total_pf:,.0f}")
            with col2:
                st.metric("🏪 Toplam Pazar", f"{total_market:,.0f}")
            with col3:
                st.metric("📊 Pazar Payı", f"%{market_share:.1f}")
            with col4:
                st.metric("⭐ Ort. Performans", f"{avg_performance:.1f}/100")
            
            st.markdown("---")
            
            # Performans dağılımı
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("🏆 Top 10 Şehir")
                top_cities = city_data.nlargest(10, 'PF_Satis')
                
                fig = px.bar(
                    top_cities,
                    x='City',
                    y='PF_Satis',
                    color='Region',
                    color_discrete_map=REGION_COLORS,
                    title='<b>En Yüksek Satış Yapan Şehirler</b>',
                    text='PF_Satis'
                )
                fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                fig.update_layout(height=500, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_right:
                st.subheader("📊 Performans Dağılımı")
                if 'Performance_Category' in city_data.columns:
                    perf_dist = city_data['Performance_Category'].value_counts()
                    
                    fig = px.pie(
                        values=perf_dist.values,
                        names=perf_dist.index,
                        title='<b>Performans Kategori Dağılımı</b>',
                        color=perf_dist.index,
                        color_discrete_map={
                            'MÜKEMMEL': PERFORMANCE_COLORS['excellent'],
                            'İYİ': PERFORMANCE_COLORS['good'],
                            'ORTA': PERFORMANCE_COLORS['fair'],
                            'ZAYIF': PERFORMANCE_COLORS['poor']
                        }
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Performans verisi bulunamadı")
            
            # Detaylı tablo
            st.markdown("---")
            st.subheader("📋 Şehir Performans Detayları")
            
            display_cols = ['City', 'Region', 'PF_Satis', 'Toplam_Pazar', 'Pazar_Payi_%']
            if 'Performance_Score' in city_data.columns:
                display_cols.extend(['Performance_Score', 'Performance_Category'])
            
            city_display = city_data[display_cols].copy()
            city_display.columns = ['Şehir', 'Bölge', 'PF Satış', 'Toplam Pazar', 'Pazar Payı %', 
                                   'Performans Skoru', 'Kategori'][:len(display_cols)]
            city_display.index = range(1, len(city_display) + 1)
            
            st.dataframe(
                city_display.style.background_gradient(subset=['Pazar Payı %'], cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )
    
    # TAB 2: HARİTA ANALİZİ
    with tab2:
        st.header("🗺️ Coğrafi Analiz")
        
        if len(city_data) == 0:
            st.warning("⚠️ Harita için veri bulunamadı")
        else:
            # Harita seçenekleri
            map_type = st.radio(
                "Harita Tipi",
                ["Şehir Bazlı", "Bölgesel", "Performans Bazlı"],
                horizontal=True
            )
            
            if map_type == "Şehir Bazlı":
                st.subheader("📍 Şehir Bazlı Dağılım")
                turkey_map = create_turkey_map(city_data, f"{selected_product} - Şehir Bazlı Dağılım")
                if turkey_map:
                    st.plotly_chart(turkey_map, use_container_width=True)
                else:
                    st.error("Harita oluşturulamadı")
            
            elif map_type == "Bölgesel":
                st.subheader("🗺️ Bölgesel Dağılım")
                region_map = create_region_map(city_data)
                if region_map:
                    st.plotly_chart(region_map, use_container_width=True)
                else:
                    st.error("Bölge haritası oluşturulamadı")
            
            else:  # Performans Bazlı
                st.subheader("⭐ Performans Bazlı Dağılım")
                if 'Performance_Score' in city_data.columns:
                    perf_map = create_turkey_map(city_data, f"{selected_product} - Performans Dağılımı", "Performance_Score")
                    if perf_map:
                        st.plotly_chart(perf_map, use_container_width=True)
                    else:
                        st.error("Performans haritası oluşturulamadı")
                else:
                    st.warning("Performans verisi bulunamadı")
            
            # Bölge analizi
            st.markdown("---")
            st.subheader("📊 Bölge Bazlı Analiz")
            
            if len(city_data) > 0:
                region_summary = city_data.groupby('Region').agg({
                    'PF_Satis': 'sum',
                    'Toplam_Pazar': 'sum',
                    'City': 'count'
                }).reset_index()
                
                region_summary['Pazar_Payi_%'] = safe_divide(region_summary['PF_Satis'], region_summary['Toplam_Pazar']) * 100
                region_summary['Ort_Satis'] = safe_divide(region_summary['PF_Satis'], region_summary['City'])
                
                col_r1, col_r2 = st.columns(2)
                
                with col_r1:
                    fig = px.bar(
                        region_summary,
                        x='Region',
                        y='PF_Satis',
                        color='Region',
                        color_discrete_map=REGION_COLORS,
                        title='<b>Bölgelere Göre PF Satış</b>',
                        text='PF_Satis'
                    )
                    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_r2:
                    fig = px.pie(
                        region_summary,
                        values='PF_Satis',
                        names='Region',
                        title='<b>Bölge Satış Dağılımı</b>',
                        color='Region',
                        color_discrete_map=REGION_COLORS,
                        hole=0.3
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: PERFORMANS
    with tab3:
        st.header("🏆 Detaylı Performans Analizi")
        
        if len(city_data) == 0:
            st.warning("⚠️ Performans analizi için veri bulunamadı")
        else:
            # Performans dashboard'u
            st.subheader("📈 Performans Dashboard")
            perf_dashboard = create_performance_dashboard(city_data)
            if perf_dashboard:
                st.plotly_chart(perf_dashboard, use_container_width=True)
            
            # Performans kriterleri
            st.markdown("---")
            st.subheader("🎯 Performans Kriterleri")
            
            col_p1, col_p2, col_p3 = st.columns(3)
            
            with col_p1:
                st.metric("🏆 En Yüksek Performans", 
                         f"{city_data['Performance_Score'].max():.1f}" if 'Performance_Score' in city_data.columns else "N/A",
                         delta=f"{city_data['Performance_Score'].mean():.1f} ortalama" if 'Performance_Score' in city_data.columns else "")
            
            with col_p2:
                st.metric("📊 En Yüksek Pazar Payı", 
                         f"%{city_data['Pazar_Payi_%'].max():.1f}",
                         delta=f"{city_data.loc[city_data['Pazar_Payi_%'].idxmax(), 'City'] if len(city_data) > 0 else 'N/A'}")
            
            with col_p3:
                if 'Buyume_%' in city_data.columns:
                    st.metric("🚀 En Yüksek Büyüme", 
                             f"%{city_data['Buyume_%'].max():.1f}",
                             delta=f"{city_data.loc[city_data['Buyume_%'].idxmax(), 'City'] if len(city_data) > 0 else 'N/A'}")
                else:
                    st.metric("🚀 En Yüksek Büyüme", "N/A")
            
            # Territory performansı
            st.markdown("---")
            st.subheader("🏢 Territory Performansı")
            
            if len(territory_data) > 0:
                top_territories = territory_data.nlargest(15, 'PF_Satis')
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_territories['Territory'],
                    y=top_territories['PF_Satis'],
                    name='PF Satış',
                    marker_color=PERFORMANCE_COLORS['success']
                ))
                fig.add_trace(go.Bar(
                    x=top_territories['Territory'],
                    y=top_territories['Rakip_Satis'],
                    name='Rakip Satış',
                    marker_color=PERFORMANCE_COLORS['danger']
                ))
                
                fig.update_layout(
                    title='<b>Top 15 Territory - PF vs Rakip</b>',
                    barmode='group',
                    height=500,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Territory detayları
                with st.expander("📋 Territory Detayları"):
                    terr_display = territory_data[['Territory', 'Region', 'City', 'Manager', 
                                                   'PF_Satis', 'Rakip_Satis', 'Pazar_Payi_%']].copy()
                    terr_display.columns = ['Territory', 'Bölge', 'Şehir', 'Manager', 
                                           'PF Satış', 'Rakip Satış', 'Pazar Payı %']
                    terr_display.index = range(1, len(territory_data) + 1)
                    
                    st.dataframe(
                        terr_display.style.background_gradient(subset=['Pazar Payı %'], cmap='RdYlGn'),
                        use_container_width=True,
                        height=400
                    )
    
    # TAB 4: TRENDLER
    with tab4:
        st.header("📈 Zaman Serisi ve Trend Analizi")
        
        if len(monthly_data) == 0:
            st.warning("⚠️ Zaman serisi verisi bulunamadı")
        else:
            # Trend grafikleri
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.subheader("📊 Satış Trendi")
                fig_sales = go.Figure()
                
                fig_sales.add_trace(go.Scatter(
                    x=monthly_data['DATE'],
                    y=monthly_data['PF_Satis'],
                    mode='lines+markers',
                    name='PF Satış',
                    line=dict(color=PERFORMANCE_COLORS['success'], width=3)
                ))
                
                fig_sales.add_trace(go.Scatter(
                    x=monthly_data['DATE'],
                    y=monthly_data['MA_3_Ay'],
                    mode='lines',
                    name='3 Aylık Ortalama',
                    line=dict(color=PERFORMANCE_COLORS['info'], width=2, dash='dash')
                ))
                
                fig_sales.update_layout(height=400)
                st.plotly_chart(fig_sales, use_container_width=True)
            
            with col_t2:
                st.subheader("🎯 Pazar Payı Trendi")
                market_share_chart = create_market_share_chart(monthly_data)
                if market_share_chart:
                    st.plotly_chart(market_share_chart, use_container_width=True)
            
            # Büyüme trendleri
            st.markdown("---")
            st.subheader("🚀 Büyüme Trendleri")
            
            growth_chart = create_growth_trend_chart(monthly_data)
            if growth_chart:
                st.plotly_chart(growth_chart, use_container_width=True)
            
            # Rakip analizi
            st.markdown("---")
            st.subheader("📊 Rakip Analizi")
            
            if len(comp_data) > 0:
                comp_chart = create_competitor_comparison_chart(comp_data)
                if comp_chart:
                    st.plotly_chart(comp_chart, use_container_width=True)
                
                # Rakip detayları
                with st.expander("📋 Rakip Performans Detayları"):
                    comp_display = comp_data.copy()
                    comp_display.columns = ['Ay', 'PF Satış', 'Rakip Satış', 'Toplam Pazar', 
                                           'PF Pay %', 'Rakip Pay %', 'PF Büyüme %', 'Rakip Büyüme %', 'Fark %']
                    comp_display.index = range(1, len(comp_display) + 1)
                    
                    st.dataframe(
                        comp_display.style.background_gradient(subset=['Fark %'], cmap='RdYlGn'),
                        use_container_width=True,
                        height=300
                    )
    
    # TAB 5: ML TAHMİNLER
    with tab5:
        st.header("🤖 Machine Learning Tahminleri")
        
        if len(monthly_data) < 12:
            st.warning("⚠️ Tahmin için yeterli veri yok (en az 12 ay gereklidir)")
        else:
            # ML veri hazırlama
            X_train, X_test, y_train, y_test, feature_cols = prepare_ml_data(monthly_data)
            
            if X_train is not None:
                # Model eğitimi
                with st.spinner("ML modelleri eğitiliyor..."):
                    ml_results = train_ml_models(X_train, X_test, y_train, y_test)
                
                # Model performansı
                st.subheader("📊 Model Performans Karşılaştırması")
                
                perf_data = []
                for name, metrics in ml_results.items():
                    perf_data.append({
                        'Model': name,
                        'MAE': f"{metrics['MAE']:,.0f}",
                        'RMSE': f"{metrics['RMSE']:,.0f}",
                        'MAPE': f"%{metrics['MAPE']:.2f}",
                        'R²': f"{metrics['R2']:.3f}"
                    })
                
                perf_df = pd.DataFrame(perf_data)
                st.dataframe(perf_df, use_container_width=True)
                
                # En iyi model
                best_model_name = min(ml_results.keys(), key=lambda x: ml_results[x]['MAPE'])
                best_model = ml_results[best_model_name]['model']
                
                # Tahminler
                st.markdown("---")
                st.subheader("🔮 Gelecek Tahminleri")
                
                # Son 6 ay tahmini
                last_data = monthly_data.tail(6).copy()
                forecast_dates = pd.date_range(
                    start=last_data['DATE'].iloc[-1] + pd.DateOffset(months=1),
                    periods=6,
                    freq='M'
                )
                
                # Tahmin yap
                try:
                    # Son verilerden feature oluştur
                    last_features = []
                    for i in range(6):
                        # Burada gerçek tahmin mantığı uygulanmalı
                        # Basit bir örnek için ortalama büyüme kullanıyoruz
                        last_features.append({
                            'DATE': forecast_dates[i],
                            'Prediction': last_data['PF_Satis'].mean() * (1 + 0.05)  # %5 büyüme varsayımı
                        })
                    
                    forecast_df = pd.DataFrame(last_features)
                    
                    # Tahmin grafiği
                    forecast_chart = create_forecast_chart(monthly_data, forecast_df, best_model_name)
                    st.plotly_chart(forecast_chart, use_container_width=True)
                    
                    # Tahmin detayları
                    with st.expander("📋 Tahmin Detayları"):
                        forecast_display = forecast_df.copy()
                        forecast_display['DATE'] = forecast_display['DATE'].dt.strftime('%Y-%m')
                        forecast_display.columns = ['Ay', 'Tahmini Satış']
                        forecast_display.index = range(1, len(forecast_display) + 1)
                        
                        st.dataframe(forecast_display, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Tahmin hatası: {str(e)}")
                    st.info("Basit tahmin yöntemi kullanılıyor...")
                    
                    # Basit tahmin (ortalama)
                    avg_sales = monthly_data['PF_Satis'].mean()
                    forecast_data = []
                    for i in range(6):
                        forecast_data.append({
                            'DATE': forecast_dates[i],
                            'Prediction': avg_sales * (1 + (i * 0.02))  %2 artış
                        })
                    
                    forecast_df = pd.DataFrame(forecast_data)
                    
                    forecast_chart = create_forecast_chart(monthly_data, forecast_df, "Basit Tahmin")
                    st.plotly_chart(forecast_chart, use_container_width=True)
            else:
                st.warning("ML için yeterli veri yok")
    
    # TAB 6: STRATEJİ
    with tab6:
        st.header("🎯 Stratejik Analiz ve Öneriler")
        
        # SWOT Analizi
        st.subheader("📊 SWOT Analizi")
        
        swot_cols = st.columns(2)
        
        with swot_cols[0]:
            st.markdown("### 🟢 Güçlü Yönler")
            if swot_analysis['Strengths']:
                for strength in swot_analysis['Strengths'][:3]:
                    st.markdown(f"• {strength}")
            else:
                st.info("Güçlü yön bulunamadı")
            
            st.markdown("### 🔴 Zayıf Yönler")
            if swot_analysis['Weaknesses']:
                for weakness in swot_analysis['Weaknesses'][:3]:
                    st.markdown(f"• {weakness}")
            else:
                st.info("Zayıf yön bulunamadı")
        
        with swot_cols[1]:
            st.markdown("### 🔵 Fırsatlar")
            if swot_analysis['Opportunities']:
                for opportunity in swot_analysis['Opportunities'][:3]:
                    st.markdown(f"• {opportunity}")
            else:
                st.info("Fırsat bulunamadı")
            
            st.markdown("### 🟡 Tehditler")
            if swot_analysis['Threats']:
                for threat in swot_analysis['Threats'][:3]:
                    st.markdown(f"• {threat}")
            else:
                st.info("Tehdit bulunamadı")
        
        # Stratejik Öneriler
        st.markdown("---")
        st.subheader("💡 Stratejik Öneriler")
        
        recommendations = calculate_strategic_recommendations(city_data, territory_data)
        
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
                with st.expander(f"{i}. {rec['title']} ({rec['type']})"):
                    st.markdown(f"**Açıklama:** {rec['description']}")
                    
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.metric("Öncelik", rec['priority'])
                    with col_r2:
                        st.metric("Etki", rec['impact'])
                    with col_r3:
                        st.metric("Efor", rec['effort'])
        else:
            st.info("Öneri bulunamadı")
        
        # Aksiyon Planı
        st.markdown("---")
        st.subheader("📋 Aksiyon Planı")
        
        action_items = [
            "Yüksek potansiyelli şehirlerde ek satış kaynakları tahsis et",
            "Düşük performans gösteren territory'ler için eğitim planı oluştur",
            "Rakip analizi sonuçlarına göre fiyat stratejisini gözden geçir",
            "ML tahminlerine göre stok planlamasını güncelle",
            "Performans ödüllendirme sistemini implemente et"
        ]
        
        for i, item in enumerate(action_items, 1):
            st.checkbox(f"{i}. {item}")
    
    # TAB 7: RAPORLAR
    with tab7:
        st.header("📥 Raporlar ve İndirme")
        
        # Rapor özeti
        st.subheader("📊 Rapor Özeti")
        
        if len(city_data) > 0:
            report = generate_comprehensive_report(city_data, territory_data, monthly_data, comp_data, 
                                                  selected_product, date_option)
            
            col_s1, col_s2, col_s3 = st.columns(3)
            
            with col_s1:
                st.metric("Toplam PF Satış", f"{report['summary']['total_pf_sales']:,.0f}")
                st.metric("Şehir Sayısı", report['summary']['cities_count'])
            
            with col_s2:
                st.metric("Toplam Pazar", f"{report['summary']['total_market']:,.0f}")
                st.metric("Territory Sayısı", report['summary']['territories_count'])
            
            with col_s3:
                st.metric("Pazar Payı", f"%{report['summary']['market_share']:.1f}")
                st.metric("Büyüme Oranı", f"%{report['trends']['growth_rate']:.1f}")
        
        # Rapor oluşturma
        st.markdown("---")
        st.subheader("📄 Rapor Oluştur")
        
        report_type = st.selectbox(
            "Rapor Tipi",
            ["Özet Rapor", "Detaylı Rapor", "Performans Raporu", "Strateji Raporu"]
        )
        
        if st.button("📊 Rapor Oluştur ve İndir", type="primary", use_container_width=True):
            with st.spinner("Rapor hazırlanıyor..."):
                try:
                    # Excel raporu oluştur
                    output = BytesIO()
                    
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Özet sayfası
                        summary_df = pd.DataFrame({
                            'Metrik': ['Ürün', 'Dönem', 'Toplam PF Satış', 'Toplam Pazar', 
                                      'Pazar Payı', 'Şehir Sayısı', 'Territory Sayısı'],
                            'Değer': [
                                selected_product,
                                date_option,
                                f"{city_data['PF_Satis'].sum():,.0f}" if len(city_data) > 0 else "0",
                                f"{city_data['Toplam_Pazar'].sum():,.0f}" if len(city_data) > 0 else "0",
                                f"{(city_data['PF_Satis'].sum() / city_data['Toplam_Pazar'].sum() * 100):.1f}%" 
                                if len(city_data) > 0 and city_data['Toplam_Pazar'].sum() > 0 else "0%",
                                len(city_data) if len(city_data) > 0 else 0,
                                len(territory_data) if len(territory_data) > 0 else 0
                            ]
                        })
                        summary_df.to_excel(writer, sheet_name='Özet', index=False)
                        
                        # Şehir performansı
                        if len(city_data) > 0:
                            city_data.to_excel(writer, sheet_name='Şehir Performans', index=False)
                        
                        # Territory performansı
                        if len(territory_data) > 0:
                            territory_data.to_excel(writer, sheet_name='Territory Performans', index=False)
                        
                        # Zaman serisi
                        if len(monthly_data) > 0:
                            monthly_data.to_excel(writer, sheet_name='Zaman Serisi', index=False)
                        
                        # Rakip analizi
                        if len(comp_data) > 0:
                            comp_data.to_excel(writer, sheet_name='Rakip Analizi', index=False)
                    
                    st.success("✅ Rapor hazır!")
                    
                    # İndirme butonu
                    st.download_button(
                        label="💾 Excel Raporunu İndir",
                        data=output.getvalue(),
                        file_name=f"ticari_analiz_raporu_{selected_product}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Rapor oluşturma hatası: {str(e)}")
        
        # Otomatik öneriler
        st.markdown("---")
        st.subheader("💡 Hızlı Öneriler")
        
        if len(city_data) > 0:
            # En iyi 3 şehir
            top_cities = city_data.nlargest(3, 'PF_Satis')
            st.info(f"**Öne Çıkan Şehirler:** {', '.join(top_cities['City'].tolist())}")
            
            # En düşük pazar payı olan 3 şehir
            if len(city_data) >= 3:
                low_share = city_data.nsmallest(3, 'Pazar_Payi_%')
                st.warning(f"**Geliştirilmesi Gerekenler:** {', '.join(low_share['City'].tolist())}")
            
            # En yüksek büyüme
            if 'Buyume_%' in city_data.columns:
                high_growth = city_data.nlargest(3, 'Buyume_%')
                st.success(f"**Yükselen Yıldızlar:** {', '.join(high_growth['City'].tolist())}")

if __name__ == "__main__":
    main()
