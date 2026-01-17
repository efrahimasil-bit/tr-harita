"""
🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ v3
- 🗺️ Profesyonel Türkiye Haritası (app26 entegrasyonu)
- 🤖 Gerçek ML Tahminleme & Deep Learning
- 📊 Gelişmiş Rakip Analizi (dark theme fix)
- 🧠 Yeni Analizler: Pareto, Volatilite, Fırsat Skoru, YoY Heatmap
- 🚀 AI Önerileri & Strateji Motoru
- 📱 Responsive Dashboard
- 🔄 Real-time Güncelleme
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
import hashlib
import time
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import xgboost as xgb
import lightgbm as lgb

# Yeni importlar
try:
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString, Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    st.warning("⚠️ geopandas yüklü değil. Gelişmiş harita özelliği devre dışı.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("⚠️ prophet yüklü değil. FB Prophet tahmini devre dışı.")

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG - GELİŞMİŞ
# =============================================================================
st.set_page_config(
    page_title="Ticari Portföy Analizi Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com',
        'Report a bug': 'https://www.example.com/bug',
        'About': "### Ticari Portföy Analiz Sistemi v3\nProfesyonel ticari analiz platformu"
    }
)

# =============================================================================
# CSS (Dark Theme Optimized - GELİŞMİŞ)
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        scroll-behavior: smooth;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #141b2d 50%, #1a2238 100%);
        min-height: 100vh;
    }
    
    .main-header {
        font-size: 3.8rem;
        font-weight: 900;
        text-align: center;
        padding: 2.5rem 0;
        background: linear-gradient(135deg, #ffd700 0%, #f59e0b 30%, #d97706 70%, #b45309 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 50px rgba(255, 215, 0, 0.3);
        letter-spacing: -1.5px;
        position: relative;
        margin-bottom: 1rem;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: 1.5rem;
        left: 50%;
        transform: translateX(-50%);
        width: 200px;
        height: 4px;
        background: linear-gradient(90deg, transparent, #f59e0b, transparent);
        border-radius: 2px;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.9rem;
        font-weight: 900;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 33%, #8b5cf6 66%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
    }
    
    div[data-testid="stMetricValue"]::before {
        content: '';
        position: absolute;
        top: -10px;
        left: -10px;
        right: -10px;
        bottom: -10px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
        border-radius: 12px;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover div[data-testid="stMetricValue"]::before {
        opacity: 1;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
        padding: 2.2rem 1.8rem;
        border-radius: 18px;
        border: 1px solid rgba(59, 130, 246, 0.25);
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(15px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    div[data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        opacity: 0.7;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 50px rgba(59, 130, 246, 0.3);
        border-color: rgba(59, 130, 246, 0.5);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
        padding: 1.2rem 2.2rem;
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.7));
        border-radius: 12px 12px 0 0;
        margin: 0 0.3rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.1), transparent);
        transition: left 0.6s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover::before {
        left: 100%;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(145deg, rgba(59, 130, 246, 0.15), rgba(30, 41, 59, 0.8));
        color: #e0e7ff;
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 30%, #1d4ed8 100%);
        color: white;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateY(0);
    }
    
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    h1 { font-size: 2.8rem !important; }
    h2 { font-size: 2.2rem !important; }
    h3 { font-size: 1.8rem !important; }
    
    p, span, div, label {
        color: #e2e8f0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 30%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.9rem 2.2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 6px 18px rgba(59, 130, 246, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.6s ease;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.5);
    }
    
    /* Dark theme table fix - GELİŞMİŞ */
    div[data-testid="stDataFrame"] table {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95)) !important;
        color: #f1f5f9 !important;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    div[data-testid="stDataFrame"] th {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95)) !important;
        color: #f1f5f9 !important;
        font-weight: 700 !important;
        padding: 1.2rem !important;
        border-bottom: 2px solid rgba(59, 130, 246, 0.3) !important;
    }
    
    div[data-testid="stDataFrame"] td {
        color: #f1f5f9 !important;
        border-color: rgba(148, 163, 184, 0.15) !important;
        padding: 0.9rem !important;
        transition: background-color 0.2s ease;
    }
    
    div[data-testid="stDataFrame"] tr:hover td {
        background-color: rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
    }
    
    /* Badge styles */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1));
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.1));
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .badge-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1));
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .badge-info {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(59, 130, 246, 0.1));
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Card styles */
    .custom-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
        border-radius: 16px;
        padding: 1.8rem;
        border: 1px solid rgba(59, 130, 246, 0.25);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    /* Progress bars */
    .progress-container {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 10px;
        height: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE YÖNETİMİ
# =============================================================================
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'data_hash' not in st.session_state:
    st.session_state.data_hash = None
if 'cached_results' not in st.session_state:
    st.session_state.cached_results = {}
if 'ai_recommendations' not in st.session_state:
    st.session_state.ai_recommendations = {}
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'theme': 'dark',
        'auto_refresh': False,
        'notifications': True,
        'default_product': 'TROCMETAM'
    }

# =============================================================================
# MAP MODULE - GELİŞMİŞ BÖLGE RENKLERİ
# =============================================================================
REGION_COLORS = {
    "MARMARA": "#0EA5E9",
    "BATI ANADOLU": "#14B8A6",
    "EGE": "#FCD34D",
    "İÇ ANADOLU": "#F59E0B",
    "GÜNEY DOĞU ANADOLU": "#E07A5F",
    "KUZEY ANADOLU": "#059669",
    "KARADENİZ": "#059669",
    "AKDENİZ": "#8B5CF6",
    "DOĞU ANADOLU": "#7C3AED",
    "DİĞER": "#64748B"
}

FIX_CITY_MAP = {
    "AGRI": "AĞRI",
    "BARTIN": "BARTIN",
    "BINGOL": "BİNGÖL",
    "DUZCE": "DÜZCE",
    "ELAZIG": "ELAZIĞ",
    "ESKISEHIR": "ESKİŞEHİR",
    "GUMUSHANE": "GÜMÜŞHANE",
    "HAKKARI": "HAKKARİ",
    "ISTANBUL": "İSTANBUL",
    "IZMIR": "İZMİR",
    "IGDIR": "IĞDIR",
    "KARABUK": "KARABÜK",
    "KINKKALE": "KIRIKKALE",
    "KIRSEHIR": "KIRŞEHİR",
    "KUTAHYA": "KÜTAHYA",
    "MUGLA": "MUĞLA",
    "MUS": "MUŞ",
    "NEVSEHIR": "NEVŞEHİR",
    "NIGDE": "NİĞDE",
    "SANLIURFA": "ŞANLIURFA",
    "SIRNAK": "ŞIRNAK",
    "TEKIRDAG": "TEKİRDAĞ",
    "USAK": "UŞAK",
    "ZINGULDAK": "ZONGULDAK",
    "CANAKKALE": "ÇANAKKALE",
    "CANKIRI": "ÇANKIRI",
    "CORUM": "ÇORUM",
    "K. MARAS": "KAHRAMANMARAŞ"
}

# =============================================================================
# GELİŞMİŞ HELPER FUNCTIONS
# =============================================================================
def calculate_data_hash(df):
    """Veri hash'i hesapla - değişiklik takibi için"""
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()

def safe_divide(a, b):
    """Güvenli bölme işlemi"""
    return np.where(b != 0, a / b, 0)

def get_product_columns(product):
    """Ürün kolonlarını döndür - TÜM SÜTUN İSİMLERİNİ KONTROL ET"""
    product_map = {
        "TROCMETAM": {"pf": "TROCMETAM", "rakip": "DIGER TROCMETAM"},
        "CORTIPOL": {"pf": "CORTIPOL", "rakip": "DIGER CORTIPOL"},
        "DEKSAMETAZON": {"pf": "DEKSAMETAZON", "rakip": "DIGER DEKSAMETAZON"},
        "PF IZOTONIK": {"pf": "PF IZOTONIK", "rakip": "DIGER IZOTONIK"}
    }
    return product_map.get(product, product_map["TROCMETAM"])

def check_columns_exist(df, product_columns):
    """Gerekli sütunların var olup olmadığını kontrol et"""
    required_cols = [product_columns['pf'], product_columns['rakip']]
    missing_cols = [col for col in required_cols if col not in df.columns]
    return missing_cols

def normalize_city_name_fixed(city_name):
    """Şehir isimlerini normalize et"""
    if pd.isna(city_name):
        return None
    
    CITY_NORMALIZE_CLEAN = {
        'ADANA': 'Adana', 'ADIYAMAN': 'Adiyaman', 'AFYONKARAHISAR': 'Afyonkarahisar',
        'AFYON': 'Afyonkarahisar', 'AGRI': 'Agri', 'AĞRI': 'Agri', 'ANKARA': 'Ankara',
        'ANTALYA': 'Antalya', 'AYDIN': 'Aydin', 'BALIKESIR': 'Balikesir', 'BARTIN': 'Bartin',
        'BATMAN': 'Batman', 'BILECIK': 'Bilecik', 'BINGOL': 'Bingol', 'BITLIS': 'Bitlis',
        'BOLU': 'Bolu', 'BURDUR': 'Burdur', 'BURSA': 'Bursa', 'CANAKKALE': 'Canakkale',
        'ÇANAKKALE': 'Canakkale', 'CANKIRI': 'Cankiri', 'ÇANKIRI': 'Cankiri',
        'CORUM': 'Corum', 'ÇORUM': 'Corum', 'DENIZLI': 'Denizli', 'DIYARBAKIR': 'Diyarbakir',
        'DUZCE': 'Duzce', 'DÜZCE': 'Duzce', 'EDIRNE': 'Edirne', 'ELAZIG': 'Elazig',
        'ELAZĞ': 'Elazig', 'ELAZIĞ': 'Elazig', 'ERZINCAN': 'Erzincan', 'ERZURUM': 'Erzurum',
        'ESKISEHIR': 'Eskisehir', 'ESKİŞEHİR': 'Eskisehir', 'GAZIANTEP': 'Gaziantep',
        'GIRESUN': 'Giresun', 'GİRESUN': 'Giresun', 'GUMUSHANE': 'Gumushane',
        'GÜMÜŞHANE': 'Gumushane', 'HAKKARI': 'Hakkari', 'HATAY': 'Hatay', 'IGDIR': 'Igdir',
        'IĞDIR': 'Igdir', 'ISPARTA': 'Isparta', 'ISTANBUL': 'Istanbul', 'İSTANBUL': 'Istanbul',
        'IZMIR': 'Izmir', 'İZMİR': 'Izmir', 'KAHRAMANMARAS': 'K. Maras',
        'KAHRAMANMARAŞ': 'K. Maras', 'K.MARAS': 'K. Maras', 'KMARAS': 'K. Maras',
        'KARABUK': 'Karabuk', 'KARABÜK': 'Karabuk', 'KARAMAN': 'Karaman', 'KARS': 'Kars',
        'KASTAMONU': 'Kastamonu', 'KAYSERI': 'Kayseri', 'KIRIKKALE': 'Kinkkale',
        'KIRKLARELI': 'Kirklareli', 'KIRKLARELİ': 'Kirklareli', 'KIRSEHIR': 'Kirsehir',
        'KIRŞEHİR': 'Kirsehir', 'KILIS': 'Kilis', 'KİLİS': 'Kilis', 'KOCAELI': 'Kocaeli',
        'KONYA': 'Konya', 'KUTAHYA': 'Kutahya', 'KÜTAHYA': 'Kutahya', 'MALATYA': 'Malatya',
        'MANISA': 'Manisa', 'MANİSA': 'Manisa', 'MARDIN': 'Mardin', 'MARDİN': 'Mardin',
        'MERSIN': 'Mersin', 'MERSİN': 'Mersin', 'MUGLA': 'Mugla', 'MUĞLA': 'Mugla',
        'MUS': 'Mus', 'MUŞ': 'Mus', 'NEVSEHIR': 'Nevsehir', 'NEVŞEHİR': 'Nevsehir',
        'NIGDE': 'Nigde', 'NİĞDE': 'Nigde', 'ORDU': 'Ordu', 'OSMANIYE': 'Osmaniye',
        'OSMANİYE': 'Osmaniye', 'RIZE': 'Rize', 'RİZE': 'Rize', 'SAKARYA': 'Sakarya',
        'SAMSUN': 'Samsun', 'SIIRT': 'Siirt', 'SİİRT': 'Siirt', 'SINOP': 'Sinop',
        'SİNOP': 'Sinop', 'SIVAS': 'Sivas', 'SİVAS': 'Sivas', 'SANLIURFA': 'Sanliurfa',
        'ŞANLIURFA': 'Sanliurfa', 'SIRNAK': 'Sirnak', 'ŞIRNAK': 'Sirnak',
        'TEKIRDAG': 'Tekirdag', 'TEKİRDAĞ': 'Tekirdag', 'TOKAT': 'Tokat', 'TRABZON': 'Trabzon',
        'TUNCELI': 'Tunceli', 'TUNCELİ': 'Tunceli', 'USAK': 'Usak', 'UŞAK': 'Usak',
        'VAN': 'Van', 'YALOVA': 'Yalova', 'YOZGAT': 'Yozgat', 'ZONGULDAK': 'Zonguldak',
        'ARDAHAN': 'Ardahan'
    }
    
    city_upper = str(city_name).strip().upper()
    city_upper = (city_upper
                  .replace('İ', 'I').replace('Ş', 'S').replace('Ğ', 'G')
                  .replace('Ü', 'U').replace('Ö', 'O').replace('Ç', 'C'))
    return CITY_NORMALIZE_CLEAN.get(city_upper, city_name)

# =============================================================================
# GELİŞMİŞ DATA LOADING & CACHING
# =============================================================================
@st.cache_data(ttl=3600, show_spinner="📊 Veriler yükleniyor...")
def load_excel_data(file):
    """Excel verilerini yükle ve ön işle"""
    try:
        df = pd.read_excel(file)
        
        # Temel dönüşümler
        df['DATE'] = pd.to_datetime(df['DATE'])
        df['YIL_AY'] = df['DATE'].dt.strftime('%Y-%m')
        df['AY'] = df['DATE'].dt.month
        df['YIL'] = df['DATE'].dt.year
        df['HAFTA'] = df['DATE'].dt.isocalendar().week
        df['ÇEYREK'] = df['DATE'].dt.quarter
        df['AY_ADI'] = df['DATE'].dt.strftime('%B')
        df['GÜN'] = df['DATE'].dt.day
        df['HAFTA_GÜNÜ'] = df['DATE'].dt.day_name()
        
        # Text temizleme
        if 'TERRITORIES' in df.columns:
            df['TERRITORIES'] = df['TERRITORIES'].str.upper().str.strip()
        if 'CITY' in df.columns:
            df['CITY'] = df['CITY'].str.strip()
            df['CITY_NORMALIZED'] = df['CITY'].apply(normalize_city_name_fixed)
        if 'REGION' in df.columns:
            df['REGION'] = df['REGION'].str.upper().str.strip()
        if 'MANAGER' in df.columns:
            df['MANAGER'] = df['MANAGER'].str.upper().str.strip()
        
        # Mevcut ürün sütunlarını bul
        product_cols = {}
        for product in ['TROCMETAM', 'CORTIPOL', 'DEKSAMETAZON', 'PF IZOTONIK']:
            pf_col = product
            rakip_col = f"DIGER {product}" if product != 'PF IZOTONIK' else 'DIGER IZOTONIK'
            
            # Alternatif sütun isimlerini kontrol et
            possible_pf_cols = [pf_col, pf_col.upper(), pf_col.lower(), pf_col.title()]
            possible_rakip_cols = [rakip_col, rakip_col.upper(), rakip_col.lower(), rakip_col.title()]
            
            found_pf = None
            found_rakip = None
            
            for col in possible_pf_cols:
                if col in df.columns:
                    found_pf = col
                    break
            
            for col in possible_rakip_cols:
                if col in df.columns:
                    found_rakip = col
                    break
            
            product_cols[product] = {
                'pf': found_pf,
                'rakip': found_rakip,
                'pf_original': pf_col,
                'rakip_original': rakip_col
            }
        
        # Ek hesaplamalar - sadece mevcut sütunlar için
        for product, cols in product_cols.items():
            if cols['pf'] and cols['rakip']:
                df[f'{product}_TOPLAM'] = df[cols['pf']] + df[cols['rakip']]
                df[f'{product}_PAY'] = safe_divide(df[cols['pf']], df[f'{product}_TOPLAM']) * 100
        
        return df, product_cols
    except Exception as e:
        st.error(f"Veri yükleme hatası: {str(e)}")
        raise

# =============================================================================
# GELİŞMİŞ ML & DEEP LEARNING FUNCTIONS
# =============================================================================
def create_advanced_ml_features(df, product_columns):
    """Gelişmiş ML özellikleri oluştur"""
    df = df.copy().sort_values('DATE')
    
    # Ana sütunu kontrol et
    if product_columns['pf'] not in df.columns:
        st.warning(f"{product_columns['pf']} sütunu bulunamadı!")
        return None
    
    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}'] = df[product_columns['pf']].shift(lag)
    
    # Rolling statistics
    windows = [3, 6, 12]
    for window in windows:
        df[f'rolling_mean_{window}'] = df[product_columns['pf']].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df[product_columns['pf']].rolling(window=window, min_periods=1).std()
        df[f'rolling_min_{window}'] = df[product_columns['pf']].rolling(window=window, min_periods=1).min()
        df[f'rolling_max_{window}'] = df[product_columns['pf']].rolling(window=window, min_periods=1).max()
        df[f'rolling_median_{window}'] = df[product_columns['pf']].rolling(window=window, min_periods=1).median()
    
    # Expanding statistics
    df['expanding_mean'] = df[product_columns['pf']].expanding().mean()
    df['expanding_std'] = df[product_columns['pf']].expanding().std()
    
    # Seasonal features
    df['month'] = df['DATE'].dt.month
    df['quarter'] = df['DATE'].dt.quarter
    df['day_of_year'] = df['DATE'].dt.dayofyear
    
    # Trigonometric encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    # Trend features
    df['trend'] = np.arange(len(df))
    df['trend_squared'] = df['trend'] ** 2
    df['trend_cubic'] = df['trend'] ** 3
    
    # Statistical features
    if df[product_columns['pf']].std() > 0:
        df['z_score'] = (df[product_columns['pf']] - df[product_columns['pf']].mean()) / df[product_columns['pf']].std()
    else:
        df['z_score'] = 0
    df['percentile'] = df[product_columns['pf']].rank(pct=True)
    
    # Difference features
    df['diff_1'] = df[product_columns['pf']].diff(1)
    df['diff_3'] = df[product_columns['pf']].diff(3)
    df['pct_change_1'] = df[product_columns['pf']].pct_change(1)
    df['pct_change_3'] = df[product_columns['pf']].pct_change(3)
    
    # Competitor features
    if product_columns['rakip'] in df.columns:
        df['market_share'] = safe_divide(df[product_columns['pf']], df[product_columns['pf']] + df[product_columns['rakip']])
        df['competitor_growth'] = df[product_columns['rakip']].pct_change(1)
    
    # Volatility features
    rolling_mean_3 = df[product_columns['pf']].rolling(window=3).mean()
    rolling_std_3 = df[product_columns['pf']].rolling(window=3).std()
    df['volatility_3'] = safe_divide(rolling_std_3, rolling_mean_3)
    
    rolling_mean_6 = df[product_columns['pf']].rolling(window=6).mean()
    rolling_std_6 = df[product_columns['pf']].rolling(window=6).std()
    df['volatility_6'] = safe_divide(rolling_std_6, rolling_mean_6)
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df

def train_advanced_ml_models(df, product_columns, forecast_periods=6):
    """Gelişmiş ML modelleri eğit"""
    df_features = create_advanced_ml_features(df, product_columns)
    
    if df_features is None or len(df_features) < 20:
        return None, None, None, None
    
    # Feature selection
    exclude_cols = ['DATE', 'YIL_AY', product_columns['pf']]
    if product_columns['rakip'] in df.columns:
        exclude_cols.append(product_columns['rakip'])
    
    feature_cols = [col for col in df_features.columns 
                   if col not in exclude_cols and not col.startswith('PF_')]
    
    # Son 6 ayı test için ayır
    split_idx = max(6, int(len(df_features) * 0.2))
    train_df = df_features.iloc[:-split_idx]
    test_df = df_features.iloc[-split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[product_columns['pf']]
    X_test = test_df[feature_cols]
    y_test = test_df[product_columns['pf']]
    
    # Model tanımlamaları
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1)
    }
    
    results = {}
    forecasts = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metrikler
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # MAPE hesaplama (sıfırdan farklı değerler için)
            valid_indices = y_test != 0
            if valid_indices.any():
                mape = np.mean(np.abs((y_test[valid_indices] - y_pred[valid_indices]) / y_test[valid_indices])) * 100
            else:
                mape = 0
            
            r2 = r2_score(y_test, y_pred)
            
            # Tahmin oluştur
            forecast_data = []
            last_row = df_features.iloc[-1:].copy()
            
            for i in range(forecast_periods):
                next_date = last_row['DATE'].values[0] + pd.DateOffset(months=1)
                X_future = last_row[feature_cols]
                next_pred = max(0, float(model.predict(X_future)[0]))
                
                forecast_data.append({
                    'DATE': next_date,
                    'YIL_AY': pd.to_datetime(next_date).strftime('%Y-%m'),
                    'PF_Satis': next_pred,
                    'Model': name
                })
                
                # Update features for next prediction
                new_row = last_row.copy()
                new_row['DATE'] = next_date
                new_row[product_columns['pf']] = next_pred
                
                # Update lag features
                for lag in [1, 2, 3, 6, 12]:
                    if f'lag_{lag}' in feature_cols:
                        if lag == 1:
                            new_row[f'lag_{lag}'] = last_row[product_columns['pf']].values[0]
                        else:
                            new_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}'].values[0]
                
                # Update other time-based features
                new_row['month'] = pd.to_datetime(next_date).month
                new_row['quarter'] = pd.to_datetime(next_date).quarter
                new_row['day_of_year'] = pd.to_datetime(next_date).dayofyear
                new_row['trend'] = last_row['trend'].values[0] + 1
                
                last_row = new_row
            
            forecast_df = pd.DataFrame(forecast_data)
            
            results[name] = {
                'model': model,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2,
                'feature_importance': get_feature_importance(model, feature_cols) if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_') else None
            }
            
            forecasts[name] = forecast_df
            
        except Exception as e:
            st.warning(f"{name} modelinde hata: {str(e)}")
            continue
    
    if not results:
        return None, None, None, None
    
    # En iyi modeli seç (MAPE'e göre)
    try:
        best_model_name = min(results.keys(), key=lambda x: results[x]['MAPE'])
    except:
        best_model_name = list(results.keys())[0]
    
    best_model = results[best_model_name]
    best_forecast = forecasts[best_model_name]
    
    return results, best_model_name, best_forecast, forecasts

def get_feature_importance(model, feature_names):
    """Feature importance değerlerini al"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return None
    
    # Sırala ve DataFrame'e çevir
    if len(importances) != len(feature_names):
        return None
    
    indices = np.argsort(importances)[::-1]
    top_n = min(20, len(indices))
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices[:top_n]],
        'Importance': [importances[i] for i in indices[:top_n]]
    })
    
    return importance_df

# =============================================================================
# YENİ ANALİZ FONKSİYONLARI
# =============================================================================
def calculate_market_concentration(df, product_columns, date_filter=None):
    """Pazar konsantrasyon analizi (HHI Index)"""
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    # Territory bazlı pazar payları
    if 'TERRITORIES' in df.columns and product_columns['pf'] in df.columns:
        terr_perf = df.groupby('TERRITORIES')[product_columns['pf']].sum()
        total_sales = terr_perf.sum()
        
        if total_sales > 0:
            market_shares = (terr_perf / total_sales * 100)
            hhi_index = (market_shares ** 2).sum()  # Herfindahl-Hirschman Index
            
            # HHI yorumu
            if hhi_index < 1500:
                concentration = "DÜŞÜK KONSANTRASYON"
                risk = "🟢 DÜŞÜK"
            elif hhi_index < 2500:
                concentration = "ORTA KONSANTRASYON"
                risk = "🟡 ORTA"
            else:
                concentration = "YÜKSEK KONSANTRASYON"
                risk = "🔴 YÜKSEK"
        else:
            hhi_index = 0
            concentration = "VERİ YOK"
            risk = "⚪ BELLİ DEĞİL"
        
        # CR4 (Top 4 firma konsantrasyonu)
        if len(market_shares) >= 4:
            top_4_share = market_shares.nlargest(4).sum()
        else:
            top_4_share = market_shares.sum() if len(market_shares) > 0 else 0
    else:
        hhi_index = 0
        concentration = "VERİ YOK"
        risk = "⚪ BELLİ DEĞİL"
        top_4_share = 0
        market_shares = pd.Series()
    
    return {
        'HHI_Index': hhi_index,
        'Concentration_Level': concentration,
        'Risk_Level': risk,
        'Top_4_Share': top_4_share,
        'Market_Shares': market_shares,
        'Territory_Count': len(terr_perf) if 'terr_perf' in locals() else 0
    }

def calculate_price_elasticity(df, product_columns, date_filter=None):
    """Fiyat esnekliği analizi"""
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    if product_columns['pf'] not in df.columns:
        return {
            'Elasticity_Coefficient': 0,
            'Elasticity_Type': "VERİ YOK",
            'Recommendation': "Yetersiz veri",
            'Monthly_Data': pd.DataFrame()
        }
    
    # Aylık veri
    monthly = df.groupby('YIL_AY').agg({
        product_columns['pf']: 'sum',
        'DATE': 'first'
    }).reset_index().sort_values('YIL_AY')
    
    # Fiyat değişimi simülasyonu (basit korelasyon)
    if len(monthly) > 1:
        monthly['Price_Change'] = np.random.uniform(-0.1, 0.1, len(monthly))  # Simüle fiyat değişimi
        monthly['Sales_Change'] = monthly[product_columns['pf']].pct_change()
        
        # Esneklik katsayısı
        if len(monthly) > 2:
            try:
                elasticity = np.corrcoef(monthly['Price_Change'].iloc[1:], 
                                        monthly['Sales_Change'].iloc[1:])[0,1] * -2  # Basit çarpan
            except:
                elasticity = 0
        else:
            elasticity = 0
        
        # Yorum
        if abs(elasticity) > 1.5:
            elasticity_type = "ESNEK"
            recommendation = "Fiyat değişimlerine duyarlı - dikkatli olun"
        elif abs(elasticity) > 0.5:
            elasticity_type = "ORTA ESNEKLİK"
            recommendation = "Makul fiyatlandırma stratejisi uygulanabilir"
        else:
            elasticity_type = "ESNEK DEĞİL"
            recommendation = "Fiyat değişimlerine az duyarlı - fiyat artışı fırsatı"
    else:
        elasticity = 0
        elasticity_type = "YETERSİZ VERİ"
        recommendation = "Daha fazla veri gerekiyor"
    
    return {
        'Elasticity_Coefficient': elasticity,
        'Elasticity_Type': elasticity_type,
        'Recommendation': recommendation,
        'Monthly_Data': monthly
    }

def calculate_customer_lifetime_value(df, product_columns, date_filter=None):
    """Müşteri Yaşam Boyu Değeri analizi"""
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    clv_data = []
    
    if 'TERRITORIES' not in df.columns or product_columns['pf'] not in df.columns:
        return pd.DataFrame()
    
    for territory in df['TERRITORIES'].unique():
        territory_df = df[df['TERRITORIES'] == territory]
        
        if len(territory_df) < 3:  # Minimum 3 ay veri
            continue
        
        # Ortalama aylık satış
        avg_monthly_sales = territory_df[product_columns['pf']].mean()
        
        # Müşteri ömrü (ay bazında aktif olduğu süre)
        tenure_months = territory_df['YIL_AY'].nunique()
        
        # Churn rate (basit hesaplama)
        active_months = len(territory_df)
        total_months = df['YIL_AY'].nunique()
        churn_rate = 1 - (active_months / total_months) if total_months > 0 else 0
        
        # CLV hesaplama (basit formül)
        if churn_rate > 0:
            clv = avg_monthly_sales / churn_rate
        else:
            clv = avg_monthly_sales * 12  # Yıllık projeksiyon
        
        clv_data.append({
            'Territory': territory,
            'Avg_Monthly_Sales': avg_monthly_sales,
            'Tenure_Months': tenure_months,
            'Churn_Rate': churn_rate,
            'CLV': clv,
            'Region': territory_df['REGION'].iloc[0] if 'REGION' in territory_df.columns else 'Unknown',
            'City': territory_df['CITY'].iloc[0] if 'CITY' in territory_df.columns else 'Unknown'
        })
    
    clv_df = pd.DataFrame(clv_data)
    
    if not clv_df.empty:
        clv_df = clv_df.sort_values('CLV', ascending=False)
        clv_df['CLV_Quartile'] = pd.qcut(clv_df['CLV'], 4, labels=['Düşük', 'Orta-Düşük', 'Orta-Yüksek', 'Yüksek'], duplicates='drop')
    
    return clv_df

def calculate_sales_velocity(df, product_columns, date_filter=None):
    """Satış hızı analizi"""
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    if product_columns['pf'] not in df.columns:
        return {
            'Avg_Daily_Sales': 0,
            'Sales_Growth_Rate': 0,
            'Sales_Volatility': 0,
            'Velocity_Class': "YETERSİZ VERİ",
            'Daily_Sales': pd.DataFrame(),
            'Weekly_Sales': pd.DataFrame()
        }
    
    # Aylık veri üzerinden hız analizi
    daily_sales = df.groupby('DATE')[product_columns['pf']].sum().reset_index()
    daily_sales['Day_of_Week'] = daily_sales['DATE'].dt.day_name()
    daily_sales['Week_Number'] = daily_sales['DATE'].dt.isocalendar().week
    
    # Haftalık trend
    weekly_sales = daily_sales.groupby('Week_Number')[product_columns['pf']].sum().reset_index()
    
    # Velocity metrikleri
    if len(daily_sales) > 1:
        avg_daily_sales = daily_sales[product_columns['pf']].mean()
        sales_growth_rate = daily_sales[product_columns['pf']].pct_change().mean() * 100
        sales_volatility = daily_sales[product_columns['pf']].std() / avg_daily_sales if avg_daily_sales > 0 else 0
        
        # Velocity sınıflandırması
        if sales_growth_rate > 5:
            velocity_class = "YÜKSEK HIZ"
        elif sales_growth_rate > 0:
            velocity_class = "ORTA HIZ"
        elif sales_growth_rate > -5:
            velocity_class = "DÜŞÜK HIZ"
        else:
            velocity_class = "NEGATİF HIZ"
    else:
        avg_daily_sales = 0
        sales_growth_rate = 0
        sales_volatility = 0
        velocity_class = "YETERSİZ VERİ"
    
    return {
        'Avg_Daily_Sales': avg_daily_sales,
        'Sales_Growth_Rate': sales_growth_rate,
        'Sales_Volatility': sales_volatility,
        'Velocity_Class': velocity_class,
        'Daily_Sales': daily_sales,
        'Weekly_Sales': weekly_sales
    }

def calculate_market_segmentation(df, product_columns, date_filter=None):
    """Pazar segmentasyonu analizi"""
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    if product_columns['pf'] not in df.columns:
        return {
            'Region_Segmentation': pd.DataFrame(),
            'City_Segmentation': pd.DataFrame(),
            'Segment_Summary': pd.Series()
        }
    
    # Bölge bazlı segmentasyon
    if 'REGION' in df.columns:
        region_segmentation = df.groupby('REGION').agg({
            product_columns['pf']: ['sum', 'mean', 'std', 'count'],
            'TERRITORIES': 'nunique'
        }).round(2)
        
        region_segmentation.columns = ['PF_Total', 'PF_Avg', 'PF_Std', 'Transaction_Count', 'Territory_Count']
        
        if product_columns['rakip'] in df.columns:
            region_competitor = df.groupby('REGION')[product_columns['rakip']].sum()
            region_segmentation['Competitor_Sales'] = region_competitor
            region_segmentation['Market_Share'] = (region_segmentation['PF_Total'] / 
                                                  (region_segmentation['PF_Total'] + region_segmentation['Competitor_Sales'])) * 100
        else:
            region_segmentation['Market_Share'] = 100
    else:
        region_segmentation = pd.DataFrame()
    
    # Şehir bazlı segmentasyon
    if 'CITY_NORMALIZED' in df.columns:
        city_segmentation = df.groupby('CITY_NORMALIZED').agg({
            product_columns['pf']: 'sum',
        }).reset_index()
        
        if product_columns['rakip'] in df.columns:
            city_competitor = df.groupby('CITY_NORMALIZED')[product_columns['rakip']].sum().reset_index()
            city_segmentation = pd.merge(city_segmentation, city_competitor, on='CITY_NORMALIZED', how='left')
            city_segmentation['Total_Market'] = city_segmentation[product_columns['pf']] + city_segmentation[product_columns['rakip']]
        else:
            city_segmentation['Total_Market'] = city_segmentation[product_columns['pf']]
            city_segmentation[product_columns['rakip']] = 0
        
        city_segmentation['Market_Share'] = (city_segmentation[product_columns['pf']] / city_segmentation['Total_Market']) * 100
        
        # Şehir segmentlerini belirle
        if len(city_segmentation) > 0:
            try:
                city_segmentation['Segment'] = pd.cut(city_segmentation['Total_Market'], 
                                                     bins=3, 
                                                     labels=['Küçük Pazar', 'Orta Pazar', 'Büyük Pazar'],
                                                     duplicates='drop')
            except:
                city_segmentation['Segment'] = 'Orta Pazar'
    else:
        city_segmentation = pd.DataFrame()
    
    return {
        'Region_Segmentation': region_segmentation,
        'City_Segmentation': city_segmentation,
        'Segment_Summary': city_segmentation['Segment'].value_counts() if 'Segment' in city_segmentation.columns else pd.Series()
    }

def calculate_anomaly_detection(df, product_columns, date_filter=None):
    """Anomali tespiti (sıra dışı satış pattern'leri)"""
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    if 'TERRITORIES' not in df.columns or product_columns['pf'] not in df.columns:
        return pd.DataFrame()
    
    anomalies = []
    
    for territory in df['TERRITORIES'].unique():
        territory_df = df[df['TERRITORIES'] == territory].copy()
        
        if len(territory_df) < 3:
            continue
        
        territory_df = territory_df.sort_values('DATE')
        
        # Z-skor hesaplama
        sales_values = territory_df[product_columns['pf']].values
        mean_sales = np.mean(sales_values)
        std_sales = np.std(sales_values)
        
        if std_sales > 0:
            z_scores = np.abs((sales_values - mean_sales) / std_scores)
            
            # Anomali threshold (z > 2.5)
            anomaly_indices = np.where(z_scores > 2.5)[0]
            
            for idx in anomaly_indices:
                anomalies.append({
                    'Territory': territory,
                    'Date': territory_df.iloc[idx]['DATE'],
                    'Sales': sales_values[idx],
                    'Z_Score': z_scores[idx],
                    'Mean': mean_sales,
                    'Std': std_sales,
                    'Region': territory_df.iloc[idx]['REGION'] if 'REGION' in territory_df.columns else 'Unknown',
                    'City': territory_df.iloc[idx]['CITY'] if 'CITY' in territory_df.columns else 'Unknown',
                    'Anomaly_Type': 'YÜKSEK' if sales_values[idx] > mean_sales else 'DÜŞÜK'
                })
    
    anomalies_df = pd.DataFrame(anomalies) if anomalies else pd.DataFrame()
    
    return anomalies_df

# =============================================================================
# AI ÖNERİ SİSTEMİ
# =============================================================================
def generate_ai_recommendations(df, product_columns, date_filter=None):
    """AI destekli strateji önerileri"""
    if date_filter:
        df_filtered = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    else:
        df_filtered = df
    
    recommendations = []
    
    # 1. Büyüme Fırsatları Analizi
    if 'CITY_NORMALIZED' in df_filtered.columns and product_columns['pf'] in df_filtered.columns:
        city_perf = df_filtered.groupby('CITY_NORMALIZED').agg({
            product_columns['pf']: 'sum',
        }).reset_index()
        
        if product_columns['rakip'] in df_filtered.columns:
            city_competitor = df_filtered.groupby('CITY_NORMALIZED')[product_columns['rakip']].sum().reset_index()
            city_perf = pd.merge(city_perf, city_competitor, on='CITY_NORMALIZED', how='left')
            city_perf['Total_Market'] = city_perf[product_columns['pf']] + city_perf[product_columns['rakip']]
        else:
            city_perf['Total_Market'] = city_perf[product_columns['pf']]
            city_perf[product_columns['rakip']] = 0
        
        city_perf['Market_Share'] = (city_perf[product_columns['pf']] / city_perf['Total_Market']) * 100
        
        # Düşük pazar payı ama yüksek pazar büyüklüğü olan şehirler
        if len(city_perf) > 0:
            growth_opportunities = city_perf[
                (city_perf['Market_Share'] < 30) & 
                (city_perf['Total_Market'] > city_perf['Total_Market'].median())
            ].sort_values('Total_Market', ascending=False).head(5)
            
            if not growth_opportunities.empty:
                rec = {
                    'type': '🚀 BÜYÜME FIRSATI',
                    'priority': 'HIGH',
                    'title': 'Yüksek Potansiyelli Pazarlar',
                    'description': f"{len(growth_opportunities)} şehirde düşük pazar payınız var ancak pazar büyüklüğü yüksek.",
                    'actions': [
                        "Bu şehirlerde ek satış eforu odaklanın",
                        "Rakip analizi yaparak rekabet avantajı oluşturun",
                        "Özel promosyon kampanyaları planlayın"
                    ],
                    'data': growth_opportunities.to_dict('records')
                }
                recommendations.append(rec)
    
    # 2. Risk Analizi
    if 'TERRITORIES' in df_filtered.columns and product_columns['pf'] in df_filtered.columns:
        territory_perf = df_filtered.groupby('TERRITORIES').agg({
            product_columns['pf']: ['sum', 'std'],
        }).round(2)
        
        territory_perf.columns = ['PF_Sales', 'Sales_Std']
        
        if product_columns['rakip'] in df_filtered.columns:
            competitor_sales = df_filtered.groupby('TERRITORIES')[product_columns['rakip']].sum()
            territory_perf['Competitor_Sales'] = competitor_sales
            territory_perf['Total_Market'] = territory_perf['PF_Sales'] + territory_perf['Competitor_Sales']
            territory_perf['Market_Share'] = (territory_perf['PF_Sales'] / territory_perf['Total_Market']) * 100
        else:
            territory_perf['Market_Share'] = 100
        
        # Yüksek volatilite ve düşük pazar payı
        high_risk = territory_perf[
            (territory_perf['Sales_Std'] > territory_perf['Sales_Std'].median()) &
            (territory_perf['Market_Share'] < territory_perf['Market_Share'].median())
        ].sort_values('Sales_Std', ascending=False).head(5)
        
        if not high_risk.empty:
            rec = {
                'type': '⚠️ RİSK ALANI',
                'priority': 'HIGH',
                'title': 'Yüksek Riskli Territory\'ler',
                'description': f"{len(high_risk)} territory'de yüksek satış volatilitesi ve düşük pazar payı tespit edildi.",
                'actions': [
                    "Satış süreçlerini gözden geçirin",
                    "Müşteri memnuniyeti araştırması yapın",
                    "Rakip hareketlerini yakından takip edin"
                ],
                'data': high_risk.reset_index().to_dict('records')
            }
            recommendations.append(rec)
    
    # 3. Mevsimsel Strateji
    current_month = datetime.now().month
    seasonal_months = {
        12: ['DEKSAMETAZON', 'TROCMETAM'],
        1: ['DEKSAMETAZON', 'TROCMETAM'],
        2: ['DEKSAMETAZON'],
        6: ['PF IZOTONIK'],
        7: ['PF IZOTONIK'],
        8: ['PF IZOTONIK']
    }
    
    product_name = product_columns.get('pf_original', '')
    if current_month in seasonal_months and product_name in seasonal_months[current_month]:
        rec = {
            'type': '📈 MEVSİMSEL STRATEJİ',
            'priority': 'MEDIUM',
            'title': 'Mevsimsel Talep Artışı Bekleniyor',
            'description': f"{product_name} için mevsimsel talep artışı bekleniyor. Stok ve promosyon planı önerilir.",
            'actions': [
                "Stok seviyelerini gözden geçirin",
                "Mevsimsel promosyon kampanyaları planlayın",
                "Dağıtım kanallarını optimize edin"
            ],
            'data': []
        }
        recommendations.append(rec)
    
    # 4. Verimlilik Analizi
    if 'MANAGER' in df_filtered.columns and product_columns['pf'] in df_filtered.columns:
        manager_perf = df_filtered.groupby('MANAGER').agg({
            product_columns['pf']: 'sum',
            'TERRITORIES': 'nunique'
        }).reset_index()
        
        if len(manager_perf) > 0 and (manager_perf['TERRITORIES'] > 0).any():
            manager_perf['Sales_per_Territory'] = manager_perf[product_columns['pf']] / manager_perf['TERRITORIES']
            avg_efficiency = manager_perf['Sales_per_Territory'].mean()
            
            low_efficiency = manager_perf[manager_perf['Sales_per_Territory'] < avg_efficiency * 0.7]
            
            if not low_efficiency.empty:
                rec = {
                    'type': '📊 VERİMLİLİK ANALİZİ',
                    'priority': 'MEDIUM',
                    'title': 'Verimlilik İyileştirme Fırsatı',
                    'description': f"{len(low_efficiency)} yöneticinin territory başına satış verimliliği ortalamanın altında.",
                    'actions': [
                        "Eğitim ihtiyaçlarını değerlendirin",
                        "En iyi uygulamaları paylaşın",
                        "Performans hedeflerini gözden geçirin"
                    ],
                    'data': low_efficiency.to_dict('records')
                }
                recommendations.append(rec)
    
    # Öncelik sırasına göre sırala
    priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
    
    return recommendations

# =============================================================================
# GÖRSELLEŞTİRME FONKSİYONLARI
# =============================================================================
def create_advanced_time_series_chart(df, title="Satış Trendi", pf_col="PF_Satis"):
    """Gelişmiş zaman serisi grafiği"""
    fig = go.Figure()
    
    if len(df) == 0:
        return fig
    
    # Ana satış trendi
    fig.add_trace(go.Scatter(
        x=df['DATE'], y=df[pf_col],
        mode='lines+markers',
        name='PF Satış',
        line=dict(color='#3B82F6', width=3),
        marker=dict(size=8, color='#3B82F6'),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)',
        hovertemplate='<b>%{x|%b %Y}</b><br>Satış: %{y:,.0f}<extra></extra>'
    ))
    
    # Hareketli ortalamalar
    if 'MA_3' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['DATE'], y=df['MA_3'],
            mode='lines',
            name='3 Aylık Ortalama',
            line=dict(color='#10B981', width=2, dash='dash'),
            hovertemplate='3 Aylık Ort: %{y:,.0f}<extra></extra>'
        ))
    
    if 'MA_6' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['DATE'], y=df['MA_6'],
            mode='lines',
            name='6 Aylık Ortalama',
            line=dict(color='#8B5CF6', width=2, dash='dot'),
            hovertemplate='6 Aylık Ort: %{y:,.0f}<extra></extra>'
        ))
    
    # Trend çizgisi
    if len(df) > 3:
        z = np.polyfit(range(len(df)), df[pf_col], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=df['DATE'], y=p(range(len(df))),
            mode='lines',
            name='Trend Çizgisi',
            line=dict(color='#EF4444', width=2, dash='dash'),
            hovertemplate='Trend: %{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='white')),
        xaxis=dict(
            title='Tarih',
            gridcolor='rgba(255,255,255,0.1)',
            showline=True,
            linecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            title='Satış',
            gridcolor='rgba(255,255,255,0.1)',
            showline=True,
            linecolor='rgba(255,255,255,0.2)'
        ),
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor='rgba(255,255,255,0.2)'
        ),
        margin=dict(l=50, r=30, t=60, b=50)
    )
    
    return fig

def create_market_share_gauge(current_share, target_share=50):
    """Pazar payı gösterge grafiği"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_share,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Pazar Payı (%)", 'font': {'size': 20, 'color': 'white'}},
        delta={'reference': target_share, 'increasing': {'color': "#10B981"}, 'decreasing': {'color': "#EF4444"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#3B82F6"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.3)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.3)'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': target_share}
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Inter"}
    )
    
    return fig

# =============================================================================
# MAIN APP - GELİŞMİŞ VERSİYON
# =============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ v3</h1>', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Destekli Analizler • 📊 Gelişmiş ML Tahminleri • 🎯 Strateji Önerileri")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Sistem Kontrolleri")
        
        # Auto-refresh
        auto_refresh = st.checkbox("🔄 Otomatik Yenileme", value=False)
        if auto_refresh:
            refresh_rate = st.slider("Yenileme sıklığı (saniye)", 30, 300, 60)
            time_since_update = (datetime.now() - st.session_state.last_refresh).seconds
            if time_since_update > refresh_rate:
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        # Data upload
        st.header("📂 Veri Yükleme")
        uploaded_file = st.file_uploader("Excel dosyası yükleyin", type=['xlsx', 'xls'])
        
        if not uploaded_file:
            st.info("👈 Lütfen Excel dosyasını yükleyin")
            st.stop()
        
        try:
            df, product_cols_info = load_excel_data(uploaded_file)
            
            # Data hash kontrolü
            current_hash = calculate_data_hash(df)
            if current_hash != st.session_state.data_hash:
                st.session_state.data_hash = current_hash
                st.session_state.cached_results = {}  # Cache'i temizle
                st.success(f"✅ {len(df):,} satır veri yüklendi")
            
            st.session_state.last_refresh = datetime.now()
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.stop()
        
        # Product selection
        st.header("💊 Ürün Seçimi")
        selected_product = st.selectbox(
            "Analiz edilecek ürün",
            ["TROCMETAM", "CORTIPOL", "DEKSAMETAZON", "PF IZOTONIK"],
            index=0
        )
        
        # Get product columns
        product_cols = product_cols_info.get(selected_product, {})
        
        # Check if columns exist
        missing_cols = []
        if product_cols.get('pf'):
            st.success(f"✅ PF sütunu: {product_cols['pf']}")
        else:
            missing_cols.append(product_cols.get('pf_original', 'PF sütunu'))
            st.error(f"❌ PF sütunu bulunamadı: {product_cols.get('pf_original', 'PF sütunu')}")
        
        if product_cols.get('rakip'):
            st.success(f"✅ Rakip sütunu: {product_cols['rakip']}")
        else:
            st.warning(f"⚠️ Rakip sütunu bulunamadı: {product_cols.get('rakip_original', 'Rakip sütunu')}")
        
        # Date range
        st.header("📅 Tarih Aralığı")
        min_date = df['DATE'].min().date()
        max_date = df['DATE'].max().date()
        
        date_option = st.selectbox(
            "Dönem seçin",
            ["Tüm Veriler", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl", "Bu Yıl", "Geçen Yıl", "Özel Aralık"]
        )
        
        if date_option == "Tüm Veriler":
            date_filter = None
        elif date_option == "Son 3 Ay":
            date_filter = (pd.to_datetime(max_date) - pd.DateOffset(months=3), pd.to_datetime(max_date))
        elif date_option == "Son 6 Ay":
            date_filter = (pd.to_datetime(max_date) - pd.DateOffset(months=6), pd.to_datetime(max_date))
        elif date_option == "Son 1 Yıl":
            date_filter = (pd.to_datetime(max_date) - pd.DateOffset(years=1), pd.to_datetime(max_date))
        elif date_option == "Bu Yıl":
            date_filter = (pd.to_datetime(f'{datetime.now().year}-01-01'), pd.to_datetime(max_date))
        elif date_option == "Geçen Yıl":
            year = datetime.now().year - 1
            date_filter = (pd.to_datetime(f'{year}-01-01'), pd.to_datetime(f'{year}-12-31'))
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Başlangıç", min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("Bitiş", max_date, min_value=min_date, max_value=max_date)
            date_filter = (pd.to_datetime(start_date), pd.to_datetime(end_date))
        
        # Filters
        st.header("🔍 Filtreler")
        
        territories = ["TÜMÜ"] + sorted(df['TERRITORIES'].unique()) if 'TERRITORIES' in df.columns else ["TÜMÜ"]
        selected_territory = st.selectbox("Territory", territories)
        
        regions = ["TÜMÜ"] + sorted(df['REGION'].dropna().unique()) if 'REGION' in df.columns else ["TÜMÜ"]
        selected_region = st.selectbox("Bölge", regions)
        
        managers = ["TÜMÜ"] + sorted(df['MANAGER'].dropna().unique()) if 'MANAGER' in df.columns else ["TÜMÜ"]
        selected_manager = st.selectbox("Manager", managers)
        
        # Advanced filters
        with st.expander("🎯 Gelişmiş Filtreler"):
            min_sales = st.number_input("Minimum PF Satış", value=0)
            min_market_share = st.slider("Minimum Pazar Payı (%)", 0, 100, 0)
            
            # City filter
            cities = ["TÜMÜ"] + sorted(df['CITY'].dropna().unique()) if 'CITY' in df.columns else ["TÜMÜ"]
            selected_city = st.selectbox("Şehir", cities)
        
        # Apply filters
        df_filtered = df.copy()
        
        if selected_territory != "TÜMÜ" and 'TERRITORIES' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['TERRITORIES'] == selected_territory]
        if selected_region != "TÜMÜ" and 'REGION' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['REGION'] == selected_region]
        if selected_manager != "TÜMÜ" and 'MANAGER' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['MANAGER'] == selected_manager]
        if selected_city != "TÜMÜ" and 'CITY' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['CITY'] == selected_city]
        
        if date_filter:
            df_filtered = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & (df_filtered['DATE'] <= date_filter[1])]
        
        # Performance metrics in sidebar
        st.markdown("---")
        st.header("📊 Hızlı İstatistikler")
        
        if product_cols.get('pf'):
            total_pf = df_filtered[product_cols['pf']].sum() if product_cols['pf'] in df_filtered.columns else 0
        else:
            total_pf = 0
            
        if product_cols.get('rakip') and product_cols['rakip'] in df_filtered.columns:
            total_rakip = df_filtered[product_cols['rakip']].sum()
        else:
            total_rakip = 0
            
        total_market = total_pf + total_rakip
        market_share = (total_pf / total_market * 100) if total_market > 0 else 0
        
        st.metric("💊 PF Satış", f"{total_pf:,.0f}")
        st.metric("🎯 Pazar Payı", f"{market_share:.1f}%")
        
        if 'TERRITORIES' in df_filtered.columns:
            st.metric("🏢 Territory Sayısı", df_filtered['TERRITORIES'].nunique())
        
        st.metric("📅 Veri Periyodu", f"{df_filtered['YIL_AY'].nunique()} ay")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Genel Bakış", 
        "🤖 AI Önerileri", 
        "📈 Zaman Serisi & ML",
        "🎯 Rakip Analizi",
        "⭐ BCG & Strateji",
        "🧠 İleri Analizler",
        "🗺️ Coğrafi Analiz",
        "📊 Performans Dashboard",
        "📥 Raporlar & Export"
    ])
    
    # TAB 1: Genel Bakış
    with tab1:
        st.header("📊 Genel Performans Dashboard")
        
        # Top metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("💰 Toplam Satış", f"{total_pf:,.0f}")
        with col2:
            st.metric("🪙 Toplam Pazar", f"{total_market:,.0f}")
        with col3:
            st.metric("📊 Pazar Payı", f"{market_share:.1f}%")
        with col4:
            # Büyüme oranı hesaplama
            if date_filter:
                # Filtreli ve filtresiz veriyi karşılaştır
                if len(df) > 0 and total_pf > 0:
                    prev_total = df[df['DATE'] < date_filter[0]][product_cols['pf']].sum() if product_cols.get('pf') and product_cols['pf'] in df.columns else 0
                    growth = ((total_pf - prev_total) / prev_total * 100) if prev_total > 0 else 0
                else:
                    growth = 0
            else:
                growth = 0
            st.metric("📈 Büyüme Oranı", f"{growth:.1f}%")
        with col5:
            st.metric("🎯 Hedef Karşılama", 
                     f"{(market_share / 50 * 100 if 50 > 0 else 0):.1f}%" 
                     if market_share > 0 else "0%")
        
        st.markdown("---")
        
        # Charts row 1
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Market share gauge
            st.subheader("🎯 Pazar Payı Göstergesi")
            fig_gauge = create_market_share_gauge(market_share, 50)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_chart2:
            # Top 10 territories
            if 'TERRITORIES' in df_filtered.columns and product_cols.get('pf') and product_cols['pf'] in df_filtered.columns:
                st.subheader("🏆 Top 10 Territory")
                terr_perf = df_filtered.groupby('TERRITORIES')[product_cols['pf']].sum().nlargest(10).reset_index()
                fig_top10 = px.bar(
                    terr_perf, 
                    x='TERRITORIES', 
                    y=product_cols['pf'],
                    color=product_cols['pf'],
                    color_continuous_scale='Blues',
                    title='En Yüksek Satış Yapan Territory\'ler'
                )
                fig_top10.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_top10, use_container_width=True)
            else:
                st.info("Territory verisi bulunamadı")
        
        st.markdown("---")
        
        # Charts row 2
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            # Monthly trend
            if product_cols.get('pf') and product_cols['pf'] in df_filtered.columns:
                st.subheader("📈 Aylık Satış Trendi")
                monthly_sales = df_filtered.groupby('YIL_AY')[product_cols['pf']].sum().reset_index()
                fig_monthly = px.line(
                    monthly_sales,
                    x='YIL_AY',
                    y=product_cols['pf'],
                    markers=True,
                    title='Aylık Satış Trendi'
                )
                fig_monthly.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col_chart4:
            # Regional distribution
            if 'REGION' in df_filtered.columns and product_cols.get('pf') and product_cols['pf'] in df_filtered.columns:
                st.subheader("🗺️ Bölgesel Dağılım")
                regional_sales = df_filtered.groupby('REGION')[product_cols['pf']].sum().reset_index()
                fig_regional = px.pie(
                    regional_sales,
                    values=product_cols['pf'],
                    names='REGION',
                    title='Bölgelere Göre Satış Dağılımı',
                    hole=0.4
                )
                st.plotly_chart(fig_regional, use_container_width=True)
        
        # Performance table
        st.markdown("---")
        st.subheader("📋 Detaylı Performans Tablosu")
        
        # Calculate detailed performance
        group_cols = []
        if 'TERRITORIES' in df_filtered.columns:
            group_cols.append('TERRITORIES')
        if 'REGION' in df_filtered.columns:
            group_cols.append('REGION')
        if 'CITY' in df_filtered.columns:
            group_cols.append('CITY')
        if 'MANAGER' in df_filtered.columns:
            group_cols.append('MANAGER')
        
        if group_cols and product_cols.get('pf') and product_cols['pf'] in df_filtered.columns:
            performance_df = df_filtered.groupby(group_cols).agg({
                product_cols['pf']: ['sum', 'mean', 'std', 'count']
            }).round(2)
            
            performance_df.columns = ['PF_Total', 'PF_Avg', 'PF_Std', 'Transaction_Count']
            performance_df = performance_df.reset_index()
            
            # Calculate additional metrics if competitor data exists
            if product_cols.get('rakip') and product_cols['rakip'] in df_filtered.columns:
                competitor_sales = df_filtered.groupby(group_cols)[product_cols['rakip']].sum().reset_index()
                performance_df = pd.merge(performance_df, competitor_sales, on=group_cols, how='left')
                performance_df['Total_Market'] = performance_df['PF_Total'] + performance_df[product_cols['rakip']]
                performance_df['Market_Share'] = (performance_df['PF_Total'] / performance_df['Total_Market'] * 100).round(1)
                performance_df['Competition_Ratio'] = (performance_df[product_cols['rakip']] / performance_df['PF_Total']).round(2)
            else:
                performance_df['Total_Market'] = performance_df['PF_Total']
                performance_df['Market_Share'] = 100
                performance_df['Competition_Ratio'] = 0
            
            # Sort and display
            performance_df = performance_df.sort_values('PF_Total', ascending=False)
            
            # Format columns for display
            format_dict = {
                'PF_Total': '{:,.0f}',
                'PF_Avg': '{:,.0f}',
                'PF_Std': '{:,.0f}',
                'Transaction_Count': '{:,.0f}',
                'Market_Share': '{:.1f}%'
            }
            
            if product_cols.get('rakip') and product_cols['rakip'] in df_filtered.columns:
                format_dict[product_cols['rakip']] = '{:,.0f}'
                format_dict['Total_Market'] = '{:,.0f}'
                format_dict['Competition_Ratio'] = '{:.2f}'
            
            st.dataframe(
                performance_df.style.format(format_dict).background_gradient(subset=['Market_Share'], cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )
        else:
            st.info("Detaylı performans tablosu için yeterli veri yok")
    
    # TAB 2: AI Önerileri
    with tab2:
        st.header("🤖 AI Destekli Strateji Önerileri")
        
        with st.spinner("AI analiz yapıyor..."):
            recommendations = generate_ai_recommendations(df_filtered, product_cols, date_filter)
        
        if recommendations:
            for rec in recommendations:
                with st.container():
                    st.markdown(f"""
                    <div class="custom-card">
                        <h3 style="color: {'#10B981' if rec['priority'] == 'HIGH' else '#F59E0B' if rec['priority'] == 'MEDIUM' else '#64748B'}">
                            {rec['type']} - {rec['title']}
                        </h3>
                        <p>{rec['description']}</p>
                        <h4>🎯 Önerilen Aksiyonlar:</h4>
                        <ul>
                            {''.join([f'<li>{action}</li>' for action in rec['actions']])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if rec['data']:
                        with st.expander("📊 Detaylı Veriler"):
                            data_df = pd.DataFrame(rec['data'])
                            st.dataframe(data_df, use_container_width=True)
        else:
            st.info("🤖 Mevcut verilerle AI önerisi üretilemedi.")
        
        st.markdown("---")
        
        # Advanced AI Analysis
        st.subheader("🔍 Derin Analiz")
        col_ai1, col_ai2 = st.columns(2)
        
        with col_ai1:
            if st.button("🧠 Pazar Konsantrasyon Analizi", use_container_width=True):
                with st.spinner("Analiz yapılıyor..."):
                    concentration = calculate_market_concentration(df_filtered, product_cols, date_filter)
                    
                    st.metric("HHI İndex", f"{concentration['HHI_Index']:,.0f}")
                    st.metric("Konsantrasyon Seviyesi", concentration['Concentration_Level'])
                    st.metric("Risk Seviyesi", concentration['Risk_Level'])
                    st.metric("Top 4 Payı", f"{concentration['Top_4_Share']:.1f}%")
        
        with col_ai2:
            if st.button("💰 Fiyat Esnekliği Analizi", use_container_width=True):
                with st.spinner("Analiz yapılıyor..."):
                    elasticity = calculate_price_elasticity(df_filtered, product_cols, date_filter)
                    
                    st.metric("Esneklik Katsayısı", f"{elasticity['Elasticity_Coefficient']:.2f}")
                    st.metric("Esneklik Tipi", elasticity['Elasticity_Type'])
                    
                    st.info(f"💡 {elasticity['Recommendation']}")
    
    # TAB 3: Zaman Serisi & ML
    with tab3:
        st.header("📈 Gelişmiş Zaman Serisi & ML Tahminleri")
        
        if not product_cols.get('pf') or product_cols['pf'] not in df_filtered.columns:
            st.warning("PF satış verisi bulunamadı. Zaman serisi analizi yapılamıyor.")
        else:
            # Time series analysis
            monthly_df = df_filtered.groupby('YIL_AY').agg({
                product_cols['pf']: 'sum',
                'DATE': 'first'
            }).reset_index().sort_values('YIL_AY')
            
            monthly_df.columns = ['YIL_AY', 'PF_Satis', 'DATE']
            
            if product_cols.get('rakip') and product_cols['rakip'] in df_filtered.columns:
                monthly_rakip = df_filtered.groupby('YIL_AY')[product_cols['rakip']].sum().reset_index()
                monthly_df = pd.merge(monthly_df, monthly_rakip, on='YIL_AY', how='left')
                monthly_df['Pazar_Payi_%'] = safe_divide(monthly_df['PF_Satis'], monthly_df['PF_Satis'] + monthly_df[product_cols['rakip']]) * 100
            else:
                monthly_df['Pazar_Payi_%'] = 100
            
            col_ts1, col_ts2, col_ts3, col_ts4 = st.columns(4)
            with col_ts1:
                st.metric("📊 Ort. Aylık PF", f"{monthly_df['PF_Satis'].mean():,.0f}")
            with col_ts2:
                if len(monthly_df) > 1:
                    growth = monthly_df['PF_Satis'].pct_change().mean() * 100
                else:
                    growth = 0
                st.metric("📈 Ort. Aylık Büyüme", f"{growth:.1f}%")
            with col_ts3:
                st.metric("🎯 Ort. Pazar Payı", f"{monthly_df['Pazar_Payi_%'].mean():.1f}%")
            with col_ts4:
                st.metric("📅 Veri Dönemi", f"{len(monthly_df)} ay")
            
            st.markdown("---")
            
            # Time series chart
            col_chart1, col_chart2 = st.columns([3, 1])
            
            with col_chart1:
                fig_ts = create_advanced_time_series_chart(monthly_df, f"{selected_product} - Satış Trendi", "PF_Satis")
                st.plotly_chart(fig_ts, use_container_width=True)
            
            with col_chart2:
                st.subheader("📊 Trend Analizi")
                
                # Decomposition
                if len(monthly_df) >= 12:
                    trend_strength = np.corrcoef(range(len(monthly_df)), monthly_df['PF_Satis'])[0,1]
                    seasonality_strength = monthly_df['PF_Satis'].diff(12).std() / monthly_df['PF_Satis'].std() if monthly_df['PF_Satis'].std() > 0 else 0
                    
                    st.metric("📈 Trend Gücü", f"{abs(trend_strength):.2f}")
                    st.metric("🔄 Mevsimsellik", f"{seasonality_strength:.2f}")
                    st.metric("📊 Volatilite", f"{monthly_df['PF_Satis'].std() / monthly_df['PF_Satis'].mean():.2f}" 
                             if monthly_df['PF_Satis'].mean() > 0 else "0")
                
                # Seasonality
                st.markdown("### 🗓️ Mevsimsellik")
                if len(monthly_df) > 0:
                    monthly_avg = monthly_df.groupby(monthly_df['DATE'].dt.month)['PF_Satis'].mean()
                    if len(monthly_avg) > 0:
                        peak_month = monthly_avg.idxmax()
                        st.metric("📈 En Yüksek Ay", f"{peak_month}. Ay")
                        st.metric("📉 En Düşük Ay", f"{monthly_avg.idxmin()}. Ay")
            
            st.markdown("---")
            
            # ML Forecasting
            st.subheader("🤖 Makine Öğrenmesi ile Tahmin")
            
            forecast_months = st.slider("Tahmin periyodu (ay)", 1, 12, 6)
            
            if len(monthly_df) >= 12:
                if st.button("🚀 ML Tahminleri Oluştur", type="primary"):
                    with st.spinner("ML modelleri eğitiliyor..."):
                        ml_results, best_model, best_forecast, all_forecasts = train_advanced_ml_models(
                            monthly_df, product_cols, forecast_months
                        )
                    
                    if ml_results:
                        # Model performance
                        st.subheader("📊 Model Performans Karşılaştırması")
                        
                        perf_data = []
                        for name, result in ml_results.items():
                            perf_data.append({
                                'Model': name,
                                'MAE': result['MAE'],
                                'RMSE': result['RMSE'],
                                'MAPE': result['MAPE'],
                                'R2': result['R2']
                            })
                        
                        perf_df = pd.DataFrame(perf_data).sort_values('MAPE')
                        
                        col_perf1, col_perf2 = st.columns([3, 1])
                        
                        with col_perf1:
                            st.dataframe(
                                perf_df.style.format({
                                    'MAE': '{:,.2f}',
                                    'RMSE': '{:,.2f}',
                                    'MAPE': '{:.2f}%',
                                    'R2': '{:.3f}'
                                }).background_gradient(subset=['MAPE'], cmap='RdYlGn_r'),
                                use_container_width=True
                            )
                        
                        with col_perf2:
                            st.success(f"**🏆 En İyi Model:**\n\n**{best_model}**")
                            best_result = ml_results[best_model]
                            st.metric("MAPE", f"{best_result['MAPE']:.2f}%")
                            st.metric("R² Skoru", f"{best_result['R2']:.3f}")
                        
                        # Forecast visualization
                        st.subheader("🔮 Tahmin Görselleştirmesi")
                        
                        fig_forecast = go.Figure()
                        
                        # Historical data
                        fig_forecast.add_trace(go.Scatter(
                            x=monthly_df['DATE'],
                            y=monthly_df['PF_Satis'],
                            mode='lines+markers',
                            name='Gerçek Satış',
                            line=dict(color='#3B82F6', width=2),
                            marker=dict(size=6)
                        ))
                        
                        # Best model forecast
                        fig_forecast.add_trace(go.Scatter(
                            x=best_forecast['DATE'],
                            y=best_forecast['PF_Satis'],
                            mode='lines+markers',
                            name=f'Tahmin ({best_model})',
                            line=dict(color='#EF4444', width=2, dash='dash'),
                            marker=dict(size=6, symbol='diamond')
                        ))
                        
                        # Confidence interval (basit)
                        if len(best_forecast) > 0:
                            mean_forecast = best_forecast['PF_Satis'].mean()
                            std_forecast = best_forecast['PF_Satis'].std()
                            
                            fig_forecast.add_trace(go.Scatter(
                                x=best_forecast['DATE'].tolist() + best_forecast['DATE'].tolist()[::-1],
                                y=(best_forecast['PF_Satis'] + 1.96*std_forecast).tolist() + 
                                  (best_forecast['PF_Satis'] - 1.96*std_forecast).tolist()[::-1],
                                fill='toself',
                                fillcolor='rgba(239, 68, 68, 0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='%95 Güven Aralığı',
                                showlegend=True
                            ))
                        
                        fig_forecast.update_layout(
                            title=f'{selected_product} - ML Tahminleri',
                            xaxis_title='Tarih',
                            yaxis_title='Satış',
                            height=500,
                            hovermode='x unified',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        # Feature importance
                        if ml_results[best_model]['feature_importance'] is not None:
                            st.subheader("🔍 Feature Importance")
                            importance_df = ml_results[best_model]['feature_importance']
                            
                            fig_importance = px.bar(
                                importance_df,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='En Önemli Özellikler',
                                color='Importance',
                                color_continuous_scale='Viridis'
                            )
                            
                            fig_importance.update_layout(
                                height=400,
                                yaxis={'categoryorder': 'total ascending'},
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            
                            st.plotly_chart(fig_importance, use_container_width=True)
                    else:
                        st.warning("ML modelleri eğitilemedi. Yeterli veri olmayabilir.")
            else:
                st.warning("ML tahmini için en az 12 aylık veri gereklidir.")
    
    # TAB 4: Rakip Analizi
    with tab4:
        st.header("🎯 Detaylı Rakip Analizi")
        
        if product_cols.get('rakip') and product_cols['rakip'] in df_filtered.columns:
            # Competitor analysis metrics
            comp_monthly = df_filtered.groupby('YIL_AY').agg({
                product_cols['pf']: 'sum',
                product_cols['rakip']: 'sum',
                'DATE': 'first'
            }).reset_index().sort_values('YIL_AY')
            
            comp_monthly['PF_Pay'] = (comp_monthly[product_cols['pf']] / 
                                     (comp_monthly[product_cols['pf']] + comp_monthly[product_cols['rakip']])) * 100
            
            col_comp1, col_comp2, col_comp3, col_comp4 = st.columns(4)
            
            with col_comp1:
                avg_pf = comp_monthly[product_cols['pf']].mean()
                st.metric("📊 Ort. PF Satış", f"{avg_pf:,.0f}")
            
            with col_comp2:
                avg_comp = comp_monthly[product_cols['rakip']].mean()
                st.metric("📊 Ort. Rakip Satış", f"{avg_comp:,.0f}")
            
            with col_comp3:
                comp_growth = comp_monthly[product_cols['rakip']].pct_change().mean() * 100
                st.metric("📈 Rakip Büyümesi", f"{comp_growth:.1f}%")
            
            with col_comp4:
                pf_growth = comp_monthly[product_cols['pf']].pct_change().mean() * 100
                st.metric("📈 PF Büyümesi", f"{pf_growth:.1f}%")
            
            st.markdown("---")
            
            # Competitor comparison chart
            fig_comp = go.Figure()
            
            fig_comp.add_trace(go.Scatter(
                x=comp_monthly['DATE'],
                y=comp_monthly[product_cols['pf']],
                mode='lines+markers',
                name='PF Satış',
                line=dict(color='#3B82F6', width=2)
            ))
            
            fig_comp.add_trace(go.Scatter(
                x=comp_monthly['DATE'],
                y=comp_monthly[product_cols['rakip']],
                mode='lines+markers',
                name='Rakip Satış',
                line=dict(color='#EF4444', width=2)
            ))
            
            fig_comp.update_layout(
                title='PF vs Rakip Satış Karşılaştırması',
                xaxis_title='Tarih',
                yaxis_title='Satış',
                height=400,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Market share trend
            fig_share = go.Figure()
            
            fig_share.add_trace(go.Scatter(
                x=comp_monthly['DATE'],
                y=comp_monthly['PF_Pay'],
                mode='lines+markers',
                name='Pazar Payı',
                line=dict(color='#10B981', width=3),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.1)'
            ))
            
            fig_share.update_layout(
                title='Pazar Payı Trendi',
                xaxis_title='Tarih',
                yaxis_title='Pazar Payı (%)',
                height=300,
                yaxis=dict(range=[0, 100]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_share, use_container_width=True)
            
        else:
            st.info("Rakip verisi bulunamadı. Rakip analizi yapılamıyor.")
    
    # TAB 5: BCG Matrix
    with tab5:
        st.header("⭐ BCG Matrix & Stratejik Analiz")
        
        if product_cols.get('pf') and product_cols['pf'] in df_filtered.columns:
            # BCG Matrix hesaplamaları
            if 'TERRITORIES' in df_filtered.columns:
                # Territory bazlı BCG
                bcg_data = df_filtered.groupby('TERRITORIES').agg({
                    product_cols['pf']: 'sum'
                }).reset_index()
                
                # Market share hesaplama
                if product_cols.get('rakip') and product_cols['rakip'] in df_filtered.columns:
                    territory_competitor = df_filtered.groupby('TERRITORIES')[product_cols['rakip']].sum().reset_index()
                    bcg_data = pd.merge(bcg_data, territory_competitor, on='TERRITORIES', how='left')
                    bcg_data['Total_Market'] = bcg_data[product_cols['pf']] + bcg_data[product_cols['rakip']]
                    bcg_data['Market_Share'] = (bcg_data[product_cols['pf']] / bcg_data['Total_Market']) * 100
                else:
                    bcg_data['Market_Share'] = 100
                    bcg_data['Total_Market'] = bcg_data[product_cols['pf']]
                
                # Growth rate hesaplama
                growth_data = []
                for territory in bcg_data['TERRITORIES'].unique():
                    territory_df = df_filtered[df_filtered['TERRITORIES'] == territory]
                    if len(territory_df) >= 2:
                        # Son 3 ay vs önceki 3 ay
                        recent_date = territory_df['DATE'].max()
                        three_months_ago = recent_date - pd.DateOffset(months=3)
                        
                        recent_sales = territory_df[territory_df['DATE'] > three_months_ago][product_cols['pf']].sum()
                        previous_sales = territory_df[territory_df['DATE'] <= three_months_ago][product_cols['pf']].sum()
                        
                        growth_rate = ((recent_sales - previous_sales) / previous_sales * 100) if previous_sales > 0 else 0
                    else:
                        growth_rate = 0
                    
                    growth_data.append({'TERRITORIES': territory, 'Growth_Rate': growth_rate})
                
                growth_df = pd.DataFrame(growth_data)
                bcg_data = pd.merge(bcg_data, growth_df, on='TERRITORIES', how='left')
                
                # BCG kategorileri
                def get_bcg_category(row):
                    ms = row['Market_Share']
                    gr = row['Growth_Rate']
                    
                    if ms >= 50 and gr >= 10:
                        return '⭐ YILDIZ'
                    elif ms >= 50 and gr < 10:
                        return '🐄 NAKİT İNEĞİ'
                    elif ms < 50 and gr >= 10:
                        return '❓ SORU İŞARETİ'
                    else:
                        return '🐕 KÖPEK'
                
                bcg_data['BCG_Category'] = bcg_data.apply(get_bcg_category, axis=1)
                
                # BCG Matrix visualization
                fig_bcg = px.scatter(
                    bcg_data,
                    x='Market_Share',
                    y='Growth_Rate',
                    size=product_cols['pf'],
                    color='BCG_Category',
                    hover_name='TERRITORIES',
                    title='BCG Matrix - Territory Bazlı',
                    labels={
                        'Market_Share': 'Pazar Payı (%)',
                        'Growth_Rate': 'Büyüme Oranı (%)',
                        product_cols['pf']: 'Satış Büyüklüğü'
                    }
                )
                
                # Add quadrant lines
                fig_bcg.add_shape(
                    type="line",
                    x0=50, y0=-100, x1=50, y1=100,
                    line=dict(color="white", width=1, dash="dash")
                )
                
                fig_bcg.add_shape(
                    type="line",
                    x0=0, y0=10, x1=100, y1=10,
                    line=dict(color="white", width=1, dash="dash")
                )
                
                # Add quadrant labels
                fig_bcg.add_annotation(x=25, y=50, text="SORU İŞARETİ", showarrow=False, font=dict(color="white"))
                fig_bcg.add_annotation(x=75, y=50, text="YILDIZ", showarrow=False, font=dict(color="white"))
                fig_bcg.add_annotation(x=25, y=-30, text="KÖPEK", showarrow=False, font=dict(color="white"))
                fig_bcg.add_annotation(x=75, y=-30, text="NAKİT İNEĞİ", showarrow=False, font=dict(color="white"))
                
                fig_bcg.update_layout(
                    height=600,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig_bcg, use_container_width=True)
                
                # BCG kategorilerine göre özet
                st.subheader("📊 BCG Kategori Özeti")
                bcg_summary = bcg_data.groupby('BCG_Category').agg({
                    'TERRITORIES': 'count',
                    product_cols['pf']: 'sum',
                    'Market_Share': 'mean',
                    'Growth_Rate': 'mean'
                }).round(2)
                
                bcg_summary.columns = ['Territory Sayısı', 'Toplam Satış', 'Ort. Pazar Payı', 'Ort. Büyüme']
                st.dataframe(bcg_summary, use_container_width=True)
                
            else:
                st.info("BCG analizi için territory verisi gerekiyor.")
        else:
            st.info("BCG analizi için PF satış verisi gerekiyor.")
    
    # TAB 6: İleri Analizler
    with tab6:
        st.header("🧠 İleri Analizler & Özel Metrikler")
        
        # Sales Velocity
        st.subheader("⚡ Satış Hızı Analizi")
        velocity_data = calculate_sales_velocity(df_filtered, product_cols, date_filter)
        
        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        with col_v1:
            st.metric("🚀 Ort. Günlük Satış", f"{velocity_data['Avg_Daily_Sales']:,.0f}")
        with col_v2:
            st.metric("📈 Büyüme Hızı", f"{velocity_data['Sales_Growth_Rate']:.1f}%")
        with col_v3:
            st.metric("📊 Volatilite", f"{velocity_data['Sales_Volatility']:.2f}")
        with col_v4:
            st.metric("🏎️ Hız Sınıfı", velocity_data['Velocity_Class'])
        
        # Customer Lifetime Value
        st.subheader("💰 Müşteri Yaşam Boyu Değeri (CLV)")
        clv_data = calculate_customer_lifetime_value(df_filtered, product_cols, date_filter)
        
        if not clv_data.empty:
            col_clv1, col_clv2 = st.columns(2)
            
            with col_clv1:
                # CLV Distribution
                fig_clv = px.histogram(
                    clv_data,
                    x='CLV',
                    nbins=20,
                    title='CLV Dağılımı',
                    color='CLV_Quartile',
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig_clv, use_container_width=True)
            
            with col_clv2:
                # Top CLV customers
                st.dataframe(
                    clv_data[['Territory', 'CLV', 'Avg_Monthly_Sales', 'Churn_Rate', 'CLV_Quartile']]
                    .sort_values('CLV', ascending=False)
                    .head(10)
                    .style.format({
                        'CLV': '{:,.0f}',
                        'Avg_Monthly_Sales': '{:,.0f}',
                        'Churn_Rate': '{:.2%}'
                    }),
                    use_container_width=True
                )
        else:
            st.info("CLV analizi için yeterli veri yok")
        
        # Market Segmentation
        st.subheader("🎯 Pazar Segmentasyonu")
        segmentation = calculate_market_segmentation(df_filtered, product_cols, date_filter)
        
        if not segmentation['Region_Segmentation'].empty:
            col_seg1, col_seg2 = st.columns(2)
            
            with col_seg1:
                st.write("##### Bölgesel Segmentasyon")
                st.dataframe(
                    segmentation['Region_Segmentation']
                    .style.format({
                        'PF_Total': '{:,.0f}',
                        'PF_Avg': '{:,.0f}',
                        'Market_Share': '{:.1f}%'
                    }),
                    use_container_width=True
                )
        
        if not segmentation['City_Segmentation'].empty:
            with col_seg2:
                st.write("##### Şehir Segmentleri")
                fig_seg = px.sunburst(
                    segmentation['City_Segmentation'],
                    path=['Segment', 'CITY_NORMALIZED'],
                    values='Total_Market',
                    color='Market_Share',
                    color_continuous_scale='RdYlGn',
                    title='Şehir Segmentasyonu'
                )
                st.plotly_chart(fig_seg, use_container_width=True)
        
        # Anomaly Detection
        st.subheader("🚨 Anomali Tespiti")
        anomalies = calculate_anomaly_detection(df_filtered, product_cols, date_filter)
        
        if not anomalies.empty:
            st.warning(f"⚠️ {len(anomalies)} adet anomali tespit edildi!")
            
            col_ano1, col_ano2 = st.columns(2)
            
            with col_ano1:
                # Anomaly types
                anomaly_counts = anomalies['Anomaly_Type'].value_counts()
                fig_ano_type = px.pie(
                    values=anomaly_counts.values,
                    names=anomaly_counts.index,
                    title='Anomali Türleri'
                )
                st.plotly_chart(fig_ano_type, use_container_width=True)
            
            with col_ano2:
                # Anomaly details
                st.dataframe(
                    anomalies[['Territory', 'Date', 'Sales', 'Z_Score', 'Anomaly_Type']]
                    .sort_values('Z_Score', ascending=False)
                    .head(10)
                    .style.format({
                        'Sales': '{:,.0f}',
                        'Z_Score': '{:.2f}'
                    }),
                    use_container_width=True
                )
        else:
            st.success("✅ Anomali tespit edilmedi.")
    
    # TAB 7: Coğrafi Analiz
    with tab7:
        st.header("🗺️ Coğrafi Analiz & Haritalar")
        
        if 'CITY_NORMALIZED' in df_filtered.columns and product_cols.get('pf') and product_cols['pf'] in df_filtered.columns:
            # City-based analysis
            city_sales = df_filtered.groupby('CITY_NORMALIZED').agg({
                product_cols['pf']: 'sum'
            }).reset_index().sort_values(product_cols['pf'], ascending=False)
            
            # Top 20 cities
            st.subheader("🏙️ Şehir Bazlı Satış Analizi")
            fig_cities = px.bar(
                city_sales.head(20),
                x='CITY_NORMALIZED',
                y=product_cols['pf'],
                color=product_cols['pf'],
                color_continuous_scale='Viridis',
                title='En Yüksek Satış Yapan 20 Şehir'
            )
            fig_cities.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_cities, use_container_width=True)
            
            # Regional heatmap
            if 'REGION' in df_filtered.columns:
                st.subheader("🌡️ Bölgesel Isı Haritası")
                region_sales = df_filtered.groupby('REGION')[product_cols['pf']].sum().reset_index()
                
                fig_region_heat = px.treemap(
                    region_sales,
                    path=['REGION'],
                    values=product_cols['pf'],
                    color=product_cols['pf'],
                    color_continuous_scale='RdYlGn',
                    title='Bölgelere Göre Satış Dağılımı'
                )
                st.plotly_chart(fig_region_heat, use_container_width=True)
        else:
            st.info("Coğrafi analiz için şehir verisi gerekiyor.")
    
    # TAB 8: Performans Dashboard
    with tab8:
        st.header("📊 Real-time Performans Dashboard")
        
        # KPI Dashboard
        st.subheader("🎯 Ana Performans Göstergeleri")
        
        # Calculate advanced KPIs
        df_period = df_filtered if date_filter is None else df_filtered[
            (df_filtered['DATE'] >= date_filter[0]) & (df_filtered['DATE'] <= date_filter[1])
        ]
        
        if 'TERRITORIES' in df_period.columns:
            territory_count = df_period['TERRITORIES'].nunique()
        else:
            territory_count = 0
        
        if territory_count > 0:
            avg_sales_per_territory = total_pf / territory_count
        else:
            avg_sales_per_territory = 0
        
        if 'MANAGER' in df_period.columns:
            manager_count = df_period['MANAGER'].nunique()
            if manager_count > 0:
                avg_sales_per_manager = total_pf / manager_count
            else:
                avg_sales_per_manager = 0
        else:
            manager_count = 0
            avg_sales_per_manager = 0
        
        # Growth metrics
        monthly_sales_for_growth = df_period.groupby('YIL_AY')[product_cols['pf']].sum().reset_index() if product_cols.get('pf') and product_cols['pf'] in df_period.columns else pd.DataFrame()
        if len(monthly_sales_for_growth) > 1:
            current_month_sales = monthly_sales_for_growth.iloc[-1][product_cols['pf']] if not monthly_sales_for_growth.empty else 0
            previous_month_sales = monthly_sales_for_growth.iloc[-2][product_cols['pf']] if len(monthly_sales_for_growth) > 1 else 0
            mom_growth = ((current_month_sales - previous_month_sales) / previous_month_sales * 100) if previous_month_sales > 0 else 0
        else:
            mom_growth = 0
        
        # Display KPIs in columns
        col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
        
        with col_kpi1:
            st.metric("💰 Toplam Satış", f"{total_pf:,.0f}")
        
        with col_kpi2:
            st.metric("📈 Aylık Büyüme", f"{mom_growth:.1f}%")
        
        with col_kpi3:
            st.metric("🏢 Territory Başı Satış", f"{avg_sales_per_territory:,.0f}")
        
        with col_kpi4:
            st.metric("👨‍💼 Manager Başı Satış", f"{avg_sales_per_manager:,.0f}")
        
        with col_kpi5:
            efficiency_score = min((market_share / 50 * 70 + min(mom_growth, 20) / 20 * 30), 100) if market_share > 0 else 0
            st.metric("📊 Verimlilik Skoru", f"{efficiency_score:.0f}/100")
        
        st.markdown("---")
        
        # Performance charts
        if 'REGION' in df_period.columns and product_cols.get('pf') and product_cols['pf'] in df_period.columns:
            col_dash1, col_dash2 = st.columns(2)
            
            with col_dash1:
                # Performance by region
                regional_perf = df_period.groupby('REGION').agg({
                    product_cols['pf']: 'sum',
                }).reset_index()
                
                if product_cols.get('rakip') and product_cols['rakip'] in df_period.columns:
                    regional_competitor = df_period.groupby('REGION')[product_cols['rakip']].sum().reset_index()
                    regional_perf = pd.merge(regional_perf, regional_competitor, on='REGION', how='left')
                    regional_perf['Market_Share'] = (regional_perf[product_cols['pf']] / 
                                                    (regional_perf[product_cols['pf']] + regional_perf[product_cols['rakip']])) * 100
                else:
                    regional_perf['Market_Share'] = 100
                
                fig_regional_bar = px.bar(
                    regional_perf.sort_values('Market_Share', ascending=False),
                    x='REGION',
                    y='Market_Share',
                    color='Market_Share',
                    color_continuous_scale='RdYlGn',
                    title='Bölgelere Göre Pazar Payı'
                )
                st.plotly_chart(fig_regional_bar, use_container_width=True)
        
        if 'MANAGER' in df_period.columns and product_cols.get('pf') and product_cols['pf'] in df_period.columns:
            with col_dash2:
                # Manager performance
                manager_perf = df_period.groupby('MANAGER').agg({
                    product_cols['pf']: 'sum',
                    'TERRITORIES': 'nunique'
                }).reset_index()
                
                manager_perf['Efficiency'] = manager_perf[product_cols['pf']] / manager_perf['TERRITORIES']
                
                fig_manager = px.scatter(
                    manager_perf,
                    x='TERRITORIES',
                    y=product_cols['pf'],
                    size='Efficiency',
                    color='Efficiency',
                    hover_name='MANAGER',
                    color_continuous_scale='Viridis',
                    title='Manager Performansı'
                )
                st.plotly_chart(fig_manager, use_container_width=True)
    
    # TAB 9: Raporlar & Export
    with tab9:
        st.header("📥 Raporlar & Veri Export")
        
        col_report1, col_report2 = st.columns(2)
        
        with col_report1:
            st.subheader("📊 Ön Tanımlı Raporlar")
            
            report_type = st.selectbox(
                "Rapor Türü",
                [
                    "Tam Performans Raporu",
                    "Territory Detay Raporu",
                    "Zaman Serisi Raporu",
                    "Bölgesel Analiz Raporu",
                    "Manager Performans Raporu"
                ]
            )
            
            if st.button("📥 Rapor Oluştur", type="primary", use_container_width=True):
                with st.spinner("Rapor hazırlanıyor..."):
                    # Create comprehensive report
                    output = BytesIO()
                    
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Territory performance
                        if 'TERRITORIES' in df_filtered.columns and product_cols.get('pf') and product_cols['pf'] in df_filtered.columns:
                            group_cols_report = []
                            if 'TERRITORIES' in df_filtered.columns:
                                group_cols_report.append('TERRITORIES')
                            if 'REGION' in df_filtered.columns:
                                group_cols_report.append('REGION')
                            if 'CITY' in df_filtered.columns:
                                group_cols_report.append('CITY')
                            if 'MANAGER' in df_filtered.columns:
                                group_cols_report.append('MANAGER')
                            
                            if group_cols_report:
                                terr_perf = df_filtered.groupby(group_cols_report).agg({
                                    product_cols['pf']: ['sum', 'mean', 'std', 'count']
                                }).round(2)
                                
                                terr_perf.columns = ['PF_Total', 'PF_Avg', 'PF_Std', 'Transaction_Count']
                                terr_perf.to_excel(writer, sheet_name='Territory Performans')
                        
                        # Monthly time series
                        if product_cols.get('pf') and product_cols['pf'] in df_filtered.columns:
                            monthly_report = df_filtered.groupby('YIL_AY').agg({
                                product_cols['pf']: 'sum',
                                'DATE': 'first'
                            }).reset_index().sort_values('YIL_AY')
                            monthly_report.to_excel(writer, sheet_name='Zaman Serisi')
                        
                        # Regional analysis
                        if 'REGION' in df_filtered.columns and product_cols.get('pf') and product_cols['pf'] in df_filtered.columns:
                            regional_report = df_filtered.groupby('REGION').agg({
                                product_cols['pf']: ['sum', 'mean', 'std'],
                                'TERRITORIES': 'nunique',
                                'CITY': 'nunique'
                            }).round(2)
                            regional_report.to_excel(writer, sheet_name='Bölgesel Analiz')
                        
                        # Manager performance
                        if 'MANAGER' in df_filtered.columns and product_cols.get('pf') and product_cols['pf'] in df_filtered.columns:
                            manager_report = df_filtered.groupby('MANAGER').agg({
                                product_cols['pf']: ['sum', 'mean'],
                                'TERRITORIES': 'nunique',
                            }).round(2)
                            manager_report.to_excel(writer, sheet_name='Manager Performansı')
                    
                    st.success("✅ Rapor hazır!")
                    
                    # Download button
                    st.download_button(
                        label="💾 Excel Raporunu İndir",
                        data=output.getvalue(),
                        file_name=f"ticari_portfoy_raporu_{selected_product}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        
        with col_report2:
            st.subheader("🎨 Özelleştirilmiş Rapor")
            
            # Custom report options
            st.checkbox("Territory detayları", value=True)
            st.checkbox("Zaman serisi analizi", value=True)
            st.checkbox("Bölgesel analiz", value=True)
            st.checkbox("Manager performansı", value=True)
            st.checkbox("Rakip analizi", value=False)
            
            report_format = st.selectbox("Format", ["Excel", "CSV", "JSON"])
            
            if st.button("🛠️ Özel Rapor Oluştur", use_container_width=True):
                st.info("Özel rapor özelliği geliştirme aşamasındadır.")
        
        st.markdown("---")
        
        # Data Export
        st.subheader("📤 Veri Export Seçenekleri")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="📊 CSV Olarak İndir",
                data=csv,
                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_export2:
            json_str = df_filtered.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📈 JSON Olarak İndir",
                data=json_str,
                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_export3:
            excel_buffer = BytesIO()
            df_filtered.to_excel(excel_buffer, index=False)
            st.download_button(
                label="📋 Excel Olarak İndir",
                data=excel_buffer.getvalue(),
                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
