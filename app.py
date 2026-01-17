"""
🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ
Territory Bazlı Performans, ML Tahminleme, Türkiye Haritası ve Rekabet Analizi

Özellikler:
- 🗺️ Türkiye il bazlı harita görselleştirme (GELİŞTİRİLMİŞ VERSİYON)
- 🤖 GERÇEK Machine Learning (Linear Regression, Ridge, Random Forest, XGBoost)
- 📊 Aylık/Yıllık dönem seçimi
- 📈 Gelişmiş rakip analizi ve trend karşılaştırması
- 🎯 Dinamik zaman aralığı filtreleme
- 🔮 6 aylık ML tahminleme
- 📉 Anomali tespiti
- 📊 Çoklu metrik karşılaştırma
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from prophet import Prophet
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

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
# CSS
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
    
    .metric-delta {
        font-size: 1.2rem !important;
        font-weight: 600;
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
    
    h1, h2, h3 {
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
    
    /* Custom alerts */
    .alert-success {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10B981;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #F59E0B;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .alert-danger {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #EF4444;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# RENK PALETLERİ
# =============================================================================
REGION_COLORS = {
    "MARMARA": "#3B82F6",
    "BATI ANADOLU": "#10B981",
    "EGE": "#F59E0B",
    "İÇ ANADOLU": "#8B5CF6",
    "GÜNEY DOĞU ANADOLU": "#EF4444",
    "KUZEY ANADOLU": "#06B6D4",
    "KARADENİZ": "#06B6D4",
    "AKDENİZ": "#3B82F6",
    "DOĞU ANADOLU": "#10B981",
    "DİĞER": "#64748B"
}

PERFORMANCE_COLORS = {
    "high": "#1F7A5A",
    "medium": "#C48A2A",
    "low": "#B23A3A",
    "positive": "#1F7A5A",
    "negative": "#B23A3A",
    "neutral": "#6B7280",
    "warning": "#C48A2A",
    "info": "#1E40AF",
    "success": "#166534",
    "danger": "#991B1B"
}

# =============================================================================
# SABİTLER VE EŞLEMELER
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

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def safe_divide(a, b):
    """Güvenli bölme işlemi"""
    return np.where(b != 0, a / b, 0)

def get_product_columns(product):
    """Ürün kolonlarını döndür - Dinamik olarak veri sütunlarına göre"""
    if product == "TROCMETAM":
        # Excel'deki gerçek sütun adlarını kontrol et
        return {"pf": "TROCMETAM", "rakip": "DIGER TROCMETAM"}
    elif product == "CORTIPOL":
        return {"pf": "CORTIPOL", "rakip": "DIGER CORTIPOL"}
    elif product == "DEKSAMETAZON":
        return {"pf": "DEKSAMETAZON", "rakip": "DIGER DEKSAMETAZON"}
    else:
        return {"pf": "PF IZOTONIK", "rakip": "DIGER IZOTONIK"}

def get_available_products(df):
    """Excel dosyasındaki mevcut ürünleri tespit et"""
    possible_products = {
        'TROCMETAM': ['TROCMETAM', 'TROC METAM', 'TROCMETAM SATIS'],
        'CORTIPOL': ['CORTIPOL', 'CORTIPOL SATIS'],
        'DEKSAMETAZON': ['DEKSAMETAZON', 'DEKSAMETAZON SATIS'],
        'PF IZOTONIK': ['PF IZOTONIK', 'IZOTONIK', 'PF_IZOTONIK']
    }
    
    available = []
    for product, possible_names in possible_products.items():
        for name in possible_names:
            if name in df.columns:
                available.append(product)
                break
    
    return available if available else ["TROCMETAM", "CORTIPOL", "DEKSAMETAZON", "PF IZOTONIK"]

def normalize_city_name_fixed(city_name):
    """Düzeltilmiş şehir normalizasyon"""
    if pd.isna(city_name):
        return None
    
    city_upper = str(city_name).strip().upper()
    
    # Fix known encoding issues
    if city_upper in FIX_CITY_MAP:
        return FIX_CITY_MAP[city_upper]
    
    # Turkish character mapping
    tr_map = {
        "İ": "I", "Ğ": "G", "Ü": "U",
        "Ş": "S", "Ö": "O", "Ç": "C",
        "Â": "A", "Î": "I", "Û": "U"
    }
    
    for k, v in tr_map.items():
        city_upper = city_upper.replace(k, v)
    
    return CITY_NORMALIZE_CLEAN.get(city_upper, city_name)

def detect_column_names(df):
    """Excel dosyasındaki sütun adlarını tespit et"""
    column_mapping = {}
    
    # Satış sütunlarını tespit et
    sales_columns = [col for col in df.columns if any(x in str(col).upper() for x in ['SATIS', 'SALES', 'TUTAR', 'MIKTAR'])]
    
    # Territory/Şehir/Bölge sütunlarını tespit et
    territory_col = None
    city_col = None
    region_col = None
    date_col = None
    manager_col = None
    
    for col in df.columns:
        col_upper = str(col).upper()
        
        if any(x in col_upper for x in ['TERRITORY', 'TERRITOR', 'BOLGE', 'BÖLGE']):
            territory_col = col
        elif any(x in col_upper for x in ['CITY', 'SEHIR', 'ŞEHİR', 'IL']):
            city_col = col
        elif 'REGION' in col_upper or 'BOLGE' in col_upper:
            region_col = col
        elif any(x in col_upper for x in ['DATE', 'TARIH', 'TARİH', 'AY', 'YIL']):
            date_col = col
        elif any(x in col_upper for x in ['MANAGER', 'SORUMLU', 'SATISCI']):
            manager_col = col
    
    return {
        'territory': territory_col or 'TERRITORIES',
        'city': city_col or 'CITY',
        'region': region_col or 'REGION',
        'date': date_col or 'DATE',
        'manager': manager_col or 'MANAGER',
        'sales_columns': sales_columns
    }

# =============================================================================
# VERİ YÜKLEME
# =============================================================================

@st.cache_data
def load_excel_data(file):
    """Excel dosyasını yükle ve sütunları otomatik tespit et"""
    try:
        df = pd.read_excel(file)
        
        # Sütun adlarını tespit et
        detected_columns = detect_column_names(df)
        
        # DataFrame'i standart formata dönüştür
        df_clean = df.copy()
        
        # Sütunları yeniden adlandır
        rename_dict = {}
        for std_name, detected_name in detected_columns.items():
            if detected_name and detected_name != std_name:
                rename_dict[detected_name] = std_name
        
        if rename_dict:
            df_clean = df_clean.rename(columns=rename_dict)
        
        # Tarih sütununu işle
        if 'DATE' in df_clean.columns:
            df_clean['DATE'] = pd.to_datetime(df_clean['DATE'], errors='coerce')
        else:
            # Tarih sütunu bulunamazsa index kullan
            df_clean['DATE'] = pd.date_range(start='2023-01-01', periods=len(df_clean), freq='M')
        
        # Ek kolonlar oluştur
        df_clean['YIL_AY'] = df_clean['DATE'].dt.strftime('%Y-%m')
        df_clean['AY'] = df_clean['DATE'].dt.month
        df_clean['YIL'] = df_clean['DATE'].dt.year
        
        # String kolonlarını temizle
        for col in ['TERRITORIES', 'CITY', 'REGION', 'MANAGER']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.upper().str.strip()
        
        # Şehir adlarını normalleştir
        if 'CITY' in df_clean.columns:
            df_clean['CITY_NORMALIZED'] = df_clean['CITY'].apply(normalize_city_name_fixed)
        
        return df_clean
        
    except Exception as e:
        st.error(f"Excel dosyası okunurken hata: {str(e)}")
        # Örnek veri oluştur
        return create_sample_data()

def create_sample_data():
    """Örnek veri oluştur (test için)"""
    dates = pd.date_range(start='2023-01-01', end='2025-12-01', freq='M')
    territories = [f'TERR_{i:03d}' for i in range(1, 51)]
    cities = ['İSTANBUL', 'ANKARA', 'İZMİR', 'BURSA', 'ANTALYA', 'ADANA', 'KONYA', 'GAZİANTEP']
    regions = ['MARMARA', 'İÇ ANADOLU', 'EGE', 'AKDENİZ', 'GÜNEYDOĞU ANADOLU']
    
    data = []
    for date in dates:
        for territory in territories[:20]:  # 20 territory
            city = np.random.choice(cities)
            region = np.random.choice(regions)
            manager = f'MANAGER_{np.random.randint(1, 6)}'
            
            # Rastgele satış verileri
            pf_sales = np.random.randint(1000, 50000)
            competitor_sales = np.random.randint(800, 45000)
            
            data.append({
                'DATE': date,
                'TERRITORIES': territory,
                'CITY': city,
                'REGION': region,
                'MANAGER': manager,
                'TROCMETAM': pf_sales,
                'DIGER TROCMETAM': competitor_sales,
                'CORTIPOL': np.random.randint(500, 25000),
                'DIGER CORTIPOL': np.random.randint(400, 20000),
                'DEKSAMETAZON': np.random.randint(300, 15000),
                'DIGER DEKSAMETAZON': np.random.randint(200, 12000),
                'PF IZOTONIK': np.random.randint(2000, 60000),
                'DIGER IZOTONIK': np.random.randint(1500, 55000)
            })
    
    df = pd.DataFrame(data)
    df['YIL_AY'] = df['DATE'].dt.strftime('%Y-%m')
    df['AY'] = df['DATE'].dt.month
    df['YIL'] = df['DATE'].dt.year
    df['CITY_NORMALIZED'] = df['CITY'].apply(normalize_city_name_fixed)
    
    return df

@st.cache_resource
def load_geojson_gpd():
    """GeoPandas ile GeoJSON yükle"""
    try:
        gdf = gpd.read_file("turkey.geojson")
        return gdf
    except:
        try:
            gdf = gpd.read_file("https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json")
            return gdf
        except Exception as e:
            st.warning(f"GeoJSON yüklenemedi, alternatif kullanılıyor: {e}")
            return create_sample_geojson()

def create_sample_geojson():
    """Örnek GeoJSON oluştur"""
    import geopandas as gpd
    from shapely.geometry import Polygon
    
    # Türkiye'nin kabaca koordinatları
    turkey_polygon = Polygon([
        (26, 42), (26, 36), (44, 36), (44, 42), (26, 42)
    ])
    
    gdf = gpd.GeoDataFrame({
        'name': ['Türkiye'],
        'geometry': [turkey_polygon]
    })
    
    return gdf

# =============================================================================
# ML ÖZELLİK MÜHENDİSLİĞİ
# =============================================================================

def create_advanced_features(df, target_col):
    """Gelişmiş ML özellikleri oluştur"""
    df = df.copy().sort_values('DATE').reset_index(drop=True)
    
    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12]:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
        df[f'rolling_min_{window}'] = df[target_col].rolling(window=window, min_periods=1).min()
        df[f'rolling_max_{window}'] = df[target_col].rolling(window=window, min_periods=1).max()
        df[f'rolling_median_{window}'] = df[target_col].rolling(window=window, min_periods=1).median()
    
    # Exponential moving averages
    for span in [3, 6, 12]:
        df[f'ema_{span}'] = df[target_col].ewm(span=span, adjust=False).mean()
    
    # Date features
    df['month'] = df['DATE'].dt.month
    df['quarter'] = df['DATE'].dt.quarter
    df['year'] = df['DATE'].dt.year
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    # Trend features
    df['trend'] = range(len(df))
    df['trend_squared'] = df['trend'] ** 2
    
    # Seasonality features
    df['is_q1'] = (df['quarter'] == 1).astype(int)
    df['is_q2'] = (df['quarter'] == 2).astype(int)
    df['is_q3'] = (df['quarter'] == 3).astype(int)
    df['is_q4'] = (df['quarter'] == 4).astype(int)
    
    # Year-over-year growth
    df['yoy_growth'] = df[target_col].pct_change(periods=12) * 100
    
    # Momentum features
    df['momentum_3'] = df[target_col] - df[target_col].shift(3)
    df['momentum_6'] = df[target_col] - df[target_col].shift(6)
    
    # Volatility
    df['volatility_6'] = df[target_col].rolling(window=6).std() / df[target_col].rolling(window=6).mean()
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df

def train_advanced_ml_models(df, target_col='PF_Satis', forecast_periods=6):
    """Gelişmiş ML modelleri ile tahmin"""
    if len(df) < 24:
        return None, None, None, "⚠️ Tahmin için yeterli veri yok (en az 24 ay gerekli)"
    
    try:
        # Özellik mühendisliği
        df_features = create_advanced_features(df, target_col)
        
        # Özellik sütunları
        feature_cols = [col for col in df_features.columns if col not in ['DATE', 'YIL_AY', target_col]]
        
        # Train/Test split (zaman serisi için)
        train_size = int(len(df_features) * 0.8)
        train_df = df_features.iloc[:train_size]
        test_df = df_features.iloc[train_size:]
        
        if len(test_df) == 0:
            test_df = df_features.iloc[-6:]  # Son 6 ayı test için kullan
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        # Modeller
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=0.1, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Metrikler
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'R2': r2,
                    'predictions': y_pred
                }
                
            except Exception as e:
                st.warning(f"{name} modeli eğitilemedi: {str(e)}")
                continue
        
        if not results:
            return None, None, None, "❌ Hiçbir model başarıyla eğitilemedi"
        
        # En iyi model (MAPE'e göre)
        valid_results = {k: v for k, v in results.items() if v['MAPE'] < 100}  # MAPE < 100 olanlar
        if not valid_results:
            valid_results = results
        
        best_model_name = min(valid_results.keys(), key=lambda x: valid_results[x]['MAPE'])
        best_model = results[best_model_name]['model']
        
        # Gelecek tahmini
        forecast_data = []
        last_row = df_features.iloc[-1:].copy()
        
        for i in range(forecast_periods):
            next_date = last_row['DATE'].values[0] + pd.DateOffset(months=1)
            
            # Özellikleri güncelle
            X_future = last_row[feature_cols]
            
            # Tahmin yap
            next_pred = best_model.predict(X_future)[0]
            next_pred = max(0, next_pred)  # Negatif olmamasını sağla
            
            forecast_data.append({
                'DATE': next_date,
                'YIL_AY': pd.to_datetime(next_date).strftime('%Y-%m'),
                'PF_Satis': next_pred,
                'Model': best_model_name,
                'Tahmin_Tipi': 'ML Tahmin'
            })
            
            # Son satırı güncelle (yeni tarih ve tahmin için)
            new_row = last_row.copy()
            new_row['DATE'] = next_date
            new_row[target_col] = next_pred
            
            # Lag değerlerini güncelle
            for lag in [1, 2, 3, 6, 12]:
                if f'lag_{lag}' in new_row.columns:
                    if lag == 1:
                        new_row[f'lag_{lag}'] = last_row[target_col].values[0]
                    else:
                        new_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}'].values[0] if f'lag_{lag-1}' in last_row.columns else 0
            
            # Diğer özellikleri güncelle
            new_row['month'] = pd.to_datetime(next_date).month
            new_row['quarter'] = pd.to_datetime(next_date).quarter
            new_row['year'] = pd.to_datetime(next_date).year
            new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
            new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)
            new_row['trend'] = last_row['trend'].values[0] + 1
            
            last_row = new_row
        
        forecast_df = pd.DataFrame(forecast_data)
        
        # Prophet modeli için tahmin (karşılaştırma için)
        try:
            prophet_df = df[['DATE', target_col]].copy()
            prophet_df.columns = ['ds', 'y']
            
            prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            prophet_model.fit(prophet_df)
            
            future = prophet_model.make_future_dataframe(periods=forecast_periods, freq='MS')
            prophet_forecast = prophet_model.predict(future)
            
            prophet_results = prophet_forecast[['ds', 'yhat']].tail(forecast_periods)
            prophet_results.columns = ['DATE', 'PF_Satis']
            prophet_results['Model'] = 'Prophet'
            prophet_results['Tahmin_Tipi'] = 'Prophet Tahmin'
            
            # Prophet tahminlerini de ekle
            all_forecasts = pd.concat([forecast_df, prophet_results], ignore_index=True)
            
        except Exception as e:
            st.warning(f"Prophet modeli çalıştırılamadı: {str(e)}")
            all_forecasts = forecast_df
        
        return results, best_model_name, all_forecasts, "✅ ML modelleri başarıyla eğitildi"
        
    except Exception as e:
        return None, None, None, f"❌ ML eğitimi hatası: {str(e)}"

# =============================================================================
# ANALİZ FONKSİYONLARI
# =============================================================================

def calculate_city_performance(df, product, date_filter=None):
    """Şehir bazlı performans"""
    cols = get_product_columns(product)
    
    # Sütunları kontrol et
    pf_col = cols['pf']
    rakip_col = cols['rakip']
    
    if pf_col not in df.columns or rakip_col not in df.columns:
        # Alternatif sütunları dene
        all_cols = df.columns.tolist()
        pf_candidates = [c for c in all_cols if product.upper() in c.upper() and 'DIGER' not in c.upper()]
        rakip_candidates = [c for c in all_cols if product.upper() in c.upper() and 'DIGER' in c.upper()]
        
        pf_col = pf_candidates[0] if pf_candidates else 'PF_Satis'
        rakip_col = rakip_candidates[0] if rakip_candidates else 'Rakip_Satis'
    
    df_filtered = df.copy()
    if date_filter:
        df_filtered = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & 
                                  (df_filtered['DATE'] <= date_filter[1])]
    
    city_perf = df_filtered.groupby(['CITY_NORMALIZED', 'REGION']).agg({
        pf_col: 'sum',
        rakip_col: 'sum'
    }).reset_index()
    
    city_perf.columns = ['City', 'Region', 'PF_Satis', 'Rakip_Satis']
    city_perf['Toplam_Pazar'] = city_perf['PF_Satis'] + city_perf['Rakip_Satis']
    city_perf['Pazar_Payi_%'] = safe_divide(city_perf['PF_Satis'], city_perf['Toplam_Pazar']) * 100
    city_perf['Bölge'] = city_perf['Region']
    
    return city_perf.sort_values('PF_Satis', ascending=False)

def calculate_territory_performance(df, product, date_filter=None):
    """Territory bazlı performans"""
    cols = get_product_columns(product)
    
    # Sütunları kontrol et
    pf_col = cols['pf']
    rakip_col = cols['rakip']
    
    if pf_col not in df.columns or rakip_col not in df.columns:
        # Alternatif sütunları dene
        all_cols = df.columns.tolist()
        pf_candidates = [c for c in all_cols if product.upper() in c.upper() and 'DIGER' not in c.upper()]
        rakip_candidates = [c for c in all_cols if product.upper() in c.upper() and 'DIGER' in c.upper()]
        
        pf_col = pf_candidates[0] if pf_candidates else 'PF_Satis'
        rakip_col = rakip_candidates[0] if rakip_candidates else 'Rakip_Satis'
    
    df_filtered = df.copy()
    if date_filter:
        df_filtered = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & 
                                  (df_filtered['DATE'] <= date_filter[1])]
    
    if 'CITY_NORMALIZED' not in df_filtered.columns and 'CITY' in df_filtered.columns:
        df_filtered['CITY_NORMALIZED'] = df_filtered['CITY']
    
    terr_perf = df_filtered.groupby(['TERRITORIES', 'REGION', 'CITY_NORMALIZED', 'MANAGER']).agg({
        pf_col: 'sum',
        rakip_col: 'sum'
    }).reset_index()
    
    terr_perf.columns = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 'Rakip_Satis']
    terr_perf['Toplam_Pazar'] = terr_perf['PF_Satis'] + terr_perf['Rakip_Satis']
    terr_perf['Pazar_Payi_%'] = safe_divide(terr_perf['PF_Satis'], terr_perf['Toplam_Pazar']) * 100
    
    total_pf = terr_perf['PF_Satis'].sum()
    terr_perf['Agirlik_%'] = safe_divide(terr_perf['PF_Satis'], total_pf) * 100
    terr_perf['Goreceli_Pazar_Payi'] = safe_divide(terr_perf['PF_Satis'], terr_perf['Rakip_Satis'])
    
    return terr_perf.sort_values('PF_Satis', ascending=False)

def calculate_time_series(df, product, territory=None, date_filter=None):
    """Zaman serisi"""
    cols = get_product_columns(product)
    
    # Sütunları kontrol et
    pf_col = cols['pf']
    rakip_col = cols['rakip']
    
    df_filtered = df.copy()
    if territory and territory != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['TERRITORIES'] == territory]
    
    if date_filter:
        df_filtered = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & 
                                   (df_filtered['DATE'] <= date_filter[1])]
    
    monthly = df_filtered.groupby('YIL_AY').agg({
        pf_col: 'sum',
        rakip_col: 'sum',
        'DATE': 'first'
    }).reset_index().sort_values('YIL_AY')
    
    monthly.columns = ['YIL_AY', 'PF_Satis', 'Rakip_Satis', 'DATE']
    monthly['Toplam_Pazar'] = monthly['PF_Satis'] + monthly['Rakip_Satis']
    monthly['Pazar_Payi_%'] = safe_divide(monthly['PF_Satis'], monthly['Toplam_Pazar']) * 100
    
    # Büyüme oranları
    monthly['PF_Buyume_%'] = monthly['PF_Satis'].pct_change() * 100
    monthly['Rakip_Buyume_%'] = monthly['Rakip_Satis'].pct_change() * 100
    monthly['Goreceli_Buyume_%'] = monthly['PF_Buyume_%'] - monthly['Rakip_Buyume_%']
    
    # Hareketli ortalamalar
    monthly['MA_3'] = monthly['PF_Satis'].rolling(window=3, min_periods=1).mean()
    monthly['MA_6'] = monthly['PF_Satis'].rolling(window=6, min_periods=1).mean()
    monthly['MA_12'] = monthly['PF_Satis'].rolling(window=12, min_periods=1).mean()
    
    # Volatilite
    monthly['Volatilite_3'] = monthly['PF_Satis'].rolling(window=3, min_periods=1).std()
    monthly['Volatilite_6'] = monthly['PF_Satis'].rolling(window=6, min_periods=1).std()
    
    return monthly

# =============================================================================
# GELİŞMİŞ ANALİZLER
# =============================================================================

def calculate_anomaly_detection(df, product):
    """Anomali tespiti"""
    ts_data = calculate_time_series(df, product)
    
    if len(ts_data) < 12:
        return pd.DataFrame()
    
    # Z-skor ile anomali tespiti
    ts_data['z_score'] = np.abs(stats.zscore(ts_data['PF_Satis'].fillna(0)))
    ts_data['Anomali'] = ts_data['z_score'] > 2.5
    
    # IQR method
    Q1 = ts_data['PF_Satis'].quantile(0.25)
    Q3 = ts_data['PF_Satis'].quantile(0.75)
    IQR = Q3 - Q1
    ts_data['IQR_Anomali'] = (ts_data['PF_Satis'] < (Q1 - 1.5 * IQR)) | (ts_data['PF_Satis'] > (Q3 + 1.5 * IQR))
    
    # Hareketli ortalama sapması
    ts_data['MA_Deviation'] = (ts_data['PF_Satis'] - ts_data['MA_6']) / ts_data['MA_6'] * 100
    ts_data['MA_Anomali'] = np.abs(ts_data['MA_Deviation']) > 50  # %50'den fazla sapma
    
    # Sonuçları birleştir
    ts_data['Toplam_Anomali'] = ts_data[['Anomali', 'IQR_Anomali', 'MA_Anomali']].any(axis=1)
    
    return ts_data[['YIL_AY', 'PF_Satis', 'MA_6', 'z_score', 'MA_Deviation', 'Toplam_Anomali']]

def calculate_seasonality_analysis(df, product):
    """Sezonallik analizi"""
    ts_data = calculate_time_series(df, product)
    
    if len(ts_data) < 24:
        return pd.DataFrame()
    
    # Aylık ortalama
    ts_data['Ay_No'] = pd.to_datetime(ts_data['DATE']).dt.month
    monthly_avg = ts_data.groupby('Ay_No')['PF_Satis'].mean().reset_index()
    monthly_avg.columns = ['Ay', 'Ortalama_Satis']
    
    # Yıllık büyüme
    yearly = ts_data.groupby(pd.to_datetime(ts_data['DATE']).dt.year)['PF_Satis'].sum().reset_index()
    yearly.columns = ['Yil', 'Toplam_Satis']
    yearly['Yillik_Buyume_%'] = yearly['Toplam_Satis'].pct_change() * 100
    
    # Mevsimsel indeks
    overall_avg = ts_data['PF_Satis'].mean()
    monthly_avg['Mevsimsel_Indeks'] = monthly_avg['Ortalama_Satis'] / overall_avg * 100
    
    return {
        'monthly': monthly_avg,
        'yearly': yearly,
        'seasonality_index': monthly_avg
    }

def calculate_market_concentration(df, product):
    """Pazar konsantrasyonu analizi (Herfindahl-Hirschman Index)"""
    terr_perf = calculate_territory_performance(df, product)
    
    if len(terr_perf) == 0:
        return 0
    
    # Pazar paylarını hesapla
    market_shares = terr_perf['PF_Satis'] / terr_perf['PF_Satis'].sum() * 100
    
    # HHI hesapla
    hhi = (market_shares ** 2).sum()
    
    # Konsantrasyon oranı (CR4)
    top4_share = terr_perf.nlargest(4, 'PF_Satis')['PF_Satis'].sum() / terr_perf['PF_Satis'].sum() * 100
    
    return {
        'HHI': hhi,
        'CR4': top4_share,
        'Market_Share_Top10': terr_perf.nlargest(10, 'PF_Satis')['Agirlik_%'].sum(),
        'Market_Share_Top20': terr_perf.nlargest(20, 'PF_Satis')['Agirlik_%'].sum()
    }

# =============================================================================
# GÖRSELLEŞTİRME FONKSİYONLARI
# =============================================================================

def create_advanced_forecast_chart(historical_df, forecast_df):
    """Gelişmiş tahmin grafiği"""
    fig = go.Figure()
    
    # Gerçek veri
    fig.add_trace(go.Scatter(
        x=historical_df['DATE'],
        y=historical_df['PF_Satis'],
        mode='lines+markers',
        name='Gerçek Satış',
        line=dict(color=PERFORMANCE_COLORS['success'], width=3),
        marker=dict(size=8, color='white'),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    
    # Hareketli ortalamalar
    if 'MA_6' in historical_df.columns:
        fig.add_trace(go.Scatter(
            x=historical_df['DATE'],
            y=historical_df['MA_6'],
            mode='lines',
            name='6 Aylık MA',
            line=dict(color='#F59E0B', width=2, dash='dash')
        ))
    
    # Tahminler
    if forecast_df is not None and len(forecast_df) > 0:
        # ML tahminleri
        ml_forecast = forecast_df[forecast_df['Model'] != 'Prophet']
        if len(ml_forecast) > 0:
            fig.add_trace(go.Scatter(
                x=ml_forecast['DATE'],
                y=ml_forecast['PF_Satis'],
                mode='lines+markers',
                name='ML Tahmin',
                line=dict(color='#3B82F6', width=3, dash='dash'),
                marker=dict(size=10, symbol='diamond', color='white')
            ))
        
        # Prophet tahminleri
        prophet_forecast = forecast_df[forecast_df['Model'] == 'Prophet']
        if len(prophet_forecast) > 0:
            fig.add_trace(go.Scatter(
                x=prophet_forecast['DATE'],
                y=prophet_forecast['PF_Satis'],
                mode='lines+markers',
                name='Prophet Tahmin',
                line=dict(color='#8B5CF6', width=3, dash='dot'),
                marker=dict(size=10, symbol='square', color='white')
            ))
    
    fig.update_layout(
        title='<b>Satış Trendi ve ML Tahminleri</b>',
        xaxis_title='<b>Tarih</b>',
        yaxis_title='<b>Satış</b>',
        height=500,
        hovermode='x unified',
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

def create_anomaly_chart(anomaly_df):
    """Anomali grafiği"""
    fig = go.Figure()
    
    # Normal veriler
    normal_data = anomaly_df[~anomaly_df['Toplam_Anomali']]
    fig.add_trace(go.Scatter(
        x=normal_data['YIL_AY'],
        y=normal_data['PF_Satis'],
        mode='markers',
        name='Normal',
        marker=dict(color=PERFORMANCE_COLORS['success'], size=10)
    ))
    
    # Anomaliler
    anomaly_data = anomaly_df[anomaly_df['Toplam_Anomali']]
    if len(anomaly_data) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_data['YIL_AY'],
            y=anomaly_data['PF_Satis'],
            mode='markers+text',
            name='Anomali',
            marker=dict(color=PERFORMANCE_COLORS['danger'], size=15, symbol='x'),
            text='⚠️',
            textposition='top center'
        ))
    
    fig.update_layout(
        title='<b>Satış Anomalileri</b>',
        xaxis_title='<b>Ay</b>',
        yaxis_title='<b>Satış</b>',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0')
    )
    
    return fig

def create_seasonality_chart(seasonality_data):
    """Sezonallik grafiği"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=seasonality_data['monthly']['Ay'],
        y=seasonality_data['monthly']['Mevsimsel_Indeks'],
        name='Mevsimsel İndeks',
        marker_color='#3B82F6',
        text=seasonality_data['monthly']['Mevsimsel_Indeks'].round(1),
        textposition='outside'
    ))
    
    fig.add_hline(y=100, line_dash="dash", line_color="white", opacity=0.5)
    
    fig.update_layout(
        title='<b>Aylık Mevsimsel İndeks</b>',
        xaxis_title='<b>Ay</b>',
        yaxis_title='<b>İndeks (Ortalama=100)</b>',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0')
    )
    
    return fig

# =============================================================================
# ANA UYGULAMA
# =============================================================================

def main():
    # Başlık
    st.markdown('<h1 class="main-header">🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #94a3b8; margin-bottom: 3rem;">'
                'GERÇEK ML Tahminleme • Modern Harita Görselleştirme • Rakip Analizi • BCG Matrix'
                '</div>', unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown('<div style="background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%); '
                   'padding: 1rem; border-radius: 12px; margin-bottom: 2rem;">'
                   '<h3 style="color: white; margin: 0; text-align: center;">📂 VERİ YÜKLEME</h3>'
                   '</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Excel Dosyası Yükleyin", type=['xlsx', 'xls'])
        
        if not uploaded_file:
            st.info("👈 Lütfen sol taraftan Excel dosyasını yükleyin")
            st.stop()
        
        try:
            df = load_excel_data(uploaded_file)
            gdf = load_geojson_gpd()
            st.success(f"✅ **{len(df):,}** satır veri yüklendi")
            
            # Mevcut ürünleri tespit et
            available_products = get_available_products(df)
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.stop()
        
        st.markdown("---")
        
        # Ürün Seçimi
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">💊 ÜRÜN SEÇİMİ</h4>', unsafe_allow_html=True)
        selected_product = st.selectbox("", available_products, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tarih Aralığı
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">📅 TARİH ARALIĞI</h4>', unsafe_allow_html=True)
        
        min_date = df['DATE'].min()
        max_date = df['DATE'].max()
        
        date_option = st.selectbox("Dönem Seçin", ["Tüm Veriler", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl", "Son 2 Yıl", "Özel Aralık"])
        
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
        elif date_option == "Son 2 Yıl":
            start_date = max_date - pd.DateOffset(years=2)
            date_filter = (start_date, max_date)
        else:
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input("Başlangıç", min_date, min_value=min_date, max_value=max_date)
            with col_date2:
                end_date = st.date_input("Bitiş", max_date, min_value=min_date, max_value=max_date)
            date_filter = (pd.to_datetime(start_date), pd.to_datetime(end_date))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Filtreler
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">🔍 FİLTRELER</h4>', unsafe_allow_html=True)
        
        territories = ["TÜMÜ"] + sorted(df['TERRITORIES'].unique())
        selected_territory = st.selectbox("Territory", territories)
        
        regions = ["TÜMÜ"] + sorted(df['REGION'].unique())
        selected_region = st.selectbox("Bölge", regions)
        
        managers = ["TÜMÜ"] + sorted(df['MANAGER'].unique())
        selected_manager = st.selectbox("Manager", managers)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Veri filtreleme
        df_filtered = df.copy()
        if selected_territory != "TÜMÜ":
            df_filtered = df_filtered[df_filtered['TERRITORIES'] == selected_territory]
        if selected_region != "TÜMÜ":
            df_filtered = df_filtered[df_filtered['REGION'] == selected_region]
        if selected_manager != "TÜMÜ":
            df_filtered = df_filtered[df_filtered['MANAGER'] == selected_manager]
        
        st.markdown("---")
        
        # ML Ayarları
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">🤖 ML AYARLARI</h4>', unsafe_allow_html=True)
        
        forecast_months = st.slider("Tahmin Periyodu (Ay)", 1, 12, 6)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ANA İÇERİK - TAB'LER
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Genel Bakış",
        "🗺️ Harita Analizi",
        "🏢 Territory Analizi",
        "📈 Zaman Serisi",
        "🤖 ML Tahminleme",
        "📊 Gelişmiş Analiz",
        "📉 Anomali Tespiti",
        "📥 Raporlar"
    ])
    
    # TAB 1: GENEL BAKIŞ
    with tab1:
        st.header("📊 Genel Performans Özeti")
        
        cols = get_product_columns(selected_product)
        
        if date_filter:
            df_period = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & (df_filtered['DATE'] <= date_filter[1])]
        else:
            df_period = df_filtered
        
        # Sütun adlarını kontrol et
        pf_col = cols['pf']
        rakip_col = cols['rakip']
        
        # Eğer sütun yoksa alternatif bul
        if pf_col not in df_period.columns:
            possible_cols = [c for c in df_period.columns if selected_product.upper() in c.upper() and 'DIGER' not in c.upper()]
            pf_col = possible_cols[0] if possible_cols else 'PF_Satis'
        
        if rakip_col not in df_period.columns:
            possible_cols = [c for c in df_period.columns if selected_product.upper() in c.upper() and 'DIGER' in c.upper()]
            rakip_col = possible_cols[0] if possible_cols else 'Rakip_Satis'
        
        # Metrikler
        total_pf = df_period[pf_col].sum() if pf_col in df_period.columns else 0
        total_rakip = df_period[rakip_col].sum() if rakip_col in df_period.columns else 0
        total_market = total_pf + total_rakip
        market_share = (total_pf / total_market * 100) if total_market > 0 else 0
        active_territories = df_period['TERRITORIES'].nunique()
        avg_monthly_pf = total_pf / df_period['YIL_AY'].nunique() if df_period['YIL_AY'].nunique() > 0 else 0
        
        # Pazar konsantrasyonu
        market_concentration = calculate_market_concentration(df_period, selected_product)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💊 PF Satış", f"{total_pf:,.0f}", f"{avg_monthly_pf:,.0f}/ay")
        with col2:
            st.metric("🏪 Toplam Pazar", f"{total_market:,.0f}", f"{total_rakip:,.0f} rakip")
        with col3:
            st.metric("📊 Pazar Payı", f"%{market_share:.1f}", 
                     f"%{100-market_share:.1f} rakip")
        with col4:
            st.metric("🏢 Active Territory", active_territories, 
                     f"{df_period['MANAGER'].nunique()} manager")
        
        st.markdown("---")
        
        # Pazar Konsantrasyonu
        st.subheader("📊 Pazar Konsantrasyonu Analizi")
        
        col_conc1, col_conc2, col_conc3, col_conc4 = st.columns(4)
        
        with col_conc1:
            hhi = market_concentration.get('HHI', 0)
            hhi_status = "Yüksek" if hhi > 2500 else "Orta" if hhi > 1500 else "Düşük"
            hhi_color = "#EF4444" if hhi > 2500 else "#F59E0B" if hhi > 1500 else "#10B981"
            st.metric("🏛️ HHI İndeksi", f"{hhi:,.0f}", hhi_status)
        
        with col_conc2:
            cr4 = market_concentration.get('CR4', 0)
            st.metric("🏆 CR4 (%)", f"{cr4:.1f}%")
        
        with col_conc3:
            top10_share = market_concentration.get('Market_Share_Top10', 0)
            st.metric("👑 Top 10 Payı", f"{top10_share:.1f}%")
        
        with col_conc4:
            top20_share = market_concentration.get('Market_Share_Top20', 0)
            st.metric("👥 Top 20 Payı", f"{top20_share:.1f}%")
        
        st.markdown("---")
        
        # Top 10 Territory
        st.subheader("🏆 Top 10 Territory Performansı")
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        
        if len(terr_perf) > 0:
            top10 = terr_perf.head(10)
            
            col_chart1, col_chart2 = st.columns([2, 1])
            
            with col_chart1:
                fig_top10 = go.Figure()
                
                fig_top10.add_trace(go.Bar(
                    x=top10['Territory'],
                    y=top10['PF_Satis'],
                    name='PF Satış',
                    marker_color=PERFORMANCE_COLORS['success'],
                    text=top10['PF_Satis'].apply(lambda x: f'{x:,.0f}'),
                    textposition='outside'
                ))
                
                fig_top10.update_layout(
                    title='<b>Top 10 Territory - PF Satış</b>',
                    xaxis_title='<b>Territory</b>',
                    yaxis_title='<b>Satış</b>',
                    height=500,
                    xaxis=dict(tickangle=-45),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig_top10, use_container_width=True)
            
            with col_chart2:
                fig_pie = px.pie(
                    top10,
                    values='PF_Satis',
                    names='Territory',
                    title='<b>Top 10 Dağılımı</b>',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                fig_pie.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("⚠️ Territory performans verisi bulunamadı")
    
    # TAB 2: HARİTA ANALİZİ
    with tab2:
        st.header("🗺️ Türkiye Harita Analizi")
        
        city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
        
        if len(city_data) == 0:
            st.warning("⚠️ Şehir verisi bulunamadı")
        else:
            # Harita için veriyi hazırla
            if gdf is not None:
                # GeoJSON ile birleştir
                gdf['name_upper'] = gdf['name'].str.upper()
                
                # Şehir adlarını normalleştir
                city_data['City_Normalized'] = city_data['City'].apply(normalize_city_name_fixed)
                
                # Birleştirme
                merged = gdf.merge(city_data, left_on='name_upper', right_on='City_Normalized', how='left')
                
                # NaN'leri doldur
                merged['PF_Satis'] = merged['PF_Satis'].fillna(0)
                merged['Pazar_Payi_%'] = merged['Pazar_Payi_%'].fillna(0)
                
                # Harita oluştur
                fig = px.choropleth_mapbox(
                    merged,
                    geojson=merged.geometry.__geo_interface__,
                    locations=merged.index,
                    color='PF_Satis',
                    color_continuous_scale="Viridis",
                    range_color=(0, merged['PF_Satis'].max()),
                    mapbox_style="carto-positron",
                    zoom=5,
                    center={"lat": 39.0, "lon": 35.0},
                    opacity=0.7,
                    hover_name='name',
                    hover_data={
                        'PF_Satis': ':,.0f',
                        'Pazar_Payi_%': ':.1f',
                        'Region': True
                    },
                    title=f'<b>{selected_product} - Şehir Bazlı Satış Dağılımı</b>'
                )
                
                fig.update_layout(
                    height=700,
                    margin={"r":0,"t":40,"l":0,"b":0},
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ GeoJSON harita verisi yüklenemedi")
        
        # Şehir Performans Tablosu
        st.markdown("---")
        st.subheader("📋 Şehir Performans Detayları")
        
        if len(city_data) > 0:
            city_display = city_data.sort_values('PF_Satis', ascending=False).head(20)
            city_display.index = range(1, len(city_display) + 1)
            
            st.dataframe(
                city_display[['City', 'Region', 'PF_Satis', 'Toplam_Pazar', 'Pazar_Payi_%']].style.format({
                    'PF_Satis': '{:,.0f}',
                    'Toplam_Pazar': '{:,.0f}',
                    'Pazar_Payi_%': '{:.1f}%'
                }),
                use_container_width=True,
                height=400
            )
    
    # TAB 3: TERRITORY ANALİZİ
    with tab3:
        st.header("🏢 Territory Bazlı Detaylı Analiz")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        
        if len(terr_perf) == 0:
            st.warning("⚠️ Territory verisi bulunamadı")
        else:
            # Filtreleme
            col_filter1, col_filter2 = st.columns(2)
            
            with col_filter1:
                sort_by = st.selectbox(
                    "Sıralama Kriteri",
                    ['PF_Satis', 'Pazar_Payi_%', 'Toplam_Pazar', 'Agirlik_%'],
                    format_func=lambda x: {
                        'PF_Satis': 'PF Satış',
                        'Pazar_Payi_%': 'Pazar Payı',
                        'Toplam_Pazar': 'Toplam Pazar',
                        'Agirlik_%': 'Ağırlık %'
                    }[x]
                )
            
            with col_filter2:
                show_n = st.slider("Gösterilecek Kayıt Sayısı", 10, 100, 25)
            
            terr_sorted = terr_perf.sort_values(sort_by, ascending=False).head(show_n)
            
            # Grafikler
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig_bar = px.bar(
                    terr_sorted,
                    x='Territory',
                    y=['PF_Satis', 'Rakip_Satis'],
                    title='<b>PF vs Rakip Satış</b>',
                    barmode='group',
                    color_discrete_map={
                        'PF_Satis': PERFORMANCE_COLORS['success'],
                        'Rakip_Satis': PERFORMANCE_COLORS['danger']
                    }
                )
                
                fig_bar.update_layout(
                    height=500,
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_chart2:
                fig_scatter = px.scatter(
                    terr_sorted,
                    x='PF_Satis',
                    y='Pazar_Payi_%',
                    size='Toplam_Pazar',
                    color='Region',
                    hover_name='Territory',
                    title='<b>Territory Performans Haritası</b>',
                    size_max=50
                )
                
                fig_scatter.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Detaylı Tablo
            st.markdown("---")
            st.subheader(f"📋 Territory Detayları (Top {show_n})")
            
            terr_display = terr_sorted.copy()
            terr_display.index = range(1, len(terr_display) + 1)
            
            st.dataframe(
                terr_display.style.format({
                    'PF_Satis': '{:,.0f}',
                    'Rakip_Satis': '{:,.0f}',
                    'Toplam_Pazar': '{:,.0f}',
                    'Pazar_Payi_%': '{:.1f}%',
                    'Agirlik_%': '{:.1f}%',
                    'Goreceli_Pazar_Payi': '{:.2f}'
                }),
                use_container_width=True,
                height=500
            )
    
    # TAB 4: ZAMAN SERİSİ
    with tab4:
        st.header("📈 Zaman Serisi Analizi")
        
        territory_for_ts = st.selectbox(
            "Territory Seçin",
            ["TÜMÜ"] + sorted(df_filtered['TERRITORIES'].unique()),
            key='ts_territory'
        )
        
        monthly_df = calculate_time_series(df_filtered, selected_product, territory_for_ts, date_filter)
        
        if len(monthly_df) == 0:
            st.warning("⚠️ Zaman serisi verisi bulunamadı")
        else:
            # Özet Metrikler
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_pf = monthly_df['PF_Satis'].mean()
                st.metric("📊 Ort. Aylık PF", f"{avg_pf:,.0f}")
            
            with col2:
                avg_growth = monthly_df['PF_Buyume_%'].mean()
                st.metric("📈 Ort. Büyüme", f"%{avg_growth:.1f}")
            
            with col3:
                avg_share = monthly_df['Pazar_Payi_%'].mean()
                st.metric("🎯 Ort. Pazar Payı", f"%{avg_share:.1f}")
            
            with col4:
                volatility = monthly_df['Volatilite_6'].mean() / monthly_df['PF_Satis'].mean() * 100
                st.metric("📉 Ort. Volatilite", f"%{volatility:.1f}")
            
            st.markdown("---")
            
            # Grafikler
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig_ts = go.Figure()
                
                fig_ts.add_trace(go.Scatter(
                    x=monthly_df['DATE'],
                    y=monthly_df['PF_Satis'],
                    mode='lines+markers',
                    name='PF Satış',
                    line=dict(color=PERFORMANCE_COLORS['success'], width=3)
                ))
                
                fig_ts.add_trace(go.Scatter(
                    x=monthly_df['DATE'],
                    y=monthly_df['MA_6'],
                    mode='lines',
                    name='6 Aylık MA',
                    line=dict(color='#F59E0B', width=2, dash='dash')
                ))
                
                fig_ts.update_layout(
                    title='<b>Satış Trendi</b>',
                    xaxis_title='<b>Tarih</b>',
                    yaxis_title='<b>Satış</b>',
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig_ts, use_container_width=True)
            
            with col_chart2:
                fig_growth = go.Figure()
                
                fig_growth.add_trace(go.Scatter(
                    x=monthly_df['DATE'],
                    y=monthly_df['PF_Buyume_%'],
                    mode='lines+markers',
                    name='PF Büyüme',
                    line=dict(color=PERFORMANCE_COLORS['success'], width=3)
                ))
                
                fig_growth.add_trace(go.Scatter(
                    x=monthly_df['DATE'],
                    y=monthly_df['Rakip_Buyume_%'],
                    mode='lines+markers',
                    name='Rakip Büyüme',
                    line=dict(color=PERFORMANCE_COLORS['danger'], width=3)
                ))
                
                fig_growth.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                
                fig_growth.update_layout(
                    title='<b>Büyüme Oranları</b>',
                    xaxis_title='<b>Tarih</b>',
                    yaxis_title='<b>Büyüme (%)</b>',
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig_growth, use_container_width=True)
            
            # Veri Tablosu
            st.markdown("---")
            st.subheader("📋 Aylık Performans Detayları")
            
            monthly_display = monthly_df.copy()
            monthly_display.index = range(1, len(monthly_display) + 1)
            
            st.dataframe(
                monthly_display[['YIL_AY', 'PF_Satis', 'Rakip_Satis', 'Pazar_Payi_%', 'PF_Buyume_%', 'MA_6']].style.format({
                    'PF_Satis': '{:,.0f}',
                    'Rakip_Satis': '{:,.0f}',
                    'Pazar_Payi_%': '{:.1f}%',
                    'PF_Buyume_%': '{:.1f}%',
                    'MA_6': '{:,.0f}'
                }),
                use_container_width=True,
                height=400
            )
    
    # TAB 5: ML TAHMİNLEME
    with tab5:
        st.header("🤖 Machine Learning Tahminleme")
        
        # Zaman serisi verisini hazırla
        monthly_df = calculate_time_series(df_filtered, selected_product, None, date_filter)
        
        if len(monthly_df) < 12:
            st.warning(f"⚠️ Tahmin için yeterli veri yok (en az 12 ay gerekli, mevcut: {len(monthly_df)})")
        else:
            with st.spinner("ML modelleri eğitiliyor..."):
                ml_results, best_model_name, forecast_df, message = train_advanced_ml_models(
                    monthly_df, 
                    'PF_Satis', 
                    forecast_months
                )
            
            st.markdown(f'<div class="alert-success">{message}</div>', unsafe_allow_html=True)
            
            if ml_results is not None:
                # Model Performansı
                st.subheader("📊 Model Performans Karşılaştırması")
                
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
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.dataframe(
                        perf_df.style.format({
                            'MAE': '{:,.0f}',
                            'RMSE': '{:,.0f}',
                            'MAPE (%)': '{:.2f}',
                            'R²': '{:.3f}'
                        }).background_gradient(subset=['MAPE (%)'], cmap='RdYlGn_r'),
                        use_container_width=True
                    )
                
                with col2:
                    best_mape = ml_results[best_model_name]['MAPE']
                    
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
                    <div style="background: rgba(30,41,59,0.8); padding: 1.5rem; border-radius: 10px; border: 2px solid {color};">
                        <h3 style="color: white; margin-top: 0;">🏆 En İyi Model</h3>
                        <h2 style="color: {color}; margin: 0.5rem 0;">{best_model_name}</h2>
                        <p style="color: #94a3b8; margin: 0.5rem 0;">MAPE: <span style="color: {color}; font-weight: bold;">{best_mape:.2f}%</span></p>
                        <p style="color: #e2e8f0; margin: 0.5rem 0;">Güven: <span style="color: {color}; font-weight: bold;">{confidence}</span></p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Tahmin Grafiği
                st.subheader("📈 Satış Tahminleri")
                
                forecast_chart = create_advanced_forecast_chart(monthly_df, forecast_df)
                st.plotly_chart(forecast_chart, use_container_width=True)
                
                # Tahmin Detayları
                st.markdown("---")
                st.subheader("📋 Tahmin Detayları")
                
                if forecast_df is not None and len(forecast_df) > 0:
                    forecast_display = forecast_df.copy()
                    forecast_display.index = range(1, len(forecast_display) + 1)
                    
                    st.dataframe(
                        forecast_display[['YIL_AY', 'PF_Satis', 'Model']].style.format({
                            'PF_Satis': '{:,.0f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Toplam tahmin
                    total_forecast = forecast_display['PF_Satis'].sum()
                    historical_avg = monthly_df['PF_Satis'].mean()
                    
                    col_f1, col_f2 = st.columns(2)
                    
                    with col_f1:
                        st.metric("💰 Tahmini Toplam Satış", f"{total_forecast:,.0f}")
                    
                    with col_f2:
                        growth_vs_avg = (total_forecast / forecast_months - historical_avg) / historical_avg * 100
                        st.metric("📈 Ort. Aylık Tahmin", f"{total_forecast/forecast_months:,.0f}", 
                                 f"%{growth_vs_avg:.1f} vs geçmiş ort.")
    
    # TAB 6: GELİŞMİŞ ANALİZ
    with tab6:
        st.header("📊 Gelişmiş Analizler")
        
        # Sezonallik Analizi
        st.subheader("📅 Sezonallik Analizi")
        
        seasonality_data = calculate_seasonality_analysis(df_filtered, selected_product)
        
        if seasonality_data and len(seasonality_data['monthly']) > 0:
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                fig_season = create_seasonality_chart(seasonality_data)
                st.plotly_chart(fig_season, use_container_width=True)
            
            with col_s2:
                st.markdown("#### 🗓️ Aylık Performans")
                monthly_display = seasonality_data['monthly'].copy()
                monthly_display.index = monthly_display['Ay']
                
                st.dataframe(
                    monthly_display[['Ortalama_Satis', 'Mevsimsel_Indeks']].style.format({
                        'Ortalama_Satis': '{:,.0f}',
                        'Mevsimsel_Indeks': '{:.1f}'
                    }),
                    use_container_width=True,
                    height=350
                )
            
            # Yıllık Büyüme
            st.markdown("---")
            st.subheader("📈 Yıllık Büyüme Analizi")
            
            if len(seasonality_data['yearly']) > 1:
                fig_yearly = go.Figure()
                
                fig_yearly.add_trace(go.Bar(
                    x=seasonality_data['yearly']['Yil'],
                    y=seasonality_data['yearly']['Toplam_Satis'],
                    name='Toplam Satış',
                    marker_color='#3B82F6',
                    text=seasonality_data['yearly']['Toplam_Satis'].apply(lambda x: f'{x:,.0f}'),
                    textposition='outside'
                ))
                
                fig_yearly.add_trace(go.Scatter(
                    x=seasonality_data['yearly']['Yil'],
                    y=seasonality_data['yearly']['Yillik_Buyume_%'],
                    name='Yıllık Büyüme',
                    yaxis='y2',
                    line=dict(color='#10B981', width=3)
                ))
                
                fig_yearly.update_layout(
                    title='<b>Yıllık Satış ve Büyüme</b>',
                    xaxis_title='<b>Yıl</b>',
                    yaxis_title='<b>Toplam Satış</b>',
                    yaxis2=dict(
                        title='<b>Büyüme (%)</b>',
                        overlaying='y',
                        side='right'
                    ),
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig_yearly, use_container_width=True)
        
        # Korelasyon Analizi
        st.markdown("---")
        st.subheader("🔗 Korelasyon Analizi")
        
        # Territory performansından sayısal kolonları al
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        
        if len(terr_perf) > 5:
            numeric_cols = ['PF_Satis', 'Rakip_Satis', 'Toplam_Pazar', 'Pazar_Payi_%', 'Agirlik_%']
            available_cols = [col for col in numeric_cols if col in terr_perf.columns]
            
            if len(available_cols) >= 2:
                corr_matrix = terr_perf[available_cols].corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig_corr.update_layout(
                    title='<b>Korelasyon Matrisi</b>',
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
    
    # TAB 7: ANOMALİ TESPİTİ
    with tab7:
        st.header("📉 Anomali Tespiti")
        
        anomaly_df = calculate_anomaly_detection(df_filtered, selected_product)
        
        if len(anomaly_df) == 0:
            st.warning("⚠️ Anomali tespiti için yeterli veri yok")
        else:
            # Anomali istatistikleri
            total_anomalies = anomaly_df['Toplam_Anomali'].sum()
            anomaly_rate = total_anomalies / len(anomaly_df) * 100
            
            col_a1, col_a2, col_a3 = st.columns(3)
            
            with col_a1:
                st.metric("⚠️ Toplam Anomali", total_anomalies)
            
            with col_a2:
                st.metric("📊 Anomali Oranı", f"%{anomaly_rate:.1f}")
            
            with col_a3:
                last_anomaly = anomaly_df[anomaly_df['Toplam_Anomali']].tail(1)
                if len(last_anomaly) > 0:
                    st.metric("🕐 Son Anomali", last_anomaly['YIL_AY'].iloc[0])
                else:
                    st.metric("🕐 Son Anomali", "Yok")
            
            st.markdown("---")
            
            # Anomali Grafiği
            st.subheader("📈 Anomali Dağılımı")
            
            anomaly_chart = create_anomaly_chart(anomaly_df)
            st.plotly_chart(anomaly_chart, use_container_width=True)
            
            # Anomali Detayları
            st.markdown("---")
            st.subheader("📋 Anomali Detayları")
            
            anomaly_details = anomaly_df[anomaly_df['Toplam_Anomali']].copy()
            
            if len(anomaly_details) > 0:
                anomaly_details.index = range(1, len(anomaly_details) + 1)
                
                st.dataframe(
                    anomaly_details[['YIL_AY', 'PF_Satis', 'MA_6', 'z_score', 'MA_Deviation']].style.format({
                        'PF_Satis': '{:,.0f}',
                        'MA_6': '{:,.0f}',
                        'z_score': '{:.2f}',
                        'MA_Deviation': '{:.1f}%'
                    }),
                    use_container_width=True
                )
                
                # Anomali açıklamaları
                st.markdown("---")
                st.subheader("📝 Anomali Açıklamaları")
                
                for idx, row in anomaly_details.iterrows():
                    explanations = []
                    
                    if row['z_score'] > 2.5:
                        explanations.append(f"Z-skor çok yüksek ({row['z_score']:.2f})")
                    
                    if abs(row['MA_Deviation']) > 50:
                        explanations.append(f"Hareketli ortalamadan %{abs(row['MA_Deviation']):.1f} sapma")
                    
                    if explanations:
                        st.markdown(f'''
                        <div class="alert-warning" style="margin-bottom: 1rem;">
                            <strong>{row['YIL_AY']}</strong> - {row['PF_Satis']:,.0f} satış<br>
                            {' • '.join(explanations)}
                        </div>
                        ''', unsafe_allow_html=True)
            else:
                st.success("🎉 Hiç anomali tespit edilmedi!")
    
    # TAB 8: RAPORLAR
    with tab8:
        st.header("📥 Rapor İndirme")
        
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.7); padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
            <h3 style="color: #e2e8f0; margin-top: 0;">📊 Detaylı Excel Raporu</h3>
            <p style="color: #94a3b8; margin-bottom: 1.5rem;">
                Tüm analizlerinizi içeren kapsamlı bir Excel raporu oluşturun. 
                Rapor aşağıdaki sayfaları içerecektir:
            </p>
            <ul style="color: #cbd5e1; margin-left: 1.5rem;">
                <li>Genel Performans Özeti</li>
                <li>Territory Performans Analizi</li>
                <li>Şehir Bazlı Dağılım</li>
                <li>Zaman Serisi Analizi</li>
                <li>ML Tahmin Sonuçları</li>
                <li>Anomali Tespit Raporu</li>
                <li>Sezonallik Analizi</li>
                <li>Pazar Konsantrasyonu</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Excel Raporu Oluştur", type="primary", use_container_width=True):
            with st.spinner("Rapor hazırlanıyor..."):
                try:
                    # Tüm analizleri hesapla
                    terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
                    city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
                    monthly_df = calculate_time_series(df_filtered, selected_product, None, date_filter)
                    
                    # ML tahmini
                    ml_results, best_model_name, forecast_df, _ = train_advanced_ml_models(
                        monthly_df, 'PF_Satis', 6
                    )
                    
                    # Anomali tespiti
                    anomaly_df = calculate_anomaly_detection(df_filtered, selected_product)
                    
                    # Sezonallik analizi
                    seasonality_data = calculate_seasonality_analysis(df_filtered, selected_product)
                    
                    # Pazar konsantrasyonu
                    market_conc = calculate_market_concentration(df_filtered, selected_product)
                    
                    # Excel dosyası oluştur
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Genel Özet
                        summary_data = pd.DataFrame({
                            'Metrik': ['Ürün', 'Dönem', 'Toplam PF Satış', 'Toplam Pazar', 
                                      'Pazar Payı', 'Territory Sayısı', 'Şehir Sayısı',
                                      'HHI İndeksi', 'CR4 (%)'],
                            'Değer': [
                                selected_product,
                                date_option,
                                f"{terr_perf['PF_Satis'].sum():,.0f}",
                                f"{terr_perf['Toplam_Pazar'].sum():,.0f}",
                                f"{(terr_perf['PF_Satis'].sum() / terr_perf['Toplam_Pazar'].sum() * 100):.1f}%" if terr_perf['Toplam_Pazar'].sum() > 0 else "0%",
                                len(terr_perf),
                                len(city_data),
                                f"{market_conc.get('HHI', 0):,.0f}",
                                f"{market_conc.get('CR4', 0):.1f}%"
                            ]
                        })
                        summary_data.to_excel(writer, sheet_name='Özet', index=False)
                        
                        # Diğer sayfalar
                        terr_perf.to_excel(writer, sheet_name='Territory Performans', index=False)
                        city_data.to_excel(writer, sheet_name='Şehir Analizi', index=False)
                        monthly_df.to_excel(writer, sheet_name='Zaman Serisi', index=False)
                        
                        if forecast_df is not None and len(forecast_df) > 0:
                            forecast_df.to_excel(writer, sheet_name='ML Tahminler', index=False)
                        
                        if len(anomaly_df) > 0:
                            anomaly_df.to_excel(writer, sheet_name='Anomaliler', index=False)
                        
                        if seasonality_data and len(seasonality_data['monthly']) > 0:
                            seasonality_data['monthly'].to_excel(writer, sheet_name='Sezonallik', index=False)
                        
                        # ML Model Performansı
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
                            pd.DataFrame(perf_data).to_excel(writer, sheet_name='ML Performans', index=False)
                    
                    st.success("✅ Rapor hazır!")
                    
                    # İndirme butonu
                    st.download_button(
                        label="💾 Excel Raporunu İndir",
                        data=output.getvalue(),
                        file_name=f"ticari_portfoy_raporu_{selected_product}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Rapor oluşturma hatası: {str(e)}")

if __name__ == "__main__":
    main()
