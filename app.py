"""
🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ
McKinsey Tema ile Territory Bazlı Performans, ML Tahminleme ve Türkiye Haritası
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
    page_title="Ticari Portföy Analizi | McKinsey Style",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# MCKINSEY CSS TEMA
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #FFFFFF;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        padding: 2rem 0;
        color: #1A1A1A;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        border-bottom: 1px solid #E0E0E0;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666666;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A1A1A;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 600;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="metric-container"] {
        background: #FFFFFF;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #0066CC;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-top: 1px solid #F0F0F0;
        border-right: 1px solid #F0F0F0;
        border-bottom: 1px solid #F0F0F0;
        transition: all 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #F8F9FA;
        border-bottom: 1px solid #E0E0E0;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #666666;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        background: transparent;
        border: none;
        margin: 0;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #0066CC;
        background: #F0F7FF;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #0066CC;
        background: #FFFFFF;
        border-bottom: 2px solid #0066CC;
        font-weight: 600;
    }
    
    h1, h2, h3, h4 {
        color: #1A1A1A !important;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h1 {
        font-size: 2.2rem;
        border-bottom: 2px solid #0066CC;
        padding-bottom: 0.5rem;
    }
    
    h2 {
        font-size: 1.8rem;
        color: #333333 !important;
    }
    
    h3 {
        font-size: 1.4rem;
        color: #444444 !important;
    }
    
    h4 {
        font-size: 1.1rem;
        color: #555555 !important;
    }
    
    p, span, div, label {
        color: #333333;
    }
    
    .stButton>button {
        background: #0066CC;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 4px;
        font-weight: 500;
        transition: all 0.2s ease;
        font-size: 0.9rem;
    }
    
    .stButton>button:hover {
        background: #0052A3;
        box-shadow: 0 2px 4px rgba(0,102,204,0.2);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F1F1F1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #B0B0B0;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #909090;
    }
    
    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background: #F8F9FA;
        border-right: 1px solid #E0E0E0;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #1A1A1A !important;
        border-bottom: 2px solid #0066CC;
        padding-bottom: 0.5rem;
    }
    
    /* Input field styling */
    .stSelectbox, .stSlider, .stRadio, .stDateInput {
        background: #FFFFFF;
        border: 1px solid #D0D0D0;
        border-radius: 4px;
    }
    
    .stSelectbox>div>div {
        border: none !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: #0066CC;
    }
    
    /* McKinsey card styling */
    .mck-card {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #E0E0E0;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        transition: all 0.2s ease;
    }
    
    .mck-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .mck-card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1A1A1A;
        margin-bottom: 1rem;
        border-bottom: 1px solid #F0F0F0;
        padding-bottom: 0.5rem;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    .status-excellent { background: #E6F4EA; color: #0D652D; }
    .status-good { background: #D9E8FB; color: #0052A3; }
    .status-fair { background: #FFF4E5; color: #996A00; }
    .status-poor { background: #FCE8E8; color: #A4262C; }
    
    /* Data table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    .dataframe thead th {
        background: #F8F9FA;
        color: #333333;
        font-weight: 600;
        border-bottom: 2px solid #E0E0E0;
    }
    
    /* Alert boxes */
    .info-box {
        background: #F0F7FF;
        border-left: 4px solid #0066CC;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #FFF4E5;
        border-left: 4px solid #FF8C00;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #E6F4EA;
        border-left: 4px solid #0D652D;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* McKinsey color classes */
    .mck-blue { color: #0066CC; }
    .mck-dark-blue { color: #004080; }
    .mck-green { color: #0D652D; }
    .mck-orange { color: #FF8C00; }
    .mck-red { color: #A4262C; }
    .mck-gray { color: #666666; }
    
    .bg-mck-blue { background-color: #0066CC; }
    .bg-mck-light-blue { background-color: #F0F7FF; }
    .bg-mck-gray { background-color: #F8F9FA; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MCKINSEY RENK PALETLERİ
# =============================================================================

# McKinsey renkleri
MCKINSEY_COLORS = {
    "primary": "#0066CC",
    "secondary": "#0D652D",
    "accent": "#FF8C00",
    "dark": "#1A1A1A",
    "medium": "#666666",
    "light": "#B0B0B0",
    "success": "#0D652D",
    "warning": "#FF8C00",
    "danger": "#A4262C",
    "info": "#0066CC"
}

# Bölge renkleri (McKinsey palette)
REGION_COLORS = {
    "MARMARA": "#0066CC",
    "EGE": "#0D652D",
    "AKDENİZ": "#FF8C00",
    "İÇ ANADOLU": "#8B5CF6",
    "KARADENİZ": "#06B6D4",
    "GÜNEYDOĞU ANADOLU": "#A4262C",
    "DOĞU ANADOLU": "#84CC16",
    "BATI KARADENİZ": "#6366F1",
    "ORTA KARADENİZ": "#14B8A6",
    "KUZEYDOĞU ANADOLU": "#F97316",
    "GÜNEYDOĞU": "#EC4899",
    "DİĞER": "#666666"
}

# Performans renkleri
PERFORMANCE_COLORS = {
    "excellent": "#0D652D",
    "good": "#0066CC",
    "fair": "#FF8C00",
    "poor": "#A4262C",
    "high": "#0D652D",
    "medium": "#FF8C00",
    "low": "#A4262C",
    "positive": "#0D652D",
    "negative": "#A4262C",
    "neutral": "#666666"
}

# Gradient skalaları (McKinsey style)
GRADIENT_SCALES = {
    "blue_scale": ["#F0F7FF", "#B3D4FF", "#66AFFF", "#0066CC", "#004080"],
    "green_scale": ["#E6F4EA", "#A7D7B8", "#68B987", "#0D652D", "#0A4A21"],
    "orange_scale": ["#FFF4E5", "#FFD6A3", "#FFB861", "#FF8C00", "#CC7000"],
    "diverging": ["#A4262C", "#FF8C00", "#FFD6A3", "#F0F7FF", "#B3D4FF", "#0066CC"]
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

# Türkiye şehir koordinatları (daha doğru)
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
    """Güvenli bölme işlemi"""
    if isinstance(a, (pd.Series, np.ndarray)) and isinstance(b, (pd.Series, np.ndarray)):
        result = a.copy() if isinstance(a, pd.Series) else a
        mask = b != 0
        if isinstance(a, pd.Series):
            result[mask] = a[mask] / b[mask]
            result[~mask] = 0
        else:
            result[mask] = a[mask] / b[mask]
            result[~mask] = 0
        return result.replace([np.inf, -np.inf], 0) if isinstance(result, pd.Series) else np.nan_to_num(result)
    else:
        if b == 0:
            return 0
        return a / b

def get_product_columns(product, df_columns):
    """Ürün sütunlarını getir"""
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
            # Eksikse sütun oluştur
            available_columns[key] = col_name
    
    return available_columns

def normalize_city_name(city_name):
    """Şehir ismini normalize et"""
    if pd.isna(city_name):
        return None
    
    city_str = str(city_name).strip().upper()
    
    # Türkçe karakter düzeltmeleri
    turkish_chars = {'İ': 'I', 'Ğ': 'G', 'Ü': 'U', 'Ş': 'S', 'Ö': 'O', 'Ç': 'C',
                     'Â': 'A', 'Î': 'I', 'Û': 'U'}
    
    for k, v in turkish_chars.items():
        city_str = city_str.replace(k, v)
    
    # Özel düzeltmeler
    corrections = {
        'ISTANBUL': 'İstanbul',
        'IZMIR': 'İzmir',
        'ANKARA': 'Ankara',
        'BURSA': 'Bursa',
        'ADANA': 'Adana',
        'GAZIANTEP': 'Gaziantep',
        'KONYA': 'Konya',
        'ANTALYA': 'Antalya',
        'KAYSERI': 'Kayseri',
        'MERSIN': 'Mersin'
    }
    
    return corrections.get(city_str, city_str.title())

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
        buyume = min(max(row['Buyume_%'], -50), 200)
        buyume_score = 50 + (buyume / 2)
        score += buyume_score * weights['buyume']
    max_score += 100 * weights['buyume']
    
    # Pazar büyüklüğü skoru
    if 'Toplam_Pazar' in row:
        market_size = np.log1p(row['Toplam_Pazar'])
        market_score = min(market_size / np.log1p(1000000) * 100, 100)
        score += market_score * weights['market_buyuklugu']
    max_score += 100 * weights['market_buyuklugu']
    
    # Stabilite skoru
    if 'Stabilite_Score' in row:
        stabilite = min(row['Stabilite_Score'], 100)
        score += stabilite * weights['stabilite']
    max_score += 100 * weights['stabilite']
    
    # Rekabet skoru
    if 'Goreceli_Pazar_Payi' in row:
        rekabet = min(row['Goreceli_Pazar_Payi'] * 20, 100)
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
# VERİ YÜKLEME
# =============================================================================

@st.cache_data
def load_excel_data(file):
    """Excel dosyasını yükle"""
    try:
        df = pd.read_excel(file)
        
        # Sütun isimlerini normalize et
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        # Tarih sütununu bul
        date_columns = ['DATE', 'TARIH', 'TARİH', 'YEAR_MONTH', 'AY-YIL', 'AY_YIL']
        date_column_found = False
        
        for date_col in date_columns:
            if date_col in df.columns:
                df['DATE'] = pd.to_datetime(df[date_col], errors='coerce')
                date_column_found = True
                break
        
        if not date_column_found:
            # İlk sütunu tarih olarak dene
            try:
                df['DATE'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
            except:
                # Son çare olarak index kullan
                df['DATE'] = pd.date_range(start='2023-01-01', periods=len(df), freq='M')
        
        # NaN tarihleri temizle
        df = df.dropna(subset=['DATE'])
        
        # Tarih sütunları oluştur
        df['YIL_AY'] = df['DATE'].dt.strftime('%Y-%m')
        df['AY'] = df['DATE'].dt.month
        df['YIL'] = df['DATE'].dt.year
        df['QUARTER'] = df['DATE'].dt.quarter
        
        # Territory ve şehir sütunlarını normalize et
        territory_columns = ['TERRITORIES', 'TERRITORY', 'TERRİTORY']
        city_columns = ['CITY', 'CİTY', 'SEHIR', 'ŞEHİR']
        region_columns = ['REGION', 'REGİON', 'BOLGE']
        
        for col_list, target in [
            (territory_columns, 'TERRITORIES'),
            (city_columns, 'CITY'),
            (region_columns, 'REGION')
        ]:
            for col in col_list:
                if col in df.columns:
                    df[target] = df[col].astype(str).str.upper().str.strip()
                    break
            if target not in df.columns:
                df[target] = 'BELİRTİLMEMİŞ'
        
        # Şehir normalizasyonu
        df['CITY_NORMALIZED'] = df['CITY'].apply(normalize_city_name)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Veri yükleme hatası: {str(e)}")
        st.stop()

# =============================================================================
# HARİTA FONKSİYONLARI - DÜZELTİLMİŞ
# =============================================================================

def create_turkey_map(city_data, title="Türkiye Satış Haritası", metric="PF_Satis"):
    """
    Türkiye haritası oluştur - DÜZELTİLMİŞ versiyon
    """
    try:
        # Veriyi hazırla
        city_data = city_data.copy()
        
        # Koordinatları ekle
        city_data['lon'] = city_data['City'].map(
            lambda x: TURKEY_CITY_COORDS.get(x, (35.0, 39.0))[0]
        )
        city_data['lat'] = city_data['City'].map(
            lambda x: TURKEY_CITY_COORDS.get(x, (35.0, 39.0))[1]
        )
        
        # NaN koordinatları kontrol et
        if city_data['lon'].isna().any() or city_data['lat'].isna().any():
            st.warning("Bazı şehirler için koordinat bulunamadı")
            # Eksik koordinatları filtrele
            city_data = city_data.dropna(subset=['lon', 'lat'])
        
        # Bubble boyutunu ayarla
        if metric == "PF_Satis":
            size_col = 'PF_Satis'
            color_col = 'Pazar_Payi_%'
            color_scale = 'RdYlGn'
            color_title = "Pazar Payı %"
        elif metric == "Performance_Score":
            size_col = 'PF_Satis'
            color_col = 'Performance_Score'
            color_scale = 'RdYlGn'
            color_title = "Performans Skoru"
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
                        font=dict(size=10)
                    ),
                    thickness=15,
                    x=1.02,
                    xpad=5
                ),
                opacity=0.8,
                sizemode='diameter'
            ),
            text=city_data.apply(
                lambda row: f"<b>{row['City']}</b><br>"
                          f"Bölge: {row.get('Region', 'N/A')}<br>"
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
                zoom=4.5,
                bearing=0,
                pitch=0
            ),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0),
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(size=18, color='#1A1A1A'),
                y=0.95
            ),
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Harita oluşturma hatası: {str(e)}")
        # Basit bir boş harita döndür
        fig = go.Figure()
        fig.update_layout(
            title="Harita oluşturulamadı",
            height=400,
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF'
        )
        return fig

# =============================================================================
# ANALİZ FONKSİYONLARI
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
        st.warning(f"{pf_col} sütunu bulunamadı!")
        # Eksik sütunları oluştur
        df_filtered[pf_col] = 0
    
    if rakip_col not in df_filtered.columns:
        df_filtered[rakip_col] = 0
    
    # Şehir bazlı toplamlar
    if 'CITY_NORMALIZED' not in df_filtered.columns:
        df_filtered['CITY_NORMALIZED'] = df_filtered['CITY'].apply(normalize_city_name)
    
    city_data = df_filtered.groupby(['CITY_NORMALIZED']).agg({
        pf_col: 'sum',
        rakip_col: 'sum'
    }).reset_index()
    
    city_data.columns = ['City', 'PF_Satis', 'Rakip_Satis']
    city_data['Toplam_Pazar'] = city_data['PF_Satis'] + city_data['Rakip_Satis']
    city_data['Pazar_Payi_%'] = safe_divide(city_data['PF_Satis'], city_data['Toplam_Pazar']) * 100
    
    # Region bilgisini ekle
    if 'REGION' in df_filtered.columns:
        region_map = df_filtered.groupby('CITY_NORMALIZED')['REGION'].first().to_dict()
        city_data['Region'] = city_data['City'].map(region_map)
    else:
        city_data['Region'] = 'Belirtilmemiş'
    
    # Pozitif satışı olan şehirler
    city_data = city_data[city_data['PF_Satis'] > 0]
    
    if len(city_data) > 0:
        # Basit performans hesaplamaları
        # Büyüme oranı (basit)
        city_data['Buyume_%'] = 0  # Burada daha detaylı hesaplama yapılabilir
        
        # Stabilite skoru (basit)
        city_data['Stabilite_Score'] = 70  # Varsayılan
        
        # Performans skoru
        city_data['Performance_Score'] = city_data.apply(calculate_performance_score, axis=1)
        city_data['Performance_Category'] = city_data['Performance_Score'].apply(
            lambda x: get_performance_category(x)[0]
        )
    
    return city_data.sort_values('PF_Satis', ascending=False)

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
        df_filtered[pf_col] = 0
    
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
    
    return monthly_data.fillna(0)

def prepare_ml_data_simple(monthly_data):
    """Basit ML veri hazırlama - hata düzeltmeli"""
    try:
        if len(monthly_data) < 6:
            return None, None, None, None, None
        
        df = monthly_data.copy().reset_index(drop=True)
        
        # Temel feature'lar
        df['month'] = df['DATE'].dt.month
        df['index'] = range(len(df))
        
        # Lag features (sadece 1 ve 2 lag)
        df['lag_1'] = df['PF_Satis'].shift(1)
        df['lag_2'] = df['PF_Satis'].shift(2)
        
        # Simple rolling features
        df['rolling_mean_3'] = df['PF_Satis'].rolling(window=3, min_periods=1).mean()
        
        # Target (sonraki ay)
        df['target'] = df['PF_Satis'].shift(-1)
        
        # NaN'leri temizle
        df_clean = df.dropna()
        
        if len(df_clean) < 4:
            return None, None, None, None, None
        
        # Feature columns
        feature_cols = ['lag_1', 'lag_2', 'rolling_mean_3', 'month', 'index']
        feature_cols = [col for col in feature_cols if col in df_clean.columns]
        
        if not feature_cols:
            return None, None, None, None, None
        
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        # Simple train/test split
        split_idx = max(3, int(len(df_clean) * 0.7))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test, feature_cols
        
    except Exception as e:
        st.warning(f"ML veri hazırlama hatası: {str(e)}")
        return None, None, None, None, None

# =============================================================================
# GÖRSELLEŞTİRME FONKSİYONLARI
# =============================================================================

def create_mckinsey_line_chart(data, x_col, y_col, title, color=MCKINSEY_COLORS['primary']):
    """McKinsey stili line chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode='lines+markers',
        line=dict(color=color, width=3),
        marker=dict(size=8, color='white', line=dict(width=2, color=color)),
        name=y_col
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=16, color='#1A1A1A')
        ),
        xaxis_title=f"<b>{x_col}</b>",
        yaxis_title=f"<b>{y_col}</b>",
        height=400,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#333333'),
        hovermode='x unified'
    )
    
    return fig

def create_mckinsey_bar_chart(data, x_col, y_col, title, color=None):
    """McKinsey stili bar chart"""
    if color is None:
        color = MCKINSEY_COLORS['primary']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data[x_col],
        y=data[y_col],
        marker_color=color,
        marker_line_width=0,
        name=y_col
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=16, color='#1A1A1A')
        ),
        xaxis_title=f"<b>{x_col}</b>",
        yaxis_title=f"<b>{y_col}</b>",
        height=400,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#333333')
    )
    
    return fig

def create_performance_scatter(city_data):
    """Performans scatter plot"""
    if len(city_data) == 0 or 'Pazar_Payi_%' not in city_data.columns:
        return None
    
    fig = go.Figure()
    
    # Renkleri performansa göre ayarla
    colors = []
    for score in city_data.get('Performance_Score', [50] * len(city_data)):
        if score >= 80:
            colors.append(PERFORMANCE_COLORS['excellent'])
        elif score >= 65:
            colors.append(PERFORMANCE_COLORS['good'])
        elif score >= 50:
            colors.append(PERFORMANCE_COLORS['fair'])
        else:
            colors.append(PERFORMANCE_COLORS['poor'])
    
    fig.add_trace(go.Scatter(
        x=city_data['Pazar_Payi_%'],
        y=city_data.get('Buyume_%', [0] * len(city_data)),
        mode='markers',
        marker=dict(
            size=city_data['PF_Satis'] / city_data['PF_Satis'].max() * 30 + 10,
            color=colors,
            line=dict(width=1, color='white')
        ),
        text=city_data['City'],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Performans Matrisi: Pazar Payı vs Büyüme</b>',
            font=dict(size=16, color='#1A1A1A')
        ),
        xaxis_title='<b>Pazar Payı (%)</b>',
        yaxis_title='<b>Büyüme Oranı (%)</b>',
        height=500,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#333333')
    )
    
    return fig

# =============================================================================
# ANA UYGULAMA
# =============================================================================

def main():
    # Başlık
    st.markdown('<h1 class="main-header">🎯 Ticari Portföy Analiz Sistemi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Territory Bazlı Performans • ML Tahminleme • Stratejik Öneriler</p>', unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown('<div style="border-bottom: 2px solid #0066CC; padding-bottom: 0.5rem; margin-bottom: 1.5rem;">'
                   '<h3 style="color: #1A1A1A; margin: 0;">📂 Veri Yükleme</h3>'
                   '</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Excel dosyasını seçin", type=['xlsx', 'xls'])
        
        if not uploaded_file:
            st.info("👈 Lütfen Excel dosyasını yükleyin")
            st.stop()
        
        try:
            df = load_excel_data(uploaded_file)
            
            st.success(f"✅ **{len(df):,}** satır veri yüklendi")
            
            with st.expander("📋 Veri Önizleme"):
                st.dataframe(df.head(), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.stop()
        
        st.markdown("---")
        
        # Ürün Seçimi
        st.markdown('<div style="margin-bottom: 1.5rem;">'
                   '<h4 style="color: #1A1A1A; margin-bottom: 0.5rem;">💊 Ürün Seçimi</h4>', 
                   unsafe_allow_html=True)
        
        # Mevcut ürünleri kontrol et
        available_products = []
        for product in COLUMN_NAMES.keys():
            cols = get_product_columns(product, df.columns)
            if cols.get('pf') in df.columns:
                available_products.append(product)
        
        if not available_products:
            st.error("❌ Excel'de beklenen ürün sütunları bulunamadı!")
            st.stop()
        
        selected_product = st.selectbox("", available_products, label_visibility="collapsed")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tarih Filtresi
        st.markdown('<div style="margin-bottom: 1.5rem;">'
                   '<h4 style="color: #1A1A1A; margin-bottom: 0.5rem;">📅 Tarih Filtresi</h4>', 
                   unsafe_allow_html=True)
        
        min_date = df['DATE'].min()
        max_date = df['DATE'].max()
        
        date_options = ["Tüm Veriler", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl", "Bu Yıl"]
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
        else:
            date_filter = None
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Territory Filtresi
        st.markdown('<div style="margin-bottom: 1.5rem;">'
                   '<h4 style="color: #1A1A1A; margin-bottom: 0.5rem;">📍 Territory Filtresi</h4>', 
                   unsafe_allow_html=True)
        
        territories = ["TÜMÜ"] + sorted([str(t) for t in df['TERRITORIES'].unique() if pd.notna(t)][:20])
        selected_territory = st.selectbox("Territory Seçin", territories)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analiz butonu
        st.markdown("---")
        if st.button("🚀 Analizi Başlat", type="primary", use_container_width=True):
            st.session_state['analysis_started'] = True
        
        if 'analysis_started' not in st.session_state:
            st.session_state['analysis_started'] = False
    
    # ANA İÇERİK
    if not st.session_state.get('analysis_started', False):
        st.info("👈 Lütfen sol taraftan filtreleri ayarlayın ve 'Analizi Başlat' butonuna tıklayın")
        st.stop()
    
    # Progress bar
    with st.spinner("Veri analizi yapılıyor..."):
        # Analizleri yap
        city_data = calculate_city_performance(df, selected_product, date_filter)
        monthly_data = calculate_time_series_analysis(df, selected_product, None, date_filter)
    
    # TAB'ler
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Genel Bakış",
        "🗺️ Coğrafi Analiz",
        "📈 Trend Analizi",
        "🤖 Tahminler"
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
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Toplam PF Satış", f"{total_pf:,.0f}")
            with col2:
                st.metric("Toplam Pazar", f"{total_market:,.0f}")
            with col3:
                st.metric("Pazar Payı", f"%{market_share:.1f}")
            with col4:
                st.metric("Şehir Sayısı", len(city_data))
            
            st.markdown("---")
            
            # Top şehirler
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("🏆 En İyi Performans Gösteren Şehirler")
                top_cities = city_data.nlargest(10, 'PF_Satis')
                
                fig = create_mckinsey_bar_chart(
                    top_cities,
                    'City',
                    'PF_Satis',
                    'En Yüksek PF Satışına Sahip Şehirler',
                    MCKINSEY_COLORS['primary']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_right:
                st.subheader("📊 Pazar Payı Dağılımı")
                if 'Performance_Category' in city_data.columns:
                    perf_dist = city_data['Performance_Category'].value_counts()
                    
                    colors = {
                        'MÜKEMMEL': PERFORMANCE_COLORS['excellent'],
                        'İYİ': PERFORMANCE_COLORS['good'],
                        'ORTA': PERFORMANCE_COLORS['fair'],
                        'ZAYIF': PERFORMANCE_COLORS['poor']
                    }
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=perf_dist.index,
                        values=perf_dist.values,
                        marker=dict(colors=[colors.get(k, '#666666') for k in perf_dist.index])
                    )])
                    
                    fig.update_layout(
                        title=dict(
                            text='<b>Performans Kategori Dağılımı</b>',
                            font=dict(size=14, color='#1A1A1A')
                        ),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Performans kategorisi hesaplanamadı")
            
            # Performans scatter plot
            st.markdown("---")
            st.subheader("📈 Performans Matrisi")
            
            perf_scatter = create_performance_scatter(city_data)
            if perf_scatter:
                st.plotly_chart(perf_scatter, use_container_width=True)
            
            # Detaylı tablo
            with st.expander("📋 Detaylı Şehir Verileri"):
                display_cols = ['City', 'Region', 'PF_Satis', 'Toplam_Pazar', 'Pazar_Payi_%']
                if 'Performance_Score' in city_data.columns:
                    display_cols.extend(['Performance_Score', 'Performance_Category'])
                
                city_display = city_data[display_cols].copy()
                city_display.columns = ['Şehir', 'Bölge', 'PF Satış', 'Toplam Pazar', 'Pazar Payı %', 
                                      'Performans Skoru', 'Kategori'][:len(display_cols)]
                
                st.dataframe(city_display, use_container_width=True)
    
    # TAB 2: COĞRAFİ ANALİZ
    with tab2:
        st.header("🗺️ Coğrafi Analiz")
        
        if len(city_data) == 0:
            st.warning("⚠️ Harita için veri bulunamadı")
        else:
            # Harita seçenekleri
            map_option = st.radio(
                "Gösterilecek Metrik",
                ["Satış Dağılımı", "Pazar Payı", "Performans"],
                horizontal=True
            )
            
            if map_option == "Satış Dağılımı":
                turkey_map = create_turkey_map(city_data, f"{selected_product} - Satış Dağılımı", "PF_Satis")
            elif map_option == "Pazar Payı":
                turkey_map = create_turkey_map(city_data, f"{selected_product} - Pazar Payı Dağılımı", "PF_Satis")
            else:
                if 'Performance_Score' in city_data.columns:
                    turkey_map = create_turkey_map(city_data, f"{selected_product} - Performans Dağılımı", "Performance_Score")
                else:
                    turkey_map = create_turkey_map(city_data, f"{selected_product} - Satış Dağılımı", "PF_Satis")
            
            if turkey_map:
                st.plotly_chart(turkey_map, use_container_width=True)
            
            # Bölge analizi
            st.markdown("---")
            st.subheader("📊 Bölge Bazlı Analiz")
            
            if 'Region' in city_data.columns:
                region_summary = city_data.groupby('Region').agg({
                    'PF_Satis': 'sum',
                    'Toplam_Pazar': 'sum'
                }).reset_index()
                
                region_summary['Pazar_Payi_%'] = safe_divide(region_summary['PF_Satis'], region_summary['Toplam_Pazar']) * 100
                
                col_r1, col_r2 = st.columns(2)
                
                with col_r1:
                    fig = create_mckinsey_bar_chart(
                        region_summary.sort_values('PF_Satis', ascending=True),
                        'PF_Satis',
                        'Region',
                        'Bölgelere Göre PF Satış',
                        MCKINSEY_COLORS['primary']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_r2:
                    fig = create_mckinsey_bar_chart(
                        region_summary.sort_values('Pazar_Payi_%', ascending=True),
                        'Pazar_Payi_%',
                        'Region',
                        'Bölgelere Göre Pazar Payı',
                        MCKINSEY_COLORS['secondary']
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: TREND ANALİZİ
    with tab3:
        st.header("📈 Trend Analizi")
        
        if len(monthly_data) == 0:
            st.warning("⚠️ Zaman serisi verisi bulunamadı")
        else:
            # Satış trendi
            st.subheader("📊 Satış Trendi")
            
            fig = create_mckinsey_line_chart(
                monthly_data,
                'DATE',
                'PF_Satis',
                'PF Satış Trendi',
                MCKINSEY_COLORS['primary']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Pazar payı trendi
            st.subheader("🎯 Pazar Payı Trendi")
            
            fig = create_mckinsey_line_chart(
                monthly_data,
                'DATE',
                'Pazar_Payi_%',
                'Pazar Payı Trendi',
                MCKINSEY_COLORS['secondary']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Büyüme oranları
            st.markdown("---")
            st.subheader("🚀 Büyüme Analizi")
            
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                fig = create_mckinsey_bar_chart(
                    monthly_data,
                    'DATE',
                    'PF_Buyume_%',
                    'PF Büyüme Oranı',
                    MCKINSEY_COLORS['primary']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_g2:
                fig = create_mckinsey_bar_chart(
                    monthly_data,
                    'DATE',
                    'Rakip_Buyume_%',
                    'Rakip Büyüme Oranı',
                    MCKINSEY_COLORS['danger']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: TAHMİNLER
    with tab4:
        st.header("🤖 Tahmin ve Öngörüler")
        
        if len(monthly_data) < 6:
            st.warning("⚠️ Tahmin için yeterli veri yok (en az 6 ay gereklidir)")
        else:
            # ML Tahminleri
            st.subheader("📈 ML Tahmin Modelleri")
            
            with st.spinner("ML modelleri eğitiliyor..."):
                X_train, X_test, y_train, y_test, feature_cols = prepare_ml_data_simple(monthly_data)
                
                if X_train is not None:
                    # Basit lineer regresyon
                    try:
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        
                        # Tahminler
                        y_pred = model.predict(X_test)
                        
                        # Performans metrikleri
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("MAE (Ortalama Mutlak Hata)", f"{mae:,.0f}")
                        with col_m2:
                            st.metric("RMSE (Kök Ortalama Kare Hata)", f"{rmse:,.0f}")
                        
                        # Tahmin grafiği
                        st.markdown("---")
                        st.subheader("🔮 Gelecek Tahminleri")
                        
                        # Son 3 ay için basit tahmin
                        last_values = monthly_data['PF_Satis'].tail(3).values
                        avg_growth = monthly_data['PF_Buyume_%'].tail(6).mean() / 100
                        
                        forecast_months = 3
                        forecast_data = []
                        
                        last_value = last_values[-1]
                        for i in range(forecast_months):
                            forecast_value = last_value * (1 + avg_growth)
                            forecast_data.append({
                                'DATE': monthly_data['DATE'].iloc[-1] + pd.DateOffset(months=i+1),
                                'Prediction': forecast_value,
                                'Actual': None
                            })
                            last_value = forecast_value
                        
                        forecast_df = pd.DataFrame(forecast_data)
                        
                        # Tarihsel ve tahmini veriyi birleştir
                        historical_df = monthly_data[['DATE', 'PF_Satis']].copy()
                        historical_df.columns = ['DATE', 'Value']
                        historical_df['Type'] = 'Tarihsel'
                        
                        forecast_display = forecast_df[['DATE', 'Prediction']].copy()
                        forecast_display.columns = ['DATE', 'Value']
                        forecast_display['Type'] = 'Tahmin'
                        
                        combined_df = pd.concat([historical_df, forecast_display])
                        
                        # Grafik
                        fig = go.Figure()
                        
                        # Tarihsel veri
                        fig.add_trace(go.Scatter(
                            x=historical_df['DATE'],
                            y=historical_df['Value'],
                            mode='lines+markers',
                            name='Tarihsel',
                            line=dict(color=MCKINSEY_COLORS['primary'], width=3),
                            marker=dict(size=6)
                        ))
                        
                        # Tahmin
                        fig.add_trace(go.Scatter(
                            x=forecast_display['DATE'],
                            y=forecast_display['Value'],
                            mode='lines+markers',
                            name='Tahmin',
                            line=dict(color=MCKINSEY_COLORS['secondary'], width=3, dash='dash'),
                            marker=dict(size=8, symbol='diamond')
                        ))
                        
                        fig.update_layout(
                            title=dict(
                                text='<b>PF Satış Tahminleri</b>',
                                font=dict(size=16, color='#1A1A1A')
                            ),
                            xaxis_title='<b>Tarih</b>',
                            yaxis_title='<b>PF Satış</b>',
                            height=500,
                            plot_bgcolor='#FFFFFF',
                            paper_bgcolor='#FFFFFF',
                            font=dict(color='#333333')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tahmin tablosu
                        with st.expander("📋 Tahmin Detayları"):
                            forecast_display = forecast_df.copy()
                            forecast_display['DATE'] = forecast_display['DATE'].dt.strftime('%Y-%m')
                            forecast_display.columns = ['Ay', 'Tahmini Satış', 'Gerçek Satış']
                            forecast_display = forecast_display[['Ay', 'Tahmini Satış']]
                            forecast_display.index = range(1, len(forecast_display) + 1)
                            
                            st.dataframe(forecast_display, use_container_width=True)
                            
                    except Exception as e:
                        st.warning(f"ML model eğitimi başarısız: {str(e)}")
                        st.info("Basit ortalama büyüme tahmini kullanılıyor...")
                        
                        # Basit tahmin göster
                        avg_sales = monthly_data['PF_Satis'].mean()
                        st.metric("Ortalama Aylık Satış", f"{avg_sales:,.0f}")
                        st.metric("Ortalama Büyüme", f"%{monthly_data['PF_Buyume_%'].mean():.1f}")
                else:
                    st.warning("ML için yeterli veri yok")
            
            # Stratejik Öneriler
            st.markdown("---")
            st.subheader("💡 Stratejik Öneriler")
            
            if len(city_data) > 0:
                # En iyi ve en kötü performanslar
                top_city = city_data.nlargest(1, 'PF_Satis').iloc[0]
                worst_city = city_data[city_data['Pazar_Payi_%'] > 0].nsmallest(1, 'Pazar_Payi_%').iloc[0]
                
                col_s1, col_s2 = st.columns(2)
                
                with col_s1:
                    st.markdown("##### 🏆 En İyi Performans")
                    st.markdown(f"**{top_city['City']}**")
                    st.markdown(f"PF Satış: {top_city['PF_Satis']:,.0f}")
                    st.markdown(f"Pazar Payı: %{top_city['Pazar_Payi_%']:.1f}")
                    st.markdown("*Öneri: Bu başarıyı diğer şehirlere yayın*")
                
                with col_s2:
                    st.markdown("##### 📉 Geliştirme Gereken")
                    st.markdown(f"**{worst_city['City']}**")
                    st.markdown(f"Pazar Payı: %{worst_city['Pazar_Payi_%']:.1f}")
                    st.markdown(f"Potansiyel: {worst_city['Toplam_Pazar']:,.0f}")
                    st.markdown("*Öneri: Bu şehirde özel kampanya başlatın*")

if __name__ == "__main__":
    main()
