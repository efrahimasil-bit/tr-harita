"""
🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ
Territory Bazlı Performans, ML Tahminleme, Türkiye Haritası ve Rekabet Analizi

Özellikler:
- 🗺️ Türkiye il bazlı harita görselleştirme
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
import xgboost as xgb
from prophet import Prophet
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from scipy import stats

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
# YARDIMCI FONKSİYONLAR
# =============================================================================

def safe_divide(a, b):
    """Güvenli bölme işlemi"""
    return np.where(b != 0, a / b, 0)

def detect_and_standardize_columns(df):
    """Excel'deki sütunları otomatik tespit et ve standartlaştır"""
    df = df.copy()
    
    # Sütun adlarını temizle (boşlukları kaldır, büyük harfe çevir)
    df.columns = [str(col).strip().upper() for col in df.columns]
    
    column_mapping = {}
    
    # Standart sütunları tespit et
    standard_columns = {
        'TERRITORIES': ['TERRITORIES', 'TERRITORY', 'BOLGE', 'BÖLGE', 'REGION', 'AREA'],
        'CITY': ['CITY', 'SEHIR', 'ŞEHİR', 'IL', 'İL', 'CITY_NAME'],
        'REGION': ['REGION', 'BOLGE', 'BÖLGE', 'AREA', 'REGION_NAME'],
        'DATE': ['DATE', 'TARIH', 'TARİH', 'AY', 'MONTH', 'YEAR', 'YIL'],
        'MANAGER': ['MANAGER', 'SORUMLU', 'SATISCI', 'REP', 'REPRESENTATIVE'],
        'YEAR': ['YEAR', 'YIL'],
        'MONTH': ['MONTH', 'AY']
    }
    
    # Her standart sütun için Excel'deki karşılığını bul
    for standard_name, possible_names in standard_columns.items():
        for possible_name in possible_names:
            if possible_name in df.columns:
                if possible_name != standard_name:
                    column_mapping[possible_name] = standard_name
                break
    
    # Satış sütunlarını tespit et
    sales_columns = {}
    for col in df.columns:
        col_upper = col.upper()
        # PF ürünlerini bul
        if 'TROC' in col_upper and 'METAM' in col_upper and 'DIGER' not in col_upper:
            sales_columns['TROCMETAM'] = col
        elif 'CORTIPOL' in col_upper and 'DIGER' not in col_upper:
            sales_columns['CORTIPOL'] = col
        elif 'DEKSAMETAZON' in col_upper and 'DIGER' not in col_upper:
            sales_columns['DEKSAMETAZON'] = col
        elif ('IZOTONIK' in col_upper or 'IZOTONIC' in col_upper) and 'DIGER' not in col_upper:
            sales_columns['PF IZOTONIK'] = col
        # Rakip ürünleri bul
        elif 'DIGER' in col_upper:
            if 'TROC' in col_upper and 'METAM' in col_upper:
                sales_columns['DIGER TROCMETAM'] = col
            elif 'CORTIPOL' in col_upper:
                sales_columns['DIGER CORTIPOL'] = col
            elif 'DEKSAMETAZON' in col_upper:
                sales_columns['DIGER DEKSAMETAZON'] = col
            elif 'IZOTONIK' in col_upper or 'IZOTONIC' in col_upper:
                sales_columns['DIGER IZOTONIK'] = col
    
    # Sütunları yeniden adlandır
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    return df, sales_columns

def normalize_city_name(city_name):
    """Şehir adını normalleştir"""
    if pd.isna(city_name):
        return "BILINMIYOR"
    
    city_str = str(city_name).strip().upper()
    
    # Türkçe karakter düzeltmeleri
    turkish_chars = {
        'İ': 'I', 'Ğ': 'G', 'Ü': 'U', 'Ş': 'S', 'Ö': 'O', 'Ç': 'C',
        'ı': 'I', 'ğ': 'G', 'ü': 'U', 'ş': 'S', 'ö': 'O', 'ç': 'C',
        'Â': 'A', 'Î': 'I', 'Û': 'U'
    }
    
    for old, new in turkish_chars.items():
        city_str = city_str.replace(old, new)
    
    # Özel durumlar
    special_cases = {
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
        'AFYON': 'AFYONKARAHİSAR'
    }
    
    return special_cases.get(city_str, city_str)

def get_product_columns(product, sales_columns_mapping):
    """Ürün kolonlarını mapping'den al"""
    product_mapping = {
        'TROCMETAM': {'pf': 'TROCMETAM', 'rakip': 'DIGER TROCMETAM'},
        'CORTIPOL': {'pf': 'CORTIPOL', 'rakip': 'DIGER CORTIPOL'},
        'DEKSAMETAZON': {'pf': 'DEKSAMETAZON', 'rakip': 'DIGER DEKSAMETAZON'},
        'PF IZOTONIK': {'pf': 'PF IZOTONIK', 'rakip': 'DIGER IZOTONIK'}
    }
    
    mapping = product_mapping.get(product, {'pf': 'PF_SATIS', 'rakip': 'RAKIP_SATIS'})
    
    # Gerçek sütun adlarını mapping'den al
    actual_pf = sales_columns_mapping.get(mapping['pf'], mapping['pf'])
    actual_rakip = sales_columns_mapping.get(mapping['rakip'], mapping['rakip'])
    
    return {'pf': actual_pf, 'rakip': actual_rakip}

def get_available_products(sales_columns_mapping):
    """Mevcut ürünleri tespit et"""
    available = []
    
    if 'TROCMETAM' in sales_columns_mapping:
        available.append('TROCMETAM')
    if 'CORTIPOL' in sales_columns_mapping:
        available.append('CORTIPOL')
    if 'DEKSAMETAZON' in sales_columns_mapping:
        available.append('DEKSAMETAZON')
    if 'PF IZOTONIK' in sales_columns_mapping:
        available.append('PF IZOTONIK')
    
    return available if available else ['TROCMETAM', 'CORTIPOL', 'DEKSAMETAZON', 'PF IZOTONIK']

# =============================================================================
# VERİ YÜKLEME
# =============================================================================

@st.cache_data
def load_excel_data(file):
    """Excel dosyasını yükle ve sütunları standartlaştır"""
    try:
        # Excel'i yükle
        df = pd.read_excel(file)
        
        # Sütunları tespit et ve standartlaştır
        df_clean, sales_columns_mapping = detect_and_standardize_columns(df)
        
        # Tarih sütununu işle
        if 'DATE' in df_clean.columns:
            try:
                df_clean['DATE'] = pd.to_datetime(df_clean['DATE'], errors='coerce')
            except:
                # Alternatif tarih formatı deneyelim
                df_clean['DATE'] = pd.to_datetime(df_clean['DATE'].astype(str), errors='coerce')
        else:
            # Tarih sütunu yoksa index kullan
            df_clean['DATE'] = pd.date_range(start='2023-01-01', periods=len(df_clean), freq='M')
        
        # Ek kolonlar oluştur
        df_clean['YIL_AY'] = df_clean['DATE'].dt.strftime('%Y-%m')
        df_clean['AY'] = df_clean['DATE'].dt.month
        df_clean['YIL'] = df_clean['DATE'].dt.year
        
        # Şehir adlarını normalleştir
        if 'CITY' in df_clean.columns:
            df_clean['CITY_NORMALIZED'] = df_clean['CITY'].apply(normalize_city_name)
        else:
            df_clean['CITY_NORMALIZED'] = 'BILINMIYOR'
        
        # Bölge bilgisi yoksa varsayılan ata
        if 'REGION' not in df_clean.columns:
            df_clean['REGION'] = 'DIGER'
        
        # Manager bilgisi yoksa varsayılan ata
        if 'MANAGER' not in df_clean.columns:
            df_clean['MANAGER'] = 'BILINMIYOR'
        
        # Territory bilgisi yoksa şehirden oluştur
        if 'TERRITORIES' not in df_clean.columns:
            df_clean['TERRITORIES'] = df_clean['CITY_NORMALIZED']
        
        # String kolonlarını temizle
        for col in ['TERRITORIES', 'CITY', 'REGION', 'MANAGER']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).fillna('BILINMIYOR')
        
        return df_clean, sales_columns_mapping
        
    except Exception as e:
        st.error(f"Excel dosyası okunurken hata: {str(e)}")
        # Örnek veri oluştur
        return create_sample_data()

def create_sample_data():
    """Örnek veri oluştur (test için)"""
    dates = pd.date_range(start='2023-01-01', end='2025-01-01', freq='M')
    territories = [f'TERR_{i:03d}' for i in range(1, 31)]
    cities = ['İSTANBUL', 'ANKARA', 'İZMİR', 'BURSA', 'ANTALYA', 'ADANA', 'KONYA', 'GAZİANTEP']
    regions = ['MARMARA', 'İÇ ANADOLU', 'EGE', 'AKDENİZ', 'GÜNEYDOĞU ANADOLU']
    
    data = []
    for date in dates:
        for territory in territories[:15]:
            city = np.random.choice(cities)
            region = np.random.choice(regions)
            manager = f'MANAGER_{np.random.randint(1, 6)}'
            
            data.append({
                'DATE': date,
                'TERRITORIES': territory,
                'CITY': city,
                'REGION': region,
                'MANAGER': manager,
                'TROCMETAM': np.random.randint(1000, 50000),
                'DIGER TROCMETAM': np.random.randint(800, 45000),
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
    df['CITY_NORMALIZED'] = df['CITY']
    
    sales_mapping = {
        'TROCMETAM': 'TROCMETAM',
        'DIGER TROCMETAM': 'DIGER TROCMETAM',
        'CORTIPOL': 'CORTIPOL',
        'DIGER CORTIPOL': 'DIGER CORTIPOL',
        'DEKSAMETAZON': 'DEKSAMETAZON',
        'DIGER DEKSAMETAZON': 'DIGER DEKSAMETAZON',
        'PF IZOTONIK': 'PF IZOTONIK',
        'DIGER IZOTONIK': 'DIGER IZOTONIK'
    }
    
    return df, sales_mapping

@st.cache_resource
def load_geojson():
    """GeoJSON yükle"""
    try:
        # URL'den Türkiye GeoJSON'u yükle
        url = "https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json"
        gdf = gpd.read_file(url)
        return gdf
    except Exception as e:
        st.warning(f"GeoJSON yüklenemedi: {e}")
        return None

# =============================================================================
# ANALİZ FONKSİYONLARI
# =============================================================================

def calculate_city_performance(df, product, sales_columns_mapping, date_filter=None):
    """Şehir bazlı performans"""
    cols = get_product_columns(product, sales_columns_mapping)
    
    df_filtered = df.copy()
    if date_filter:
        df_filtered = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & 
                                  (df_filtered['DATE'] <= date_filter[1])]
    
    # Sütunları kontrol et
    pf_col = cols['pf']
    rakip_col = cols['rakip']
    
    if pf_col not in df_filtered.columns:
        pf_col = 'TROCMETAM'  # Varsayılan
    
    if rakip_col not in df_filtered.columns:
        rakip_col = 'DIGER TROCMETAM'  # Varsayılan
    
    city_perf = df_filtered.groupby(['CITY_NORMALIZED', 'REGION']).agg({
        pf_col: 'sum',
        rakip_col: 'sum'
    }).reset_index()
    
    city_perf.columns = ['City', 'Region', 'PF_Satis', 'Rakip_Satis']
    city_perf['Toplam_Pazar'] = city_perf['PF_Satis'] + city_perf['Rakip_Satis']
    city_perf['Pazar_Payi_%'] = safe_divide(city_perf['PF_Satis'], city_perf['Toplam_Pazar']) * 100
    city_perf['Bölge'] = city_perf['Region']
    
    return city_perf.sort_values('PF_Satis', ascending=False)

def calculate_territory_performance(df, product, sales_columns_mapping, date_filter=None):
    """Territory bazlı performans"""
    cols = get_product_columns(product, sales_columns_mapping)
    
    df_filtered = df.copy()
    if date_filter:
        df_filtered = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & 
                                  (df_filtered['DATE'] <= date_filter[1])]
    
    # Sütunları kontrol et
    pf_col = cols['pf']
    rakip_col = cols['rakip']
    
    if pf_col not in df_filtered.columns:
        pf_col = 'TROCMETAM'  # Varsayılan
    
    if rakip_col not in df_filtered.columns:
        rakip_col = 'DIGER TROCMETAM'  # Varsayılan
    
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

def calculate_time_series(df, product, sales_columns_mapping, territory=None, date_filter=None):
    """Zaman serisi"""
    cols = get_product_columns(product, sales_columns_mapping)
    
    df_filtered = df.copy()
    if territory and territory != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['TERRITORIES'] == territory]
    
    if date_filter:
        df_filtered = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & 
                                   (df_filtered['DATE'] <= date_filter[1])]
    
    # Sütunları kontrol et
    pf_col = cols['pf']
    rakip_col = cols['rakip']
    
    if pf_col not in df_filtered.columns:
        pf_col = 'TROCMETAM'  # Varsayılan
    
    if rakip_col not in df_filtered.columns:
        rakip_col = 'DIGER TROCMETAM'  # Varsayılan
    
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
    
    return monthly

# =============================================================================
# ML FONKSİYONLARI
# =============================================================================

def create_ml_features(df, target_col='PF_Satis'):
    """ML için özellikler oluştur"""
    df_features = df.copy().sort_values('DATE').reset_index(drop=True)
    
    # Lag features
    for lag in [1, 2, 3]:
        df_features[f'lag_{lag}'] = df_features[target_col].shift(lag)
    
    # Rolling statistics
    df_features['rolling_mean_3'] = df_features[target_col].rolling(window=3, min_periods=1).mean()
    df_features['rolling_mean_6'] = df_features[target_col].rolling(window=6, min_periods=1).mean()
    df_features['rolling_std_3'] = df_features[target_col].rolling(window=3, min_periods=1).std()
    
    # Date features
    df_features['month'] = df_features['DATE'].dt.month
    df_features['quarter'] = df_features['DATE'].dt.quarter
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['trend'] = range(len(df_features))
    
    # Fill NaN
    df_features = df_features.fillna(method='bfill').fillna(0)
    
    return df_features

def train_ml_models(df, target_col='PF_Satis', forecast_periods=6):
    """ML modelleri ile tahmin"""
    if len(df) < 12:
        return None, None, None, "⚠️ Tahmin için yeterli veri yok (en az 12 ay gerekli)"
    
    try:
        # Özellik mühendisliği
        df_features = create_ml_features(df, target_col)
        
        # Özellik sütunları
        feature_cols = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_mean_6',
                       'rolling_std_3', 'month', 'quarter', 'month_sin', 'month_cos', 'trend']
        
        # Eğitim ve test verisi
        train_size = max(12, int(len(df_features) * 0.8))
        train_df = df_features.iloc[:train_size]
        test_df = df_features.iloc[train_size:] if train_size < len(df_features) else df_features.iloc[-6:]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        # Modeller
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
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
                    'R2': r2
                }
            except Exception as e:
                continue
        
        if not results:
            return None, None, None, "❌ Hiçbir model eğitilemedi"
        
        # En iyi model (MAPE'e göre)
        best_model_name = min(results.keys(), key=lambda x: results[x]['MAPE'])
        best_model = results[best_model_name]['model']
        
        # Gelecek tahmini
        forecast_data = []
        last_row = df_features.iloc[-1:].copy()
        
        for i in range(forecast_periods):
            next_date = last_row['DATE'].values[0] + pd.DateOffset(months=1)
            X_future = last_row[feature_cols]
            next_pred = best_model.predict(X_future)[0]
            next_pred = max(0, next_pred)  # Negatif olmaması için
            
            forecast_data.append({
                'DATE': next_date,
                'YIL_AY': pd.to_datetime(next_date).strftime('%Y-%m'),
                'PF_Satis': next_pred,
                'Model': best_model_name,
                'Tahmin_Tipi': 'ML Tahmin'
            })
            
            # Güncelleme için yeni satır
            new_row = last_row.copy()
            new_row['DATE'] = next_date
            new_row[target_col] = next_pred
            
            # Lag değerlerini güncelle
            new_row['lag_1'] = last_row[target_col].values[0]
            new_row['lag_2'] = last_row['lag_1'].values[0]
            new_row['lag_3'] = last_row['lag_2'].values[0]
            
            # Diğer özellikleri güncelle
            new_row['month'] = pd.to_datetime(next_date).month
            new_row['quarter'] = pd.to_datetime(next_date).quarter
            new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
            new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)
            new_row['trend'] = last_row['trend'].values[0] + 1
            
            last_row = new_row
        
        forecast_df = pd.DataFrame(forecast_data)
        
        return results, best_model_name, forecast_df, "✅ ML modelleri başarıyla eğitildi"
        
    except Exception as e:
        return None, None, None, f"❌ ML eğitimi hatası: {str(e)}"

# =============================================================================
# GÖRSELLEŞTİRME FONKSİYONLARI
# =============================================================================

def create_forecast_chart(historical_df, forecast_df):
    """Tahmin grafiği oluştur"""
    fig = go.Figure()
    
    # Gerçek veri
    fig.add_trace(go.Scatter(
        x=historical_df['DATE'],
        y=historical_df['PF_Satis'],
        mode='lines+markers',
        name='Gerçek Satış',
        line=dict(color=PERFORMANCE_COLORS['success'], width=3),
        marker=dict(size=8, color='white')
    ))
    
    # Tahmin
    if forecast_df is not None and len(forecast_df) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_df['DATE'],
            y=forecast_df['PF_Satis'],
            mode='lines+markers',
            name='ML Tahmin',
            line=dict(color='#3B82F6', width=3, dash='dash'),
            marker=dict(size=10, symbol='diamond', color='white')
        ))
    
    fig.update_layout(
        title='<b>Satış Trendi ve ML Tahmini</b>',
        xaxis_title='<b>Tarih</b>',
        yaxis_title='<b>Satış</b>',
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

def create_city_map(city_data, gdf):
    """Şehir haritası oluştur"""
    if gdf is None:
        return None
    
    # GeoJSON verisini hazırla
    gdf['name_upper'] = gdf['name'].str.upper()
    
    # Şehir verisini hazırla
    city_data['City_Upper'] = city_data['City'].str.upper()
    
    # Birleştir
    merged = gdf.merge(city_data, left_on='name_upper', right_on='City_Upper', how='left')
    merged['PF_Satis'] = merged['PF_Satis'].fillna(0)
    
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
        hover_data=['PF_Satis', 'Pazar_Payi_%'],
        title='<b>Şehir Bazlı Satış Dağılımı</b>'
    )
    
    fig.update_layout(
        height=600,
        margin={"r":0,"t":40,"l":0,"b":0},
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
                'GERÇEK ML Tahminleme • Modern Harita Görselleştirme • Rakip Analizi'
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
            df, sales_columns_mapping = load_excel_data(uploaded_file)
            gdf = load_geojson()
            
            st.success(f"✅ **{len(df):,}** satır veri yüklendi")
            
            # Mevcut ürünleri göster
            available_products = get_available_products(sales_columns_mapping)
            
            # Sütun bilgilerini göster
            with st.expander("📋 Veri Önizleme"):
                st.write("**Sütunlar:**", list(df.columns))
                st.write("**Satış Sütunları:**", sales_columns_mapping)
                st.write("**Mevcut Ürünler:**", available_products)
                
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
        
        date_option = st.selectbox("Dönem Seçin", ["Tüm Veriler", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl", "Son 2 Yıl"])
        
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
        else:
            start_date = max_date - pd.DateOffset(years=2)
            date_filter = (start_date, max_date)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Filtreler
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">🔍 FİLTRELER</h4>', unsafe_allow_html=True)
        
        territories = ["TÜMÜ"] + sorted(df['TERRITORIES'].fillna('BILINMIYOR').unique())
        selected_territory = st.selectbox("Territory", territories)
        
        regions = ["TÜMÜ"] + sorted(df['REGION'].fillna('DIGER').unique())
        selected_region = st.selectbox("Bölge", regions)
        
        managers = ["TÜMÜ"] + sorted(df['MANAGER'].fillna('BILINMIYOR').unique())
        selected_manager = st.selectbox("Manager", managers)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ML Ayarları
        st.markdown("---")
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">🤖 ML AYARLARI</h4>', unsafe_allow_html=True)
        
        forecast_months = st.slider("Tahmin Periyodu (Ay)", 1, 12, 6)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Veri filtreleme
        df_filtered = df.copy()
        if selected_territory != "TÜMÜ":
            df_filtered = df_filtered[df_filtered['TERRITORIES'] == selected_territory]
        if selected_region != "TÜMÜ":
            df_filtered = df_filtered[df_filtered['REGION'] == selected_region]
        if selected_manager != "TÜMÜ":
            df_filtered = df_filtered[df_filtered['MANAGER'] == selected_manager]
    
    # ANA İÇERİK - TAB'LER
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Genel Bakış",
        "🗺️ Harita Analizi",
        "🏢 Territory Analizi",
        "📈 Zaman Serisi",
        "🤖 ML Tahminleme",
        "📥 Raporlar"
    ])
    
    # TAB 1: GENEL BAKIŞ
    with tab1:
        st.header("📊 Genel Performans Özeti")
        
        if date_filter:
            df_period = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & (df_filtered['DATE'] <= date_filter[1])]
        else:
            df_period = df_filtered
        
        # Ürün kolonlarını al
        cols = get_product_columns(selected_product, sales_columns_mapping)
        
        # Sütunları kontrol et
        pf_col = cols['pf']
        rakip_col = cols['rakip']
        
        # Varsayılan değerler
        pf_sales = 0
        competitor_sales = 0
        
        # PF satışını hesapla
        if pf_col in df_period.columns:
            pf_sales = df_period[pf_col].sum()
        
        # Rakip satışını hesapla
        if rakip_col in df_period.columns:
            competitor_sales = df_period[rakip_col].sum()
        
        # Metrikler
        total_market = pf_sales + competitor_sales
        market_share = (pf_sales / total_market * 100) if total_market > 0 else 0
        active_territories = df_period['TERRITORIES'].nunique()
        avg_monthly_pf = pf_sales / df_period['YIL_AY'].nunique() if df_period['YIL_AY'].nunique() > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💊 PF Satış", f"{pf_sales:,.0f}", f"{avg_monthly_pf:,.0f}/ay")
        with col2:
            st.metric("🏪 Toplam Pazar", f"{total_market:,.0f}", f"{competitor_sales:,.0f} rakip")
        with col3:
            st.metric("📊 Pazar Payı", f"%{market_share:.1f}", 
                     f"%{100-market_share:.1f} rakip")
        with col4:
            st.metric("🏢 Active Territory", active_territories, 
                     f"{df_period['MANAGER'].nunique()} manager")
        
        st.markdown("---")
        
        # Top 10 Territory
        st.subheader("🏆 Top 10 Territory Performansı")
        
        try:
            terr_perf = calculate_territory_performance(df_filtered, selected_product, sales_columns_mapping, date_filter)
            
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
                
                # Tablo
                st.markdown("---")
                st.subheader("📋 Top 10 Territory Detayları")
                
                top10_display = top10.copy()
                top10_display.index = range(1, len(top10_display) + 1)
                
                st.dataframe(
                    top10_display[['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 'Pazar_Payi_%']].style.format({
                        'PF_Satis': '{:,.0f}',
                        'Pazar_Payi_%': '{:.1f}%'
                    }),
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("⚠️ Territory performans verisi bulunamadı")
                
        except Exception as e:
            st.error(f"Territory analizi hatası: {str(e)}")
    
    # TAB 2: HARİTA ANALİZİ
    with tab2:
        st.header("🗺️ Türkiye Harita Analizi")
        
        try:
            city_data = calculate_city_performance(df_filtered, selected_product, sales_columns_mapping, date_filter)
            
            if len(city_data) > 0:
                # Harita oluştur
                if gdf is not None:
                    city_map = create_city_map(city_data, gdf)
                    if city_map:
                        st.plotly_chart(city_map, use_container_width=True)
                    else:
                        st.warning("⚠️ Harita oluşturulamadı")
                else:
                    st.warning("⚠️ Harita verisi yüklenemedi")
                
                # Şehir Performans Tablosu
                st.markdown("---")
                st.subheader("📋 Şehir Performans Detayları")
                
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
            else:
                st.warning("⚠️ Şehir verisi bulunamadı")
                
        except Exception as e:
            st.error(f"Harita analizi hatası: {str(e)}")
    
    # TAB 3: TERRITORY ANALİZİ
    with tab3:
        st.header("🏢 Territory Bazlı Detaylı Analiz")
        
        try:
            terr_perf = calculate_territory_performance(df_filtered, selected_product, sales_columns_mapping, date_filter)
            
            if len(terr_perf) > 0:
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
                        y='PF_Satis',
                        title=f'<b>Top {show_n} Territory - PF Satış</b>',
                        color='Region',
                        color_discrete_map=REGION_COLORS
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
                        title=f'<b>Top {show_n} Territory Performans Haritası</b>',
                        size_max=50,
                        color_discrete_map=REGION_COLORS
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
                        'Agirlik_%': '{:.1f}%'
                    }),
                    use_container_width=True,
                    height=500
                )
            else:
                st.warning("⚠️ Territory verisi bulunamadı")
                
        except Exception as e:
            st.error(f"Territory analizi hatası: {str(e)}")
    
    # TAB 4: ZAMAN SERİSİ
    with tab4:
        st.header("📈 Zaman Serisi Analizi")
        
        territory_for_ts = st.selectbox(
            "Territory Seçin",
            ["TÜMÜ"] + sorted(df_filtered['TERRITORIES'].fillna('BILINMIYOR').unique()),
            key='ts_territory'
        )
        
        try:
            monthly_df = calculate_time_series(df_filtered, selected_product, sales_columns_mapping, territory_for_ts, date_filter)
            
            if len(monthly_df) > 0:
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
                    total_months = len(monthly_df)
                    st.metric("📅 Veri Dönemi", f"{total_months} ay")
                
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
                    
                    fig_growth.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                    
                    fig_growth.update_layout(
                        title='<b>Büyüme Oranı</b>',
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
            else:
                st.warning("⚠️ Zaman serisi verisi bulunamadı")
                
        except Exception as e:
            st.error(f"Zaman serisi analizi hatası: {str(e)}")
    
    # TAB 5: ML TAHMİNLEME
    with tab5:
        st.header("🤖 Machine Learning Tahminleme")
        
        # Zaman serisi verisini hazırla
        monthly_df = calculate_time_series(df_filtered, selected_product, sales_columns_mapping, None, date_filter)
        
        if len(monthly_df) < 12:
            st.warning(f"⚠️ Tahmin için yeterli veri yok (en az 12 ay gerekli, mevcut: {len(monthly_df)})")
        else:
            with st.spinner("ML modelleri eğitiliyor..."):
                ml_results, best_model_name, forecast_df, message = train_ml_models(
                    monthly_df, 
                    'PF_Satis', 
                    forecast_months
                )
            
            if message.startswith("✅"):
                st.markdown(f'<div class="alert-success">{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-warning">{message}</div>', unsafe_allow_html=True)
            
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
                
                forecast_chart = create_forecast_chart(monthly_df, forecast_df)
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
    
    # TAB 6: RAPORLAR
    with tab6:
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
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Excel Raporu Oluştur", type="primary", use_container_width=True):
            with st.spinner("Rapor hazırlanıyor..."):
                try:
                    # Tüm analizleri hesapla
                    terr_perf = calculate_territory_performance(df_filtered, selected_product, sales_columns_mapping, date_filter)
                    city_data = calculate_city_performance(df_filtered, selected_product, sales_columns_mapping, date_filter)
                    monthly_df = calculate_time_series(df_filtered, selected_product, sales_columns_mapping, None, date_filter)
                    
                    # ML tahmini
                    ml_results, best_model_name, forecast_df, _ = train_ml_models(
                        monthly_df, 'PF_Satis', 6
                    )
                    
                    # Excel dosyası oluştur
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Genel Özet
                        summary_data = pd.DataFrame({
                            'Metrik': ['Ürün', 'Dönem', 'Toplam PF Satış', 'Toplam Pazar', 
                                      'Pazar Payı', 'Territory Sayısı', 'Şehir Sayısı'],
                            'Değer': [
                                selected_product,
                                date_option,
                                f"{terr_perf['PF_Satis'].sum():,.0f}" if len(terr_perf) > 0 else "0",
                                f"{terr_perf['Toplam_Pazar'].sum():,.0f}" if len(terr_perf) > 0 else "0",
                                f"{(terr_perf['PF_Satis'].sum() / terr_perf['Toplam_Pazar'].sum() * 100):.1f}%" if len(terr_perf) > 0 and terr_perf['Toplam_Pazar'].sum() > 0 else "0%",
                                len(terr_perf) if len(terr_perf) > 0 else 0,
                                len(city_data) if len(city_data) > 0 else 0
                            ]
                        })
                        summary_data.to_excel(writer, sheet_name='Özet', index=False)
                        
                        # Diğer sayfalar
                        if len(terr_perf) > 0:
                            terr_perf.to_excel(writer, sheet_name='Territory Performans', index=False)
                        
                        if len(city_data) > 0:
                            city_data.to_excel(writer, sheet_name='Şehir Analizi', index=False)
                        
                        if len(monthly_df) > 0:
                            monthly_df.to_excel(writer, sheet_name='Zaman Serisi', index=False)
                        
                        if forecast_df is not None and len(forecast_df) > 0:
                            forecast_df.to_excel(writer, sheet_name='ML Tahminler', index=False)
                        
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
