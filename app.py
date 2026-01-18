"""
🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ
Territory Bazlı Performans, ML Tahminleme, Türkiye Haritası ve Rekabet Analizi

Özellikler:
- 🗺️ Türkiye il bazlı harita görselleştirme (GELİŞMİŞ VERSİYON)
- 🤖 GERÇEK Machine Learning (Linear Regression, Ridge, Random Forest)
- 📊 Aylık/Yıllık dönem seçimi
- 📈 Gelişmiş rakip analizi ve trend karşılaştırması
- 🎯 Dinamik zaman aralığı filtreleme
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import warnings
import requests
from urllib.request import urlopen

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
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SADE RENK PALETİ
# =============================================================================
# Sade ve profesyonel bölge renkleri
REGION_COLORS = {
    "MARMARA": "#3B82F6",      # Mavi
    "BATI ANADOLU": "#10B981", # Yeşil
    "EGE": "#F59E0B",          # Turuncu
    "İÇ ANADOLU": "#8B5CF6",   # Mor
    "GÜNEY DOĞU ANADOLU": "#EF4444", # Kırmızı
    "KUZEY ANADOLU": "#06B6D4",      # Camgöbeği
    "KARADENİZ": "#06B6D4",          # Camgöbeği
    "AKDENİZ": "#3B82F6",            # Mavi
    "DOĞU ANADOLU": "#10B981",       # Yeşil
    "DİĞER": "#64748B"               # Gri
}

# PERFORMANS RENKLERİ - Sade
PERFORMANCE_COLORS = {
    "high": "#1F7A5A",       # Koyu Yeşil – Yüksek Performans
    "medium": "#C48A2A",     # Altın Sarısı – Orta Performans
    "low": "#B23A3A",        # Bordo – Düşük Performans
    "positive": "#1F7A5A",   # Koyu Yeşil – Pozitif
    "negative": "#B23A3A",   # Bordo – Negatif
    "neutral": "#6B7280",    # Kurumsal Gri – Nötr
    "warning": "#C48A2A",    # Altın – Uyarı
    "info": "#1E40AF",       # Lacivert – Bilgi
    "success": "#166534",    # Koyu Yeşil – Başarı
    "danger": "#991B1B"      # Koyu Kırmızı – Risk / Tehlike
}

# BCG MATRIX RENKLERİ
BCG_COLORS = {
    "⭐ Star": "#F59E0B",      # Turuncu
    "🐄 Cash Cow": "#10B981",  # Yeşil
    "❓ Question Mark": "#3B82F6",  # Mavi
    "🐶 Dog": "#64748B"        # Gri
}

# YATIRIM STRATEJİSİ RENKLERİ
STRATEGY_COLORS = {
    "🚀 Agresif": "#EF4444",      # Kırmızı
    "⚡ Hızlandırılmış": "#F59E0B",  # Turuncu
    "🛡️ Koruma": "#10B981",        # Yeşil
    "💎 Potansiyel": "#3B82F6",     # Mavi
    "👁️ İzleme": "#64748B"         # Gri
}

# GRADIENT SCALES for Visualizations
GRADIENT_SCALES = {
    "blue_green": ["#3B82F6", "#06B6D4", "#10B981"],
    "sequential_blue": ["#DBEAFE", "#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6"],
    "diverging": ["#EF4444", "#F59E0B", "#10B981", "#3B82F6", "#8B5CF6"],
    "temperature": ["#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE"]
}

# =============================================================================
# CONSTANTS
# =============================================================================

# Excel'deki sütun isimlerini kontrol etmek için
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
# HELPER FUNCTIONS - DÜZELTİLMİŞ
# =============================================================================

def safe_divide_series(a, b):
    """Güvenli bölme işlemi - pandas Series için"""
    # Series'i numpy array'e çevir
    a_array = np.array(a)
    b_array = np.array(b)
    
    # Bölme işlemi
    result = np.zeros_like(a_array, dtype=float)
    mask = b_array != 0
    result[mask] = a_array[mask] / b_array[mask]
    
    # Sonsuz değerleri temizle
    result = np.where(np.isinf(result), 0, result)
    result = np.where(np.isnan(result), 0, result)
    
    return result

def safe_divide(a, b):
    """Güvenli bölme işlemi - skaler veya array için"""
    if isinstance(a, (pd.Series, np.ndarray)) or isinstance(b, (pd.Series, np.ndarray)):
        return safe_divide_series(a, b)
    
    # Skaler değerler için
    if b == 0:
        return 0
    return a / b

def get_product_columns(product, df_columns):
    """Excel'deki sütun isimlerini kontrol ederek döndür"""
    product_map = COLUMN_NAMES.get(product, {})
    
    # Sütun isimlerini kontrol et ve mevcut olanları kullan
    available_columns = {}
    for key, col_name in product_map.items():
        # Sütun ismini farklı formatlarda kontrol et
        variations = [
            col_name,
            col_name.lower(),
            col_name.upper(),
            col_name.replace(' ', '_'),
            col_name.replace(' ', ''),
            col_name.replace(' ', ' ').strip()
        ]
        
        for var in variations:
            if var in df_columns:
                available_columns[key] = var
                break
        
        # Eğer bulunamadıysa, sütun adını verilen haliyle kullan
        if key not in available_columns:
            available_columns[key] = col_name
    
    return available_columns

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

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_excel_data(file):
    """Excel dosyasını yükle ve sütun isimlerini normalize et"""
    try:
        df = pd.read_excel(file)
        
        # Sütun isimlerini normalize et (boşlukları temizle, büyük harf yap)
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        # Tarih sütununu işle
        date_columns = ['DATE', 'TARIH', 'TARİH', 'YEAR_MONTH', 'AY-YIL', 'AY_YIL']
        date_column_found = False
        
        for date_col in date_columns:
            if date_col in df.columns:
                df['DATE'] = pd.to_datetime(df[date_col], errors='coerce')
                date_column_found = True
                break
        
        if not date_column_found:
            # Tarih sütunu bulunamadı, ilk sütunu tarih olarak kullanmaya çalış
            df['DATE'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        
        # NaN tarihleri temizle
        df = df.dropna(subset=['DATE'])
        
        # Diğer tarih sütunlarını oluştur
        df['YIL_AY'] = df['DATE'].dt.strftime('%Y-%m')
        df['AY'] = df['DATE'].dt.month
        df['YIL'] = df['DATE'].dt.year
        
        # Territor ve şehir sütunlarını normalize et
        territory_columns = ['TERRITORIES', 'TERRITORY', 'TERRİTORY', 'TERRITOR', 'BOLGE', 'BÖLGE']
        city_columns = ['CITY', 'CİTY', 'SEHIR', 'ŞEHİR', 'İL']
        region_columns = ['REGION', 'REGİON', 'BOLGE', 'BÖLGE']
        manager_columns = ['MANAGER', 'SATIS_TEMSILCISI', 'SORUMLU']
        
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
        df['CITY_NORMALIZED'] = df['CITY'].apply(normalize_city_name_fixed)
        
        return df
    
    except Exception as e:
        st.error(f"❌ Veri yükleme hatası: {str(e)}")
        st.stop()

@st.cache_resource
def load_turkey_geojson():
    """Türkiye GeoJSON verisini yükle"""
    try:
        # İnternetten Türkiye GeoJSON verisini al
        url = "https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json"
        response = requests.get(url)
        
        if response.status_code == 200:
            geojson_data = response.json()
            
            # GeoPandas ile işle
            gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
            
            # Şehir isimlerini normalize et
            gdf['name_normalized'] = gdf['name'].apply(lambda x: normalize_city_name_fixed(x))
            
            return gdf, geojson_data
        else:
            st.warning("⚠️ İnternetten GeoJSON yüklenemedi, örnek veri kullanılıyor...")
            return create_sample_geojson()
    
    except Exception as e:
        st.warning(f"⚠️ GeoJSON yükleme hatası: {str(e)}. Örnek veri kullanılıyor...")
        return create_sample_geojson()

def create_sample_geojson():
    """Örnek Türkiye GeoJSON verisi oluştur"""
    # Basit bir Türkiye haritası için örnek koordinatlar
    turkey_cities_coords = {
        'İstanbul': {'lon': 28.9795, 'lat': 41.0151},
        'Ankara': {'lon': 32.8597, 'lat': 39.9334},
        'İzmir': {'lon': 27.1428, 'lat': 38.4237},
        'Bursa': {'lon': 29.0588, 'lat': 40.1885},
        'Antalya': {'lon': 30.7133, 'lat': 36.8969},
        'Adana': {'lon': 35.3213, 'lat': 37.0000},
        'Konya': {'lon': 32.4846, 'lat': 37.9838},
        'Gaziantep': {'lon': 37.3826, 'lat': 37.0662},
        'Kayseri': {'lon': 35.4833, 'lat': 38.7312},
        'Mersin': {'lon': 34.6415, 'lat': 36.8121}
    }
    
    features = []
    for city, coords in turkey_cities_coords.items():
        feature = {
            "type": "Feature",
            "properties": {"name": city},
            "geometry": {
                "type": "Point",
                "coordinates": [coords['lon'], coords['lat']]
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
# MODERN HARİTA OLUŞTURUCU - GÜNCELLENMİŞ
# =============================================================================

def create_modern_turkey_map(city_data, gdf, geojson_data, title="Türkiye Satış Haritası"):
    """
    Modern Türkiye haritası - Düzeltilmiş versiyon
    """
    try:
        # Veriyi hazırla
        city_data = city_data.copy()
        city_data['City_Fixed'] = city_data['City'].apply(normalize_city_name_fixed)
        
        if city_data['City_Fixed'].isnull().all():
            st.warning("⚠️ Şehir verileri eşleştirilemiyor.")
            return None
        
        # Bubble haritası oluştur (daha basit ve güvenilir)
        fig = go.Figure()
        
        # Koordinatları ayarla
        city_data['lon'] = 35.0  # Türkiye merkezi
        city_data['lat'] = 39.0  # Türkiye merkezi
        
        # Büyük şehirler için özel koordinatlar
        city_coords = {
            'İstanbul': (28.9795, 41.0151),
            'Ankara': (32.8597, 39.9334),
            'İzmir': (27.1428, 38.4237),
            'Bursa': (29.0588, 40.1885),
            'Antalya': (30.7133, 36.8969),
            'Adana': (35.3213, 37.0000),
            'Konya': (32.4846, 37.9838),
            'Gaziantep': (37.3826, 37.0662),
            'Kayseri': (35.4833, 38.7312),
            'Mersin': (34.6415, 36.8121)
        }
        
        for idx, row in city_data.iterrows():
            city_name = row['City_Fixed']
            if city_name in city_coords:
                city_data.at[idx, 'lon'] = city_coords[city_name][0]
                city_data.at[idx, 'lat'] = city_coords[city_name][1]
            else:
                # Rastgele dağıt
                city_data.at[idx, 'lon'] = np.random.uniform(26, 44)
                city_data.at[idx, 'lat'] = np.random.uniform(36, 42)
        
        # Bubble boyutunu ayarla
        max_sales = city_data['PF_Satis'].max() if city_data['PF_Satis'].max() > 0 else 1
        city_data['bubble_size'] = (city_data['PF_Satis'] / max_sales * 50) + 10
        
        fig.add_trace(go.Scattermapbox(
            lat=city_data['lat'],
            lon=city_data['lon'],
            mode='markers',
            marker=dict(
                size=city_data['bubble_size'],
                color=city_data['Pazar_Payi_%'],
                colorscale='RdYlGn',
                cmin=0,
                cmax=100,
                showscale=True,
                colorbar=dict(
                    title="Pazar Payı %",
                    thickness=20,
                    titleside="right"
                ),
                opacity=0.8
            ),
            text=city_data['City_Fixed'] + '<br>PF Satış: ' + city_data['PF_Satis'].astype(str) + '<br>Pazar Payı: ' + city_data['Pazar_Payi_%'].round(1).astype(str) + '%',
            hoverinfo='text'
        ))
        
        # Modern layout ayarları
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox=dict(
                center=dict(lat=39.0, lon=35.0),
                zoom=5,
                bearing=0,
                pitch=0
            ),
            height=700,
            margin=dict(l=0, r=0, t=80, b=0),
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(size=24, color='white'),
                y=0.95
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            hoverlabel=dict(
                bgcolor="rgba(15, 23, 41, 0.9)",
                font_size=12,
                font_family="Inter, sans-serif"
            )
        )
        
        return fig
    
    except Exception as e:
        st.error(f"❌ Harita oluşturma hatası: {str(e)}")
        return None

# =============================================================================
# ANALYSIS FUNCTIONS - GÜNCELLENMİŞ
# =============================================================================

def calculate_city_performance(df, product, date_filter=None):
    """Şehir bazlı performans"""
    # Sütun isimlerini kontrol et
    cols = get_product_columns(product, df.columns)
    
    if date_filter:
        df_filtered = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])].copy()
    else:
        df_filtered = df.copy()
    
    # Sütunların mevcut olup olmadığını kontrol et
    pf_col = cols.get('pf')
    rakip_col = cols.get('rakip')
    
    if pf_col not in df_filtered.columns:
        st.error(f"❌ '{pf_col}' sütunu veri setinde bulunamadı!")
        return pd.DataFrame()
    
    # Rakip sütunu yoksa, sıfır olarak kabul et
    if rakip_col not in df_filtered.columns:
        df_filtered[rakip_col] = 0
    
    # Gruplama yap
    city_perf = df_filtered.groupby(['CITY_NORMALIZED', 'REGION']).agg({
        pf_col: 'sum',
        rakip_col: 'sum'
    }).reset_index()
    
    city_perf.columns = ['City', 'Region', 'PF_Satis', 'Rakip_Satis']
    city_perf['Toplam_Pazar'] = city_perf['PF_Satis'] + city_perf['Rakip_Satis']
    city_perf['Pazar_Payi_%'] = safe_divide(city_perf['PF_Satis'], city_perf['Toplam_Pazar']) * 100
    city_perf['Bölge'] = city_perf['Region']
    
    # Sadece PF satışı olan şehirleri filtrele
    city_perf = city_perf[city_perf['PF_Satis'] > 0]
    
    return city_perf

def calculate_territory_performance(df, product, date_filter=None):
    """Territory bazlı performans - DÜZELTİLMİŞ"""
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
    
    # Gruplama yap
    terr_perf = df_filtered.groupby(['TERRITORIES', 'REGION', 'CITY', 'MANAGER']).agg({
        pf_col: 'sum',
        rakip_col: 'sum'
    }).reset_index()
    
    terr_perf.columns = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 'Rakip_Satis']
    terr_perf['Toplam_Pazar'] = terr_perf['PF_Satis'] + terr_perf['Rakip_Satis']
    
    # DÜZELTME: safe_divide fonksiyonunu doğrudan kullan
    terr_perf['Pazar_Payi_%'] = terr_perf.apply(
        lambda row: safe_divide(row['PF_Satis'], row['Toplam_Pazar']) * 100, 
        axis=1
    )
    
    total_pf = terr_perf['PF_Satis'].sum()
    
    # DÜZELTME: Agirlik_% hesaplaması
    terr_perf['Agirlik_%'] = terr_perf.apply(
        lambda row: safe_divide(row['PF_Satis'], total_pf) * 100 if total_pf > 0 else 0,
        axis=1
    )
    
    # DÜZELTME: Goreceli_Pazar_Payi hesaplaması
    terr_perf['Goreceli_Pazar_Payi'] = terr_perf.apply(
        lambda row: safe_divide(row['PF_Satis'], row['Rakip_Satis']),
        axis=1
    )
    
    return terr_perf.sort_values('PF_Satis', ascending=False)

def calculate_time_series(df, product, territory=None, date_filter=None):
    """Zaman serisi"""
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
    
    monthly = df_filtered.groupby('YIL_AY').agg({
        pf_col: 'sum',
        rakip_col: 'sum',
        'DATE': 'first'
    }).reset_index().sort_values('YIL_AY')
    
    monthly.columns = ['YIL_AY', 'PF_Satis', 'Rakip_Satis', 'DATE']
    monthly['Toplam_Pazar'] = monthly['PF_Satis'] + monthly['Rakip_Satis']
    
    # DÜZELTME: safe_divide kullanımı
    monthly['Pazar_Payi_%'] = monthly.apply(
        lambda row: safe_divide(row['PF_Satis'], row['Toplam_Pazar']) * 100,
        axis=1
    )
    
    # Büyüme hesaplamaları
    monthly['PF_Buyume_%'] = monthly['PF_Satis'].pct_change() * 100
    monthly['Rakip_Buyume_%'] = monthly['Rakip_Satis'].pct_change() * 100
    monthly['Goreceli_Buyume_%'] = monthly['PF_Buyume_%'] - monthly['Rakip_Buyume_%']
    monthly['MA_3'] = monthly['PF_Satis'].rolling(window=3, min_periods=1).mean()
    monthly['MA_6'] = monthly['PF_Satis'].rolling(window=6, min_periods=1).mean()
    
    return monthly.fillna(0)

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
    
    monthly = df_filtered.groupby('YIL_AY').agg({
        pf_col: 'sum',
        rakip_col: 'sum'
    }).reset_index().sort_values('YIL_AY')
    
    monthly.columns = ['YIL_AY', 'PF', 'Rakip']
    
    # DÜZELTME: safe_divide kullanımı
    monthly['PF_Pay_%'] = monthly.apply(
        lambda row: safe_divide(row['PF'], (row['PF'] + row['Rakip'])) * 100,
        axis=1
    )
    
    monthly['Rakip_Pay_%'] = 100 - monthly['PF_Pay_%']
    monthly['PF_Buyume'] = monthly['PF'].pct_change() * 100
    monthly['Rakip_Buyume'] = monthly['Rakip'].pct_change() * 100
    monthly['Fark'] = monthly['PF_Buyume'] - monthly['Rakip_Buyume']
    
    return monthly.fillna(0)

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_modern_forecast_chart(historical_df, forecast_df):
    """Modern tahmin grafiği"""
    fig = go.Figure()
    
    # Gerçek veri
    fig.add_trace(go.Scatter(
        x=historical_df['DATE'],
        y=historical_df['PF_Satis'],
        mode='lines+markers',
        name='Gerçek Satış',
        line=dict(
            color=PERFORMANCE_COLORS['success'],
            width=3,
            shape='spline'
        ),
        marker=dict(
            size=8,
            color='white',
            line=dict(width=2, color=PERFORMANCE_COLORS['success'])
        ),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    
    # Tahmin
    if forecast_df is not None and len(forecast_df) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_df['DATE'],
            y=forecast_df['PF_Satis'],
            mode='lines+markers',
            name='Tahmin',
            line=dict(
                color=PERFORMANCE_COLORS['info'],
                width=3,
                dash='dash',
                shape='spline'
            ),
            marker=dict(
                size=10,
                symbol='diamond',
                color='white',
                line=dict(width=2, color=PERFORMANCE_COLORS['info'])
            ),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
    
    # Modern layout
    fig.update_layout(
        title=dict(
            text='<b>Satış Trendi ve ML Tahmin</b>',
            font=dict(size=20, color='white')
        ),
        xaxis_title='<b>Tarih</b>',
        yaxis_title='<b>PF Satış</b>',
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
            x=1,
            bgcolor='rgba(30, 41, 59, 0.8)'
        )
    )
    
    return fig

def create_modern_competitor_chart(comp_data):
    """Modern rakip karşılaştırma"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=comp_data['YIL_AY'],
        y=comp_data['PF'],
        name='PF',
        marker_color=PERFORMANCE_COLORS['success'],
        marker=dict(
            line=dict(width=2, color='rgba(255, 255, 255, 0.8)')
        )
    ))
    
    fig.add_trace(go.Bar(
        x=comp_data['YIL_AY'],
        y=comp_data['Rakip'],
        name='Rakip',
        marker_color=PERFORMANCE_COLORS['danger'],
        marker=dict(
            line=dict(width=2, color='rgba(255, 255, 255, 0.8)')
        )
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>PF vs Rakip Satış Karşılaştırması</b>',
            font=dict(size=20, color='white')
        ),
        xaxis_title='<b>Ay</b>',
        yaxis_title='<b>Satış</b>',
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
            x=1,
            bgcolor='rgba(30, 41, 59, 0.8)'
        )
    )
    
    return fig

# =============================================================================
# MODERN DATA TABLE STYLING
# =============================================================================

def style_dataframe(df, color_column=None, gradient_columns=None):
    """Modern dataframe stilini uygula"""
    if gradient_columns is None:
        gradient_columns = []
    
    styled_df = df.style
    
    # Genel stil
    styled_df = styled_df.set_properties(**{
        'background-color': 'rgba(30, 41, 59, 0.7)',
        'color': '#e2e8f0',
        'border': '1px solid rgba(59, 130, 246, 0.2)',
        'font-family': 'Inter, sans-serif'
    })
    
    # Başlık satırı
    styled_df = styled_df.set_table_styles([{
        'selector': 'thead th',
        'props': [
            ('background-color', 'rgba(59, 130, 246, 0.3)'),
            ('color', 'white'),
            ('font-weight', '700'),
            ('border', '1px solid rgba(59, 130, 246, 0.4)'),
            ('padding', '12px 8px'),
            ('text-align', 'center')
        ]
    }])
    
    # Hücreler
    styled_df = styled_df.set_table_styles([{
        'selector': 'td',
        'props': [
            ('padding', '10px 8px'),
            ('text-align', 'center')
        ]
    }])
    
    # Gradient columns
    for col in gradient_columns:
        if col in df.columns:
            try:
                styled_df = styled_df.background_gradient(
                    subset=[col], 
                    cmap='RdYlGn',
                    vmin=df[col].min() if len(df) > 0 else 0,
                    vmax=df[col].max() if len(df) > 0 else 100
                )
            except:
                pass
    
    # Renk sütunu
    if color_column and color_column in df.columns:
        def color_cells(val):
            if isinstance(val, (int, float)):
                if val >= 70:
                    return 'background-color: rgba(16, 185, 129, 0.3); color: #10B981; font-weight: 600'
                elif val >= 40:
                    return 'background-color: rgba(245, 158, 11, 0.3); color: #F59E0B; font-weight: 600'
                else:
                    return 'background-color: rgba(239, 68, 68, 0.3); color: #EF4444; font-weight: 600'
            return ''
        
        styled_df = styled_df.applymap(color_cells, subset=[color_column])
    
    return styled_df

# =============================================================================
# MAIN APP - DÜZELTİLMİŞ
# =============================================================================

def main():
    # Başlık ve açıklama
    st.markdown('<h1 class="main-header">🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #94a3b8; margin-bottom: 3rem;">'
                'GERÇEK ML Tahminleme • Modern Harita Görselleştirme • Rakip Analizi • Performans Takibi'
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
            
            # GeoJSON yükle
            gdf, geojson_data = load_turkey_geojson()
            
            st.success(f"✅ **{len(df):,}** satır veri yüklendi")
            st.info(f"📅 Veri aralığı: {df['DATE'].min().strftime('%Y-%m')} - {df['DATE'].max().strftime('%Y-%m')}")
            
            # Sütunları göster
            with st.expander("📋 Excel Sütunları"):
                st.write("Mevcut sütunlar:", df.columns.tolist())
                
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.stop()
        
        st.markdown("---")
        
        # Ürün Seçimi
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">💊 ÜRÜN SEÇİMİ</h4>', unsafe_allow_html=True)
        
        # Excel'deki sütunlara göre ürün listesi oluştur
        available_products = []
        for product in COLUMN_NAMES.keys():
            cols = get_product_columns(product, df.columns)
            if cols.get('pf') in df.columns:
                available_products.append(product)
        
        if not available_products:
            st.error("❌ Excel'de beklenen ürün sütunları bulunamadı!")
            st.info("Lütfen veri setinizi kontrol edin. Beklenen sütunlar: TROCMETAM, CORTIPOL, DEKSAMETAZON, PF IZOTONIK")
            st.stop()
        
        selected_product = st.selectbox("", available_products, label_visibility="collapsed")
        
        # Seçilen ürünün sütunlarını göster
        cols = get_product_columns(selected_product, df.columns)
        st.caption(f"📊 PF Sütunu: {cols.get('pf')}")
        st.caption(f"🎯 Rakip Sütunu: {cols.get('rakip')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tarih Aralığı
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">📅 TARİH ARALIĞI</h4>', unsafe_allow_html=True)
        
        min_date = df['DATE'].min()
        max_date = df['DATE'].max()
        
        date_option = st.selectbox("Dönem Seçin", ["Tüm Veriler", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl", "2025", "2024", "Özel Aralık"])
        
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
        elif date_option == "2025":
            date_filter = (pd.to_datetime('2025-01-01'), pd.to_datetime('2025-12-31'))
        elif date_option == "2024":
            date_filter = (pd.to_datetime('2024-01-01'), pd.to_datetime('2024-12-31'))
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
        
        territories = ["TÜMÜ"] + sorted([str(t) for t in df['TERRITORIES'].unique() if pd.notna(t)])
        selected_territory = st.selectbox("Territory", territories)
        
        regions = ["TÜMÜ"] + sorted([str(r) for r in df['REGION'].unique() if pd.notna(r)])
        selected_region = st.selectbox("Bölge", regions)
        
        managers = ["TÜMÜ"] + sorted([str(m) for m in df['MANAGER'].unique() if pd.notna(m)])
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
        
        # Renk Legend
        st.markdown('<h4 style="color: #e2e8f0;">🎨 BÖLGE RENKLERİ</h4>', unsafe_allow_html=True)
        for region, color in list(REGION_COLORS.items())[:5]:
            st.markdown(f'<div style="display: flex; align-items: center; margin: 0.3rem 0;">'
                       f'<div style="width: 12px; height: 12px; background-color: {color}; border-radius: 2px; margin-right: 8px;"></div>'
                       f'<span style="color: #cbd5e1; font-size: 0.9rem;">{region}</span>'
                       f'</div>', unsafe_allow_html=True)
    
    # ANA İÇERİK - TAB'LER
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Genel Bakış",
        "🗺️ Modern Harita",
        "🏢 Territory Analizi",
        "📈 Zaman Serisi & ML",
        "📊 Rakip Analizi",
        "📥 Raporlar"
    ])
    
    # TAB 1: GENEL BAKIŞ
    with tab1:
        st.header("📊 Genel Performans Özeti")
        
        cols = get_product_columns(selected_product, df.columns)
        
        if date_filter:
            df_period = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & (df_filtered['DATE'] <= date_filter[1])]
        else:
            df_period = df_filtered
        
        # HATA DÜZELTME: Sütun isimlerini kontrol et
        pf_col = cols.get('pf')
        rakip_col = cols.get('rakip')
        
        if pf_col not in df_period.columns:
            st.error(f"❌ '{pf_col}' sütunu veri setinde bulunamadı!")
            st.info(f"📋 Mevcut sütunlar: {', '.join(df_period.columns.tolist())}")
        else:
            # Rakip sütunu yoksa sıfır olarak kabul et
            if rakip_col not in df_period.columns:
                df_period[rakip_col] = 0
            
            # Metrikler
            total_pf = df_period[pf_col].sum()
            total_rakip = df_period[rakip_col].sum()
            total_market = total_pf + total_rakip
            market_share = safe_divide(total_pf, total_market) * 100
            active_territories = df_period['TERRITORIES'].nunique()
            avg_monthly_pf = safe_divide(total_pf, df_period['YIL_AY'].nunique())
            
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
            
            # Top 10 Territory
            st.subheader("🏆 Top 10 Territory Performansı")
            terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
            
            if len(terr_perf) == 0:
                st.warning("⚠️ Seçilen filtrelerde territory verisi bulunamadı")
            else:
                top10 = terr_perf.head(10)
                
                # Toplam Pazar % ekle
                total_market_all = terr_perf['Toplam_Pazar'].sum()
                top10['Toplam_Pazar_%'] = top10.apply(
                    lambda row: safe_divide(row['Toplam_Pazar'], total_market_all) * 100,
                    axis=1
                )
                
                col_chart1, col_chart2 = st.columns([2, 1])
                
                with col_chart1:
                    fig_top10 = go.Figure()
                    
                    fig_top10.add_trace(go.Bar(
                        x=top10['Territory'],
                        y=top10['PF_Satis'],
                        name='PF Satış',
                        marker_color=PERFORMANCE_COLORS['success'],
                        text=top10['PF_Satis'].apply(lambda x: f'{x:,.0f}'),
                        textposition='outside',
                        marker=dict(
                            line=dict(width=2, color='rgba(255, 255, 255, 0.8)')
                        )
                    ))
                    
                    fig_top10.add_trace(go.Bar(
                        x=top10['Territory'],
                        y=top10['Rakip_Satis'],
                        name='Rakip Satış',
                        marker_color=PERFORMANCE_COLORS['danger'],
                        text=top10['Rakip_Satis'].apply(lambda x: f'{x:,.0f}'),
                        textposition='outside',
                        marker=dict(
                            line=dict(width=2, color='rgba(255, 255, 255, 0.8)')
                        )
                    ))
                    
                    fig_top10.update_layout(
                        title=dict(
                            text='<b>Top 10 Territory - PF vs Rakip</b>',
                            font=dict(size=18, color='white')
                        ),
                        xaxis_title='<b>Territory</b>',
                        yaxis_title='<b>Satış</b>',
                        barmode='group',
                        height=500,
                        xaxis=dict(tickangle=-45),
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
                    
                    st.plotly_chart(fig_top10, use_container_width=True)
                
                with col_chart2:
                    # Top 5 Territory için pasta grafiği
                    top5 = top10.head(5)
                    fig_pie = px.pie(
                        top5,
                        values='PF_Satis',
                        names='Territory',
                        title='<b>Top 5 Territory Dağılımı</b>',
                        color_discrete_sequence=GRADIENT_SCALES['blue_green'],
                        hole=0.4
                    )
                    
                    fig_pie.update_layout(
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0'),
                        legend=dict(
                            orientation="v",
                            yanchor="middle",
                            y=0.5,
                            xanchor="right",
                            x=1.3
                        )
                    )
                    
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(line=dict(color='rgba(255, 255, 255, 0.8)', width=2))
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Detaylı Tablo
                st.markdown("---")
                st.subheader("📋 Top 10 Territory Detayları")
                
                display_cols = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 'Toplam_Pazar', 'Toplam_Pazar_%', 'Pazar_Payi_%', 'Agirlik_%']
                
                top10_display = top10[display_cols].copy()
                top10_display.columns = ['Territory', 'Region', 'City', 'Manager', 'PF Satış', 'Toplam Pazar', 'Toplam Pazar %', 'Pazar Payı %', 'Ağırlık %']
                top10_display.index = range(1, len(top10_display) + 1)
                
                # Modern tablo stilini uygula
                styled_df = style_dataframe(
                    top10_display,
                    color_column='Pazar Payı %',
                    gradient_columns=['Toplam Pazar %', 'Ağırlık %']
                )
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=400
                )
    
    # TAB 2: MODERN HARİTA - GÜNCELLENMİŞ
    with tab2:
        st.header("🗺️ Modern Türkiye Haritası")
        
        city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
        
        if len(city_data) == 0:
            st.warning("⚠️ Seçilen filtrelerde şehir verisi bulunamadı")
        else:
            # Quick Stats
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_pf = city_data['PF_Satis'].sum()
            total_market = city_data['Toplam_Pazar'].sum()
            avg_share = city_data['Pazar_Payi_%'].mean()
            active_cities = len(city_data[city_data['PF_Satis'] > 0])
            top_city = city_data.loc[city_data['PF_Satis'].idxmax(), 'City'] if len(city_data) > 0 else "Yok"
            
            with col1:
                st.metric("💊 PF Satış", f"{total_pf:,.0f}")
            with col2:
                st.metric("🏪 Toplam Pazar", f"{total_market:,.0f}")
            with col3:
                st.metric("📊 Ort. Pazar Payı", f"%{avg_share:.1f}")
            with col4:
                st.metric("🏙️ Aktif Şehir", active_cities)
            with col5:
                st.metric("🏆 Lider Şehir", top_city)
            
            st.markdown("---")
            
            # Modern Harita
            st.subheader("📍 İl Bazlı Dağılım")
            
            try:
                turkey_map = create_modern_turkey_map(
                    city_data, 
                    gdf, 
                    geojson_data,
                    title=f"{selected_product} Dağılımı"
                )
                
                if turkey_map:
                    st.plotly_chart(turkey_map, use_container_width=True)
                else:
                    # Alternatif görselleştirme
                    st.warning("⚠️ Harita oluşturulamadı. Alternatif görselleştirme gösteriliyor...")
                    
                    fig_bar = px.bar(
                        city_data.nlargest(15, 'PF_Satis'),
                        x='City',
                        y='PF_Satis',
                        title='<b>En Yüksek Satış Yapan Şehirler</b>',
                        color='Region',
                        color_discrete_map=REGION_COLORS,
                        text='PF_Satis'
                    )
                    fig_bar.update_layout(height=600)
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Harita oluşturma hatası: {str(e)}")
                # Alternatif görselleştirme
                fig_bar = px.bar(
                    city_data.nlargest(15, 'PF_Satis'),
                    x='City',
                    y='PF_Satis',
                    title='<b>En Yüksek Satış Yapan Şehirler</b>',
                    color='Region',
                    color_discrete_map=REGION_COLORS,
                    text='PF_Satis'
                )
                fig_bar.update_layout(height=600)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("---")
            
            # Şehir Analizi
            col_analysis1, col_analysis2 = st.columns(2)
            
            with col_analysis1:
                st.subheader("🏆 Top 10 Şehir")
                top_cities = city_data.nlargest(10, 'PF_Satis')
                
                fig_bar = px.bar(
                    top_cities,
                    x='City',
                    y='PF_Satis',
                    title='<b>En Yüksek Satış Yapan Şehirler</b>',
                    color='Region',
                    color_discrete_map=REGION_COLORS,
                    hover_data=['Region', 'PF_Satis', 'Pazar_Payi_%'],
                    text='PF_Satis'
                )
                
                fig_bar.update_layout(
                    height=500,
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    yaxis_title='<b>PF Satış</b>',
                    xaxis_title='<b>Şehir</b>'
                )
                
                fig_bar.update_traces(
                    texttemplate='%{text:,.0f}',
                    textposition='outside',
                    marker=dict(line=dict(width=2, color='rgba(255, 255, 255, 0.8)'))
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_analysis2:
                st.subheader("🗺️ Bölge Dağılımı")
                
                region_perf = city_data.groupby('Region').agg({
                    'PF_Satis': 'sum',
                    'Toplam_Pazar': 'sum'
                }).reset_index()
                
                region_perf['Pazar_Payi_%'] = region_perf.apply(
                    lambda row: safe_divide(row['PF_Satis'], row['Toplam_Pazar']) * 100,
                    axis=1
                )
                
                fig_pie = px.pie(
                    region_perf,
                    values='PF_Satis',
                    names='Region',
                    title='<b>Bölgelere Göre Satış Dağılımı</b>',
                    color='Region',
                    color_discrete_map=REGION_COLORS,
                    hole=0.3
                )
                
                fig_pie.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="right",
                        x=1.3
                    )
                )
                
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    marker=dict(line=dict(color='rgba(255, 255, 255, 0.8)', width=2))
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
    
    # TAB 3: TERRITORY ANALİZİ - DÜZELTİLMİŞ
    with tab3:
        st.header("🏢 Territory Bazlı Detaylı Analiz")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        
        if len(terr_perf) == 0:
            st.warning("⚠️ Seçilen filtrelerde territory verisi bulunamadı")
        else:
            # TOPLAM PAZAR YÜZDESİ HESAPLA
            total_market_all = terr_perf['Toplam_Pazar'].sum()
            terr_perf['Toplam_Pazar_%'] = terr_perf.apply(
                lambda row: safe_divide(row['Toplam_Pazar'], total_market_all) * 100,
                axis=1
            )
            
            # Filtreleme ve sıralama
            col_filter1, col_filter2 = st.columns([1, 2])
            
            with col_filter1:
                sort_options = {
                    'PF_Satis': 'PF Satış',
                    'Pazar_Payi_%': 'Pazar Payı %',
                    'Toplam_Pazar': 'Toplam Pazar',
                    'Toplam_Pazar_%': 'Toplam Pazar %',
                    'Agirlik_%': 'Ağırlık %'
                }
                sort_by = st.selectbox(
                    "Sıralama Kriteri",
                    options=list(sort_options.keys()),
                    format_func=lambda x: sort_options[x]
                )
            
            with col_filter2:
                show_n = st.slider("Gösterilecek Territory Sayısı", 10, 100, 25, 5)
            
            terr_sorted = terr_perf.sort_values(sort_by, ascending=False).head(show_n)
            
            # Visualizations
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.subheader("📊 PF vs Rakip Satış")
                
                fig_bar = go.Figure()
                
                fig_bar.add_trace(go.Bar(
                    x=terr_sorted['Territory'],
                    y=terr_sorted['PF_Satis'],
                    name='PF Satış',
                    marker_color=PERFORMANCE_COLORS['success'],
                    text=terr_sorted['PF_Satis'].apply(lambda x: f'{x:,.0f}'),
                    textposition='outside',
                    marker=dict(
                        line=dict(width=1.5, color='rgba(255, 255, 255, 0.8)')
                    )
                ))
                
                fig_bar.add_trace(go.Bar(
                    x=terr_sorted['Territory'],
                    y=terr_sorted['Rakip_Satis'],
                    name='Rakip Satış',
                    marker_color=PERFORMANCE_COLORS['danger'],
                    text=terr_sorted['Rakip_Satis'].apply(lambda x: f'{x:,.0f}'),
                    textposition='outside',
                    marker=dict(
                        line=dict(width=1.5, color='rgba(255, 255, 255, 0.8)')
                    )
                ))
                
                fig_bar.update_layout(
                    title=dict(
                        text=f'<b>Top {show_n} Territory - PF vs Rakip</b>',
                        font=dict(size=18, color='white')
                    ),
                    xaxis_title='<b>Territory</b>',
                    yaxis_title='<b>Satış</b>',
                    barmode='group',
                    height=600,
                    xaxis=dict(tickangle=-45),
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
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_viz2:
                st.subheader("🎯 Pazar Payı Dağılımı")
                
                fig_scatter = px.scatter(
                    terr_sorted,
                    x='PF_Satis',
                    y='Pazar_Payi_%',
                    size='Toplam_Pazar',
                    color='Region',
                    color_discrete_map=REGION_COLORS,
                    hover_name='Territory',
                    hover_data={
                        'Region': True,
                        'PF_Satis': ':,.0f',
                        'Rakip_Satis': ':,.0f',
                        'Pazar_Payi_%': ':.1f',
                        'Toplam_Pazar_%': ':.1f'
                    },
                    size_max=50,
                    title=f'<b>Territory Performans Haritası</b>'
                )
                
                fig_scatter.update_layout(
                    height=600,
                    plot_bgcolor='rgba(15, 23, 41, 0.9)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    xaxis_title='<b>PF Satış</b>',
                    yaxis_title='<b>Pazar Payı %</b>',
                    legend=dict(
                        title='<b>Bölge</b>',
                        bgcolor='rgba(30, 41, 59, 0.8)'
                    )
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("---")
            
            # Detaylı Territory Listesi
            st.subheader(f"📋 Detaylı Territory Listesi (Top {show_n})")
            
            display_cols = [
                'Territory', 'Region', 'City', 'Manager',
                'PF_Satis', 'Rakip_Satis', 'Toplam_Pazar', 'Toplam_Pazar_%',
                'Pazar_Payi_%', 'Goreceli_Pazar_Payi', 'Agirlik_%'
            ]
            
            terr_display = terr_sorted[display_cols].copy()
            terr_display.columns = [
                'Territory', 'Region', 'City', 'Manager',
                'PF Satış', 'Rakip Satış', 'Toplam Pazar', 'Toplam Pazar %',
                'Pazar Payı %', 'Göreceli Pay', 'Ağırlık %'
            ]
            terr_display.index = range(1, len(terr_display) + 1)
            
            # Modern tablo stilini uygula
            styled_territory = style_dataframe(
                terr_display,
                color_column='Pazar Payı %',
                gradient_columns=['Toplam Pazar %', 'Ağırlık %', 'Göreceli Pay']
            )
            
            st.dataframe(
                styled_territory,
                use_container_width=True,
                height=600
            )
    
    # TAB 4: ZAMAN SERİSİ & ML
    with tab4:
        st.header("📈 Zaman Serisi Analizi")
        
        territory_for_ts = st.selectbox(
            "Territory Seçin",
            ["TÜMÜ"] + sorted([str(t) for t in df_filtered['TERRITORIES'].unique() if pd.notna(t)]),
            key='ts_territory'
        )
        
        monthly_df = calculate_time_series(df_filtered, selected_product, territory_for_ts, date_filter)
        
        if len(monthly_df) == 0:
            st.warning("⚠️ Seçilen filtrelerde veri bulunamadı")
        else:
            # Özet Metrikler
            col_ts1, col_ts2, col_ts3, col_ts4 = st.columns(4)
            
            with col_ts1:
                avg_pf = monthly_df['PF_Satis'].mean()
                st.metric("📊 Ort. Aylık PF", f"{avg_pf:,.0f}")
            
            with col_ts2:
                avg_growth = monthly_df['PF_Buyume_%'].mean()
                st.metric("📈 Ort. Büyüme", f"%{avg_growth:.1f}")
            
            with col_ts3:
                avg_share = monthly_df['Pazar_Payi_%'].mean()
                st.metric("🎯 Ort. Pazar Payı", f"%{avg_share:.1f}")
            
            with col_ts4:
                total_months = len(monthly_df)
                st.metric("📅 Veri Dönemi", f"{total_months} ay")
            
            st.markdown("---")
            
            # Grafikler
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("📊 Satış Trendi")
                fig_ts = go.Figure()
                
                fig_ts.add_trace(go.Scatter(
                    x=monthly_df['DATE'],
                    y=monthly_df['PF_Satis'],
                    mode='lines+markers',
                    name='PF Satış',
                    line=dict(color=PERFORMANCE_COLORS['success'], width=3, shape='spline'),
                    marker=dict(size=8, color='white', line=dict(width=2, color=PERFORMANCE_COLORS['success'])),
                    fill='tozeroy',
                    fillcolor='rgba(16, 185, 129, 0.1)'
                ))
                
                fig_ts.add_trace(go.Scatter(
                    x=monthly_df['DATE'],
                    y=monthly_df['Rakip_Satis'],
                    mode='lines+markers',
                    name='Rakip Satış',
                    line=dict(color=PERFORMANCE_COLORS['danger'], width=3, shape='spline'),
                    marker=dict(size=8, color='white', line=dict(width=2, color=PERFORMANCE_COLORS['danger'])),
                    fill='tozeroy',
                    fillcolor='rgba(239, 68, 68, 0.1)'
                ))
                
                fig_ts.update_layout(
                    height=500,
                    xaxis_title='<b>Tarih</b>',
                    yaxis_title='<b>Satış</b>',
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
                
                st.plotly_chart(fig_ts, use_container_width=True)
            
            with col_chart2:
                st.subheader("🎯 Pazar Payı Trendi")
                fig_share = go.Figure()
                
                fig_share.add_trace(go.Scatter(
                    x=monthly_df['DATE'],
                    y=monthly_df['Pazar_Payi_%'],
                    mode='lines+markers',
                    name='Pazar Payı %',
                    line=dict(color=PERFORMANCE_COLORS['info'], width=3, shape='spline'),
                    marker=dict(size=8, color='white', line=dict(width=2, color=PERFORMANCE_COLORS['info'])),
                    fill='tozeroy',
                    fillcolor='rgba(59, 130, 246, 0.1)'
                ))
                
                fig_share.add_hline(
                    y=50,
                    line_dash="dash",
                    line_color=PERFORMANCE_COLORS['warning'],
                    opacity=0.5,
                    line_width=2,
                    annotation_text="50% Eşik"
                )
                
                fig_share.update_layout(
                    height=500,
                    xaxis_title='<b>Tarih</b>',
                    yaxis_title='<b>Pazar Payı (%)</b>',
                    yaxis=dict(range=[0, 100]),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig_share, use_container_width=True)
    
    # TAB 5: RAKİP ANALİZİ
    with tab5:
        st.header("📊 Detaylı Rakip Analizi")
        
        comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
        
        if len(comp_data) == 0:
            st.warning("⚠️ Seçilen filtrelerde veri bulunamadı")
        else:
            # Özet Metrikler
            col1, col2, col3, col4 = st.columns(4)
            
            avg_pf_share = comp_data['PF_Pay_%'].mean()
            avg_pf_growth = comp_data['PF_Buyume'].mean()
            avg_rakip_growth = comp_data['Rakip_Buyume'].mean()
            win_months = len(comp_data[comp_data['Fark'] > 0])
            
            with col1:
                st.metric("🎯 Ort. PF Pazar Payı", f"%{avg_pf_share:.1f}")
            with col2:
                st.metric("📈 Ort. PF Büyüme", f"%{avg_pf_growth:.1f}")
            with col3:
                st.metric("📉 Ort. Rakip Büyüme", f"%{avg_rakip_growth:.1f}")
            with col4:
                st.metric("🏆 Kazanılan Aylar", f"{win_months}/{len(comp_data)}")
            
            st.markdown("---")
            
            # Grafikler
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.subheader("💰 Satış Karşılaştırması")
                comp_chart = create_modern_competitor_chart(comp_data)
                st.plotly_chart(comp_chart, use_container_width=True)
            
            with col_g2:
                st.subheader("📈 Büyüme Trendi")
                fig_growth = go.Figure()
                
                fig_growth.add_trace(go.Scatter(
                    x=comp_data['YIL_AY'],
                    y=comp_data['PF_Buyume'],
                    mode='lines+markers',
                    name='PF Büyüme',
                    line=dict(color=PERFORMANCE_COLORS['success'], width=3, shape='spline'),
                    marker=dict(size=8, color='white', line=dict(width=2, color=PERFORMANCE_COLORS['success']))
                ))
                
                fig_growth.add_trace(go.Scatter(
                    x=comp_data['YIL_AY'],
                    y=comp_data['Rakip_Buyume'],
                    mode='lines+markers',
                    name='Rakip Büyüme',
                    line=dict(color=PERFORMANCE_COLORS['danger'], width=3, shape='spline'),
                    marker=dict(size=8, color='white', line=dict(width=2, color=PERFORMANCE_COLORS['danger']))
                ))
                
                fig_growth.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color=PERFORMANCE_COLORS['neutral'],
                    opacity=0.5
                )
                
                fig_growth.update_layout(
                    height=500,
                    xaxis_title='<b>Ay</b>',
                    yaxis_title='<b>Büyüme %</b>',
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
                    )
                )
                
                st.plotly_chart(fig_growth, use_container_width=True)
            
            # Detaylı Tablo
            st.markdown("---")
            st.subheader("📋 Aylık Performans Detayları")
            
            comp_display = comp_data[['YIL_AY', 'PF', 'Rakip', 'PF_Pay_%', 'PF_Buyume', 'Rakip_Buyume', 'Fark']].copy()
            comp_display.columns = ['Ay', 'PF Satış', 'Rakip Satış', 'PF Pay %', 'PF Büyüme %', 'Rakip Büyüme %', 'Fark %']
            comp_display.index = range(1, len(comp_display) + 1)
            
            styled_comp = style_dataframe(
                comp_display,
                color_column='Fark %',
                gradient_columns=['PF Pay %', 'PF Büyüme %', 'Rakip Büyüme %']
            )
            
            st.dataframe(
                styled_comp,
                use_container_width=True,
                height=400
            )
    
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
                <li>Territory Performans</li>
                <li>Zaman Serisi Analizi</li>
                <li>Şehir Bazlı Analiz</li>
                <li>Rakip Analizi</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Excel Raporu Oluştur", type="primary", use_container_width=True):
            with st.spinner("Rapor hazırlanıyor..."):
                try:
                    # Tüm analizleri hesapla
                    terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
                    
                    if len(terr_perf) > 0:
                        total_market_all = terr_perf['Toplam_Pazar'].sum()
                        terr_perf['Toplam_Pazar_%'] = terr_perf.apply(
                            lambda row: safe_divide(row['Toplam_Pazar'], total_market_all) * 100,
                            axis=1
                        )
                    
                    monthly_df = calculate_time_series(df_filtered, selected_product, None, date_filter)
                    city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
                    comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        if len(terr_perf) > 0:
                            terr_perf.to_excel(writer, sheet_name='Territory Performans', index=False)
                        
                        if len(monthly_df) > 0:
                            monthly_df.to_excel(writer, sheet_name='Zaman Serisi', index=False)
                        
                        if len(city_data) > 0:
                            city_data.to_excel(writer, sheet_name='Şehir Analizi', index=False)
                        
                        if len(comp_data) > 0:
                            comp_data.to_excel(writer, sheet_name='Rakip Analizi', index=False)
                        
                        # Özet sayfası
                        summary_data = {
                            'Metrik': ['Ürün', 'Dönem', 'Toplam PF Satış', 'Toplam Pazar', 'Pazar Payı', 'Territory Sayısı'],
                            'Değer': [
                                selected_product,
                                date_option,
                                f"{terr_perf['PF_Satis'].sum():,.0f}" if len(terr_perf) > 0 else "0",
                                f"{terr_perf['Toplam_Pazar'].sum():,.0f}" if len(terr_perf) > 0 else "0",
                                f"{(terr_perf['PF_Satis'].sum() / terr_perf['Toplam_Pazar'].sum() * 100):.1f}%" if len(terr_perf) > 0 and terr_perf['Toplam_Pazar'].sum() > 0 else "0%",
                                len(terr_perf) if len(terr_perf) > 0 else 0
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Özet', index=False)
                    
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
