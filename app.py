"""
🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ v2
- 🗺️ Profesyonel Türkiye Haritası (app26 entegrasyonu)
- 🤖 Gerçek ML Tahminleme
- 📊 Gelişmiş Rakip Analizi (dark theme fix)
- 🧠 Yeni Analizler: Pareto, Volatilite, Fırsat Skoru, YoY Heatmap
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

# Yeni importlar
try:
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    st.warning("⚠️ geopandas yüklü değil. Gelişmiş harita özelliği devre dışı.")

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
# CSS (Dark Theme Optimized)
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #ffd700 0%, #f59e0b 50%, #d97706 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(255, 215, 0, 0.4);
        letter-spacing: -1px;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.9);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.4);
        border-color: rgba(59, 130, 246, 0.6);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
        padding: 1rem 2rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px 8px 0 0;
        margin: 0 0.25rem;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.2);
        color: #e0e7ff;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700;
    }
    
    p, span, div, label {
        color: #cbd5e1;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    /* Dark theme table fix */
    div[data-testid="stDataFrame"] table {
        background-color: rgba(30, 41, 59, 0.9) !important;
        color: #f1f5f9 !important;
    }
    
    div[data-testid="stDataFrame"] th {
        background-color: rgba(15, 23, 42, 1.0) !important;
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stDataFrame"] td {
        color: #f1f5f9 !important;
        border-color: rgba(148, 163, 184, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MAP MODULE - BÖLGE RENKLERİ (APP26 ENTEGRASYON)
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

def normalize_city_for_map(name):
    """Harita için şehir normalizasyonu"""
    if pd.isna(name):
        return None
    name = str(name).upper().strip()
    tr_map = {"İ": "I", "Ğ": "G", "Ü": "U", "Ş": "S", "Ö": "O", "Ç": "C"}
    for k, v in tr_map.items():
        name = name.replace(k, v)
    return name

@st.cache_resource
def load_geo_map():
    """GeoJSON yükle (app26 tarzı)"""
    if not GEOPANDAS_AVAILABLE:
        return None
    
    try:
        gdf = gpd.read_file("turkey.geojson")
        gdf["raw_name"] = gdf["name"].str.upper()
        gdf["fixed_name"] = gdf["raw_name"].replace(FIX_CITY_MAP)
        gdf["CITY_KEY"] = gdf["fixed_name"].apply(normalize_city_for_map)
        return gdf
    except:
        return None

def lines_to_lonlat(geom):
    """Geometri -> lon/lat"""
    lons, lats = [], []
    if isinstance(geom, LineString):
        xs, ys = geom.xy
        lons += list(xs) + [None]
        lats += list(ys) + [None]
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            xs, ys = line.xy
            lons += list(xs) + [None]
            lats += list(ys) + [None]
    return lons, lats

def get_region_center(gdf_region):
    """Bölge merkezi"""
    centroid = gdf_region.geometry.unary_union.centroid
    return centroid.x, centroid.y

def map_create_figure(gdf, manager_filter, view_mode, filtered_pf_total, filtered_total_market):
    """Profesyonel harita (app26 tarzı - dark theme optimized)"""
    gdf = gdf.copy()
    
    if manager_filter != "TÜMÜ":
        gdf = gdf[gdf["Ticaret Müdürü"] == manager_filter]
    
    fig = go.Figure()
    
    # Bölge bazlı choropleth
    for region in gdf["Bölge"].unique():
        region_gdf = gdf[gdf["Bölge"] == region]
        color = REGION_COLORS.get(region, "#CCCCCC")
        
        fig.add_choropleth(
            geojson=json.loads(region_gdf.to_json()),
            locations=region_gdf.index,
            z=[1] * len(region_gdf),
            colorscale=[[0, color], [1, color]],
            marker_line_color="white",
            marker_line_width=1.5,
            showscale=False,
            customdata=list(zip(
                region_gdf["Şehir"],
                region_gdf["Bölge"],
                region_gdf["PF Kutu"],
                region_gdf["Pazar Payı %"]
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Bölge: %{customdata[1]}<br>"
                "PF Kutu: %{customdata[2]:,.0f}<br>"
                "Pazar Payı: %{customdata[3]:.1f}%"
                "<extra></extra>"
            ),
            name=region
        )
    
    # Sınır çizgileri
    lons, lats = [], []
    for geom in gdf.geometry.boundary:
        lo, la = lines_to_lonlat(geom)
        lons += lo
        lats += la
    
    fig.add_scattergeo(
        lon=lons, lat=lats, mode="lines",
        line=dict(color="rgba(255,255,255,0.6)", width=1),
        hoverinfo="skip", showlegend=False
    )
    
    # Etiketler (Bölge / Şehir)
    if view_mode == "Bölge Görünümü":
        label_lons, label_lats, label_texts = [], [], []
        for region in gdf["Bölge"].unique():
            region_gdf = gdf[gdf["Bölge"] == region]
            total = region_gdf["PF Kutu"].sum()
            if total > 0:
                percent = (total / filtered_pf_total * 100) if filtered_pf_total > 0 else 0
                region_market = region_gdf["Toplam Kutu"].sum()
                pazar_payi = (total / region_market * 100) if region_market > 0 else 0
                lon, lat = get_region_center(region_gdf)
                label_lons.append(lon)
                label_lats.append(lat)
                label_texts.append(
                    f"<b>{region}</b><br>"
                    f"{total:,.0f} ({percent:.1f}%)<br>"
                    f"Pazar: {pazar_payi:.1f}%"
                )
        
        fig.add_scattergeo(
            lon=label_lons, lat=label_lats, mode="text", text=label_texts,
            textfont=dict(size=10, color="white", family="Arial Black"),
            hoverinfo="skip", showlegend=False
        )
    
    else:  # Şehir Görünümü
        city_lons, city_lats, city_texts = [], [], []
        for idx, row in gdf.iterrows():
            if row["PF Kutu"] > 0:
                percent = (row["PF Kutu"] / filtered_pf_total * 100) if filtered_pf_total > 0 else 0
                centroid = row.geometry.centroid
                city_lons.append(centroid.x)
                city_lats.append(centroid.y)
                city_texts.append(
                    f"<b>{row['Şehir']}</b><br>"
                    f"{row['PF Kutu']:,.0f} ({percent:.1f}%)<br>"
                    f"Pazar: {row['Pazar Payı %']:.1f}%"
                )
        
        fig.add_scattergeo(
            lon=city_lons, lat=city_lats, mode="text", text=city_texts,
            textfont=dict(size=8, color="white", family="Arial"),
            hoverinfo="skip", showlegend=False
        )
    
    fig.update_layout(
        geo=dict(
            projection=dict(type="mercator"),
            center=dict(lat=39, lon=35),
            lonaxis=dict(range=[25, 45]),
            lataxis=dict(range=[35, 43]),
            visible=False,
            bgcolor="rgba(0,0,0,0)"
        ),
        height=750,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

# =============================================================================
# ŞEHİR NORMALIZASYON (MEVCUT)
# =============================================================================
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

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def safe_divide(a, b):
    return np.where(b != 0, a / b, 0)

def get_product_columns(product, df_columns=None):
    """
    Ürün kolonlarını akıllıca tespit et
    Önce tam eşleşme ara, bulamazsa fuzzy matching yap
    """
    # Standart mapping
    standard_mapping = {
        "TROCMETAM": {"pf": "TROCMETAM", "rakip": "DIGER TROCMETAM"},
        "CORTIPOL": {"pf": "CORTIPOL", "rakip": "DIGER CORTIPOL"},
        "DEKSAMETAZON": {"pf": "DEKSAMETAZON", "rakip": "DIGER DEKSAMETAZON"},
        "PF IZOTONIK": {"pf": "PF IZOTONIK", "rakip": "DIGER IZOTONIK"}
    }
    
    base_cols = standard_mapping.get(product, {"pf": product, "rakip": f"DIGER {product}"})
    
    # Eğer df_columns verilmemişse standart döndür
    if df_columns is None:
        return base_cols
    
    # Kolon kontrolü ve fuzzy matching
    result = {}
    
    for key, col_name in base_cols.items():
        if col_name in df_columns:
            result[key] = col_name
        else:
            # Fuzzy matching - case insensitive, strip whitespace
            normalized_cols = {c.upper().strip(): c for c in df_columns}
            search_key = col_name.upper().strip()
            
            if search_key in normalized_cols:
                result[key] = normalized_cols[search_key]
            else:
                # Partial match
                matches = [c for c in df_columns if search_key in c.upper() or c.upper() in search_key]
                if matches:
                    result[key] = matches[0]
                else:
                    # Son çare: içinde product adı geçen kolonları bul
                    product_word = product.split()[0] if ' ' in product else product
                    if key == 'pf':
                        matches = [c for c in df_columns if product_word.upper() in c.upper() and 'DIGER' not in c.upper()]
                    else:
                        matches = [c for c in df_columns if product_word.upper() in c.upper() and 'DIGER' in c.upper()]
                    
                    if matches:
                        result[key] = matches[0]
                    else:
                        result[key] = None
    
    return result

def normalize_city_name_fixed(city_name):
    if pd.isna(city_name):
        return None
    city_upper = str(city_name).strip().upper()
    city_upper = (city_upper
                  .replace('İ', 'I').replace('Ş', 'S').replace('Ğ', 'G')
                  .replace('Ü', 'U').replace('Ö', 'O').replace('Ç', 'C'))
    return CITY_NORMALIZE_CLEAN.get(city_upper, city_name)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_excel_data(file):
    """Excel dosyasını yükle ve validate et"""
    try:
        df = pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Excel dosyası okunamadı: {str(e)}")
        st.stop()
    
    # Gerekli kolonları kontrol et
    required_base_cols = ['DATE', 'TERRITORIES', 'CITY', 'REGION', 'MANAGER']
    missing_cols = [col for col in required_base_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"""
        ❌ **Eksik Kolonlar Tespit Edildi:**
        
        Gerekli kolonlar: `{', '.join(required_base_cols)}`
        
        Eksik olanlar: `{', '.join(missing_cols)}`
        
        **Mevcut kolonlar:**
        ```
        {', '.join(df.columns.tolist())}
        ```
        
        💡 Lütfen Excel dosyanızın bu kolonları içerdiğinden emin olun.
        """)
        st.stop()
    
    # Tarih dönüşümü
    try:
        df['DATE'] = pd.to_datetime(df['DATE'])
    except:
        st.error("❌ 'DATE' kolonu tarih formatına dönüştürülemedi. Lütfen tarih formatını kontrol edin.")
        st.stop()
    
    df['YIL_AY'] = df['DATE'].dt.strftime('%Y-%m')
    df['AY'] = df['DATE'].dt.month
    df['YIL'] = df['DATE'].dt.year
    
    # String kolonları normalize et
    df['TERRITORIES'] = df['TERRITORIES'].str.upper().str.strip()
    df['CITY'] = df['CITY'].str.strip()
    df['CITY_NORMALIZED'] = df['CITY'].apply(normalize_city_name_fixed)
    df['REGION'] = df['REGION'].str.upper().str.strip()
    df['MANAGER'] = df['MANAGER'].str.upper().str.strip()
    
    return df

@st.cache_data
def load_geojson_safe():
    paths = ['/mnt/user-data/uploads/turkey.geojson', 'turkey.geojson', './turkey.geojson']
    for path in paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            continue
    return None

# =============================================================================
# ML FUNCTIONS
# =============================================================================
def create_ml_features(df):
    df = df.copy()
    df = df.sort_values('DATE').reset_index(drop=True)
    
    df['lag_1'] = df['PF_Satis'].shift(1)
    df['lag_2'] = df['PF_Satis'].shift(2)
    df['lag_3'] = df['PF_Satis'].shift(3)
    df['rolling_mean_3'] = df['PF_Satis'].rolling(window=3, min_periods=1).mean()
    df['rolling_mean_6'] = df['PF_Satis'].rolling(window=6, min_periods=1).mean()
    df['rolling_std_3'] = df['PF_Satis'].rolling(window=3, min_periods=1).std()
    df['month'] = df['DATE'].dt.month
    df['quarter'] = df['DATE'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['trend_index'] = range(len(df))
    df = df.fillna(method='bfill').fillna(0)
    
    return df

def train_ml_models(df, forecast_periods=3):
    df_features = create_ml_features(df)
    
    if len(df_features) < 10:
        return None, None, None
    
    feature_cols = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_mean_6',
                    'rolling_std_3', 'month', 'quarter', 'month_sin', 'month_cos', 'trend_index']
    
    split_idx = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['PF_Satis']
    X_test = test_df[feature_cols]
    y_test = test_df['PF_Satis']
    
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
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        results[name] = {'model': model, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    best_model_name = min(results.keys(), key=lambda x: results[x]['MAPE'])
    best_model = results[best_model_name]['model']
    
    forecast_data = []
    last_row = df_features.iloc[-1:].copy()
    
    for i in range(forecast_periods):
        next_date = last_row['DATE'].values[0] + pd.DateOffset(months=1)
        X_future = last_row[feature_cols]
        next_pred = best_model.predict(X_future)[0]
        
        forecast_data.append({
            'DATE': next_date,
            'YIL_AY': pd.to_datetime(next_date).strftime('%Y-%m'),
            'PF_Satis': max(0, next_pred),
            'Model': best_model_name
        })
        
        new_row = last_row.copy()
        new_row['DATE'] = next_date
        new_row['PF_Satis'] = next_pred
        new_row['lag_1'] = last_row['PF_Satis'].values[0]
        new_row['lag_2'] = last_row['lag_1'].values[0]
        new_row['lag_3'] = last_row['lag_2'].values[0]
        new_row['rolling_mean_3'] = (new_row['lag_1'] + new_row['lag_2'] + new_row['lag_3']) / 3
        new_row['month'] = pd.to_datetime(next_date).month
        new_row['quarter'] = pd.to_datetime(next_date).quarter
        new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)
        new_row['trend_index'] = last_row['trend_index'].values[0] + 1
        last_row = new_row
    
    forecast_df = pd.DataFrame(forecast_data)
    return results, best_model_name, forecast_df

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def calculate_city_performance(df, product, date_filter=None):
    """Şehir bazlı performans - geliştirilmiş hata yönetimi"""
    cols = get_product_columns(product, df.columns)
    
    # Kolon kontrolü
    if cols['pf'] is None or cols['rakip'] is None:
        st.error(f"""
        ❌ **Ürün kolonları bulunamadı!**
        
        Aranan: `{product}` ve `DIGER {product}`
        
        **Mevcut ürün kolonları:**
        ```
        {', '.join([c for c in df.columns if any(p in c.upper() for p in ['TROC', 'CORTI', 'DEKSA', 'IZOTO', 'DIGER'])])}
        ```
        
        💡 Lütfen doğru ürünü seçtiğinizden emin olun veya Excel dosyasındaki kolon isimlerini kontrol edin.
        """)
        st.stop()
    
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    city_perf = df.groupby(['CITY_NORMALIZED']).agg({
        cols['pf']: 'sum', cols['rakip']: 'sum'
    }).reset_index()
    city_perf.columns = ['City', 'PF_Satis', 'Rakip_Satis']
    city_perf['Toplam_Pazar'] = city_perf['PF_Satis'] + city_perf['Rakip_Satis']
    city_perf['Pazar_Payi_%'] = safe_divide(city_perf['PF_Satis'], city_perf['Toplam_Pazar']) * 100
    return city_perf

def calculate_territory_performance(df, product, date_filter=None):
    cols = get_product_columns(product)
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    terr_perf = df.groupby(['TERRITORIES', 'REGION', 'CITY', 'MANAGER']).agg({
        cols['pf']: 'sum', cols['rakip']: 'sum'
    }).reset_index()
    terr_perf.columns = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 'Rakip_Satis']
    terr_perf['Toplam_Pazar'] = terr_perf['PF_Satis'] + terr_perf['Rakip_Satis']
    terr_perf['Pazar_Payi_%'] = safe_divide(terr_perf['PF_Satis'], terr_perf['Toplam_Pazar']) * 100
    total_pf = terr_perf['PF_Satis'].sum()
    terr_perf['Agirlik_%'] = safe_divide(terr_perf['PF_Satis'], total_pf) * 100
    terr_perf['Goreceli_Pazar_Payi'] = safe_divide(terr_perf['PF_Satis'], terr_perf['Rakip_Satis'])
    return terr_perf.sort_values('PF_Satis', ascending=False)

def calculate_time_series(df, product, territory=None, date_filter=None):
    cols = get_product_columns(product)
    df_filtered = df.copy()
    if territory and territory != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['TERRITORIES'] == territory]
    if date_filter:
        df_filtered = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & (df_filtered['DATE'] <= date_filter[1])]
    
    monthly = df_filtered.groupby('YIL_AY').agg({
        cols['pf']: 'sum', cols['rakip']: 'sum', 'DATE': 'first'
    }).reset_index().sort_values('YIL_AY')
    monthly.columns = ['YIL_AY', 'PF_Satis', 'Rakip_Satis', 'DATE']
    monthly['Toplam_Pazar'] = monthly['PF_Satis'] + monthly['Rakip_Satis']
    monthly['Pazar_Payi_%'] = safe_divide(monthly['PF_Satis'], monthly['Toplam_Pazar']) * 100
    monthly['PF_Buyume_%'] = monthly['PF_Satis'].pct_change() * 100
    monthly['Rakip_Buyume_%'] = monthly['Rakip_Satis'].pct_change() * 100
    monthly['Goreceli_Buyume_%'] = monthly['PF_Buyume_%'] - monthly['Rakip_Buyume_%']
    monthly['MA_3'] = monthly['PF_Satis'].rolling(window=3, min_periods=1).mean()
    monthly['MA_6'] = monthly['PF_Satis'].rolling(window=6, min_periods=1).mean()
    return monthly

def calculate_competitor_analysis(df, product, date_filter=None):
    cols = get_product_columns(product)
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    monthly = df.groupby('YIL_AY').agg({
        cols['pf']: 'sum', cols['rakip']: 'sum'
    }).reset_index().sort_values('YIL_AY')
    monthly.columns = ['YIL_AY', 'PF', 'Rakip']
    monthly['PF_Pay_%'] = (monthly['PF'] / (monthly['PF'] + monthly['Rakip'])) * 100
    monthly['Rakip_Pay_%'] = 100 - monthly['PF_Pay_%']
    monthly['PF_Buyume'] = monthly['PF'].pct_change() * 100
    monthly['Rakip_Buyume'] = monthly['Rakip'].pct_change() * 100
    monthly['Fark'] = monthly['PF_Buyume'] - monthly['Rakip_Buyume']
    return monthly

def calculate_bcg_matrix(df, product, date_filter=None):
    cols = get_product_columns(product)
    if date_filter:
        df_filtered = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    else:
        df_filtered = df.copy()
    
    terr_perf = calculate_territory_performance(df_filtered, product)
    df_sorted = df_filtered.sort_values('DATE')
    mid_point = len(df_sorted) // 2
    first_half = df_sorted.iloc[:mid_point].groupby('TERRITORIES')[cols['pf']].sum()
    second_half = df_sorted.iloc[mid_point:].groupby('TERRITORIES')[cols['pf']].sum()
    
    growth_rate = {}
    for terr in first_half.index:
        if terr in second_half.index and first_half[terr] > 0:
            growth_rate[terr] = ((second_half[terr] - first_half[terr]) / first_half[terr]) * 100
        else:
            growth_rate[terr] = 0
    
    terr_perf['Pazar_Buyume_%'] = terr_perf['Territory'].map(growth_rate).fillna(0)
    median_share = terr_perf['Goreceli_Pazar_Payi'].median()
    median_growth = terr_perf['Pazar_Buyume_%'].median()
    
    def assign_bcg(row):
        if row['Goreceli_Pazar_Payi'] >= median_share and row['Pazar_Buyume_%'] >= median_growth:
            return "⭐ Star"
        elif row['Goreceli_Pazar_Payi'] >= median_share and row['Pazar_Buyume_%'] < median_growth:
            return "🐄 Cash Cow"
        elif row['Goreceli_Pazar_Payi'] < median_share and row['Pazar_Buyume_%'] >= median_growth:
            return "❓ Question Mark"
        else:
            return "🐶 Dog"
    
    terr_perf['BCG_Kategori'] = terr_perf.apply(assign_bcg, axis=1)
    return terr_perf

# =============================================================================
# YENİ ANALİZLER
# =============================================================================
def calculate_pareto_analysis(df, product, date_filter=None):
    """Pareto 80/20 Analizi"""
    cols = get_product_columns(product)
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    city_perf = df.groupby('CITY_NORMALIZED')[cols['pf']].sum().sort_values(ascending=False).reset_index()
    city_perf.columns = ['City', 'PF_Satis']
    city_perf['Kumulative_%'] = (city_perf['PF_Satis'].cumsum() / city_perf['PF_Satis'].sum() * 100).round(1)
    city_perf['Sira'] = range(1, len(city_perf) + 1)
    
    sehir_80 = len(city_perf[city_perf['Kumulative_%'] <= 80])
    return city_perf, sehir_80

def calculate_volatility_score(df, product, date_filter=None):
    """Volatilite/Stabilite Skoru"""
    cols = get_product_columns(product)
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    monthly = df.groupby(['CITY_NORMALIZED', 'YIL_AY'])[cols['pf']].sum().reset_index()
    monthly['Buyume_%'] = monthly.groupby('CITY_NORMALIZED')[cols['pf']].pct_change() * 100
    
    volatility = monthly.groupby('CITY_NORMALIZED')['Buyume_%'].std().reset_index()
    volatility.columns = ['City', 'Volatilite_Std']
    volatility = volatility.sort_values('Volatilite_Std')
    return volatility

def calculate_opportunity_score(df, product, date_filter=None):
    """Fırsat Skoru"""
    city_perf = calculate_city_performance(df, product, date_filter)
    city_perf['Firsat_Skoru'] = city_perf['Toplam_Pazar'] * (1 - city_perf['Pazar_Payi_%'] / 100)
    city_perf = city_perf.sort_values('Firsat_Skoru', ascending=False)
    return city_perf

def calculate_yoy_heatmap(df, product):
    """YoY Heatmap"""
    cols = get_product_columns(product)
    df_copy = df.copy()
    df_copy['Yil'] = df_copy['DATE'].dt.year
    df_copy['Ay'] = df_copy['DATE'].dt.month
    
    pivot = df_copy.groupby(['Yil', 'Ay'])[cols['pf']].sum().reset_index()
    pivot_table = pivot.pivot(index='Yil', columns='Ay', values=cols['pf']).fillna(0)
    return pivot_table

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_turkey_map_fixed(city_data, geojson, title="Türkiye Satış Haritası"):
    if geojson is None:
        st.error("❌ GeoJSON yüklenemedi")
        return None
    
    fig = px.choropleth_mapbox(
        city_data, geojson=geojson, locations='City', featureidkey="properties.name",
        color='PF_Satis', hover_name='City',
        hover_data={'PF_Satis': ':,.0f', 'Pazar_Payi_%': ':.1f', 'City': False},
        color_continuous_scale="YlOrRd", labels={'PF_Satis': 'PF Satış'},
        title=title, mapbox_style="carto-positron",
        center={"lat": 39.0, "lon": 35.0}, zoom=5
    )
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))
    return fig

def create_forecast_chart(historical_df, forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_df['DATE'], y=historical_df['PF_Satis'],
        mode='lines+markers', name='Gerçek Satış',
        line=dict(color='#3B82F6', width=2), marker=dict(size=6)
    ))
    if forecast_df is not None and len(forecast_df) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_df['DATE'], y=forecast_df['PF_Satis'],
            mode='lines+markers', name='Tahmin',
            line=dict(color='#EF4444', width=2, dash='dash'), marker=dict(size=6, symbol='diamond')
        ))
    fig.update_layout(title='Satış Trendi ve ML Tahmin', xaxis_title='Tarih',
                      yaxis_title='PF Satış', height=400, hovermode='x unified')
    return fig

def create_competitor_comparison_chart(comp_data):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=comp_data['YIL_AY'], y=comp_data['PF'], name='PF', marker_color='#3B82F6'))
    fig.add_trace(go.Bar(x=comp_data['YIL_AY'], y=comp_data['Rakip'], name='Rakip', marker_color='#EF4444'))
    fig.update_layout(title='PF vs Rakip Satış', xaxis_title='Ay', yaxis_title='Satış', barmode='group', height=400)
    return fig

def create_market_share_trend(comp_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comp_data['YIL_AY'], y=comp_data['PF_Pay_%'],
                             mode='lines+markers', name='PF Pazar Payı', fill='tozeroy',
                             line=dict(color='#3B82F6', width=2)))
    fig.add_trace(go.Scatter(x=comp_data['YIL_AY'], y=comp_data['Rakip_Pay_%'],
                             mode='lines+markers', name='Rakip Pazar Payı', fill='tozeroy',
                             line=dict(color='#EF4444', width=2)))
    fig.update_layout(title='Pazar Payı Trendi (%)', xaxis_title='Ay', yaxis_title='Pazar Payı (%)',
                      height=400, yaxis=dict(range=[0, 100]))
    return fig

def create_growth_comparison(comp_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comp_data['YIL_AY'], y=comp_data['PF_Buyume'],
                             mode='lines+markers', name='PF Büyüme', line=dict(color='#3B82F6', width=2)))
    fig.add_trace(go.Scatter(x=comp_data['YIL_AY'], y=comp_data['Rakip_Buyume'],
                             mode='lines+markers', name='Rakip Büyüme', line=dict(color='#EF4444', width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(title='Büyüme Oranları Karşılaştırması (%)', xaxis_title='Ay',
                      yaxis_title='Büyüme (%)', height=400)
    return fig

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.markdown('<h1 class="main-header">🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ</h1>', unsafe_allow_html=True)
    st.markdown("**GERÇEK ML Tahminleme • Profesyonel Harita • Rakip Analizi • Yeni Analizler**")
    
    st.sidebar.header("📂 Dosya Yükleme")
    uploaded_file = st.sidebar.file_uploader("Excel Dosyası Yükleyin", type=['xlsx', 'xls'])
    
    if not uploaded_file:
        st.info("👈 Lütfen sol taraftan Excel dosyasını yükleyin")
        st.stop()
    
    try:
        df = load_excel_data(uploaded_file)
        geojson = load_geojson_safe()
        geo_gdf = load_geo_map() if GEOPANDAS_AVAILABLE else None
        st.sidebar.success(f"✅ {len(df)} satır veri yüklendi")
    except Exception as e:
        st.error(f"❌ Veri yükleme hatası: {str(e)}")
        st.stop()
    
    st.sidebar.markdown("---")
    st.sidebar.header("💊 Ürün Seçimi")
    selected_product = st.sidebar.selectbox("Ürün", ["TROCMETAM", "CORTIPOL", "DEKSAMETAZON", "PF IZOTONIK"])
    
    st.sidebar.markdown("---")
    st.sidebar.header("📅 Tarih Aralığı")
    min_date = df['DATE'].min()
    max_date = df['DATE'].max()
    date_option = st.sidebar.selectbox("Dönem Seçin", ["Tüm Veriler", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl", "2025", "2024", "Özel Aralık"])
    
    if date_option == "Tüm Veriler":
        date_filter = None
    elif date_option == "Son 3 Ay":
        date_filter = (max_date - pd.DateOffset(months=3), max_date)
    elif date_option == "Son 6 Ay":
        date_filter = (max_date - pd.DateOffset(months=6), max_date)
    elif date_option == "Son 1 Yıl":
        date_filter = (max_date - pd.DateOffset(years=1), max_date)
    elif date_option == "2025":
        date_filter = (pd.to_datetime('2025-01-01'), pd.to_datetime('2025-12-31'))
    elif date_option == "2024":
        date_filter = (pd.to_datetime('2024-01-01'), pd.to_datetime('2024-12-31'))
    else:
        col_date1, col_date2 = st.sidebar.columns(2)
        with col_date1:
            start_date = st.date_input("Başlangıç", min_date, min_value=min_date, max_value=max_date)
        with col_date2:
            end_date = st.date_input("Bitiş", max_date, min_value=min_date, max_value=max_date)
        date_filter = (pd.to_datetime(start_date), pd.to_datetime(end_date))
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filtreler")
    territories = ["TÜMÜ"] + sorted(df['TERRITORIES'].unique())
    selected_territory = st.sidebar.selectbox("Territory", territories)
    regions = ["TÜMÜ"] + sorted(df['REGION'].unique())
    selected_region = st.sidebar.selectbox("Bölge", regions)
    managers = ["TÜMÜ"] + sorted(df['MANAGER'].unique())
    selected_manager = st.sidebar.selectbox("Manager", managers)
    
    df_filtered = df.copy()
    if selected_territory != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['TERRITORIES'] == selected_territory]
    if selected_region != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['REGION'] == selected_region]
    if selected_manager != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['MANAGER'] == selected_manager]
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Genel Bakış", "🗺️ Türkiye Haritası", "🏢 Territory Analizi",
        "📈 Zaman Serisi & ML", "🎯 Rakip Analizi", "⭐ BCG & Strateji",
        "🧠 Yeni Analizler", "📥 Raporlar"
    ])
    
    # TAB 1: GENEL BAKIŞ
    with tab1:
        st.header("📊 Genel Performans Özeti")
        cols = get_product_columns(selected_product)
        
        if date_filter:
            df_period = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & (df_filtered['DATE'] <= date_filter[1])]
        else:
            df_period = df_filtered
        
        total_pf = df_period[cols['pf']].sum()
        total_rakip = df_period[cols['rakip']].sum()
        total_market = total_pf + total_rakip
        market_share = (total_pf / total_market * 100) if total_market > 0 else 0
        active_territories = df_period['TERRITORIES'].nunique()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💊 PF Satış", f"{total_pf:,.0f}")
        with col2:
            st.metric("🪙 Toplam Pazar", f"{total_market:,.0f}")
        with col3:
            st.metric("📊 Pazar Payı", f"%{market_share:.1f}")
        with col4:
            st.metric("🏢 Territory Sayısı", active_territories)
        
        st.markdown("---")
        st.subheader("🏆 Top 10 Territory")
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        top10 = terr_perf.head(10)
        
        fig_top10 = go.Figure()
        fig_top10.add_trace(go.Bar(x=top10['Territory'], y=top10['PF_Satis'], name='PF Satış',
                                    marker_color='#3B82F6', text=top10['PF_Satis'].apply(lambda x: f'{x:,.0f}'),
                                    textposition='outside'))
        fig_top10.add_trace(go.Bar(x=top10['Territory'], y=top10['Rakip_Satis'], name='Rakip Satış',
                                    marker_color='#EF4444', text=top10['Rakip_Satis'].apply(lambda x: f'{x:,.0f}'),
                                    textposition='outside'))
        fig_top10.update_layout(title='Top 10 Territory - PF vs Rakip', xaxis_title='Territory',
                                yaxis_title='Satış', barmode='group', height=500, xaxis=dict(tickangle=-45))
        st.plotly_chart(fig_top10, use_container_width=True)
        
        display_cols = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 'Pazar_Payi_%', 'Agirlik_%']
        top10_display = top10[display_cols].copy()
        top10_display.columns = ['Territory', 'Region', 'City', 'Manager', 'PF Satış', 'Pazar Payı %', 'Ağırlık %']
        top10_display.index = range(1, len(top10_display) + 1)
        st.dataframe(top10_display.style.format({'PF Satış': '{:,.0f}', 'Pazar Payı %': '{:.1f}', 'Ağırlık %': '{:.1f}'}),
                     use_container_width=True)
    
    # TAB 2: TÜRKİYE HARİTASI (ENTEGRASYON)
    with tab2:
        st.header("🗺️ Türkiye İl Bazlı Satış Haritası")
        
        city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
        total_pf = city_data['PF_Satis'].sum()
        total_market = city_data['Toplam_Pazar'].sum()
        avg_share = city_data['Pazar_Payi_%'].mean()
        active_cities = len(city_data[city_data['PF_Satis'] > 0])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💊 Toplam PF Satış", f"{total_pf:,.0f}")
        with col2:
            st.metric("🪙 Toplam Pazar", f"{total_market:,.0f}")
        with col3:
            st.metric("📊 Ort. Pazar Payı", f"%{avg_share:.1f}")
        with col4:
            st.metric("🏙️ Aktif Şehir", active_cities)
        
        st.markdown("---")
        
        # Harita tipi seçimi
        map_type = st.selectbox("🗺️ Harita Tipi", ["Bölge Renkli Harita (Yeni - app26)", "Heatmap (Eski)"])
        
        if map_type == "Bölge Renkli Harita (Yeni - app26)" and GEOPANDAS_AVAILABLE and geo_gdf is not None:
            st.subheader("🎨 Profesyonel Bölge Haritası")
            
            # Görünüm modu
            view_mode = st.radio("Görünüm Modu", ["Bölge Görünümü", "Şehir Görünümü"], horizontal=True)
            
            # Manager filtresi (harita için)
            map_manager_filter = st.selectbox("Manager Filtresi (Harita)", ["TÜMÜ"] + sorted(df['MANAGER'].unique()), key='map_manager')
            
            # Veriyi hazırla
            cols = get_product_columns(selected_product)
            if date_filter:
                df_map = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & (df_filtered['DATE'] <= date_filter[1])]
            else:
                df_map = df_filtered.copy()
            
            city_perf = df_map.groupby(['CITY_NORMALIZED', 'REGION', 'MANAGER']).agg({
                cols['pf']: 'sum', cols['rakip']: 'sum'
            }).reset_index()
            city_perf.columns = ['City', 'Region', 'Manager', 'PF_Satis', 'Rakip_Satis']
            city_perf['Toplam_Pazar'] = city_perf['PF_Satis'] + city_perf['Rakip_Satis']
            city_perf['Pazar_Payi_%'] = safe_divide(city_perf['PF_Satis'], city_perf['Toplam_Pazar']) * 100
            
            # Geo merge
            city_perf['CITY_KEY'] = city_perf['City'].apply(normalize_city_for_map)
            merged_geo = geo_gdf.merge(city_perf, on='CITY_KEY', how='left')
            merged_geo['Şehir'] = merged_geo['fixed_name']
            merged_geo['PF Kutu'] = merged_geo['PF_Satis'].fillna(0)
            merged_geo['Toplam Kutu'] = merged_geo['Toplam_Pazar'].fillna(0)
            merged_geo['Bölge'] = merged_geo['Region'].fillna("DİĞER")
            merged_geo['Ticaret Müdürü'] = merged_geo['Manager'].fillna("YOK")
            merged_geo['Pazar Payı %'] = merged_geo['Pazar_Payi_%'].fillna(0)
            
            filtered_pf_total = merged_geo['PF Kutu'].sum()
            filtered_total_market = merged_geo['Toplam Kutu'].sum()
            
            # Harita oluştur
            fig_map = map_create_figure(merged_geo, map_manager_filter, view_mode, filtered_pf_total, filtered_total_market)
            st.plotly_chart(fig_map, use_container_width=True)
            
        else:
            # Eski heatmap
            if geojson:
                st.subheader("📍 İl Bazlı Dağılım (Heatmap)")
                city_data_fixed = city_data.copy()
                city_data_fixed['City'] = city_data_fixed['City'].apply(normalize_city_name_fixed)
                turkey_map = create_turkey_map_fixed(city_data_fixed, geojson, f"{selected_product} - Şehir Bazlı Satış Dağılımı")
                if turkey_map:
                    st.plotly_chart(turkey_map, use_container_width=True)
                else:
                    st.error("❌ Harita oluşturulamadı")
            else:
                st.warning("⚠️ turkey.geojson bulunamadı")
        
        st.markdown("---")
        st.subheader("🏆 Top 10 Şehir")
        top_cities = city_data.nlargest(10, 'PF_Satis')
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_bar = px.bar(top_cities, x='City', y='PF_Satis',
                            title='En Yüksek Satış Yapan Şehirler',
                            color='Pazar_Payi_%', color_continuous_scale='Blues')
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_chart2:
            fig_pie = px.pie(top_cities, values='PF_Satis', names='City', title='Top 10 Şehir Satış Dağılımı')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # TAB 3: TERRITORY ANALİZİ
    with tab3:
        st.header("🏢 Territory Bazlı Detaylı Analiz")
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        
        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            sort_by = st.selectbox("Sıralama", ['PF_Satis', 'Pazar_Payi_%', 'Toplam_Pazar', 'Agirlik_%'],
                                   format_func=lambda x: {'PF_Satis': 'PF Satış', 'Pazar_Payi_%': 'Pazar Payı %',
                                                          'Toplam_Pazar': 'Toplam Pazar', 'Agirlik_%': 'Ağırlık %'}[x])
        with col_f2:
            show_n = st.slider("Gösterilecek Territory Sayısı", 10, 50, 20)
        
        terr_sorted = terr_perf.sort_values(sort_by, ascending=False).head(show_n)
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("#### 📊 PF vs Rakip Satış")
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=terr_sorted['Territory'], y=terr_sorted['PF_Satis'],
                                     name='PF Satış', marker_color='#3B82F6'))
            fig_bar.add_trace(go.Bar(x=terr_sorted['Territory'], y=terr_sorted['Rakip_Satis'],
                                     name='Rakip Satış', marker_color='#EF4444'))
            fig_bar.update_layout(barmode='group', height=500, xaxis=dict(tickangle=-45))
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_v2:
            st.markdown("#### 🎯 Pazar Payı Dağılımı")
            fig_pie = px.pie(terr_sorted.head(10), values='PF_Satis', names='Territory',
                            title='Top 10 Territory - PF Satış Dağılımı')
            fig_pie.update_layout(height=500)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Detaylı Territory Listesi")
        display_cols = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 'Rakip_Satis',
                       'Toplam_Pazar', 'Pazar_Payi_%', 'Goreceli_Pazar_Payi', 'Agirlik_%']
        terr_display = terr_sorted[display_cols].copy()
        terr_display.columns = ['Territory', 'Region', 'City', 'Manager', 'PF Satış', 'Rakip Satış',
                               'Toplam Pazar', 'Pazar Payı %', 'Göreceli Pay', 'Ağırlık %']
        terr_display.index = range(1, len(terr_display) + 1)
        st.dataframe(terr_display.style.format({'PF Satış': '{:,.0f}', 'Rakip Satış': '{:,.0f}',
                                                'Toplam Pazar': '{:,.0f}', 'Pazar Payı %': '{:.1f}',
                                                'Göreceli Pay': '{:.2f}', 'Ağırlık %': '{:.1f}'})
                    .background_gradient(subset=['Pazar Payı %'], cmap='RdYlGn'),
                    use_container_width=True)
    
    # TAB 4: ZAMAN SERİSİ & ML (Mevcut kod korundu)
    with tab4:
        st.header("📈 Zaman Serisi Analizi & GERÇEK ML Tahminleme")
        territory_for_ts = st.selectbox("Territory Seçin", ["TÜMÜ"] + sorted(df_filtered['TERRITORIES'].unique()), key='ts_territory')
        monthly_df = calculate_time_series(df_filtered, selected_product, territory_for_ts, date_filter)
        
        if len(monthly_df) == 0:
            st.warning("⚠️ Seçilen filtrelerde veri bulunamadı")
        else:
            col_ts1, col_ts2, col_ts3, col_ts4 = st.columns(4)
            with col_ts1:
                st.metric("📊 Ort. Aylık PF", f"{monthly_df['PF_Satis'].mean():,.0f}")
            with col_ts2:
                st.metric("📈 Ort. Büyüme", f"%{monthly_df['PF_Buyume_%'].mean():.1f}")
            with col_ts3:
                st.metric("🎯 Ort. Pazar Payı", f"%{monthly_df['Pazar_Payi_%'].mean():.1f}")
            with col_ts4:
                st.metric("📅 Veri Dönemi", f"{len(monthly_df)} ay")
            
            st.markdown("---")
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("#### 📊 Satış Trendi")
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(x=monthly_df['DATE'], y=monthly_df['PF_Satis'],
                                           mode='lines+markers', name='PF Satış',
                                           line=dict(color='#3B82F6', width=3), marker=dict(size=8)))
                fig_ts.add_trace(go.Scatter(x=monthly_df['DATE'], y=monthly_df['Rakip_Satis'],
                                           mode='lines+markers', name='Rakip Satış',
                                           line=dict(color='#EF4444', width=3), marker=dict(size=8)))
                fig_ts.add_trace(go.Scatter(x=monthly_df['DATE'], y=monthly_df['MA_3'],
                                           mode='lines', name='3 Aylık Ort.',
                                           line=dict(color='#10B981', width=2, dash='dash')))
                fig_ts.update_layout(xaxis_title='Tarih', yaxis_title='Satış', height=400, hovermode='x unified')
                st.plotly_chart(fig_ts, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 🎯 Pazar Payı Trendi")
                fig_share = go.Figure()
                fig_share.add_trace(go.Scatter(x=monthly_df['DATE'], y=monthly_df['Pazar_Payi_%'],
                                              mode='lines+markers', fill='tozeroy',
                                              line=dict(color='#8B5CF6', width=2), marker=dict(size=8)))
                fig_share.update_layout(xaxis_title='Tarih', yaxis_title='Pazar Payı (%)', height=400)
                st.plotly_chart(fig_share, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🤖 GERÇEK Machine Learning Satış Tahmini")
            forecast_months = st.slider("Tahmin Periyodu (Ay)", 1, 6, 3)
            
            if len(monthly_df) < 10:
                st.warning("⚠️ Tahmin için yeterli veri yok (en az 10 ay gerekli)")
            else:
                with st.spinner("ML modelleri eğitiliyor..."):
                    ml_results, best_model_name, forecast_df = train_ml_models(monthly_df, forecast_months)
                
                if ml_results:
                    st.markdown("#### 📊 Model Performans Karşılaştırması")
                    perf_data = [{'Model': name, 'MAE': m['MAE'], 'RMSE': m['RMSE'], 'MAPE (%)': m['MAPE']}
                                for name, m in ml_results.items()]
                    perf_df = pd.DataFrame(perf_data).sort_values('MAPE (%)')
                    col_perf1, col_perf2 = st.columns([2, 1])
                    with col_perf1:
                        st.dataframe(perf_df.style.format({'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'MAPE (%)': '{:.2f}'})
                                    .background_gradient(subset=['MAPE (%)'], cmap='RdYlGn_r'), use_container_width=True)
                    with col_perf2:
                        st.success(f"**🏆 En İyi Model:**\n\n{best_model_name}")
                        best_mape = ml_results[best_model_name]['MAPE']
                        confidence = "🟢 YÜKSEK" if best_mape < 10 else ("🟡 ORTA" if best_mape < 20 else "🔴 DÜŞÜK")
                        st.metric("Güven Seviyesi", confidence)
                        st.metric("MAPE", f"{best_mape:.2f}%")
                    
                    st.markdown("#### 📈 Gerçek vs ML Tahmini")
                    fig_ml = create_forecast_chart(monthly_df, forecast_df)
                    st.plotly_chart(fig_ml, use_container_width=True)
    
    # TAB 5: RAKİP ANALİZİ (DÜZELTİLMİŞ TABLO)
    with tab5:
        st.header("📊 Detaylı Rakip Analizi")
        comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
        
        if len(comp_data) == 0:
            st.warning("⚠️ Seçilen filtrelerde veri bulunamadı")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Ort. PF Pazar Payı", f"%{comp_data['PF_Pay_%'].mean():.1f}")
            with col2:
                st.metric("📈 Ort. PF Büyüme", f"%{comp_data['PF_Buyume'].mean():.1f}")
            with col3:
                st.metric("📉 Ort. Rakip Büyüme", f"%{comp_data['Rakip_Buyume'].mean():.1f}")
            with col4:
                win_months = len(comp_data[comp_data['Fark'] > 0])
                st.metric("🏆 Kazanılan Aylar", f"{win_months}/{len(comp_data)}")
            
            st.markdown("---")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.subheader("💰 Satış Karşılaştırması")
                st.plotly_chart(create_competitor_comparison_chart(comp_data), use_container_width=True)
            with col_g2:
                st.subheader("📊 Pazar Payı Trendi")
                st.plotly_chart(create_market_share_trend(comp_data), use_container_width=True)
            
            st.markdown("---")
            st.subheader("📈 Büyüme Karşılaştırması")
            st.plotly_chart(create_growth_comparison(comp_data), use_container_width=True)
            
            st.markdown("---")
            st.subheader("📋 Aylık Performans Detayları")
            
            comp_display = comp_data[['YIL_AY', 'PF', 'Rakip', 'PF_Pay_%', 'PF_Buyume', 'Rakip_Buyume', 'Fark']].copy()
            comp_display.columns = ['Ay', 'PF Satış', 'Rakip Satış', 'PF Pay %', 'PF Büyüme %', 'Rakip Büyüme %', 'Fark %']
            
            # DARK THEME UYUMLU STİL (DÜZELTİLMİŞ)
            def highlight_winner(row):
                fark = row['Fark %']
                if pd.isna(fark):
                    return [''] * len(row)
                elif fark > 0:
                    return ['background-color: rgba(16,185,129,0.25); color: white'] * len(row)
                elif fark < 0:
                    return ['background-color: rgba(239,68,68,0.25); color: white'] * len(row)
                else:
                    return [''] * len(row)
            
            styler = (comp_display.style
                     .format({'PF Satış': '{:,.0f}', 'Rakip Satış': '{:,.0f}',
                             'PF Pay %': '{:.1f}', 'PF Büyüme %': '{:.1f}',
                             'Rakip Büyüme %': '{:.1f}', 'Fark %': '{:.1f}'})
                     .apply(highlight_winner, axis=1)
                     .set_properties(**{
                         "background-color": "rgba(30,41,59,0.85)",
                         "color": "#f1f5f9",
                         "border-color": "rgba(148,163,184,0.2)"
                     })
                     .set_table_styles([
                         {"selector": "th", "props": [("background-color", "rgba(15,23,42,1)"),
                                                      ("color", "#f1f5f9"), ("font-weight", "700")]},
                         {"selector": "td", "props": [("padding", "8px")]},
                     ]))
            
            st.dataframe(styler, use_container_width=True, hide_index=True)
    
    # TAB 6: BCG MATRIX (Mevcut kod korundu)
    with tab6:
        st.header("⭐ BCG Matrix & Yatırım Stratejisi")
        bcg_df = calculate_bcg_matrix(df_filtered, selected_product, date_filter)
        
        bcg_counts = bcg_df['BCG_Kategori'].value_counts()
        col_bcg1, col_bcg2, col_bcg3, col_bcg4 = st.columns(4)
        with col_bcg1:
            star_count = bcg_counts.get("⭐ Star", 0)
            star_pf = bcg_df[bcg_df['BCG_Kategori'] == "⭐ Star"]['PF_Satis'].sum()
            st.metric("⭐ Star", f"{star_count}", delta=f"{star_pf:,.0f} PF")
        with col_bcg2:
            cow_count = bcg_counts.get("🐄 Cash Cow", 0)
            cow_pf = bcg_df[bcg_df['BCG_Kategori'] == "🐄 Cash Cow"]['PF_Satis'].sum()
            st.metric("🐄 Cash Cow", f"{cow_count}", delta=f"{cow_pf:,.0f} PF")
        with col_bcg3:
            q_count = bcg_counts.get("❓ Question Mark", 0)
            q_pf = bcg_df[bcg_df['BCG_Kategori'] == "❓ Question Mark"]['PF_Satis'].sum()
            st.metric("❓ Question", f"{q_count}", delta=f"{q_pf:,.0f} PF")
        with col_bcg4:
            dog_count = bcg_counts.get("🐶 Dog", 0)
            dog_pf = bcg_df[bcg_df['BCG_Kategori'] == "🐶 Dog"]['PF_Satis'].sum()
            st.metric("🐶 Dog", f"{dog_count}", delta=f"{dog_pf:,.0f} PF")
        
        st.markdown("---")
        st.subheader("🎯 BCG Matrix")
        color_map = {"⭐ Star": "#FFD700", "🐄 Cash Cow": "#10B981",
                    "❓ Question Mark": "#3B82F6", "🐶 Dog": "#9CA3AF"}
        
        fig_bcg = px.scatter(bcg_df, x='Goreceli_Pazar_Payi', y='Pazar_Buyume_%', size='PF_Satis',
                            color='BCG_Kategori', color_discrete_map=color_map, hover_name='Territory',
                            hover_data={'PF_Satis': ':,.0f', 'Pazar_Payi_%': ':.1f',
                                       'Goreceli_Pazar_Payi': ':.2f', 'Pazar_Buyume_%': ':.1f'},
                            labels={'Goreceli_Pazar_Payi': 'Göreceli Pazar Payı (PF/Rakip)',
                                   'Pazar_Buyume_%': 'Pazar Büyüme Oranı (%)'}, size_max=50)
        
        median_share = bcg_df['Goreceli_Pazar_Payi'].median()
        median_growth = bcg_df['Pazar_Buyume_%'].median()
        fig_bcg.add_hline(y=median_growth, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        fig_bcg.add_vline(x=median_share, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        fig_bcg.update_layout(title='BCG Matrix - Stratejik Konumlandırma', height=600,
                             plot_bgcolor='#0f172a', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'))
        st.plotly_chart(fig_bcg, use_container_width=True)
    
    # TAB 7: YENİ ANALİZLER
    with tab7:
        st.header("🧠 Yeni Analizler")
        
        # 1. PARETO ANALİZİ
        st.subheader("📊 Pareto Analizi (80/20 Kuralı)")
        pareto_df, sehir_80 = calculate_pareto_analysis(df_filtered, selected_product, date_filter)
        
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("🎯 %80 Satış", f"{sehir_80} şehirden", delta="geliyor")
        with col_p2:
            top_10_contrib = pareto_df.head(10)['PF_Satis'].sum() / pareto_df['PF_Satis'].sum() * 100
            st.metric("🏆 Top 10 Katkı", f"%{top_10_contrib:.1f}")
        with col_p3:
            risk_level = "🟢 DÜŞÜK" if sehir_80 > 20 else ("🟡 ORTA" if sehir_80 > 10 else "🔴 YÜKSEK")
            st.metric("⚠️ Konsantrasyon Riski", risk_level)
        
        fig_pareto = go.Figure()
        top30 = pareto_df.head(30)
        fig_pareto.add_trace(go.Bar(x=top30['City'], y=top30['PF_Satis'], name='PF Satış',
                                    marker_color='#3B82F6', yaxis='y'))
        fig_pareto.add_trace(go.Scatter(x=top30['City'], y=top30['Kumulative_%'], name='Kümülatif %',
                                       mode='lines+markers', marker=dict(size=8, color='#EF4444'),
                                       line=dict(width=3, color='#EF4444'), yaxis='y2'))
        fig_pareto.add_hline(y=80, line_dash="dash", line_color="#10B981", annotation_text="80% hedefi", yref='y2')
        fig_pareto.update_layout(title="Pareto Analizi: %80 Satışı Hangi Şehirler Yapıyor?",
                                height=500, xaxis=dict(tickangle=-45),
                                yaxis=dict(title='PF Satış'), yaxis2=dict(title='Kümülatif %',
                                overlaying='y', side='right', range=[0, 100]))
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        st.markdown("---")
        
        # 2. VOLATİLİTE ANALİZİ
        st.subheader("📉 Volatilite & Stabilite Analizi")
        vol_df = calculate_volatility_score(df_filtered, selected_product, date_filter)
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("##### 🟢 En Stabil 10 Şehir")
            stable = vol_df.head(10)
            st.dataframe(stable.style.format({'Volatilite_Std': '{:.2f}'}), use_container_width=True, hide_index=True)
        with col_v2:
            st.markdown("##### 🔴 En Volatil 10 Şehir")
            volatile = vol_df.tail(10).sort_values('Volatilite_Std', ascending=False)
            st.dataframe(volatile.style.format({'Volatilite_Std': '{:.2f}'}), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # 3. FIRSAT SKORU
        st.subheader("💎 Fırsat Skoru Analizi")
        opp_df = calculate_opportunity_score(df_filtered, selected_product, date_filter)
        top15_opp = opp_df.head(15)
        
        fig_opp = px.bar(top15_opp, x='City', y='Firsat_Skoru', color='Pazar_Payi_%',
                        title='Top 15 Fırsat Şehri', color_continuous_scale='Reds',
                        labels={'Firsat_Skoru': 'Fırsat Skoru (Potansiyel)', 'Pazar_Payi_%': 'Mevcut Pay %'})
        fig_opp.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig_opp, use_container_width=True)
        
        st.markdown("---")
        
        # 4. YOY HEATMAP
        st.subheader("🔥 Yıl x Ay Satış Isı Haritası")
        yoy_pivot = calculate_yoy_heatmap(df_filtered, selected_product)
        
        fig_heatmap = px.imshow(yoy_pivot, labels=dict(x="Ay", y="Yıl", color="PF Satış"),
                               color_continuous_scale='YlOrRd', aspect="auto", text_auto='.0f')
        fig_heatmap.update_layout(title="Yıllık/Aylık Satış Trendi", height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # TAB 8: RAPORLAR
    with tab8:
        st.header("📥 Rapor İndirme")
        st.markdown("Detaylı analizlerin Excel raporlarını indirebilirsiniz.")
        
        if st.button("📥 Excel Raporu Oluştur", type="primary"):
            with st.spinner("Rapor hazırlanıyor..."):
                terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
                monthly_df = calculate_time_series(df_filtered, selected_product, None, date_filter)
                bcg_df = calculate_bcg_matrix(df_filtered, selected_product, date_filter)
                city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
                comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
                pareto_df, _ = calculate_pareto_analysis(df_filtered, selected_product, date_filter)
                vol_df = calculate_volatility_score(df_filtered, selected_product, date_filter)
                opp_df = calculate_opportunity_score(df_filtered, selected_product, date_filter)
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    terr_perf.to_excel(writer, sheet_name='Territory Performans', index=False)
                    monthly_df.to_excel(writer, sheet_name='Zaman Serisi', index=False)
                    bcg_df.to_excel(writer, sheet_name='BCG Matrix', index=False)
                    city_data.to_excel(writer, sheet_name='Şehir Analizi', index=False)
                    comp_data.to_excel(writer, sheet_name='Rakip Analizi', index=False)
                    pareto_df.to_excel(writer, sheet_name='Pareto Analizi', index=False)
                    vol_df.to_excel(writer, sheet_name='Volatilite', index=False)
                    opp_df.to_excel(writer, sheet_name='Fırsat Skoru', index=False)
                
                st.success("✅ Rapor hazır!")
                st.download_button(label="💾 Excel Raporunu İndir", data=output.getvalue(),
                                  file_name=f"ticari_portfoy_raporu_{selected_product}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                  mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
