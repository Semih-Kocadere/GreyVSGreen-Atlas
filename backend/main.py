"""
Grey vs Green Atlas - Backend API
==================================
İstanbul'un yeşil/gri alan dengesini izleyen ve tahmin eden web uygulaması.

Teknolojiler:
- FastAPI: Modern, hızlı web framework
- SQLModel: SQL veritabanı ORM
- JWT: Güvenli kullanıcı kimlik doğrulama
- Leaflet.js: İnteraktif harita görselleştirme

Veri Kaynakları:
- İBB Açık Veri Portalı
- TÜİK (Türkiye İstatistik Kurumu)
- Sentinel-2 Uydu Görüntüleri
"""

from datetime import datetime, timedelta
from typing import Optional
import os
import json
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlmodel import SQLModel, Field, Session, create_engine, select

# Tile servisi import et
from tile_service import get_tile_response

# Model servisi import et
try:
    from model_service import (
        model_service,
        get_available_periods,
        get_tiles_for_period,
        predict_period,
        calculate_statistics_from_softmax
    )
    MODEL_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Model servisi yüklenemedi: {e}")
    MODEL_SERVICE_AVAILABLE = False


# ============================================================================
# YAPILANDIRMA (Configuration)
# ============================================================================

# JWT Token ayarları
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-please-very-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 12 * 60  # 12 saat

# Veritabanı bağlantısı
DB_URL = os.getenv("DB_URL", "sqlite:///./db.sqlite")

# Veri dosyalarının bulunduğu klasör
DATA_DIR = Path(__file__).parent / "data"


# ============================================================================
# VERİ YÜKLEME FONKSİYONLARI (Data Loading Functions)
# ============================================================================

def load_istanbul_data():
    """
    İstanbul için JSON dosyasından veri yükler.
    
    Dosya: backend/data/istanbul_data.json
    İçerik: Tarihsel veriler, tahminler, bölgeler, ilçeler, özet istatistikler
    
    Returns:
        dict: İstanbul verileri veya None (dosya yoksa)
    """
    data_file = DATA_DIR / "istanbul_data.json"
    
    if not data_file.exists():
        print(f"⚠️  UYARI: {data_file} bulunamadı!")
        return None
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✅ İstanbul verisi yüklendi: {len(data.get('districts', []))} ilçe")
            return data
    except Exception as e:
        print(f"❌ HATA: İstanbul verisi yüklenemedi: {e}")
        return None


# Veriyi uygulama başlangıcında yükle ve cache'le
ISTANBUL_DATA = load_istanbul_data()


# ============================================================================
# VERİTABANI AYARLARI (Database Setup)
# ============================================================================

# SQLite veritabanı bağlantısı oluştur
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

# Şifre hash'leme için bcrypt kullan
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(SQLModel, table=True):
    """
    Kullanıcı veritabanı modeli.
    
    Attributes:
        id: Otomatik artan benzersiz kimlik
        email: Kullanıcı email adresi (unique)
        password_hash: Bcrypt ile hash'lenmiş şifre
        full_name: Kullanıcının tam adı (opsiyonel)
        is_active: Hesap aktif mi? (varsayılan: True)
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    email: EmailStr
    password_hash: str
    full_name: Optional[str] = None
    is_active: bool = True


def create_db_and_tables():
    """Veritabanı tablolarını oluştur."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """
    Veritabanı session'ı oluştur ve yönet.
    Dependency injection için kullanılır.
    """
    with Session(engine) as session:
        yield session


# ============================================================================
# PYDANTIC ŞEMALARI (Request/Response Models)
# ============================================================================

class UserCreate(BaseModel):
    """Kullanıcı kaydı için gelen veri şeması."""
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserOut(BaseModel):
    """API'den dönen kullanıcı verisi (şifre hariç)."""
    id: int
    email: EmailStr
    full_name: Optional[str]


class Token(BaseModel):
    """JWT token response şeması."""
    access_token: str
    token_type: str = "bearer"


# ============================================================================
# KİMLİK DOĞRULAMA FONKSİYONLARI (Authentication Functions)
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Şifre doğrulaması yap.
    
    Args:
        plain_password: Kullanıcının girdiği düz metin şifre
        hashed_password: Veritabanındaki hash'lenmiş şifre
    
    Returns:
        bool: Şifre eşleşirse True, değilse False
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    """
    JWT access token oluştur.
    
    Args:
        data: Token'a eklenecek veriler (genellikle {"sub": email})
        minutes: Token geçerlilik süresi (dakika)
    
    Returns:
        str: JWT token string'i
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# OAuth2 token şeması (header'da Bearer token bekler)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session)
) -> User:
    """
    JWT token'dan mevcut kullanıcıyı al.
    Korumalı endpoint'lerde dependency olarak kullanılır.
    
    Args:
        token: Authorization header'dan gelen JWT token
        session: Veritabanı session'ı
    
    Returns:
        User: Aktif kullanıcı nesnesi
    
    Raises:
        HTTPException: Token geçersiz veya kullanıcı bulunamazsa
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Kimlik doğrulama başarısız",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Token'ı decode et
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        
        if email is None:
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    # Kullanıcıyı veritabanından bul
    user = session.exec(select(User).where(User.email == email)).first()
    
    if user is None or not user.is_active:
        raise credentials_exception
    
    return user


# ============================================================================
# FASTAPİ UYGULAMASI (FastAPI Application)
# ============================================================================


# FastAPI uygulaması ve CORS ayarları
app = FastAPI(
    title="Grey vs Green Atlas API",
    description="İstanbul yeşil alan takip ve tahmin sistemi",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uygulama başlatılırken sadece veritabanı tabloları oluşturulsun
@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    print("🚀 Backend başlatıldı!")
    # Model yükleme işlemi, ilk tahmin isteğinde model_service tarafından tetiklenecek
    # Softmaps klasörü eksikse veya model dosyaları yoksa, ilgili endpoint hata mesajı dönecek


# ============================================================================
# KİMLİK DOĞRULAMA API'LERİ (Authentication Endpoints)
# ============================================================================

@app.post("/auth/register", response_model=UserOut)
def register(payload: UserCreate, session: Session = Depends(get_session)):
    """
    Yeni kullanıcı kaydı.
    
    Body Parametreleri:
        email (str): Kullanıcı email adresi
        password (str): Şifre (en az 6 karakter)
        full_name (str, opsiyonel): Tam ad
    
    Döner:
        UserOut: Oluşturulan kullanıcı bilgileri (şifre hariç)
    
    Hatalar:
        400: Email zaten kayıtlı
    """
    # Email zaten kayıtlı mı kontrol et
    existing_user = session.exec(
        select(User).where(User.email == payload.email)
    ).first()
    
    if existing_user:
        raise HTTPException(400, "Bu email adresi zaten kayıtlı")
    
    # Yeni kullanıcı oluştur
    new_user = User(
        email=payload.email,
        password_hash=pwd_context.hash(payload.password),
        full_name=payload.full_name
    )
    
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    
    return UserOut(
        id=new_user.id,
        email=new_user.email,
        full_name=new_user.full_name
    )


@app.post("/auth/login", response_model=Token)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session)
):
    """
    Kullanıcı girişi - JWT token döner.
    
    Form Parametreleri:
        username (str): Email adresi
        password (str): Şifre
    
    Döner:
        Token: JWT access token ve token tipi
    
    Hatalar:
        400: Email veya şifre hatalı
    """
    # Kullanıcıyı bul (OAuth2 form'da username alanı email için kullanılır)
    user = session.exec(
        select(User).where(User.email == form.username)
    ).first()
    
    # Kullanıcı yoksa veya şifre yanlışsa hata ver
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(400, "Email veya şifre hatalı")
    
    # JWT token oluştur
    access_token = create_access_token({"sub": user.email})
    
    return Token(access_token=access_token)


@app.get("/me", response_model=UserOut)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Mevcut kullanıcının bilgilerini döner.
    
    Headers:
        Authorization: Bearer <token>
    
    Döner:
        UserOut: Kullanıcı bilgileri
    
    Hatalar:
        401: Token geçersiz veya eksik
    """
    return UserOut(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name
    )


# ============================================================================
# ŞEHİR VERİLERİ API'LERİ (City Data Endpoints)
# ============================================================================

@app.get("/api/city/istanbul")
def get_istanbul_summary(current_user: User = Depends(get_current_user)):
    """
    İstanbul özet istatistiklerini döner.
    Dashboard sayfasında kullanılır.
    
    Headers:
        Authorization: Bearer <token>
    
    Döner:
        dict: Şimdi, +6 ay, +12 ay tahminleri
    
    Örnek Response:
        {
            "title": "İstanbul — Yeşil/Gri Özet",
            "now": {"green": 31.2, "grey": 61.8, "water": 7.0},
            "+6m": {...},
            "+12m": {...},
            "note": "Veriler: İBB, TÜİK"
        }
    """
    if not ISTANBUL_DATA:
        raise HTTPException(500, "İstanbul verisi yüklenemedi")
    
    return {
        "title": "İstanbul — Yeşil/Gri Özet",
        "now": ISTANBUL_DATA["predictions"][0],
        "+6m": ISTANBUL_DATA["predictions"][1],
        "+12m": ISTANBUL_DATA["predictions"][2],
        "note": f"Veriler: {', '.join(ISTANBUL_DATA['metadata']['sources'][:2])}"
    }


# ============================================================================
# HARİTA API'LERİ (Map Endpoints)
# ============================================================================

@app.get("/api/map/turkey")
def get_turkey_map_data(current_user: User = Depends(get_current_user)):
    """
    Türkiye geneli büyük şehirlerin harita verilerini döner.
    GeoJSON formatında Point feature'lar.
    
    Kullanım: map.html sayfasında Türkiye haritası için
    Durum: Sadece İstanbul aktif, diğer şehirler "Yakında Aktif"
    
    Headers:
        Authorization: Bearer <token>
    
    Döner:
        dict: GeoJSON FeatureCollection formatında şehir verileri
    
    Örnek Response:
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [28.9784, 41.0082]},
                    "properties": {
                        "name": "İstanbul",
                        "active": true,
                        "population": 15840900,
                        "url": "/istanbul-detail.html"
                    }
                },
                ...
            ]
        }
    """
    # Türkiye'nin büyük şehirleri (TÜİK verileri) - İstanbul aktif ve özel
    cities = [
        # En büyük şehir - Aktif ve özel vurgu
        {"name": "İstanbul", "lat": 41.0082, "lng": 28.9784, "active": True, "population": 15840900},
        
        # Diğer büyük şehirler
        {"name": "Ankara", "lat": 39.9334, "lng": 32.8597, "active": False, "population": 5663322},
        {"name": "İzmir", "lat": 38.4237, "lng": 27.1428, "active": False, "population": 4425789},
        {"name": "Bursa", "lat": 40.1826, "lng": 29.0665, "active": False, "population": 3147818},
        {"name": "Antalya", "lat": 36.8969, "lng": 30.7133, "active": False, "population": 2619832},
        {"name": "Adana", "lat": 37.0000, "lng": 35.3213, "active": False, "population": 2258718},
        {"name": "Konya", "lat": 37.8714, "lng": 32.4846, "active": False, "population": 2277017},
        {"name": "Gaziantep", "lat": 37.0662, "lng": 37.3833, "active": False, "population": 2101157},
        {"name": "Şanlıurfa", "lat": 37.1591, "lng": 38.7969, "active": False, "population": 2115256},
        {"name": "Kocaeli", "lat": 40.8533, "lng": 29.8815, "active": False, "population": 1997258},
        {"name": "Mersin", "lat": 36.8121, "lng": 34.6415, "active": False, "population": 1891145},
        {"name": "Kayseri", "lat": 38.7312, "lng": 35.4787, "active": False, "population": 1434357},
        {"name": "Eskişehir", "lat": 39.7767, "lng": 30.5206, "active": False, "population": 887475},
        {"name": "Diyarbakır", "lat": 37.9144, "lng": 40.2306, "active": False, "population": 1783431},
        {"name": "Samsun", "lat": 41.2867, "lng": 36.3300, "active": False, "population": 1356079},
        {"name": "Denizli", "lat": 37.7765, "lng": 29.0864, "active": False, "population": 1040915},
        {"name": "Adapazarı", "lat": 40.7569, "lng": 30.4046, "active": False, "population": 439262},
        {"name": "Malatya", "lat": 38.3552, "lng": 38.3095, "active": False, "population": 803930},
        {"name": "Kahramanmaraş", "lat": 37.5847, "lng": 36.9233, "active": False, "population": 1168163},
        {"name": "Erzurum", "lat": 39.9208, "lng": 41.2675, "active": False, "population": 762062},
        {"name": "Van", "lat": 38.4891, "lng": 43.4089, "active": False, "population": 1136757},
        {"name": "Batman", "lat": 37.8812, "lng": 41.1351, "active": False, "population": 608659},
        {"name": "Elazığ", "lat": 38.6810, "lng": 39.2264, "active": False, "population": 591098},
        {"name": "Sivas", "lat": 39.7477, "lng": 37.0179, "active": False, "population": 646608},
        {"name": "Manisa", "lat": 38.6191, "lng": 27.4289, "active": False, "population": 1429643},
        {"name": "Tekirdağ", "lat": 40.9833, "lng": 27.5167, "active": False, "population": 1055412},
        {"name": "Balıkesir", "lat": 39.6484, "lng": 27.8826, "active": False, "population": 1257590},
        {"name": "Aydın", "lat": 37.8560, "lng": 27.8416, "active": False, "population": 1119084},
        {"name": "Trabzon", "lat": 41.0015, "lng": 39.7178, "active": False, "population": 811901},
        {"name": "Ordu", "lat": 40.9839, "lng": 37.8764, "active": False, "population": 771932},
    ]
    
    # GeoJSON Feature listesi oluştur
    features = []
    for city in cities:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [city["lng"], city["lat"]]  # GeoJSON formatı: [lng, lat]
            },
            "properties": {
                "name": city["name"],
                "active": city["active"],
                "population": city["population"],
                # Aktif şehirlere detay sayfası linki ekle
                "url": "/istanbul-detail.html" if city["active"] else None
            }
        })
    
    return {
        "type": "FeatureCollection",
        "features": features
    }

@app.get("/api/map/istanbul/{timeframe}")
def get_istanbul_map_data(
    timeframe: str,
    current_user: User = Depends(get_current_user)
):
    """
    İstanbul ilçe bazlı yeşil/gri alan verilerini döner.
    Harita görselleştirmesi için GeoJSON formatında.
    
    Kullanım: map.html sayfasında İstanbul zoom görünümü için
    
    Path Parametresi:
        timeframe (str): Zaman dilimi
            - 'now': Şu anki durum
            - '6m': 6 ay sonra tahmini
            - '12m': 12 ay sonra tahmini
    
    Headers:
        Authorization: Bearer <token>
    
    Döner:
        dict: İlçelerin GeoJSON Point feature'ları
    
    Hatalar:
        400: Geçersiz timeframe
        500: İstanbul verisi yüklenemedi
    
    Örnek Response:
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [29.0875, 41.1060]},
                    "properties": {
                        "name": "Sarıyer",
                        "green": 65.0,
                        "grey": 28.0,
                        "water": 7.0,
                        "timeframe": "now"
                    }
                },
                ...
            ],
            "metadata": {
                "city": "Istanbul",
                "timeframe": "now",
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    """
    # Timeframe geçerliliğini kontrol et
    if timeframe not in ["now", "6m", "12m"]:
        raise HTTPException(
            400,
            "Geçersiz timeframe. Kullanılabilir: now, 6m, 12m"
        )
    
    # İstanbul verisi yüklü mü kontrol et
    if not ISTANBUL_DATA or "districts" not in ISTANBUL_DATA:
        raise HTTPException(500, "İstanbul ilçe verisi bulunamadı")
    
    # İlçe verilerini timeframe'e göre hazırla
    districts_with_predictions = []
    for district in ISTANBUL_DATA["districts"]:
        # Su yüzeyini sabit tut (%7)
        water_percentage = 7
        
        # Gri alanı hesapla: %100 - yeşil - su
        now_grey = 100 - district["now_green"] - water_percentage
        future_grey = 100 - district["future_green"] - water_percentage
        
        districts_with_predictions.append({
            "name": district["name"],
            "lat": district["lat"],
            "lng": district["lng"],
            # Her timeframe için ayrı veriler
            "now": {
                "green": district["now_green"],
                "grey": now_grey,
                "water": water_percentage
            },
            "6m": {
                "green": district["future_green"],
                "grey": future_grey,
                "water": water_percentage
            },
            "12m": {
                "green": district["future_green"],
                "grey": future_grey,
                "water": water_percentage
            }
        })
    
    # GeoJSON Feature'ları oluştur
    features = []
    for district in districts_with_predictions:
        data = district[timeframe]  # İstenen timeframe'in verisini al
        
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [district["lng"], district["lat"]]
            },
            "properties": {
                "name": district["name"],
                "green": data["green"],
                "grey": data["grey"],
                "water": data["water"],
                "timeframe": timeframe
            }
        })
    
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "city": "Istanbul",
            "timeframe": timeframe,
            "timestamp": datetime.utcnow().isoformat()
        }
    }


# ============================================================================
# DETAYLI ANALİZ API'LERİ (Detailed Analysis Endpoints)
# ============================================================================

@app.get("/api/istanbul/detailed")
def get_istanbul_detailed_analysis(current_user: User = Depends(get_current_user)):
    """
    İstanbul için kapsamlı detaylı analiz verilerini döner.
    
    Kullanım: istanbul-detail.html sayfasında görselleştirmeler için
    
    İçerik:
        - Bölgesel grid verileri (9 bölge)
        - Tarihsel trendler (2019-2024)
        - Gelecek tahminleri (şimdi, +6 ay, +12 ay)
        - İlçe bazlı değişimler (39 ilçe)
        - Özet istatistikler (nüfus, alan, yeşil alan/kişi)
        - Metadata (veri kaynakları, son güncelleme)
    
    Headers:
        Authorization: Bearer <token>
    
    Döner:
        dict: Kapsamlı İstanbul analiz verileri
    
    Hatalar:
        500: İstanbul verisi yüklenemedi
    
    Örnek Response:
        {
            "city": "Istanbul",
            "grid": [
                {
                    "id": "tarihi_yarimada",
                    "name": "Tarihi Yarımada",
                    "lat": 41.0082,
                    "lng": 28.9784,
                    "now": {"green": 15, "grey": 78, "water": 7},
                    "6m": {...},
                    "12m": {...}
                },
                ...
            ],
            "historical": [
                {"year": 2019, "green": 36.8, "grey": 56.2, "water": 7.0},
                ...
            ],
            "predictions": [
                {"timeframe": "now", "green": 31.2, ...},
                ...
            ],
            "district_changes": [...],
            "summary": {
                "population": 15907951,
                "total_area_km2": 5461,
                "green_per_capita_m2": 10.7,
                ...
            },
            "metadata": {
                "sources": ["İBB", "TÜİK", "Sentinel-2"],
                ...
            }
        }
    """
    # İstanbul verisi yüklü mü kontrol et
    if not ISTANBUL_DATA:
        raise HTTPException(500, "İstanbul detaylı verisi yüklenemedi")
    
    # Bölgeleri grid formatına dönüştür
    # Her bölge haritada bir daire olarak gösterilecek
    grid_data = []
    for region in ISTANBUL_DATA["regions"]:
        # Bölge isminden id oluştur (küçük harf, boşluksuz)
        region_id = (
            region["name"]
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("ç", "c")
            .replace("ğ", "g")
            .replace("ı", "i")
            .replace("ö", "o")
            .replace("ş", "s")
            .replace("ü", "u")
        )
        
        grid_data.append({
            "id": region_id,
            "name": region["name"],
            "lat": region["center"][0],  # Enlem
            "lng": region["center"][1],  # Boylam
            "now": region["now"],        # Şu anki durum
            "6m": region["6m"],          # 6 ay sonra
            "12m": region["12m"]         # 12 ay sonra
        })
    
    # Tüm verileri tek response'ta döndür
    return {
        "city": "Istanbul",
        "grid": grid_data,                           # Bölgesel grid (9 bölge)
        "historical": ISTANBUL_DATA["historical"],   # 2019-2024 tarihsel veriler
        "predictions": ISTANBUL_DATA["predictions"], # 3 zaman dilimi tahmini
        "district_changes": ISTANBUL_DATA["districts"], # 39 ilçe detayları
        "summary": ISTANBUL_DATA["summary"],         # Özet istatistikler
        "metadata": ISTANBUL_DATA["metadata"],       # Veri kaynağı bilgileri
        "timestamp": datetime.utcnow().isoformat()   # API çağrı zamanı
    }


# ============================================================================
# TREND TAHMİNİ API'LERİ (Trend Prediction Endpoints)
# ============================================================================



# ============================================================================
# /api/trend/predict (Frontend için ana tahmin verisi endpointi)
# ============================================================================
from fastapi import Response

@app.get("/api/trend/predict")
def get_trend_predict(current_user: User = Depends(get_current_user)):
    """
    Trend tahmin verilerini döndürür (mock). Frontendin zaman çizelgesi ve harita için ana veri kaynağı.
    """
    # Örnek/mock veri (gerçek model çıktısı ile değiştirilebilir)
    years = [f"2024 Q{i}" for i in range(1, 5)] + [f"2025 Q{i}" for i in range(1, 5)] + [f"2026 Q1"]
    predictions = []
    import numpy as np
    for i, y in enumerate(years):
        # 256x256 boyutunda, 0-2 arası değerlerden oluşan örnek bir mask (her yıl için farklı pattern)
        mask = np.full((256, 256), i % 3, dtype=int)
        # Alternatif: rastgele maske için aşağıdaki satırı kullanabilirsin
        # mask = np.random.randint(0, 3, (256, 256)).tolist()
        predictions.append({
            "current": {
                "timeframe": y,
                "year": int(y.split()[0]),
                "quarter": int(y.split()[1][1:]),
                "green": 32.5 - i * 0.3,
                "grey": 60.2 + i * 0.3,
                "water": 7.3
            },
            "prediction": {
                "timeframe": years[min(i+1, len(years)-1)],
                "year": int(years[min(i+1, len(years)-1)].split()[0]),
                "quarter": int(years[min(i+1, len(years)-1)].split()[1][1:]),
                "green": 32.5 - (i+1) * 0.3,
                "grey": 60.2 + (i+1) * 0.3,
                "water": 7.3
            },
            "changes": {
                "green": -0.3,
                "grey": +0.3,
                "water": 0.0
            },
            "class_mask": mask.tolist()
        })
    return {
        "years": years,
        "predictions": predictions
    }

@app.get("/api/trend/historical")
def get_trend_historical(
    start_year: int = 2018,
    end_year: int = 2024,
    current_user: User = Depends(get_current_user)
):
    """
    Geçmiş dönem trend verilerini döndürür.
    
    Query Parametreleri:
        start_year (int): Başlangıç yılı (varsayılan: 2018)
        end_year (int): Bitiş yılı (varsayılan: 2024)
    
    Headers:
        Authorization: Bearer <token>
    
    Döner:
        dict: Çeyreklik geçmiş veriler
    
    Örnek Response:
        {
            "data": [
                {"period": "2018 Q1", "green": 35.2, "grey": 57.8, "water": 7.0},
                {"period": "2018 Q2", "green": 35.0, "grey": 58.0, "water": 7.0},
                ...
            ],
            "metadata": {
                "start": "2018 Q1",
                "end": "2024 Q4",
                "count": 28
            }
        }
    """
    # Mock data - gerçek veri dosyalarından yüklenecek
    data = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            # Trend: Yeşil azalıyor, gri artıyor
            period_idx = (year - start_year) * 4 + (quarter - 1)
            data.append({
                "period": f"{year} Q{quarter}",
                "green": 35.2 - (period_idx * 0.15),
                "grey": 57.8 + (period_idx * 0.15),
                "water": 7.0
            })
    
    return {
        "data": data,
        "metadata": {
            "start": f"{start_year} Q1",
            "end": f"{end_year} Q4",
            "count": len(data)
        }
    }


@app.get("/api/trend/tiles/{year}/{quarter}/{prediction_type}")
def get_trend_tiles_metadata(
    year: int,
    quarter: int,
    prediction_type: str,
    current_user: User = Depends(get_current_user)
):
    """
    Belirli bir dönem için tile metadata'sını döndürür.
    
    Path Parametreleri:
        year (int): Yıl
        quarter (int): Çeyrek (1-4)
        prediction_type (str): 'actual' veya 'predicted'
    
    Headers:
        Authorization: Bearer <token>
    
    Döner:
        dict: Tile URL şablonu ve metadata
    
    Örnek Response:
        {
            "tile_url": "/api/tiles/2025/1/predicted/{z}/{x}/{y}.png",
            "bounds": {
                "north": 41.5,
                "south": 40.5,
                "east": 29.5,
                "west": 28.0
            },
            "available_indices": ["ndvi", "ndwi", "ndbi", "rgb", "prediction"],
            "metadata": {
                "year": 2025,
                "quarter": 1,
                "type": "predicted"
            }
        }
    """
    if prediction_type not in ["actual", "predicted"]:
        raise HTTPException(400, "prediction_type must be 'actual' or 'predicted'")
    
    return {
        "tile_url": f"/api/tiles/{year}/{quarter}/{prediction_type}/{{z}}/{{x}}/{{y}}.png",
        "bounds": {
            "north": 41.5,
            "south": 40.5,
            "east": 29.5,
            "west": 28.0
        },
        "available_indices": ["ndvi", "ndwi", "ndbi", "rgb", "prediction"],
        "metadata": {
            "year": year,
            "quarter": quarter,
            "type": prediction_type
        }
    }


@app.get("/api/trend/comparison")
def get_trend_comparison(
    period1: str,
    period2: str,
    current_user: User = Depends(get_current_user)
):
    """
    İki dönem arasındaki farkları analiz eder.
    
    Query Parametreleri:
        period1 (str): İlk dönem (örn: "2024_Q4")
        period2 (str): İkinci dönem (örn: "2025_Q1")
    
    Headers:
        Authorization: Bearer <token>
    
    Döner:
        dict: Karşılaştırma ve değişim metrikleri
    
    Örnek Response:
        {
            "period1": {"label": "2024 Q4", "green": 32.5, ...},
            "period2": {"label": "2025 Q1", "green": 31.8, ...},
            "changes": {
                "green": {"absolute": -0.7, "percentage": -2.15},
                "grey": {"absolute": +0.9, "percentage": +1.50},
                "water": {"absolute": -0.2, "percentage": -2.74}
            },
            "analysis": {
                "trend": "Urbanization increasing",
                "green_loss_rate": "2.15% per quarter",
                "hotspots": ["Başakşehir", "Esenyurt", "Beylikdüzü"]
            }
        }
    """
    # Mock comparison data
    return {
        "period1": {
            "label": period1.replace("_", " "),
            "green": 32.5,
            "grey": 60.2,
            "water": 7.3
        },
        "period2": {
            "label": period2.replace("_", " "),
            "green": 31.8,
            "grey": 61.1,
            "water": 7.1
        },
        "changes": {
            "green": {"absolute": -0.7, "percentage": -2.15},
            "grey": {"absolute": +0.9, "percentage": +1.50},
            "water": {"absolute": -0.2, "percentage": -2.74}
        },
        "analysis": {
            "trend": "Urbanization increasing",
            "green_loss_rate": "2.15% per quarter",
            "hotspots": ["Başakşehir", "Esenyurt", "Beylikdüzü"]
        }
    }


# ============================================================================
# TILE API'LERİ (Tile Service Endpoints)
# ============================================================================

@app.get("/api/tiles/{year}/{quarter}/{index}/{z}/{x}/{y}.png")
def get_satellite_tile(
    year: int,
    quarter: int,
    index: str,
    z: int,
    x: int,
    y: int
):
    """
    Uydu görüntüsü tile'ı döndür.
    
    Google Drive'dan .npy tile dosyalarını okur, renklendirir ve PNG olarak servis eder.
    
    Kullanım: istanbul-detail.html sayfasında zaman serisi analizi için
    
    Path Parametreleri:
        year (int): Yıl (2018-2025)
        quarter (int): Çeyrek (1-4)
        index (str): Görselleştirme tipi
            - 'ndvi': Yeşil alan (Normalized Difference Vegetation Index)
            - 'ndwi': Su alanı (Normalized Difference Water Index)
            - 'ndbi': Beton/Yapı (Normalized Difference Built-up Index)
            - 'rgb': Doğal görünüm
        z (int): Zoom seviyesi (0-18)
        x (int): Tile X koordinatı
        y (int): Tile Y koordinatı
    
    Headers:
        Authorization: Bearer <token>
    
    Döner:
        StreamingResponse: PNG image (256x256 piksel)
    
    Örnek:
        GET /api/tiles/2025/4/ndvi/12/2048/1360.png
        → 2025 Q4, NDVI index, zoom 12, x=2048, y=1360
    
    Notlar:
        - Tile'lar Google Drive'dan indirilir ve local cache'lenir
        - Cache 24 saat boyunca geçerlidir
        - Dosya yoksa transparent PNG döner (404 değil)
    """
    return get_tile_response(year, quarter, index, z, x, y)


@app.get("/api/tiles/available")
def get_available_tiles():
    """
    Mevcut tile yıl/çeyrek kombinasyonlarını listele.
    
    Headers:
        Authorization: Bearer <token>
    
    Döner:
        dict: Mevcut veri setleri
    
    Örnek Response:
        {
            "datasets": [
                {"year": 2018, "quarter": 1, "label": "2018 Q1"},
                {"year": 2018, "quarter": 2, "label": "2018 Q2"},
                ...
                {"year": 2025, "quarter": 4, "label": "2025 Q4"}
            ],
            "count": 32
        }
    """
    # 2018 Q1'den 2025 Q4'e kadar tüm çeyrekler
    datasets = []
    for year in range(2018, 2026):
        for quarter in range(1, 5):
            datasets.append({
                "year": year,
                "quarter": quarter,
                "label": f"{year} Q{quarter}"
            })
    
    return {
        "datasets": datasets,
        "count": len(datasets)
    }

# =========================================================================
# TREND TILE (T+1) OVERLAY ENDPOINT (prediction_outputs_trend_tiles)
# =========================================================================

import io
import numpy as np
from PIL import Image
from fastapi.responses import StreamingResponse
import math

@app.get("/api/trend/tiles/{year}/{quarter}/{z}/{x}/{y}.png")
def get_trend_tile(year: int, quarter: int, z: int, x: int, y: int):
    """
    Trend tahmin maskesi tile'ı döndürür (t+1 Conv3D overlay).
    prediction_outputs_trend_tiles klasöründeki .npy dosyalarını renklendirip PNG olarak sunar.
    Sadece trend tile overlay için kullanılır.
    """
    # AOI
    LON_MIN, LAT_MIN = 28.62, 40.75
    LON_MAX, LAT_MAX = 29.56, 41.18
    n = 2 ** z
    lon_deg = (x + 0.5) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n)))
    lat_deg = math.degrees(lat_rad)
    if not (LON_MIN <= lon_deg <= LON_MAX and LAT_MIN <= lat_deg <= LAT_MAX):
        empty_img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        empty_img.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type='image/png')
    lon_norm = (lon_deg - LON_MIN) / (LON_MAX - LON_MIN)
    lat_norm = (lat_deg - LAT_MIN) / (LAT_MAX - LAT_MIN)
    row_index = int((1.0 - lat_norm) * 18)
    col_index = int(lon_norm * 40)
    row_index = max(0, min(17, row_index))
    col_index = max(0, min(39, col_index))
    patch_row = row_index * 256
    patch_col = col_index * 256
    filename = f"{year}_Q{quarter}_{patch_row:05d}_{patch_col:05d}_trend_tplus1.npy"
    patch_path = Path(__file__).parent / "data" / "prediction_outputs_trend_tiles" / filename
    if not patch_path.exists():
        empty_img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        empty_img.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type='image/png')
    try:
        arr = np.load(patch_path)
        mask = np.argmax(arr, axis=0).astype(np.uint8)
        palette = np.array([
            [180, 180, 180],  # gri
            [34, 139, 34],    # yeşil
            [30, 144, 255]    # su
        ], dtype=np.uint8)
        rgb = palette[mask]
        img = Image.fromarray(rgb, mode='RGB')
    except Exception as e:
        print(f"Trend tile görselleştirme hatası: {e}")
        empty_img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        empty_img.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type='image/png')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type='image/png',
        headers={
            'Cache-Control': 'public, max-age=86400',
            'Access-Control-Allow-Origin': '*',
        }
    )

# =========================================================================
# FOLIUM HARİTA ENDPOINTİ (Trend Prediction için PNG olarak)
# =========================================================================
from fastapi.responses import FileResponse
import tempfile
import folium
import base64
import re


# Folium harita endpointi: PNG tile'ları birleştirip folium haritasında overlay olarak gösterir (HTML döner)
from fastapi import Request, Query, Header

from fastapi.responses import HTMLResponse
from pathlib import Path
from PIL import Image
import base64
import io
import re

@app.get("/api/trend/folium_map/{year}/{quarter}/{horizon}")
def get_trend_folium_map(year: int, quarter: int, horizon: int):
    TILE_DIR = Path(__file__).parent / "data" / "prediction_outputs_trend_tiles"
    suffix = f"tplus{horizon}"
    tile_re = re.compile(rf"{year}_Q{quarter}_(\d{{5}})_(\d{{5}})_trend_{suffix}\.png")
    tiles = []
    for f in TILE_DIR.glob(f"{year}_Q{quarter}_*_trend_{suffix}.png"):
        m = tile_re.match(f.name)
        if m:
            row = int(m.group(1))
            col = int(m.group(2))
            tiles.append((row, col, f))
    if not tiles:
        raise HTTPException(404, f"Hiç PNG tile bulunamadı (horizon={horizon}).")
    tile_size = Image.open(tiles[0][2]).width
    max_row = max(r for r,_,_ in tiles)
    max_col = max(c for _,c,_ in tiles)
    min_row = min(r for r,_,_ in tiles)
    min_col = min(c for _,c,_ in tiles)
    mosaic_h = (max_row - min_row) + tile_size
    mosaic_w = (max_col - min_col) + tile_size
    mosaic = Image.new("RGBA", (mosaic_w, mosaic_h), (0,0,0,0))
    for row, col, f in tiles:
        img = Image.open(f).convert("RGBA")
        mosaic.paste(img, (col - min_col, row - min_row))
    buf = io.BytesIO()
    mosaic.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    url = f"data:image/png;base64,{data}"
    html_str = f"""
    <div style='width:100%;display:flex;justify-content:center;align-items:center;background:#f3f4f6;'>
        <img src='{url}' alt='Tahmin Mozaik' style='max-width:100%;height:auto;border-radius:16px;box-shadow:0 2px 16px rgba(0,0,0,0.12);margin:32px 0;'/>
    </div>
    """
    return HTMLResponse(content=html_str, media_type="text/html")

# =========================================================================
# PNG TILE MOZAİK VE FOLIUM HARİTA ENDPOINTİ
# =========================================================================
from PIL import Image


# PNG mozaik indirme endpointi kaldırıldı


# Folium HTML indirme endpointi
@app.get("/api/trend/folium_mosaic/{year}/{quarter}/{horizon}/html")
def download_trend_folium_html(year: int, quarter: int, horizon: int, current_user: User = Depends(get_current_user)):
    # Aynı mozaik PNG'yi oluştur ve folium haritası üret
    TILE_DIR = Path(__file__).parent / "data" / "prediction_outputs_trend_tiles"
    suffix = f"tplus{horizon}"
    tile_files = list(TILE_DIR.glob(f"{year}_Q{quarter}_*_trend_{suffix}.png"))
    if not tile_files:
        raise HTTPException(404, f"Hiç PNG tile bulunamadı (horizon={horizon}).")
    import re
    tile_re = re.compile(rf"{year}_Q{quarter}_(\\d{{5}})_(\\d{{5}})_trend_{suffix}.png")
    coords_files = [(tile_re.match(f.name), f) for f in tile_files]
    coords_files = [(m, f) for m, f in coords_files if m]
    if not coords_files:
        raise HTTPException(500, f"PNG tile dosya isimleri beklenen formatta değil (horizon={horizon})")
    tile_size = 256
    row_vals = [int(m.group(1)) for m, _ in coords_files]
    col_vals = [int(m.group(2)) for m, _ in coords_files]
    min_row, max_row = min(row_vals), max(row_vals)
    min_col, max_col = min(col_vals), max(col_vals)
    mosaic_w = (max_col - min_col) + tile_size
    mosaic_h = (max_row - min_row) + tile_size
    mosaic = Image.new('RGBA', (mosaic_w, mosaic_h))
    for m, f in coords_files:
        row = int(m.group(1))
        col = int(m.group(2))
        img = Image.open(f).convert('RGBA')
        mosaic.paste(img, (col - min_col, row - min_row))
    import tempfile
    tmp_png = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    mosaic.save(tmp_png.name)
    tmp_png.flush()
    import folium, base64
    lat_min, lon_min = 40.8, 28.6
    lat_max, lon_max = 41.4, 29.4
    bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    with open(tmp_png.name, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    url = f"data:image/png;base64,{data}"
    folium.raster_layers.ImageOverlay(
        image=url,
        bounds=bounds,
        opacity=0.75,
        name=f"{year}_Q{quarter}_mosaic_{suffix}.png"
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    tmp_html = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
    m.save(tmp_html.name)
    tmp_html.flush()
    return FileResponse(tmp_html.name, media_type="text/html", filename=f"trend_folium_{year}_Q{quarter}_{suffix}.html")

# ============================================================================
# FRONTEND SUNUCU (Static Files Server)
# ============================================================================

# Frontend dosyalarını statik olarak sun
# NOT: Bu mount en son olmalı, yoksa API route'ları çalışmaz!
try:
    app.mount(
        "/",
        StaticFiles(directory="../frontend", html=True),
        name="frontend"
    )
    print("✅ Frontend dosyaları hazır (../frontend)")
except RuntimeError:
    # Frontend klasörü yoksa hata verme (geliştirme aşamasında olabilir)
    print("⚠️  Frontend klasörü bulunamadı")
    pass
