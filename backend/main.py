""" 
Backend API part of the Grey vs Green Atlas application.
"""

from datetime import datetime, timedelta
from typing import Optional
import os
import json
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlmodel import SQLModel, Field, Session, create_engine, select

import io
import numpy as np
from PIL import Image
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
import math


import tempfile
import folium
import base64
import re


# Folium harita endpointi: PNG tile'ları birleştirip folium haritasında overlay olarak gösterir (HTML döner)
from fastapi import Request, Query, Header


# Import tile service
from tile_service import get_tile_response

# Import model service
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
    print(f"⚠️  The model service could not be loaded: {e}")
    MODEL_SERVICE_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

# JWT Token settings
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-please-very-secret")
ALGORITHM = "HS256" # Used SHA-256 for encoding
ACCESS_TOKEN_EXPIRE_MINUTES = 12 * 60  # 12 hours

# Database connection
DB_URL = os.getenv("DB_URL", "sqlite:///./db.sqlite")

# The directory containing data files
DATA_DIR = Path(__file__).parent / "data"


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_istanbul_data():
    """
    Load data for Istanbul from a JSON file.(This file should be prepared in advance)
    
    File: backend/data/istanbul_data.json
    Content: Historical data, predictions, regions, districts, summary statistics
    
    Returns:
        dict: Istanbul data or None (if file not found)
    """
    data_file = DATA_DIR / "istanbul_data.json"
    
    if not data_file.exists():
        print(f"⚠️  WARNING: {data_file} not found!")
        return None
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✅ Istanbul data loaded: {len(data.get('districts', []))} districts")
            return data
    except Exception as e:
        print(f"❌ ERROR: Failed to load Istanbul data: {e}")
        return None


# Load data at application startup and cache
ISTANBUL_DATA = load_istanbul_data()


# ============================================================================
# DATABASE SETUP
# ============================================================================

# Create SQLite database connection
engine = create_engine(DB_URL, connect_args={"check_same_thread": False}) # Used check_same_thread to allow multi-threading

# Use bcrypt for password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(SQLModel, table=True):
    """
    User database model.
    
    Attributes:
        id: Auto-incrementing unique identifier
        email: User email address (unique)
        password_hash: Password hashed with bcrypt
        full_name: User's full name (optional)
        is_active: Is the account active? (default: True)
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    email: EmailStr
    password_hash: str
    full_name: Optional[str] = None
    is_active: bool = True


def create_db_and_tables():
    """Create database tables based on SQLModel models."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """
    Create and manage database session.
    Used for dependency injection.
    """
    with Session(engine) as session:
        yield session


# ============================================================================
# PYDANTIC SCHEMAS (Request/Response Models)
# ============================================================================

class UserCreate(BaseModel):
    """Schema for incoming user registration data."""
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserOut(BaseModel):
    """Schema for user data returned by the API (excluding password)."""
    id: int
    email: EmailStr
    full_name: Optional[str]


class Token(BaseModel):
    """JWT token response schema."""
    access_token: str
    token_type: str = "bearer"


# ============================================================================
# Authentication Functions
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password.
    
    Args:
        plain_password: User's plain text password
        hashed_password: Hashed password from the database
    
    Returns:
        bool: True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Data to include in the token (usually {"sub": email})
        minutes: Token expiration time in minutes
    
    Returns:
        str: JWT token string
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# OAuth2 scheme for token extraction from requests
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session)
) -> User:
    """
    Get current user from JWT token.
    Used as a dependency in protected endpoints.
    
    Args:
        token: JWT token from the Authorization header
        session: Database session
    
    Returns:
        User: Active user object
    
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication failed",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the token
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
# FASTAPI APPLICATION
# ============================================================================


# FastAPI application and CORS settings
app = FastAPI(
    title="Grey vs Green Atlas API",
    description="İstanbul yeşil alan takip ve tahmin sistemi",
    version="1.0.0"
)
# Allow CORS from all origins (for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables only when the application starts
@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    print("Backend started!!!")
    # Model loading will be triggered by model_service on the first prediction request
    # If the Softmaps folder is missing or model files are not present, the relevant endpoint will return an error message


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/register", response_model=UserOut)
def register(payload: UserCreate, session: Session = Depends(get_session)):
    """
    New user registration.
    
    Body Parameters:
        email (str): User email address
        password (str): Password (at least 6 characters)
        full_name (str, optional): Full name
    
    Returns:
        UserOut: Created user information (excluding password)
    
    Errors:
        400: Email already registered
    """
    # Check if email is already registered
    existing_user = session.exec(
        select(User).where(User.email == payload.email)
    ).first()
    
    if existing_user:
        raise HTTPException(400, "Email already registered")
    
    # Create new user
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
    User login and JWT token generation.
    
    Form Parameters:
        username (str): Email address
        password (str): Password
    
    Returns:
        Token: JWT access token and token type
    
    Errors:
        400: Invalid email or password
    """
    # Find user (OAuth2 form's username field is used for email)
    user = session.exec(
        select(User).where(User.email == form.username)
    ).first()
    
    # If user does not exist or password is incorrect, raise error
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(400, "Invalid email or password")
    
    # Create JWT token
    access_token = create_access_token({"sub": user.email})
    
    return Token(access_token=access_token)


@app.get("/me", response_model=UserOut)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current logged-in user information.
    
    Headers:
        Authorization: Bearer <token>
    
    Returns:
        UserOut: User information
    
    Errors:
        401: Invalid or missing token
    """
    return UserOut(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name
    )


# ============================================================================
# City Data Endpoints
# ============================================================================

# Used on dashboard main page for Istanbul summary
@app.get("/api/city/istanbul")
def get_istanbul_summary(current_user: User = Depends(get_current_user)):
    """
    Returns summary statistics for Istanbul.
    
    Headers:
        Authorization: Bearer <token>
    
    Returns:
        dict: Now, +6 months, +12 months predictions
    
    Example Response:
        {
            "title": "Istanbul — Green/Grey Summary",
            "now": {"green": 31.2, "grey": 61.8, "water": 7.0},
            "+6m": {...},
            "+12m": {...},
            "note": "Data sources: İBB, TÜİK"
        }
    """
    if not ISTANBUL_DATA:
        raise HTTPException(500, "Istanbul data could not be loaded")
    
    return {
        "title": "Istanbul — Green/Grey Summary",
        "now": ISTANBUL_DATA["predictions"][0],
        "+6m": ISTANBUL_DATA["predictions"][1],
        "+12m": ISTANBUL_DATA["predictions"][2],
        "note": f"Data sources: {', '.join(ISTANBUL_DATA['metadata']['sources'][:2])}"
    }


# ============================================================================
# Map Endpoints
# ============================================================================

# Shows major cities across Turkey
@app.get("/api/map/turkey")
def get_turkey_map_data(current_user: User = Depends(get_current_user)):
    """
    Returns map data for major cities across Turkey.
    Point features in GeoJSON format.
    
    Usage: For the Turkey map on the map.html page
    Status: Only Istanbul is active, other cities are "Coming Soon"
    
    Headers:
        Authorization: Bearer <token>
    
    Returns:
        dict: GeoJSON FeatureCollection format city data
    
    Example Response:
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
    # Major cities in Turkey (TÜİK data) - Istanbul is active and special
    cities = [
        # Largest city - Active and special emphasis
        {"name": "Istanbul", "lat": 41.0082, "lng": 28.9784, "active": True, "population": 15840900},
        
        # Other major cities
        {"name": "Ankara", "lat": 39.9334, "lng": 32.8597, "active": False, "population": 5663322},
        {"name": "Izmir", "lat": 38.4237, "lng": 27.1428, "active": False, "population": 4425789},
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

# Shows Istanbul districts information when zoomed in
@app.get("/api/map/istanbul/{timeframe}")
def get_istanbul_map_data(
    timeframe: str,
    current_user: User = Depends(get_current_user)
):
    """
    Returns green/grey area data for Istanbul districts.
    In GeoJSON format for map visualization.
    
    Usage: For Istanbul zoom view on the map.html page
    
    Path Parameter:
        timeframe (str): Timeframe
            - 'now': Current state
            - '6m': Estimated 6 months later
            - '12m': Estimated 12 months later
    
    Headers:
        Authorization: Bearer <token>
    
    Returns:
        dict: GeoJSON Point features of districts
    
    Errors:
        400: Invalid timeframe
        500: Istanbul data not found
    
    Example Response:
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
    # Timeframe validity check
    if timeframe not in ["now", "6m", "12m"]:
        raise HTTPException(
            400,
            "Invalid timeframe. Available: now, 6m, 12m"
        )
    
    # Check if Istanbul data is loaded
    if not ISTANBUL_DATA or "districts" not in ISTANBUL_DATA:
        raise HTTPException(500, "Istanbul district data not found")
    
    # Prepare district data according to timeframe
    districts_with_predictions = []
    for district in ISTANBUL_DATA["districts"]:
        # Keep water surface constant (7%)
        water_percentage = 7
        
        # Calculate grey area: 100% - green - water
        now_grey = 100 - district["now_green"] - water_percentage
        future_grey = 100 - district["future_green"] - water_percentage
        
        districts_with_predictions.append({
            "name": district["name"],
            "lat": district["lat"],
            "lng": district["lng"],
            # Separate data for each timeframe
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
    
    # Create GeoJSON Features
    features = []
    for district in districts_with_predictions:
        data = district[timeframe]  # Get data for requested timeframe
        
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
# Istanbul Detailed Analysis Endpoints
# ============================================================================

# Gets comprehensive detailed analysis data for Istanbul
@app.get("/api/istanbul/detailed")
def get_istanbul_detailed_analysis(current_user: User = Depends(get_current_user)):
    """
    Returns comprehensive detailed analysis data for Istanbul.
    
    Usage: For visualizations on the istanbul-detail.html page
    
    Content:
        - Regional grid data (9 regions)
        - Historical trends (2019-2024)
        - Future predictions (now, +6 months, +12 months)
        - District-based changes (39 districts)
        - Summary statistics (population, area, green space per capita)
        - Metadata (data sources, last update)
    
    Headers:
        Authorization: Bearer <token>
    
    Returns:
        dict: Comprehensive Istanbul analysis data
    
    Errors:
        500: Istanbul data not loaded
    
    Example Response:
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
    # Check if Istanbul data is loaded
    if not ISTANBUL_DATA:
        raise HTTPException(500, "Istanbul detailed data not loaded")
    
    # Convert regions to grid format
    # Each region will be shown as a circle on the map
    grid_data = []
    for region in ISTANBUL_DATA["regions"]:
        # Create id from region name (lowercase, no spaces)
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
            "lat": region["center"][0],  # Latitude
            "lng": region["center"][1],  # Longitude
            "now": region["now"],        # Current state
            "6m": region["6m"],          # 6 months later
            "12m": region["12m"]         # 12 months later
        })
    
    # Return all data in a single response
    return {
        "city": "Istanbul",
        "grid": grid_data,                           # Regional grid (9 regions)
        "historical": ISTANBUL_DATA["historical"],   # Historical data 2019-2024
        "predictions": ISTANBUL_DATA["predictions"], # Predictions for 3 timeframes
        "district_changes": ISTANBUL_DATA["districts"], # 39 district details
        "summary": ISTANBUL_DATA["summary"],         # Summary statistics
        "metadata": ISTANBUL_DATA["metadata"],       # Data source information
        "timestamp": datetime.utcnow().isoformat()   # API call time
    }


# ============================================================================
# Tile Service Endpoints for Istanbul Area
# ============================================================================

## Serves satellite image tiles (NDVI, NDWI, NDBI, RGB) for Istanbul area
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
    Serves satellite image tiles for Istanbul area.
    
    Reads .npy tile files from Google Drive, colors them, and serves as PNG.
    
    Usage: for time series analysis on istanbul-detail.html page
    
    Path Parameters:
        year (int): Year (2018-2025)
        quarter (int): Quarter (1-4)
        index (str): Visualization type
            - 'ndvi': Green area (Normalized Difference Vegetation Index)
            - 'ndwi': Water area (Normalized Difference Water Index)
            - 'ndbi': Built-up area (Normalized Difference Built-up Index)
            - 'rgb': Natural view
        z (int): Zoom level (0-18)
        x (int): Tile X coordinate
        y (int): Tile Y coordinate
    
    Headers:
        Authorization: Bearer <token>
    
    Returns:
        StreamingResponse: PNG image (256x256 pixels)
    
    Example:
        GET /api/tiles/2025/4/ndvi/12/2048/1360.png
        → 2025 Q4, NDVI index, zoom 12, x=2048, y=1360
    
    Notes:
        - Tiles are downloaded from Google Drive and locally cached
        - Cache is valid for 24 hours
        - If file is missing, returns transparent PNG (not 404)
    """
    return get_tile_response(year, quarter, index, z, x, y)

## Lists available tile year/quarter combinations
@app.get("/api/tiles/available")
def get_available_tiles():
    """
    Lists available tile year/quarter combinations.
    
    Headers:
        Authorization: Bearer <token>
    
    Returns:
        dict: Available datasets
    
    Example Response:
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
    # All quarters from 2018 Q1 to 2025 Q4
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
# TREND TILE OVERLAY ENDPOINT (prediction_outputs_trend_tiles)
# =========================================================================

from pathlib import Path
import math
import io
import numpy as np
from PIL import Image
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

# ...

@app.get("/api/trend/tiles/{year}/{quarter}/{horizon}/{z}/{x}/{y}.png")
def get_trend_tile(year: int, quarter: int, horizon: int, z: int, x: int, y: int):
    """
    Trend tahmin tile'ı döner (t+1, t+4, t+8).

    - data/prediction_outputs_trend_tiles içindeki
      2026_Q1_row_col_trend_tplusX.npy / .png dosyalarını kullanır.
    - .png varsa direkt onu döner, yoksa .npy maskesinden RGBA üretir.
    """

    # Sadece 1,4,8 destekli
    horizon_suffix = {1: "tplus1", 4: "tplus4", 8: "tplus8"}
    if horizon not in horizon_suffix:
        raise HTTPException(status_code=404, detail="Unsupported horizon")

    suffix = horizon_suffix[horizon]

    # -------- 1) Tile -> Lat/Lon, AOI kontrolü --------
    # İstanbul AOI (senin kullandığın bbox)
    LON_MIN, LAT_MIN = 28.62, 40.75
    LON_MAX, LAT_MAX = 29.56, 41.18

    n = 2 ** z
    lon_deg = (x + 0.5) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n)))
    lat_deg = math.degrees(lat_rad)

    # AOI dışındaysa tamamen şeffaf PNG
    if not (LON_MIN <= lon_deg <= LON_MAX and LAT_MIN <= lat_deg <= LAT_MAX):
        empty = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        empty.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    # -------- 2) Lat/Lon -> patch row/col (tile index) --------
    # Buradaki grid yapısı senin softmax/tile üretim koduna göre
    # (örnek: 18x40 patch, her patch 256x256)
    lon_norm = (lon_deg - LON_MIN) / (LON_MAX - LON_MIN)
    lat_norm = (lat_deg - LAT_MIN) / (LAT_MAX - LAT_MIN)

    row_index = int((1.0 - lat_norm) * 18)
    col_index = int(lon_norm * 40)
    row_index = max(0, min(17, row_index))
    col_index = max(0, min(39, col_index))

    patch_row = row_index * 256
    patch_col = col_index * 256

    # -------- 3) Dosya yolları --------
    out_dir = Path(__file__).parent / "data" / "prediction_outputs_trend_tiles"
    png_path = out_dir / f"{year}_Q{quarter}_{patch_row:05d}_{patch_col:05d}_trend_{suffix}.png"
    npy_path = out_dir / f"{year}_Q{quarter}_{patch_row:05d}_{patch_col:05d}_trend_{suffix}.npy"

    # Önce hazır PNG varsa onu döneriz (trend_prediction_generate.py’nin ürettiği)
    if png_path.exists():
        buf = io.BytesIO(png_path.read_bytes())
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=86400",
                "Access-Control-Allow-Origin": "*",
            },
        )

    # PNG yoksa NPY'den üret
    if not npy_path.exists():
        # Hiç veri yoksa şeffaf
        empty = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        empty.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    try:
        pred_mask = np.load(npy_path).astype(np.uint8)  # (H, W), 0..3

        if pred_mask.ndim != 2:
            raise ValueError(f"Beklenmeyen mask shape: {pred_mask.shape}")

        h, w = pred_mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # trend_prediction_generate.py ile aynı palette
        palette = {
            0: (0, 0, 0, 0),        # background -> tam şeffaf
            1: (0, 200, 0, 160),    # yeşil
            2: (130, 130, 130, 160),# beton/gri
            3: (0, 180, 255, 160),  # su
        }

        for k, col in palette.items():
            rgba[pred_mask == k] = col

        img = Image.fromarray(rgba, mode="RGBA")

    except Exception as e:
        print(f"Trend tile görselleştirme hatası: {e}")
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
        },
    )


# =========================================================================
# Folium Map Endpoint for Trend Tiles (This is used for quick preview)
# =========================================================================


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



# ============================================================================
# FRONTEND SERVER (Static Files Server)
# ============================================================================

# Serve frontend files statically
# NOTE: This mount should be last, otherwise API routes won't work!
try:
    app.mount(
        "/",
        StaticFiles(directory="../frontend", html=True),
        name="frontend"
    )
    print("Frontend files ready (../frontend)")
except RuntimeError:
    # Don't raise error if frontend folder is missing (might be during development)
    print("⚠️  Frontend folder not found")
    pass
