// ============================================================================
// Authentication
// ============================================================================

// JWT token check - redirect to login if not present
const token = localStorage.getItem('token');
if (!token) {
  location.href = '/';
}

/**
 * Fetch user info and display in topbar
 * API: GET /me
 */
async function whoami() {
  try {
    const response = await fetch('/me', {
      headers: { Authorization: `Bearer ${token}` }
    });
    
    if (!response.ok) {
      // Redirect to login if token is invalid
      localStorage.removeItem('token');
      location.href = '/';
      return;
    }
    
    const user = await response.json();
    document.getElementById('who').textContent = user.email;
  } catch (error) {
    console.error('Kullanıcı bilgisi alınamadı:', error);
    location.href = '/';
  }
}

whoami();

/**
 * Logout button - remove token and redirect to login
 */
document.getElementById('logout').onclick = () => {
  localStorage.removeItem('token');
  location.href = '/';
};

/**
 * Sidebar toggle functionality
 */
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');
let isCollapsed = true; // Start collapsed on map page

// Initialize sidebar as collapsed
sidebar.classList.add('collapsed');
sidebarToggle.textContent = '☰';

sidebarToggle.onclick = () => {
  isCollapsed = !isCollapsed;
  sidebar.classList.toggle('collapsed', isCollapsed);
  sidebarToggle.textContent = isCollapsed ? '☰' : '✕';
};

/**
 * Sidebar resize functionality
 */
const resizer = document.getElementById('sidebar-resizer');
let isResizing = false;
let startX = 0;
let startWidth = 0;

resizer.addEventListener('mousedown', (e) => {
  isResizing = true;
  startX = e.clientX;
  startWidth = sidebar.offsetWidth;
  document.body.style.cursor = 'col-resize';
  document.body.style.userSelect = 'none';
});

document.addEventListener('mousemove', (e) => {
  if (!isResizing) return;
  const diff = e.clientX - startX;
  const newWidth = Math.max(250, Math.min(600, startWidth + diff));
  sidebar.style.width = newWidth + 'px';
});

document.addEventListener('mouseup', () => {
  if (!isResizing) return;
  isResizing = false;
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
});

/**
 * Right panel open/close functionality
 */
const rightPanel = document.getElementById('right-panel');
const panelCloseBtn = document.getElementById('panel-close');
let panelToggleBtn = null;

// Make panel draggable
let isDraggingPanel = false;
let dragOffsetX = 0;
let dragOffsetY = 0;

// Drag functionality
const panelHeader = rightPanel.querySelector('.panel-header');
panelHeader.style.cursor = 'move';
panelHeader.addEventListener('mousedown', (e) => {
  if (e.target.closest('.panel-close')) return; // Don't drag when clicking close button
  
  isDraggingPanel = true;
  
  // Calculate offset from mouse position to panel's current position
  const rect = rightPanel.getBoundingClientRect();
  dragOffsetX = e.clientX - rect.left;
  dragOffsetY = e.clientY - rect.top;
  
  document.body.style.userSelect = 'none';
  rightPanel.style.transition = 'none'; // Disable transition during drag
});

document.addEventListener('mousemove', (e) => {
  if (!isDraggingPanel) return;
  
  // Calculate new position
  let newX = e.clientX - dragOffsetX;
  let newY = e.clientY - dragOffsetY;
  
  // Constrain to viewport
  const maxX = window.innerWidth - rightPanel.offsetWidth;
  const maxY = window.innerHeight - rightPanel.offsetHeight;
  
  newX = Math.max(0, Math.min(newX, maxX));
  newY = Math.max(0, Math.min(newY, maxY));
  
  // Apply new position
  rightPanel.style.right = 'auto';
  rightPanel.style.bottom = 'auto';
  rightPanel.style.left = newX + 'px';
  rightPanel.style.top = newY + 'px';
});

document.addEventListener('mouseup', () => {
  if (!isDraggingPanel) return;
  isDraggingPanel = false;
  document.body.style.userSelect = '';
  rightPanel.style.transition = ''; // Re-enable transition
});

// Close panel
panelCloseBtn.addEventListener('click', () => {
  rightPanel.style.transform = 'translateX(420px)';
  rightPanel.style.opacity = '0';
  setTimeout(() => {
    rightPanel.style.display = 'none';
    // Create toggle button to reopen panel
    if (!panelToggleBtn) {
      panelToggleBtn = document.createElement('button');
      panelToggleBtn.id = 'panel-toggle';
      panelToggleBtn.innerHTML = '⚙️';
      panelToggleBtn.style.cssText = `
        position: absolute;
        top: 20px;
        right: 20px;
        width: 50px;
        height: 50px;
        border-radius: 12px;
        border: 2px solid rgba(59, 130, 246, 0.3);
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.85));
        color: white;
        font-size: 24px;
        cursor: pointer;
        z-index: 1000;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
      `;
      panelToggleBtn.addEventListener('mouseenter', () => {
        panelToggleBtn.style.transform = 'scale(1.1)';
        panelToggleBtn.style.borderColor = 'rgba(59, 130, 246, 0.5)';
      });
      panelToggleBtn.addEventListener('mouseleave', () => {
        panelToggleBtn.style.transform = 'scale(1)';
        panelToggleBtn.style.borderColor = 'rgba(59, 130, 246, 0.3)';
      });
      panelToggleBtn.addEventListener('click', () => {
        rightPanel.style.display = 'flex';
        rightPanel.style.right = '20px';
        rightPanel.style.top = '20px';
        rightPanel.style.left = 'auto';
        rightPanel.style.bottom = '20px';
        setTimeout(() => {
          rightPanel.style.transform = 'translateX(0)';
          rightPanel.style.opacity = '1';
        }, 10);
        panelToggleBtn.remove();
        panelToggleBtn = null;
      });
      document.body.appendChild(panelToggleBtn);
    }
  }, 300);
});


// ============================================================================
// MAP INITIALIZATION
// ============================================================================

/**
 * Create Leaflet map
 * Initial view: Turkey center (39.0, 35.0) zoom 7.4
 */
const map = L.map('map', {
  zoomControl: true,
  scrollWheelZoom: true,
  doubleClickZoom: false,  // Çift tıklama ile ani zoom kapalı
  dragging: true,
  zoomSnap: 0.10,  // Zoom adımlarını 0.10'a düşürür (çok yumuşak)
  zoomDelta: 0.5,  // Butonlar için zoom değişimi
  wheelPxPerZoomLevel: 120  // Mouse wheel hassasiyeti (yavaş zoom)
}).setView([39.0, 35.0], 7.4);

/**
 * Add OpenStreetMap tile layer (free map layer)
 * Tiles: https://tile.openstreetmap.org/{z}/{x}/{y}.png
 * {s}: server (a,b,c), {z}: zoom, {x}/{y}: coordinates
 */
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors',
  maxZoom: 18  // Maksimum zoom seviyesii
}).addTo(map);


// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

let cityMarkers = [];           // Turkey city markers (10 cities)
let detailMarkers = [];         // Istanbul district markers (15 districts)
let currentTimeframe = 'now';   // Active timeframe: 'now', '6m', '12m'
let currentViewMode = 'markers'; // Visualization mode: 'markers' or 'heatmap'
let heatmapLayer = null;        // Leaflet heatmap layer object
let currentData = null;         // Last loaded data (for re-render)
let showingDetails = false;     // Are Istanbul details being shown?
let satelliteLayer = null;      // Satellite imagery layer (NDVI/NDWI/NDBI/RGB)
let currentIndex = 'ndvi';      // Active index: 'ndvi', 'ndwi', 'ndbi', 'rgb'


// ============================================================================
// LOAD TURKEY CITIES
// ============================================================================

/**
 * Add Turkey's 10 largest cities to the map
 * API: GET /api/map/turkey
 * 
 * Workflow:
 *   1. Get city data from backend in GeoJSON format
 *   2. Create circleMarker for each city
 *   3. Istanbul is green (active), others are grey (inactive)
 *   4. Clicking active cities opens detail view
 *   5. Popup shows city name, population, and status
 */
async function loadTurkeyCities() {
  try {
    // Fetch city data from backend API
    const response = await fetch('/api/map/turkey', {
      headers: { Authorization: `Bearer ${token}` }
    });
    
    if (!response.ok) {
      throw new Error('Şehir verileri yüklenemedi');
    }
    
    const data = await response.json();
    
    data.features.forEach(feature => {
      const props = feature.properties;
      const coords = feature.geometry.coordinates;
      
      // Create marker
      const isActive = props.active;
      const marker = L.circleMarker([coords[1], coords[0]], {
        radius: isActive ? 20 : 15,
        fillColor: isActive ? '#10b981' : '#94a3b8',
        color: 'white',
        weight: 3,
        opacity: 1,
        fillOpacity: isActive ? 0.9 : 0.5
      }).addTo(map);
      
      // Popup content
      const popupContent = `
        <div style="padding:12px;min-width:200px">
          <div style="font-weight:900;font-size:18px;margin-bottom:8px;color:#0284c7">${props.name}</div>
          <div style="font-size:13px;color:#64748b;margin-bottom:12px">
            👥 Nüfus: <strong>${props.population.toLocaleString()}</strong>
          </div>
          ${isActive ? 
            '<button onclick="location.href=\'/istanbul-detail.html\'" style="width:100%;padding:12px;background:linear-gradient(135deg,#0284c7,#0891b2);color:white;border:none;border-radius:10px;font-weight:700;cursor:pointer;font-size:14px">🔬 Detaylı Analiz</button>' : 
            '<div style="text-align:center;color:#94a3b8;font-size:12px;font-weight:600">Yakında Aktif</div>'
          }
        </div>
      `;
      
      marker.bindPopup(popupContent);
      
      // Click handler for active cities
      if(isActive) {
        marker.on('click', () => {
          // Show Istanbul details on map
          showIstanbulDetails();
        });
      }
      
      cityMarkers.push(marker);
    });
    
    document.getElementById('loading').style.display = 'none';
  } catch(err) {
    console.error('Şehirler yüklenirken hata oluştu:', err);
    alert('Şehirler yüklenirken hata oluştu');
  }
}

// Show Istanbul details on map
async function showIstanbulDetails() {
  showingDetails = true;
  map.setView([41.0082, 28.9784], 11);
  
  loadMapData(currentTimeframe);
}

// Add satellite imagery layer - Real satellite imagery
function addSatelliteLayer() {
  // Remove old layer
  if(satelliteLayer) {
    map.removeLayer(satelliteLayer);
  }
  
  // Esri World Imagery - free, real satellite imagery
  satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
    maxZoom: 19,
    minZoom: 1
  }).addTo(map);
  
  console.log('✅ Satellite imagery layer added');
}

// Load Istanbul district data
async function loadMapData(timeframe, animate = true) {
  try {
    const r = await fetch(`/api/map/istanbul/${timeframe}`, {
      headers: {Authorization: `Bearer ${token}`}
    });
    
    if(!r.ok) throw new Error('Failed to load map data');
    
    const data = await r.json();
    currentData = data;
    
    // Clear existing layers
    detailMarkers.forEach(m => map.removeLayer(m));
    detailMarkers = [];
    if(heatmapLayer) {
      map.removeLayer(heatmapLayer);
      heatmapLayer = null;
    }
    
    // Hide city markers when showing details
    cityMarkers.forEach(m => m.setStyle({opacity: 0.2, fillOpacity: 0.1}));
    
    // Calculate averages
    let totalGreen = 0, totalGrey = 0, totalWater = 0;
    const count = data.features.length;
    
    // Process data and calculate totals
    data.features.forEach(feature => {
      const props = feature.properties;
      totalGreen += props.green;
      totalGrey += props.grey;
      totalWater += props.water;
    });
    
    // Render based on view mode
    if(currentViewMode === 'markers') {
      renderMarkers(data, animate);
    } else {
      renderHeatmap(data);
    }
    
    // Update averages
    document.getElementById('avg-green').textContent = Math.round(totalGreen / count) + '%';
    document.getElementById('avg-grey').textContent = Math.round(totalGrey / count) + '%';
    document.getElementById('avg-water').textContent = Math.round(totalWater / count) + '%';
    
    // Hide loading
    document.getElementById('loading').style.display = 'none';
    
  } catch(err) {
    console.error('Harita yüklenirken hata oluştu:', err);
    alert('Harita yüklenirken hata oluştu');
  }
}

// Render markers
function renderMarkers(data, animate = true) {
  data.features.forEach((feature, index) => {
    const props = feature.properties;
    const coords = feature.geometry.coordinates;
    
    // Create custom icon based on green percentage
    const greenPercent = props.green;
    let markerColor = greenPercent > 40 ? '#10b981' : greenPercent > 25 ? '#f59e0b' : '#ef4444';
    
    const marker = L.circleMarker([coords[1], coords[0]], {
      radius: animate ? 0 : 15,
      fillColor: markerColor,
      color: 'white',
      weight: 3,
      opacity: animate ? 0 : 1,
      fillOpacity: animate ? 0 : 0.8
    }).addTo(map);
    
    // Animate marker appearance
    if(animate) {
      setTimeout(() => {
        marker.setStyle({
          radius: 15,
          opacity: 1,
          fillOpacity: 0.8
        });
      }, index * 50);
    }
    
    // Create popup
    const popupContent = `
      <div class="popup-title">${props.name}</div>
      <div class="popup-stats">
        <div class="popup-stat">
          <span class="popup-stat-label">🌳 Yeşil Alan</span>
          <span class="popup-stat-value">${props.green}%</span>
        </div>
        <div class="popup-stat">
          <span class="popup-stat-label">🏢 Beton</span>
          <span class="popup-stat-value">${props.grey}%</span>
        </div>
        <div class="popup-stat">
          <span class="popup-stat-label">💧 Su</span>
          <span class="popup-stat-value">${props.water}%</span>
        </div>
      </div>
    `;
    
    marker.bindPopup(popupContent);
    detailMarkers.push(marker);
  });
}

// Render heatmap
function renderHeatmap(data) {
  const heatData = [];
  
  data.features.forEach(feature => {
    const props = feature.properties;
    const coords = feature.geometry.coordinates;
    
    // Use inverse of green (more grey = more heat)
    const intensity = (100 - props.green) / 100;
    heatData.push([coords[1], coords[0], intensity]);
  });
  
  heatmapLayer = L.heatLayer(heatData, {
    radius: 50,
    blur: 40,
    maxZoom: 11,
    max: 1.0,
    gradient: {
      0.0: '#10b981',
      0.3: '#22c55e',
      0.5: '#f59e0b',
      0.7: '#ef4444',
      1.0: '#dc2626'
    }
  }).addTo(map);
}

// View mode buttons (marker/heatmap)
document.querySelectorAll('.view-btn[data-mode]').forEach(btn => {
  btn.addEventListener('click', () => {
    // Update active state
    document.querySelectorAll('.view-btn[data-mode]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    
    // Update mode and reload
    currentViewMode = btn.dataset.mode;
    if(currentData) {
      // Clear current view
      detailMarkers.forEach(m => map.removeLayer(m));
      detailMarkers = [];
      if(heatmapLayer) {
        map.removeLayer(heatmapLayer);
        heatmapLayer = null;
      }
      
      // Render new view
      if(currentViewMode === 'markers') {
        renderMarkers(currentData, true);
      } else {
        renderHeatmap(currentData);
      }
    }
  });
});

// Satellite toggle button
let satelliteEnabled = false;
const satelliteToggle = document.getElementById('satellite-toggle');

satelliteToggle.addEventListener('click', () => {
  satelliteEnabled = !satelliteEnabled;
  
  if (satelliteEnabled) {
    satelliteToggle.innerHTML = '<span>�️ Uydu Görünümü (Açık)</span>';
    satelliteToggle.classList.add('active');
    
    // Add satellite layer (only if showing Istanbul details)
    addSatelliteLayer();
  } else {
    satelliteToggle.innerHTML = '<span>�️ Normal Harita</span>';
    satelliteToggle.classList.remove('active');
    
    // Remove satellite layer
    if (satelliteLayer) {
      map.removeLayer(satelliteLayer);
      satelliteLayer = null;
    }
  }
});

// Timeframe buttons
document.querySelectorAll('.timeframe-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    // Update active state
    document.querySelectorAll('.timeframe-btn').forEach(b => {
      b.classList.remove('active');
      b.querySelector('.icon').textContent = '';
    });
    btn.classList.add('active');
    btn.querySelector('.icon').textContent = '✓';
    
    // Load new data
    currentTimeframe = btn.dataset.time;
    
    // Reload data
    if(showingDetails) {
      loadMapData(currentTimeframe);
    }
  });
});

// Back to Turkey button
const backControl = L.control({position: 'topleft'});
backControl.onAdd = function() {
  const div = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
  div.innerHTML = '<button id="back-to-turkey" style="display:none;padding:12px 20px;background:white;border:none;color:#0284c7;border-radius:8px;font-weight:700;cursor:pointer;font-size:14px;box-shadow:0 2px 8px rgba(0,0,0,0.15)">← Türkiye Haritası</button>';
  return div;
};
backControl.addTo(map);

document.addEventListener('click', (e) => {
  if(e.target && e.target.id === 'back-to-turkey') {
    showingDetails = false;
    map.setView([39.0, 35.0], 6.2);
    
    // Clear all layers
    detailMarkers.forEach(m => map.removeLayer(m));
    detailMarkers = [];
    if(heatmapLayer) {
      map.removeLayer(heatmapLayer);
      heatmapLayer = null;
    }
    if(satelliteLayer) {
      map.removeLayer(satelliteLayer);
      satelliteLayer = null;
    }
    
    // Show city markers again
    cityMarkers.forEach(m => m.setStyle({opacity: 1, fillOpacity: m.options.fillColor === '#10b981' ? 0.9 : 0.5}));
    document.getElementById('back-to-turkey').style.display = 'none';
    
    // Reset info panel
    document.getElementById('avg-green').textContent = '-';
    document.getElementById('avg-grey').textContent = '-';
    document.getElementById('avg-water').textContent = '-';
  }
});

// Update showIstanbulDetails to show back button
const originalShowDetails = showIstanbulDetails;
showIstanbulDetails = async function() {
  await originalShowDetails();
  document.getElementById('back-to-turkey').style.display = 'block';
};

// Initial load - show Turkey cities
loadTurkeyCities();