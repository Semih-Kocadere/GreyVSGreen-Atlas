// ============================================================================
// AUTHENTICATION
// ============================================================================
const token = localStorage.getItem('token');
if (!token) {
  location.href = '/';
}

async function whoami() {
  try {
    const response = await fetch('/me', {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!response.ok) {
      localStorage.removeItem('token');
      location.href = '/';
      return;
    }
    const user = await response.json();
    document.getElementById('who').textContent = user.email;
  } catch (error) {
    console.error('User information could not be retrieved:', error);
    location.href = '/';
  }
}

whoami();

document.getElementById('logout').onclick = () => {
  localStorage.removeItem('token');
  location.href = '/';
};

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================
let predictionData = null; // All prediction sequences from the API
let overlayLayer = null; // Map overlay layer

// ============================================================================
// DATA LOADING
// ============================================================================
async function loadPredictionData() {
  showLoading(true);
  try {
    const response = await fetch('/api/trend/predict', {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!response.ok) {
      throw new Error('Prediction data could not be loaded');
    }
    const data = await response.json();
    predictionData = data;
    updateUI();
    createTrendChart();
    console.log('Prediction data loaded:', predictionData);
  } catch (error) {
    console.error('Prediction data loading error:', error);
    showErrorMessage('Prediction data could not be loaded. Please ensure the models are loaded.');
    showLoading(false);
  } finally {
    showLoading(false);
  }
}

// Trend Chart.js function
let trendChartInstance = null;
function createTrendChart() {
  const chartEl = document.getElementById('trendChart');
  if (!chartEl || !predictionData || !predictionData.predictions) return;

  // X axis: periods
  const labels = predictionData.predictions.map(p => p.prediction.timeframe);
  // Y axis: green, grey, water ratios
  const green = predictionData.predictions.map(p => p.prediction.green);
  const grey = predictionData.predictions.map(p => p.prediction.grey);
  const water = predictionData.predictions.map(p => p.prediction.water);

  if (trendChartInstance) {
    trendChartInstance.destroy();
  }
  trendChartInstance = new Chart(chartEl.getContext('2d'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: '🌳 Yeşil Alan',
          data: green,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16,185,129,0.15)',
          borderWidth: 3,
          tension: 0.3,
          fill: true,
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          label: '🏢 Beton',
          data: grey,
          borderColor: '#9ca3af',
          backgroundColor: 'rgba(156,163,175,0.15)',
          borderWidth: 3,
          tension: 0.3,
          fill: true,
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          label: '💧 Su',
          data: water,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.15)',
          borderWidth: 3,
          tension: 0.3,
          fill: true,
          pointRadius: 4,
          pointHoverRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        title: { display: false }
      },
      scales: {
        y: { beginAtZero: true, max: 100, title: { display: true, text: '%' } }
      }
    }
  });
}


function updateUI() {
  if (!predictionData) return;

  // Find Horizon dropdown selected prediction
  const horizonValue = document.getElementById('horizon-select').value;

  // Find appropriate prediction
  let pred = null;
  if (predictionData.predictions && Array.isArray(predictionData.predictions)) {
    pred = predictionData.predictions.find(
      p => String(p.horizon) === String(horizonValue)
    );
    // If no horizon found, show the first prediction
    if (!pred) pred = predictionData.predictions[0];
  }
  if (!pred) return;

  // Update statistic cards
  updateStatCard('green', pred.current.green, pred.prediction.green, pred.changes.green);
  updateStatCard('gray',  pred.current.grey,  pred.prediction.grey,  pred.changes.grey);
  updateStatCard('water', pred.current.water, pred.prediction.water, pred.changes.water);

  // Add mask overlay to the map (if any)
  addPredictionOverlay(pred);

  // --- TREND TILE LAYER INTEGRATION ---
  const year    = pred.prediction.year    || pred.current.year;
  const quarter = pred.prediction.quarter || pred.current.quarter;
  const horizon = pred.horizon || parseInt(horizonValue, 10) || 1;

  // Update global values so zoom handler etc. use the same info
  window.currentTrendYear = year;
  window.currentTrendQuarter = quarter;
  window.currentTrendHorizon = horizon;

  // Remove old trend tile layer
  if (window.trendTileLayer) {
    window.map.removeLayer(window.trendTileLayer);
    window.trendTileLayer = null;
  }

  // New trend tile URL (including horizon)
  const tileUrl =
    `/api/trend/tiles/${year}/${quarter}/${horizon}/{z}/{x}/{y}.png`;

  // Add new trend tile layer
  window.trendTileLayer = L.tileLayer(tileUrl, {
    attribution: 'Trend Prediction',
    opacity: window.trendOpacity,
    maxZoom: 18,
    tileSize: 256
  });
  window.trendTileLayer.addTo(window.map);

  console.log(`🟢 updateUI → trend tiles: ${year} Q${quarter} t+${horizon}`);
}

function updateStatCard(type, current, predicted, change) {
  const currentEl = document.getElementById(`stat-${type}-current`);
  const predictedEl = document.getElementById(`stat-${type}-predicted`);
  const changeEl = document.getElementById(`stat-${type}-change`);
  
  if (currentEl) currentEl.textContent = `${current.toFixed(1)}%`;
  if (predictedEl) predictedEl.textContent = `${predicted.toFixed(1)}%`;
  
  if (changeEl) {
    const sign = change >= 0 ? '+' : '';
    changeEl.textContent = `${sign}${change.toFixed(1)}%`;
    changeEl.className = 'stat-change ' + (change >= 0 ? 'positive' : 'negative');
  }
}

function addPredictionOverlay(pred) {
  // Remove old overlays
  if (window.overlayLayers && Array.isArray(window.overlayLayers)) {
    window.overlayLayers.forEach(l => map.removeLayer(l));
  }
  window.overlayLayers = [];
  // Only overlay for Istanbul prediction
  if (window.overlayLayers && Array.isArray(window.overlayLayers)) {
    window.overlayLayers.forEach(l => map.removeLayer(l));
  }
  window.overlayLayers = [];
  if (pred && pred.class_mask) {
    const mask = pred.class_mask;
    // Fixed size: 256x256
    const w = 256;
    const h = 256;
    // If mask size is different, resize (if needed)
    // (Here, by default, mask[x][y] is used if present)
    // Colors: 0-bg, 1-green, 2-gray, 3-water
    const colors = {
      0: [0,0,0,0],        // background: transparent
      1: [16,185,129,220], // green: #10b981
      2: [128,128,128,220],// gray: #808080
      3: [59,130,246,220]  // water: #3b82f6
    };
    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let cls = 0;
        if (mask[x] && typeof mask[x][y] !== 'undefined') {
          cls = mask[x][y];
        }
        const idx = (y * w + x) * 4;
        const color = colors[cls] || [0,0,0,0];
        imgData.data[idx] = color[0];
        imgData.data[idx+1] = color[1];
        imgData.data[idx+2] = color[2];
        imgData.data[idx+3] = Math.round(color[3] * (opacitySlider.value/100));
      }
    }
    ctx.putImageData(imgData, 0, 0);
    const dataUrl = canvas.toDataURL('image/png');
    // Bounds exactly matching backend AOI
    const bounds = [[40.75, 28.62], [41.18, 29.56]];
    const layer = L.imageOverlay(dataUrl, bounds, {opacity: 0.7});
    layer.addTo(map);
    window.overlayLayers.push(layer);
  }
}

function showErrorMessage(message) {
  const errorDiv = document.createElement('div');
  errorDiv.style.cssText = `
    position: fixed;
    top: 80px;
    right: 20px;
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    padding: 16px 24px;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(239, 68, 68, 0.4);
    z-index: 9999;
    font-weight: 600;
    max-width: 400px;
  `;
  errorDiv.textContent = '⚠️ ' + message;
  document.body.appendChild(errorDiv);
  
  setTimeout(() => {
    errorDiv.remove();
  }, 5000);
}



// --- MAP INITIALIZATION AND TREND TILE LAYER ---
window.addEventListener('DOMContentLoaded', () => {
  // Core area where the dataset actually exists (the overlay itself)
  const dataBounds = [[40.75, 28.62], [41.18, 29.56]];

  // Wider Istanbul area to show to the user
  const viewBounds = [[40.60, 28.30], [41.40, 29.90]];

  window.map = L.map('map', {
    center: [41.0, 29.0],
    zoom: 11,         
    minZoom: 10,       
    maxZoom: 14,
    zoomControl: true,
    scrollWheelZoom: true,
    doubleClickZoom: false,
    dragging: true,
    zoomSnap: 0.15,
    zoomDelta: 0.25,
    wheelPxPerZoomLevel: 120,
    maxBounds: dataBounds  // Restrict map bounds to data area
  });

  // Fit view to Istanbul area
  window.map.fitBounds(viewBounds);

  // Base layer
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap',
    maxZoom: 18
  }).addTo(window.map);

  // Trend tile initial values
  window.currentTrendYear = 2026;
  window.currentTrendQuarter = 1;
  window.currentTrendHorizon = 1;   // t+1
  window.trendOpacity = 0.7;
  window.trendTileLayer = null;

  // Update trend tile layer according to backend endpoint
  window.updateTrendTileLayer = function() {
    if (window.trendTileLayer) {
      window.map.removeLayer(window.trendTileLayer);
    }

    const tileUrl =
      `/api/trend/tiles/${window.currentTrendYear}/${window.currentTrendQuarter}/${window.currentTrendHorizon}/{z}/{x}/{y}.png`;

    window.trendTileLayer = L.tileLayer(tileUrl, {
      attribution: 'Trend Prediction',
      minZoom: 9,
      maxZoom: 18,
      opacity: window.trendOpacity,
      tileSize: 256
    });

    window.trendTileLayer.addTo(window.map);
    console.log(`🟢 Trend tile layer updated: ${window.currentTrendYear} Q${window.currentTrendQuarter} t+${window.currentTrendHorizon}`);
  };

  // Opacity slider
  const opacitySlider = document.getElementById('opacity-slider');
  if (opacitySlider) {
    // Reflect initial value in UI
    opacitySlider.value = window.trendOpacity * 100;
    const valueLabel = document.getElementById('opacity-value');
    if (valueLabel) valueLabel.textContent = `${opacitySlider.value}%`;

    opacitySlider.addEventListener('input', (e) => {
      window.trendOpacity = e.target.value / 100;
      if (valueLabel) valueLabel.textContent = e.target.value + '%';
      if (window.trendTileLayer) {
        window.trendTileLayer.setOpacity(window.trendOpacity);
      }
    });
  }

  // Zoom control: lower the threshold if you want to disable when zoomed out too much
  window.map.on('zoomend', () => {
    const currentZoom = window.map.getZoom();
    if (currentZoom < 9 && window.trendTileLayer) {
      window.map.removeLayer(window.trendTileLayer);
      window.trendTileLayer = null;
      console.log('ℹ️ Zoom level low - trend tile layer hidden');
    } else if (currentZoom >= 9 && !window.trendTileLayer) {
      window.updateTrendTileLayer();
      console.log('✅ Zoom level sufficient - trend tile layer shown');
    }
  });

  // Add trend tile layer initially
  window.updateTrendTileLayer();
});


// Comparison map (initially hidden)
let mapCompare = null;

// Loading overlay control
function showLoading(show) {
  const overlay = document.getElementById('loading-overlay');
  if (overlay) overlay.style.display = show ? 'flex' : 'none';
}

// ============================================================================
// MODE SWITCHING
// ============================================================================
const modeButtons = document.querySelectorAll('.modern-mode-btn');
const mapElement = document.getElementById('map');
const mapCompareElement = document.getElementById('map-compare');

modeButtons.forEach(btn => {
  btn.onclick = () => {
    modeButtons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    
    const mode = btn.dataset.mode;
    
    if (mode === 'comparison') {
      // Comparison mode: two maps side by side
      mapElement.classList.add('map-split');
      mapCompareElement.classList.remove('map-hidden');
      mapCompareElement.classList.add('map-split');
      
      // Initialize second map (if not already initialized)
      if (!mapCompare) {
        mapCompare = L.map('map-compare').setView([41.0082, 28.9784], 11);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '© OpenStreetMap contributors'
        }).addTo(mapCompare);
      }
      
      // Update map sizes
      setTimeout(() => {
        map.invalidateSize();
        mapCompare.invalidateSize();
      }, 100);
      
    } else {
      // Single map mode
      mapElement.classList.remove('map-split');
      mapCompareElement.classList.add('map-hidden');
      mapCompareElement.classList.remove('map-split');
      
      setTimeout(() => {
        map.invalidateSize();
      }, 100);
    }
  };
});

// ============================================================================
// OPACITY CONTROL
// ============================================================================
const opacitySlider = document.getElementById('opacity-slider');
const opacityValue = document.getElementById('opacity-value');

opacitySlider.oninput = () => {
  opacityValue.textContent = opacitySlider.value + '%';
  
  // Update overlay opacity
  if (overlayLayer) {
    const opacity = opacitySlider.value / 100;
    overlayLayer.setStyle({ fillOpacity: opacity * 0.5 });
  }
};

// ============================================================================
// LAYER TOGGLE
// ============================================================================
const layerToggles = {
  green: document.getElementById('layer-green'),
  gray: document.getElementById('layer-gray'),
  water: document.getElementById('layer-water')
};

// Layer visibility states
let visibleLayers = { green: true, gray: true, water: true };

Object.keys(layerToggles).forEach(layerName => {
  if (layerToggles[layerName]) {
    layerToggles[layerName].onchange = (e) => {
      visibleLayers[layerName] = e.target.checked;
      updateUI(); // Katman değişince overlay güncellensin
      console.log(`Layer ${layerName}: ${visibleLayers[layerName] ? 'visible' : 'hidden'}`);
    };
  }
});

// Timeline and slider related code removed

// ============================================================================
// SIDEBAR TOGGLE
// ============================================================================
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');
const mainContent = document.querySelector('.main');
let isCollapsed = false;

sidebarToggle.onclick = () => {
  isCollapsed = !isCollapsed;
  sidebar.classList.toggle('collapsed', isCollapsed);
  if (mainContent) {
    mainContent.classList.toggle('sidebar-collapsed', isCollapsed);
  }
  sidebarToggle.textContent = isCollapsed ? '☰' : '✕';
  sidebarToggle.title = isCollapsed ? 'Sidebar Open' : 'Sidebar Close';
  
  // Update map sizes
  setTimeout(() => {
    map.invalidateSize();
    if (mapCompare) mapCompare.invalidateSize();
  }, 300);
};

// ============================================================================
// SIDEBAR WIDTH ADJUSTMENT
// ============================================================================
const resizer = document.getElementById('sidebar-resizer');
let isResizing = false;
let startX = 0;
let startWidth = 0;

resizer.addEventListener('mousedown', (e) => {
  isResizing = true;
  startX = e.clientX;
  startWidth = sidebar.offsetWidth;
  resizer.classList.add('resizing');
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
  resizer.classList.remove('resizing');
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
  
  setTimeout(() => {
    map.invalidateSize();
    if (mapCompare) mapCompare.invalidateSize();
  }, 100);
});

// Horizon dropdown change updates trend tiles
const horizonSelect = document.getElementById('horizon-select');
if (horizonSelect) {
  // Set horizon based on selected value as soon as the page loads
  window.currentTrendHorizon = parseInt(horizonSelect.value, 10) || 1;

  horizonSelect.addEventListener('change', () => {
    window.currentTrendHorizon = parseInt(horizonSelect.value, 10) || 1;

    console.log('🔄 Horizon değişti → t+', window.currentTrendHorizon);
    if (window.updateTrendTileLayer) {
      window.updateTrendTileLayer();
    }
  });
}

// ============================================================================
// LOAD DATA ON PAGE LOAD
// ============================================================================
loadPredictionData();


// ============================================================================
// FOLIUM MAP INTEGRATION
// ============================================================================


// --- PNG Mosaic Map shown with fetch+blob ---
async function showMosaicImage() {
  const year = window.currentTrendYear || 2026;
  const quarter = window.currentTrendQuarter || 1;
  const token = localStorage.getItem('token');
  if (!token) {
    alert('You must be logged in!');
    return;
  }
  const pngUrl = `/api/trend/folium_mosaic/${year}/${quarter}/download`;
  const foliumContainer = document.getElementById('folium-map-container');
  if (foliumContainer) {
    foliumContainer.innerHTML = '';
    try {
      const resp = await fetch(pngUrl, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (!resp.ok) throw new Error('PNG mosaic map could not be retrieved');
      const blob = await resp.blob();
      const imgUrl = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgUrl;
      img.alt = 'Prediction Mosaic Map';
      img.style.maxWidth = '100%';
      img.style.borderRadius = '12px';
      img.style.boxShadow = '0 2px 16px rgba(0,0,0,0.08)';
      foliumContainer.appendChild(img);
    } catch (e) {
      foliumContainer.innerHTML = '<span style="color:#ef4444">PNG mozaik haritası yüklenemedi</span>';
    }
  }
}

// --- Folium map automatically shown with iframe ---
async function showFoliumMapIframe() {
  const year = window.currentTrendYear || 2026;
  const quarter = window.currentTrendQuarter || 1;
  const horizon = document.getElementById('horizon-select')?.value || 1;
  const token = localStorage.getItem('token');
  const foliumContainer = document.getElementById('folium-map-container');
  if (foliumContainer) {
    let foliumUrl = `/api/trend/folium_map/${year}/${quarter}/${horizon}`;
    if (token) foliumUrl += `?token=${encodeURIComponent(token)}`;
    foliumContainer.innerHTML = '';
  const iframe = document.createElement('iframe');
  iframe.src = foliumUrl;
  iframe.style.width = '100%';
  iframe.style.height = '100%';
  iframe.style.border = 'none';
  iframe.style.borderRadius = '12px';
  iframe.style.display = 'block';
  foliumContainer.appendChild(iframe);
  }
}

// --- Page load and horizon change automatically show folium map ---
window.addEventListener('DOMContentLoaded', () => {
  showFoliumMapIframe();
  const horizonSelect = document.getElementById('horizon-select');
  if (horizonSelect) {
    horizonSelect.addEventListener('change', () => {
      showFoliumMapIframe();
    });
  }
});
