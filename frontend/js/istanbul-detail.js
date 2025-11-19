console.log('🚀 Script starting...');
const token = localStorage.getItem('token');
if(!token){ location.href='/'; }

// ============================================================================
// Authentication
// ============================================================================
async function whoami(){
  const r = await fetch('/me',{headers:{Authorization:`Bearer ${token}`}});
  if(!r.ok){ localStorage.removeItem('token'); location.href='/'; return; }
  const me = await r.json();
  document.getElementById('who').textContent = me.email;
}
whoami();
document.getElementById('logout').onclick = () => {
  localStorage.removeItem('token'); location.href='/';
};

// Sidebar toggle
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');
let isCollapsed = true; // Start collapsed on istanbul-detail page

// Initialize sidebar as collapsed
sidebar.classList.add('collapsed');
sidebarToggle.textContent = '☰';

sidebarToggle.onclick = () => {
  isCollapsed = !isCollapsed;
  sidebar.classList.toggle('collapsed', isCollapsed);
  sidebarToggle.textContent = isCollapsed ? '☰' : '✕';
  sidebarToggle.title = isCollapsed ? 'Sidebar Open' : 'Sidebar Close';
  // When the main left sidebar is opened, close the left detail sidebar
  if (!isCollapsed) {
    const detailSidebarLeft = document.getElementById('detail-sidebar-left');
    if (detailSidebarLeft && !detailSidebarLeft.classList.contains('collapsed')) {
      detailLeftCollapsed = true;
      detailSidebarLeft.classList.add('collapsed');
    }
  }
};

// Sidebar resize
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

// Detail sidebar toggles (left and right)
const detailSidebarLeft = document.getElementById('detail-sidebar-left');
const detailSidebarRight = document.getElementById('detail-sidebar-right');
const detailToggleLeft = document.getElementById('detail-toggle-left');
const detailToggleRight = document.getElementById('detail-toggle-right');
const detailSidebarToggleLeft = document.getElementById('detail-sidebar-toggle-left');
const detailSidebarToggleRight = document.getElementById('detail-sidebar-toggle-right');

let detailLeftCollapsed = false;
let detailRightCollapsed = false;

// Toggle left detail sidebar
function toggleDetailLeft() {
  detailLeftCollapsed = !detailLeftCollapsed;
  detailSidebarLeft.classList.toggle('collapsed', detailLeftCollapsed);
  detailToggleLeft.classList.toggle('show', detailLeftCollapsed);
  detailToggleLeft.textContent = detailLeftCollapsed ? '▶' : '◀';
}

// Toggle right detail sidebar
function toggleDetailRight() {
  detailRightCollapsed = !detailRightCollapsed;
  detailSidebarRight.classList.toggle('collapsed', detailRightCollapsed);
  detailToggleRight.classList.toggle('show', detailRightCollapsed);
  detailToggleRight.textContent = detailRightCollapsed ? '◀' : '▶';
}

if(detailSidebarToggleLeft) detailSidebarToggleLeft.onclick = toggleDetailLeft;
if(detailToggleLeft) detailToggleLeft.onclick = toggleDetailLeft;
if(detailSidebarToggleRight) detailSidebarToggleRight.onclick = toggleDetailRight;
if(detailToggleRight) detailToggleRight.onclick = toggleDetailRight;

// Bottom bar toggle
const bottomBar = document.getElementById('bottom-bar');
const bottomBarToggle = document.getElementById('bottom-bar-toggle');
const detailToggleBottom = document.getElementById('detail-toggle-bottom');
let bottomBarCollapsed = false;

function toggleBottomBar() {
  bottomBarCollapsed = !bottomBarCollapsed;
  if(bottomBar) bottomBar.classList.toggle('collapsed', bottomBarCollapsed);
  if(detailToggleBottom) {
    detailToggleBottom.classList.toggle('show', bottomBarCollapsed);
    detailToggleBottom.textContent = bottomBarCollapsed ? '▲' : '▼';
  }
}

if(bottomBarToggle) bottomBarToggle.onclick = toggleBottomBar;
if(detailToggleBottom) detailToggleBottom.onclick = toggleBottomBar;

// Initialize map
// GEE AOI: Rectangle([28.62, 40.75, 29.56, 41.18])
const istanbulBounds = [[40.75, 28.62], [41.18, 29.56]];

const map = L.map('map', {
  center: [40.965, 29.09], // AOI merkezi
  zoom: 11,
  minZoom: 10,
  maxZoom: 18,
  zoomControl: true,
  scrollWheelZoom: true,
  doubleClickZoom: false,  // Çift tıklama ile ani zoom kapalı
  dragging: true,
  zoomSnap: 0.10,  // Zoom steps of 0.10 (very smooth)
  zoomDelta: 0.5,  // Zoom change for buttons
  wheelPxPerZoomLevel: 120  // Mouse wheel sensitivity (slow zoom)
});

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap',
  maxZoom: 18
}).addTo(map);

// Satellite tile layer
let satelliteTileLayer = null;
let currentSatelliteYear = 2018;
let currentSatelliteQuarter = 1;
let currentSatelliteIndex = 'ndvi';
let satelliteOpacity = 0.7;

// Initialize year/quarter from dropdown selected value
function initializeSatelliteFromDropdown() {
  const dropdown = document.getElementById('satellite-time');
  if (dropdown && dropdown.value) {
    const [year, quarter] = dropdown.value.split('_');
    currentSatelliteYear = parseInt(year);
    currentSatelliteQuarter = parseInt(quarter);
    console.log(`🛰️ Initialized from dropdown: ${currentSatelliteYear} Q${currentSatelliteQuarter}`);
  }
}

function updateSatelliteTileLayer() {
  // Remove old layer
  if (satelliteTileLayer) {
    map.removeLayer(satelliteTileLayer);
  }
  
  // Check if layer should be visible
  const isVisible = document.getElementById('tile-layer-toggle').checked;
  if (!isVisible) return;
  
  // Create new tile layer
  const tileUrl = `http://localhost:8000/api/tiles/${currentSatelliteYear}/${currentSatelliteQuarter}/${currentSatelliteIndex}/{z}/{x}/{y}.png`;
  
  satelliteTileLayer = L.tileLayer(tileUrl, {
    attribution: '© Sentinel-2 ESA',
    minZoom: 12, 
    maxZoom: 15,
    opacity: satelliteOpacity,
    tileSize: 256,
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  satelliteTileLayer.addTo(map);
  console.log(`🛰️ Tile layer updated: ${currentSatelliteYear} Q${currentSatelliteQuarter}, ${currentSatelliteIndex}`);
}

// Satellite controls event listeners
document.getElementById('satellite-time').addEventListener('change', (e) => {
  const [year, quarter] = e.target.value.split('_');
  currentSatelliteYear = parseInt(year);
  currentSatelliteQuarter = parseInt(quarter);
  updateSatelliteTileLayer();
});

// Satellite index buttons (using view-btn with data-index)
document.querySelectorAll('.view-btn[data-index]').forEach(btn => {
  btn.addEventListener('click', (e) => {
    // Remove active class from satellite buttons only
    document.querySelectorAll('.view-btn[data-index]').forEach(b => b.classList.remove('active'));
    // Add active to clicked button
    btn.classList.add('active');
    
    // Update index
    currentSatelliteIndex = btn.dataset.index;
    updateSatelliteTileLayer();
  });
});

document.getElementById('tile-opacity').addEventListener('input', (e) => {
  satelliteOpacity = e.target.value / 100;
  document.getElementById('opacity-value').textContent = e.target.value + '%';
  
  if (satelliteTileLayer) {
    satelliteTileLayer.setOpacity(satelliteOpacity);
  }
});

document.getElementById('tile-layer-toggle').addEventListener('change', (e) => {
  if (e.target.checked) {
    updateSatelliteTileLayer();
  } else if (satelliteTileLayer) {
    map.removeLayer(satelliteTileLayer);
    satelliteTileLayer = null;
  }
});

// Zoom control - Hide tile layer at low zoom
map.on('zoomend', () => {
  const currentZoom = map.getZoom();
  const checkbox = document.getElementById('tile-layer-toggle');
  
  // If zoom is less than 12 and layer is visible, automatically hide it
  if (currentZoom < 12 && checkbox.checked && satelliteTileLayer) {
    map.removeLayer(satelliteTileLayer);
    satelliteTileLayer = null;
    console.log('ℹ️ Zoom seviyesi düşük - tile layer gizlendi');
  }
  // If zoom is greater than 12 and checkbox is checked, show the layer
  else if (currentZoom >= 12 && checkbox.checked && !satelliteTileLayer) {
    updateSatelliteTileLayer();
    console.log('✅ Zoom seviyesi yeterli - tile layer gösteriliyor');
  }
});

// Initialize satellite layer
initializeSatelliteFromDropdown();
updateSatelliteTileLayer();

let gridLayers = [];
let currentTimeframe = 'now';
let currentLayer = 'all';
let currentView = 'single';
let allData = null;

// Timeframe mapping
const timeframes = ['now', '6m', '12m'];
const timeframeDates = ['Ekim 2024', 'Nisan 2025', 'Ekim 2025'];

// Load detailed data
async function loadData() {
  try {
    const r = await fetch('/api/istanbul/detailed', {
      headers: {Authorization: `Bearer ${token}`}
    });
    if(!r.ok) {
      console.error('API Response:', r.status, r.statusText);
      throw new Error(`Failed to load data: ${r.status}`);
    }
    
    allData = await r.json();
    console.log('✅ Loaded data:', allData);
    
    // Render grid
    renderGrid(currentTimeframe, currentLayer);
    
    // Update stats
    updateStats();
    
    // Create charts
    createCharts();
    
    // Update district list
    updateDistrictList();
    
    // Update summary
    updateSummary();
    
  } catch(err) {
    console.error('❌ Error loading data:', err);
    
    // Show error message on map
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(239, 68, 68, 0.95);
      color: white;
      padding: 20px 30px;
      border-radius: 12px;
      font-weight: 600;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
      z-index: 1000;
    `;
    errorDiv.textContent = '⚠️ Veri yüklenemedi. Backend çalışıyor mu?';
    document.querySelector('.map-container').appendChild(errorDiv);
  } finally {
    // Always hide loading spinner
    const loadingElement = document.getElementById('loading');
    if(loadingElement) {
      loadingElement.style.display = 'none';
    }
  }
}

// Render regions as large circular markers
function renderGrid(timeframe, layer) {
  // Clear existing layers
  gridLayers.forEach(l => map.removeLayer(l));
  gridLayers = [];
  
  if(!allData) return;
  
  allData.grid.forEach(region => {
    const data = region[timeframe];
    
    // Determine color and size based on layer
    let color, radius;
    if(layer === 'all') {
      // Composite color based on green percentage
      const greenPercent = data.green;
      color = greenPercent > 40 ? '#10b981' : greenPercent > 25 ? '#f59e0b' : '#ef4444';
      radius = 2000; // 2km radius
    } else if(layer === 'green') {
      color = '#10b981';
      radius = (data.green / 100) * 3000;
    } else if(layer === 'grey') {
      color = '#ef4444';
      radius = (data.grey / 100) * 3000;
    } else if(layer === 'water') {
      color = '#3b82f6';
      radius = (data.water / 100) * 3000;
    }
    
    // Create circle marker
    const circle = L.circle([region.lat, region.lng], {
      color: 'white',
      weight: 3,
      fillColor: color,
      fillOpacity: 0.6,
      radius: radius
    }).addTo(map);
    
    circle.bindPopup(`
      <div style="padding:12px;min-width:220px">
        <div style="font-weight:900;font-size:16px;margin-bottom:10px;color:#0284c7">${region.name}</div>
        <div style="font-size:14px;line-height:1.8">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span>🌳 Yeşil:</span>
            <strong style="color:#10b981">${data.green}%</strong>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span>🏢 Beton:</span>
            <strong style="color:#ef4444">${data.grey}%</strong>
          </div>
          <div style="display:flex;justify-content:space-between">
            <span>💧 Su:</span>
            <strong style="color:#3b82f6">${data.water}%</strong>
          </div>
        </div>
      </div>
    `);
    
    gridLayers.push(circle);
  });
}

// Update stats
function updateStats() {
  const pred = allData.predictions.find(p => p.period === currentTimeframe);
  if(!pred) return;
  
  // Check if stat elements exist before updating
  const statGreen = document.getElementById('stat-green');
  const statGrey = document.getElementById('stat-grey');
  const statWater = document.getElementById('stat-water');
  
  if(statGreen) statGreen.textContent = pred.green + '%';
  if(statGrey) statGrey.textContent = pred.grey + '%';
  if(statWater) statWater.textContent = pred.water + '%';
}

// Create charts
function createCharts() {
  const trendCanvas = document.getElementById('trendChart');
  if(!trendCanvas) {
    console.log('⚠️ Chart canvas not found');
    return;
  }
  
  try {
    // Historical trend chart - horizontal timeline
    const trendCtx = trendCanvas.getContext('2d');
    new Chart(trendCtx, {
      type: 'line',
      data: {
        labels: allData.historical.map(h => h.year),
        datasets: [
          {
            label: '🌳 Yeşil Alan',
            data: allData.historical.map(h => h.green),
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.15)',
            borderWidth: 3,
            tension: 0.3,
            fill: true,
            pointRadius: 5,
            pointHoverRadius: 7,
            pointBackgroundColor: '#10b981',
            pointBorderColor: '#fff',
            pointBorderWidth: 2
          },
          {
            label: '🏢 Beton',
            data: allData.historical.map(h => h.grey),
            borderColor: '#9ca3af',
            backgroundColor: 'rgba(156, 163, 175, 0.15)',
            borderWidth: 3,
            tension: 0.3,
            fill: true,
            pointRadius: 5,
            pointHoverRadius: 7,
            pointBackgroundColor: '#9ca3af',
            pointBorderColor: '#fff',
            pointBorderWidth: 2
          },
          {
            label: '💧 Su Alanı',
            data: allData.historical.map(h => h.water),
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.15)',
            borderWidth: 3,
            tension: 0.3,
            fill: true,
            pointRadius: 5,
            pointHoverRadius: 7,
            pointBackgroundColor: '#3b82f6',
            pointBorderColor: '#fff',
            pointBorderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'right',
            labels: {
              color: 'white',
              font: {size: 13, weight: '600'},
              padding: 15,
              usePointStyle: true,
              pointStyle: 'circle'
            }
          },
          tooltip: {
            backgroundColor: 'rgba(15, 23, 42, 0.95)',
            titleColor: 'white',
            bodyColor: 'white',
            borderColor: 'rgba(59, 130, 246, 0.5)',
            borderWidth: 1,
            padding: 12,
            displayColors: true,
            callbacks: {
              label: function(context) {
                return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: false,
            min: 0,
            max: 70,
            ticks: {
              stepSize: 10,
              color: 'rgba(255, 255, 255, 0.7)',
              font: {size: 12},
              callback: function(value) {
                return value + '%';
              }
            },
            grid: {
              color: 'rgba(255, 255, 255, 0.08)',
              drawBorder: false
            }
          },
          x: {
            ticks: {
              color: 'rgba(255, 255, 255, 0.7)',
              font: {size: 12, weight: '600'}
            },
            grid: {
              display: false
            }
          }
        }
      }
    });
    console.log('✅ Chart created successfully');
  } catch(err) {
    console.error('❌ Error creating chart:', err);
  }
}

// Update district list
function updateDistrictList() {
  const list = document.getElementById('district-list');
  if(!list || !allData || !allData.district_changes) {
    console.log('⚠️ District list element or data not found');
    return;
  }
  
  list.innerHTML = '';
  
  allData.district_changes.forEach(d => {
    const item = document.createElement('div');
    item.className = 'district-item';
    item.innerHTML = `
      <div class="district-name">${d.name}</div>
      <div class="district-stats">
        <span class="district-green">${d.now_green}% → ${d.future_green}%</span>
        <span class="district-risk risk-${d.risk}">${d.risk}</span>
      </div>
    `;
    list.appendChild(item);
  });
}

// Update summary
function updateSummary() {
  if(!allData || !allData.summary) {
    console.log('⚠️ Summary data not found');
    return;
  }
  
  const s = allData.summary;
  const totalArea = document.getElementById('total-area');
  const population = document.getElementById('population');
  const greenPerCapita = document.getElementById('green-per-capita');
  const lossRate = document.getElementById('loss-rate');
  
  if(totalArea) totalArea.textContent = s.total_area_km2.toLocaleString() + ' km²';
  if(population) population.textContent = s.population.toLocaleString();
  if(greenPerCapita) greenPerCapita.textContent = s.green_per_capita_m2.toFixed(2) + ' m²';
  if(lossRate) lossRate.textContent = s.green_loss_rate_yearly.toFixed(1) + '%';
}

// Timeline slider (only if element exists)
const timelineSlider = document.getElementById('timeline-slider');
if(timelineSlider) {
  timelineSlider.addEventListener('input', (e) => {
    const index = parseInt(e.target.value);
    currentTimeframe = timeframes[index];
    
    // Update labels
    document.querySelectorAll('.timeline-label').forEach((label, i) => {
      label.classList.toggle('active', i === index);
    });
    
    // Update date
    const currentDateEl = document.getElementById('current-date');
    if(currentDateEl) currentDateEl.textContent = timeframeDates[index];
    
    // Re-render grid
    renderGrid(currentTimeframe, currentLayer);
    updateStats();
  });
}

// Layer buttons (using view-btn with data-layer)
document.querySelectorAll('.view-btn[data-layer]').forEach(btn => {
  btn.addEventListener('click', () => {
    // Remove active class from layer buttons only
    document.querySelectorAll('.view-btn[data-layer]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentLayer = btn.dataset.layer;
    renderGrid(currentTimeframe, currentLayer);
  });
});

// Note: View mode buttons removed - using single view only

// Initial load - wait for DOM to be ready
console.log('🔄 Script loaded, readyState:', document.readyState);

if (document.readyState === 'loading') {
  console.log('⏳ DOM still loading, adding event listener');
  document.addEventListener('DOMContentLoaded', () => {
    console.log('✅ DOMContentLoaded fired, calling loadData()');
    loadData();
  });
} else {
  // DOM already loaded
  console.log('✅ DOM already loaded, calling loadData() immediately');
  loadData();
}