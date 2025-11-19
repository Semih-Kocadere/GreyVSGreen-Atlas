// ============================================================================
// Authentication
// ============================================================================

const token = localStorage.getItem('token');
if (!token) {
  location.href = '/';
}

/**
 * Fetch and display user information
 */
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
    console.error('Kullanıcı bilgisi alınamadı:', error);
    location.href = '/';
  }
}

whoami();


// ============================================================================
// Logout
// ============================================================================

document.getElementById('logout').onclick = () => {
  localStorage.removeItem('token');
  location.href = '/';
};


// ============================================================================
// SIDEBAR FUNCTIONS
// ============================================================================

const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');
let isCollapsed = false;

/**
 * Toggle sidebar open/close
 */
sidebarToggle.onclick = () => {
  isCollapsed = !isCollapsed;
  sidebar.classList.toggle('collapsed', isCollapsed);
  sidebarToggle.textContent = isCollapsed ? '☰' : '✕';
  sidebarToggle.title = isCollapsed ? 'Sidebar Aç' : 'Sidebar Kapat';
};

/**
 * Sidebar width adjustment (drag)
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
  document.documentElement.style.setProperty('--sidebar-width', newWidth + 'px');
});

document.addEventListener('mouseup', () => {
  if (!isResizing) return;
  isResizing = false;
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
});


// ============================================================================
// DATA SOURCE ACTIONS
// ============================================================================

/**
 * Add new data source button
 */
document.querySelector('.add-source-btn').onclick = () => {
  alert('Yeni veri kaynağı ekleme özelliği yakında eklenecek!');
};

/**
 * Synchronization buttons
 * 
 * What does synchronization do?
 * ---------------------------
 * 1. Automatically downloads new data from remote sources (ESA Copernicus, Google Earth Engine, etc.)
 * 2. Adds updated satellite images and land-cover layers to the system
 * 3. Updates the database to the most current state (e.g., January 2024 -> June 2024)
 * 4. Feeds data flow for trend analyses and time series modeling
 * 5. Recalculates indices such as NDVI, NDWI, NDBI
 * 
 * Example: When Sentinel-2 is synchronized:
 * - Images from the last 30 days are fetched from the ESA API
 * - Images with cloud coverage below 20% are selected
 * - Automatic preprocessing is performed (atmospheric correction, masking)
 * - Saved to PostgreSQL/PostGIS
 * - Indices are calculated and prepared for analysis
 */
document.querySelectorAll('.btn-primary').forEach(btn => {
  btn.onclick = function(e) {
    e.preventDefault();
    const sourceName = this.closest('.source-card').querySelector('.source-name').textContent;
    
    if (this.textContent.includes('Aktif Et')) {
      alert(`${sourceName} veri kaynağı aktif edilecek...`);
      // TODO: API call to activate the source
    } else {
      alert(`✅ ${sourceName} senkronizasyonu başlatılıyor...\n\n` +
            `🔄 Uzak sunucudan yeni veriler indiriliyor\n` +
            `📊 Veri tabanı güncelleniyor\n` +
            `🧮 İndeksler yeniden hesaplanıyor\n\n` +
            `Bu işlem birkaç dakika sürebilir.`);
      // TODO: API call to start synchronization
      // fetch(`/api/sync/${sourceId}`, { method: 'POST' })
    }
  };
});

/**
 * Pause buttons
 */
document.querySelectorAll('.btn-danger').forEach(btn => {
  btn.onclick = function(e) {
    e.preventDefault();
    const sourceName = this.closest('.source-card').querySelector('.source-name').textContent;
    
    if (confirm(`${sourceName} senkronizasyonu duraklatılsın mı?`)) {
      alert('Senkronizasyon duraklatılıyor...');
      // TODO: API call to pause synchronization
    }
  };
});

/**
 * Settings buttons
 */
document.querySelectorAll('.btn-secondary').forEach(btn => {
  btn.onclick = function(e) {
    e.preventDefault();
    const sourceName = this.closest('.source-card').querySelector('.source-name').textContent;
    alert(`${sourceName} ayarları açılacak...`);
    // TODO: Open settings modal or page
  };
});


// ============================================================================
// REAL-TIME DATA LOADING
// ============================================================================

/**
 * Update data source statistics in real-time
 * 
 * Normally, this data would come from the backend API:
 * GET /api/data-sources -> [{id, name, stats: {lastUpdate, dataSize, ...}}]
 * 
 * For now, it works with simulated data.
 */
async function loadDataSourceStats() {
  // Simulated data (would come from API in real application)
  const mockData = {
    sentinel2: {
      lastUpdate: '2 saat önce',
      dataSize: '1.24 TB',
      imageCount: '492 görüntü'
    },
    landsat: {
      dateRange: '2018-2024',
      dataSize: '867 GB',
      imageCount: '318 görüntü',
      syncProgress: '71%'
    },
    dynamicworld: {
      lastUpdate: '12 dakika önce',
      coverage: 'Güncel (2024)'
    },
    worldcover: {
      lastUpdate: '5 gün önce'
    },
    ibb: {
      lastUpdate: '3 saat önce',
      dataSize: '152 MB',
      apiCalls: '18,743'
    }
  };

  // Update data for each card
  document.querySelectorAll('.source-card[data-source]').forEach(card => {
    const sourceId = card.getAttribute('data-source');
    const data = mockData[sourceId];
    
    if (!data) return;

    // Update values for each data-field
    card.querySelectorAll('[data-field]').forEach(field => {
      const fieldName = field.getAttribute('data-field');
      if (data[fieldName]) {
        // Animated update
        field.style.opacity = '0.5';
        setTimeout(() => {
          field.textContent = data[fieldName];
          field.style.opacity = '1';
        }, 300);
      }
    });
  });

  console.log('✅ Veri kaynağı istatistikleri güncellendi');
}

// Load data when the page is loaded
loadDataSourceStats();

// Update data every 30 seconds (real-time simulation)
setInterval(loadDataSourceStats, 30000);

/**
 * For real API integration:
 * 
 * async function loadDataSourceStats() {
 *   try {
 *     const response = await fetch('/api/data-sources/stats', {
 *       headers: { Authorization: `Bearer ${token}` }
 *     });
 *     
 *     if (!response.ok) throw new Error('Stats yüklenemedi');
 *     
 *     const sources = await response.json();
 *     
 *     sources.forEach(source => {
 *       const card = document.querySelector(`[data-source="${source.id}"]`);
 *       if (!card) return;
 *       
 *       Object.entries(source.stats).forEach(([key, value]) => {
 *         const field = card.querySelector(`[data-field="${key}"]`);
 *         if (field) field.textContent = value;
 *       });
 *     });
 *   } catch (error) {
 *     console.error('Veri kaynağı istatistikleri yüklenemedi:', error);
 *   }
 * }
 */