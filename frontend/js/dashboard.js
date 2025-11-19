// ============================================================================
// Authentication Check
// ============================================================================

// Retrieve token from localStorage
const token = localStorage.getItem('token');

// If no token, redirect to login page
if (!token) {
  location.href = '/';
}

/**
 * Fetch current user information and display on the page
 * API: GET /me
 */
async function whoami() {
  try {
    const response = await fetch('/me', {
      headers: { Authorization: `Bearer ${token}` }
    });
    
    // If token is invalid, redirect to login
    if (!response.ok) {
      localStorage.removeItem('token');
      location.href = '/';
      return;
    }
    
    // Display user email in the topbar
    const user = await response.json();
    document.getElementById('who').textContent = user.email;
    
    // Update user information in the sidebar
    const sidebarUserName = document.getElementById('sidebar-user-name');
    if (sidebarUserName) {
      sidebarUserName.textContent = user.email;
    }
    
  } catch (error) {
    console.error('Kullanıcı bilgisi alınamadı:', error);
    location.href = '/';
  }
}
// Fetch current user information on page load
whoami();


// ============================================================================
// Logout
// ============================================================================

/**
 * When the logout button is clicked, remove the token and redirect to login
 */
document.getElementById('logout').onclick = () => {
  localStorage.removeItem('token');
  location.href = '/';
};


// ============================================================================
// Istanbul Data Display
// ============================================================================

/**
 * General toggle function for KPI buttons
 * Works independently for each city
 */
const kpiStates = {}; // Separate state for each city

// Add event listener to all KPI buttons
document.querySelectorAll('.btn-kpi').forEach(btn => {
  const cityName = btn.getAttribute('data-city');
  kpiStates[cityName] = false; // Initially closed
  
  btn.onclick = async (event) => {
    event.stopPropagation();
    event.preventDefault();
    
    const card = event.target.closest('.card');
    const kpisElement = document.getElementById(`kpis-${cityName}`);
    const noteElement = document.getElementById(`note-${cityName}`);
    
    // Toggle: If open, close it; if closed, open it
    if (kpiStates[cityName]) {
      kpisElement.style.display = 'none';
      noteElement.style.display = 'none';
      card.classList.remove('kpi-expanded');
      kpiStates[cityName] = false;
      console.log(`${cityName} KPI kapatıldı`);
      return;
    }
    
    // Open KPI - fetch data
    console.log(`${cityName} KPI açılıyor...`);
    try {
      const response = await fetch(`/api/city/${cityName}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      if (!response.ok) {
        alert('Veri alınamadı. Lütfen tekrar deneyin.');
        return;
      }
      
      const data = await response.json();
      
      // Fill KPI cards
      document.getElementById(`k-green-${cityName}`).textContent = data.now.green + '%';
      document.getElementById(`k-grey-${cityName}`).textContent = data.now.grey + '%';
      document.getElementById(`k-water-${cityName}`).textContent = data.now.water + '%';
      
      // Show data source note
      if (noteElement) {
        noteElement.textContent = data.note;
        noteElement.style.display = 'block';
      }
      
      // Make KPI cards visible
      kpisElement.style.display = 'grid';
      card.classList.add('kpi-expanded');
      kpiStates[cityName] = true;
      console.log(`${cityName} KPI açıldı`);
      
    } catch (error) {
      console.error(`${cityName} verileri yüklenemedi:`, error);
      alert('Bir hata oluştu. Lütfen tekrar deneyin.');
    }
  };
});


// ============================================================================
// Open/Close Sidebar (Sidebar Toggle)
// ============================================================================

const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');
const mainContent = document.querySelector('.main');
let isCollapsed = false;

/**
 * When the sidebar toggle button is clicked, open/close the sidebar
 * The button icon and tooltip also change (☰ ↔ ✕)
 */
sidebarToggle.onclick = () => {
  isCollapsed = !isCollapsed;
  // Add/remove 'collapsed' class (display:none in CSS)
  sidebar.classList.toggle('collapsed', isCollapsed);
  // Add/remove class to main content
  if (mainContent) {
    mainContent.classList.toggle('sidebar-collapsed', isCollapsed);
  }
  // Change button content
  sidebarToggle.textContent = isCollapsed ? '☰' : '✕';
  sidebarToggle.title = isCollapsed ? 'Open Sidebar' : 'Close Sidebar';
  // Add/remove class for color change
  if (isCollapsed) {
    sidebarToggle.classList.add('toggled');
  } else {
    sidebarToggle.classList.remove('toggled');
    sidebar.style.width = '';
    document.documentElement.style.setProperty('--sidebar-width', '300px');
  }
};


// ============================================================================
// Sidebar Resize
// ============================================================================

const resizer = document.getElementById('sidebar-resizer');
let isResizing = false; 
let startX = 0;         
let startWidth = 0;      

// Remove transition during dragging
function removeSidebarTransition() {
  sidebar.style.transition = 'none';
}
// Add transition back after dragging
function restoreSidebarTransition() {
  sidebar.style.transition = '';
}

/**
 * When mousedown on the resizer bar, start dragging
 */
resizer.addEventListener('mousedown', (e) => {
  isResizing = true;
  startX = e.clientX;
  startWidth = sidebar.offsetWidth;
  removeSidebarTransition();
  // Visual feedback: change resizer and cursor
  resizer.classList.add('resizing');
  document.body.style.cursor = 'col-resize';
  document.body.style.userSelect = 'none';  // Prevent text selection
});

/**
 * Adjust sidebar width while mouse moves
 * Min: 250px, Max: 600px
 */
document.addEventListener('mousemove', (e) => {
  if (!isResizing) return;

  // Calculate movement distance
  const diff = e.clientX - startX;
  const minSidebar = 60; // px, values below this are considered closed
  const maxSidebar = 600;
  const newWidth = Math.max(minSidebar, Math.min(maxSidebar, startWidth + diff));

  // Update sidebar width
  sidebar.style.width = newWidth + 'px';
  document.documentElement.style.setProperty('--sidebar-width', newWidth + 'px');

  // If narrowed too much, automatically close
  if (newWidth <= minSidebar + 2) {
    sidebar.classList.add('collapsed');
    if (mainContent) mainContent.classList.add('sidebar-collapsed');
    isCollapsed = true;
  } else {
    sidebar.classList.remove('collapsed');
    if (mainContent) mainContent.classList.remove('sidebar-collapsed');
    isCollapsed = false;
  }
});

/**
 * When mouse is released, stop dragging
 */
document.addEventListener('mouseup', () => {
  if (!isResizing) return;
  isResizing = false;
  resizer.classList.remove('resizing');
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
  restoreSidebarTransition();
});


// ============================================================================
// INITIALIZATION
// ============================================================================

// Save the initial sidebar width to a CSS variable
document.documentElement.style.setProperty(
  '--sidebar-width',
  sidebar.offsetWidth + 'px'
);