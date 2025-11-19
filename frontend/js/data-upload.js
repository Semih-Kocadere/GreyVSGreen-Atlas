// ============================================================================
// Authentication
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
    console.error('Kullanıcı bilgisi alınamadı:', error);
    location.href = '/';
  }
}

whoami();

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

sidebarToggle.onclick = () => {
  isCollapsed = !isCollapsed;
  sidebar.classList.toggle('collapsed', isCollapsed);
  sidebarToggle.textContent = isCollapsed ? '☰' : '✕';
  sidebarToggle.title = isCollapsed ? 'Sidebar Aç' : 'Sidebar Kapat';
};

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
// FILE UPLOAD
// ============================================================================

const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const progressOverlay = document.getElementById('progress-overlay');
const progressBar = document.getElementById('progress-bar');
const progressPercent = document.getElementById('progress-percent');
const progressFilename = document.getElementById('progress-filename');

/**
 * Drag & Drop - Change style when file is dragged over
 */
uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.classList.add('dragging');
});

uploadArea.addEventListener('dragleave', () => {
  uploadArea.classList.remove('dragging');
});

/**
 * Drag & Drop - Upload when file is dropped
 */
uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('dragging');
  
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFileUpload(files);
  }
});

/**
 * File input change - when button is clicked
 */
fileInput.addEventListener('change', (e) => {
  const files = e.target.files;
  if (files.length > 0) {
    handleFileUpload(files);
  }
});

/**
 * File upload process (simulated)
 * In a real application, it would be sent to the backend with FormData
 */
function handleFileUpload(files) {
  const file = files[0]; // Take the first file
  
  // Show progress overlay
  progressFilename.textContent = file.name;
  progressOverlay.classList.add('active');
  
  // Simulated upload progress
  let progress = 0;
  const interval = setInterval(() => {
    progress += Math.random() * 15;
    
    if (progress >= 100) {
      progress = 100;
      clearInterval(interval);
      
      // Upload completed
      setTimeout(() => {
        progressOverlay.classList.remove('active');
        progressBar.style.width = '0%';
        progressPercent.textContent = '0%';
        alert(`✅ ${file.name} başarıyla yüklendi!`);
        
        // Refresh file list (would be fetched from API in real application)
        // location.reload();
      }, 500);
    }
    
    progressBar.style.width = progress + '%';
    progressPercent.textContent = Math.round(progress) + '%';
  }, 200);
}


// ============================================================================
// FILE ACTIONS
// ============================================================================

/**
 * View buttons
 */
document.querySelectorAll('.btn-primary').forEach(btn => {
  btn.onclick = function(e) {
    e.preventDefault();
    const fileName = this.closest('.file-card').querySelector('.file-name').textContent;
    alert(`${fileName} görüntüleniyor...\n\nGerçek uygulamada bu dosya için önizleme/viewer açılacak.`);
  };
});

/**
 * Delete buttons
 */
document.querySelectorAll('.btn-danger').forEach(btn => {
  btn.onclick = function(e) {
    e.preventDefault();
    const fileName = this.closest('.file-card').querySelector('.file-name').textContent;
    
    if (confirm(`${fileName} dosyasını silmek istediğinizden emin misiniz?`)) {
      alert('Dosya siliniyor...\n\nGerçek uygulamada backend API\'ye DELETE isteği gönderilecek.');
      // TODO: API call to delete file
      // this.closest('.file-card').remove();
    }
  };
});