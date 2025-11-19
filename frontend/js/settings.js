// ============================================================================
// AUTHENTICATION
// ============================================================================

const token = localStorage.getItem('token');
if (!token) {
  location.href = '/';
}

/**
 * Fetch and display user info
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
    document.getElementById('email').value = user.email;
    document.getElementById('fullname').value = user.full_name || '';
  } catch (error) {
    console.error('Kullanıcı bilgisi alınamadı:', error);
    location.href = '/';
  }
}

whoami();


// ============================================================================
// LOGOUT
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

sidebarToggle.onclick = () => {
  isCollapsed = !isCollapsed;
  sidebar.classList.toggle('collapsed', isCollapsed);
  sidebarToggle.textContent = isCollapsed ? '☰' : '✕';
  sidebarToggle.title = isCollapsed ? 'Sidebar Aç' : 'Sidebar Kapat'; // Keep Turkish for UI
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
// PROFILE SETTINGS
// ============================================================================

/**
 * Save profile information
 */
document.getElementById('save-profile').onclick = async () => {
  const fullname = document.getElementById('fullname').value;
  const newPassword = document.getElementById('new-password').value;
  const confirmPassword = document.getElementById('confirm-password').value;
  
  // Password check
  if (newPassword && newPassword !== confirmPassword) {
    alert('❌ Şifreler eşleşmiyor!');
    return;
  }
  
  // Simulated save
  alert('✅ Profil bilgileri kaydedildi!\n\n' + 
        `Ad Soyad: ${fullname}\n` +
        (newPassword ? 'Şifre güncellendi' : 'Şifre değiştirilmedi'));
  
  // Clear password fields
  document.getElementById('new-password').value = '';
  document.getElementById('confirm-password').value = '';
  
  // TODO: Backend API call
  // await fetch('/api/profile', { method: 'PATCH', ... });
};

/**
 * Cancel button
 */
document.getElementById('cancel-profile').onclick = () => {
  document.getElementById('new-password').value = '';
  document.getElementById('confirm-password').value = '';
  whoami(); // Load original values
};


// ============================================================================
// LANGUAGE SETTINGS
// ============================================================================

/**
 * Save language selection
 */
document.getElementById('save-language').onclick = () => {
  const selectedLang = document.querySelector('input[name="language"]:checked').value;
  const langName = selectedLang === 'tr' ? 'Türkçe' : 'English';
  
  alert(`✅ Dil değiştirildi: ${langName}\n\nSayfa yeniden yüklenecek...`);
  
  // Save to LocalStorage
  localStorage.setItem('language', selectedLang);
  
  // TODO: Reload page and apply translations
  // location.reload();
};


// ============================================================================
// THEME SETTINGS
// ============================================================================

/**
 * Save theme selection
 */
document.getElementById('save-theme').onclick = () => {
  const selectedTheme = document.querySelector('input[name="theme"]:checked').value;
  const themeNames = {
    'light': 'Açık Tema',
    'dark': 'Koyu Tema',
    'auto': 'Otomatik'
  };
  
  alert(`✅ Tema değiştirildi: ${themeNames[selectedTheme]}\n\nDeğişiklikler uygulanacak...`);
  
  // Save to LocalStorage
  localStorage.setItem('theme', selectedTheme);
  
  // TODO: Apply theme changes
  // applyTheme(selectedTheme);
};


// ============================================================================
// NOTIFICATION SETTINGS
// ============================================================================

/**
 * Save notification preferences
 */
document.getElementById('save-notifications').onclick = () => {
  const emailNotif = document.getElementById('email-notifications').checked;
  const browserNotif = document.getElementById('browser-notifications').checked;
  const weeklyReport = document.getElementById('weekly-report').checked;
  const dataUpdates = document.getElementById('data-updates').checked;
  
  alert('✅ Bildirim tercihleri kaydedildi!\n\n' +
        `E-posta: ${emailNotif ? 'Açık' : 'Kapalı'}\n` +
        `Tarayıcı: ${browserNotif ? 'Açık' : 'Kapalı'}\n` +
        `Haftalık Rapor: ${weeklyReport ? 'Açık' : 'Kapalı'}\n` +
        `Veri Güncellemeleri: ${dataUpdates ? 'Açık' : 'Kapalı'}`);
  
  // TODO: Backend API call
  // await fetch('/api/notifications', { method: 'PATCH', ... });
};


// ============================================================================
// PRIVACY SETTINGS
// ============================================================================

/**
 * Save privacy settings
 */
document.getElementById('save-privacy').onclick = () => {
  const twoFactor = document.getElementById('two-factor').checked;
  const profilePrivacy = document.getElementById('profile-privacy').checked;
  const activityLog = document.getElementById('activity-log').checked;
  
  alert('✅ Gizlilik ayarları kaydedildi!\n\n' +
        `İki Faktörlü Doğrulama: ${twoFactor ? 'Açık' : 'Kapalı'}\n` +
        `Profil Gizliliği: ${profilePrivacy ? 'Açık' : 'Kapalı'}\n` +
        `Aktivite Geçmişi: ${activityLog ? 'Açık' : 'Kapalı'}`);
  
  // TODO: Backend API call
  // await fetch('/api/privacy', { method: 'PATCH', ... });
};


// ============================================================================
// DELETE ACCOUNT
// ============================================================================

/**
 * Delete account process
 */
document.getElementById('delete-account').onclick = () => {
  const confirmation = confirm(
    '⚠️ UYARI: HESABINIZI SİLMEK ÜZEREYİZ!\n\n' +
    'Bu işlem geri alınamaz ve tüm verileriniz kalıcı olarak silinecektir.\n\n' +
    'Hesabınızı silmek istediğinizden emin misiniz?'
  );
  
  if (!confirmation) {
    return;
  }
  
  const finalConfirmation = prompt(
    'Son onay: Hesabınızı silmek için "SİL" yazın (büyük harflerle):'
  );
  
  if (finalConfirmation === 'SİL') {
    alert('❌ Hesabınız siliniyor...\n\nOturum kapatılacak ve ana sayfaya yönlendirileceksiniz.');
    
    // TODO: Backend API call
    // await fetch('/api/account', { method: 'DELETE', ... });
    
    // Logout
    localStorage.removeItem('token');
    location.href = '/';
  } else {
    alert('✅ İşlem iptal edildi. Hesabınız güvende!');
  }
};