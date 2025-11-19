// Translations
const translations = {
  tr: {
    title: 'Hoş Geldiniz',
    subtitle: 'Şehirlerin yeşil geleceğini birlikte izleyelim',
    emailLabel: 'E-posta',
    passwordLabel: 'Şifre',
    loginBtn: 'Giriş Yap',
    error: 'Hatalı e-posta veya şifre',
    footerText: 'Grey vs Green Atlas © 2025'
  },
  en: {
    title: 'Welcome Back',
    subtitle: 'Monitor the green future of cities together',
    emailLabel: 'Email',
    passwordLabel: 'Password',
    loginBtn: 'Sign In',
    error: 'Incorrect email or password',
    footerText: 'Grey vs Green Atlas © 2025'
  }
};

// Language handling
let currentLang = localStorage.getItem('lang') || 'tr';

function updateLanguage() {
  const t = translations[currentLang];
  document.getElementById('title').textContent = t.title;
  document.getElementById('subtitle').textContent = t.subtitle;
  document.getElementById('emailLabel').textContent = t.emailLabel;
  document.getElementById('passwordLabel').textContent = t.passwordLabel;
  document.getElementById('loginBtn').textContent = t.loginBtn;
  document.getElementById('footerText').textContent = t.footerText;
  document.getElementById('langText').textContent = currentLang === 'tr' ? 'EN' : 'TR';
  document.documentElement.lang = currentLang;
}

document.getElementById('langToggle').addEventListener('click', () => {
  currentLang = currentLang === 'tr' ? 'en' : 'tr';
  localStorage.setItem('lang', currentLang);
  updateLanguage();
});

// Theme handling
const themeToggle = document.getElementById('themeToggle');
const themeIcon = themeToggle.querySelector('i');
let currentTheme = localStorage.getItem('theme') || 'dark';

function updateTheme() {
  document.body.setAttribute('data-theme', currentTheme);
  themeIcon.className = currentTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
}

themeToggle.addEventListener('click', () => {
  currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('theme', currentTheme);
  updateTheme();
});

updateTheme();
updateLanguage();

// Form handling
const form = document.getElementById('form');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const email = document.getElementById('email').value;
  const password = document.getElementById('password').value;
  const body = new URLSearchParams({username: email, password});
  try {
    const res = await fetch('/auth/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body
    });
    if (!res.ok) throw new Error(translations[currentLang].error);
    const data = await res.json();
    localStorage.setItem('token', data.access_token);
    location.href = '/dashboard.html';
  } catch(err) {
    const el = document.getElementById('err');
    el.style.display = 'block';
    el.textContent = err.message;
    setTimeout(() => el.style.display = 'none', 5000);
  }
});