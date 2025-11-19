const token=localStorage.getItem('token');if(!token){location.href='/';}
async function whoami(){try{const r=await fetch('/me',{headers:{Authorization:'Bearer '+token}});if(!r.ok){localStorage.removeItem('token');location.href='/';return;}const u=await r.json();document.getElementById('who').textContent=u.email;}catch(e){location.href='/';}}
whoami();
document.getElementById('logout').onclick=()=>{localStorage.removeItem('token');location.href='/';};
const sidebarToggle=document.getElementById('sidebar-toggle');const sidebar=document.getElementById('sidebar');let isCollapsed=false;
sidebarToggle.onclick=()=>{isCollapsed=!isCollapsed;sidebar.classList.toggle('collapsed',isCollapsed);sidebarToggle.textContent=isCollapsed?'☰':'✕';};
const resizer=document.getElementById('sidebar-resizer');let isResizing=false;let startX=0;let startWidth=0;
resizer.addEventListener('mousedown',(e)=>{isResizing=true;startX=e.clientX;startWidth=sidebar.offsetWidth;document.body.style.cursor='col-resize';document.body.style.userSelect='none';});
document.addEventListener('mousemove',(e)=>{if(!isResizing)return;const diff=e.clientX-startX;const newWidth=Math.max(250,Math.min(600,startWidth+diff));sidebar.style.width=newWidth+'px';});
document.addEventListener('mouseup',()=>{if(!isResizing)return;isResizing=false;document.body.style.cursor='';document.body.style.userSelect='';});
document.querySelectorAll('.btn-primary').forEach(btn=>{btn.onclick=function(){alert('Rapor indiriliyor...\n\nGerçek uygulamada PDF dosyası oluşturulacak.');};});
document.querySelectorAll('.btn-secondary').forEach(btn=>{btn.onclick=function(){alert('Excel raporu oluşturuluyor...\n\nGerçek uygulamada Excel dosyası indirilecek.');};});
