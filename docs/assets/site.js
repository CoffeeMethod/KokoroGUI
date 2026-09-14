// Shared across every docs page: the light/dark toggle (persisted in
// localStorage), the mobile nav button, and copy buttons on code blocks.
(function(){
  function initTheme(){
    var root = document.documentElement;
    var stored = null;
    try { stored = localStorage.getItem('kg-theme'); } catch (e) {}
    if (stored) root.setAttribute('data-theme', stored);

    var btn = document.getElementById('themeToggle');
    if (!btn) return;
    var sun = document.getElementById('iconSun');
    var moon = document.getElementById('iconMoon');

    function isDark(){
      var t = root.getAttribute('data-theme');
      if (t) return t === 'dark';
      return !window.matchMedia('(prefers-color-scheme: light)').matches;
    }
    function syncIcons(){
      if (!sun || !moon) return;
      sun.style.display = isDark() ? 'block' : 'none';
      moon.style.display = isDark() ? 'none' : 'block';
    }
    syncIcons();
    btn.addEventListener('click', function(){
      var next = isDark() ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('kg-theme', next); } catch (e) {}
      syncIcons();
    });
  }

  function initMenu(){
    var btn = document.getElementById('menuToggle');
    var header = document.querySelector('header.site');
    if (!btn || !header) return;
    btn.addEventListener('click', function(){
      var open = header.classList.toggle('open');
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    header.querySelectorAll('.navlinks a').forEach(function(a){
      a.addEventListener('click', function(){ header.classList.remove('open'); });
    });
  }

  function initCopy(){
    if (!navigator.clipboard) return;
    document.querySelectorAll('.code-block[data-copy]').forEach(function(block){
      var head = block.querySelector('.head');
      var pre = block.querySelector('pre');
      if (!head || !pre) return;
      var btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.type = 'button';
      btn.textContent = 'Copy';
      btn.addEventListener('click', function(){
        // Copy the commands only: drop comment lines and the "$ " prompt.
        var text = pre.innerText.split('\n').filter(function(l){
          return l.trim() && l.trim().charAt(0) !== '#';
        }).map(function(l){ return l.replace(/^\$\s+/, '').replace(/\s+#.*$/, ''); }).join('\n');
        navigator.clipboard.writeText(text).then(function(){
          btn.textContent = 'Copied';
          btn.classList.add('done');
          setTimeout(function(){ btn.textContent = 'Copy'; btn.classList.remove('done'); }, 1600);
        });
      });
      head.appendChild(btn);
    });
  }

  function init(){ initTheme(); initMenu(); initCopy(); }
  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
