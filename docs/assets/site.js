// Shared across every docs page: light/dark toggle, persisted in localStorage.
(function(){
  function init(){
    var root = document.documentElement;
    var stored = localStorage.getItem('kg-theme');
    if (stored) root.setAttribute('data-theme', stored);

    var btn = document.getElementById('themeToggle');
    if (!btn) return;
    var sun = document.getElementById('iconSun');
    var moon = document.getElementById('iconMoon');

    function isDark(){
      return root.getAttribute('data-theme') === 'dark' ||
        (!root.getAttribute('data-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    }
    function syncIcons(){
      if (!sun || !moon) return;
      sun.style.display = isDark() ? 'none' : 'block';
      moon.style.display = isDark() ? 'block' : 'none';
    }
    syncIcons();
    btn.addEventListener('click', function(){
      var next = isDark() ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      localStorage.setItem('kg-theme', next);
      syncIcons();
    });
  }
  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
