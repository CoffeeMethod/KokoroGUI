// Shared across every docs page: the light/dark toggle (persisted in
// localStorage), the mobile nav button, copy buttons on code blocks, tab
// groups, and the transport bar along the bottom of the window.
(function(){
  function store(key, value){
    try {
      if (value === undefined) return localStorage.getItem(key);
      localStorage.setItem(key, value);
    } catch (e) {}
    return null;
  }

  function initTheme(){
    var root = document.documentElement;
    var stored = store('kg-theme');
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
      store('kg-theme', next);
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

  // role="tablist" groups: click or arrow keys pick a tab, the chosen one
  // is remembered per group (data-remember) for the next visit.
  function initTabs(){
    document.querySelectorAll('[role="tablist"]').forEach(function(list){
      var tabs = Array.prototype.slice.call(list.querySelectorAll('[role="tab"]'));
      var key = list.getAttribute('data-remember');
      function select(tab, focus){
        tabs.forEach(function(t){
          var on = t === tab;
          t.setAttribute('aria-selected', on ? 'true' : 'false');
          t.tabIndex = on ? 0 : -1;
          var panel = document.getElementById(t.getAttribute('aria-controls'));
          if (panel) panel.hidden = !on;
        });
        if (focus) tab.focus();
        if (key) store(key, tab.id);
      }
      tabs.forEach(function(tab, i){
        tab.addEventListener('click', function(){ select(tab, false); });
        tab.addEventListener('keydown', function(e){
          var next = null;
          if (e.key === 'ArrowRight') next = tabs[(i + 1) % tabs.length];
          if (e.key === 'ArrowLeft') next = tabs[(i - 1 + tabs.length) % tabs.length];
          if (e.key === 'Home') next = tabs[0];
          if (e.key === 'End') next = tabs[tabs.length - 1];
          if (next){ e.preventDefault(); select(next, true); }
        });
      });
      var remembered = key && document.getElementById(store(key) || '');
      select(tabs.indexOf(remembered) >= 0 ? remembered : tabs[0], false);
    });
  }

  // The transport: one lane, one clip per <main> section, sized by its
  // height. The playhead is the scroll position. The clock is how far into
  // the page you'd be if it were read aloud at 150 words a minute, as
  // HH:MM:SS:FF at 25 fps. Play scrolls the page at that pace.
  var WORDS_PER_SECOND = 2.5;
  var COLORS = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'];

  function timecode(seconds){
    var frames = Math.floor(seconds * 25);
    var f = frames % 25, s = Math.floor(frames / 25);
    function two(n){ return (n < 10 ? '0' : '') + n; }
    return two(Math.floor(s / 3600)) + ':' + two(Math.floor(s / 60) % 60) + ':' + two(s % 60) + ':' + two(f);
  }

  // A section's clip name: data-clip, else a numbered label's name
  // ("01 · Syntax" gives "Syntax"), else its heading.
  function clipName(section){
    if (section.dataset.clip) return section.dataset.clip;
    var idx = section.querySelector('h2 .idx');
    var numbered = idx && idx.textContent.match(/^\s*\d+\s*·\s*(.+)$/);
    if (numbered) return numbered[1].trim();
    var h = section.querySelector('h1, h2');
    var text = h ? h.textContent : '';
    if (idx) text = text.replace(idx.textContent, '');
    return (text || section.id || '').trim();
  }

  function initTransport(){
    var main = document.querySelector('main');
    if (!main) return;
    var sections = Array.prototype.slice.call(main.querySelectorAll(':scope > section'));
    if (sections.length < 2) return;

    var bar = document.createElement('div');
    bar.className = 'transport';
    bar.setAttribute('role', 'region');
    bar.setAttribute('aria-label', 'Page timeline');
    bar.innerHTML =
      '<button class="tp-btn" type="button" aria-label="Read the page at speaking pace">' +
        '<svg class="i-play" viewBox="0 0 10 10"><path d="M2 1l7 4-7 4z" fill="currentColor"/></svg>' +
        '<svg class="i-pause" viewBox="0 0 10 10" style="display:none"><path d="M2 1h2v8H2zM6 1h2v8H6z" fill="currentColor"/></svg>' +
      '</button>' +
      '<span class="tp-clock" aria-live="off">00:00:00:00</span>' +
      '<div class="tp-lane"></div>' +
      '<span class="tp-label">read aloud · 150 wpm</span>';
    document.body.appendChild(bar);
    var lane = bar.querySelector('.tp-lane');
    var clock = bar.querySelector('.tp-clock');
    var btn = bar.querySelector('.tp-btn');
    var head = document.createElement('div');
    head.className = 'tp-head';

    var clips = sections.map(function(section, i){
      var words = (section.innerText || '').split(/\s+/).filter(Boolean).length;
      // A section without data-cc gets the next color, so its label dot
      // and its clip on the bar match.
      if (!section.dataset.cc) section.dataset.cc = COLORS[i % COLORS.length];
      var el = document.createElement('button');
      el.type = 'button';
      el.className = 'tp-clip ' + section.dataset.cc;
      var name = clipName(section);
      el.textContent = name;
      el.title = name;
      el.addEventListener('click', function(){ section.scrollIntoView({ behavior: 'smooth' }); });
      lane.appendChild(el);
      return { section: section, el: el, words: words, top: 0, height: 1, start: 0 };
    });
    lane.appendChild(head);

    function measure(){
      var total = 0;
      clips.forEach(function(c){
        var r = c.section.getBoundingClientRect();
        c.top = r.top + window.scrollY;
        c.height = Math.max(1, r.height);
        c.start = total;
        total += c.words / WORDS_PER_SECOND;
        c.el.style.flex = c.height + ' 1 0';
      });
      update();
    }

    // Where the reading line sits: the top of the page when scrolled to the
    // top, the bottom when scrolled to the bottom, in between otherwise.
    function readingY(){
      var max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      return window.scrollY + window.innerHeight * Math.min(1, window.scrollY / max);
    }

    function update(){
      var y = readingY();
      var current = clips[0], frac = 0;
      for (var i = 0; i < clips.length; i++){
        if (y >= clips[i].top) current = clips[i];
      }
      frac = Math.min(1, Math.max(0, (y - current.top) / current.height));
      var secs = current.start + frac * current.words / WORDS_PER_SECOND;
      clock.textContent = timecode(secs);
      clips.forEach(function(c){ c.el.classList.toggle('on', c === current); });
      var laneRect = lane.getBoundingClientRect();
      var elRect = current.el.getBoundingClientRect();
      head.style.left = (elRect.left - laneRect.left + frac * elRect.width) + 'px';
    }

    // A frame's step is often under a pixel, so the position is kept as a
    // float here and written with an instant scroll (the page's smooth
    // scrolling would otherwise animate every frame).
    var playing = false, last = 0, pos = 0;
    function setPlaying(on){
      playing = on;
      bar.querySelector('.i-play').style.display = on ? 'none' : 'block';
      bar.querySelector('.i-pause').style.display = on ? 'block' : 'none';
      btn.setAttribute('aria-label', on ? 'Pause' : 'Read the page at speaking pace');
      if (on){ last = 0; pos = window.scrollY; requestAnimationFrame(step); }
    }
    function step(t){
      if (!playing) return;
      if (last){
        var y = readingY(), current = clips[0];
        for (var i = 0; i < clips.length; i++) if (y >= clips[i].top) current = clips[i];
        var max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
        // The reading line moves faster than the scroll position (see
        // readingY), so slow the scroll by that ratio.
        var ratio = 1 + window.innerHeight / max;
        var pxPerSec = current.height / Math.max(1, current.words / WORDS_PER_SECOND) / ratio;
        pos = Math.min(max, pos + pxPerSec * (t - last) / 1000);
        window.scrollTo({ top: pos, behavior: 'instant' });
        update();
        if (pos >= max){ setPlaying(false); return; }
      }
      last = t;
      requestAnimationFrame(step);
    }
    btn.addEventListener('click', function(){ setPlaying(!playing); });
    ['wheel', 'touchstart', 'keydown'].forEach(function(ev){
      window.addEventListener(ev, function(e){
        if (!playing) return;
        if (ev === 'keydown' && e.target === btn) return;
        setPlaying(false);
      }, { passive: true });
    });

    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', measure);
    window.addEventListener('load', measure);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(measure);
    measure();
  }

  function init(){ initTheme(); initMenu(); initCopy(); initTabs(); initTransport(); }
  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
