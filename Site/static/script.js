// ════════════════════════════════════════════════════════════
//  SineBot — script.js
// ════════════════════════════════════════════════════════════

const TMDB_KEY = '8265bd1679663a7ea12ac168da84d2e8';
const PAGE_SIZE = 24;

// ── MOBILE MENU (global — used by all pages) ─────────────────
function toggleMobileMenu() {
  const menu = document.getElementById('mobileMenu');
  const btn  = document.getElementById('navHamburger');
  if (!menu) return;
  const open = menu.classList.contains('open');
  if (open) { closeMobileMenu(); }
  else       { menu.classList.add('open'); btn?.classList.add('open'); }
}
function closeMobileMenu() {
  document.getElementById('mobileMenu')?.classList.remove('open');
  document.getElementById('navHamburger')?.classList.remove('open');
}

// ── POSTER CACHE & ASYNC LOADER ──────────────────────────────
const _posterCache = {};

async function loadPosterAsync(movieId, tmdbId) {
  if (!tmdbId) return;
  const key = String(tmdbId);

  if (_posterCache[key] !== undefined) {
    setPosterEl(movieId, _posterCache[key]);
    return;
  }
  try {
    const r = await fetch(
      `https://api.themoviedb.org/3/movie/${tmdbId}?api_key=${TMDB_KEY}&language=tr-TR`,
      { signal: AbortSignal.timeout(6000) }
    );
    if (r.ok) {
      const d = await r.json();
      const url = d.poster_path ? `https://image.tmdb.org/t/p/w342${d.poster_path}` : '';
      _posterCache[key] = url;
      setPosterEl(movieId, url);
    } else { _posterCache[key] = ''; }
  } catch { _posterCache[key] = ''; }
}

function setPosterEl(movieId, url) {
  const wrap = document.getElementById(`poster-${movieId}`);
  if (!wrap) return;
  if (url) {
    const img = document.createElement('img');
    img.src = url; img.alt = 'poster'; img.loading = 'lazy';
    img.style.cssText = 'width:100%;height:100%;object-fit:cover;opacity:0;transition:opacity .4s';
    img.onload  = () => { img.style.opacity = '1'; };
    img.onerror = () => { wrap.innerHTML = posterFallback('🎬'); };
    // remove fallback
    const fb = wrap.querySelector('.film-pfallback');
    if (fb) fb.remove();
    wrap.insertBefore(img, wrap.firstChild);
  }
}

function posterFallback(icon = '🎬') {
  return `<div class="film-pfallback"><span>${icon}</span><span>Poster Yok</span></div>`;
}

// ── BUILD FILM CARD ───────────────────────────────────────────
function buildFilmCard(movie, index = 0, globalIndex = 0) {
  const card = document.createElement('div');
  card.className = 'film-card';
  card.style.animationDelay = `${(index % PAGE_SIZE) * 0.04}s`;
  card.setAttribute('id', `card-${movie.movieId}`);

  const posterUrl  = movie.poster_url || '';
  const rating     = movie.avg_rating ? movie.avg_rating : null;
  const year       = movie.year || '';
  const genres     = (movie.genre_names || []).slice(0, 2);

  // rank badge
  let rankHtml = '';
  if (globalIndex === 0)      rankHtml = `<div class="film-rank r1">1</div>`;
  else if (globalIndex === 1) rankHtml = `<div class="film-rank r2">2</div>`;
  else if (globalIndex === 2) rankHtml = `<div class="film-rank r3">3</div>`;
  else                        rankHtml = `<div class="film-rank rn">${globalIndex+1}</div>`;

  const ratingHtml = rating
    ? `<div class="film-imdb">⭐ ${rating}</div>` : '';

  const posterInner = posterUrl
    ? `<img src="${posterUrl}" alt="${movie.title}" loading="lazy"
           onerror="this.parentNode.innerHTML='${posterFallback()}';">`
    : posterFallback();

  const tagHtml = genres.map(g =>
    `<span class="film-tag tag-genre">${g}</span>`
  ).join('') + (year ? `<span class="film-tag tag-year">${year}</span>` : '');

  card.innerHTML = `
    <div class="film-poster" id="poster-${movie.movieId}">
      ${posterInner}
      <div class="film-poster-gradient"></div>
      <div class="film-top">
        ${rankHtml}
        ${ratingHtml}
      </div>
      <div class="film-bottom">
        ${year ? `<span class="film-year-pill">${year}</span>` : ''}
      </div>
      <div class="film-hover-overlay">
        <div class="play-btn">▶</div>
      </div>
    </div>
    <div class="film-info">
      <div class="film-title">${movie.title}</div>
      <div class="film-tags">${tagHtml}</div>
    </div>
  `;

  card.addEventListener('click', () => {
    window.location.href = `/movie/${movie.movieId}`;
  });

  // async poster if missing
  if (!posterUrl && movie.tmdbId) {
    loadPosterAsync(movie.movieId, movie.tmdbId);
  }

  return card;
}

// ════════════════════════════════════════════════════════════
//  HOME PAGE — BROWSE / KEŞFET
// ════════════════════════════════════════════════════════════

// Filter state
const filterState = {
  genres: [],       // [] = all
  year:   2010,
  minScore: 0,
  maxScore: 10,
  offset: 0
};

let browseOffset = 0;
let browseHasMore = false;

async function loadFilms(reset = true) {
  const grid = document.getElementById('filmGrid');
  if (!grid) return;

  if (reset) {
    browseOffset = 0;
    grid.innerHTML = `<div class="grid-loading"><div class="spinner"></div><div class="grid-loading-txt">Filmler yükleniyor...</div></div>`;
    const lmw = document.getElementById('loadMoreWrap');
    if (lmw) lmw.style.display = 'none';
  }

  try {
    const body = {
      preferred_year:   filterState.year || 2010,
      strict_year_filter: filterState.year !== 0,
      selected_genres:  filterState.genres,
      min_score:        filterState.minScore,
      max_score:        filterState.maxScore,
      offset:           browseOffset,
      limit:            PAGE_SIZE,
      selected_movie_id: null
    };

    const r = await fetch('/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await r.json();

    if (!r.ok || !data.success) throw new Error(data.error || 'Hata');

    const films = data.recommendations || [];

    if (reset) grid.innerHTML = '';

    if (reset && films.length === 0) {
      grid.innerHTML = `
        <div class="grid-empty">
          <div class="grid-empty-icon">🎬</div>
          <div class="grid-empty-t">Film Bulunamadı</div>
          <div class="grid-empty-s">Filtreleri değiştirerek tekrar deneyin.</div>
        </div>`;
      return;
    }

    films.forEach((film, i) => {
      const card = buildFilmCard(film, i, browseOffset + i);
      grid.appendChild(card);
    });

    const cnt = document.getElementById('filmCount');
    if (cnt) cnt.textContent = `${browseOffset + films.length} film`;

    browseHasMore = data.has_more && films.length === PAGE_SIZE;
    const lmw = document.getElementById('loadMoreWrap');
    if (lmw) lmw.style.display = browseHasMore ? 'block' : 'none';

  } catch (err) {
    console.error('Film yükleme hatası:', err);
    if (reset) {
      grid.innerHTML = `
        <div class="grid-empty">
          <div class="grid-empty-icon">⚠️</div>
          <div class="grid-empty-t">Yükleme Hatası</div>
          <div class="grid-empty-s">${err.message}</div>
        </div>`;
    }
  }
}

async function loadMoreFilms() {
  browseOffset += PAGE_SIZE;
  await loadFilms(false);
}

function reloadFilms() {
  loadFilms(true);
}

// ── FILTER DROPDOWN LOGIC ─────────────────────────────────────
function toggleFdd(id) {
  const el = document.getElementById(id);
  if (!el) return;
  const wasOpen = el.classList.contains('open');
  document.querySelectorAll('.fdd.open').forEach(d => d.classList.remove('open'));
  if (!wasOpen) el.classList.add('open');
}

// Genre filter
let selectedGenres = [];
function initGenreFilter() {
  document.querySelectorAll('#genreMenuScroll .fdd-item').forEach(item => {
    item.addEventListener('click', () => {
      const gId = item.dataset.genreId;
      if (gId === '' || gId === undefined) {
        // "Hepsi" selected
        selectedGenres = [];
        document.querySelectorAll('#genreMenuScroll .fdd-item').forEach(i => i.classList.remove('sel'));
        document.getElementById('genre-all')?.classList.add('sel');
        document.getElementById('fddTurlerVal').textContent = 'Hepsi';
      } else {
        // Toggle genre
        document.getElementById('genre-all')?.classList.remove('sel');
        if (selectedGenres.includes(gId)) {
          selectedGenres = selectedGenres.filter(g => g !== gId);
          item.classList.remove('sel');
        } else {
          selectedGenres.push(gId);
          item.classList.add('sel');
        }
        if (selectedGenres.length === 0) {
          document.getElementById('genre-all')?.classList.add('sel');
          document.getElementById('fddTurlerVal').textContent = 'Hepsi';
        } else {
          document.getElementById('fddTurlerVal').textContent =
            selectedGenres.length === 1 ? item.textContent.trim().replace('✓','').trim()
            : `${selectedGenres.length} tür seçili`;
        }
      }
      filterState.genres = selectedGenres;
      reloadFilms();
    });
  });
}

function resetGenreFilter() {
  selectedGenres = [];
  filterState.genres = [];
  document.querySelectorAll('#genreMenuScroll .fdd-item').forEach(i => i.classList.remove('sel'));
  document.getElementById('genre-all')?.classList.add('sel');
  document.getElementById('fddTurlerVal').textContent = 'Hepsi';
  document.getElementById('fddTurler')?.classList.remove('open');
  reloadFilms();
}

// Year filter
function initYearFilter() {
  document.querySelectorAll('#fddYillar .fdd-item').forEach(item => {
    item.addEventListener('click', () => {
      document.querySelectorAll('#fddYillar .fdd-item').forEach(i => i.classList.remove('sel'));
      item.classList.add('sel');
      const yr = parseInt(item.dataset.year);
      filterState.year = yr || 2010;
      document.getElementById('fddYillarVal').textContent = item.dataset.label || 'Hepsi';
      document.getElementById('fddYillar').classList.remove('open');
      reloadFilms();
    });
  });
}

// Score filter
function initScoreFilter() {
  document.querySelectorAll('#fddPuan .fdd-item').forEach(item => {
    item.addEventListener('click', () => {
      document.querySelectorAll('#fddPuan .fdd-item').forEach(i => i.classList.remove('sel'));
      item.classList.add('sel');
      const mn = parseFloat(item.dataset.min) || 0;
      filterState.minScore = mn;
      document.getElementById('fddPuanVal').textContent = item.dataset.label || 'Hepsi';
      document.getElementById('fddPuan').classList.remove('open');
      reloadFilms();
    });
  });
}

function clearAllFilters() {
  selectedGenres = [];
  filterState.genres   = [];
  filterState.year     = 2010;
  filterState.minScore = 0;
  filterState.maxScore = 10;

  // Reset all fdd-item selections
  document.querySelectorAll('.fdd-item').forEach(i => i.classList.remove('sel'));
  document.querySelectorAll('.fdd-item[data-genre-id=""]').forEach(i => i.classList.add('sel'));
  document.querySelectorAll('.fdd-item[data-year="0"]').forEach(i => i.classList.add('sel'));
  document.querySelectorAll('.fdd-item[data-min="0"]').forEach(i => i.classList.add('sel'));
  document.querySelectorAll('.fdd-item[data-sort="recommend"]').forEach(i => i.classList.add('sel'));

  document.getElementById('fddTurlerVal').textContent   = 'Hepsi';
  document.getElementById('fddYillarVal').textContent   = 'Hepsi';
  document.getElementById('fddPuanVal').textContent     = 'Hepsi';
  document.getElementById('fddSiralamaVal').textContent = 'Öneriye Göre';

  reloadFilms();
}

// ── NAV SEARCH ────────────────────────────────────────────────
let _navSearchTimer = null;

function initNavSearch() {
  const inp  = document.getElementById('navSearchInput');
  const drop = document.getElementById('navSearchDrop');
  if (!inp || !drop) return;

  inp.addEventListener('input', function () {
    const q = this.value.trim();
    clearTimeout(_navSearchTimer);
    drop.style.display = 'none';
    drop.innerHTML = '';
    if (q.length < 5) return;

    _navSearchTimer = setTimeout(async () => {
      try {
        const r = await fetch(`/api/search_movies?q=${encodeURIComponent(q)}`);
        const d = await r.json();
        renderNavSearch(d.results || []);
      } catch {}
    }, 280);
  });

  function renderNavSearch(results) {
    drop.innerHTML = '';
    if (!results.length) {
      drop.innerHTML = '<div class="nsd-empty">Sonuç bulunamadı</div>';
      drop.style.display = 'block';
      return;
    }
    results.forEach(m => {
      const tags = (m.genre_names || []).slice(0,2)
        .map(g => `<span class="film-tag tag-genre" style="font-size:.6rem;">${g}</span>`).join('');
      const item = document.createElement('div');
      item.className = 'nsd-item';
      item.innerHTML = `
        <div style="flex:1;min-width:0;">
          <div class="nsd-title">${m.title}</div>
          <div class="nsd-meta"><span>${m.year}</span>${tags}</div>
        </div>
        ${m.avg_rating ? `<span class="nsd-rating">⭐ ${m.avg_rating}</span>` : ''}
      `;
      item.addEventListener('click', () => {
        window.location.href = `/movie/${m.movieId}`;
      });
      drop.appendChild(item);
    });
    drop.style.display = 'block';
  }
}

// ════════════════════════════════════════════════════════════
//  RECOMMENDATIONS PAGE — form + results
// ════════════════════════════════════════════════════════════
let recOffset = 0;
let recFormData = null;
const REC_LIMIT = 12;

// Movie search (rec page)
let _recSearchTimer = null;

function initMovieSearch() {
  const inp    = document.getElementById('movieSearch');
  const drop   = document.getElementById('movieSearchDropdown');
  const hidden = document.getElementById('selectedMovieId');
  const badge  = document.getElementById('selectedMovieBadge');
  const name   = document.getElementById('selectedMovieName');
  const rmBtn  = document.getElementById('removeSelectedMovie');

  if (!inp) return;

  inp.addEventListener('input', function () {
    const q = this.value.trim();
    clearTimeout(_recSearchTimer);
    drop.style.display = 'none'; drop.innerHTML = '';
    if (q.length < 5) return;

    _recSearchTimer = setTimeout(async () => {
      try {
        const r = await fetch(`/api/search_movies?q=${encodeURIComponent(q)}`);
        const d = await r.json();
        renderRecSearch(d.results || []);
      } catch {}
    }, 280);
  });

  function renderRecSearch(results) {
    drop.innerHTML = '';
    if (!results.length) {
      drop.innerHTML = '<div class="msd-empty">Sonuç bulunamadı</div>';
      drop.style.display = 'block'; return;
    }
    results.forEach(m => {
      const tags = (m.genre_names || []).slice(0,3)
        .map(g => `<span class="film-tag tag-genre">${g}</span>`).join('');
      const item = document.createElement('div');
      item.className = 'msd-item';
      item.innerHTML = `
        <div style="flex:1;min-width:0;">
          <div class="msd-title">${m.title}</div>
          <div class="msd-meta"><span>${m.year}</span>${tags}</div>
        </div>
        ${m.avg_rating ? `<span class="msd-rating">⭐ ${m.avg_rating}</span>` : ''}
      `;
      item.addEventListener('click', () => {
        hidden.value = m.movieId;
        name.textContent = `${m.title} (${m.year})`;
        badge.style.display = 'inline-flex';
        inp.style.display = 'none';
        drop.style.display = 'none'; drop.innerHTML = '';
      });
      drop.appendChild(item);
    });
    drop.style.display = 'block';
  }

  rmBtn?.addEventListener('click', () => {
    hidden.value = '';
    badge.style.display = 'none';
    inp.style.display = ''; inp.value = ''; inp.focus();
  });

  document.addEventListener('click', e => {
    if (!e.target.closest('#movieSearchWrapper')) drop.style.display = 'none';
  });
}

// Form submit
const _recForm = document.getElementById('recommendForm');
if (_recForm) {
  _recForm.addEventListener('submit', async function (e) {
    e.preventDefault();
    const fd = new FormData(this);
    const genres = fd.getAll('genres');
    if (!genres.length) { alert('Lütfen en az bir film türü seçin!'); return; }
    if (genres.length > 5) { alert('En fazla 5 tür seçebilirsiniz!'); return; }

    const selMovId = document.getElementById('selectedMovieId')?.value || '';

    recFormData = {
      preferred_year:    parseInt(document.getElementById('preferred_year').value) || 2010,
      selected_genres:   genres,
      min_score:         parseFloat(document.getElementById('minScore')?.value || 0),
      max_score:         parseFloat(document.getElementById('maxScore')?.value || 10),
      selected_movie_id: selMovId ? parseInt(selMovId) : null
    };
    recOffset = 0;

    document.getElementById('formSection').style.display = 'none';
    document.getElementById('loadingIndicator').style.display = 'flex';
    document.getElementById('errorMessage').style.display = 'none';
    document.getElementById('recommendationsList').innerHTML = '';

    await fetchRec();
  });
}

async function fetchRec() {
  try {
    const r = await fetch('/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...recFormData, offset: recOffset, limit: REC_LIMIT })
    });
    const data = await r.json();
    if (!r.ok || !data.success) throw new Error(data.error || 'Hata');
    displayRecResults(data.recommendations, recOffset === 0);
    const lm = document.getElementById('loadMoreRecContainer');
    if (lm) lm.style.display = (data.has_more && recOffset < REC_LIMIT) ? 'block' : 'none';
  } catch (err) {
    showRecError(err.message);
  } finally {
    document.getElementById('loadingIndicator').style.display = 'none';
  }
}

async function loadMoreRec() {
  recOffset += REC_LIMIT;
  document.getElementById('loadMoreRecContainer').style.display = 'none';
  await fetchRec();
}

function displayRecResults(films, isFirst) {
  const grid = document.getElementById('recommendationsList');
  if (isFirst) grid.innerHTML = '';
  if (isFirst && !films.length) { showRecError('Hiç film bulunamadı. Kriterleri değiştirin.'); return; }

  films.forEach((film, i) => {
    const card = buildFilmCard(film, i, recOffset + i);
    grid.appendChild(card);
  });

  const cnt = document.getElementById('resultsCount');
  if (cnt) cnt.textContent = `${recOffset + films.length} film`;

  document.getElementById('resultsSection').style.display = 'block';
  if (isFirst) {
    setTimeout(() => {
      document.getElementById('resultsSection')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }
}

function showRecError(msg) {
  const box = document.getElementById('errorMessage');
  const txt = document.getElementById('errorText');
  if (txt) txt.textContent = msg;
  if (box) box.style.display = 'block';
  document.getElementById('formSection').style.display = 'none';
  document.getElementById('resultsSection').style.display = 'none';
}

function resetRecForm() {
  document.getElementById('formSection').style.display = 'block';
  document.getElementById('resultsSection').style.display = 'none';
  document.getElementById('errorMessage').style.display = 'none';
  document.getElementById('loadingIndicator').style.display = 'none';
  document.getElementById('recommendationsList').innerHTML = '';
  document.getElementById('recommendForm')?.reset();
  document.querySelectorAll('.genre-chip').forEach(c => c.classList.remove('sel'));
  document.querySelectorAll('.genre-input').forEach(c => { c.checked = false; });
  const mb = document.getElementById('selectedMovieBadge');
  const si = document.getElementById('movieSearch');
  if (mb) mb.style.display = 'none';
  if (si) { si.style.display = ''; si.value = ''; }
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── INIT ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Home page
  const grid = document.getElementById('filmGrid');
  if (grid) {
    initGenreFilter();
    initYearFilter();
    initScoreFilter();
    loadFilms(true);
  }

  // Rec page
  initMovieSearch();
  initNavSearch();

  // Health check
  fetch('/health').then(r => r.json()).then(d => console.log('✓ API:', d)).catch(() => {});
});
