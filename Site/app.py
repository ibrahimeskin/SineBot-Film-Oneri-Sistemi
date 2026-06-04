"""
Film Öneri Web Sitesi
Kullanıcı özellikleri alır ve 10 film önerir
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import warnings
import requests as http_requests

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(__file__)
ENV_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', '.env'))
load_dotenv(ENV_PATH)

app = Flask(__name__, template_folder='templates', static_folder='static')
OMDB_API_KEY = os.getenv('OMDB_API_KEY', '').strip()

# Poster URL cache (imdbId -> poster_url)
_poster_cache = {}

# ─────────────────────────────────────────────
# MODELİ YÜKLE
# ─────────────────────────────────────────────
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'ModelResult', 'best_model.joblib'))
MOVIES_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'ProcessedData', 'movies_cleaned.csv'))
REPORT_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'ModelResult', 'rapor.csv'))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")
if not os.path.exists(MOVIES_PATH):
    raise FileNotFoundError(f"Film verisi bulunamadı: {MOVIES_PATH}")

# Model bundle'ını yükle
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle['model']
scaler = model_bundle['scaler']
feature_cols = model_bundle['feature_cols']
best_name = model_bundle['best_name']

# Film verisini yükle
movies_df = pd.read_csv(MOVIES_PATH, dtype={'movieId': 'int32'})
movies_df = movies_df.dropna(subset=['title', 'genre_ids'])

# Film istatistiklerini yükle (Puan aralığı filtresi için)
STATS_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'ModelResult', 'movie_stats.csv'))
if os.path.exists(STATS_PATH):
    stats_df = pd.read_csv(STATS_PATH)
    movies_df = movies_df.merge(stats_df[['movieId', 'avg_movie_rating']], on='movieId', how='left')
    movies_df['avg_movie_rating'] = movies_df['avg_movie_rating'].fillna(3.5) # Varsayılan ortalama puan
else:
    movies_df['avg_movie_rating'] = 3.5

# Linkleri yükle
LINKS_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'RawData', 'links.csv'))
if os.path.exists(LINKS_PATH):
    links_df = pd.read_csv(LINKS_PATH)
    movies_df = movies_df.merge(links_df, on='movieId', how='left')
else:
    movies_df['imdbId'] = np.nan
    movies_df['tmdbId'] = np.nan

print(f"✓ Model yüklendi: {best_name}")
print(f"✓ {len(movies_df):,} film yüklendi")
print(f"✓ Özellikler: {', '.join(feature_cols[:5])}...")

# ─────────────────────────────────────────────
# GENRE'LER VE İSTATİSTİKLER
# ─────────────────────────────────────────────
# dartaClean.py içindeki alfabetik genre sırasına göre ID -> ad eşlemesi
GENRE_ID_TO_NAME = {
    '1': 'Action',    '2': 'Adventure',  '3': 'Animation',
    '4': 'Children',  '5': 'Comedy',     '6': 'Crime',
    '7': 'Documentary','8': 'Drama',     '9': 'Fantasy',
    '10': 'Film-Noir','11': 'Horror',    '12': 'IMAX',
    '13': 'Musical',  '14': 'Mystery',   '15': 'Romance',
    '16': 'Sci-Fi',   '17': 'Thriller',  '18': 'War',
    '19': 'Western',
}

# Genre ID'leri çıkar
all_genres = set()
for genre_str in movies_df['genre_ids'].fillna('0'):
    if str(genre_str) != '0' and str(genre_str) != 'nan':
        all_genres.update(str(genre_str).split('|'))
all_genres = sorted([g for g in all_genres if g])
AVAILABLE_GENRES = all_genres

# Genre listesi: (id, name) çiftleri — frontend'e gönderilecek
GENRE_LIST = [
    {'id': g, 'name': GENRE_ID_TO_NAME.get(g, f'Genre {g}')}
    for g in AVAILABLE_GENRES
]

# İstenen üzere 100'e kadar tür eklemek için gerçekçi alt-tür ve kategoriler listesi
ekstra_turler = [
    "Cyberpunk", "Steampunk", "Post-Apocalyptic", "Zombie", "Space Opera",
    "Martial Arts", "Superhero", "Spy", "Neo-Noir", "Detective",
    "Historical Fiction", "Biographical", "Epic", "Experimental", "Mockumentary",
    "Slasher", "Psychological Thriller", "Dark Comedy", "Romantic Comedy", "Slapstick",
    "Parody", "Satire", "Legal Drama", "Medical Drama", "Teen",
    "Coming-of-Age", "Found Footage", "Monster", "Disaster", "Survival",
    "Heist", "Road Movie", "Buddy Cop", "Political Thriller", "Erotic Thriller",
    "Supernatural", "Occult", "Magic Realism", "Cyber Thriller", "Techno-Thriller",
    "Time Travel", "Alternate History", "Dystopian", "Utopian", "Space Western",
    "Mecha", "Kaiju", "Samurai", "Ninja", "Wuxia",
    "Giallo", "Spaghetti Western", "Acid Western", "Revisionist Western", "Musical Comedy",
    "Rockumentary", "Concert Film", "Dance", "Sports", "eSports",
    "Racing", "Biker", "Surfing", "Skateboarding", "High School",
    "College", "Workplace", "Family", "Holiday", "Christmas",
    "Halloween", "Thanksgiving", "Wedding", "Pregnancy", "Parenting",
    "Aging", "Ghosts", "Vampires", "Werewolves", "Witches",
    "Demons", "Angels", "Mythology", "Fairy Tale", "Urban Legend",
    "Conspiracy", "Cult", "Independent", "Avant-Garde", "Surrealist",
    "Absurdist", "Silent Film", "Stop Motion", "Claymation", "Puppetry"
]

current_genre_count = len(GENRE_LIST)
for i in range(100 - current_genre_count):
    mock_id = f'mock_{i + current_genre_count + 1}'
    tur_adi = ekstra_turler[i] if i < len(ekstra_turler) else f"Özel Tür {i}"
    GENRE_LIST.append({'id': mock_id, 'name': tur_adi})

print(f"✓ Mevcut türler: {', '.join(AVAILABLE_GENRES[:10])}...")

# ─────────────────────────────────────────────
# RAPOR OKUMA (CSV)
# ─────────────────────────────────────────────
def load_report_data():
    if not os.path.exists(REPORT_PATH):
        return {
            'general_stats': {},
            'most_rated': [],
            'top_avg': []
        }

    report_df = pd.read_csv(REPORT_PATH)

    def format_int(value):
        try:
            if pd.isna(value):
                return '—'
            return f"{int(float(value)):,}"
        except Exception:
            return str(value)

    def format_float(value, digits=3):
        try:
            if pd.isna(value):
                return '—'
            return f"{float(value):.{digits}f}"
        except Exception:
            return str(value)

    general_df = report_df[report_df['bolum'] == 'Genel Istatistikler']
    general_df = general_df.dropna(subset=['metrik'])
    general_stats = {
        str(row['metrik']): format_int(row['deger'])
        for _, row in general_df.iterrows()
    }

    most_rated_df = report_df[report_df['bolum'] == 'En Cok Puanlanan 10 Film']
    most_rated_df = most_rated_df.dropna(subset=['title'])
    most_rated = []
    for _, row in most_rated_df[['sira', 'title', 'puan_sayisi']].iterrows():
        movie_match = movies_df[movies_df['title'] == row['title']]
        poster_url = ''
        tmdb_id_val = ''
        if not movie_match.empty:
            m = movie_match.iloc[0]
            imdb_id = str(m['imdbId']).split('.')[0].zfill(7) if pd.notna(m.get('imdbId')) else ''
            tmdb_id_val = str(m['tmdbId']).split('.')[0] if pd.notna(m.get('tmdbId')) else ''
            poster_url = get_poster_url(imdb_id, tmdb_id_val)
            
        most_rated.append({
            'movieId': int(m['movieId']) if not movie_match.empty else None,
            'sira': format_int(row['sira']),
            'title': row['title'],
            'puan_sayisi': format_int(row['puan_sayisi']),
            'poster_url': poster_url,
            'tmdb_id': tmdb_id_val
        })

    top_avg_df = report_df[report_df['bolum'] == 'En Yuksek Ortalama Puan']
    top_avg_df = top_avg_df.dropna(subset=['title'])
    top_avg = []
    for _, row in top_avg_df[['sira', 'title', 'ortalama_puan', 'puan_sayisi']].iterrows():
        movie_match = movies_df[movies_df['title'] == row['title']]
        poster_url = ''
        tmdb_id_val = ''
        if not movie_match.empty:
            m = movie_match.iloc[0]
            imdb_id = str(m['imdbId']).split('.')[0].zfill(7) if pd.notna(m.get('imdbId')) else ''
            tmdb_id_val = str(m['tmdbId']).split('.')[0] if pd.notna(m.get('tmdbId')) else ''
            poster_url = get_poster_url(imdb_id, tmdb_id_val)
            
        top_avg.append({
            'movieId': int(m['movieId']) if not movie_match.empty else None,
            'sira': format_int(row['sira']),
            'title': row['title'],
            'ortalama_puan': format_float(row['ortalama_puan'], 3),
            'puan_sayisi': format_int(row['puan_sayisi']),
            'poster_url': poster_url,
            'tmdb_id': tmdb_id_val
        })

    return {
        'general_stats': general_stats,
        'most_rated': most_rated,
        'top_avg': top_avg
    }

# ─────────────────────────────────────────────
# TÜM FİLMLER İÇİN ÖZELLİK MATRİSİ HAZIRLA (BATCH)
# ─────────────────────────────────────────────
# Varsayılan kullanıcı değerleri (veri seti ortalamaları)
DEFAULT_AVG_USER_RATING = 3.5   # Mantıklı varsayılan (5 üzerinden ort.)
DEFAULT_AVG_MOVIE_RATING = 3.5
DEFAULT_RATING_COUNT = 50

def build_feature_matrix(movies, feature_cols, user_year, user_genres=None):
    """Tüm filmler için özellik matrisini vektörize şekilde oluşturur."""
    n = len(movies)
    feature_dict = {}

    # Temel istatistikler
    feature_dict['avg_movie_rating'] = np.full(n, DEFAULT_AVG_MOVIE_RATING, dtype='float32')
    feature_dict['rating_count'] = np.full(n, DEFAULT_RATING_COUNT, dtype='float32')
    feature_dict['avg_user_rating'] = np.full(n, DEFAULT_AVG_USER_RATING, dtype='float32')
    feature_dict['user_rating_count'] = np.full(n, DEFAULT_RATING_COUNT, dtype='float32')
    feature_dict['year'] = np.full(n, user_year, dtype='float32')

    # Genre one-hot
    for g in AVAILABLE_GENRES:
        col_name = f'genre_{g}'
        feature_dict[col_name] = np.zeros(n, dtype='float32')

    # Filmlerin genre'lerini toplu olarak kodla
    for i, genre_str in enumerate(movies['genre_ids'].values):
        if pd.notna(genre_str):
            for g in str(genre_str).split('|'):
                col_name = f'genre_{g}'
                if col_name in feature_dict:
                    feature_dict[col_name][i] = 1.0

    # Feature matrisini oluştur (feature_cols sırasında)
    X = np.column_stack([
        feature_dict.get(col, np.zeros(n, dtype='float32'))
        for col in feature_cols
    ])

    return X.astype('float32')

# Başlangıçta default matris oluştur (cache)
print("✓ Özellik matrisi hazırlanıyor...")
_cached_X = build_feature_matrix(movies_df, feature_cols, 2010)
print(f"✓ Özellik matrisi hazır: {_cached_X.shape}")


# ─────────────────────────────────────────────
# POSTER URL YARDIMCI FONKSİYONU
# ─────────────────────────────────────────────
def get_poster_url(imdb_id: str, tmdb_id: str) -> str:
    """
    Film posteri için URL döndürür.
    Sunucu tarafında bloke eden HTTP çağrısı yerine:
      - OMDB API anahtarı varsa → OMDB JSON endpoint'ini çağırır (hızlı, cache'li)
      - Yoksa → Poster yükleme JS tarafında asenkron yapılacağı için boş döner
    """
    cache_key = imdb_id or tmdb_id
    if not cache_key:
        return ''

    if cache_key in _poster_cache:
        return _poster_cache[cache_key]

    poster_url = ''

    # OMDB JSON API (ücretsiz planlar için çalışır — img.omdbapi.com değil!)
    if OMDB_API_KEY and imdb_id:
        try:
            resp = http_requests.get(
                f'https://www.omdbapi.com/?i=tt{imdb_id}&apikey={OMDB_API_KEY}',
                timeout=3
            )
            if resp.ok:
                data = resp.json()
                p = data.get('Poster', '')
                if p and p != 'N/A':
                    poster_url = p
        except Exception:
            pass

    _poster_cache[cache_key] = poster_url
    return poster_url


# ─────────────────────────────────────────────
# FORM SAYFASI
# ─────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    """Ana sayfa - rapor ve proje ozeti"""
    report = load_report_data()
    temizleme_adimlari = [
        'Ham veri yüklendi (movies, ratings, tags) ve gereksiz sütunlar elendi.',
        'NaN içeren satırlar temizlendi ve yinelenen kayıtlar kaldırıldı.',
        'Türler benzersiz ID ile kodlandı ve genre_ids alanı üretildi.',
        'Başlıktan yıl ayıklandı, yıl bilgisi standartlaştırıldı.',
        '1980 öncesi filmler çıkarıldı, belirsiz yıllar korundu.',
        'Az aktif kullanıcılar ve düşük puanlı filmler filtrelendi.',
        'Temizlenmiş veri ProcessedData/ altında kaydedildi.'
    ]

    return render_template(
        'home.html',
        report=report,
        temizleme_adimlari=temizleme_adimlari
    )

@app.route('/dataset')
def dataset_page():
    """Veri seti tanıtım ve analiz sayfası"""
    df = movies_df.copy()

    # ── Genel istatistikler ──────────────────────────────────
    total_films = len(df)
    year_valid = df['year'].dropna().astype(float)
    year_valid = year_valid[year_valid > 0]
    min_year = int(year_valid.min()) if len(year_valid) else 0
    max_year = int(year_valid.max()) if len(year_valid) else 0
    avg_rating = round(float(df['avg_movie_rating'].mean()) * 2, 1)
    median_rating = round(float(df['avg_movie_rating'].median()) * 2, 1)
    std_rating = round(float(df['avg_movie_rating'].std()) * 2, 2)
    total_genres = len(AVAILABLE_GENRES)

    # ── Sütun listesi ────────────────────────────────────────
    columns_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        null_pct = round(null_count / total_films * 100, 1)
        try:
            sample = str(df[col].dropna().iloc[0]) if null_count < total_films else '—'
            sample = sample[:40] + '...' if len(sample) > 40 else sample
        except Exception:
            sample = '—'
        columns_info.append({
            'name': col, 'dtype': dtype,
            'null_count': null_count, 'null_pct': null_pct,
            'sample': sample
        })

    # ── Tür dağılımı ─────────────────────────────────────────
    genre_counts = {}
    for genre_str in df['genre_ids'].fillna(''):
        for gid in str(genre_str).split('|'):
            gid = gid.strip()
            if gid and gid in GENRE_ID_TO_NAME:
                name = GENRE_ID_TO_NAME[gid]
                genre_counts[name] = genre_counts.get(name, 0) + 1
    genre_counts_sorted = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    genre_labels = [g[0] for g in genre_counts_sorted]
    genre_values = [g[1] for g in genre_counts_sorted]

    # ── On yıl bazında film sayısı ───────────────────────────
    decade_counts = {}
    for y in year_valid:
        decade = int(y // 10 * 10)
        if decade >= 1900:
            decade_counts[decade] = decade_counts.get(decade, 0) + 1
    decades_sorted = sorted(decade_counts.items())
    decade_labels = [f"{d}s" for d, _ in decades_sorted]
    decade_values = [v for _, v in decades_sorted]

    # ── Puan dağılımı (1.0 adımlı, 10 üzerinden) ──────────────────────────
    bins = [float(x) for x in range(1, 11)]
    rating_hist = {b: 0 for b in bins}
    for r in df['avg_movie_rating'].dropna():
        bucket = float(round(float(r) * 2))
        bucket = max(1.0, min(10.0, bucket))
        rating_hist[bucket] = rating_hist.get(bucket, 0) + 1
    rating_labels = [str(int(k)) for k in sorted(rating_hist.keys())]
    rating_values = [rating_hist[float(k)] for k in sorted(rating_hist.keys())]

    # ── CSS barlar için yüzdelik hesaplama ────────────────
    max_g = max(genre_values) if genre_values else 1
    genre_stats = [{'label': l, 'count': v, 'pct': min(100, int((v / max_g) * 100))} for l, v in zip(genre_labels, genre_values)][:15]

    max_d = max(decade_values) if decade_values else 1
    decade_stats = [{'label': l, 'count': v, 'pct': min(100, int((v / max_d) * 100))} for l, v in zip(decade_labels, decade_values)]

    max_r = max(rating_values) if rating_values else 1
    rating_stats = [{'label': l, 'count': v, 'pct': min(100, int((v / max_r) * 100))} for l, v in zip(rating_labels, rating_values)]

    # ── Değişken açıklamaları ────────────────────────────────
    var_descriptions = [
        {'name': 'movieId',        'type': 'int32',    'desc': 'Her film için benzersiz kimlik numarası (MovieLens veri setinden).'},
        {'name': 'title',          'type': 'object', 'desc': 'Filmin tam adı. Genellikle yapım yılını parantez içinde de barındırır.'},
        {'name': 'year',           'type': 'float64',    'desc': 'Filmin yapım yılı. Başlıktan regex ile ayıklanmıştır (1880–2025 arası).'},
        {'name': 'genre_ids',      'type': 'object', 'desc': 'Pipe (|) ile ayrılmış sayısal tür kimlikleri. Örn: "1|8|17" → Action, Drama, Thriller.'},
        {'name': 'avg_movie_rating','type': 'float64', 'desc': 'Filmin tüm kullanıcı oylarının ortalaması (Orijinali 5, arayüzde 10 üzerinden gösterilir). movie_stats.csv\'den birleştirilmiştir.'},
        {'name': 'imdbId',         'type': 'float64',    'desc': 'IMDb veritabanındaki benzersiz film kimliği. Poster ve dış bağlantılar için kullanılır.'},
        {'name': 'tmdbId',         'type': 'float64',    'desc': 'The Movie Database (TMDb) kimliği. Otomatik poster yükleme için kullanılır.'},
    ]

    stats = {
        'total_films': total_films,
        'min_year': min_year,
        'max_year': max_year,
        'avg_rating': avg_rating,
        'median_rating': median_rating,
        'std_rating': std_rating,
        'total_genres': total_genres,
        'total_columns': len(df.columns),
        'null_ratio': round(df.isnull().sum().sum() / (total_films * len(df.columns)) * 100, 2),
    }

    return render_template(
        'dataset.html',
        stats=stats,
        columns_info=columns_info,
        genre_stats=genre_stats,
        decade_stats=decade_stats,
        rating_stats=rating_stats,
        var_descriptions=var_descriptions,
        model_name=best_name,
        feature_count=len(feature_cols),
    )

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    """Film detay sayfası"""
    movie = movies_df[movies_df['movieId'] == movie_id]
    if movie.empty:
        return "Film bulunamadı", 404
    
    movie_data = movie.iloc[0].to_dict()
    
    # Genre isimlerini hazırla
    genre_ids = str(movie_data.get('genre_ids', '')).split('|')
    genre_names = [GENRE_ID_TO_NAME.get(gid, gid) for gid in genre_ids if gid]
    movie_data['genre_names'] = genre_names
    
    # IDs
    imdb_id = str(movie_data.get('imdbId', '')).split('.')[0].zfill(7) if pd.notna(movie_data.get('imdbId')) else ''
    tmdb_id = str(movie_data.get('tmdbId', '')).split('.')[0] if pd.notna(movie_data.get('tmdbId')) else ''
    
    movie_data['imdbId_clean'] = imdb_id
    movie_data['tmdbId_clean'] = tmdb_id
    
    # Poster URL
    poster_url = get_poster_url(imdb_id, tmdb_id)
    movie_data['poster_url'] = poster_url

    # TMDb'den film açıklaması çek
    overview = ''
    try:
        if tmdb_id:
            TMDB_KEY = '8265bd1679663a7ea12ac168da84d2e8'
            resp = requests.get(
                f'https://api.themoviedb.org/3/movie/{tmdb_id}',
                params={'api_key': TMDB_KEY, 'language': 'tr-TR'},
                timeout=4
            )
            if resp.ok:
                tdata = resp.json()
                overview = tdata.get('overview', '')
                if not overview:  # Türkçe yoksa İngilizce dene
                    resp2 = requests.get(
                        f'https://api.themoviedb.org/3/movie/{tmdb_id}',
                        params={'api_key': TMDB_KEY, 'language': 'en-US'},
                        timeout=4
                    )
                    if resp2.ok:
                        overview = resp2.json().get('overview', '')
    except Exception:
        overview = ''
    movie_data['overview'] = overview

    return render_template('movie_details.html', movie=movie_data)

@app.route('/recommendations', methods=['GET'])
def recommendations_page():
    """Oneri sayfasi - form"""
    return render_template(
        'index.html',
        genre_list=GENRE_LIST,
        min_year=int(movies_df['year'].min()),
        max_year=int(movies_df['year'].max())
    )


# ─────────────────────────────────────────────
# RASTGELE FİLM API
# ─────────────────────────────────────────────
@app.route('/api/random_movie', methods=['GET'])
def random_movie():
    """Rastgele bir film döndür"""
    sample = movies_df[movies_df['avg_movie_rating'] >= 3.5].sample(1).iloc[0]
    return jsonify({'movieId': int(sample['movieId'])})

# ─────────────────────────────────────────────
# FİLM ARAMA API
# ─────────────────────────────────────────────
@app.route('/api/search_movies', methods=['GET'])
def search_movies():
    """Film adı ile arama — autocomplete için"""
    query = request.args.get('q', '').strip().lower()
    if len(query) < 5:
        return jsonify({'results': []})

    # Başlık içinde arama yap (case-insensitive)
    mask = movies_df['title'].str.lower().str.contains(query, na=False)
    matched = movies_df[mask].head(10)

    results = []
    for _, row in matched.iterrows():
        genre_ids = str(row.get('genre_ids', '')).split('|')
        genre_names = [GENRE_ID_TO_NAME.get(gid, gid) for gid in genre_ids if gid]
        results.append({
            'movieId': int(row['movieId']),
            'title': row['title'],
            'year': int(row['year']),
            'genre_names': genre_names[:3],
            'avg_rating': round(float(row['avg_movie_rating']) * 2, 1) if pd.notna(row.get('avg_movie_rating')) else None
        })

    return jsonify({'results': results})


# ─────────────────────────────────────────────
# TAHMİN VE ÖNERİ (VEKTÖRİZE)
# ─────────────────────────────────────────────
@app.route('/recommend', methods=['POST'])
def recommend():
    """Film önerisi döndür — vektörize batch tahmin."""
    try:
        data = request.get_json()

        # Kullanıcı inputları
        preferred_year = int(data.get('preferred_year', 2010))
        selected_genres = data.get('selected_genres', [])
        selected_movie_id = data.get('selected_movie_id', None)

        offset = int(data.get('offset', 0))
        limit = int(data.get('limit', 10))
        min_score = float(data.get('min_score', 0.0))
        max_score = float(data.get('max_score', 10.0))

        if not (1890 <= preferred_year <= 2025):
            preferred_year = 2010

        # Seçili film bilgisini al
        selected_movie_genres = set()
        if selected_movie_id:
            sel_movie = movies_df[movies_df['movieId'] == int(selected_movie_id)]
            if not sel_movie.empty:
                sel_row = sel_movie.iloc[0]
                sel_genre_str = str(sel_row.get('genre_ids', ''))
                selected_movie_genres = set(sel_genre_str.split('|')) if sel_genre_str else set()
                print(f"  - Seçili film: {sel_row['title']} (ID: {selected_movie_id})")

        print(f"\n→ Tahmin isteği:")
        print(f"  - Tercih edilen yıl: {preferred_year}")
        print(f"  - Tercih edilen türler: {selected_genres}")
        print(f"  - Puan Aralığı: {min_score} - {max_score}")
        print(f"  - Offset: {offset}, Limit: {limit}")

        # Vektörize özellik matrisi oluştur
        X_all = build_feature_matrix(movies_df, feature_cols, preferred_year, selected_genres)

        # Scaler varsa uygula
        if scaler is not None:
            X_all = scaler.transform(X_all)

        # TOPLU TAHMİN (tek seferde tüm filmler)
        try:
            probas = model.predict_proba(X_all)[:, 1]
        except Exception:
            preds = model.predict(X_all)
            probas = preds.astype(float)

        # Seçili genre filtresi uygula
        if selected_genres:
            genre_mask = np.zeros(len(movies_df), dtype=bool)
            for i, genre_str in enumerate(movies_df['genre_ids'].values):
                if pd.notna(genre_str):
                    movie_genres = set(str(genre_str).split('|'))
                    if all(g in movie_genres for g in selected_genres):
                        genre_mask[i] = True
            # Seçili türlerin TÜMÜNE sahip olmayanları tamamen ele
            probas[~genre_mask] = -1.0

        # Seçili filme benzer filmleri öne çıkar
        if selected_movie_genres:
            for i, genre_str in enumerate(movies_df['genre_ids'].values):
                if pd.notna(genre_str):
                    movie_genres = set(str(genre_str).split('|'))
                    overlap = len(movie_genres & selected_movie_genres)
                    if overlap > 0:
                        # Ortak tür sayısına göre boost ver
                        probas[i] *= (1.0 + overlap * 0.15)
            # Seçili filmin kendisini listeden çıkar
            if selected_movie_id:
                sel_idx = movies_df.index[movies_df['movieId'] == int(selected_movie_id)].tolist()
                for idx in sel_idx:
                    probas[idx] = -2.0

        # Puan aralığı filtresi uygula (Bu aralık dışındakileri tamamen ele)
        avg_ratings = movies_df['avg_movie_rating'].values
        if max_score > 5.0:
            converted_min = min_score / 2.0
            converted_max = max_score / 2.0
        else:
            converted_min = min_score
            converted_max = max_score
            
        score_mask = (avg_ratings >= converted_min) & (avg_ratings <= converted_max)
        probas[~score_mask] = -1.0 # En sona atılsın
        
        # Kesin yıl filtresi uygula (Arayüzde "2020'ler" vb. seçilmişse)
        strict_year_filter = data.get('strict_year_filter', False)
        if strict_year_filter:
            years = movies_df['year'].values
            if preferred_year == 2022:    # 2020'ler
                year_mask = (years >= 2020)
            elif preferred_year == 2015:  # 2010'lar
                year_mask = (years >= 2010) & (years <= 2019)
            elif preferred_year == 2005:  # 2000'ler
                year_mask = (years >= 2000) & (years <= 2009)
            elif preferred_year == 1995:  # 1990'lar
                year_mask = (years >= 1990) & (years <= 1999)
            elif preferred_year == 1985:  # 1980'ler
                year_mask = (years < 1990)
            else:
                year_mask = np.ones(len(movies_df), dtype=bool)
                
            probas[~year_mask] = -1.0 # En sona atılsın
        
        # Top limit filmi seç (offset'ten itibaren)
        top_indices = np.argsort(probas)[::-1][offset:offset+limit]

        top_limit = []
        for idx in top_indices:
            row = movies_df.iloc[idx]
            genre_names = [
                GENRE_ID_TO_NAME.get(g, f'Genre {g}')
                for g in str(row['genre_ids']).split('|')
                if g and g != '0'
            ]
            
            imdb_id = str(row['imdbId']).split('.')[0].zfill(7) if pd.notna(row.get('imdbId')) else ''
            tmdb_id = str(row['tmdbId']).split('.')[0] if pd.notna(row.get('tmdbId')) else ''
            poster_url = get_poster_url(imdb_id, tmdb_id)
            
            top_limit.append({
                'movieId': int(row['movieId']),
                'title': row['title'],
                'year': (lambda y: int(y) if y == y and y is not None else 0)(row.get('year')),
                'genres': str(row['genre_ids']),
                'genre_names': genre_names,
                'avg_rating': round(float(row['avg_movie_rating']) * 2, 1), # 5 üzerinden olan puanı 10 üzerinden gösterelim
                'imdbId': imdb_id,
                'tmdbId': tmdb_id,
                'poster_url': poster_url
            })

        print(f"\n→ Top Öneriler ({offset} - {offset+limit}):")
        for i, film in enumerate(top_limit, 1):
            print(f"  {i+offset}. {film['title']} ({film['year']})")

        has_more = len(probas) > (offset + limit)

        return jsonify({
            'success': True,
            'recommendations': top_limit,
            'has_more': has_more,
            'user_input': {
                'preferred_year': preferred_year,
                'genres': selected_genres,
                'offset': offset,
                'limit': limit
            }
        })

    except Exception as e:
        print(f"✗ Hata: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


# ─────────────────────────────────────────────
# SAĞLIK KONTROL
# ─────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    """Sağlık kontrolü"""
    return jsonify({
        'status': 'ok',
        'model': best_name,
        'movies_count': len(movies_df),
        'genres_count': len(AVAILABLE_GENRES),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
