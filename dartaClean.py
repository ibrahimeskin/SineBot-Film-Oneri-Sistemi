import pandas as pd
import numpy as np
import os
import time


def build_genre_id_map(genres: pd.Series) -> dict:
    unique_genres = sorted(
        {
            token.strip()
            for value in genres.dropna().astype(str)
            for token in value.split('|')
            if token.strip() and token.strip() != '(no genres listed)'
        }
    )
    return {genre: idx + 1 for idx, genre in enumerate(unique_genres)}


def encode_genre_ids(genres_value: object, genre_map: dict) -> str:
    if pd.isna(genres_value):
        return '0'

    genre_ids = [
        str(genre_map[token.strip()])
        for token in str(genres_value).split('|')
        if token.strip() in genre_map
    ]
    return '|'.join(genre_ids) if genre_ids else '0'

# ─────────────────────────────────────────────
# 1. ADIM: ÇIKTI DİZİNİNİ HAZIRLA
# ─────────────────────────────────────────────
os.makedirs('ProcessedData', exist_ok=True)

# ─────────────────────────────────────────────
# 2. ADIM: VERİYİ OPTİMİZE YÜKLE
#   - Sadece gerekli sütunlar (usecols)
#   - dtype ile tip dönüşümü yükleme sırasında
#     (sonradan astype() yerine — tek geçiş)
#   - engine='c' varsayılan ama belirtmek netlik sağlar
# ─────────────────────────────────────────────
t0 = time.time()

try:
    movies = pd.read_csv(
        'RawData/movies.csv',
        usecols=['movieId', 'title', 'genres'],
        dtype={'movieId': 'int32'},
    )

    ratings = pd.read_csv(
        'RawData/ratings.csv',
        usecols=['userId', 'movieId', 'rating'],  
        dtype={
            'userId':  'int32',
            'movieId': 'int32',
            'rating':  'float32',
        },
    )

    tags = pd.read_csv(
        'RawData/tags.csv',
        usecols=['userId', 'movieId', 'tag'],  
        dtype={
            'userId':  'int32',
            'movieId': 'int32',
        },
    )

    print(f"[{time.time()-t0:.1f}s] Veri yüklendi — "
          f"movies:{len(movies):,}  ratings:{len(ratings):,}  tags:{len(tags):,}")

except FileNotFoundError as e:
    raise SystemExit(f"Dosya bulunamadı: {e}. Lütfen RawData/ klasörünü kontrol et.")

# ─────────────────────────────────────────────
# 3. ADIM: VERİ TEMİZLEME — tek zincirde
#   - dropna + drop_duplicates tek geçişte
#   - inplace=False + yeniden atama: daha güvenli,
#     büyük DataFrame'lerde inplace() bazen yavaş
# ─────────────────────────────────────────────
movies  = movies.dropna().drop_duplicates()
ratings = ratings.dropna().drop_duplicates()
tags    = tags.dropna().drop_duplicates()

genre_id_map = build_genre_id_map(movies['genres'])
movies['genre_ids'] = movies['genres'].apply(lambda value: encode_genre_ids(value, genre_id_map))
movies = movies.drop(columns=['genres'])

print(f"[{time.time()-t0:.1f}s] Temizleme tamamlandı.")

# ─────────────────────────────────────────────
# 4. ADIM: YILI AYIKLA — vektörize slice (regex'ten hızlı)
#   "Toy Story (1995)" → son 5 char = "(1995" → [1:5] = "1995"
#   Fallback: regex ile kontrol
# ─────────────────────────────────────────────
# Hızlı yol: başlık genellikle "(YYYY)" ile biter
movies['year'] = movies['title'].str[-5:-1]
# Geçersiz değerleri temizle (4 haneli sayı değilse NaN)
movies['year'] = pd.to_numeric(
    movies['year'].where(movies['year'].str.match(r'^\d{4}$')),
    errors='coerce'
).astype('Int16')  # nullable int — NaN destekler, float32'den küçük

print(f"[{time.time()-t0:.1f}s] Yıl ayıklandı.")

# ─────────────────────────────────────────────
# 5. ADIM: FİLTRELEME — NumPy masking (isin'den hızlı)
#   value_counts yerine groupby+size veya np.unique+bincount
# ─────────────────────────────────────────────
MIN_RATINGS      = 50   # film başına minimum rating sayısı
MIN_USER_RATINGS = 20   # kullanıcı başına minimum rating sayısı
MIN_YEAR         = 1980 # bu yıldan önceki filmler çıkarılır

# ── 1980 öncesi filmleri çıkar ──────────────────────────────────────────────
movies = movies[movies['year'].isna() | (movies['year'] >= MIN_YEAR)].reset_index(drop=True)
print(f"[{time.time()-t0:.1f}s] {MIN_YEAR} öncesi filmler çıkarıldı — movies:{len(movies):,}")

# ── 10'dan az rating veren kullanıcıları çıkar ──────────────────────────────
user_ids_arr              = ratings['userId'].to_numpy()
unique_users, user_counts = np.unique(user_ids_arr, return_counts=True)
active_user_ids           = unique_users[user_counts >= MIN_USER_RATINGS]
active_user_set           = set(active_user_ids)

ratings = ratings[ratings['userId'].isin(active_user_set)].reset_index(drop=True)
print(f"[{time.time()-t0:.1f}s] Az aktif kullanıcılar çıkarıldı — ratings:{len(ratings):,}")

# ── 20'den az rating alan filmleri çıkar ────────────────────────────────────
movie_ids_arr  = ratings['movieId'].to_numpy()
unique_ids, counts = np.unique(movie_ids_arr, return_counts=True)
popular_movie_ids  = unique_ids[counts >= MIN_RATINGS]

# Hem ratings/tags hem de movies tablosunu filtrele
popular_set       = set(popular_movie_ids)           # O(1) lookup
movies            = movies[movies['movieId'].isin(popular_set)].reset_index(drop=True)
mask_ratings      = ratings['movieId'].isin(popular_set)
mask_tags         = tags['movieId'].isin(popular_set)

ratings_optimized = ratings[mask_ratings].reset_index(drop=True)
tags_optimized    = tags[mask_tags].reset_index(drop=True)

print(f"[{time.time()-t0:.1f}s] Filtreleme tamamlandı — "
      f"movies:{len(movies):,}  ratings:{len(ratings_optimized):,}  tags:{len(tags_optimized):,}")

# ─────────────────────────────────────────────
# 6. ADIM: KAYDET — chunksize ile bellek dostu yazım
# ─────────────────────────────────────────────
CHUNK = 500_000  # 500K satır/parça

def save_csv(df: pd.DataFrame, path: str, chunk: int = CHUNK) -> None:
    """Büyük DataFrame'i parçalı yazar; ilk parça header içerir."""
    mode = 'w'
    for i, start in enumerate(range(0, len(df), chunk)):
        df.iloc[start:start + chunk].to_csv(
            path,
            mode=mode,
            header=(i == 0),
            index=False,
        )
        mode = 'a'  # sonraki parçalar ekler

save_csv(movies,            'ProcessedData/movies_cleaned.csv')
save_csv(ratings_optimized, 'ProcessedData/ratings_optimized.csv')
save_csv(tags_optimized,    'ProcessedData/tags_optimized.csv')

print(f"[{time.time()-t0:.1f}s] Tüm veriler 'ProcessedData/' klasörüne kaydedildi.")

# ─────────────────────────────────────────────
# 7. ADIM: ÖZET RAPOR
# ─────────────────────────────────────────────
print("\n─── Özet ───────────────────────────────────")
print(f"  movies   : {len(movies):>10,} satır")
print(f"  ratings  : {len(ratings_optimized):>10,} satır")
print(f"  tags     : {len(tags_optimized):>10,} satır")
print(f"  RAM (ratings): {ratings_optimized.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"  Toplam süre  : {time.time()-t0:.2f} sn")
print("────────────────────────────────────────────")