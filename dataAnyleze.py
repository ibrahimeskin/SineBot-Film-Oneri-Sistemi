import pandas as pd
import numpy as np
import os
import time

os.makedirs('ModelResult', exist_ok=True)
t0 = time.time()

# ─── VERİYİ YÜKLE ────────────────────────────────────────────────────────────
try:
    movies  = pd.read_csv('ProcessedData/movies_cleaned.csv',    dtype={'movieId': 'int32', 'year': 'Int16'})
    ratings = pd.read_csv('ProcessedData/ratings_optimized.csv', dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
    tags    = pd.read_csv('ProcessedData/tags_optimized.csv',    dtype={'userId': 'int32', 'movieId': 'int32'})
except FileNotFoundError as e:
    raise SystemExit(f"Dosya bulunamadı: {e}. Önce temizleme scriptini çalıştırın.")

print(f"[{time.time()-t0:.1f}s] Veriler yüklendi.")

# ─── HESAPLAMALAR ─────────────────────────────────────────────────────────────
num_users   = ratings['userId'].nunique()
num_movies  = movies['movieId'].nunique()
num_ratings = len(ratings)

rating_counts = ratings.groupby('movieId', sort=False).size().reset_index(name='puan_sayisi')

en_cok_puanlanan = (
    rating_counts.merge(movies[['movieId', 'title']], on='movieId')
    .nlargest(10, 'puan_sayisi')[['title', 'puan_sayisi']]
    .reset_index(drop=True)
)
en_cok_puanlanan.index += 1

rating_stats = ratings.groupby('movieId', sort=False)['rating'].agg(
    puan_sayisi='count', ortalama_puan='mean'
).reset_index()

en_yuksek_ortalama = (
    rating_stats[rating_stats['puan_sayisi'] >= 50]
    .merge(movies[['movieId', 'title']], on='movieId')
    .nlargest(10, 'ortalama_puan')[['title', 'ortalama_puan', 'puan_sayisi']]
    .reset_index(drop=True)
)
en_yuksek_ortalama['ortalama_puan'] = en_yuksek_ortalama['ortalama_puan'].round(3)
en_yuksek_ortalama.index += 1

temizleme_data = [
    ('Ham veri yüklendi',               'movies.csv, ratings.csv, tags.csv — yalnızca gerekli sütunlar ve optimize dtype'),
    ('dropna() uygulandı',              'NaN içeren satırlar tüm tablolardan kaldırıldı'),
    ('drop_duplicates() uygulandı',     'Birebir aynı satırlar kaldırıldı'),
    ('Tür → ID kodlaması',              "Her tür benzersiz bir ID'ye eşlendi; genre_ids sütunu oluşturuldu"),
    ('Yıl ayıklandı',                   'Başlık sonundaki "(YYYY)" deseni ile vektörize slice; nullable Int16'),
    ('1980 öncesi filmler çıkarıldı',   'MIN_YEAR = 1980; yılı bilinmeyen filmler korundu'),
    ('Az aktif kullanıcılar çıkarıldı', 'MIN_USER_RATINGS = 20; NumPy unique+bincount maskeleme'),
    ('Az popüler filmler çıkarıldı',    'MIN_RATINGS = 50; ratings, tags ve movies\'den temizlendi'),
    ('Temizlenmiş veriler kaydedildi',  'ProcessedData/ klasörüne 500K satır/parça ile yazıldı'),
]

ornek_veri_data = [
    ('movies_cleaned',    len(movies),  movies.shape[1],  round(movies.memory_usage(deep=True).sum()/1e6, 2),  ', '.join(movies.columns)),
    ('ratings_optimized', len(ratings), ratings.shape[1], round(ratings.memory_usage(deep=True).sum()/1e6, 2), ', '.join(ratings.columns)),
    ('tags_optimized',    len(tags),    tags.shape[1],    round(tags.memory_usage(deep=True).sum()/1e6, 2),    ', '.join(tags.columns)),
]

os.makedirs('ModelResult', exist_ok=True)

# ─── TEK CSV ÇIKTI ───────────────────────────────────────────────────────────
genel_istatistikler_df = pd.DataFrame([
    {'metrik': 'Toplam Kullanici Sayisi', 'deger': num_users},
    {'metrik': 'Toplam Film Sayisi', 'deger': num_movies},
    {'metrik': 'Toplam Puanlama Sayisi', 'deger': num_ratings},
])
genel_istatistikler_df.insert(0, 'bolum', 'Genel Istatistikler')

en_cok_puanlanan_df = en_cok_puanlanan.copy()
en_cok_puanlanan_df.insert(0, 'sira', range(1, len(en_cok_puanlanan_df) + 1))
en_cok_puanlanan_df.insert(0, 'bolum', 'En Cok Puanlanan 10 Film')

en_yuksek_ortalama_df = en_yuksek_ortalama.copy()
en_yuksek_ortalama_df.insert(0, 'sira', range(1, len(en_yuksek_ortalama_df) + 1))
en_yuksek_ortalama_df.insert(0, 'bolum', 'En Yuksek Ortalama Puan')

csv_out = pd.concat(
    [genel_istatistikler_df, en_cok_puanlanan_df, en_yuksek_ortalama_df],
    ignore_index=True,
    sort=False
)

out_path = 'ModelResult/rapor.csv'
csv_out.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"[{time.time()-t0:.2f}s] '{out_path}' kaydedildi.")
print(f"""
─── Özet ────────────────────────────────────────
  Kullanici sayisi  : {num_users:>10,}
  Film sayisi       : {num_movies:>10,}
  Puanlama sayisi   : {num_ratings:>10,}
    Cikti             : {out_path}
    Bolum             : Tek CSV
─────────────────────────────────────────────────""")