"""
Film Öneri ML Pipeline
======================
Algoritmalar (sırayla):
  1. ZeroR        — Baseline: hep çoğunluk sınıfını tahmin et
  2. OneR         — En iyi tek özelliği kullan
  3. KNN          — K en yakın komşu
  4. Naive Bayes  — Gaussian Naive Bayes
  5. Decision Tree— Karar ağacı

Hedef (target):
  rating >= 3.5  →  liked = 1
  rating <  3.5  →  liked = 0

Özellikler (features):
  - avg_movie_rating   : filmin ortalama puanı
  - rating_count       : filmin toplam oy sayısı
  - avg_user_rating    : kullanıcının ortalama verdiği puan
  - user_rating_count  : kullanıcının toplam oy sayısı
  - movie_year         : filmin yapım yılı
    - genre_* (one-hot)  : film türlerinin sayısal ID'leri
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, roc_auc_score)
from sklearn.base            import BaseEstimator, ClassifierMixin  # OneR için
from sklearn.dummy           import DummyClassifier          # ZeroR
from sklearn.neighbors       import KNeighborsClassifier     # KNN
from sklearn.naive_bayes     import GaussianNB               # Naive Bayes
from sklearn.tree            import DecisionTreeClassifier   # Decision Tree

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────

def banner(text: str) -> None:
    print(f"\n{'═'*55}")
    print(f"  {text}")
    print(f"{'═'*55}")

def evaluate(name: str, model, X_tr, y_tr, X_te, y_te,
             cv: StratifiedKFold, needs_scale: bool = False, sample_weight=None) -> dict:
    """Modeli eğit, test et, CV uygula, sonuçları yazdır."""
    t0 = time.time()

    # Ölçekleme gerektiren modeller için (KNN, NB)
    if needs_scale:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
    else:
        X_tr_s, X_te_s = X_tr, X_te

    # Sample weight desteklenmiyorsa düz eğitim yap
    try:
        if sample_weight is not None:
            model.fit(X_tr_s, y_tr, sample_weight=sample_weight)
        else:
            model.fit(X_tr_s, y_tr)
    except TypeError:
        model.fit(X_tr_s, y_tr)
    
    y_pred = model.predict(X_te_s)
    elapsed = time.time() - t0

    acc  = accuracy_score(y_te, y_pred)
    # ROC-AUC: sadece predict_proba destekleyenler için
    try:
        proba = model.predict_proba(X_te_s)[:, 1]
        auc   = roc_auc_score(y_te, proba)
    except Exception:
        auc = float('nan')

    # Çapraz doğrulama
    cv_scores = cross_val_score(model, X_tr_s, y_tr,
                                cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Test Accuracy    : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC          : {auc:.4f}" if not np.isnan(auc) else "  ROC-AUC          : N/A")
    print(f"  CV Accuracy      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Eğitim süresi    : {elapsed:.3f} sn")
    print(f"\n  Confusion Matrix :")
    cm = confusion_matrix(y_te, y_pred)
    print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
    print(f"\n  Sınıf Raporu     :")
    print(classification_report(y_te, y_pred,
                                target_names=['Beğenmedi (0)', 'Beğendi (1)'],
                                digits=4, zero_division=0))

    return {
        'model': name,
        'accuracy': acc,
        'auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'time_sec': elapsed,
    }


class OneRClassifier(BaseEstimator, ClassifierMixin):
    """
    OneR: Tüm özellikler üzerinde en düşük hata oranını veren
    tek özelliği ve eşiği seçer; basit threshold kuralı uygular.
    sklearn DummyClassifier'dan farklı olarak gerçekten öğrenir.

    sklearn 1.6+ uyumlu: BaseEstimator + ClassifierMixin miras alır.
    Bu sayede cross_val_score, clone(), Pipeline gibi araçlarla çalışır.
    """
    def __init__(self):
        # BaseEstimator.get_params() __init__ parametrelerini okur;
        # bu yüzden tüm hiperparametreler burada tanımlanmalı.
        # Eğitimde öğrenilen değerler trailing underscore (_) ile ayrılır.
        pass

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)   # ClassifierMixin gerektirir

        best_err        = np.inf
        self.best_feat_ = 0
        self.threshold_ = 0.0
        self.direction_ = 1            # 1: >=thr sonucu 1,  -1: <thr sonucu 1

        for col in range(X.shape[1]):
            vals = X[:, col]
            thresholds = np.nanpercentile(vals, np.linspace(5, 95, 20))
            for thr in np.unique(thresholds):   # tekrar eşikleri atla
                for direction in (1, -1):
                    pred = (vals >= thr) if direction == 1 else (vals < thr)
                    errors = (pred.astype(int) != y)
                    err  = np.average(errors, weights=sample_weight) if sample_weight is not None else np.mean(errors)
                    if err < best_err:
                        best_err        = err
                        self.best_feat_ = col
                        self.threshold_ = thr
                        self.direction_ = direction
        return self

    def predict(self, X):
        X    = np.asarray(X)
        vals = X[:, self.best_feat_]
        mask = (vals >= self.threshold_) if self.direction_ == 1 \
               else (vals < self.threshold_)
        return mask.astype(int)


# ─────────────────────────────────────────────
# 1. VERİ YÜKLEME
# ─────────────────────────────────────────────
banner("1. VERİ YÜKLEME")

# DATA_DIR may be produced by a preprocessing pipeline. Try common locations
CANDIDATE_DIRS = ['ModelResult', 'ProcessedData']
found_dir = None
for d in CANDIDATE_DIRS:
    if all(os.path.exists(os.path.join(d, f)) for f in ['movies_cleaned.csv', 'ratings_optimized.csv']):
        found_dir = d
        break

if found_dir is None:
    os.makedirs('ModelResult', exist_ok=True)
    raise SystemExit("'movies_cleaned.csv' veya 'ratings_optimized.csv' bulunamadı. Önce data_pipeline_optimized.py çalıştır veya ProcessedData içindeki dosyaları ModelResult'a kopyala.")

# Girdi (input) ve çıktı (output) dizinlerini ayır
INPUT_DIR = found_dir
OUTPUT_DIR = 'ModelResult'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"  Girdi dizini : {INPUT_DIR}")
print(f"  Çıktı dizini : {OUTPUT_DIR}")

t0 = time.time()
movies  = pd.read_csv(f'{INPUT_DIR}/movies_cleaned.csv',
                      dtype={'movieId': 'int32'})
ratings = pd.read_csv(f'{INPUT_DIR}/ratings_optimized.csv',
                      dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

# Tags varsa yükle, yoksa atla
tags_path = f'{INPUT_DIR}/tags_optimized.csv'
tags = pd.read_csv(tags_path) if os.path.exists(tags_path) else None

print(f"  movies  : {len(movies):>10,} satır")
print(f"  ratings : {len(ratings):>10,} satır")
print(f"  Yükleme : {time.time()-t0:.2f} sn")

# ─────────────────────────────────────────────
# 2. ÖZELLİK MÜHENDİSLİĞİ
# ─────────────────────────────────────────────
banner("2. ÖZELLİK MÜHENDİSLİĞİ")
feat_t0 = time.time()

# 2a. Hedef değişken
ratings['liked'] = (ratings['rating'] >= 3.5).astype('int8')
print(f"  Sınıf dağılımı  →  liked=1: {ratings['liked'].mean()*100:.1f}%  |  liked=0: {(1-ratings['liked'].mean())*100:.1f}%")

# 2b. Film ve kullanıcı istatistikleri train/test split sonrası hesaplanacak
# (Data leakage önlemi — aşağıda bölüm 3'ten sonra yapılıyor)
movie_stats = None  # placeholder
user_stats = None   # placeholder

# 2d. Türleri one-hot encode et
genre_source_col = 'genre_ids' if 'genre_ids' in movies.columns else 'genres'
genres_expanded = movies[genre_source_col].fillna('0').astype(str).str.get_dummies(sep='|')
genres_expanded = genres_expanded.drop(columns=['(no genres listed)'], errors='ignore')
genres_expanded = genres_expanded.drop(columns=['0'], errors='ignore')
genres_expanded.columns = [f'genre_{col}' for col in genres_expanded.columns]
movies_with_genres = pd.concat([movies[['movieId', 'year']], genres_expanded], axis=1)

# 2e. Birleştir (istatistikler henüz yok, split sonrası eklenecek)
df = (ratings[['userId', 'movieId', 'liked', 'rating']]
      .merge(movies_with_genres, on='movieId', how='left'))

# 2f. Eksik değerleri doldur (year için medyan, diğerleri 0)
df['year'] = df['year'].fillna(df['year'].median())
df = df.fillna(0)

print(f"  Toplam örnek sayısı   : {len(df):,}")
print(f"  Özellik mühendisliği süresi : {time.time() - feat_t0:.2f} sn")

# ─────────────────────────────────────────────
# 3. EĞİTİM / TEST BÖLME (Data Leakage Önlemi)
# ─────────────────────────────────────────────
banner("3. EĞİTİM / TEST BÖLME")

# Büyük veri setlerinde hız için örnekleme (isteğe bağlı)
MAX_SAMPLES = 200_000
if len(df) > MAX_SAMPLES:
    indices = np.random.RandomState(42).choice(len(df), MAX_SAMPLES, replace=False)
    df = df.iloc[indices].reset_index(drop=True)
    print(f"  Hız için {MAX_SAMPLES:,} örneğe örneklendi.")

# Önce split yap
from sklearn.model_selection import train_test_split as tts_split
train_df, test_df = tts_split(df, test_size=0.20, random_state=42, stratify=df['liked'])

# İstatistikleri SADECE train verisi üzerinden hesapla (data leakage önlemi)
movie_stats = train_df.groupby('movieId').agg(
    avg_movie_rating=('rating', 'mean'),
    rating_count=('rating', 'count'),
).astype('float32').reset_index()

user_stats = train_df.groupby('userId').agg(
    avg_user_rating=('rating', 'mean'),
    user_rating_count=('rating', 'count'),
).astype('float32').reset_index()

# İstatistikleri kaydet (Web uygulamasında puan aralığı filtresi için kullanılacak)
movie_stats.to_csv(f'{OUTPUT_DIR}/movie_stats.csv', index=False)

# İstatistikleri train ve test'e birleştir
train_df = train_df.merge(movie_stats, on='movieId', how='left').merge(user_stats, on='userId', how='left')
test_df = test_df.merge(movie_stats, on='movieId', how='left').merge(user_stats, on='userId', how='left')

# Eksik istatistikleri doldur (test'te yeni filmler/kullanıcılar olabilir)
for col in ['avg_movie_rating', 'rating_count', 'avg_user_rating', 'user_rating_count']:
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    test_df[col] = test_df[col].fillna(median_val)

feature_cols = [c for c in train_df.columns if c not in ('userId', 'movieId', 'liked', 'rating')]
print(f"  Toplam özellik sayısı : {len(feature_cols)}")
print(f"  Özellikler            : {', '.join(feature_cols[:8])}{'...' if len(feature_cols)>8 else ''}")

# Sample weight oluştur (train verisi üzerinden)
sample_weight = (train_df['user_rating_count'] / train_df['user_rating_count'].max()).values
w_train = sample_weight

X_train = train_df[feature_cols].to_numpy(dtype='float32')
y_train = train_df['liked'].to_numpy()
X_test = test_df[feature_cols].to_numpy(dtype='float32')
y_test = test_df['liked'].to_numpy()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"  Eğitim : {len(X_train):,}  |  Test : {len(X_test):,}")
print(f"  CV     : {cv.n_splits}-fold Stratified")

# ─────────────────────────────────────────────
# 4. ALGORİTMALAR
# ─────────────────────────────────────────────
results = []

# ── ALGORİTMA 1: ZeroR ──────────────────────
banner("ALGORİTMA 1 — ZeroR (Baseline)")
print("  Strateji: Her zaman eğitim setindeki çoğunluk sınıfını tahmin et.")
zeror = DummyClassifier(strategy='most_frequent', random_state=42)
results.append(evaluate("ZeroR", zeror, X_train, y_train, X_test, y_test, cv, sample_weight=w_train))

# ── ALGORİTMA 2: OneR ───────────────────────
banner("ALGORİTMA 2 — OneR (En İyi Tek Kural)")
print("  Strateji: En düşük hata oranını veren tek özellik + eşik kuralı.")
oner = OneRClassifier()
results.append(evaluate("OneR", oner, X_train, y_train, X_test, y_test, cv, sample_weight=w_train))
print(f"\n  Seçilen özellik indeksi : {oner.best_feat_}")
print(f"  Seçilen özellik adı     : {feature_cols[oner.best_feat_]}")
print(f"  Eşik değeri             : {oner.threshold_:.4f}")

# ── ALGORİTMA 3: KNN ────────────────────────
banner("ALGORİTMA 3 — KNN (K En Yakın Komşu)")
print("  K=11, metric=minkowski. Özellikler StandardScaler ile ölçeklendi.")
knn = KNeighborsClassifier(n_neighbors=11, metric='minkowski', n_jobs=-1)
results.append(evaluate("KNN (k=11)", knn,
                         X_train, y_train, X_test, y_test, cv,
                         needs_scale=True, sample_weight=w_train))

# ── ALGORİTMA 4: Naive Bayes ────────────────
banner("ALGORİTMA 4 — Naive Bayes (Gaussian)")
print("  Gaussian NB: sürekli özellikler için normal dağılım varsayımı.")
nb = GaussianNB()
results.append(evaluate("Naive Bayes", nb,
                         X_train, y_train, X_test, y_test, cv,
                         needs_scale=True, sample_weight=w_train))

# ── ALGORİTMA 5: Decision Tree ──────────────
banner("ALGORİTMA 5 — Decision Tree")
print("  max_depth=10, criterion=gini, min_samples_leaf=20.")
dt = DecisionTreeClassifier(
    max_depth=10,
    criterion='gini',
    min_samples_leaf=20,
    random_state=42,
)
results.append(evaluate("Decision Tree", dt,
                         X_train, y_train, X_test, y_test, cv, sample_weight=w_train))

# Önemli özellikler (Decision Tree)
importances = pd.Series(dt.feature_importances_, index=feature_cols)
top5 = importances.nlargest(5)
print("\n  En önemli 5 özellik:")
for feat, val in top5.items():
    bar = '█' * int(val * 60)
    print(f"    {feat:<25} {val:.4f}  {bar}")

# ─────────────────────────────────────────────
# 5. KARŞILAŞTIRMA TABLOSU
# ─────────────────────────────────────────────
banner("5. KARŞILAŞTIRMA TABLOSU")

res_df = pd.DataFrame(results).set_index('model')
res_df['accuracy_%'] = (res_df['accuracy'] * 100).round(2)
res_df['cv_mean_%']  = (res_df['cv_mean']  * 100).round(2)
res_df['cv_std_%']   = (res_df['cv_std']   * 100).round(2)
res_df['auc']        = res_df['auc'].round(4)
res_df['time_sec']   = res_df['time_sec'].round(3)

print(res_df[['accuracy_%', 'auc', 'cv_mean_%', 'cv_std_%', 'time_sec']].to_string())

best_name = res_df['cv_mean'].idxmax()
print(f"\n  ★ En iyi model (CV Accuracy): {best_name}  →  {res_df.loc[best_name,'cv_mean_%']}%")
print(f"  Test Accuracy              : {res_df.loc[best_name,'accuracy_%']}%")

# ─────────────────────────────────────────────
# 6. GÖRSELLEŞTİRME (Matplotlib)
# ─────────────────────────────────────────────
banner("6. GÖRSELLEŞTİRME")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

matplotlib.rcParams.update({
    'font.family':  'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'figure.dpi':         120,
})

COLORS    = ['#6C8EBF', '#82B366', '#D6A041', '#AE4132', '#7B61C4']
MODEL_NAMES = res_df.index.tolist()

plt.close('all')
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Film Öneri Modeli — Algoritma Karşılaştırması', fontsize=16, fontweight='bold', y=0.98)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 6.1 Test Accuracy ──
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(MODEL_NAMES, res_df['accuracy_%'], color=COLORS, edgecolor='white', linewidth=0.8)
ax1.set_title('Test Accuracy (%)', fontweight='bold')
ax1.set_ylim(max(0, res_df['accuracy_%'].min() - 5), min(100, res_df['accuracy_%'].max() + 5))
ax1.set_xticklabels(MODEL_NAMES, rotation=30, ha='right', fontsize=8)
for b in bars:
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
             f"{b.get_height():.2f}%", ha='center', va='bottom', fontsize=8)

# ── 6.2 ROC-AUC ──
ax2 = fig.add_subplot(gs[0, 1])
auc_vals  = res_df['auc'].tolist()
auc_names = MODEL_NAMES
auc_colors = [COLORS[i] if not np.isnan(v) else '#cccccc' for i, v in enumerate(auc_vals)]
auc_plot  = [v if not np.isnan(v) else 0 for v in auc_vals]
bars2 = ax2.bar(auc_names, auc_plot, color=auc_colors, edgecolor='white', linewidth=0.8)
ax2.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Random (0.5)')
ax2.set_title('ROC-AUC', fontweight='bold')
ax2.set_ylim(0, 1.05)
ax2.set_xticklabels(auc_names, rotation=30, ha='right', fontsize=8)
ax2.legend(fontsize=7)
for b, v in zip(bars2, auc_vals):
    label = f"{v:.4f}" if not np.isnan(v) else "N/A"
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
             label, ha='center', va='bottom', fontsize=8)

# ── 6.3 CV Accuracy ± std ──
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(MODEL_NAMES, res_df['cv_mean_%'], color=COLORS,
        yerr=res_df['cv_std_%'], capsize=5, edgecolor='white', linewidth=0.8,
        error_kw=dict(elinewidth=1.5, ecolor='#444'))
ax3.set_title('CV Accuracy ± Std (%)', fontweight='bold')
ax3.set_ylim(max(0, res_df['cv_mean_%'].min() - 6), min(100, res_df['cv_mean_%'].max() + 6))
ax3.set_xticklabels(MODEL_NAMES, rotation=30, ha='right', fontsize=8)

# ── 6.4 Eğitim Süresi ──
ax4 = fig.add_subplot(gs[1, 0])
ax4.barh(MODEL_NAMES, res_df['time_sec'], color=COLORS, edgecolor='white', linewidth=0.8)
ax4.set_title('Eğitim Süresi (sn)', fontweight='bold')
ax4.set_xlabel('Saniye')
for i, v in enumerate(res_df['time_sec']):
    ax4.text(v + 0.001, i, f"{v:.3f}s", va='center', fontsize=8)

# ── 6.5 ROC Eğrileri (proba destekleyen modeller) ──
ax5 = fig.add_subplot(gs[1, 1:])
ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4, label='Random')


model_map = {
    'ZeroR':         (zeror, X_test,                             False),
    'OneR':          (oner,  X_test,                             False),
    'KNN (k=11)':    (knn,   StandardScaler().fit(X_train).transform(X_test), False),
    'Naive Bayes':   (nb,    StandardScaler().fit(X_train).transform(X_test), False),
    'Decision Tree': (dt,    X_test,                             False),
}
# Scalerları doğru şekilde yeniden fit et
scaler_knn = StandardScaler().fit(X_train)
scaler_nb  = StandardScaler().fit(X_train)
model_map['KNN (k=11)']  = (knn, scaler_knn.transform(X_test), False)
model_map['Naive Bayes'] = (nb,  scaler_nb.transform(X_test),  False)

# Her model için ayrı görselleştirme (ROC + Confusion) ve süre ölçümü
for mname in MODEL_NAMES:
    mdl, X_te_plot, _ = model_map[mname]
    t_img0 = time.time()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # ROC
    try:
        prob = mdl.predict_proba(X_te_plot)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        axes[0].plot(fpr, tpr, color='C0', linewidth=2)
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4)
        axes[0].set_title('ROC Eğrisi')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
    except Exception:
        axes[0].text(0.5, 0.5, 'ROC (predict_proba yok)', ha='center', va='center')
        axes[0].set_title('ROC N/A')

    # Confusion Matrix
    try:
        preds = mdl.predict(X_te_plot)
        ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=axes[1], colorbar=False)
        axes[1].set_title('Confusion Matrix')
    except Exception:
        axes[1].text(0.5, 0.5, 'Tahmin yapılamadı', ha='center', va='center')
        axes[1].set_title('Confusion N/A')

    fig.suptitle(f'{mname} — ROC & Confusion')
    safe_name = mname.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    out_img = f'{OUTPUT_DIR}/{safe_name}.png'
    plt.savefig(out_img, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  {mname} görselleştirmesi kaydedildi → {out_img}  (süre: {time.time() - t_img0:.3f}s)")

for i, (mname, color) in enumerate(zip(MODEL_NAMES, COLORS)):
    mdl, X_te_roc, _ = model_map[mname]
    try:
        prob = mdl.predict_proba(X_te_roc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_val = res_df.loc[mname, 'auc']
        ax5.plot(fpr, tpr, color=color, linewidth=2,
                 label=f"{mname}  (AUC={auc_val:.3f})")
    except Exception:
        pass

ax5.set_title('ROC Eğrileri', fontweight='bold')
ax5.set_xlabel('False Positive Rate')
ax5.set_ylabel('True Positive Rate')
ax5.legend(fontsize=8, loc='lower right')

# ── 6.6 En İyi Modelin Confusion Matrix ──
ax6 = fig.add_subplot(gs[2, 0])
best_model_obj = {'ZeroR': zeror, 'OneR': oner, 'KNN (k=11)': knn,
                  'Naive Bayes': nb, 'Decision Tree': dt}[best_name]
best_X_test    = model_map[best_name][1]
ConfusionMatrixDisplay.from_predictions(
    y_test, best_model_obj.predict(best_X_test),
    display_labels=['Beğenmedi', 'Beğendi'],
    colorbar=False, ax=ax6,
    cmap='Blues',
)
ax6.set_title(f'Confusion Matrix — {best_name}', fontweight='bold', fontsize=9)

# ── 6.7 Decision Tree Özellik Önemleri ──
ax7 = fig.add_subplot(gs[2, 1:])
top_n   = min(12, len(feature_cols))
imp_top = importances.nlargest(top_n).sort_values()
ax7.barh(imp_top.index, imp_top.values,
         color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n)),
         edgecolor='white', linewidth=0.8)
ax7.set_title('Decision Tree — En Önemli Özellikler', fontweight='bold')
ax7.set_xlabel('Önem Skoru')
for i, v in enumerate(imp_top.values):
    ax7.text(v + 0.001, i, f"{v:.4f}", va='center', fontsize=8)

plt.savefig(f'{OUTPUT_DIR}/model_comparison.png', bbox_inches='tight', dpi=150)
print(f"  Grafik '{OUTPUT_DIR}/model_comparison.png' olarak kaydedildi.")

# ─────────────────────────────────────────────
# 7. EN İYİ MODELİ KAYDET (joblib)
# ─────────────────────────────────────────────
banner("7. EN İYİ MODELİ KAYDET")

import joblib

# Web'de kullanmak için gerekli her şeyi bir dict içinde sakla:
#   model        → tahmin yapan nesne
#   scaler       → (varsa) StandardScaler
#   feature_cols → özellik sırası (JSON input sıralaması için)
#   threshold    → liked eşiği
#   best_name    → model adı
needs_scaler = best_name in ('KNN (k=11)', 'Naive Bayes')
export_scaler = scaler_knn if best_name == 'KNN (k=11)' \
               else scaler_nb if best_name == 'Naive Bayes' \
               else None

model_bundle = {
    'model':        best_model_obj,
    'scaler':       export_scaler,
    'feature_cols': feature_cols,
    'threshold':    3.5,
    'best_name':    best_name,
    'accuracy':     float(res_df.loc[best_name, 'accuracy']),
    'auc':          float(res_df.loc[best_name, 'auc']),
}

model_path = f'{OUTPUT_DIR}/best_model.joblib'
joblib.dump(model_bundle, model_path)
print(f"  Model paketi kaydedildi  → {model_path}")
print(f"  Model adı                : {best_name}")
print(f"  Test Accuracy            : {res_df.loc[best_name,'accuracy_%']}%")
print(f"  AUC                      : {res_df.loc[best_name,'auc']}")

# ─────────────────────────────────────────────
# 8. SONUÇLARI CSV KAYDET
# ─────────────────────────────────────────────
out_path = f'{OUTPUT_DIR}/model_results.csv'
res_df.reset_index().to_csv(out_path, index=False)
print(f"\n  Karşılaştırma tablosu    → {out_path}")

# Eğitim sürelerini açıkça yazdır
print("\n  Eğitim süreleri (her model):")
for idx, row in res_df.iterrows():
    print(f"    {idx:<15} : {row['time_sec']:.3f} sn")
print(f"\n{'═'*55}")
print("  Pipeline tamamlandı.")
print(f"{'═'*55}\n")