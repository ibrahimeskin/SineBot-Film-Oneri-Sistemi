# SineBot — Makine Öğrenmesi Destekli Film Öneri Sistemi

Web tabanlı, **Decision Tree** algoritması kullanan kişiselleştirilmiş film öneri sistemi.

## 🎬 Özellikler

- 🤖 Decision Tree (CART) algoritması ile film önerisi
- 🎲 Rastgele film keşfi
- 🎭 Çoklu tür filtresi (AND mantığı)
- 📅 Yıl/dönem bazlı filtreleme
- ⭐ Puan aralığı filtresi
- 🔍 Gerçek zamanlı film arama (autocomplete)
- 📖 TMDb API ile film açıklamaları
- 📊 Veri seti analiz sayfası
- 🌙 Modern karanlık tema

## 🛠️ Kurulum

### Gereksinimler
```bash
pip install flask pandas numpy scikit-learn requests joblib python-dotenv
```

### Veri Dosyaları
[MovieLens ml-32m](http://grouplens.org/datasets/) veri setini indirip `Site/data/` klasörüne çıkarın.

### Çalıştırma
```bash
cd Site
python app.py
```

Tarayıcıda `http://127.0.0.1:5000` adresini açın.

## 📊 Veri Seti

**MovieLens ml-32m** (GroupLens Research)
- 200.948 kullanıcı
- 87.585 film  
- 32.000.204 puanlama

### Atıf
> F. Maxwell Harper ve Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

## 🧠 Kullanılan Algoritma

**Decision Tree (CART — Gini Safsızlığı)**
- Scikit-Learn `DecisionTreeClassifier`
- Film türü, yıl, puan özellikleri ile eğitilmiş
- AND mantığıyla çoklu tür filtresi
- Benzer film seçiminde boost mekanizması

## 📁 Proje Yapısı

```
Site/
├── app.py              # Flask backend + ML modeli
├── templates/
│   ├── home.html       # Ana sayfa (film keşfet)
│   ├── index.html      # AI öneri formu
│   ├── dataset.html    # Veri seti analizi
│   └── movie_details.html
└── static/
    ├── style.css
    └── script.js
```

## 📄 Lisans

Veri Madenciliği Projesi © 2026
