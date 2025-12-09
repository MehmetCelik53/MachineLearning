# IEEE-CIS Fraud Detection

E-ticaret işlemlerinde fraud tespiti için makine öğrenmesi pipeline'ı. Vesta Corporation'ın IEEE-CIS Kaggle yarışması veri seti üzerinde geliştirildi.

## 1. Problem Tanımı

**İş Problemi:** E-ticaret platformlarında gerçekleşen işlemlerin fraud olup olmadığını tespit etmek. Her 100 işlemin sadece 3-4 tanesi fraud (%3.5 fraud rate) - bu ciddi bir class imbalance sorunu. Model sadece "fraud değil" dese bile %96 accuracy alır, ama bu tamamen yanıltıcı.

**Zorluklar:**
- **Class Imbalance:** Fraud oranı %3.5, standart accuracy metriği anlamsız
- **Temporal Dependency:** Fraud pattern'ları zamanla değişiyor, random split data leakage yaratır
- **Interpretability:** Regülatöre "neden bu işlemi bloke ettiniz" sorusuna cevap verebilmek şart
- **Black-box Features:** Veri setindeki 339 V feature'ı Vesta'nın proprietary skorları, ne anlama geldikleri bilinmiyor

**Veri Seti:**
- 590K eğitim, 506K test kaydı
- 434 feature (Transaction, Identity, V features)
- Kaynak: [IEEE-CIS Fraud Detection - Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

---

## 2. Baseline Süreci ve Skoru

**03_modeling.ipynb** notebook'unda iki baseline model kuruldu:

| Model | Features | Val AUC | Precision | Recall | F1 |
|-------|----------|---------|-----------|--------|-----|
| **Full Model** | 392 (tüm V, id_, D, M, C dahil) | 0.9187 | 0.2814 | 0.7237 | 0.4052 |
| **Interpretable Model** | 16 (sadece yorumlanabilir) | 0.8453 | 0.1538 | 0.6647 | 0.2498 |

**Baseline Yaklaşımı:**
- LightGBM with `class_weight='balanced'`
- Time-based split: %60 Train, %20 Val, %20 Test
- Native categorical encoding (LightGBM'in category desteği)
- Yüksek kardinaliteli değişkenler için Frequency Encoding

**Önemli Bulgu:** Full model yüksek AUC verse de 392 feature'ın çoğu black-box V değişkenleri. Interpretable model 16 feature ile AUC 0.8453 - kabul edilebilir ama iyileştirme alanı var.

---

## 3. Feature Engineering Denemeleri ve Sonuçları

**04_FeatureEngineering.ipynb** notebook'unda 16 interpretable feature'dan 36 yeni feature türetildi:

### UID (User Identification) - Chris Deotte'nin Kaggle 1. yaklaşımı
```
uid = card1 + addr1 + floor(day - D1)
```
D1 kullanıcının ilk işleminden bu yana geçen gün sayısı. Bu formül aynı kart+adres kombinasyonunu tutarlı şekilde tanımlıyor.

### Türetilen Feature Grupları

| Grup | Features | Açıklama |
|------|----------|----------|
| **UID Aggregations** | uid_avg_amt, uid_std_amt, uid_max_amt, uid_txn_count | Kullanıcı bazlı işlem istatistikleri |
| **Temporal UID** | uid_week_avg_amt, uid_month_avg_amt, uid_week_txn_count | Haftalık/aylık kullanıcı davranışı |
| **Anomaly Ratios** | amt_vs_uid_avg, amt_vs_uid_week_avg | İşlem tutarının kullanıcı ortalamasına oranı |
| **Card Aggregations** | card1_amt_mean, card1_amt_std, n_addr, n_devices | Kart bazlı istatistikler |
| **Frequency Encodings** | card1_FE, addr1_FE, card1_addr1_FE, addr2_freq, card3_freq | Frekans bazlı encoding |
| **Amount Features** | amt_dollars, user_anchor_D1 | İşlem tutarı türevleri |

### Feature Engineering Sonuçları (04_FE sonrası)

| Metric | Baseline (16 feat) | FE Model (36 feat) | Değişim |
|--------|-------------------|-------------------|---------|
| Val AUC | 0.8453 | 0.8766 | +3.7% |
| Precision | 0.1538 | 0.2100 | +36.5% |
| Recall | 0.6647 | 0.6800 | +2.3% |

**Anahtar Bulgular:**
- Kullanıcı bazlı aggregation'lar (uid_avg_amt, uid_std_amt) en etkili feature'lar
- Anomaly ratio'lar (amt_vs_uid_avg) fraud pattern'larını iyi yakalıyor
- Frequency encoding'ler nadir değerleri tanımlamada güçlü

---

## 4. Validasyon Şeması

**Seçim: Time-Based GroupKFold**

### Neden Time-Based?
Fraud detection'da random split kullanmak data leakage yaratır. Model gelecekteki işlemleri görerek öğrenir - production'da böyle bir lüks yok. Örneğin random split'te Kasım verisinden öğrenip Ekim'i tahmin edebilir, bu gerçekçi değil.

### Uygulama
```python
# 05_ModelOptimization.ipynb
CV = GroupKFold(n_splits=3)
GROUPS = month_num  # Aylar grup etiketi olarak kullanılıyor

# GroupKFold mantığı: Aynı ay içindeki işlemler asla train ve val'e ayrılmaz
# 3 fold ile veri 3 parçaya bölünüyor, her fold'da farklı ay(lar) validation oluyor
# Fold 1: Train months [5,6] -> Val months [7]
# Fold 2: Train months [5,7] -> Val months [6]  
# Fold 3: Train months [6,7] -> Val months [5]
```

### Avantajları
- **Gerçekçi:** Her fold'da model geçmişi görüp geleceği tahmin ediyor
- **Data leakage yok:** Temporal bağımlılık korunuyor
- **Stabil skorlar:** CV std 0.01-0.02 aralığında, model tutarlı

### Train/Val/Test Split
- %60 Train (erken dönem)
- %20 Validation (orta dönem)
- %20 Test (son dönem - sadece final evaluation)

---

## 5. Final Pipeline Feature Seti ve Ön İşleme

### Feature Selection: SHAP + Native Importance Rank Sum

**06_ModelEvaluation.ipynb**'da her feature için iki rank hesaplandı:
1. **SHAP Rank:** SHAP importance'a göre sıralama
2. **Native Rank:** LightGBM'in native feature importance'ına göre sıralama

```
rank_sum = shap_rank + native_rank
```

**Threshold Analizi (50-72 arası test edildi):**

| Threshold | Features | AUC | Recall | Excluded |
|-----------|----------|-----|--------|----------|
| 64 | 33 | 0.8820 | 0.6326 | addr2_freq, card3_freq, uid_month_avg_amt |
| **66** | **34** | **0.8842** | **0.6513** | **addr2_freq, card3_freq** |
| 68 | 34 | 0.8842 | 0.6513 | addr2_freq, card3_freq |
| 72 | 36 | 0.8811 | 0.6326 | (none) |

**Seçim: Threshold = 66**
- 34 feature (2 excluded: `addr2_freq`, `card3_freq`)
- En iyi AUC + Recall dengesi
- Elenen feature'lar hem SHAP hem native metrikte düşük sırada

### Ön İşleme Stratejisi
1. **Missing Values:** D1 için 0 ile doldur (ilk işlem varsayımı)
2. **Categorical:** LightGBM native categorical support
3. **High Cardinality:** Frequency Encoding (P_emaildomain, R_emaildomain, DeviceInfo)
4. **Feature Selection:** SHAP + Native rank sum <= 66

### Final Feature List (34 features)
```
TransactionAmt, ProductCD, card1, card2, card4, card5, card6, addr1, 
P_emaildomain, R_emaildomain, D1, DeviceType, uid_avg_amt, uid_std_amt,
uid_max_amt, uid_min_amt, uid_total_amt, uid_txn_count, uid_week_avg_amt,
uid_week_txn_count, uid_month_txn_count, amt_vs_uid_avg, amt_vs_uid_week_avg,
amt_vs_uid_month_avg, card1_amt_mean, card1_amt_std, card1_amt_max, n_addr,
n_devices, card1_addr1_amt_mean, card1_addr1_amt_std, card1_FE, addr1_FE,
card1_addr1_FE
```

---

## 6. Final Model vs Baseline Başarı Farkı

### Model Karşılaştırması

| Model | Features | Val AUC | Precision | Recall | F1 |
|-------|----------|---------|-----------|--------|-----|
| 03_Baseline Full | 392 | 0.9187 | 0.2814 | 0.7237 | 0.4052 |
| 03_Baseline Interp | 16 | 0.8453 | 0.1538 | 0.6647 | 0.2498 |
| **07_Pipeline Final** | **34** | **0.8842** | **0.2512** | **0.6513** | **0.3625** |

### İyileştirme Analizi

**Interpretable Baseline → Final Pipeline:**
- Features: 16 → 34 (+18 feature engineering ile)
- AUC: 0.8453 → 0.8842 (+4.6%)
- Recall: 0.6647 → 0.6513 (-2.0%)
- F1: 0.2498 → 0.3625 (+45.1%)

**Full Baseline vs Final Pipeline:**
- Features: 392 → 34 (-91% feature azaltma)
- AUC: 0.9187 → 0.8842 (-3.8%)
- Interpretability: Black-box → Fully explainable

### Sonuç
Black-box V features olmadan, sadece 34 interpretable feature ile Full baseline'ın %96.2'si performansa ulaştık. Trade-off kabul edilebilir çünkü:
- Her tahmin SHAP ile açıklanabilir
- Feature'lar business-meaningful
- Regulatory compliance sağlanıyor

---

## 7. Business Gereksinimleri ile Uyum

### Gereksinim 1: Yüksek Recall (Fraud Kaçırma Maliyeti)
✅ **Recall: 0.6513** - Her 100 fraud'dan 65'ini yakalıyoruz  
⚠️ 35 fraud kaçıyor, ama precision da önemli - çok fazla false alarm müşteri deneyimini bozar

### Gereksinim 2: Precision/False Alarm Dengesi
✅ **Precision: 0.2512** - Her 4 alarm'dan 1'i gerçek fraud  
Operasyon ekibi kapasitesi dahilinde. False positive oranı kabul edilebilir.

### Gereksinim 3: Açıklanabilirlik (Regulatory)
✅ **SHAP Integration** - Her tahmin için "neden" sorusuna cevap var  
Waterfall plot ile tek işlem bazında açıklama üretilebilir.

### Gereksinim 4: Operasyonel Verimlilik
✅ **Top 10% Lift: 8-10x** - En riskli %10'a odaklanarak fraud yakalama 8-10 kat artıyor  
✅ **KS: 0.55 (Excellent)** - Fraud/non-fraud ayrımı çok güçlü

### Gereksinim 5: Risk Segmentasyonu
✅ **5 Risk Kategorisi:**
- Very Low (0-10%): Otomatik onay
- Low (10-30%): Hafif monitoring
- Medium (30-50%): Enhanced review
- High (50-70%): Manuel inceleme
- Very High (70-100%): Otomatik blok

---

## 8. Production Deployment ve Monitoring

### Deployment Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION FLOW                          │
├─────────────────────────────────────────────────────────────┤
│  Transaction → Feature Eng → Model → Risk Score → Decision  │
│                                                             │
│  [Raw Data]   [36 features]  [LightGBM]  [0-1]   [5 bucket] │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Adımları

1. **Model Yükleme:**
```python
import joblib
pipeline = joblib.load('models/fraud_detection/pipeline/lgb_pipeline.pkl')
```

2. **Feature Engineering:**
```python
# Frequency maps yükle
with open('models/fraud_detection/frequency_maps.json', 'r') as f:
    freq_maps = json.load(f)
```

3. **Inference:**
```python
def predict_fraud(transaction, threshold=0.5):
    features = engineer_features(transaction, freq_maps)
    proba = pipeline.predict_proba(features)[0, 1]
    risk = categorize_risk(proba)
    return {'probability': proba, 'risk': risk, 'decision': proba >= threshold}
```

### Monitoring Metrikleri

| Metrik | Açıklama | Alarm Eşiği | Aksiyon |
|--------|----------|-------------|---------|
| **Daily AUC** | Günlük model performansı | < 0.80 | Model refresh tetikle |
| **Fraud Rate** | Günlük fraud oranı | > 5% veya < 2% | Distribution shift kontrolü |
| **False Positive Rate** | Yanlış alarm oranı | > 80% | Threshold ayarla |
| **Prediction Latency** | Tahmin süresi (p99) | > 100ms | Infra optimizasyonu |
| **Feature Drift** | Feature dağılım değişimi | PSI > 0.2 | Feature engineering review |

### Model Refresh Planı

| Periyot | Aksiyon |
|---------|---------|
| **Haftalık** | AUC, precision, recall monitoring |
| **Aylık** | Feature distribution kontrolü (PSI) |
| **3 Aylık** | Full model retrain + hyperparameter optimization |
| **6 Aylık** | Feature engineering review + yeni feature ekleme |

### Shadow Mode Deployment
İlk 2 hafta mevcut rule-based sistem ile paralel çalıştır:
1. Her iki sistemin kararlarını karşılaştır
2. Disagreement case'leri analiz et
3. Güven oluştuktan sonra kademeli geçiş

---

## Proje Yapısı

```
├── data/
│   ├── processed/           # Train/val/test splits
│   ├── train_optimized.csv  # Optimize edilmiş veri
│   └── frequency_maps.json  # Encoding map'leri
├── models/
│   └── fraud_detection/
│       ├── optimized/       # Optuna ile tune edilmiş model
│       └── pipeline/        # Production pipeline
├── notebooks/
│   └── modeling/
│       ├── 01_fixingdtypes.ipynb    # Veri tipi optimizasyonu
│       ├── 02_EDA.ipynb             # Keşifsel analiz
│       ├── 03_modeling.ipynb        # Baseline modeller
│       ├── 04_FeatureEngineering.ipynb  # Feature türetme
│       ├── 05_ModelOptimization.ipynb   # Optuna hyperparameter tuning
│       ├── 06_ModelEvaluation.ipynb     # SHAP, KS, Decile analizi
│       └── 07_MLPipeline.ipynb          # Final pipeline
├── src/
│   ├── app.py              # Streamlit demo
│   ├── config.py           # Uygulama konfigürasyonu
│   └── inference.py        # Production inference
└── README.md
```

---

## Demo Uygulaması (Streamlit)

Fraud detection modelini interaktif olarak test edebileceğiniz bir web arayüzü.

### Kurulum ve Çalıştırma

```bash
# 1. Gereksinimleri yükle
pip install -r requirements.txt

# 2. Streamlit uygulamasını başlat
streamlit run src/app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılır.

### Özellikler

- **Manuel İşlem Girişi:** Form üzerinden işlem detaylarını girerek fraud olasılığı hesaplama
- **Risk Skoru:** 0-100 arası fraud olasılığı ve 5 kademeli risk kategorisi (Very Low → Very High)
- **SHAP Açıklaması:** Her tahmin için hangi feature'ların kararı nasıl etkilediğini görme
- **Batch Prediction:** CSV dosyası yükleyerek toplu tahmin yapma

### Ekran Görüntüsü

```
┌─────────────────────────────────────────────────────────────┐
│  🔍 Fraud Detection System                              │
├─────────────────────────────────────────────────────────────┤
│  Transaction Amount: [___$150___]                       │
│  Product Code:       [___W___▼]                         │
│  Card Brand:         [___visa___▼]                      │
│  ...                                                    │
│                                                         │
│  [🔍 Analyze Transaction]                               │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Risk Score: 73%  [██████████░░░░] HIGH         │   │
│  │  Recommendation: Manuel inceleme gerekli        │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---