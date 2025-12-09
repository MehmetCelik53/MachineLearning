"""
Fraud Detection Model Configuration
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection" / "pipeline" / "lgb_pipeline_lite.pkl"
PIPELINE_METADATA_PATH = BASE_DIR / "models" / "fraud_detection" / "pipeline" / "pipeline_metadata.json"
FREQUENCY_MAPS_PATH = BASE_DIR / "models" / "fraud_detection" / "frequency_maps.json"

# Model settings
PROBABILITY_THRESHOLD = 0.50

# ============================================================
# USER INPUT OPTIONS - Kullanıcının seçeceği değerler
# ============================================================

# İşlem Tutarı
TRANSACTION_AMT = {
    'min': 0.0,
    'max': 50000.0,
    'default': 100.0,
    'description': 'İşlem tutarı (USD)'
}

# Ürün Tipi
PRODUCT_CODES = {
    'W': 'W - Web İşlemi',
    'H': 'H - Hardware',
    'C': 'C - Cash',
    'S': 'S - Service',
    'R': 'R - Retail'
}

# Kart Markası
CARD_BRANDS = {
    'visa': 'Visa',
    'mastercard': 'Mastercard', 
    'american express': 'American Express',
    'discover': 'Discover'
}

# Kart Tipi
CARD_TYPES = {
    'debit': 'Banka Kartı (Debit)',
    'credit': 'Kredi Kartı (Credit)'
}

# Email Domain'leri - Purchaser (Satın Alan)
P_EMAIL_DOMAINS = [
    ('gmail.com', 'Gmail (gmail.com) - %46'),
    ('yahoo.com', 'Yahoo (yahoo.com) - %20'),
    ('hotmail.com', 'Hotmail (hotmail.com) - %9'),
    ('anonymous.com', 'Anonim (anonymous.com) - %7'),
    ('aol.com', 'AOL (aol.com) - %6'),
    ('comcast.net', 'Comcast (comcast.net)'),
    ('icloud.com', 'iCloud (icloud.com)'),
    ('outlook.com', 'Outlook (outlook.com)'),
    ('msn.com', 'MSN (msn.com)'),
    ('att.net', 'AT&T (att.net)'),
    ('verizon.net', 'Verizon (verizon.net)'),
    ('live.com', 'Live (live.com)'),
    ('ymail.com', 'YMail (ymail.com)'),
    ('other', 'Diğer (nadir görülen)')
]

# Email Domain'leri - Recipient (Alıcı Taraf)
R_EMAIL_DOMAINS = [
    ('gmail.com', 'Gmail (gmail.com) - %42'),
    ('hotmail.com', 'Hotmail (hotmail.com) - %20'),
    ('anonymous.com', 'Anonim (anonymous.com) - %15'),
    ('yahoo.com', 'Yahoo (yahoo.com) - %9'),
    ('aol.com', 'AOL (aol.com)'),
    ('outlook.com', 'Outlook (outlook.com)'),
    ('comcast.net', 'Comcast (comcast.net)'),
    ('icloud.com', 'iCloud (icloud.com)'),
    ('live.com', 'Live (live.com)'),
    ('other', 'Diğer (nadir görülen)')
]

# Kart Numaraları (Anonimleştirilmiş ID'ler - veri setindeki gerçek aralıklar)
CARD_RANGES = {
    'card1': {
        'min': 1000, 'max': 19000, 'default': 10000, 'step': 100,
        'label': 'Kart ID 1',
        'help': 'Kart tanımlayıcı (1000-19000 arası). Yaygın değerler: 7919, 9500, 15885'
    },
    'card2': {
        'min': 100.0, 'max': 600.0, 'default': 321.0, 'step': 1.0,
        'label': 'Kart ID 2',
        'help': 'Kart grubu (100-600 arası). Yaygın değerler: 321, 111, 555, 490'
    },
    'card3': {
        'min': 100.0, 'max': 230.0, 'default': 150.0, 'step': 1.0,
        'label': 'Kart Tipi Kodu',
        'help': 'Kart tipi (100-230 arası). En yaygın: 150 (%88), 185 (%10)'
    },
    'card5': {
        'min': 100.0, 'max': 240.0, 'default': 226.0, 'step': 1.0,
        'label': 'Kart Kategorisi',
        'help': 'Kart kategorisi (100-240 arası). En yaygın: 226 (%51), 224 (%14)'
    }
}

# Adres Kodları (Anonimleştirilmiş)
ADDRESS_RANGES = {
    'addr1': {
        'min': 100.0, 'max': 540.0, 'default': 299.0, 'step': 1.0,
        'label': 'Fatura Adresi Kodu',
        'help': 'Adres kodu (100-540 arası). Yaygın değerler: 299, 325, 204, 264'
    },
    'addr2': {
        'min': 10.0, 'max': 100.0, 'default': 87.0, 'step': 1.0,
        'label': 'Bölge Kodu',
        'help': 'Bölge/Ülke kodu (10-100 arası). %99 işlem 87 kodlu'
    }
}

# Mesafe Metrikleri
DISTANCE_RANGES = {
    'dist1': {
        'min': 0.0, 'max': 10000.0, 'default': 0.0, 'step': 1.0,
        'label': 'Mesafe 1',
        'help': 'Fatura ve kargo adresi arası mesafe. 0 = aynı adres'
    },
    'dist2': {
        'min': 0.0, 'max': 10000.0, 'default': 0.0, 'step': 1.0,
        'label': 'Mesafe 2',
        'help': 'İkincil mesafe metriği. 0 = aynı lokasyon'
    }
}

# Risk Levels
RISK_LEVELS = {
    "low": (0.0, 0.3),
    "medium": (0.3, 0.6),
    "high": (0.6, 1.0)
}

RISK_MESSAGES = {
    "low": "✅ Düşük Risk - İşlem güvenli görünüyor",
    "medium": "⚠️ Orta Risk - Manuel inceleme önerilir",
    "high": "🚨 Yüksek Risk - İşlem muhtemelen sahte"
}

RISK_COLORS = {
    "low": "#28a745",      # Green
    "medium": "#ffc107",   # Yellow
    "high": "#dc3545"      # Red
}

# App settings
APP_TITLE = "Fraud Detection System"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "LightGBM Lite Pipeline ile Gerçek Zamanlı Dolandırıcılık Tespiti"
