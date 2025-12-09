"""
Fraud Detection Streamlit Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    APP_TITLE, 
    APP_VERSION, 
    APP_DESCRIPTION,
    RISK_COLORS,
    PRODUCT_CODES,
    CARD_BRANDS,
    CARD_TYPES,
    P_EMAIL_DOMAINS,
    R_EMAIL_DOMAINS,
    CARD_RANGES,
    ADDRESS_RANGES,
    DISTANCE_RANGES,
    TRANSACTION_AMT
)
from src.inference import predictor


# Page config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .section-title {
        color: #1f77b4;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .help-text {
        font-size: 0.8rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown(f'<div class="main-header">🔍 {APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{APP_DESCRIPTION} | v{APP_VERSION}</div>', unsafe_allow_html=True)
    
    # Sidebar - Info
    with st.sidebar:
        st.header("ℹ️ Hakkında")
        st.markdown("""
        Bu uygulama **LightGBM** modeli kullanarak 
        gerçek zamanlı dolandırıcılık tespiti yapar.
        
        **Model Performansı:**
        - 🎯 Validation AUC: 0.9395
        - 📊 809 özellik (Full Model)
        - ⚡ Hızlı inference
        
        **Nasıl Çalışır:**
        1. İşlem bilgilerini girin
        2. "Tahmin Et" butonuna tıklayın
        3. Risk seviyesini görün
        """)
        
        st.divider()
        st.header("📊 Risk Seviyeleri")
        st.markdown("""
        - 🟢 **Düşük (0-30%)**: Güvenli işlem
        - 🟡 **Orta (30-60%)**: Manuel inceleme
        - 🔴 **Yüksek (60-100%)**: Riskli işlem
        """)
        
        st.divider()
        st.caption("IEEE-CIS Fraud Detection verisi ile eğitildi")
    
    # Main content - Two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 İşlem Bilgileri")
        
        # ============ İŞLEM BİLGİLERİ ============
        with st.expander("💰 İşlem Bilgileri", expanded=True):
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                transaction_amt = st.number_input(
                    "İşlem Tutarı ($)",
                    min_value=TRANSACTION_AMT['min'],
                    max_value=TRANSACTION_AMT['max'],
                    value=TRANSACTION_AMT['default'],
                    step=10.0,
                    help=TRANSACTION_AMT['description']
                )
            
            with col_t2:
                product_cd = st.selectbox(
                    "Ürün Tipi",
                    options=list(PRODUCT_CODES.keys()),
                    format_func=lambda x: PRODUCT_CODES[x],
                    index=0,
                    help="İşlem yapılan ürün kategorisi"
                )
        
        # ============ KART BİLGİLERİ ============
        with st.expander("💳 Kart Bilgileri", expanded=True):
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                card4 = st.selectbox(
                    "Kart Markası",
                    options=list(CARD_BRANDS.keys()),
                    format_func=lambda x: CARD_BRANDS[x],
                    index=0
                )
            
            with col_c2:
                card6 = st.selectbox(
                    "Kart Tipi",
                    options=list(CARD_TYPES.keys()),
                    format_func=lambda x: CARD_TYPES[x],
                    index=0
                )
            
            st.markdown("---")
            st.markdown("**Kart Tanımlayıcıları** *(Anonimleştirilmiş değerler)*")
            
            col_card1, col_card2 = st.columns(2)
            
            with col_card1:
                card1 = st.number_input(
                    CARD_RANGES['card1']['label'],
                    min_value=CARD_RANGES['card1']['min'],
                    max_value=CARD_RANGES['card1']['max'],
                    value=CARD_RANGES['card1']['default'],
                    step=CARD_RANGES['card1']['step'],
                    help=CARD_RANGES['card1']['help']
                )
                
                card3 = st.number_input(
                    CARD_RANGES['card3']['label'],
                    min_value=CARD_RANGES['card3']['min'],
                    max_value=CARD_RANGES['card3']['max'],
                    value=CARD_RANGES['card3']['default'],
                    step=CARD_RANGES['card3']['step'],
                    help=CARD_RANGES['card3']['help']
                )
            
            with col_card2:
                card2 = st.number_input(
                    CARD_RANGES['card2']['label'],
                    min_value=CARD_RANGES['card2']['min'],
                    max_value=CARD_RANGES['card2']['max'],
                    value=CARD_RANGES['card2']['default'],
                    step=CARD_RANGES['card2']['step'],
                    help=CARD_RANGES['card2']['help']
                )
                
                card5 = st.number_input(
                    CARD_RANGES['card5']['label'],
                    min_value=CARD_RANGES['card5']['min'],
                    max_value=CARD_RANGES['card5']['max'],
                    value=CARD_RANGES['card5']['default'],
                    step=CARD_RANGES['card5']['step'],
                    help=CARD_RANGES['card5']['help']
                )
        
        # ============ ADRES BİLGİLERİ ============
        with st.expander("📍 Adres ve Mesafe Bilgileri", expanded=False):
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                addr1 = st.number_input(
                    ADDRESS_RANGES['addr1']['label'],
                    min_value=ADDRESS_RANGES['addr1']['min'],
                    max_value=ADDRESS_RANGES['addr1']['max'],
                    value=ADDRESS_RANGES['addr1']['default'],
                    step=ADDRESS_RANGES['addr1']['step'],
                    help=ADDRESS_RANGES['addr1']['help']
                )
                
                dist1 = st.number_input(
                    DISTANCE_RANGES['dist1']['label'],
                    min_value=DISTANCE_RANGES['dist1']['min'],
                    max_value=DISTANCE_RANGES['dist1']['max'],
                    value=DISTANCE_RANGES['dist1']['default'],
                    step=DISTANCE_RANGES['dist1']['step'],
                    help=DISTANCE_RANGES['dist1']['help']
                )
            
            with col_a2:
                addr2 = st.number_input(
                    ADDRESS_RANGES['addr2']['label'],
                    min_value=ADDRESS_RANGES['addr2']['min'],
                    max_value=ADDRESS_RANGES['addr2']['max'],
                    value=ADDRESS_RANGES['addr2']['default'],
                    step=ADDRESS_RANGES['addr2']['step'],
                    help=ADDRESS_RANGES['addr2']['help']
                )
                
                dist2 = st.number_input(
                    DISTANCE_RANGES['dist2']['label'],
                    min_value=DISTANCE_RANGES['dist2']['min'],
                    max_value=DISTANCE_RANGES['dist2']['max'],
                    value=DISTANCE_RANGES['dist2']['default'],
                    step=DISTANCE_RANGES['dist2']['step'],
                    help=DISTANCE_RANGES['dist2']['help']
                )
        
        # ============ EMAIL BİLGİLERİ ============
        with st.expander("📧 Email Domain Bilgileri", expanded=False):
            st.markdown("*Email domain'leri fraud tespitinde önemli bir sinyal. Gmail ve Hotmail en yaygın domain'ler.*")
            
            col_e1, col_e2 = st.columns(2)
            
            with col_e1:
                p_email = st.selectbox(
                    "Satın Alan Email Domain",
                    options=[d[0] for d in P_EMAIL_DOMAINS],
                    format_func=lambda x: dict(P_EMAIL_DOMAINS).get(x, x),
                    index=0,
                    help="Ödemeyi yapan kişinin email domain'i"
                )
            
            with col_e2:
                r_email = st.selectbox(
                    "Alıcı Email Domain",
                    options=[d[0] for d in R_EMAIL_DOMAINS],
                    format_func=lambda x: dict(R_EMAIL_DOMAINS).get(x, x),
                    index=0,
                    help="Ürünü alan kişinin email domain'i"
                )
        
        # Predict button
        st.divider()
        predict_btn = st.button("🔮 Tahmin Et", type="primary", use_container_width=True)
    
    with col2:
        st.header("📊 Sonuç")
        
        if predict_btn:
            # Prepare raw input for prediction
            raw_input = {
                'TransactionAmt': transaction_amt,
                'ProductCD': product_cd,
                'card4': card4,
                'card6': card6,
                'card1': card1,
                'card2': card2,
                'card3': card3,
                'card5': card5,
                'addr1': addr1,
                'addr2': addr2,
                'dist1': dist1,
                'dist2': dist2,
                'P_emaildomain': p_email,
                'R_emaildomain': r_email
            }
            
            try:
                # Make prediction
                with st.spinner("İşlem analiz ediliyor..."):
                    probability, risk_level, message, color = predictor.predict(raw_input)
                
                # Display result
                st.markdown(f"""
                <div class="result-box" style="background-color: {color}20; border: 2px solid {color};">
                    <h3 style="color: {color}; margin: 0;">{message}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col_m1, col_m2 = st.columns(2)
                
                with col_m1:
                    st.metric(
                        label="Fraud Olasılığı",
                        value=f"{probability*100:.1f}%"
                    )
                
                with col_m2:
                    risk_tr = {"low": "DÜŞÜK", "medium": "ORTA", "high": "YÜKSEK"}
                    st.metric(
                        label="Risk Seviyesi",
                        value=risk_tr.get(risk_level, risk_level.upper())
                    )
                
                # Progress bar
                st.progress(probability)
                
                # Additional info
                st.divider()
                st.subheader("📋 İşlem Özeti")
                
                summary_data = {
                    'Özellik': ['Tutar', 'Ürün', 'Kart Markası', 'Kart Tipi', 'Satın Alan Email', 'Alıcı Email'],
                    'Değer': [
                        f"${transaction_amt:,.2f}", 
                        PRODUCT_CODES[product_cd],
                        CARD_BRANDS[card4], 
                        CARD_TYPES[card6],
                        p_email,
                        r_email
                    ]
                }
                st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
                
                # Risk factors
                st.divider()
                st.subheader("⚠️ Risk Faktörleri")
                
                risk_factors = []
                if transaction_amt > 1000:
                    risk_factors.append("• Yüksek işlem tutarı")
                if p_email == 'anonymous.com' or r_email == 'anonymous.com':
                    risk_factors.append("• Anonim email kullanımı")
                if p_email != r_email:
                    risk_factors.append("• Farklı email domain'leri")
                if dist1 > 100 or dist2 > 100:
                    risk_factors.append("• Yüksek mesafe değeri")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(factor)
                else:
                    st.success("Belirgin risk faktörü tespit edilmedi")
                
            except Exception as e:
                st.error(f"Tahmin hatası: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        else:
            # Default state
            st.info("👈 İşlem bilgilerini girip **Tahmin Et** butonuna tıklayın")
            
            # Sample transactions
            st.divider()
            st.subheader("📌 Hızlı Test")
            st.markdown("*Örnek senaryoları test etmek için aşağıdaki değerleri kullanın:*")
            
            st.markdown("""
            **🟢 Düşük Riskli İşlem:**
            - Tutar: $100
            - Visa, Debit
            - Gmail → Gmail
            - Mesafe: 0
            
            **🔴 Yüksek Riskli İşlem:**
            - Tutar: $5000+
            - Discover, Credit
            - Anonymous email
            - Yüksek mesafe
            """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.85rem;">
        Fraud Detection System v1.0 | LightGBM Lite Pipeline | 
        <a href="https://github.com/MehmetCelik53/MachineLearning" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
