"""
Fraud Detection Model Inference
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

from src.config import (
    MODEL_PATH, 
    PIPELINE_METADATA_PATH,
    FREQUENCY_MAPS_PATH,
    RISK_LEVELS, 
    RISK_MESSAGES,
    RISK_COLORS
)


class FraudPredictor:
    """Fraud Detection predictor using LightGBM Lite Pipeline"""
    
    def __init__(self):
        self.pipeline = None
        self.feature_columns = None
        self.frequency_maps = None
        self._model_loaded = False
    
    def load_model(self):
        """Load pipeline, metadata and frequency maps"""
        if not self._model_loaded:
            # Load pipeline
            self.pipeline = joblib.load(MODEL_PATH)
            
            # Load feature columns from metadata
            with open(PIPELINE_METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            self.feature_columns = metadata['feature_columns']
            
            # Load frequency maps for encoding
            with open(FREQUENCY_MAPS_PATH, 'r') as f:
                self.frequency_maps = json.load(f)
            
            self._model_loaded = True
            print(f"Model loaded: {len(self.feature_columns)} features expected")
    
    def get_frequency(self, col: str, value) -> float:
        """Get frequency encoding for a value"""
        if self.frequency_maps is None:
            return 0.001
        
        freq_map = self.frequency_maps.get(col, {})
        str_value = str(value)
        
        # Return frequency or default for unknown values
        return freq_map.get(str_value, 0.0001)
    
    def get_risk_level(self, probability: float) -> str:
        """Determine risk level based on probability"""
        for level, (low, high) in RISK_LEVELS.items():
            if low <= probability < high:
                return level
        return "high"
    
    def engineer_features(self, raw_input: Dict) -> pd.DataFrame:
        """
        Transform raw user input into full feature set expected by pipeline.
        
        Raw input contains:
        - TransactionAmt, ProductCD, card4, card6, dist1, dist2
        - card1, card2, card3, card5, addr1, addr2
        - P_emaildomain, R_emaildomain
        
        This function:
        1. Applies frequency encoding to card/addr/email fields
        2. Creates derived features expected by pipeline
        """
        self.load_model()
        
        # Extract raw values
        transaction_amt = raw_input.get('TransactionAmt', 100.0)
        
        # Build feature dict with frequency encodings
        features = {
            'TransactionAmt': transaction_amt,
            'ProductCD': raw_input.get('ProductCD', 'W'),
            'card4': raw_input.get('card4', 'visa'),
            'card6': raw_input.get('card6', 'debit'),
            'dist1': raw_input.get('dist1', 0.0),
            'dist2': raw_input.get('dist2', 0.0),
            
            # Frequency encoded features
            'card1_freq': self.get_frequency('card1', raw_input.get('card1', 10000)),
            'card2_freq': self.get_frequency('card2', raw_input.get('card2', 321.0)),
            'card3_freq': self.get_frequency('card3', raw_input.get('card3', 150.0)),
            'card5_freq': self.get_frequency('card5', raw_input.get('card5', 226.0)),
            'addr1_freq': self.get_frequency('addr1', raw_input.get('addr1', 299.0)),
            'addr2_freq': self.get_frequency('addr2', raw_input.get('addr2', 87.0)),
            'P_emaildomain_freq': self.get_frequency('P_emaildomain', raw_input.get('P_emaildomain', 'gmail.com')),
            'R_emaildomain_freq': self.get_frequency('R_emaildomain', raw_input.get('R_emaildomain', 'gmail.com')),
            
            # Derived features - defaults for pipeline compatibility
            # Removed: _D1_missing, amt_cents_exact, email_match, 3-way features
            'user_anchor_D1': 0,
            'uid_avg_amt': transaction_amt,
            'uid_std_amt': transaction_amt * 0.5,
            'uid_max_amt': transaction_amt * 1.5,
            'uid_total_amt': transaction_amt,
            'amt_vs_uid_month_avg': 1.0,
            'card1_amt_mean': transaction_amt,
            'card1_amt_max': transaction_amt * 1.5,
            'n_addr': 1,
            'n_devices': 1,
            'card1_addr1_amt_mean': transaction_amt,
            'card1_addr1_amt_std': transaction_amt * 0.3,
            'card1_FE': self.get_frequency('card1', raw_input.get('card1', 10000)),
            'addr1_FE': self.get_frequency('addr1', raw_input.get('addr1', 299.0)),
            'card1_addr1_FE': 0.001,
            'amt_vs_card1_mean': 1.0,
            'card2_FE': self.get_frequency('card2', raw_input.get('card2', 321.0)),
            'card3_FE': self.get_frequency('card3', raw_input.get('card3', 150.0)),
            'card5_FE': self.get_frequency('card5', raw_input.get('card5', 226.0)),
            'P_emaildomain_FE': self.get_frequency('P_emaildomain', raw_input.get('P_emaildomain', 'gmail.com')),
            'R_emaildomain_FE': self.get_frequency('R_emaildomain', raw_input.get('R_emaildomain', 'gmail.com')),
            'is_random_amount': 1 if (transaction_amt % 1) not in [0, 0.5, 0.99, 0.95] else 0
        }
        
        # Create DataFrame with correct column order
        df = pd.DataFrame([features])
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only required columns in correct order
        df = df[self.feature_columns]
        
        return df
    
    def predict(self, raw_input: Dict) -> Tuple[float, str, str, str]:
        """
        Make fraud prediction from raw user input.
        
        Returns: (probability, risk_level, message, color)
        """
        # Ensure model is loaded
        self.load_model()
        
        # Engineer features from raw input
        df = self.engineer_features(raw_input)
        
        # Get prediction
        probability = self.pipeline.predict_proba(df)[0, 1]
        risk_level = self.get_risk_level(probability)
        message = RISK_MESSAGES[risk_level]
        color = RISK_COLORS[risk_level]
        
        return float(probability), risk_level, message, color
    
    def get_feature_names(self) -> list:
        """Get list of input feature names"""
        return list(INPUT_FEATURES.keys())
    
    def get_feature_info(self) -> Dict:
        """Get feature information for UI"""
        return INPUT_FEATURES


# Singleton instance
predictor = FraudPredictor()
