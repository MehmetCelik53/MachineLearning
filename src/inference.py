"""
Fraud Detection Model Inference
Pipeline version - uses sklearn Pipeline from 07_MLPipeline.ipynb
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
    """Fraud Detection predictor using sklearn Pipeline"""
    
    def __init__(self):
        self.pipeline = None
        self.feature_columns = None
        self.frequency_maps = None
        self._model_loaded = False
    
    def load_model(self):
        """Load pipeline, metadata and frequency maps"""
        if not self._model_loaded:
            # Load sklearn Pipeline
            self.pipeline = joblib.load(MODEL_PATH)
            
            # Load feature columns from pipeline metadata
            try:
                with open(PIPELINE_METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
                self.feature_columns = metadata['lgb_pipeline']['feature_columns']
            except FileNotFoundError:
                # Fallback: try to get from pipeline's classifier
                clf = self.pipeline.named_steps.get('classifier')
                if clf and hasattr(clf, 'feature_name_'):
                    self.feature_columns = clf.feature_name_
                else:
                    self.feature_columns = None
            
            # Try to load frequency maps (optional)
            try:
                with open(FREQUENCY_MAPS_PATH, 'r') as f:
                    self.frequency_maps = json.load(f)
            except FileNotFoundError:
                self.frequency_maps = {}
                print("Warning: frequency_maps.json not found, using defaults")
            
            self._model_loaded = True
            n_features = len(self.feature_columns) if self.feature_columns else "unknown"
            print(f"Pipeline loaded: {n_features} features")
    
    def get_frequency(self, col: str, value) -> float:
        """Get frequency encoding for a value"""
        if not self.frequency_maps:
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
        Transform raw user input into feature set expected by pipeline.
        All categorical features are frequency encoded.
        """
        self.load_model()
        
        # Extract raw values
        transaction_amt = raw_input.get('TransactionAmt', 100.0)
        
        # Build feature dict - ALL features must be numeric (frequency encoded)
        features = {
            'TransactionAmt': float(transaction_amt),
            
            # Categorical -> Frequency encoding
            'ProductCD_freq': self.get_frequency('ProductCD', raw_input.get('ProductCD', 'W')),
            'card4_freq': self.get_frequency('card4', raw_input.get('card4', 'visa')),
            'card6_freq': self.get_frequency('card6', raw_input.get('card6', 'debit')),
            
            # Distance features
            'dist1': float(raw_input.get('dist1', 0.0)),
            'dist2': float(raw_input.get('dist2', 0.0)),
            
            # Card frequency encoded features
            'card1_freq': self.get_frequency('card1', raw_input.get('card1', 10000)),
            'card2_freq': self.get_frequency('card2', raw_input.get('card2', 321.0)),
            'card3_freq': self.get_frequency('card3', raw_input.get('card3', 150.0)),
            'card5_freq': self.get_frequency('card5', raw_input.get('card5', 226.0)),
            
            # Address frequency encoded
            'addr1_freq': self.get_frequency('addr1', raw_input.get('addr1', 299.0)),
            'addr2_freq': self.get_frequency('addr2', raw_input.get('addr2', 87.0)),
            
            # Email frequency encoded
            'P_emaildomain_freq': self.get_frequency('P_emaildomain', raw_input.get('P_emaildomain', 'gmail.com')),
            'R_emaildomain_freq': self.get_frequency('R_emaildomain', raw_input.get('R_emaildomain', 'gmail.com')),
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # If model has feature_columns, ensure all exist with correct order
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.feature_columns]
        
        # Ensure all columns are float
        df = df.astype(float)
        
        return df
    
    def predict(self, raw_input: Dict) -> Tuple[float, str, str, str]:
        """
        Make fraud prediction from raw user input.
        
        Returns: (probability, risk_level, message, color)
        """
        # Ensure model is loaded
        self.load_model()
        
        try:
            # Engineer features from raw input
            df = self.engineer_features(raw_input)
            
            # Get prediction using pipeline
            probability = self.pipeline.predict_proba(df)[0, 1]
            risk_level = self.get_risk_level(probability)
            message = RISK_MESSAGES[risk_level]
            color = RISK_COLORS[risk_level]
            
            return float(probability), risk_level, message, color
        except Exception as e:
            # Return error state
            return 0.5, "medium", f"Tahmin hatası: {str(e)}", RISK_COLORS["medium"]
    
    def get_feature_names(self) -> list:
        """Get list of expected feature column names"""
        self.load_model()
        return list(self.feature_columns) if self.feature_columns else []
    
    def get_input_fields(self) -> list:
        """Get list of raw input field names for UI"""
        return [
            'TransactionAmt', 'ProductCD', 'card4', 'card6',
            'dist1', 'dist2', 'card1', 'card2', 'card3', 'card5',
            'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain'
        ]


# Singleton instance
predictor = FraudPredictor()
