import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

class PricingEngine:
    """
    Dynamic Pricing Engine using XGBoost Classification.
    
    Uses probability-based revenue maximization:
    Revenue = Price × P(booked | Price, features)
    Optimal Price = argmax(Revenue)
    """
    
    def __init__(self, features):
        self.features = features
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.target = 'is_booked'
        self.is_trained = False

    def train(self, df):
        """
        Trains the XGBoost classifier to predict booking probability.
        
        Returns:
            X_test, y_test: Test data for further evaluation
        """
        model_df = df.dropna(subset=self.features + [self.target])
        X = model_df[self.features]
        y = model_df[self.target].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Training XGBoost Classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate with classification metrics
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Model AUC: {auc:.4f}")
        print(f"Model Accuracy: {acc:.4f}")
        
        return X_test, y_test

    def predict_booking_probability(self, df):
        """
        Predicts the probability of booking for given features.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(df[self.features])[:, 1]

    def get_optimal_price(self, base_features_dict, price_range=None):
        """
        Calculates optimal price using probability-based revenue maximization.
        
        Revenue = Price × P(booked | Price, features)
        
        Args:
            base_features_dict: Dict of feature values (excluding price)
            price_range: Array of candidate prices to evaluate
            
        Returns:
            opt_price: Optimal price that maximizes expected revenue
            max_revenue: Expected revenue at optimal price
            elasticity: Price elasticity of demand at optimal price
            price_range: Array of prices evaluated
            booking_probs: Booking probabilities at each price
        """
        if price_range is None:
            price_range = np.linspace(30, 500, 100)
        
        # Create dataframe with all candidate prices
        temp_df = pd.DataFrame([base_features_dict] * len(price_range))
        temp_df['price'] = price_range
        
        # Predict booking probability at each price point
        booking_probs = self.predict_booking_probability(temp_df)
        
        # Calculate expected revenue: Price × P(booked)
        expected_revenues = price_range * booking_probs
        
        # Find optimal price (maximize revenue)
        idx_max = np.argmax(expected_revenues)
        opt_price = price_range[idx_max]
        max_revenue = expected_revenues[idx_max]
        
        # Calculate price elasticity at optimal price using numerical differentiation
        # Elasticity = (dQ/dP) × (P/Q)
        if idx_max > 0 and idx_max < len(price_range) - 1:
            dQ = booking_probs[idx_max + 1] - booking_probs[idx_max - 1]
            dP = price_range[idx_max + 1] - price_range[idx_max - 1]
            elasticity = (dQ / dP) * (opt_price / booking_probs[idx_max])
        else:
            elasticity = np.nan
        
        return opt_price, max_revenue, elasticity, price_range, booking_probs

    def get_feature_importance(self):
        """Returns feature importance from the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return dict(zip(self.features, self.model.feature_importances_))
