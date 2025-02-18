import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import pickle
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Load and prepare the data."""
    try:
        # Load datasets
        house_data = pd.read_csv('data/kc_house_data.csv', dtype={'zipcode': str})
        demographics = pd.read_csv('data/zipcode_demographics.csv', dtype={'zipcode': str})
        
        # Merge datasets
        data = house_data.merge(demographics, on='zipcode', how='left')
        logger.info(f"Loaded {len(data)} records")
        
        # Fill missing values
        data['waterfront'] = data['waterfront'].fillna(0)
        data['view'] = data['view'].fillna(0)
        data['yr_renovated'] = data['yr_renovated'].fillna(0)
        data['condition'] = data['condition'].fillna(data['condition'].median())
        data['grade'] = data['grade'].fillna(data['grade'].median())
        data['sqft_living15'] = data['sqft_living15'].fillna(data['sqft_living'])
        data['sqft_lot15'] = data['sqft_lot15'].fillna(data['sqft_lot'])
        
        # Split features and target
        y = data['price']
        X = data.drop(['price', 'id', 'date', 'zipcode'], axis=1)
        
        # Save feature names
        feature_names = list(X.columns)
        Path('model').mkdir(exist_ok=True)
        with open('model/model_features.json', 'w') as f:
            json.dump(feature_names, f)
        logger.info(f"Saved {len(feature_names)} feature names")
        
        # Save feature defaults for prediction
        feature_defaults = {
            'waterfront': 0,
            'view': 0,
            'condition': int(data['condition'].median()),
            'grade': int(data['grade'].median()),
            'yr_built': int(data['yr_built'].median()),
            'yr_renovated': 0,
            'lat': float(data['lat'].median()),
            'long': float(data['long'].median()),
            'sqft_living15': None,  # Will be set to sqft_living
            'sqft_lot15': None,  # Will be set to sqft_lot
        }
        
        with open('model/feature_defaults.json', 'w') as f:
            json.dump(feature_defaults, f)
        logger.info("Saved feature defaults")
        
        return X, y
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_knn_model(X, y):
    """Train and save KNN model."""
    try:
        # Create and train KNN pipeline
        knn_pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', KNeighborsRegressor())
        ])
        
        knn_pipeline.fit(X, y)
        logger.info("Trained KNN model")
        
        # Save model
        with open('model/knn_model.pkl', 'wb') as f:
            pickle.dump(knn_pipeline, f)
        logger.info("Saved KNN model")
        
        return knn_pipeline
    except Exception as e:
        logger.error(f"Error training KNN model: {str(e)}")
        raise

def train_rf_model(X, y):
    """Train and save Random Forest model."""
    try:
        # Create and train RF pipeline
        rf_pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ))
        ])
        
        rf_pipeline.fit(X, y)
        logger.info("Trained Random Forest model")
        
        # Save model
        with open('model/rf_model.pkl', 'wb') as f:
            pickle.dump(rf_pipeline, f)
        logger.info("Saved Random Forest model")
        
        return rf_pipeline
    except Exception as e:
        logger.error(f"Error training Random Forest model: {str(e)}")
        raise

def main():
    """Main function to train and save models."""
    try:
        logger.info("Starting model training")
        
        # Load data
        X, y = load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train models
        knn_model = train_knn_model(X_train, y_train)
        rf_model = train_rf_model(X_train, y_train)
        
        # Evaluate models
        knn_score = knn_model.score(X_test, y_test)
        rf_score = rf_model.score(X_test, y_test)
        
        logger.info(f"KNN R² score: {knn_score:.4f}")
        logger.info(f"Random Forest R² score: {rf_score:.4f}")
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 