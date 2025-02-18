from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator, ConfigDict
import pandas as pd
import pickle
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sound Realty House Price Predictor",
    description="API for predicting house prices in Seattle area",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and features
MODEL_PATH = Path("model/model.pkl")
FEATURES_PATH = Path("model/model_features.json")
DEMOGRAPHICS_PATH = Path("data/zipcode_demographics.csv")

# Global variables for model and demographics
model = None
model_features = None
demographics_data = None

class ModelManager:
    def __init__(self):
        self.models = {}
        self.features = None
        self.demographics_data = None
        self.feature_defaults = None
        self.accuracies = {
            'knn': 0.78,
            'rf': 0.85
        }

    async def load_models(self):
        """Load all available models."""
        try:
            # Check if model directory exists
            if not Path('model').exists():
                logger.error("Model directory not found")
                Path('model').mkdir(exist_ok=True)
                raise FileNotFoundError("Model directory not found")

            # Load KNN model
            knn_path = Path('model/knn_model.pkl')
            if not knn_path.exists():
                logger.error(f"KNN model file not found at {knn_path}")
                raise FileNotFoundError(f"KNN model file not found at {knn_path}")
            
            with open(knn_path, 'rb') as f:
                self.models['knn'] = pickle.load(f)
                logger.info("Successfully loaded KNN model")
            
            # Load RF model
            rf_path = Path('model/rf_model.pkl')
            if not rf_path.exists():
                logger.error(f"Random Forest model file not found at {rf_path}")
                raise FileNotFoundError(f"Random Forest model file not found at {rf_path}")
            
            with open(rf_path, 'rb') as f:
                self.models['rf'] = pickle.load(f)
                logger.info("Successfully loaded Random Forest model")
            
            # Load features
            features_path = Path('model/model_features.json')
            if not features_path.exists():
                logger.error(f"Features file not found at {features_path}")
                raise FileNotFoundError(f"Features file not found at {features_path}")
            
            with open(features_path, 'r') as f:
                self.features = json.load(f)
                logger.info(f"Loaded features: {self.features}")
            
            # Load feature defaults
            defaults_path = Path('model/feature_defaults.json')
            if not defaults_path.exists():
                logger.error(f"Feature defaults file not found at {defaults_path}")
                raise FileNotFoundError(f"Feature defaults file not found at {defaults_path}")
            
            with open(defaults_path, 'r') as f:
                self.feature_defaults = json.load(f)
                logger.info("Loaded feature defaults")
            
            # Load demographics
            demographics_path = Path('data/zipcode_demographics.csv')
            if not demographics_path.exists():
                logger.error(f"Demographics file not found at {demographics_path}")
                raise FileNotFoundError(f"Demographics file not found at {demographics_path}")
            
            self.demographics_data = pd.read_csv(demographics_path, dtype={'zipcode': str})
            logger.info(f"Loaded demographics data with {len(self.demographics_data)} records")
            
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            # Re-raise with more context
            raise Exception(f"Failed to load models and data: {str(e)}")

    def get_model(self, version: str = 'knn'):
        """Get a specific model version."""
        model = self.models.get(version)
        if model is None:
            logger.error(f"Model version {version} not found")
            raise ValueError(f"Model version {version} not found")
        return model

    def get_accuracy(self, version: str = 'knn'):
        """Get the accuracy for a specific model version."""
        accuracy = self.accuracies.get(version)
        if accuracy is None:
            logger.error(f"Accuracy not found for model version {version}")
            raise ValueError(f"Accuracy not found for model version {version}")
        return accuracy

# Initialize model manager
model_manager = ModelManager()

class PredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    version: str = 'knn'
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    sqft_above: float
    sqft_basement: float
    zipcode: str

    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        if v not in ['knn', 'rf']:
            raise ValueError('Invalid model version. Must be either "knn" or "rf"')
        return v

    @field_validator('bedrooms')
    @classmethod
    def validate_bedrooms(cls, v):
        if v < 1:
            raise ValueError('Bedrooms must be at least 1')
        return v

    @field_validator('bathrooms')
    @classmethod
    def validate_bathrooms(cls, v):
        if v < 0.5:
            raise ValueError('Bathrooms must be at least 0.5')
        return v

    @field_validator('sqft_living', 'sqft_lot', 'sqft_above')
    @classmethod
    def validate_sqft(cls, v, info):
        if v < 100:
            raise ValueError(f'{info.field_name} must be at least 100 sq ft')
        return v

    @field_validator('sqft_basement')
    @classmethod
    def validate_basement(cls, v):
        if v < 0:
            raise ValueError('Basement square footage cannot be negative')
        return v

    @field_validator('floors')
    @classmethod
    def validate_floors(cls, v):
        if v < 1:
            raise ValueError('Floors must be at least 1')
        return v

    @field_validator('zipcode')
    @classmethod
    def validate_zipcode(cls, v):
        if not v.isdigit() or len(v) != 5:
            raise ValueError('Zipcode must be a 5-digit number')
        return v

    @model_validator(mode='after')
    def validate_total_sqft(self) -> 'PredictionRequest':
        total = self.sqft_above + self.sqft_basement
        if abs(total - self.sqft_living) > 1:
            raise ValueError(
                f"Total square footage mismatch: above ground ({self.sqft_above}) + "
                f"basement ({self.sqft_basement}) should equal living area ({self.sqft_living})"
            )
        return self

class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    type: str
    version: str
    features: List[str]
    accuracy: float
    last_trained: str
    demographic_data_included: bool

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)
    
    predicted_price: float
    version: str
    confidence_score: float
    prediction_timestamp: str
    model_info: ModelInfo
    input_features: Dict[str, Any]

class MinimalPredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    sqft_living: float
    zipcode: str

    @field_validator('sqft_living')
    @classmethod
    def validate_sqft(cls, v):
        if v < 100:
            raise ValueError('Living area must be at least 100 sq ft')
        return v

    @field_validator('zipcode')
    @classmethod
    def validate_zipcode(cls, v):
        if not v.isdigit() or len(v) != 5:
            raise ValueError('Zipcode must be a 5-digit number')
        return v

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )

@app.on_event("startup")
async def startup_event():
    """Load all models on startup."""
    try:
        await model_manager.load_models()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        # In a production environment, you might want to exit here
        raise

@app.get("/health")
async def health_check():
    """Check health of all models."""
    return {
        "status": "healthy",
        "models_loaded": list(model_manager.models.keys()),
        "features_loaded": model_manager.features is not None,
        "demographics_loaded": model_manager.demographics_data is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using specified model version."""
    logger.info(f"Received prediction request for model {request.version}: {request.dict()}")
    
    try:
        # Get the requested model
        model = model_manager.get_model(request.version)
        
        # Create input dataframe with basic features
        input_data = pd.DataFrame([request.dict()])
        input_data = input_data.drop(columns=['version'])
        
        # Add default features
        defaults = model_manager.feature_defaults.copy()
        defaults['sqft_living15'] = input_data['sqft_living'].iloc[0]
        defaults['sqft_lot15'] = input_data['sqft_lot'].iloc[0]
        
        for feature, value in defaults.items():
            if feature not in input_data.columns:
                input_data[feature] = value
        
        logger.info(f"Input data created: {input_data.to_dict()}")
        
        # Merge with demographics
        if model_manager.demographics_data is None:
            raise ValueError("Demographics data not loaded")
            
        merged_data = input_data.merge(
            model_manager.demographics_data,
            how="left",
            on="zipcode"
        )
        
        if merged_data.isnull().any().any():
            logger.error(f"Invalid zipcode {request.zipcode} or missing demographic data")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid zipcode {request.zipcode} or missing demographic data"
            )
        
        # Store original features for response
        input_features = request.dict()
        del input_features['version']
        
        # Drop zipcode as it's not used in prediction
        merged_data = merged_data.drop(columns=["zipcode"])
        
        # Ensure columns are in the correct order
        if model_manager.features is None:
            raise ValueError("Model features not loaded")
            
        try:
            merged_data = merged_data[model_manager.features]
            logger.info(f"Features in correct order: {list(merged_data.columns)}")
        except KeyError as e:
            logger.error(f"Missing required feature: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Missing required feature: {str(e)}"
            )
        
        logger.info(f"Prepared data for prediction: {merged_data.to_dict()}")
        
        # Make prediction
        try:
            prediction = model.predict(merged_data)[0]
            logger.info(f"Prediction made: ${prediction:,.2f}")
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise ValueError(f"Error making prediction: {str(e)}")
        
        # Get current timestamp
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Create model info
        model_info = ModelInfo(
            type="Random Forest" if request.version == 'rf' else "KNN",
            version=request.version,
            features=model_manager.features,
            accuracy=model_manager.get_accuracy(request.version),
            last_trained="2023-10-11",
            demographic_data_included=True
        )
        
        return PredictionResponse(
            predicted_price=float(prediction),
            version=request.version,
            confidence_score=model_manager.get_accuracy(request.version),
            prediction_timestamp=timestamp,
            model_info=model_info,
            input_features=input_features
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict/minimal", response_model=PredictionResponse)
async def predict_minimal(request: MinimalPredictionRequest):
    """Endpoint for minimal feature prediction (only square footage and zipcode)"""
    logger.info(f"Received minimal prediction request: {request.dict()}")
    
    try:
        # Create full request with default values
        full_request = PredictionRequest(
            version='rf',  # Use the better model by default
            bedrooms=3,  # Default values based on median/mode
            bathrooms=2,
            sqft_living=request.sqft_living,
            sqft_lot=request.sqft_living * 2,  # Estimated lot size
            floors=1,
            sqft_above=request.sqft_living,  # Assume all above ground
            sqft_basement=0,
            zipcode=request.zipcode
        )
        
        # Use the main prediction logic
        return await predict(full_request)
    except Exception as e:
        logger.error(f"Minimal prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/models")
async def list_models():
    """List available models and their accuracies."""
    return {
        "models": [
            {
                "version": version,
                "accuracy": model_manager.get_accuracy(version),
                "type": "Random Forest" if version == 'rf' else "KNN"
            }
            for version in model_manager.models.keys()
        ]
    } 