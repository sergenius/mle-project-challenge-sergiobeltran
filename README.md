# Sound Realty House Price Predictor

A machine learning-powered application that helps predict house prices in the Seattle area, combining property features with demographic data for accurate estimations.

## Project Overview

This project implements a house price prediction service for Sound Realty, featuring:
- A machine learning model trained on Seattle area housing data
- A REST API for serving predictions
- A user-friendly web interface for realtors
- Integration of demographic data for better predictions

## Key Features

- **Multiple Models**:
  - KNN Model (R² ≈ 78%)
  - Random Forest Model (R² ≈ 85%)
  - Model selection in UI
  
- **Smart Predictions**:
  - Comprehensive feature set
  - Demographic data integration
  - Confidence scores
  - Real-time validation

- **User Experience**:
  - Clean, modern interface
  - Instant feedback
  - Detailed results
  - Error handling

## Technical Stack

- **Backend**:
  - FastAPI (Python web framework)
  - scikit-learn (Machine Learning)
  - Pandas (Data Processing)
  - Pydantic (Data Validation)

- **Frontend**:
  - HTML5/CSS3/JavaScript
  - Modern responsive design
  - Real-time validation

- **Infrastructure**:
  - Docker & Docker Compose
  - Nginx (Web Server)
  - REST API architecture

## Project Structure

```
project/
├── app/
│   └── main.py              # FastAPI application
├── data/
│   ├── kc_house_data.csv    # House sales data
│   └── zipcode_demographics.csv  # Demographic data
├── docs/                    # Project documentation
├── frontend/
│   ├── index.html          # Web interface
│   ├── styles.css          # Styling
│   ├── script.js           # Frontend logic
│   └── nginx.conf          # Nginx configuration
├── model/
│   ├── knn_model.pkl       # KNN model
│   ├── rf_model.pkl        # Random Forest model
│   ├── model_features.json # Feature list
│   └── feature_defaults.json # Default values
├── train_models.py         # Model training script
├── Dockerfile              # API Dockerfile
├── docker-compose.yml      # Service orchestration
└── requirements.txt        # Python dependencies
```

## Model Performance

Current model performance:
- **KNN Model**:
  - R² Score: 0.78
  - Good for basic predictions
  - Faster inference

- **Random Forest Model**:
  - R² Score: 0.85
  - Better accuracy
  - More feature importance insights

## Installation & Setup

1. Clone the repository:
\`\`\`bash
git clone [repository-url]
\`\`\`

2. Install Docker and Docker Compose

3. Train the models:
\`\`\`bash
python train_models.py
\`\`\`

4. Build and run the application:
\`\`\`bash
docker-compose up --build
\`\`\`

5. Access the application:
- Web Interface: http://localhost
- API Documentation: http://localhost:8000/docs

## API Endpoints

- `POST /predict`: Get house price prediction
- `POST /predict/minimal`: Simplified prediction with fewer inputs
- `GET /health`: Check service health
- `GET /models`: List available models

## Usage Example

```json
// Example API request
POST /predict
{
    "version": "rf",
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 2100,
    "sqft_lot": 5000,
    "floors": 2,
    "sqft_above": 1800,
    "sqft_basement": 300,
    "zipcode": "98074"
}

// Example response
{
    "predicted_price": 750000,
    "version": "rf",
    "confidence_score": 0.85,
    "prediction_timestamp": "2024-02-17T23:56:09.123456Z",
    "model_info": {
        "type": "Random Forest",
        "version": "rf",
        "features": [...],
        "accuracy": 0.85,
        "last_trained": "2023-10-11",
        "demographic_data_included": true
    },
    "input_features": {...}
}
```

## Future Improvements

1. Model Enhancements:
   - Additional feature engineering
   - Regular model retraining
   - Confidence intervals
   - A/B testing framework

2. Technical Improvements:
   - Redis caching
   - Load balancing
   - Authentication
   - Monitoring dashboard

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- King County Housing Dataset
- Sound Realty for project requirements
- Census Bureau for demographic data

![phData Logo](phData.png "phData Logo")

