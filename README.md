# Sound Realty House Price Predictor

A machine learning-powered application that helps predict house prices in the Seattle area, combining property features with demographic data for accurate estimations.

## Project Overview

This project implements a house price prediction service for Sound Realty, featuring:
- A machine learning model trained on Seattle area housing data
- A REST API for serving predictions
- A user-friendly web interface for realtors
- Integration of demographic data for better predictions

## Key Features

- **Accurate Price Predictions**: Uses KNN Regression or Random Forest with demographic data integration
- **Real-time API**: Fast, scalable predictions via REST API
- **User-friendly Interface**: Clean, intuitive web interface for easy use
- **Data Validation**: Comprehensive input validation and error handling
- **Docker Deployment**: Containerized application for easy deployment and scaling

## Technical Stack

- **Backend**:
  - FastAPI (Python web framework)
  - scikit-learn (Machine Learning)
  - Pandas (Data Processing)
  - Pydantic (Data Validation)

- **Frontend**:

![Frontend](image.png "Frontend")

  - HTML5/CSS3/JavaScript
  - Modern responsive design
  - Real-time validation

- **Infrastructure**:
  - Docker & Docker Compose
  - Nginx (Web Server)
  - REST API architecture

# Model Performance

Based on our evaluation:
- RMSE (Root Mean Square Error): $X
- MAE (Mean Absolute Error): $Y
- R² Score: Z

Key findings from model evaluation:
1. Strong correlation between living space and price
2. Significant impact of location (zipcode) on prices
3. Improved accuracy with demographic data integration

## Installation & Setup

1. Clone the repository:
\`\`\`bash
git clone [repository-url]
\`\`\`

2. Install Docker and Docker Compose

3. Build and run the application:
\`\`\`bash
python train_models.py
docker-compose up --build
\`\`\`

4. Access the application:
- Web Interface: http://localhost
- API Documentation: http://localhost:8000/docs

## API Endpoints

- `POST /predict`: Get house price prediction
- `GET /health`: Check service health
- `GET /model/info`: Get model information

## Usage Example

```json
// Example API request
POST /predict
{
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
    "confidence_score": 0.95,
    "model_version": "1.0.0"
}
```

## Future Improvements

1. Model Enhancements:
   - Integration of more features (school ratings, crime rates)
   - Regular model retraining with new data from the app
   - Confidence interval calculations

2. Technical Improvements:
   - Automated model retraining pipeline
   - Caching layer for frequent predictions
   - A/B testing infrastructure


![phData Logo](phData.png "phData Logo")

