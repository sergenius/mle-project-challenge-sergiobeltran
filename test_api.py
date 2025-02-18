import pandas as pd
import requests
import json
from typing import Dict
import time

def load_test_data(file_path: str = "data/future_unseen_examples.csv") -> pd.DataFrame:
    """Load the test data from the future unseen examples."""
    return pd.read_csv(file_path, dtype={'zipcode': str})

def test_health_endpoint(base_url: str = "http://localhost:8000") -> None:
    """Test the health check endpoint."""
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
    print("Health check passed!")

def test_prediction(row: Dict, base_url: str = "http://localhost:8000") -> Dict:
    """Test the prediction endpoint with a single row of data."""
    response = requests.post(
        f"{base_url}/predict",
        json=row
    )
    return response.json()

def main():
    # Load test data
    print("Loading test data...")
    test_data = load_test_data()
    
    # Test health endpoint
    print("\nTesting health endpoint...")
    test_health_endpoint()
    
    # Test predictions
    print("\nTesting predictions...")
    test_cases = test_data.head(5)  # Test with first 5 examples
    
    results = []
    for _, row in test_cases.iterrows():
        # Convert row to dict and keep only required columns
        row_dict = {
            'bedrooms': float(row['bedrooms']),
            'bathrooms': float(row['bathrooms']),
            'sqft_living': float(row['sqft_living']),
            'sqft_lot': float(row['sqft_lot']),
            'floors': float(row['floors']),
            'sqft_above': float(row['sqft_above']),
            'sqft_basement': float(row['sqft_basement']),
            'zipcode': str(row['zipcode'])
        }
        
        # Time the prediction
        start_time = time.time()
        result = test_prediction(row_dict)
        end_time = time.time()
        
        results.append({
            'input': row_dict,
            'prediction': result,
            'response_time': end_time - start_time
        })
    
    # Print results
    print("\nTest Results:")
    print("-" * 50)
    for i, result in enumerate(results, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {json.dumps(result['input'], indent=2)}")
        print(f"Predicted Price: ${result['prediction']['predicted_price']:,.2f}")
        print(f"Response Time: {result['response_time']*1000:.2f}ms")
        print(f"Confidence Score: {result['prediction']['confidence_score']}")
        print("-" * 50)
    
    # Print summary
    avg_response_time = sum(r['response_time'] for r in results) / len(results)
    print(f"\nSummary:")
    print(f"Total test cases: {len(results)}")
    print(f"Average response time: {avg_response_time*1000:.2f}ms")
    print(f"All tests completed successfully!")

if __name__ == "__main__":
    main() 