document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const result = document.getElementById('result');
    const loading = document.getElementById('loading');
    const predictedPrice = document.getElementById('predictedPrice');
    const confidenceScore = document.getElementById('confidenceScore');
    const modelVersion = document.getElementById('modelVersion');
    const modelAccuracy = document.getElementById('modelAccuracy');

    // API endpoint URL
    const API_URL = '/predict';

    // Helper function to validate numbers
    const validateNumber = (value, min, fieldName) => {
        const num = parseFloat(value);
        if (isNaN(num)) {
            throw new Error(`${fieldName} must be a valid number`);
        }
        if (num < min) {
            throw new Error(`${fieldName} must be at least ${min}`);
        }
        return num;
    };

    // Helper function to show error message
    const showError = (message) => {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.style.backgroundColor = '#fee2e2';
        errorDiv.style.color = '#dc2626';
        errorDiv.style.padding = '1rem';
        errorDiv.style.borderRadius = '0.5rem';
        errorDiv.style.marginBottom = '1rem';
        errorDiv.textContent = message;
        
        // Remove any existing error message
        const existingError = form.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }
        
        form.insertBefore(errorDiv, form.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    };

    // Update model info display
    const updateModelInfo = (selectedModel) => {
        const modelInfo = {
            knn: { name: 'KNN Model', accuracy: 78 },
            rf: { name: 'Random Forest', accuracy: 85 }
        };
        
        const info = modelInfo[selectedModel];
        modelVersion.textContent = `Model: ${info.name}`;
        modelAccuracy.textContent = `Accuracy: ${info.accuracy}%`;
    };

    // Add model selection change handler
    const modelRadios = form.querySelectorAll('input[name="model_version"]');
    modelRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            updateModelInfo(e.target.value);
        });
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Remove any existing error message
        const existingError = form.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }

        try {
            // Show loading, hide result
            loading.classList.remove('hidden');
            result.classList.add('hidden');

            // Get form data
            const formData = new FormData(form);
            const data = {
                version: formData.get('model_version')  // Get selected model version
            };

            // Validate and convert form data
            try {
                data.bedrooms = validateNumber(formData.get('bedrooms'), 1, 'Bedrooms');
                data.bathrooms = validateNumber(formData.get('bathrooms'), 0.5, 'Bathrooms');
                data.sqft_living = validateNumber(formData.get('sqft_living'), 100, 'Living Area');
                data.sqft_lot = validateNumber(formData.get('sqft_lot'), 100, 'Lot Size');
                data.floors = validateNumber(formData.get('floors'), 1, 'Floors');
                data.sqft_above = validateNumber(formData.get('sqft_above'), 100, 'Above Ground Area');
                data.sqft_basement = validateNumber(formData.get('sqft_basement'), 0, 'Basement Area');
                
                // Validate zipcode
                const zipcode = formData.get('zipcode');
                if (!/^\d{5}$/.test(zipcode)) {
                    throw new Error('Zipcode must be a 5-digit number');
                }
                data.zipcode = zipcode;

                // Validate total square footage
                const totalSqft = data.sqft_above + data.sqft_basement;
                if (Math.abs(totalSqft - data.sqft_living) > 1) {
                    throw new Error(
                        `Total square footage mismatch: above ground (${data.sqft_above}) + ` +
                        `basement (${data.sqft_basement}) should equal living area (${data.sqft_living})`
                    );
                }
            } catch (validationError) {
                showError(validationError.message);
                loading.classList.add('hidden');
                return;
            }

            console.log('Sending data:', data);  // Debug log

            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            const responseData = await response.json();

            if (!response.ok) {
                throw new Error(responseData.detail || 'API request failed');
            }

            // Format the predicted price
            const formattedPrice = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 0,
                maximumFractionDigits: 0
            }).format(responseData.predicted_price);

            // Update the UI
            predictedPrice.textContent = formattedPrice;
            confidenceScore.textContent = `Confidence: ${(responseData.confidence_score * 100).toFixed(1)}%`;
            modelVersion.textContent = `Model: ${responseData.model_info.type}`;
            modelAccuracy.textContent = `Accuracy: ${(responseData.model_info.accuracy * 100).toFixed(1)}%`;

            // Hide loading, show result
            loading.classList.add('hidden');
            result.classList.remove('hidden');

        } catch (error) {
            console.error('Error:', error);
            showError(error.message || 'An error occurred while getting the prediction.');
            loading.classList.add('hidden');
        }
    });

    // Add real-time validation feedback
    const inputs = form.querySelectorAll('input');
    inputs.forEach(input => {
        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'validation-feedback';
        feedbackDiv.style.color = '#dc2626';
        feedbackDiv.style.fontSize = '0.8rem';
        feedbackDiv.style.marginTop = '0.25rem';
        input.parentNode.appendChild(feedbackDiv);

        input.addEventListener('input', () => {
            try {
                if (input.name === 'zipcode') {
                    if (!/^\d{5}$/.test(input.value)) {
                        throw new Error('Must be a 5-digit number');
                    }
                } else if (input.type === 'number') {
                    const min = parseFloat(input.min);
                    validateNumber(input.value, min, input.name);
                }
                feedbackDiv.textContent = '';
                input.setCustomValidity('');
            } catch (error) {
                feedbackDiv.textContent = error.message;
                input.setCustomValidity(error.message);
            }
        });

        // Add blur event for square footage validation
        if (['sqft_living', 'sqft_above', 'sqft_basement'].includes(input.name)) {
            input.addEventListener('blur', () => {
                const living = parseFloat(form.querySelector('[name="sqft_living"]').value) || 0;
                const above = parseFloat(form.querySelector('[name="sqft_above"]').value) || 0;
                const basement = parseFloat(form.querySelector('[name="sqft_basement"]').value) || 0;

                if (living && above && basement) {
                    const total = above + basement;
                    if (Math.abs(total - living) > 1) {
                        feedbackDiv.textContent = 'Total square footage must match living area';
                        input.setCustomValidity('Square footage mismatch');
                    } else {
                        feedbackDiv.textContent = '';
                        input.setCustomValidity('');
                    }
                }
            });
        }
    });

    // Initialize model info display
    updateModelInfo('knn');
}); 