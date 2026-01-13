// Healthcare Cost Prediction - Frontend JavaScript

// Update age slider display
const ageSlider = document.getElementById('age');
const ageValue = document.getElementById('ageValue');

ageSlider.addEventListener('input', (e) => {
    ageValue.textContent = e.target.value;
});

// Form submission
const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Get form data
    const formData = {
        age: parseInt(document.getElementById('age').value),
        sex: document.getElementById('sex').value,
        bmi: parseFloat(document.getElementById('bmi').value),
        children: parseInt(document.getElementById('children').value),
        smoker: document.getElementById('smoker').value,
        region: document.getElementById('region').value
    };
    
    // Show loading state
    showState('loading');
    predictBtn.disabled = true;
    
    try {
        // Make API call
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        
        // Display results
        displayResults(data, formData);
        showState('results');
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred while generating the prediction.');
    } finally {
        predictBtn.disabled = false;
    }
});

// Show different states
function showState(state) {
    const states = ['initialState', 'loadingState', 'resultsState', 'errorState'];
    states.forEach(s => {
        document.getElementById(s).style.display = s === state + 'State' ? 'block' : 'none';
    });
}

// Display results
function displayResults(data, formData) {
    // Patient Summary
    document.getElementById('summaryAge').textContent = formData.age;
    document.getElementById('summaryBMI').textContent = formData.bmi.toFixed(1);
    document.getElementById('summaryBMICategory').textContent = data.bmi_category;
    
    const smokerStatus = formData.smoker === 'yes' ? 'Yes ⚠️' : 'No ✅';
    const smokerRisk = formData.smoker === 'yes' ? 'High Risk' : 'Low Risk';
    document.getElementById('summarySmoker').textContent = smokerStatus;
    document.getElementById('summaryRiskText').textContent = smokerRisk;
    
    // Prediction Amount
    document.getElementById('predictionAmount').textContent = 
        '$' + data.prediction.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    
    // Cost Breakdown
    document.getElementById('monthlyCost').textContent = 
        '$' + Math.round(data.monthly_cost).toLocaleString('en-US');
    document.getElementById('dailyCost').textContent = 
        '$' + Math.round(data.daily_cost).toLocaleString('en-US');
    
    // Risk Badge
    const riskIcons = {
        'low': '✅',
        'medium': '⚠️',
        'high': '🚨'
    };
    document.getElementById('riskIcon').textContent = riskIcons[data.risk_level];
    
    const riskBadge = document.getElementById('riskBadge');
    riskBadge.textContent = data.risk_text;
    riskBadge.className = 'risk-badge ' + data.risk_level;
    
    // Risk Factors
    const riskFactorsList = document.getElementById('riskFactorsList');
    riskFactorsList.innerHTML = '';
    
    if (data.risk_factors && data.risk_factors.length > 0) {
        document.getElementById('riskFactorsSection').style.display = 'block';
        data.risk_factors.forEach(factor => {
            const factorDiv = document.createElement('div');
            factorDiv.className = 'risk-factor ' + factor.severity;
            factorDiv.innerHTML = `
                <div class="risk-icon">${factor.icon}</div>
                <div class="risk-title">${factor.title}</div>
                <div class="risk-description">${factor.description}</div>
            `;
            riskFactorsList.appendChild(factorDiv);
        });
    } else {
        document.getElementById('riskFactorsSection').style.display = 'none';
    }
    
    // Recommendations
    const recommendationsList = document.getElementById('recommendationsList');
    recommendationsList.innerHTML = '';
    
    if (data.recommendations && data.recommendations.length > 0) {
        document.getElementById('recommendationsSection').style.display = 'block';
        data.recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            recommendationsList.appendChild(li);
        });
    } else {
        document.getElementById('recommendationsSection').style.display = 'none';
    }
    
    // Scroll to results smoothly
    document.querySelector('.results-section').scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
    });
}

// Show error
function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    showState('error');
}

// Input validation
document.getElementById('bmi').addEventListener('input', function(e) {
    const value = parseFloat(e.target.value);
    if (value < 10) e.target.value = 10;
    if (value > 60) e.target.value = 60;
});

// Initialize tooltips or additional features here
console.log('🏥 Healthcare Cost Prediction System Loaded');
