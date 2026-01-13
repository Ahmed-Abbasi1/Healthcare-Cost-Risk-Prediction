// Main JavaScript for Healthcare Cost Prediction

document.addEventListener('DOMContentLoaded', function() {
    // Initialize
    initializeTabs();
    setupEventListeners();
    restoreActiveTab();
});

// Tab Switching Function
function switchTab(tabName) {
    // If switching to predict tab, clear results
    if (tabName === 'predict') {
        localStorage.removeItem('resultsVisible');
        localStorage.removeItem('resultsData');
        localStorage.removeItem('formData');
        
        // Hide results tab button
        const resultsTabBtn = document.getElementById('results-tab-btn');
        if (resultsTabBtn) {
            resultsTabBtn.style.display = 'none';
        }
        
        // Reset results display
        const resultsContent = document.getElementById('results-content');
        const resultsPlaceholder = document.getElementById('results-placeholder');
        if (resultsContent) resultsContent.style.display = 'none';
        if (resultsPlaceholder) resultsPlaceholder.style.display = 'block';
    }
    
    // Remove active class from all tabs and contents
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    // Add active class to selected tab and content
    const selectedBtn = document.querySelector(`[data-tab="${tabName}"]`);
    const selectedContent = document.getElementById(`${tabName}-tab`);
    
    if (selectedBtn && selectedContent) {
        selectedBtn.classList.add('active');
        selectedContent.classList.add('active');
        // Save the active tab to localStorage
        localStorage.setItem('activeTab', tabName);
    }
}

function initializeTabs() {
    // Setup tab click handlers
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            switchTab(tabName);
        });
    });
}

function restoreActiveTab() {
    // Get the saved tab from localStorage
    const savedTab = localStorage.getItem('activeTab');
    
    // Check if results were visible before refresh AND user was on results tab
    const resultsVisible = localStorage.getItem('resultsVisible');
    if (resultsVisible === 'true' && savedTab === 'results') {
        const resultsTabBtn = document.getElementById('results-tab-btn');
        resultsTabBtn.style.display = 'flex';
        
        // Restore results data
        const resultsData = localStorage.getItem('resultsData');
        const formData = localStorage.getItem('formData');
        
        if (resultsData && formData) {
            const data = JSON.parse(resultsData);
            const form = JSON.parse(formData);
            
            // Hide placeholder, show results
            document.getElementById('results-placeholder').style.display = 'none';
            document.getElementById('results-content').style.display = 'block';
            
            // Restore all the results
            document.getElementById('predictedCost').textContent = data.annual_cost;
            document.getElementById('riskCategory').textContent = data.risk_category;
            
            const riskBadge = document.getElementById('riskBadge');
            riskBadge.className = `risk-badge risk-${data.risk_category.toLowerCase()}`;
            
            displayRiskFactors(data.risk_factors || []);
            displayRecommendations(data.recommendations || []);
            displayInputSummary(form);
        }
    }
    
    if (savedTab) {
        // Check if the tab button exists (e.g., results tab might be hidden)
        const tabBtn = document.querySelector(`[data-tab="${savedTab}"]`);
        if (tabBtn && tabBtn.style.display !== 'none') {
            // Don't use switchTab here to avoid clearing results
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            tabBtn.classList.add('active');
            const content = document.getElementById(`${savedTab}-tab`);
            if (content) content.classList.add('active');
        }
    }
}

function setupEventListeners() {
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', handleSubmit);
}

async function handleSubmit(e) {
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
    
    // Validate
    if (!validateForm(formData)) {
        return;
    }
    
    // Show loading state
    showLoading();
    
    try {
        // Make API request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }
        
        const result = await response.json();
        
        // Display results with form data (automatically switches to results tab)
        displayResults(result, formData);
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

function validateForm(data) {
    // Check for empty required fields
    if (!data.age) {
        showError('Please enter your age');
        return false;
    }
    
    if (!data.sex) {
        showError('Please select your sex');
        return false;
    }
    
    if (!data.region) {
        showError('Please select your region');
        return false;
    }
    
    if (!data.bmi) {
        showError('Please enter your BMI');
        return false;
    }
    
    if (!data.smoker) {
        showError('Please select your smoking status');
        return false;
    }
    
    if (data.children === null || data.children === undefined || data.children === '') {
        showError('Please enter the number of children');
        return false;
    }
    
    // Age validation
    if (data.age < 18 || data.age > 100) {
        showError('Age must be between 18 and 100 years');
        return false;
    }
    
    // BMI validation
    if (data.bmi < 10 || data.bmi > 60) {
        showError('BMI must be between 10 and 60');
        return false;
    }
    
    // Children validation
    if (data.children < 0 || data.children > 5) {
        showError('Number of children must be between 0 and 5');
        return false;
    }
    
    return true;
}

function showLoading() {
    const button = document.getElementById('predictButton');
    button.disabled = true;
    button.querySelector('.button-text').style.display = 'none';
    button.querySelector('.button-loader').style.display = 'inline';
}

function hideLoading() {
    const button = document.getElementById('predictButton');
    button.disabled = false;
    button.querySelector('.button-text').style.display = 'inline';
    button.querySelector('.button-loader').style.display = 'none';
}

function displayResults(data, formData) {
    // Show results tab button
    const resultsTabBtn = document.getElementById('results-tab-btn');
    resultsTabBtn.style.display = 'flex';
    
    // Save that results are visible
    localStorage.setItem('resultsVisible', 'true');
    localStorage.setItem('resultsData', JSON.stringify(data));
    localStorage.setItem('formData', JSON.stringify(formData));
    
    // Hide placeholder, show results content
    document.getElementById('results-placeholder').style.display = 'none';
    document.getElementById('results-content').style.display = 'block';
    
    // Update predicted cost
    document.getElementById('predictedCost').textContent = data.annual_cost;
    
    // Update risk category
    const riskCategory = data.risk_category;
    document.getElementById('riskCategory').textContent = riskCategory;
    
    // Update risk badge styling
    const riskBadge = document.getElementById('riskBadge');
    riskBadge.className = `risk-badge risk-${riskCategory.toLowerCase()}`;
    
    // Display risk factors
    displayRiskFactors(data.risk_factors || []);
    
    // Display recommendations
    displayRecommendations(data.recommendations || []);
    
    // Display input summary with the form data we sent
    displayInputSummary(formData);
    
    // Switch to results tab
    switchTab('results');
}

function animateNumber(elementId, finalValue) {
    const element = document.getElementById(elementId);
    element.textContent = finalValue;
    element.style.animation = 'none';
    setTimeout(() => {
        element.style.animation = 'pulse 0.5s ease-in-out';
    }, 10);
}

function displayRiskFactors(factors) {
    const container = document.getElementById('riskFactors');
    
    if (!factors || factors.length === 0) {
        container.innerHTML = '<p style="color: var(--success);">✓ No significant risk factors identified</p>';
        return;
    }
    
    container.innerHTML = '';
    
    factors.forEach(factor => {
        const factorDiv = document.createElement('div');
        factorDiv.className = 'risk-factor-item';
        
        // Handle both object and string formats
        if (typeof factor === 'object') {
            factorDiv.innerHTML = `
                <span class="risk-icon">${factor.icon || '⚠️'}</span>
                <div class="risk-content">
                    <strong>${factor.title || 'Risk Factor'}</strong>
                    <p>${factor.description || ''}</p>
                </div>
            `;
        } else {
            factorDiv.innerHTML = `
                <span class="risk-icon">⚠️</span>
                <span class="risk-text">${factor}</span>
            `;
        }
        
        container.appendChild(factorDiv);
    });
}

function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendations');
    
    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<p>No specific recommendations at this time.</p>';
        return;
    }
    
    container.innerHTML = '';
    
    recommendations.forEach(rec => {
        const recDiv = document.createElement('div');
        recDiv.className = 'recommendation-item';
        recDiv.innerHTML = `
            <span class="rec-icon">💡</span>
            <span class="rec-text">${rec}</span>
        `;
        container.appendChild(recDiv);
    });
}

function displayInputSummary(inputData) {
    const container = document.getElementById('inputSummary');
    
    if (!inputData) return;
    
    // Format the display values
    const sexDisplay = inputData.sex.charAt(0).toUpperCase() + inputData.sex.slice(1);
    const smokerDisplay = inputData.smoker === 'yes' ? 'Yes' : 'No';
    const regionDisplay = inputData.region.charAt(0).toUpperCase() + inputData.region.slice(1);
    
    container.innerHTML = `
        <div class="input-item"><strong>Age:</strong> ${inputData.age} years</div>
        <div class="input-item"><strong>Sex:</strong> ${sexDisplay}</div>
        <div class="input-item"><strong>BMI:</strong> ${inputData.bmi}</div>
        <div class="input-item"><strong>Children:</strong> ${inputData.children}</div>
        <div class="input-item"><strong>Smoker:</strong> ${smokerDisplay}</div>
        <div class="input-item"><strong>Region:</strong> ${regionDisplay}</div>
    `;
}

function showError(message) {
    alert('Error: ' + message + '\n\nPlease try again or contact support.');
}

// Add pulse animation
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    .risk-low {
        color: var(--success);
    }
    
    .risk-medium {
        color: var(--warning);
    }
    
    .risk-high {
        color: var(--danger);
    }
`;
document.head.appendChild(style);
