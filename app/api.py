from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

app = FastAPI(
    title="Healthcare Cost Prediction API",
    description="AI-powered healthcare cost prediction with explainability",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Load model and preprocessors
@app.on_event("startup")
async def load_models():
    global model, scaler, label_encoders, feature_names
    try:
        base_path = Path(__file__).parent.parent  # Go up to project root
        model = joblib.load(base_path / 'models' / 'best_model.pkl')
        scaler = joblib.load(base_path / 'data' / 'processed' / 'scaler.pkl')
        label_encoders = joblib.load(base_path / 'data' / 'processed' / 'label_encoders.pkl')
        feature_names = joblib.load(base_path / 'data' / 'processed' / 'feature_names.pkl')
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        model = None


# Request/Response models
class PatientInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Patient age (18-100)")
    sex: str = Field(..., pattern="^(male|female)$", description="Patient sex")
    bmi: float = Field(..., ge=10, le=60, description="Body Mass Index (10-60)")
    children: int = Field(..., ge=0, le=5, description="Number of children (0-5)")
    smoker: str = Field(..., pattern="^(yes|no)$", description="Smoking status")
    region: str = Field(..., pattern="^(northeast|northwest|southeast|southwest)$", description="US Region")

    class Config:
        schema_extra = {
            "example": {
                "age": 30,
                "sex": "male",
                "bmi": 25.0,
                "children": 0,
                "smoker": "no",
                "region": "northeast"
            }
        }


class PredictionResponse(BaseModel):
    prediction: float
    annual_cost: str
    monthly_cost: str
    daily_cost: str
    risk_level: str
    risk_category: str
    risk_factors: list
    recommendations: list
    bmi_category: str
    age_group: str


def preprocess_input(data: dict) -> pd.DataFrame:
    """Preprocess user input to match training data format"""
    df = pd.DataFrame([data])
    
    # BMI category
    bmi = df['bmi'].values[0]
    if bmi < 18.5:
        df['bmi_category'] = 'underweight'
    elif 18.5 <= bmi < 25:
        df['bmi_category'] = 'normal'
    elif 25 <= bmi < 30:
        df['bmi_category'] = 'overweight'
    else:
        df['bmi_category'] = 'obese'
    
    # Age group
    age = df['age'].values[0]
    if age < 30:
        df['age_group'] = 'young'
    elif 30 <= age < 50:
        df['age_group'] = 'middle_aged'
    else:
        df['age_group'] = 'senior'
    
    # Interaction features
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['smoker_obese'] = int((df['smoker'].values[0] == 'yes') and (df['bmi_category'].values[0] == 'obese'))
    
    # Polynomial features
    df['age_squared'] = df['age'] ** 2
    df['bmi_squared'] = df['bmi'] ** 2
    
    # Family size
    df['has_children'] = int(df['children'].values[0] > 0)
    
    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except:
                df[col] = 0
    
    # Ensure all features are present
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_names]
    return df


def analyze_risk_factors(data: dict, prediction: float):
    """Analyze risk factors and generate recommendations"""
    risk_factors = []
    recommendations = []
    
    bmi = data['bmi']
    age = data['age']
    smoker = data['smoker']
    children = data['children']
    
    # Risk factors
    if smoker == "yes":
        risk_factors.append({
            "icon": "🚬",
            "title": "Smoking Status",
            "description": "Major cost driver - can increase costs by 2-3x",
            "severity": "high"
        })
        recommendations.append("🚭 Smoking Cessation: Quitting smoking is the single most impactful change - can save $10,000+ annually")
    
    if bmi >= 30:
        risk_factors.append({
            "icon": "⚖️",
            "title": f"BMI: {bmi:.1f} (Obese)",
            "description": "Obesity significantly increases healthcare costs",
            "severity": "high"
        })
        recommendations.append("🏃 Weight Management: Reducing BMI to healthy range (18.5-25) through diet and exercise")
    elif bmi >= 25:
        risk_factors.append({
            "icon": "⚖️",
            "title": f"BMI: {bmi:.1f} (Overweight)",
            "description": "Elevated BMI may lead to higher medical expenses",
            "severity": "medium"
        })
        recommendations.append("🥗 Healthy Lifestyle: Maintain balanced diet and regular physical activity")
    
    if age >= 50:
        risk_factors.append({
            "icon": "👴",
            "title": f"Age: {age} years",
            "description": "Older age associated with higher medical expenses",
            "severity": "medium"
        })
    
    if children >= 3:
        risk_factors.append({
            "icon": "👨‍👩‍👧‍👦",
            "title": f"{children} Dependents",
            "description": "Multiple dependents increase family insurance costs",
            "severity": "low"
        })
    
    # General recommendations
    recommendations.append("🏥 Preventive Care: Regular check-ups and screenings can catch issues early")
    recommendations.append("💊 Generic Medications: Choose generic over brand-name when possible")
    
    # Risk level
    if prediction < 5000:
        risk_level = "Low Risk"
        risk_category = "low"
    elif prediction < 15000:
        risk_level = "Medium Risk"
        risk_category = "medium"
    else:
        risk_level = "High Risk"
        risk_category = "high"
    
    return risk_factors, recommendations, risk_level, risk_category


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    html_file = Path(__file__).parent / "templates" / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding='utf-8'), status_code=200)
    return HTMLResponse(content="<h1>Healthcare Cost Prediction API</h1><p>Go to <a href='/docs'>/docs</a> for API documentation</p>", status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientInput):
    """Make a prediction for the given patient data"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert to dict
        data = patient.dict()
        
        # Preprocess
        processed_data = preprocess_input(data)
        
        # Scale
        scaled_data = scaler.transform(processed_data)
        
        # Predict
        prediction = float(model.predict(scaled_data)[0])
        
        # Calculate costs
        monthly_cost = prediction / 12
        daily_cost = prediction / 365
        
        # Analyze risks
        risk_factors, recommendations, risk_level, risk_category = analyze_risk_factors(data, prediction)
        
        # BMI category
        bmi = data['bmi']
        if bmi < 18.5:
            bmi_category = "Underweight"
        elif bmi < 25:
            bmi_category = "Normal"
        elif bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"
        
        # Age group
        age = data['age']
        if age < 30:
            age_group = "Young Adult"
        elif age < 50:
            age_group = "Middle Aged"
        else:
            age_group = "Senior"
        
        return PredictionResponse(
            prediction=prediction,
            annual_cost=f"${prediction:,.2f}",
            monthly_cost=f"${monthly_cost:,.2f}",
            daily_cost=f"${daily_cost:,.2f}",
            risk_level=risk_level,
            risk_category=risk_category,
            risk_factors=risk_factors,
            recommendations=recommendations,
            bmi_category=bmi_category,
            age_group=age_group
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features": len(feature_names),
        "feature_names": feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
        "performance": {
            "r2_score": "85.07%",
            "rmse": "$4,509",
            "training_samples": 1337
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
