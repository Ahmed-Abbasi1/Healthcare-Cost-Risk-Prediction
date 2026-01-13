# 🚀 Vercel Deployment Guide

## Healthcare Cost Prediction System - Deployment Instructions

---

## ✅ Prerequisites Checklist

Before deploying, make sure you have:

1. **Vercel Account** - Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI** (Optional) - Install with: `npm install -g vercel`
3. **GitHub Account** (Recommended) - For automatic deployments
4. **All model files present** in the repository

---

## 📁 Project Structure for Vercel

```
Healthcare Risk and Cost Prediction System/
├── app/
│   ├── api.py                 # Main FastAPI application
│   ├── static/                # Static files (CSS, JS)
│   └── templates/             # HTML templates
├── models/
│   ├── best_model.pkl         # Trained ML model
│   └── model_metadata.json    # Model information
├── data/processed/
│   ├── scaler.pkl            # Feature scaler
│   ├── label_encoders.pkl    # Categorical encoders
│   └── feature_names.pkl     # Feature list
├── index.py                   # Vercel entry point ⭐
├── vercel.json               # Vercel configuration ⭐
├── requirements-vercel.txt   # Production dependencies ⭐
└── .vercelignore            # Files to exclude ⭐
```

**⭐ = Files created for Vercel deployment**

---

## 🎯 Deployment Options

### **Option 1: Deploy via Vercel Dashboard (Recommended for Beginners)**

#### Step 1: Prepare GitHub Repository

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Prepare for Vercel deployment"

# Create GitHub repository and push
# Go to github.com → Create new repository → Follow instructions
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

#### Step 2: Deploy on Vercel

1. **Go to**: [vercel.com/new](https://vercel.com/new)
2. **Import Git Repository**: Click "Import" next to your GitHub repository
3. **Configure Project**:
   - **Framework Preset**: `Other`
   - **Root Directory**: `./` (leave as default)
   - **Build Command**: Leave empty
   - **Output Directory**: Leave empty
   - **Install Command**: `pip install -r requirements-vercel.txt`

4. **Environment Variables** (if needed):
   - Click "Environment Variables"
   - Add any secrets or API keys (none required for this project)

5. **Deploy**: Click "Deploy" button

6. **Wait**: Deployment takes 2-5 minutes

7. **Success**: You'll get a URL like `https://your-project.vercel.app`

---

### **Option 2: Deploy via Vercel CLI**

```bash
# Install Vercel CLI globally
npm install -g vercel

# Navigate to project directory
cd "D:\Projects\Data Scientist Projects\Healthcare Risk and Cost Prediction System"

# Login to Vercel
vercel login

# Deploy (first time)
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? Your account
# - Link to existing project? No
# - Project name? healthcare-cost-prediction (or your choice)
# - Directory? ./ (default)
# - Want to modify settings? No

# Deploy to production
vercel --prod
```

---

## 🔧 Vercel Configuration Explained

### **vercel.json**

```json
{
  "version": 2,
  "builds": [
    {
      "src": "app/api.py",      // Points to your FastAPI app
      "use": "@vercel/python"    // Uses Python runtime
    }
  ],
  "routes": [
    {
      "src": "/(.*)",            // All routes
      "dest": "app/api.py"       // Go to FastAPI app
    }
  ],
  "env": {
    "PYTHONPATH": "."            // Ensures imports work
  }
}
```

### **Entry Point Settings in Vercel Dashboard**

When Vercel asks for configuration:

| Setting | Value |
|---------|-------|
| **Entry Point** | `app/api.py` or `index.py` |
| **Python Runtime** | `python-3.9` or `python-3.10` |
| **Install Command** | `pip install -r requirements-vercel.txt` |
| **Build Command** | Leave empty |

---

## ⚠️ Important Notes

### **1. File Size Limitations**

Vercel has a **250MB deployment limit**. Your project should be under this. Check:

```bash
# Check model file sizes
dir models /s
dir data\processed /s
```

If your model files are too large, consider:
- Using model compression
- Hosting models separately (AWS S3, Google Cloud Storage)
- Using model quantization

### **2. Cold Starts**

Serverless functions have cold starts (2-5 seconds first request). This is normal.

### **3. Required Files**

Ensure these files exist and are committed:
- ✅ `models/best_model.pkl`
- ✅ `data/processed/scaler.pkl`
- ✅ `data/processed/label_encoders.pkl`
- ✅ `data/processed/feature_names.pkl`

### **4. Static Files**

Your static files (CSS, JS) in `app/static/` will be served correctly.

---

## 🧪 Testing Your Deployment

Once deployed, test these endpoints:

### **1. Health Check**
```bash
curl https://your-project.vercel.app/health
```

Expected: `{"status":"healthy","model_loaded":true}`

### **2. API Documentation**
Visit: `https://your-project.vercel.app/docs`

### **3. Main Page**
Visit: `https://your-project.vercel.app/`

### **4. Make a Prediction**
```bash
curl -X POST https://your-project.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "sex": "male",
    "bmi": 25.0,
    "children": 0,
    "smoker": "no",
    "region": "northeast"
  }'
```

---

## 🐛 Troubleshooting

### **Issue: "Build Failed"**

**Solution 1**: Check dependency versions
```bash
# Use requirements-vercel.txt (already created)
# It has minimal, production-ready dependencies
```

**Solution 2**: Check Python version
- Vercel supports Python 3.9, 3.10, 3.11
- Specify in `vercel.json`:
```json
{
  "builds": [
    {
      "src": "app/api.py",
      "use": "@vercel/python",
      "config": {
        "runtime": "python3.10"
      }
    }
  ]
}
```

### **Issue: "Model not loaded"**

**Solution**: Ensure model files are NOT in `.vercelignore`
- Check `.vercelignore` doesn't exclude `models/` or `data/processed/`

### **Issue: "Module not found"**

**Solution**: Check imports in `app/api.py`
- Relative imports should work with `PYTHONPATH` setting
- If issues persist, modify imports to be absolute

### **Issue: "Deployment too large"**

**Solution**: Reduce package sizes
```bash
# Remove heavy dependencies from requirements-vercel.txt:
# - Remove: matplotlib, seaborn, shap, xgboost (if not used in API)
# - Keep: pandas, numpy, scikit-learn, fastapi, joblib
```

### **Issue: "Function timeout"**

**Solution**: Vercel free tier has 10-second timeout
- Consider Vercel Pro ($20/month) for 60-second timeout
- Or optimize model loading (lazy loading)

---

## 🔄 Redeployment

### **Automatic Redeployment (GitHub)**
- Push changes to GitHub → Vercel auto-deploys
```bash
git add .
git commit -m "Update model"
git push
```

### **Manual Redeployment (CLI)**
```bash
vercel --prod
```

---

## 🎉 Success Checklist

- ✅ Vercel project created
- ✅ Deployment successful (no errors)
- ✅ Health check returns `{"status":"healthy","model_loaded":true}`
- ✅ API docs accessible at `/docs`
- ✅ Main page loads correctly at `/`
- ✅ Prediction endpoint works (test with sample data)
- ✅ All static files (CSS/JS) load correctly

---

## 📚 Additional Resources

- **Vercel Python Docs**: https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/
- **Vercel CLI Reference**: https://vercel.com/docs/cli

---

## 🆘 Need Help?

If deployment fails:

1. **Check Vercel Logs**: 
   - Dashboard → Your Project → Deployments → Click deployment → Logs

2. **Common fixes**:
   - Ensure `requirements-vercel.txt` has correct dependencies
   - Verify model files are included (not in .vercelignore)
   - Check Python version compatibility

3. **Alternative**: Deploy to Render, Railway, or PythonAnywhere

---

**Good luck with your deployment! 🚀**
