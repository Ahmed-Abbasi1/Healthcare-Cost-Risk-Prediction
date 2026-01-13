"""
Model Training Module for Healthcare Cost Prediction
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os
import json
from datetime import datetime


class ModelTrainer:
    """
    Handles model training, hyperparameter tuning, and model selection
    """
    
    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        """
        Initialize the trainer
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            y_test: Test target (optional)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """
        Initialize all models to be trained
        """
        print("\nInitializing models...")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        }
        
        print(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_baseline_models(self, cv=5):
        """
        Train all baseline models with cross-validation
        
        Args:
            cv (int): Number of cross-validation folds
        """
        if not self.models:
            self.initialize_models()
            
        print("\n" + "="*70)
        print("TRAINING BASELINE MODELS")
        print("="*70)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
            )
            cv_rmse = np.sqrt(-cv_scores)
            
            # Calculate R² score
            cv_r2 = cross_val_score(
                model, self.X_train, self.y_train,
                cv=cv, scoring='r2', n_jobs=-1
            )
            
            self.results[name] = {
                'model': model,
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                'cv_r2_mean': cv_r2.mean(),
                'cv_r2_std': cv_r2.std(),
                'trained': True
            }
            
            print(f"  CV RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std():.2f})")
            print(f"  CV R²: {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")
        
        print("\n" + "="*70)
        print("BASELINE TRAINING COMPLETE")
        print("="*70)
        
        return self.results
    
    def tune_hyperparameters(self, model_name='XGBoost', param_grid=None, cv=5):
        """
        Perform hyperparameter tuning for specified model
        
        Args:
            model_name (str): Name of the model to tune
            param_grid (dict): Parameter grid for GridSearchCV
            cv (int): Number of cross-validation folds
        """
        print(f"\n" + "="*70)
        print(f"HYPERPARAMETER TUNING: {model_name}")
        print("="*70)
        
        if param_grid is None:
            # Default parameter grids
            param_grids = {
                'Random Forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Gradient Boosting': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10]
                },
                'XGBoost': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'Ridge Regression': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
                },
                'Lasso Regression': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
                }
            }
            param_grid = param_grids.get(model_name, {})
        
        if not param_grid:
            print(f"No parameter grid defined for {model_name}")
            return None
        
        # Get base model
        base_model = self.models[model_name]
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=cv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        print("\nStarting grid search...")
        grid_search.fit(self.X_train, self.y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = np.sqrt(-grid_search.best_score_)
        
        print(f"\n✓ Tuning complete!")
        print(f"Best parameters: {best_params}")
        print(f"Best CV RMSE: {best_score:.2f}")
        
        # Update results
        self.results[f'{model_name} (Tuned)'] = {
            'model': best_model,
            'best_params': best_params,
            'cv_rmse_mean': best_score,
            'tuned': True
        }
        
        return best_model, best_params
    
    def select_best_model(self):
        """
        Select the best performing model based on CV RMSE
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        # Create comparison dataframe
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'CV RMSE': result['cv_rmse_mean'],
                'CV R²': result.get('cv_r2_mean', 'N/A')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('CV RMSE')
        
        print("\n", comparison_df.to_string(index=False))
        
        # Select best model
        best_name = comparison_df.iloc[0]['Model']
        self.best_model_name = best_name
        self.best_model = self.results[best_name]['model']
        
        print(f"\n✓ Best Model: {best_name}")
        print(f"  CV RMSE: {comparison_df.iloc[0]['CV RMSE']:.2f}")
        
        return self.best_model, self.best_model_name
    
    def save_model(self, model=None, model_name=None, output_dir='models'):
        """
        Save trained model to disk
        
        Args:
            model: Model to save (default: best model)
            model_name (str): Name for saved model (default: best model name)
            output_dir (str): Directory to save model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if model is None:
            model = self.best_model
            model_name = self.best_model_name
        
        if model is None:
            print("No model to save. Train models first.")
            return
        
        # Save model
        model_path = f'{output_dir}/best_model.pkl'
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_count': self.X_train.shape[1],
            'training_samples': len(self.X_train),
            'cv_rmse': self.results[model_name]['cv_rmse_mean'],
        }
        
        metadata_path = f'{output_dir}/model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✓ Model saved to {model_path}")
        print(f"✓ Metadata saved to {metadata_path}")
        
        return model_path
    
    def full_training_pipeline(self, tune_best_model=True):
        """
        Execute complete training pipeline
        
        Args:
            tune_best_model (bool): Whether to tune hyperparameters for best model
        """
        print("\n" + "="*70)
        print("STARTING FULL TRAINING PIPELINE")
        print("="*70)
        
        # Train baseline models
        self.initialize_models()
        self.train_baseline_models()
        
        # Select best baseline model
        self.select_best_model()
        
        # Optionally tune best model
        if tune_best_model:
            print("\n" + "="*70)
            print("TUNING BEST MODEL")
            print("="*70)
            self.tune_hyperparameters(self.best_model_name)
            self.select_best_model()  # Re-select after tuning
        
        # Save best model
        self.save_model()
        
        print("\n" + "="*70)
        print("TRAINING PIPELINE COMPLETE")
        print("="*70)
        
        return self.best_model


def load_model(model_path='models/best_model.pkl'):
    """
    Load a saved model
    
    Args:
        model_path (str): Path to saved model
        
    Returns:
        model: Loaded model
    """
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("✓ Model loaded successfully")
    return model


if __name__ == "__main__":
    # Load processed data
    from preprocess import load_processed_data
    
    X_train, X_test, y_train, y_test, _, _, _ = load_processed_data()
    
    # Train models
    trainer = ModelTrainer(X_train, y_train, X_test, y_test)
    best_model = trainer.full_training_pipeline(tune_best_model=False)
    
    print("\n✓ Training complete!")
