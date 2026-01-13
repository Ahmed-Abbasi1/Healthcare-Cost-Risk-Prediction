"""
Model Evaluation Module for Healthcare Cost Prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import os


class ModelEvaluator:
    """
    Handles model evaluation, metrics calculation, and visualization
    """
    
    def __init__(self, model, X_train, X_test, y_train, y_test, feature_names=None):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names if feature_names else list(X_train.columns)
        
        # Generate predictions
        self.y_train_pred = model.predict(X_train)
        self.y_test_pred = model.predict(X_test)
        
        self.metrics = {}
        
    def calculate_metrics(self):
        """
        Calculate all evaluation metrics
        """
        print("\n" + "="*70)
        print("MODEL EVALUATION METRICS")
        print("="*70)
        
        # Training metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))
        train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        train_mape = np.mean(np.abs((self.y_train - self.y_train_pred) / self.y_train)) * 100
        
        # Test metrics
        test_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))
        test_mae = mean_absolute_error(self.y_test, self.y_test_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        test_mape = np.mean(np.abs((self.y_test - self.y_test_pred) / self.y_test)) * 100
        
        self.metrics = {
            'train': {
                'RMSE': train_rmse,
                'MAE': train_mae,
                'R²': train_r2,
                'MAPE': train_mape
            },
            'test': {
                'RMSE': test_rmse,
                'MAE': test_mae,
                'R²': test_r2,
                'MAPE': test_mape
            }
        }
        
        # Print metrics
        print("\nTraining Set:")
        print(f"  RMSE: ${train_rmse:,.2f}")
        print(f"  MAE:  ${train_mae:,.2f}")
        print(f"  R²:   {train_r2:.4f}")
        print(f"  MAPE: {train_mape:.2f}%")
        
        print("\nTest Set:")
        print(f"  RMSE: ${test_rmse:,.2f}")
        print(f"  MAE:  ${test_mae:,.2f}")
        print(f"  R²:   {test_r2:.4f}")
        print(f"  MAPE: {test_mape:.2f}%")
        
        # Overfitting check
        rmse_diff = abs(train_rmse - test_rmse)
        r2_diff = abs(train_r2 - test_r2)
        
        print("\nOverfitting Analysis:")
        print(f"  RMSE difference: ${rmse_diff:,.2f}")
        print(f"  R² difference:   {r2_diff:.4f}")
        
        if rmse_diff < 1000 and r2_diff < 0.05:
            print("  ✓ Model generalizes well")
        elif rmse_diff < 2000 and r2_diff < 0.1:
            print("  ⚠ Slight overfitting detected")
        else:
            print("  ⚠ Significant overfitting detected")
        
        return self.metrics
    
    def plot_predictions(self, save_path='reports/figures/predictions.png'):
        """
        Plot actual vs predicted values
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Training set
        axes[0].scatter(self.y_train, self.y_train_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[0].plot([self.y_train.min(), self.y_train.max()], 
                     [self.y_train.min(), self.y_train.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Charges ($)', fontsize=12)
        axes[0].set_ylabel('Predicted Charges ($)', fontsize=12)
        axes[0].set_title(f'Training Set\nR² = {self.metrics["train"]["R²"]:.4f}', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Test set
        axes[1].scatter(self.y_test, self.y_test_pred, alpha=0.6, edgecolors='k', linewidth=0.5, color='orange')
        axes[1].plot([self.y_test.min(), self.y_test.max()], 
                     [self.y_test.min(), self.y_test.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[1].set_xlabel('Actual Charges ($)', fontsize=12)
        axes[1].set_ylabel('Predicted Charges ($)', fontsize=12)
        axes[1].set_title(f'Test Set\nR² = {self.metrics["test"]["R²"]:.4f}', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Predictions plot saved to {save_path}")
        plt.close()
        
    def plot_residuals(self, save_path='reports/figures/residuals.png'):
        """
        Plot residual analysis
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Calculate residuals
        train_residuals = self.y_train - self.y_train_pred
        test_residuals = self.y_test - self.y_test_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training residuals vs predicted
        axes[0, 0].scatter(self.y_train_pred, train_residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 0].set_xlabel('Predicted Charges ($)', fontsize=12)
        axes[0, 0].set_ylabel('Residuals ($)', fontsize=12)
        axes[0, 0].set_title('Training Set: Residuals vs Predicted', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test residuals vs predicted
        axes[0, 1].scatter(self.y_test_pred, test_residuals, alpha=0.6, edgecolors='k', linewidth=0.5, color='orange')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Charges ($)', fontsize=12)
        axes[0, 1].set_ylabel('Residuals ($)', fontsize=12)
        axes[0, 1].set_title('Test Set: Residuals vs Predicted', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training residuals distribution
        axes[1, 0].hist(train_residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals ($)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Training Set: Residual Distribution', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Test residuals distribution
        axes[1, 1].hist(test_residuals, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Residuals ($)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Test Set: Residual Distribution', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Residuals plot saved to {save_path}")
        plt.close()
    
    def plot_feature_importance(self, top_n=15, save_path='reports/figures/feature_importance.png'):
        """
        Plot feature importance (for tree-based models)
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Get feature importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices], color='steelblue', edgecolor='black')
        plt.yticks(range(top_n), [self.feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved to {save_path}")
        plt.close()
    
    def explain_with_shap(self, sample_size=100, save_path='reports/figures/shap_summary.png'):
        """
        Generate SHAP explanations
        
        Args:
            sample_size (int): Number of samples for SHAP analysis
            save_path (str): Path to save SHAP plot
        """
        print("\nGenerating SHAP explanations...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Sample data for SHAP (for performance)
        X_sample = self.X_test.sample(min(sample_size, len(self.X_test)), random_state=42)
        
        # Create SHAP explainer
        explainer = shap.Explainer(self.model, self.X_train)
        shap_values = explainer(X_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP summary plot saved to {save_path}")
        plt.close()
        
        return explainer, shap_values
    
    def generate_report(self, save_path='reports/evaluation_report.txt'):
        """
        Generate a comprehensive evaluation report
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        report = []
        report.append("="*70)
        report.append("HEALTHCARE COST PREDICTION MODEL - EVALUATION REPORT")
        report.append("="*70)
        report.append("")
        
        # Metrics
        report.append("PERFORMANCE METRICS")
        report.append("-"*70)
        report.append("\nTraining Set:")
        for metric, value in self.metrics['train'].items():
            if metric == 'MAPE':
                report.append(f"  {metric}: {value:.2f}%")
            elif metric == 'R²':
                report.append(f"  {metric}: {value:.4f}")
            else:
                report.append(f"  {metric}: ${value:,.2f}")
        
        report.append("\nTest Set:")
        for metric, value in self.metrics['test'].items():
            if metric == 'MAPE':
                report.append(f"  {metric}: {value:.2f}%")
            elif metric == 'R²':
                report.append(f"  {metric}: {value:.4f}")
            else:
                report.append(f"  {metric}: ${value:,.2f}")
        
        report.append("")
        report.append("="*70)
        
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\n✓ Evaluation report saved to {save_path}")
        
        return '\n'.join(report)
    
    def full_evaluation_pipeline(self):
        """
        Run complete evaluation pipeline
        """
        print("\n" + "="*70)
        print("STARTING FULL EVALUATION PIPELINE")
        print("="*70)
        
        self.calculate_metrics()
        self.plot_predictions()
        self.plot_residuals()
        
        try:
            self.plot_feature_importance()
        except:
            print("⚠ Could not generate feature importance plot")
        
        try:
            self.explain_with_shap()
        except Exception as e:
            print(f"⚠ Could not generate SHAP explanations: {e}")
        
        self.generate_report()
        
        print("\n" + "="*70)
        print("EVALUATION PIPELINE COMPLETE")
        print("="*70)
        
        return self.metrics


if __name__ == "__main__":
    # Load data and model
    from preprocess import load_processed_data
    from train import load_model
    
    X_train, X_test, y_train, y_test, _, _, feature_names = load_processed_data()
    model = load_model()
    
    # Evaluate model
    evaluator = ModelEvaluator(model, X_train, X_test, y_train, y_test, feature_names)
    metrics = evaluator.full_evaluation_pipeline()
    
    print("\n✓ Evaluation complete!")
