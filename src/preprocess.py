"""
Data Preprocessing Module for Healthcare Cost Prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib


class DataPreprocessor:
    """
    Handles all data preprocessing tasks including:
    - Loading data
    - Cleaning and validation
    - Feature engineering
    - Encoding categorical variables
    - Scaling numerical features
    - Train-test split
    """
    
    def __init__(self, data_path='data/raw/insurance.csv'):
        """
        Initialize the preprocessor
        
        Args:
            data_path (str): Path to the raw CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self):
        """Load data from CSV file"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """Basic data exploration"""
        if self.df is None:
            self.load_data()
            
        print("\n" + "="*50)
        print("DATA OVERVIEW")
        print("="*50)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nBasic Statistics:\n{self.df.describe()}")
        print(f"\nCategorical Features:")
        for col in self.df.select_dtypes(include=['object']).columns:
            print(f"  {col}: {self.df[col].unique()}")
        
        return self.df.info()
    
    def clean_data(self):
        """Clean and validate data"""
        if self.df is None:
            self.load_data()
            
        print("\nCleaning data...")
        initial_shape = self.df.shape
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        
        # Handle missing values (if any)
        if self.df.isnull().sum().sum() > 0:
            print("Handling missing values...")
            self.df = self.df.dropna()
        
        # Validate data types
        assert self.df['age'].dtype in [np.int64, np.float64], "Age should be numeric"
        assert self.df['bmi'].dtype in [np.int64, np.float64], "BMI should be numeric"
        assert self.df['children'].dtype in [np.int64, np.float64], "Children should be numeric"
        assert self.df['charges'].dtype in [np.int64, np.float64], "Charges should be numeric"
        
        print(f"Data cleaned. Shape: {initial_shape} -> {self.df.shape}")
        return self.df
    
    def engineer_features(self):
        """
        Create additional features for better predictions
        """
        if self.df is None:
            self.load_data()
            
        print("\nEngineering features...")
        
        # BMI categories based on WHO standards
        def bmi_category(bmi):
            if bmi < 18.5:
                return 'underweight'
            elif 18.5 <= bmi < 25:
                return 'normal'
            elif 25 <= bmi < 30:
                return 'overweight'
            else:
                return 'obese'
        
        self.df['bmi_category'] = self.df['bmi'].apply(bmi_category)
        
        # Age groups
        def age_group(age):
            if age < 30:
                return 'young'
            elif 30 <= age < 50:
                return 'middle_aged'
            else:
                return 'senior'
        
        self.df['age_group'] = self.df['age'].apply(age_group)
        
        # Interaction features
        self.df['age_bmi_interaction'] = self.df['age'] * self.df['bmi']
        self.df['smoker_obese'] = ((self.df['smoker'] == 'yes') & 
                                    (self.df['bmi_category'] == 'obese')).astype(int)
        
        # Polynomial features for age and BMI
        self.df['age_squared'] = self.df['age'] ** 2
        self.df['bmi_squared'] = self.df['bmi'] ** 2
        
        # Family size indicator
        self.df['has_children'] = (self.df['children'] > 0).astype(int)
        
        print(f"Feature engineering complete. New shape: {self.df.shape}")
        print(f"New features: {[col for col in self.df.columns if col not in ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']]}")
        
        return self.df
    
    def encode_features(self):
        """
        Encode categorical variables
        """
        if self.df is None:
            self.load_data()
            
        print("\nEncoding categorical features...")
        
        # Separate target variable
        y = self.df['charges'].copy()
        X = self.df.drop('charges', axis=1)
        
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Label encoding for categorical features
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
            print(f"  Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        
        Args:
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
        """
        X, y = self.encode_features()
        
        print(f"\nSplitting data (test_size={test_size}, random_state={random_state})...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """
        Scale numerical features using StandardScaler
        """
        if self.X_train is None:
            self.split_data()
            
        print("\nScaling features...")
        
        # Fit scaler on training data and transform both train and test
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        print("Features scaled successfully")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def save_processed_data(self, output_dir='data/processed'):
        """
        Save processed data and preprocessing objects
        
        Args:
            output_dir (str): Directory to save processed files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving processed data to {output_dir}...")
        
        # Save train/test splits
        self.X_train_scaled.to_csv(f'{output_dir}/X_train.csv', index=False)
        self.X_test_scaled.to_csv(f'{output_dir}/X_test.csv', index=False)
        pd.DataFrame(self.y_train).to_csv(f'{output_dir}/y_train.csv', index=False)
        pd.DataFrame(self.y_test).to_csv(f'{output_dir}/y_test.csv', index=False)
        
        # Save preprocessing objects
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{output_dir}/label_encoders.pkl')
        joblib.dump(self.feature_names, f'{output_dir}/feature_names.pkl')
        
        print("✓ Processed data saved successfully")
        
    def full_pipeline(self):
        """
        Execute the complete preprocessing pipeline
        """
        print("\n" + "="*70)
        print("STARTING FULL PREPROCESSING PIPELINE")
        print("="*70)
        
        self.load_data()
        self.explore_data()
        self.clean_data()
        self.engineer_features()
        self.split_data()
        self.scale_features()
        self.save_processed_data()
        
        print("\n" + "="*70)
        print("PREPROCESSING PIPELINE COMPLETED")
        print("="*70)
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test


def load_processed_data(data_dir='data/processed'):
    """
    Load previously processed data
    
    Args:
        data_dir (str): Directory containing processed files
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names
    """
    print(f"Loading processed data from {data_dir}...")
    
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    
    scaler = joblib.load(f'{data_dir}/scaler.pkl')
    label_encoders = joblib.load(f'{data_dir}/label_encoders.pkl')
    feature_names = joblib.load(f'{data_dir}/feature_names.pkl')
    
    print("✓ Processed data loaded successfully")
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names


if __name__ == "__main__":
    # Run the preprocessing pipeline
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.full_pipeline()
    
    print("\nPreprocessing complete!")
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
