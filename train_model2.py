import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json
from datetime import datetime

def preprocess_data(df):
    df_clean = df.dropna(subset=['rating'])
    
    # Convert timestamp to features
    df_clean['hour_of_day'] = df_clean['timestamp'].dt.hour if isinstance(df_clean['timestamp'].iloc[0], datetime) else pd.to_datetime(df_clean['timestamp']).dt.hour
    df_clean['day_of_week'] = df_clean['timestamp'].dt.dayofweek if isinstance(df_clean['timestamp'].iloc[0], datetime) else pd.to_datetime(df_clean['timestamp']).dt.dayofweek
    
    # Calculate watch ratio
    df_clean['watch_ratio'] = df_clean['watch_time_minutes'] / df_clean['movie_duration']
    
    # Features and target
    X = df_clean.drop(['rating', 'timestamp', 'user_id', 'movie_id'], axis=1)
    y = df_clean['rating']
    
    # Define categorical and numerical features
    categorical_features = ['device', 'user_gender', 'subscription_type', 'genre']
    numerical_features = ['watch_time_minutes', 'completed', 'user_age', 
                          'release_year', 'movie_duration', 'popularity_score',
                          'hour_of_day', 'day_of_week', 'watch_ratio']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return X, y, preprocessor, categorical_features, numerical_features

def train_and_save_model(data_path, model_dir, model_type='gradient_boosting'):
    """Train a model and save it to the specified directory"""
    os.makedirs(model_dir, exist_ok=True)
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    X, y, preprocessor, categorical_features, numerical_features = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_name = "GradientBoostingRecommender"
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train model
    print(f"Training {model_name}...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model evaluation:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Save model
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names (for inference)
    feature_info = {
        'categorical_features': categorical_features,
        'numerical_features': numerical_features
    }
    feature_path = os.path.join(model_dir, 'feature_info.json')
    with open(feature_path, 'w') as f:
        json.dump(feature_info, f)
    
    # Save model metadata
    metadata = {
        'model_name': model_name,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_source': data_path,
        'num_training_samples': len(X_train),
        'model_type': model_type,
        'evaluation': {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2)
        },
        'feature_importance': None 
    }
    
    if hasattr(model, 'feature_importances_'):
        try:
            feature_names = (
                numerical_features +
                preprocessor.transformers_[1][1].get_feature_names_out(categorical_features).tolist()
            )
            # Match with importances
            importances = model.feature_importances_
            importance_dict = dict(zip(feature_names, importances))
            metadata['feature_importance'] = importance_dict
        except:
            pass
    
    # Save metadata
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model metadata saved to {metadata_path}")
    return metadata

if __name__ == "__main__":
    # Train the second model (Gradient Boosting on snapshot 2)
    print("\n=== Training Model 2: Gradient Boosting on Snapshot 2 ===")
    model2_metadata = train_and_save_model(
        './datasets/streaming_data_snapshot2.csv',
        './models/model_snapshot2',
        model_type='gradient_boosting'
    )
    
    print("\nModel has been trained and saved successfully!")