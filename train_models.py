import pandas as pd
import numpy as np
import os
import pickle
import argparse
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train real estate price prediction models')
    parser.add_argument('--rental_data', type=str, default='unegui_data/unegui_rental_data.csv',
                        help='Path to rental data CSV')
    parser.add_argument('--sales_data', type=str, default='unegui_data/unegui_sales_data.csv',
                        help='Path to sales data CSV')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained models')
    return parser.parse_args()

def load_and_prepare_data(rental_path, sales_path):
    """Load and prepare the rental and sales data"""
    print(f"Loading rental data from: {rental_path}")
    print(f"Loading sales data from: {sales_path}")
    
    dfs = []
    
    # Load rental data
    if os.path.exists(rental_path):
        try:
            rental_df = pd.read_csv(rental_path, encoding='utf-8-sig')
            rental_df['Type'] = 'Rent'
            dfs.append(rental_df)
            print(f"Loaded {len(rental_df)} rental records")
        except Exception as e:
            print(f"Error loading rental data: {e}")
    else:
        print(f"Warning: Rental data file not found at {rental_path}")
    
    # Load sales data
    if os.path.exists(sales_path):
        try:
            sales_df = pd.read_csv(sales_path, encoding='utf-8-sig')
            sales_df['Type'] = 'Sale'
            dfs.append(sales_df)
            print(f"Loaded {len(sales_df)} sales records")
        except Exception as e:
            print(f"Error loading sales data: {e}")
    else:
        print(f"Warning: Sales data file not found at {sales_path}")
    
    if not dfs:
        raise ValueError("No valid data files found")
    
    # Combine datasets
    df = pd.concat(dfs, ignore_index=True)
    
    # Clean and prepare data
    print("Preparing data...")
    df['Үнэ'] = pd.to_numeric(df['Үнэ'], errors='coerce')
    
    # Convert date columns if they exist
    date_columns = ['Нийтэлсэн', 'Огноо', 'Date', 'date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Converted {col} to datetime")
            break
    else:
        # If no date column found, add current date
        print("No date column found, using file modification date")
        df['Нийтэлсэн'] = datetime.now()
    
    # Drop missing values in key columns
    df = df.dropna(subset=['Үнэ'])
    print(f"Data preparation complete. Final dataset size: {len(df)} records")
    
    return df

def engineer_features(df, property_type):
    """Engineer features for time series prediction"""
    print(f"Engineering features for {property_type} properties...")
    
    # Group by date and calculate daily stats
    date_col = None
    for col in ['Нийтэлсэн', 'Огноо', 'Date', 'date']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError("No date column found in data")
    
    daily_data = df.groupby(df[date_col].dt.date).agg(
        {'Үнэ': ['mean', 'median', 'std', 'count']}).reset_index()
    daily_data.columns = ['date', 'price_mean', 'price_median', 'price_std', 'count']
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    
    # Extract time-based features
    daily_data['year'] = daily_data['date'].dt.year
    daily_data['month'] = daily_data['date'].dt.month
    daily_data['day'] = daily_data['date'].dt.day
    daily_data['dayofweek'] = daily_data['date'].dt.dayofweek
    daily_data['is_weekend'] = daily_data['dayofweek'].isin([5, 6]).astype(int)
    
    # Create lag features
    for i in range(1, 8):
        daily_data[f'price_lag_{i}'] = daily_data['price_mean'].shift(i)
    
    # Create rolling window features
    for window in [3, 7, 14, 30]:
        if len(daily_data) > window:
            daily_data[f'rolling_mean_{window}'] = daily_data['price_mean'].rolling(window).mean()
            daily_data[f'rolling_median_{window}'] = daily_data['price_mean'].rolling(window).median()
            daily_data[f'rolling_std_{window}'] = daily_data['price_mean'].rolling(window).std()
    
    # Drop rows with NaN values (resulting from shifts/rolling)
    daily_data = daily_data.dropna()
    
    return daily_data

def train_models(data, property_type, output_dir):
    """Train models for a specific property type and save the best ones"""
    print(f"\nTraining models for {property_type} properties...")
    
    # Prepare feature matrix X and target vector y
    X = data.drop(['date', 'price_mean', 'price_median', 'price_std', 'count'], axis=1, errors='ignore')
    
    # Two different target variables
    y_mean = data['price_mean']
    y_median = data['price_median'] if 'price_median' in data.columns else data['price_mean']
    
    # Split data into training and testing sets (time-based split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    
    y_mean_train, y_mean_test = y_mean[:split_idx], y_mean[split_idx:]
    y_median_train, y_median_test = y_median[:split_idx], y_median[split_idx:]
    
    print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
    
    # Create models
    models = {
        'mean_price': {
            'Linear': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        },
        'median_price': {
            'Linear': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
    }
    
    results = {}
    best_models = {}
    
    # Train and evaluate models for mean prices
    print("\nTraining models for predicting MEAN prices:")
    best_rmse = float('inf')
    best_model_name = None
    
    for name, model in models['mean_price'].items():
        model.fit(X_train, y_mean_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_mean_test, y_pred))
        mae = mean_absolute_error(y_mean_test, y_pred)
        r2 = r2_score(y_mean_test, y_pred)
        
        print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        
        results[f'mean_{name}'] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_models['mean'] = model
    
    print(f"Best model for mean price prediction: {best_model_name} (RMSE: {best_rmse:.2f})")
    
    # Train and evaluate models for median prices
    print("\nTraining models for predicting MEDIAN prices:")
    best_rmse = float('inf')
    best_model_name = None
    
    for name, model in models['median_price'].items():
        model.fit(X_train, y_median_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_median_test, y_pred))
        mae = mean_absolute_error(y_median_test, y_pred)
        r2 = r2_score(y_median_test, y_pred)
        
        print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        
        results[f'median_{name}'] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_models['median'] = model
    
    print(f"Best model for median price prediction: {best_model_name} (RMSE: {best_rmse:.2f})")
    
    # Save the best models
    os.makedirs(output_dir, exist_ok=True)
    
    mean_model_path = os.path.join(output_dir, f"{property_type.lower()}_mean_price_model.pkl")
    with open(mean_model_path, 'wb') as f:
        pickle.dump(best_models['mean'], f)
    print(f"Saved mean price model to {mean_model_path}")
    
    median_model_path = os.path.join(output_dir, f"{property_type.lower()}_median_price_model.pkl")
    with open(median_model_path, 'wb') as f:
        pickle.dump(best_models['median'], f)
    print(f"Saved median price model to {median_model_path}")
    
    # Also save as default model if it's the sale property type
    if property_type.lower() == 'sale':
        default_path = os.path.join(output_dir, "default_price_prediction_model.pkl")
        with open(default_path, 'wb') as f:
            pickle.dump(best_models['mean'], f)
        print(f"Saved default price prediction model to {default_path}")
        
        # For backwards compatibility, also save to root directory
        with open("apartment_price_prediction_model.pkl", 'wb') as f:
            pickle.dump(best_models['mean'], f)
        print("Also saved as apartment_price_prediction_model.pkl in root directory")
    
    return results

def main():
    """Main function to run the model training pipeline"""
    print("\n=== Real Estate Price Prediction Model Training ===\n")
    
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Load and prepare data
        df = load_and_prepare_data(args.rental_data, args.sales_data)
        
        all_results = {}
        
        # Process each property type
        for property_type in df['Type'].unique():
            print(f"\n\n=== Processing {property_type} Properties ===")
            
            # Filter data for the current property type
            type_df = df[df['Type'] == property_type].copy()
            print(f"Found {len(type_df)} records for {property_type} properties")
            
            if len(type_df) < 30:
                print(f"Skipping {property_type} - insufficient data (minimum 30 records required)")
                continue
            
            # Engineer features
            processed_data = engineer_features(type_df, property_type)
            
            if len(processed_data) < 10:
                print(f"Skipping {property_type} - insufficient processed data after feature engineering")
                continue
            
            # Train models and save results
            results = train_models(processed_data, property_type, args.output_dir)
            all_results[property_type] = results
        
        # Print summary of results
        print("\n=== Training Complete ===")
        print("\nSummary of Results:")
        
        for prop_type, results in all_results.items():
            print(f"\n{prop_type} Properties:")
            
            for model_name, metrics in results.items():
                print(f"  {model_name}:")
                print(f"    RMSE: {metrics['rmse']:.2f}")
                print(f"    MAE: {metrics['mae']:.2f}")
                print(f"    R²: {metrics['r2']:.4f}")
        
        print("\nAll models have been successfully saved.")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
