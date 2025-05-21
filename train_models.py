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
    
    # For feature engineering, we'll work with the raw data without grouping by date
    # This avoids potential data loss when there are few distinct dates
    
    # Create a working copy
    processed_df = df.copy()
    
    # Ensure we have a date column
    date_col = None
    for col in ['Нийтэлсэн', 'Огноо', 'Date', 'date']:
        if col in processed_df.columns:
            date_col = col
            break
    
    if date_col is None:
        print("No date column found, using current date")
        processed_df['date'] = datetime.now()
        date_col = 'date'
    
    # Create a proper date column for consistency
    processed_df['date'] = pd.to_datetime(processed_df[date_col])
    
    # Extract time-based features
    processed_df['year'] = processed_df['date'].dt.year
    processed_df['month'] = processed_df['date'].dt.month
    processed_df['day'] = processed_df['date'].dt.day
    processed_df['dayofweek'] = processed_df['date'].dt.dayofweek
    processed_df['is_weekend'] = processed_df['dayofweek'].isin([5, 6]).astype(int)
    
    # If we have enough data, we can use groupby approach, otherwise use individual records
    if len(processed_df['date'].dt.date.unique()) >= 10:
        print(f"Using time series approach with {len(processed_df['date'].dt.date.unique())} unique dates")
        # Group by date and calculate daily stats
        daily_data = processed_df.groupby(processed_df['date'].dt.date).agg(
            {'Үнэ': ['mean', 'median', 'std', 'count']}).reset_index()
        daily_data.columns = ['date', 'price_mean', 'price_median', 'price_std', 'count']
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        
        # Extract time-based features again for the grouped data
        daily_data['year'] = daily_data['date'].dt.year
        daily_data['month'] = daily_data['date'].dt.month
        daily_data['day'] = daily_data['date'].dt.day
        daily_data['dayofweek'] = daily_data['date'].dt.dayofweek
        daily_data['is_weekend'] = daily_data['dayofweek'].isin([5, 6]).astype(int)
        
        # Create lag features if we have enough data
        if len(daily_data) >= 8:
            for i in range(1, min(8, len(daily_data))):
                daily_data[f'price_lag_{i}'] = daily_data['price_mean'].shift(i)
        
        # Create rolling window features
        for window in [3, 7, 14]:
            if len(daily_data) > window:
                daily_data[f'rolling_mean_{window}'] = daily_data['price_mean'].rolling(window).mean()
                daily_data[f'rolling_median_{window}'] = daily_data['price_mean'].rolling(window).median()
                daily_data[f'rolling_std_{window}'] = daily_data['price_mean'].rolling(window).std()
        
        # Drop NaN values but keep at least 10 rows
        before_len = len(daily_data)
        daily_data = daily_data.dropna()
        print(f"After removing NaNs: {len(daily_data)} rows (from {before_len})")
        
        if len(daily_data) < 10:
            print("Time series approach resulted in too few records, using individual records instead")
            return prepare_individual_records(processed_df)
            
        return daily_data
    else:
        print(f"Insufficient unique dates ({len(processed_df['date'].dt.date.unique())}), using individual records approach")
        return prepare_individual_records(processed_df)

def prepare_individual_records(df):
    """Prepare individual property records for modeling when time series isn't viable"""
    print("Preparing individual property records...")
    
    # Create a copy to work with
    processed_df = df.copy()
    
    # These columns will be our features
    # First, extract any numeric columns 
    numeric_cols = []
    for col in processed_df.columns:
        if col not in ['Үнэ', 'date', 'Type', 'Link', 'URL', 'url', 'link']:
            try:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                if not processed_df[col].isna().all():
                    numeric_cols.append(col)
            except:
                continue
    
    print(f"Found {len(numeric_cols)} numeric columns to use as features")
    
    # Add engineered features based on date columns
    if 'date' in processed_df.columns:
        processed_df['year'] = processed_df['date'].dt.year
        processed_df['month'] = processed_df['date'].dt.month
        processed_df['day'] = processed_df['date'].dt.day
        processed_df['dayofweek'] = processed_df['date'].dt.dayofweek
        processed_df['is_weekend'] = processed_df['dayofweek'].isin([5, 6]).astype(int)
    
    # Try to find area column
    area_cols = ['Талбай', 'Area', 'area', 'Square', 'square', 'тал']
    area_col = None
    for col in area_cols:
        if col in processed_df.columns:
            area_col = col
            break
    
    # Add additional features based on text columns
    if 'Байршил' in processed_df.columns:
        # Create district indicator variables
        districts = processed_df['Байршил'].str.extract(r'(БГД|БЗД|СХД|ХУД|СБД|ЧД|БХД|НД|ХЭД)', expand=False)
        for district in districts.dropna().unique():
            processed_df[f'district_{district}'] = (districts == district).astype(int)
    
    # Create mean price and median price (target variables)
    processed_df['price_mean'] = processed_df['Үнэ']
    processed_df['price_median'] = processed_df['Үнэ']
    
    # Drop unnecessary columns
    drop_cols = ['Үнэ', 'Type', 'Link', 'URL', 'url', 'link', 'date']
    processed_df = processed_df.drop([col for col in drop_cols if col in processed_df.columns], axis=1)
    
    # Fill NaN values with the mean of each column
    for col in processed_df.columns:
        if col not in ['price_mean', 'price_median'] and processed_df[col].dtype != 'object':
            processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
    
    # Drop any remaining NaN values
    processed_df = processed_df.dropna(subset=['price_mean', 'price_median'])
    
    print(f"Final dataset size: {len(processed_df)} records with {len(processed_df.columns)} features")
    return processed_df

def train_models(data, property_type, output_dir):
    """Train models for a specific property type and save the best ones"""
    print(f"\nTraining models for {property_type} properties...")
    print(f"Dataset shape: {data.shape}")
    
    # Prepare feature matrix X and target vector y
    target_cols = ['price_mean', 'price_median']
    feature_cols = [col for col in data.columns if col not in target_cols and col != 'date']
    
    if len(feature_cols) == 0:
        print("Error: No feature columns available for training")
        return {}
    
    print(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")
    X = data[feature_cols]
    
    # Handle any remaining NaN values in features
    for col in X.columns:
        if X[col].isna().any():
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
    
    # Two different target variables
    if 'price_mean' not in data.columns:
        print("Error: price_mean column not found in data")
        return {}
    
    y_mean = data['price_mean']
    y_median = data['price_median'] if 'price_median' in data.columns else data['price_mean']
    
    # Split data into training and testing sets
    # For time series data, use temporal split
    # For individual records, use random split
    is_time_series = 'date' in data.columns and len(data) > 10
    
    if is_time_series:
        # Time-based split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_mean_train, y_mean_test = y_mean[:split_idx], y_mean[split_idx:]
        y_median_train, y_median_test = y_median[:split_idx], y_median[split_idx:]
    else:
        # Random split for non-time series data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_mean_train, y_mean_test = train_test_split(
            X, y_mean, test_size=0.2, random_state=42)
        _, _, y_median_train, y_median_test = train_test_split(
            X, y_median, test_size=0.2, random_state=42)
    
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
        try:
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
                
        except Exception as e:
            print(f"Error training {name} model: {e}")
    
    if 'mean' in best_models:
        print(f"Best model for mean price prediction: {best_model_name} (RMSE: {best_rmse:.2f})")
    else:
        print("Failed to train any models for mean price prediction")
        return results
    
    # Train and evaluate models for median prices
    print("\nTraining models for predicting MEDIAN prices:")
    best_rmse = float('inf')
    best_model_name = None
    
    for name, model in models['median_price'].items():
        try:
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
                
        except Exception as e:
            print(f"Error training {name} model: {e}")
    
    if 'median' in best_models:
        print(f"Best model for median price prediction: {best_model_name} (RMSE: {best_rmse:.2f})")
    else:
        print("Failed to train any models for median price prediction")
        return results
    
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
