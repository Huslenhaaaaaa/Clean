import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def train_prediction_models():
    """
    Train machine learning models to predict real estate prices based on time series data.
    Save the best performing model as a pickle file.
    """
    print("Starting model training process...")
    
    # Create directory for models if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 1. Load data
    rental_files = glob.glob("unegui_data/unegui_rental_data.csv")
    sales_files = glob.glob("unegui_data/unegui_sales_data.csv")
    
    def load_and_process(files, label):
        all_data = []
        for f in files:
            try:
                df = pd.read_csv(f, encoding='utf-8-sig')
                date_str = os.path.basename(f).split('_')[-1].split('.')[0]
                df['Нийтэлсэн'] = pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')
                df['Нийтэлсэн'] = pd.to_datetime(df['Нийтэлсэн'], errors='coerce')

                df['Type'] = label
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        if not all_data:
            return pd.DataFrame()
            
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Check for and remove duplicates based on URL/link
        if 'Link' in combined_df.columns:
            url_col = 'Link'
        elif 'URL' in combined_df.columns:
            url_col = 'URL'
        elif 'url' in combined_df.columns:
            url_col = 'url'
        elif 'link' in combined_df.columns:
            url_col = 'link'
        elif 'Зар' in combined_df.columns:
            url_col = 'Зар'
        else:
            if 'ad_id' in combined_df.columns:
                url_col = 'ad_id'
            else:
                print("No URL or ad_id column found. Cannot check for duplicates.")
                return combined_df
        
        # Remove duplicates
        duplicate_count = combined_df.duplicated(subset=[url_col]).sum()
        combined_df = combined_df.drop_duplicates(subset=[url_col], keep='first')
        
        if duplicate_count > 0:
            print(f"Removed {duplicate_count} duplicate listings")
            
        return combined_df

    rental_df = load_and_process(rental_files, 'Rent')
    sales_df = load_and_process(sales_files, 'Sale')

    if not rental_df.empty and not sales_df.empty:
        df = pd.concat([rental_df, sales_df], ignore_index=True)
    elif not rental_df.empty:
        df = rental_df
    elif not sales_df.empty:
        df = sales_df
    else:
        print("No data files found.")
        return
    
    # Basic data preprocessing
    df['Үнэ'] = pd.to_numeric(df['Үнэ'], errors='coerce')
    
    # 2. Prepare time series data for each property type
    model_results = {}
    
    for property_type in df['Type'].unique():
        print(f"\nTraining models for {property_type} properties...")
        
        # Filter data by property type
        type_df = df[df['Type'] == property_type].copy()
        
        # Group by date and calculate daily average prices
        daily_data = type_df.groupby(type_df['Нийтэлсэн'].dt.date).agg({
            'Үнэ': 'mean',
            'ad_id': 'count'
        }).reset_index()
        
        # Sort by date
        daily_data = daily_data.sort_values('Нийтэлсэн')
        
        # Make sure there's enough data
        if len(daily_data) < 30:
            print(f"Not enough data for {property_type} (only {len(daily_data)} days). Skipping.")
            continue
        
        print(f"Data timespan: {daily_data['Нийтэлсэн'].min()} to {daily_data['Нийтэлсэн'].max()}")
        
        # 3. Feature engineering for time series prediction
        # Create features from date
        daily_data['year'] = daily_data['Нийтэлсэн'].dt.year
        daily_data['month'] = daily_data['Нийтэлсэн'].dt.month
        daily_data['day'] = daily_data['Нийтэлсэн'].dt.day
        daily_data['dayofweek'] = daily_data['Нийтэлсэн'].dt.dayofweek
        
        # Create lag features (previous days' prices)
        for i in range(1, 8):  # 1-week lag features
            daily_data[f'lag_{i}'] = daily_data['Үнэ'].shift(i)
        
        # Create rolling averages
        daily_data['rolling_mean_7'] = daily_data['Үнэ'].rolling(window=7).mean()
        daily_data['rolling_mean_14'] = daily_data['Үнэ'].rolling(window=14).mean()
        
        # Drop rows with NaN values (initial rows with lag features)
        daily_data = daily_data.dropna()
        
        # Create features and target variables
        X = daily_data.drop(['Нийтэлсэн', 'Үнэ', 'ad_id'], axis=1)
        y = daily_data['Үнэ']
        
        # 4. Train-test split (chronological for time series)
        # Use last 20% of data as test set
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # 5. Train and evaluate different models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        best_rmse = float('inf')
        best_model_name = None
        best_model = None
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
            
            # Track best model
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
                best_model = model
        
        print(f"Best model: {best_model_name} with RMSE: {best_rmse:.2f}")
        
        # Save best model
        model_filename = f"models/{property_type.lower()}_xgboost_price_prediction.pkl"
        with open(model_filename, 'wb') as file:
            pickle.dump(best_model, file)
        print(f"Model saved to {model_filename}")
        
        # Also save a generic model file (for backward compatibility)
        if property_type.lower() == 'sale':  # Prioritize sale model as default
            with open("models/xgboost_price_prediction.pkl", 'wb') as file:
                pickle.dump(best_model, file)
            print("Default model saved to models/xgboost_price_prediction.pkl")
        
        # Store results for this property type
        model_results[property_type] = {
            'best_model': best_model_name,
            'rmse': best_rmse,
            'mae': mae,
            'r2': r2
        }
    
    return model_results

if __name__ == "__main__":
    print("Real Estate Price Prediction Model Training")
    print("=" * 50)
    results = train_prediction_models()
    print("\nTraining Complete!")
    if results:
        print("\nSummary of Results:")
        for prop_type, metrics in results.items():
            print(f"\n{prop_type} Properties:")
            print(f"  Best Model: {metrics['best_model']}")
            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  R² Score: {metrics['r2']:.2f}")
