import pandas as pd
import numpy as np
import os
import glob
import pickle
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


def train_prediction_models():
    """
    Train time series prediction models for real estate (rent/sale) prices.
    Saves the best model per type and returns performance metrics.
    """
    print("Starting model training process...")

    os.makedirs('models', exist_ok=True)

    # Load CSV data
    rental_path = "unegui_data/unegui_rental_data.csv"
    sales_path = "unegui_data/unegui_sales_data.csv"

    def load_data(path, label):
        if not os.path.exists(path):
            print(f"{label} file not found at: {path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
            date_str = os.path.basename(path).split('_')[-1].split('.')[0]
            df['Нийтэлсэн'] = pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')
            df['Type'] = label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return pd.DataFrame()

        # Remove duplicates based on available link/id column
        for col in ['Link', 'URL', 'url', 'link', 'Зар', 'ad_id']:
            if col in df.columns:
                df = df.drop_duplicates(subset=col)
                break
        else:
            print(f"No ID column found in {label} data to deduplicate.")

        return df

    rental_df = load_data(rental_path, 'Rent')
    sales_df = load_data(sales_path, 'Sale')

    if rental_df.empty and sales_df.empty:
        print("No valid data found to train models.")
        return

    df = pd.concat([rental_df, sales_df], ignore_index=True)
    df['Үнэ'] = pd.to_numeric(df['Үнэ'], errors='coerce')
    df.dropna(subset=['Нийтэлсэн', 'Үнэ'], inplace=True)

    model_results = {}

    for prop_type in df['Type'].unique():
        print(f"\nTraining model for: {prop_type}")

        type_df = df[df['Type'] == prop_type].copy()

        daily_data = type_df.groupby(type_df['Нийтэлсэн'].dt.date).agg(
            {'Үнэ': 'mean', 'Type': 'count'}).reset_index()
        daily_data.rename(columns={'Нийтэлсэн': 'date', 'Type': 'count'}, inplace=True)
        daily_data['date'] = pd.to_datetime(daily_data['date'])

        if len(daily_data) < 30:
            print(f"Skipping {prop_type} — only {len(daily_data)} days of data.")
            continue

        # Feature engineering
        daily_data['year'] = daily_data['date'].dt.year
        daily_data['month'] = daily_data['date'].dt.month
        daily_data['day'] = daily_data['date'].dt.day
        daily_data['dayofweek'] = daily_data['date'].dt.dayofweek

        for i in range(1, 8):
            daily_data[f'lag_{i}'] = daily_data['Үнэ'].shift(i)

        daily_data['rolling_7'] = daily_data['Үнэ'].rolling(7).mean()
        daily_data['rolling_14'] = daily_data['Үнэ'].rolling(14).mean()
        daily_data.dropna(inplace=True)

        # Define X and y
        X = daily_data.drop(['date', 'Үнэ', 'count'], axis=1)
        y = daily_data['Үнэ']

        # Time series split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }

        best_rmse = float('inf')
        best_model = None
        best_name = None
        best_metrics = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_name = name
                best_metrics = {'rmse': rmse, 'mae': mae, 'r2': r2}

        print(f"→ Best model for {prop_type}: {best_name}")

        model_file = f"models/{prop_type.lower()}_price_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Saved model: {model_file}")

        if prop_type.lower() == 'sale':
            with open("models/xgboost_price_prediction.pkl", 'wb') as f:
                pickle.dump(best_model, f)
            print("Also saved as default: models/xgboost_price_prediction.pkl")

        model_results[prop_type] = {
            'best_model': best_name,
            **best_metrics
        }

    return model_results


if __name__ == "__main__":
    print("Real Estate Price Prediction Model Training")
    print("=" * 50)
    results = train_prediction_models()
    print("\nTraining Complete!")

    if results:
        print("\nSummary of Results:")
        for t, r in results.items():
            print(f"\n{t} Properties:")
            print(f"  Best Model: {r['best_model']}")
            print(f"  RMSE: {r['rmse']:.2f}")
            print(f"  MAE: {r['mae']:.2f}")
            print(f"  R² Score: {r['r2']:.2f}")
