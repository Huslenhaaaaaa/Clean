import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')
# Set page configuration
st.set_page_config(
    page_title="Mongolia Real Estate Market Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f1f5f9;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data with duplicate URL checking
@st.cache_data(ttl=3600)
def load_data():
    # Find matching files
    rental_files = glob.glob("unegui_data/unegui_rental_data.csv")
    sales_files = glob.glob("unegui_data/unegui_sales_data.csv")

    def load_and_process(files, label):
        all_data = []
        for f in files:
            try:
                df = pd.read_csv(f, encoding='utf-8-sig')
                date_str = os.path.basename(f).split('_')[-1].split('.')[0]
                df['Scraped_date'] = pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')
                df['Type'] = label
                all_data.append(df)
            except Exception as e:
                st.warning(f"Error loading {f}: {e}")
        
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
        elif 'Зар' in combined_df.columns:  # This might be a link column in Mongolian
            url_col = 'Зар'
        else:
            # If no URL column is found, try to use ad_id as a unique identifier
            if 'ad_id' in combined_df.columns:
                url_col = 'ad_id'
            else:
                st.warning("No URL or ad_id column found. Cannot check for duplicates.")
                return combined_df
        
        # Count duplicates before removal
        duplicate_count = combined_df.duplicated(subset=[url_col]).sum()
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=[url_col], keep='first')
        
        # Log the number of duplicates removed
        if duplicate_count > 0:
            st.info(f"Removed {duplicate_count} duplicate listings based on {url_col}")
            
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
        st.error("❌ No rental or sales CSV files found.")
        return None

    # Numeric conversions
    df['Үнэ'] = pd.to_numeric(df['Үнэ'], errors='coerce')
    df['Rooms'] = df['ӨрөөнийТоо'].str.extract(r'(\d+)').astype(float)
    df['Area_m2'] = df['Талбай'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
    
    # Calculate price per m2 safely (avoid division by zero)
    df['Price_per_m2'] = np.where(df['Area_m2'] > 0, df['Үнэ'] / df['Area_m2'], np.nan)

    # Clean posted date
    def fix_posted_date(text):
        today = pd.Timestamp.today().normalize()
        if isinstance(text, str):
            if "өнөөдөр" in text.lower():
                return today
            elif "өчигдөр" in text.lower():
                return today - pd.Timedelta(days=1)
        try:
            return pd.to_datetime(text)
        except:
            return pd.NaT

    df['Fixed Posted Date'] = df['Нийтэлсэн'].apply(fix_posted_date)

    # Balcony and Garage detection
    df['HasBalcony'] = df['Тагт'].apply(lambda x: 'Yes' if isinstance(x, str) and 'байгаа' in x.lower() else 'No')
    df['HasGarage'] = df['Гараж'].apply(lambda x: 'Yes' if isinstance(x, str) and 'байгаа' in x.lower() else 'No')

    # Parse location
    def extract_location_details(location):
        if pd.isna(location): return pd.Series([None, None])
        parts = str(location).split(',')
        district = parts[0].strip() if parts else None
        subdistrict = parts[1].strip() if len(parts) > 1 else None
        return pd.Series([district, subdistrict])

    def extract_primary_district(location):
        if pd.isna(location): return None
        return str(location).split(',')[0].strip()

    def clean_sub_district(location):
        if pd.isna(location): return None
        parts = str(location).split(',')
        return parts[-1].strip() if len(parts) > 1 else None

    df[['District', 'Sub_District']] = df['Байршил'].apply(extract_location_details)
    df['Primary_District'] = df['Байршил'].apply(extract_primary_district)
    df['Clean_Sub_District'] = df['Байршил'].apply(clean_sub_district)

    return df

# Main function to build the dashboard
def main():
    st.markdown('<div class="main-header">Mongolia Real Estate Market Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Add data summary in expander
    with st.expander("Data Source Information"):
        st.info(f"Total listings loaded: {len(df)} (after removing duplicates)")
        if 'Scraped_date' in df.columns:
            # Check if there are valid date values
            valid_dates = df['Scraped_date'].dropna()
            if not valid_dates.empty:
                st.write(f"Data date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
            else:
                st.warning("No valid dates found in the data.")
        if 'Type' in df.columns:
            type_counts = df['Type'].value_counts()
            st.write("Property Types:")
            st.write(f"- Rent: {type_counts.get('Rent', 0)}")
            st.write(f"- Sale: {type_counts.get('Sale', 0)}")
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Add a date range filter if we have time-series data
    if 'Scraped_date' in df.columns:
        # Get valid dates, ensure they're not NaT
        valid_dates = df['Scraped_date'].dropna()
        
        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            
            # Ensure we have valid dates before creating the date_input widget
            if isinstance(min_date, date) and isinstance(max_date, date):
                date_range = st.sidebar.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    df = df[(df['Scraped_date'].dt.date >= start_date) & 
                            (df['Scraped_date'].dt.date <= end_date)]
    
    # Property Type Filter
    property_types = ['All']
    if 'Type' in df.columns:
        property_types += list(df['Type'].unique())
    selected_type = st.sidebar.selectbox("Property Type", property_types)
    if selected_type != 'All':
        df = df[df['Type'] == selected_type]
    
    # District Filter
    if 'Primary_District' in df.columns:
        valid_districts = [x for x in df['Primary_District'].dropna().unique() if isinstance(x, str)]
        districts = ['All'] + sorted(valid_districts)
        selected_district = st.sidebar.selectbox("District", districts)
        if selected_district != 'All':
            df = df[df['Primary_District'] == selected_district]
    
    # Room Filter
    if 'Rooms' in df.columns:
        valid_rooms = df['Rooms'].dropna()
        if not valid_rooms.empty:
            room_options = ['All'] + sorted([str(int(x)) for x in valid_rooms.unique() if pd.notna(x) and x > 0 and x < 10])
            selected_rooms = st.sidebar.multiselect("Number of Rooms", room_options, default=['All'])
            
            if 'All' not in selected_rooms and selected_rooms:
                # Convert string selections to numeric for filtering
                numeric_rooms = [float(x) for x in selected_rooms]
                df = df[df['Rooms'].isin(numeric_rooms)]
    
    # Price Range Filter
    if 'Үнэ' in df.columns and not df['Үнэ'].dropna().empty:
        min_price = int(df['Үнэ'].dropna().min())
        max_price = int(df['Үнэ'].dropna().max())
        
        if min_price < max_price:  # Ensure valid range
            price_range = st.sidebar.slider(
                "Price Range (₮)",
                min_price,
                max_price,
                (min_price, max_price),
                step=max(1, int((max_price - min_price) / 100))
            )
            df = df[(df['Үнэ'] >= price_range[0]) & (df['Үнэ'] <= price_range[1])]
    
    # Feature filters
    features_col1, features_col2 = st.sidebar.columns(2)
    with features_col1:
        if 'HasBalcony' in df.columns:
            balcony_option = st.radio("Balcony", ['All', 'Yes', 'No'])
            if balcony_option != 'All':
                df = df[df['HasBalcony'] == balcony_option]
    
    with features_col2:
        if 'HasGarage' in df.columns:
            garage_option = st.radio("Garage", ['All', 'Yes', 'No'])
            if garage_option != 'All':
                df = df[df['HasGarage'] == garage_option]
    
    # Display total counts after filtering
    st.sidebar.markdown("### Data Summary")
    st.sidebar.info(f"Total listings: {len(df)}")
    
    # Skip dashboard rendering if dataframe is empty after filtering
    if df.empty:
        st.warning("No data available with the current filters. Please adjust your filter criteria.")
        return
    
    # Main dashboard layout with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Market Overview", "Price Analysis", "Location Insights", "Property Features"])
    
    with tab1:
        st.markdown('<div class="sub-header">Market Overview</div>', unsafe_allow_html=True)
        
        # Summary metrics
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            avg_price = df['Үнэ'].mean()
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-label">Average Price</div>
                    <div class="metric-value">{avg_price:,.0f} ₮</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with metrics_col2:
            if 'Price_per_m2' in df.columns:
                avg_price_per_m2 = df['Price_per_m2'].mean()
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="metric-label">Avg Price per m²</div>
                        <div class="metric-value">{avg_price_per_m2:,.0f} ₮</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        with metrics_col3:
            if 'Area_m2' in df.columns:
                avg_area = df['Area_m2'].mean()
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="metric-label">Average Area</div>
                        <div class="metric-value">{avg_area:.1f} m²</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        with metrics_col4:
            if 'Rooms' in df.columns:
                avg_rooms = df['Rooms'].mean()
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="metric-label">Average Rooms</div>
                        <div class="metric-value">{avg_rooms:.1f}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        # Distribution of Rooms
        if 'Rooms' in df.columns:
            st.markdown("#### Room Distribution")
            room_counts = df['Rooms'].value_counts().sort_index()
            # Filter to just keep common room counts (1-6)
            room_counts = room_counts[room_counts.index.isin([1, 2, 3, 4, 5, 6])]
            
            fig_rooms = px.bar(
                x=room_counts.index,
                y=room_counts.values,
                labels={'x': 'Number of Rooms', 'y': 'Count'},
                color=room_counts.values,
                color_continuous_scale='Blues'
            )
            fig_rooms.update_layout(
                xaxis_title="Number of Rooms",
                yaxis_title="Number of Listings",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_rooms, use_container_width=True)
        
        # Distribution by District
        if 'Primary_District' in df.columns and df['Primary_District'].nunique() > 1:
            st.markdown("#### Listings by District")
            district_counts = df['Primary_District'].value_counts().nlargest(10)
            
            fig_district = px.bar(
                x=district_counts.index,
                y=district_counts.values,
                labels={'x': 'District', 'y': 'Count'},
                color=district_counts.values,
                color_continuous_scale='Greens'
            )
            fig_district.update_layout(
                xaxis_title="District",
                yaxis_title="Number of Listings",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_district, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="sub-header">Price Analysis</div>', unsafe_allow_html=True)
        
        # Price by Room Count
        if 'Rooms' in df.columns and 'Үнэ' in df.columns:
            st.markdown("#### Average Price by Room Count")
            # Group by rooms and calculate average price
            price_by_rooms = df.groupby('Rooms')['Үнэ'].mean().reset_index()
            price_by_rooms = price_by_rooms[price_by_rooms['Rooms'].between(1, 6)]  # Filter to common room counts
            
            fig_price_rooms = px.bar(
                price_by_rooms,
                x='Rooms',
                y='Үнэ',
                labels={'Үнэ': 'Average Price (₮)', 'Rooms': 'Number of Rooms'},
                color='Үнэ',
                color_continuous_scale='Reds'
            )
            fig_price_rooms.update_layout(
                xaxis_title="Number of Rooms",
                yaxis_title="Average Price (₮)",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_price_rooms, use_container_width=True)
        
        # Price Distribution
        if 'Үнэ' in df.columns:
            st.markdown("#### Price Distribution")
            
            fig_price_hist = px.histogram(
                df,
                x='Үнэ',
                nbins=30,
                labels={'Үнэ': 'Price (₮)'},
                color_discrete_sequence=['#3B82F6']
            )
            fig_price_hist.update_layout(
                xaxis_title="Price (₮)",
                yaxis_title="Number of Listings"
            )
            st.plotly_chart(fig_price_hist, use_container_width=True)
        
        # Price per Square Meter by District
        if 'Price_per_m2' in df.columns and 'Primary_District' in df.columns:
            st.markdown("#### Price per m² by District")
            
            # Group by district and calculate average price per m²
            price_per_m2_by_district = df.groupby('Primary_District')['Price_per_m2'].mean().reset_index()
            # Sort by price and take top 10
            price_per_m2_by_district = price_per_m2_by_district.sort_values('Price_per_m2', ascending=False).head(10)
            
            fig_price_district = px.bar(
                price_per_m2_by_district,
                x='Primary_District',
                y='Price_per_m2',
                labels={'Price_per_m2': 'Price per m² (₮)', 'Primary_District': 'District'},
                color='Price_per_m2',
                color_continuous_scale='Purples'
            )
            fig_price_district.update_layout(
                xaxis_title="District",
                yaxis_title="Average Price per m² (₮)",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_price_district, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="sub-header">Location Insights</div>', unsafe_allow_html=True)
        
        # District popularity with price overlay
        if 'Primary_District' in df.columns and 'Үнэ' in df.columns:
            st.markdown("#### District Analysis: Listings vs Price")
            
            # Group by district
            district_data = df.groupby('Primary_District').agg({
                'Үнэ': 'mean',
                'ad_id': 'count'  # Count of listings
            }).reset_index()
            
            district_data = district_data.rename(columns={'ad_id': 'Number of Listings'})
            district_data = district_data.sort_values('Number of Listings', ascending=False).head(10)
            
            # Create dual-axis chart
            fig = go.Figure()
            
            # Add bar chart for number of listings
            fig.add_trace(go.Bar(
                x=district_data['Primary_District'],
                y=district_data['Number of Listings'],
                name='Number of Listings',
                marker_color='#60A5FA'
            ))
            
            # Add line chart for average price
            fig.add_trace(go.Scatter(
                x=district_data['Primary_District'],
                y=district_data['Үнэ'],
                name='Average Price (₮)',
                marker_color='#EF4444',
                mode='lines+markers',
                yaxis='y2'
            ))
            
            # Set up the layout with dual y-axes
            fig.update_layout(
                title='District Popularity vs. Average Price',
                xaxis_title='District',
                yaxis=dict(
                    title=dict(
                        text='Number of Listings',
                        font=dict(color='#60A5FA')
                    ),
                    tickfont=dict(color='#60A5FA')
                ),
                yaxis2=dict(
                    title=dict(
                        text='Average Price (₮)',
                        font=dict(color='#EF4444')
                    ),
                    tickfont=dict(color='#EF4444'),
                    anchor='x',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Popular locations/neighborhoods analysis
        if 'Байршил' in df.columns:
            st.markdown("#### Top Neighborhoods")
            
            # Count listings by location and get top 15
            location_counts = df['Байршил'].value_counts().head(15)
            
            fig_locations = px.bar(
                x=location_counts.index,
                y=location_counts.values,
                labels={'x': 'Neighborhood', 'y': 'Number of Listings'},
                color=location_counts.values,
                color_continuous_scale='Teal'
            )
            fig_locations.update_layout(
                xaxis_title="Neighborhood",
                yaxis_title="Number of Listings",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_locations, use_container_width=True)
    
    with tab4:
        st.markdown('<div class="sub-header">Property Features</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Balcony Distribution
            if 'HasBalcony' in df.columns:
                balcony_counts = df['HasBalcony'].value_counts()
                
                fig_balcony = px.pie(
                    values=balcony_counts.values,
                    names=balcony_counts.index,
                    title="Properties with Balcony",
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                fig_balcony.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_balcony, use_container_width=True)
        
        with col2:
            # Garage Distribution
            if 'HasGarage' in df.columns:
                garage_counts = df['HasGarage'].value_counts()
                
                fig_garage = px.pie(
                    values=garage_counts.values,
                    names=garage_counts.index,
                    title="Properties with Garage",
                    color_discrete_sequence=px.colors.sequential.Greens
                )
                fig_garage.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_garage, use_container_width=True)
        
        # Analysis of other features
        st.markdown("#### Building Features")
        
        if 'Building Year' in df.columns or 'Ашиглалтандорсонон' in df.columns:
            # Use either column that contains building year information
            year_col = 'Building Year' if 'Building Year' in df.columns else 'Ашиглалтандорсонон'
            
            # Try to convert to numeric - this needs proper preprocessing based on the actual data format
            try:
                df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
                year_counts = df[year_col].value_counts().sort_index().head(20)
                
                fig_year = px.bar(
                    x=year_counts.index,
                    y=year_counts.values,
                    labels={'x': 'Year Built', 'y': 'Count'},
                    title="Properties by Year Built",
                    color=year_counts.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_year, use_container_width=True)
            except:
                st.write("Building year data requires additional preprocessing")
        
        # Floor distribution
        if 'Хэдэндавхарт' in df.columns:
            st.markdown("#### Floor Distribution")
            
            # Extract floor numbers
            df['Floor'] = df['Хэдэндавхарт'].astype(str).str.extract(r'(\d+)').astype(float)
            floor_counts = df['Floor'].value_counts().sort_index()
            
            # Filter to reasonable floor numbers
            floor_counts = floor_counts[floor_counts.index <= 25]
            
            fig_floor = px.bar(
                x=floor_counts.index,
                y=floor_counts.values,
                labels={'x': 'Floor', 'y': 'Count'},
                color=floor_counts.values,
                color_continuous_scale='YlOrRd'
            )
            fig_floor.update_layout(
                xaxis_title="Floor",
                yaxis_title="Number of Listings",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_floor, use_container_width=True)
    
    # Add data trends over time section if we have time-series data
    if 'Scraped_date' in df.columns and df['Scraped_date'].nunique() > 1:
        st.markdown("---")
        st.markdown('<div class="sub-header">Data Trends Over Time</div>', unsafe_allow_html=True)
        
        # Group by date and calculate daily averages
        time_data = df.groupby(df['Scraped_date'].dt.date).agg({
            'Үнэ': 'mean',
            'ad_id': 'count',
            'Price_per_m2': 'mean'
        }).reset_index()
        
        # Plot price trends over time
        st.markdown("#### Price Trends")
        
        fig_trends = go.Figure()
        
        fig_trends.add_trace(go.Scatter(
            x=time_data['Scraped_date'],
            y=time_data['Үнэ'],
            mode='lines+markers',
            name='Average Price (₮)',
            line=dict(color='#2563EB', width=2)
        ))
        
        fig_trends.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Price (₮)",
            title="Average Price Trend Over Time"
        )
        
        st.plotly_chart(fig_trends, use_container_width=True)
        
        # Plot listing volume over time
        st.markdown("#### Listing Volume Trends")
        
        fig_volume = go.Figure()
        
        fig_volume.add_trace(go.Scatter(
            x=time_data['Scraped_date'],
            y=time_data['ad_id'],
            mode='lines+markers',
            name='Number of Listings',
            line=dict(color='#10B981', width=2),
            fill='tozeroy'
        ))
        
        fig_volume.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Listings",
            title="Daily Listing Volume"
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)

# Function to load ML models
@st.cache_resource
def load_prediction_models():
    """Load pre-trained ML models for price predictions."""
    models = {}
    
    # Look for available model files in the models directory
    model_files = glob.glob("models/*.pkl")
    if not model_files:
        st.warning("No ML models found in 'models/' directory.")
        return None
    
    try:
        # Load each model
        for model_file in model_files:
            model_name = os.path.basename(model_file).replace('.pkl', '')
            with open(model_file, 'rb') as f:
                models[model_name] = pickle.load(f)
            st.success(f"Loaded model: {model_name}")
    except Exception as e:
        st.error(f"Error loading ML models: {e}")
        return None
    
    return models

# Function to make predictions
def predict_prices(df, models, prediction_period=12, freq='M'):
    """
    Generate price predictions based on loaded models
    
    Args:
        df: DataFrame with historical data
        models: Dictionary of loaded ML models
        prediction_period: Number of periods to predict
        freq: Frequency for predictions ('M' for monthly, 'W' for weekly)
    
    Returns:
        Dictionary of prediction DataFrames for different property types/districts
    """
    predictions = {}
    
    if not models:
        return predictions
    
    # Get the latest date in our dataset
    try:
        last_date = df['Scraped_date'].max()
    except:
        st.error("Could not determine last date in dataset")
        return predictions
    
    # Generate future dates for prediction
    if freq == 'M':
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=prediction_period,
            freq='MS'  # Month Start
        )
    else:
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=prediction_period,
            freq='W'  # Weekly
        )
    
    # For each model, generate predictions
    for model_name, model in models.items():
        try:
            # Parse model name to determine what it predicts
            # Example format: "rental_BayangolDistrict_model" or "sales_all_districts_model"
            parts = model_name.split('_')
            property_type = parts[0]  # 'rental' or 'sales'
            
            # Create prediction input features
            # This would depend on what features your model expects
            # Here's a simple example assuming time-based features only
            X_future = pd.DataFrame({
                'date_number': range(len(future_dates)),  # Numerical representation of date
                'month': future_dates.month,  # Month as a feature
                'year': future_dates.year,    # Year as a feature
            })
            
            # Some models might need additional features like:
            # 'district_encoded', 'rooms', etc.
            # You would need to prepare these based on your model's requirements
            
            # Make predictions
            future_prices = model.predict(X_future)
            
            # Create prediction dataframe
            pred_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Price': future_prices
            })
            
            predictions[model_name] = pred_df
            
        except Exception as e:
            st.error(f"Error making predictions with model {model_name}: {e}")
    
    return predictions

# Function to display prediction graphs
def display_prediction_graphs(df, predictions):
    """Display interactive graphs of historical data and predictions"""
    
    if not predictions:
        st.warning("No predictions available to display.")
        return
    
    # For each prediction model, create a visualization
    for model_name, pred_df in predictions.items():
        try:
            # Parse the model name to get property type and district
            parts = model_name.split('_')
            property_type = parts[0].capitalize()
            
            if 'district' in model_name.lower():
                district = parts[1].replace('District', ' District')
                title = f"{property_type} Price Predictions - {district}"
            else:
                title = f"{property_type} Price Predictions - All Areas"
            
            # Create a figure with historical + predicted data
            fig = go.Figure()
            
            # Get historical data for this property type/district
            if 'Type' in df.columns:
                historical_type = 'Rent' if property_type.lower() == 'rental' else 'Sale'
                hist_data = df[df['Type'] == historical_type].copy()
            else:
                hist_data = df.copy()
            
            # If district-specific model, filter historical data
            if 'district' in model_name.lower() and 'Primary_District' in hist_data.columns:
                district_name = parts[1]
                # Clean up district name for matching
                clean_district = district_name.replace('District', '').strip()
                hist_data = hist_data[hist_data['Primary_District'].str.contains(clean_district, case=False, na=False)]
            
            # Group by date and get average price
            if not hist_data.empty and 'Scraped_date' in hist_data.columns:
                hist_data_agg = hist_data.groupby(hist_data['Scraped_date'].dt.date)['Үнэ'].mean().reset_index()
                
                # Add historical data trace
                fig.add_trace(go.Scatter(
                    x=hist_data_agg['Scraped_date'],
                    y=hist_data_agg['Үнэ'],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='#3366CC', width=2)
                ))
            
            # Add prediction trace
            fig.add_trace(go.Scatter(
                x=pred_df['Date'],
                y=pred_df['Predicted_Price'],
                mode='lines+markers',
                name='Predictions',
                line=dict(color='#FF9900', width=2, dash='dash')
            ))
            
            # Add confidence interval if available
            if 'Upper_Bound' in pred_df.columns and 'Lower_Bound' in pred_df.columns:
                fig.add_trace(go.Scatter(
                    x=pred_df['Date'].tolist() + pred_df['Date'].tolist()[::-1],
                    y=pred_df['Upper_Bound'].tolist() + pred_df['Lower_Bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,153,0,0.2)',
                    line=dict(color='rgba(255,153,0,0)'),
                    name='95% Confidence Interval'
                ))
            
            # Improve layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Price (₮)",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"
                ),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights about the prediction
            if len(pred_df) > 0:
                latest_price = hist_data_agg['Үнэ'].iloc[-1] if not hist_data_agg.empty else 0
                future_price = pred_df['Predicted_Price'].iloc[-1]
                percent_change = ((future_price - latest_price) / latest_price * 100) if latest_price > 0 else 0
                
                direction = "increase" if percent_change > 0 else "decrease"
                
                st.markdown(f"""
                <div class="highlight">
                    <strong>Forecast Insights:</strong><br>
                    The model predicts a {abs(percent_change):.1f}% {direction} in {property_type.lower()} prices 
                    by {pred_df['Date'].iloc[-1].strftime('%B %Y')}.
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error displaying predictions for {model_name}: {e}")

# Function to add the ML prediction section to the dashboard
def add_ml_prediction_section(df):
    """Add ML prediction section to the dashboard"""
    
    st.markdown('<div class="main-header">Price Predictions</div>', unsafe_allow_html=True)
    
    # Load ML models
    models = load_prediction_models()
    
    if not models:
        st.warning("""
        No machine learning models found. To enable predictions:
        1. Create a 'models/' directory in your project
        2. Add trained .pkl model files (e.g., 'rental_BayangolDistrict_model.pkl')
        """)
        return
    
    # Add prediction settings in sidebar
    st.sidebar.markdown("### Prediction Settings")
    prediction_period = st.sidebar.slider("Prediction Period (Months)", 3, 24, 12)
    
    # Generate predictions
    predictions = predict_prices(df, models, prediction_period=prediction_period)
    
    # Create tabs for different prediction views
    pred_tab1, pred_tab2 = st.tabs(["Price Predictions", "Prediction Insights"])
    
    with pred_tab1:
        # Display prediction graphs
        display_prediction_graphs(df, predictions)
    
    with pred_tab2:
        # Add more detailed analysis of predictions
        st.markdown("#### Price Trend Analysis")
        
        # Create a table comparing current prices to predicted future prices
        if predictions:
            comparison_data = []
            
            for model_name, pred_df in predictions.items():
                try:
                    # Parse model info
                    parts = model_name.split('_')
                    property_type = parts[0].capitalize()
                    
                    if len(parts) > 1:
                        location = parts[1].replace('District', ' District')
                    else:
                        location = "All Areas"
                    
                    # Get current price (from historical data)
                    if 'Type' in df.columns:
                        historical_type = 'Rent' if property_type.lower() == 'rental' else 'Sale'
                        hist_data = df[df['Type'] == historical_type].copy()
                    else:
                        hist_data = df.copy()
                    
                    # Filter by location if needed
                    if location != "All Areas" and 'Primary_District' in hist_data.columns:
                        clean_location = location.replace(' District', '').strip()
                        hist_data = hist_data[hist_data['Primary_District'].str.contains(clean_location, case=False, na=False)]
                    
                    current_price = hist_data['Үнэ'].mean() if not hist_data.empty else 0
                    
                    # Get predicted future prices
                    future_short = pred_df['Predicted_Price'].iloc[2] if len(pred_df) > 2 else None
                    future_medium = pred_df['Predicted_Price'].iloc[5] if len(pred_df) > 5 else None
                    future_long = pred_df['Predicted_Price'].iloc[-1] if not pred_df.empty else None
                    
                    # Calculate percent changes
                    short_change = ((future_short - current_price) / current_price * 100) if current_price > 0 and future_short is not None else 0
                    medium_change = ((future_medium - current_price) / current_price * 100) if current_price > 0 and future_medium is not None else 0
                    long_change = ((future_long - current_price) / current_price * 100) if current_price > 0 and future_long is not None else 0
                    
                    # Add to comparison data
                    comparison_data.append({
                        "Property Type": property_type,
                        "Location": location,
                        "Current Avg Price": current_price,
                        "3-Month Forecast": future_short,
                        "3-Month Change %": short_change,
                        "6-Month Forecast": future_medium,
                        "6-Month Change %": medium_change,
                        f"{prediction_period}-Month Forecast": future_long,
                        f"{prediction_period}-Month Change %": long_change
                    })
                
                except Exception as e:
                    st.error(f"Error processing prediction data for {model_name}: {e}")
            
            if comparison_data:
                # Convert to DataFrame for display
                comparison_df = pd.DataFrame(comparison_data)
                
                # Format for display
                for col in ["Current Avg Price", "3-Month Forecast", "6-Month Forecast", f"{prediction_period}-Month Forecast"]:
                    if col in comparison_df.columns:
                        comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:,.0f} ₮" if pd.notna(x) else "N/A")
                
                for col in ["3-Month Change %", "6-Month Change %", f"{prediction_period}-Month Change %"]:
                    if col in comparison_df.columns:
                        comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
                
                # Display the comparison table
                st.dataframe(comparison_df, use_container_width=True)
                
                # Add investment recommendation section
                st.markdown("#### Investment Recommendations")
                
                # Find areas with highest predicted growth
                best_areas = sorted([d for d in comparison_data if pd.notna(d[f"{prediction_period}-Month Change %"])], 
                                   key=lambda x: x[f"{prediction_period}-Month Change %"], 
                                   reverse=True)[:3]
                
                if best_areas:
                    st.markdown("**Top areas for potential investment:**")
                    for i, area in enumerate(best_areas, 1):
                        st.markdown(f"""
                        {i}. **{area['Location']} ({area['Property Type']})** - 
                        Predicted growth: {area[f"{prediction_period}-Month Change %"]:.1f}%
                        """)
        else:
            st.info("No prediction data available to analyze.")

# Add feature importance visualization if models expose that information
def display_feature_importance(models):
    """Display feature importance from the ML models if available"""
    
    if not models:
        return
    
    st.markdown("#### Model Feature Importance")
    st.write("What factors most influence property prices according to our models:")
    
    # Check if any model exposes feature importances
    for model_name, model in models.items():
        # For tree-based models like RandomForest, XGBoost
        if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
            try:
                # Create feature importance DataFrame
                feature_importance = pd.DataFrame({
                    'Feature': model.feature_names_in_,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Create bar chart of feature importance
                fig = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Feature Importance - {model_name}",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    xaxis_title="Importance Score",
                    yaxis_title="Feature"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                break
            except Exception as e:
                st.error(f"Error displaying feature importance: {e}")

# Function to simulate predictions for demonstration purposes
def generate_demo_predictions(df):
    """Generate simulated predictions when no models are available"""
    
    st.info("⚠️ Using simulated predictions for demonstration. To use real ML predictions, add trained models to your project.")
    
    predictions = {}
    
    # Get the latest date in our dataset
    try:
        last_date = df['Scraped_date'].max()
    except:
        # If no date column, use today
        last_date = datetime.now()
    
    # Generate future dates for prediction (12 months)
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=12,
        freq='MS'  # Month Start
    )
    
    # Get unique districts and property types
    districts = []
    if 'Primary_District' in df.columns:
        districts = df['Primary_District'].dropna().unique()[:3]  # Take top 3 districts
    
    property_types = []
    if 'Type' in df.columns:
        property_types = df['Type'].unique()
    else:
        property_types = ['Sale', 'Rent']  # Default if not available
    
    # Create demo models for different combinations
    for prop_type in property_types:
        # Convert type to match expected model name format
        type_key = 'rental' if prop_type == 'Rent' else 'sales'
        
        # Create all areas prediction
        base_price = df[df['Type'] == prop_type]['Үнэ'].mean() if 'Type' in df.columns else df['Үнэ'].mean()
        
        # Generate a slightly increasing trend with some randomness
        trend_factor = np.linspace(1.0, 1.15, len(future_dates))  # 15% increase over period
        seasonal_factor = 1 + 0.03 * np.sin(np.pi * np.arange(len(future_dates)) / 6)  # Seasonal variation
        random_factor = np.random.normal(1, 0.02, size=len(future_dates))  # Random noise
        
        predicted_prices = base_price * trend_factor * seasonal_factor * random_factor
        
        # Create prediction dataframe for all areas
        all_areas_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': predicted_prices,
            'Lower_Bound': predicted_prices * 0.95,  # 5% lower bound
            'Upper_Bound': predicted_prices * 1.05   # 5% upper bound
        })
        
        predictions[f"{type_key}_all_areas_model"] = all_areas_df
        
        # Create district-specific predictions
        for district in districts:
            # Clean district name for key
            district_key = district.replace(' ', '')
            
            # Filter data for this district
            district_data = df[(df['Type'] == prop_type) & (df['Primary_District'] == district)] if 'Type' in df.columns else df[df['Primary_District'] == district]
            
            # Get base price for district
            district_base_price = district_data['Үнэ'].mean() if not district_data.empty else base_price
            
            # Generate slightly different trend for this district
            district_trend = np.linspace(1.0, 1.1 + np.random.uniform(-0.05, 0.15), len(future_dates))
            district_seasonal = 1 + 0.04 * np.sin(np.pi * np.arange(len(future_dates)) / 6 + np.random.uniform(0, 2))
            district_random = np.random.normal(1, 0.03, size=len(future_dates))
            
            district_prices = district_base_price * district_trend * district_seasonal * district_random
            
            district_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Price': district_prices,
                'Lower_Bound': district_prices * 0.93,  # 7% lower bound
                'Upper_Bound': district_prices * 1.07   # 7% upper bound
            })
            
            predictions[f"{type_key}_{district_key}District_model"] = district_df
    
    return predictions

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6B7280; padding: 1rem;">
        Data scraped from Unegui.mn | Dashboard updated: {datetime.now().strftime("%Y-%m-%d")}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
