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


def add_prediction_section(df):
    """
    Add a machine learning prediction section to the dashboard
    """
    st.markdown("---")
    st.markdown('<div class="sub-header">Price Predictions</div>', unsafe_allow_html=True)
    
    # Check if we have enough time series data
    if 'Scraped_date' not in df.columns or df['Scraped_date'].nunique() < 5:
        st.warning("Not enough time series data available for predictions. Need at least 5 different dates.")
        return
    
    # Create tabs for different prediction types
    pred_tab1, pred_tab2, pred_tab3 = st.tabs(["Time Series Forecast", "Price by Features", "District Comparison"])
    
    # Tab 1: Time Series Forecasting
    with pred_tab1:
        st.markdown("#### Price Trend Forecast")
        st.info("This forecast predicts future average prices based on historical trends")
        
        # Select forecast type
        forecast_type = st.radio(
            "Select property type to forecast:",
            ["All Properties", "Rental Only", "Sales Only"],
            horizontal=True
        )
        
        # Filter data based on selection
        forecast_df = df.copy()
        if forecast_type == "Rental Only" and 'Type' in df.columns:
            forecast_df = forecast_df[forecast_df['Type'] == 'Rent']
        elif forecast_type == "Sales Only" and 'Type' in df.columns:
            forecast_df = forecast_df[forecast_df['Type'] == 'Sale']
        
        # Check if we have enough data after filtering
        if len(forecast_df) < 20:
            st.warning("Not enough data for this property type to make reliable predictions.")
            return
        
        # Create time series data
        ts_data = forecast_df.groupby(forecast_df['Scraped_date'].dt.date).agg({
            'Үнэ': 'mean',
            'ad_id': 'count'
        }).reset_index()
        
        # Only proceed if we have enough data points
        if len(ts_data) >= 5:
            # Number of days to forecast
            forecast_days = st.slider("Number of days to forecast:", 7, 30, 14)
            
            # Choose forecasting model
            model_type = st.selectbox(
                "Select forecasting model:",
                ["Simple Trend (Linear)", "ARIMA (Time Series)"]
            )
            
            # Prepare data for forecasting
            ts_data = ts_data.sort_values('Scraped_date')
            ts_data['Days'] = range(len(ts_data))
            max_date = ts_data['Scraped_date'].max()
            
            if model_type == "Simple Trend (Linear)":
                # Simple linear trend model
                X = ts_data['Days'].values.reshape(-1, 1)
                y = ts_data['Үнэ'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Create forecast dates and features
                future_dates = [max_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
                future_X = np.array(range(len(ts_data), len(ts_data) + forecast_days)).reshape(-1, 1)
                
                # Make predictions
                future_y = model.predict(future_X)
                
                # Create forecast DataFrame
                forecast_result = pd.DataFrame({
                    'Scraped_date': future_dates,
                    'Forecast': future_y,
                    'Type': 'Forecast'
                })
                
                # Add confidence intervals (simple approach)
                train_pred = model.predict(X)
                mae = mean_absolute_error(y, train_pred)
                forecast_result['Upper'] = forecast_result['Forecast'] + mae * 1.96
                forecast_result['Lower'] = forecast_result['Forecast'] - mae * 1.96
                
                # Plot results
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=ts_data['Scraped_date'],
                    y=ts_data['Үнэ'],
                    mode='lines+markers',
                    name='Historical Prices',
                    line=dict(color='#2563EB', width=2)
                ))
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=forecast_result['Scraped_date'],
                    y=forecast_result['Forecast'],
                    mode='lines+markers',
                    name='Price Forecast',
                    line=dict(color='#EF4444', width=2, dash='dash')
                ))
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_result['Scraped_date'].tolist() + forecast_result['Scraped_date'].tolist()[::-1],
                    y=forecast_result['Upper'].tolist() + forecast_result['Lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(231, 84, 128, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    name='95% Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f"{forecast_type} Price Forecast ({forecast_days} days ahead)",
                    xaxis_title="Date",
                    yaxis_title="Average Price (₮)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display forecast stats
                current_price = ts_data['Үнэ'].iloc[-1]
                final_forecast = forecast_result['Forecast'].iloc[-1]
                price_change = ((final_forecast - current_price) / current_price) * 100
                
                st.metric(
                    label=f"Forecasted price in {forecast_days} days",
                    value=f"{final_forecast:,.0f} ₮",
                    delta=f"{price_change:.1f}%"
                )
                
                # Model metrics
                st.markdown("##### Model Performance")
                r2 = r2_score(y, train_pred)
                st.write(f"R² Score: {r2:.2f}")
                st.write(f"Mean Absolute Error: {mae:,.0f} ₮")
                
            elif model_type == "ARIMA (Time Series)":
                try:
                    # ARIMA forecasting
                    st.info("Training ARIMA model (this may take a moment)...")
                    
                    # Use a simple ARIMA model with fixed parameters for simplicity
                    model = ARIMA(ts_data['Үнэ'].values, order=(1, 1, 1))
                    model_fit = model.fit()
                    
                    # Make prediction
                    forecast_result = model_fit.forecast(steps=forecast_days)
                    
                    # Create forecast DataFrame
                    future_dates = [max_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
                    forecast_df = pd.DataFrame({
                        'Scraped_date': future_dates,
                        'Forecast': forecast_result,
                    })
                    
                    # Plot results
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=ts_data['Scraped_date'],
                        y=ts_data['Үнэ'],
                        mode='lines+markers',
                        name='Historical Prices',
                        line=dict(color='#2563EB', width=2)
                    ))
                    
                    # Add forecast line
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Scraped_date'],
                        y=forecast_df['Forecast'],
                        mode='lines+markers',
                        name='ARIMA Forecast',
                        line=dict(color='#10B981', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"{forecast_type} ARIMA Price Forecast ({forecast_days} days ahead)",
                        xaxis_title="Date",
                        yaxis_title="Average Price (₮)",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display forecast stats
                    current_price = ts_data['Үнэ'].iloc[-1]
                    final_forecast = forecast_df['Forecast'].iloc[-1]
                    price_change = ((final_forecast - current_price) / current_price) * 100
                    
                    st.metric(
                        label=f"Forecasted price in {forecast_days} days",
                        value=f"{final_forecast:,.0f} ₮",
                        delta=f"{price_change:.1f}%"
                    )
                    
                    # Add model performance
                    st.markdown("##### Model Performance")
                    st.write("ARIMA Model Summary")
                    st.text(str(model_fit.summary()))
                    
                except Exception as e:
                    st.error(f"Error in ARIMA modeling: {e}")
                    st.warning("Try using 'Simple Trend (Linear)' model instead.")
        else:
            st.warning("Not enough time series data points for forecasting.")
    
    # Tab 2: Price by Features Prediction
    with pred_tab2:
        st.markdown("#### Price Prediction by Property Features")
        st.info("Predict property prices based on features like size, rooms, and location")
        
        # Only run if we have necessary features
        if all(col in df.columns for col in ['Rooms', 'Area_m2', 'Primary_District']):
            
            # Choose property type
            prop_type = 'All'
            if 'Type' in df.columns:
                prop_type = st.radio(
                    "Property type to model:",
                    ["All", "Rent", "Sale"],
                    horizontal=True
                )
                
                if prop_type != 'All':
                    model_df = df[df['Type'] == prop_type].copy()
                else:
                    model_df = df.copy()
            else:
                model_df = df.copy()
            
            # Ensure we have enough data
            if len(model_df) < 50:
                st.warning("Not enough data for this property type to build a reliable model.")
                return
            
            # Select algorithm
            algo = st.selectbox(
                "Select prediction algorithm:",
                ["Random Forest", "Linear Regression"]
            )
            
            # Prepare data
            model_df = model_df.dropna(subset=['Area_m2', 'Rooms', 'Primary_District', 'Үнэ'])
            
            # One-hot encode districts
            district_dummies = pd.get_dummies(model_df['Primary_District'], prefix='district')
            model_data = pd.concat([
                model_df[['Area_m2', 'Rooms', 'Үнэ']],
                district_dummies
            ], axis=1)
            
            # Drop any remaining NaN
            model_data = model_data.dropna()
            
            if len(model_data) < 50:
                st.warning("Not enough clean data to build a reliable model after filtering.")
                return
            
            # Split features and target
            X = model_data.drop('Үнэ', axis=1)
            y = model_data['Үнэ']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            if algo == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = LinearRegression()
            
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            
            train_mae = mean_absolute_error(y_train, train_preds)
            test_mae = mean_absolute_error(y_test, test_preds)
            
            train_r2 = r2_score(y_train, train_preds)
            test_r2 = r2_score(y_test, test_preds)
            
            # Display model performance metrics
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.metric("Training R² Score", f"{train_r2:.2f}")
                st.metric("Training MAE", f"{train_mae:,.0f} ₮")
                
            with metric_col2:
                st.metric("Testing R² Score", f"{test_r2:.2f}")
                st.metric("Testing MAE", f"{test_mae:,.0f} ₮")
            
            # Feature importance (for Random Forest)
            if algo == "Random Forest":
                st.markdown("##### Feature Importance")
                importance = model.feature_importances_
                feature_names = X.columns
                
                # Sort features by importance
                indices = np.argsort(importance)[::-1]
                
                # Plot feature importance
                fig = go.Figure(go.Bar(
                    x=[importance[i] * 100 for i in indices[:10]],  # Show top 10 features
                    y=[feature_names[i] for i in indices[:10]],
                    orientation='h',
                    marker_color='#3B82F6'
                ))
                
                fig.update_layout(
                    title="Top 10 Features by Importance",
                    xaxis_title="Importance (%)",
                    yaxis_title="Feature",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Interactive prediction tool
            st.markdown("### Predict Property Price")
            st.markdown("Use the sliders below to predict property price based on features")
            
            # Get input features from user
            pred_col1, pred_col2 = st.columns(2)
            
            with pred_col1:
                pred_area = st.slider(
                    "Property Area (m²):",
                    min_value=int(model_df['Area_m2'].min()),
                    max_value=int(model_df['Area_m2'].max()),
                    value=int(model_df['Area_m2'].median()),
                    step=1
                )
                
            with pred_col2:
                pred_rooms = st.slider(
                    "Number of Rooms:",
                    min_value=int(model_df['Rooms'].min()),
                    max_value=int(model_df['Rooms'].max()),
                    value=int(model_df['Rooms'].median()),
                    step=1
                )
            
            pred_district = st.selectbox(
                "Select District:",
                sorted(model_df['Primary_District'].unique())
            )
            
            # Create prediction input
            pred_input = pd.DataFrame({
                'Area_m2': [pred_area],
                'Rooms': [pred_rooms]
            })
            
            # One-hot encode the district
            for district in district_dummies.columns:
                if district == f'district_{pred_district}':
                    pred_input[district] = 1
                else:
                    pred_input[district] = 0
            
            # Make prediction
            prediction = model.predict(pred_input)[0]
            
            # Display prediction result with styling
            st.markdown(
                f"""
                <div class="card" style="background-color:#f0f9ff;padding:20px;border-radius:10px;text-align:center;">
                    <div class="metric-label" style="font-size:1.2rem;color:#6B7280;">Predicted Property Price</div>
                    <div class="metric-value" style="font-size:2.5rem;font-weight:700;color:#1E40AF;">{prediction:,.0f} ₮</div>
                    <div style="font-size:0.9rem;color:#6B7280;">Based on {algo} model | Accuracy: {test_r2:.2f} R²</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Price scatter plot
            st.markdown("##### Actual vs. Predicted Prices")
            
            # Create scatter plot of actuals vs. predicted (test set)
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=y_test,
                y=test_preds,
                mode='markers',
                marker=dict(
                    size=8,
                    color='#3B82F6',
                    opacity=0.6
                ),
                name='Test Data'
            ))
            
            # Add perfect prediction line
            min_val = min(y_test.min(), test_preds.min())
            max_val = max(y_test.max(), test_preds.max())
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction'
            ))
            
            fig.update_layout(
                title="Model Performance: Actual vs. Predicted Prices",
                xaxis_title="Actual Price (₮)",
                yaxis_title="Predicted Price (₮)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required features (Rooms, Area_m2, Primary_District) not found in the dataset.")
    
    # Tab 3: District Price Comparison and Forecast
    with pred_tab3:
        st.markdown("#### District Price Comparison & Forecast")
        st.info("Compare price trends across different districts")
        
        # Only proceed if we have districts and enough time series data
        if 'Primary_District' in df.columns and df['Scraped_date'].nunique() >= 5:
            # Get top districts by listing count
            top_districts = df['Primary_District'].value_counts().nlargest(5).index.tolist()
            
            # Let user select districts to compare
            selected_districts = st.multiselect(
                "Select districts to compare:",
                options=sorted(df['Primary_District'].unique()),
                default=top_districts[:3] if len(top_districts) >= 3 else top_districts
            )
            
            if not selected_districts:
                st.warning("Please select at least one district to analyze")
                return
            
            # Filter data for selected districts
            district_df = df[df['Primary_District'].isin(selected_districts)].copy()
            
            # Group by date and district
            district_ts = district_df.groupby([
                district_df['Scraped_date'].dt.date, 'Primary_District'
            ]).agg({
                'Үнэ': 'mean',
                'ad_id': 'count'
            }).reset_index()
            
            # Plot price trends by district
            fig = go.Figure()
            
            for district in selected_districts:
                district_data = district_ts[district_ts['Primary_District'] == district]
                if not district_data.empty:
                    fig.add_trace(go.Scatter(
                        x=district_data['Scraped_date'],
                        y=district_data['Үнэ'],
                        mode='lines+markers',
                        name=district
                    ))
            
            fig.update_layout(
                title="Price Trends by District",
                xaxis_title="Date",
                yaxis_title="Average Price (₮)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add district forecast
            st.markdown("#### District Price Forecast")
            
            # Select a district to forecast
            forecast_district = st.selectbox(
                "Select district to forecast:",
                selected_districts
            )
            
            forecast_days = st.slider(
                "Number of days to forecast:",
                7, 30, 14,
                key="district_forecast_days"
            )
            
            # Get data for selected district
            district_data = district_ts[district_ts['Primary_District'] == forecast_district].sort_values('Scraped_date')
            
            if len(district_data) >= 5:  # Need enough data points
                # Simple linear trend model
                district_data = district_data.sort_values('Scraped_date')
                district_data['Days'] = range(len(district_data))
                
                X = district_data['Days'].values.reshape(-1, 1)
                y = district_data['Үнэ'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Create forecast dates and features
                max_date = district_data['Scraped_date'].max()
                future_dates = [pd.to_datetime(max_date) + pd.Timedelta(days=i+1) for i in range(forecast_days)]
                future_X = np.array(range(len(district_data), len(district_data) + forecast_days)).reshape(-1, 1)
                
                # Make predictions
                future_y = model.predict(future_X)
                
                # Create forecast DataFrame
                forecast_result = pd.DataFrame({
                    'Scraped_date': future_dates,
                    'Forecast': future_y,
                    'Type': 'Forecast'
                })
                
                # Plot results
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=district_data['Scraped_date'],
                    y=district_data['Үнэ'],
                    mode='lines+markers',
                    name='Historical Prices',
                    line=dict(color='#2563EB', width=2)
                ))
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=forecast_result['Scraped_date'],
                    y=forecast_result['Forecast'],
                    mode='lines+markers',
                    name='Price Forecast',
                    line=dict(color='#EF4444', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{forecast_district} District Price Forecast ({forecast_days} days ahead)",
                    xaxis_title="Date",
                    yaxis_title="Average Price (₮)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display growth rate
                current_price = district_data['Үнэ'].iloc[-1]
                final_forecast = forecast_result['Forecast'].iloc[-1]
                price_change = ((final_forecast - current_price) / current_price) * 100
                
                # Display metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        label="Current Average Price",
                        value=f"{current_price:,.0f} ₮"
                    )
                
                with metric_col2:
                    st.metric(
                        label=f"Forecasted Price ({forecast_days} days)",
                        value=f"{final_forecast:,.0f} ₮"
                    )
                
                with metric_col3:
                    st.metric(
                        label="Predicted Change",
                        value=f"{price_change:.1f}%",
                        delta=f"{price_change:.1f}%"
                    )
                
                # District growth rate comparison
                st.markdown("#### District Growth Rate Comparison")
                
                # Calculate growth rates for all selected districts
                growth_rates = []
                
                for district in selected_districts:
                    district_series = district_ts[district_ts['Primary_District'] == district].sort_values('Scraped_date')
                    
                    if len(district_series) >= 5:
                        district_series['Days'] = range(len(district_series))
                        
                        X = district_series['Days'].values.reshape(-1, 1)
                        y = district_series['Үнэ'].values
                        
                        # Fit linear model to estimate trend
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        # Calculate monthly growth rate (30 days)
                        daily_change = model.coef_[0]
                        starting_price = district_series['Үнэ'].iloc[0]
                        monthly_change_rate = (daily_change * 30) / starting_price * 100
                        
                        growth_rates.append({
                            'District': district,
                            'Monthly Growth Rate (%)': monthly_change_rate,
                            'Average Price': district_series['Үнэ'].mean()
                        })
                
                if growth_rates:
                    growth_df = pd.DataFrame(growth_rates)
                    
                    # Sort by growth rate
                    growth_df = growth_df.sort_values('Monthly Growth Rate (%)', ascending=False)
                    
                    # Plot growth rates
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=growth_df['District'],
                        y=growth_df['Monthly Growth Rate (%)'],
                        marker_color=['#10B981' if x > 0 else '#EF4444' for x in growth_df['Monthly Growth Rate (%)']]
                    ))
                    
                    fig.update_layout(
                        title="Estimated Monthly Price Growth Rate by District",
                        xaxis_title="District",
                        yaxis_title="Monthly Growth Rate (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display as a table too
                    st.markdown("##### District Growth Rate Details")
                    
                    # Format the dataframe for display
                    display_df = growth_df.copy()
                    display_df['Average Price'] = display_df['Average Price'].map('{:,.0f} ₮'.format)
                    display_df['Monthly Growth Rate (%)'] = display_df['Monthly Growth Rate (%)'].map('{:+.2f}%'.format)
                    
                    st.dataframe(
                        display_df,
                        column_config={
                            "District": "District",
                            "Monthly Growth Rate (%)": st.column_config.TextColumn("Monthly Growth Rate"),
                            "Average Price": st.column_config.TextColumn("Average Price (₮)")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Investment recommendation based on growth rate
                    st.markdown("##### Investment Insights")
                    
                    top_growing = growth_df.iloc[0]['District']
                    top_growth_rate = growth_df.iloc[0]['Monthly Growth Rate (%)']
                    
                    if top_growth_rate > 0:
                        st.success(f"**{top_growing}** shows the strongest growth potential at {top_growth_rate:.2f}% per month")
                    else:
                        st.warning("No districts showing positive growth in the analysis period")
                
            else:
                st.warning(f"Not enough time series data for {forecast_district} district. Need at least 5 data points.")
        else:
            st.warning("Required data (district information with time series) not available for analysis.")

# Add this function to your main function
# In the main() function, add this line at the end (before the footer):
# add_prediction_section(df)
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6B7280; padding: 1rem;">
        Data scraped from Unegui.mn | Dashboard updated: {datetime.now().strftime("%Y-%m-%d")}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
