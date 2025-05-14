import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import os
import pickle
import xgboost as xgb
from datetime import timedelta
import os.path
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
    tab1, tab2, tab3, tab4, tab5,  = st.tabs(["Market Overview", "Price Analysis", "Location Insights", "Property Features", "Browse Listings", "Price Predictions"])
    
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
    if 'Primary_District' in df.columns and 'Rooms' in df.columns and 'Үнэ' in df.columns:
        st.markdown("#### Price Heatmap by District and Room Count")
        
    
    # Group by district and room count, calculate average price
    heatmap_data = df.groupby(['Primary_District', 'Rooms'])['Үнэ'].mean().reset_index()
    
    # Filter to keep only districts with sufficient data
    district_counts = df['Primary_District'].value_counts()
    valid_districts = district_counts[district_counts > 5].index
    heatmap_data = heatmap_data[heatmap_data['Primary_District'].isin(valid_districts)]
    
    # Filter to common room counts
    heatmap_data = heatmap_data[heatmap_data['Rooms'].between(1, 6)]
    
    # Create pivot table for heatmap
    pivot_data = heatmap_data.pivot(index='Primary_District', columns='Rooms', values='Үнэ')
    
    # Create heatmap
    fig_heatmap = px.imshow(
        pivot_data,
        labels=dict(x="Number of Rooms", y="District", color="Average Price (₮)"),
        color_continuous_scale="Viridis",
        text_auto='.0f',  # Show the values on the heatmap cells
        aspect="auto"
    )
    
    fig_heatmap.update_layout(
        title="Average Price by District and Room Count",
        xaxis_title="Number of Rooms",
        yaxis_title="District",
        coloraxis_colorbar=dict(title="Avg Price (₮)")
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
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
        
    with tab5:
        st.markdown('<div class="sub-header">Browse All Listings</div>', unsafe_allow_html=True)
        browse_cols = st.columns([2, 1, 1])
        

    
    with browse_cols[0]:
        search_term = st.text_input("Search by title or description", "")
    
    with browse_cols[1]:
        sort_by = st.selectbox(
            "Sort by",
            ["Newest First", "Price: Low to High", "Price: High to Low", "Area: Largest First"]
        )
    
    with browse_cols[2]:
        records_per_page = st.selectbox("Items per page", [10, 20, 50, 100], index=1)
    
    # Create a copy of the filtered dataframe for display
    browse_df = df.copy()
    
    # Apply search if provided
    if search_term:
        # Create a combined search field from multiple columns
        browse_df['search_text'] = ''
        
        # Add title/description fields to search text if they exist
        if 'Гарчиг' in browse_df.columns:
            browse_df['search_text'] += browse_df['Гарчиг'].fillna('').astype(str) + ' '
        if 'Тайлбар' in browse_df.columns:
            browse_df['search_text'] += browse_df['Тайлбар'].fillna('').astype(str) + ' '
        if 'Байршил' in browse_df.columns:
            browse_df['search_text'] += browse_df['Байршил'].fillna('').astype(str)
        
        # Filter based on search term
        browse_df = browse_df[browse_df['search_text'].str.contains(search_term, case=False, na=False)]
    
    # Apply sorting
    if sort_by == "Newest First" and 'Fixed Posted Date' in browse_df.columns:
        browse_df = browse_df.sort_values('Fixed Posted Date', ascending=False)
    elif sort_by == "Price: Low to High":
        browse_df = browse_df.sort_values('Үнэ', ascending=True)
    elif sort_by == "Price: High to Low":
        browse_df = browse_df.sort_values('Үнэ', ascending=False)
    elif sort_by == "Area: Largest First" and 'Area_m2' in browse_df.columns:
        browse_df = browse_df.sort_values('Area_m2', ascending=False)
    
    # Show count of listings after filtering
    st.info(f"Found {len(browse_df)} listings matching your criteria")
    
    # Pagination
    if len(browse_df) > 0:
        total_pages = max(1, len(browse_df) // records_per_page + (1 if len(browse_df) % records_per_page > 0 else 0))
        page_col1, page_col2 = st.columns([6, 1])
        
        with page_col2:
            current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        with page_col1:
            st.markdown(f"Page {current_page} of {total_pages}")
        
        # Calculate slice indices for pagination
        start_idx = (current_page - 1) * records_per_page
        end_idx = min(start_idx + records_per_page, len(browse_df))
        
        # Get page of data
        page_df = browse_df.iloc[start_idx:end_idx].copy()
        
        # Prepare data for display
        display_df = pd.DataFrame()
        
        # Add columns to display
        if 'Гарчиг' in page_df.columns:
            display_df['Title'] = page_df['Гарчиг']
        
        if 'Үнэ' in page_df.columns:
            display_df['Price (₮)'] = page_df['Үнэ'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        
        if 'Байршил' in page_df.columns:
            display_df['Location'] = page_df['Байршил']
        
        if 'ӨрөөнийТоо' in page_df.columns:
            display_df['Rooms'] = page_df['ӨрөөнийТоо']
        
        if 'Area_m2' in page_df.columns:
            display_df['Area (m²)'] = page_df['Area_m2'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        
        if 'Тайлбар' in page_df.columns:
            # Truncate description to avoid very wide columns
            display_df['Description'] = page_df['Тайлбар'].apply(
                lambda x: str(x)[:100] + '...' if isinstance(x, str) and len(str(x)) > 100 else x
            )
        
        if 'Fixed Posted Date' in page_df.columns:
            display_df['Posted Date'] = page_df['Fixed Posted Date'].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else "N/A"
            )
        
        # Display as a table
        st.dataframe(display_df, use_container_width=True)
        
        # Add clickable links if available
        if 'Link' in page_df.columns or 'URL' in page_df.columns or 'url' in page_df.columns:
            st.markdown("### Listing Details")
            
            # Determine which column has the URL
            url_col = None
            for col in ['Link', 'URL', 'url', 'link']:
                if col in page_df.columns:
                    url_col = col
                    break
            
            # Display individual listing cards with links
            for i, row in page_df.iterrows():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    title = row['Гарчиг'] if 'Гарчиг' in page_df.columns else f"Listing #{i}"
                    st.markdown(f"**{title}**")
                    
                    if 'Байршил' in page_df.columns:
                        st.markdown(f"📍 {row['Байршил']}")
                    
                    if 'Тайлбар' in page_df.columns and pd.notna(row['Тайлбар']):
                        st.markdown(f"{row['Тайлбар']}")
                
                with col2:
                    if 'Үнэ' in page_df.columns:
                        st.markdown(f"**Price:** {row['Үнэ']:,.0f} ₮")
                    
                    if 'Area_m2' in page_df.columns and pd.notna(row['Area_m2']):
                        st.markdown(f"**Area:** {row['Area_m2']:.1f} m²")
                    
                    if 'ӨрөөнийТоо' in page_df.columns:
                        st.markdown(f"**Rooms:** {row['ӨрөөнийТоо']}")
                    
                    if url_col and pd.notna(row[url_col]):
                        st.markdown(f"[View Original Listing]({row[url_col]})")
                
                st.markdown("---")
    else:
        st.warning("No listings found matching your criteria. Try adjusting your filters.")
        

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6B7280; padding: 1rem;">
        Data scraped from Unegui.mn | Dashboard updated: {datetime.now().strftime("%Y-%m-%d")}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
