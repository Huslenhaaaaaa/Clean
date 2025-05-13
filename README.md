                                           General 
                                             
My repository automatically scrapes all the ad information relating to buying or renting apartments from Unegui.mn
It automatically scrapes and updates the stores information so overtime it will keep gathering more and more data 
which I think would be useful for maybe developing a perdiction model. 

Then based off that data is a Streamlit dashboard that shows off the useful parts of the information for users who 
are looking to move and wish to see thier options. 


Automated Data Collection: Daily scraping of Unegui.mn listings via GitHub Actions
Real-Time Market Insights: Current property prices, trends, and availability
Interactive Filtering: Customize analysis by location, price range, property features
District Comparison: Compare housing options across different districts of Mongolia
Property Feature Analysis: Explore the impact of amenities on pricing
Market Trend Visualization: Track price movements and listing volumes over time
Price Prediction: Machine learning model to forecast property price trends (coming soon  before friday ideally)


Repository Structure
├── .github/workflows/
│   └── scrape_rental.yml    # GitHub Actions workflow that triggers the daily scraping
├── unegui_data/             # Storage for scraped real estate data
├── Analysis.ipynb           # Data analysis and visualization exploration
├── Model.ipynb             
├── Scraper.py               # Scraping script
├── apartment_price_prediction_model.pkl 
├── app.py                   # Streamlit dashboard application
└── requirements.txt        
