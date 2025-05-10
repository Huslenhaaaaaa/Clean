import requests
from bs4 import BeautifulSoup
from datetime import date, datetime
import time
import pandas as pd
import os
import logging
import random
import hashlib
import concurrent.futures
import json
import re
from urllib.parse import urljoin

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"scraper_{date.today().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
BASE_DELAY = 0.5  # Reduced base delay between requests
JITTER = 0.5  # Random jitter to add to delays
CACHE_FILE = "scraped_urls.json"  # Use JSON for more efficient storage
MAX_WORKERS = 10  # Number of concurrent workers
BATCH_SIZE = 20  # Save data after processing this many items

class UneguiScraper:
    def __init__(self, base_url, max_pages=90):
        self.base_url = base_url
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        })
        self.scraped_urls = self.load_scraped_urls()
        self.ad_data_cache = {}  # Cache for storing ad data
        
    def load_scraped_urls(self):
        """Load previously scraped URLs from cache file"""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error loading {CACHE_FILE}, creating new cache")
                return {}
        return {}
        
    def save_scraped_url(self, url, data=None):
        """Save URL to cache file after successful scraping"""
        self.scraped_urls[url] = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        # Don't write to disk on every URL, will save periodically instead
    
    def save_cache(self):
        """Save the URL cache to disk"""
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_urls, f)
        logger.info(f"Cache saved with {len(self.scraped_urls)} URLs")
    
    def make_request(self, url, retry_count=0):
        """Make an HTTP request with retry logic"""
        try:
            # Add randomized delay to be respectful to the server
            # Only add delay if it's not the first attempt
            if retry_count > 0:
                time.sleep(BASE_DELAY + random.random() * JITTER)
            
            response = self.session.get(url, timeout=15)  # Reduced timeout
            response.raise_for_status()
            return response
        except (requests.RequestException, requests.Timeout) as e:
            if retry_count < MAX_RETRIES:
                backoff_time = (2 ** retry_count) + random.random()
                logger.warning(f"Request failed for {url}: {str(e)}. Retrying in {backoff_time:.2f} seconds...")
                time.sleep(backoff_time)
                return self.make_request(url, retry_count + 1)
            else:
                logger.error(f"Failed to retrieve {url} after {MAX_RETRIES} attempts: {str(e)}")
                return None

    def scrape_page(self, page_num):
        """Scrape all ad links from a single page"""
        url = self.base_url if page_num == 1 else f"{self.base_url}?page={page_num}"
        
        response = self.make_request(url)
        if not response:
            return []
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all ad links based on the class "mask"
            ad_links = soup.find_all('a', class_='mask')
            
            # Extract the href attributes (the links)
            links = [urljoin("https://www.unegui.mn", link['href']) 
                    for link in ad_links if 'href' in link.attrs]
            
            logger.info(f"Found {len(links)} links on page {url}")
            return links
        except Exception as e:
            logger.error(f"Error parsing page {url}: {str(e)}")
            return []

    def extract_data_from_soup(self, soup, url):
        """Extract ad data from BeautifulSoup object"""
        ad_data = {
            'Шал': 'N/A',
            'Тагт': 'N/A',
            'Гараж': 'N/A',
            'Цонх': 'N/A',
            'Хаалга': 'N/A',
            'Цонхнытоо': 'N/A',
            'Барилгынявц': 'N/A',
            'Ашиглалтандорсонон': 'N/A',
            'Барилгындавхар': 'N/A',
            'Талбай': 'N/A',
            'Хэдэндавхарт': 'N/A',
            'Лизингээравахболомж': 'N/A',
            'Дүүрэг': 'N/A',
            'Байршил': 'N/A',
            'Үзсэн': 'N/A',
            'Scraped_date': date.today().strftime("%d/%m/%Y"),
            'link': url,
            'Үнэ': 'N/A',
            'ӨрөөнийТоо': 'N/A',
            'Зарыг гарчиг': 'N/A',
            'Зарын тайлбар': 'N/A',
            'Нийтэлсэн': 'N/A',
            'ad_id': self.generate_ad_id(url)
        }
        
        # Optimize data extraction with CSS selectors where possible
        features = soup.find_all('span', class_=['key', 'key-chars'])
        for feature in features:
            key = feature.text.strip().replace(':', '')
            value_element = feature.find_next('span') or feature.find_next('a', class_='value-chars')
            if value_element:
                value = value_element.text.strip()
                # Map key to ad_data key
                if key == 'Шал':
                    ad_data['Шал'] = value
                elif key == 'Тагт':
                    ad_data['Тагт'] = value
                elif key == 'Гараж':
                    ad_data['Гараж'] = value
                elif key == 'Цонх':
                    ad_data['Цонх'] = value
                elif key == 'Хаалга':
                    ad_data['Хаалга'] = value
                elif key == 'Цонхны тоо':
                    ad_data['Цонхнытоо'] = value
                elif key == 'Барилгын явц':
                    ad_data['Барилгынявц'] = value
                elif key == 'Ашиглалтанд орсон он':
                    ad_data['Ашиглалтандорсонон'] = value
                elif key == 'Барилгын давхар':
                    ad_data['Барилгындавхар'] = value
                elif key == 'Талбай':
                    ad_data['Талбай'] = value
                elif key == 'Хэдэн давхарт':
                    ad_data['Хэдэндавхарт'] = value
                elif key == 'Лизингээр авах боломж':
                    ad_data['Лизингээравахболомж'] = value
        
        # Handle address
        try:
            address = soup.find('span', itemprop="address")
            if address and '—' in address.text:
                parts = address.text.split('—')
                ad_data['Дүүрэг'] = parts[0].strip()
                ad_data['Байршил'] = parts[1].strip()
        except Exception as e:
            logger.warning(f"Error extracting address from {url}: {str(e)}")
        
        # Extract views count
        views_element = soup.find('span', class_='counter-views')
        if views_element:
            ad_data['Үзсэн'] = views_element.text.strip().replace(' ', '')
        
        # Extract price from meta tag
        price_meta = soup.find('meta', {'itemprop': 'price'})
        if price_meta:
            price = price_meta.get('content', 'N/A')
            # Convert to integer if possible (remove .00)
            try:
                price = str(int(float(price)))
            except:
                pass
            ad_data['Үнэ'] = price
        
        # Extract room count
        location_div = soup.find('div', class_='wrap js-single-item__location')
        if location_div and location_div.find_all('span'):
            ad_data['ӨрөөнийТоо'] = location_div.find_all('span')[-1].text.strip()
        
        # Extract title
        title_element = soup.find('h1', class_='title-announcement')
        if title_element:
            ad_data['Зарыг гарчиг'] = title_element.text.strip().replace('\n', ' ')
        
        # Extract description
        desc_element = soup.find('div', class_='announcement-description')
        if desc_element:
            ad_data['Зарын тайлбар'] = desc_element.text.strip().replace('\n', ' ')
        
        # Extract posted date
        date_element = soup.find('span', class_='date-meta')
        if date_element:
            ad_data['Нийтэлсэн'] = date_element.text.strip().replace('Нийтэлсэн: ', '')
        
        return ad_data

    def scrape_ad(self, url):
        """Scrape detailed information from a single ad page"""
        # Check if URL has already been scraped
        if url in self.scraped_urls:
            logger.info(f"Using cached data for: {url}")
            # If we have the data in cache, use it
            if "data" in self.scraped_urls[url] and self.scraped_urls[url]["data"]:
                return self.scraped_urls[url]["data"]
            # Otherwise skip it
            return None
        
        response = self.make_request(url)
        if not response:
            return None
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            ad_data = self.extract_data_from_soup(soup, url)
            
            # Cache the data
            self.save_scraped_url(url, ad_data)
            
            return ad_data
        except Exception as e:
            logger.error(f"Error scraping ad {url}: {str(e)}", exc_info=True)
            # Mark as attempted but failed
            self.save_scraped_url(url)
            return None
    
    def generate_ad_id(self, url):
        """Generate a unique ID for each ad based on URL"""
        # Extract numeric ID from URL if possible
        match = re.search(r'(\d+)/?$', url)
        if match:
            return match.group(1)
        # Fallback to hash
        return hashlib.md5(url.encode()).hexdigest()
    
    def scrape_pages(self):
        """Scrape links from all pages in parallel"""
        all_links = []
        
        # First get a list of page URLs
        page_urls = []
        for page_num in range(1, self.max_pages + 1):
            page_urls.append(page_num)
        
        # Use thread pool to scrape pages in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(page_urls))) as executor:
            # Submit all page scraping tasks
            future_to_page = {executor.submit(self.scrape_page, page_num): page_num for page_num in page_urls}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    links = future.result()
                    if links:
                        all_links.extend(links)
                        logger.info(f"Collected {len(links)} links from page {page_num}")
                    else:
                        logger.warning(f"No links found on page {page_num}")
                except Exception as e:
                    logger.error(f"Error scraping page {page_num}: {str(e)}")
        
        # Deduplicate links
        all_links = list(set(all_links))
        logger.info(f"Total unique links found: {len(all_links)}")
        return all_links
    
    def scrape_ads_batch(self, links_batch):
        """Scrape a batch of ad links"""
        results = []
        
        # Use thread pool to scrape ads in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all ad scraping tasks
            future_to_url = {executor.submit(self.scrape_ad, url): url for url in links_batch}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    ad_data = future.result()
                    if ad_data:
                        results.append(ad_data)
                except Exception as e:
                    logger.error(f"Error processing result for {url}: {str(e)}")
        
        return results
    
    def run(self):
        """Main scraping process"""
        # Load existing master data
        existing_data = self.load_existing_data()
        
        # Start with existing data if available
        if not existing_data.empty:
            # Use a dictionary to index by ad_id for faster lookups and merging
            existing_data_dict = {row['ad_id']: row for row in existing_data.to_dict('records')}
            logger.info(f"Loaded {len(existing_data_dict)} existing records from master file")
        else:
            existing_data_dict = {}
            logger.info("No existing data found. Starting fresh collection.")
        
        # Get all links
        all_links = self.scrape_pages()
        
        if not all_links:
            logger.warning("No links found to scrape.")
            return
        
        logger.info(f"Starting to scrape {len(all_links)} individual ads...")
        
        # Track new ads found in this run
        new_ads_count = 0
        today_data = []
        
        # Process in batches for better performance and periodic saving
        for i in range(0, len(all_links), BATCH_SIZE):
            batch = all_links[i:i+BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(all_links)+BATCH_SIZE-1)//BATCH_SIZE}: {len(batch)} ads")
            
            batch_results = self.scrape_ads_batch(batch)
            
            # Process results: add to today's data and update master dataset
            for ad in batch_results:
                today_data.append(ad)
                
                # Add to or update the master dataset
                if ad['ad_id'] not in existing_data_dict:
                    existing_data_dict[ad['ad_id']] = ad
                    new_ads_count += 1
            
            # Save periodically
            self.save_cache()
            logger.info(f"Progress: {i+len(batch)}/{len(all_links)} ads processed. Found {new_ads_count} new ads so far.")
        
        # Save final results
        if today_data or existing_data_dict:
            # Convert dictionary back to list for saving
            all_data = list(existing_data_dict.values())
            self.save_data(all_data)
            
            # Save today's data separately
            if today_data:
                df_today = pd.DataFrame(today_data)
                today_file = f"unegui_data_today_{date.today().strftime('%Y%m%d')}.csv"
                df_today.to_csv(today_file, index=False, encoding='utf-8-sig')
                logger.info(f"Today's data saved to {today_file}: {len(today_data)} records")
            
            logger.info(f"Scraping completed. Total ads in master file: {len(all_data)}")
            logger.info(f"New ads found in this run: {new_ads_count}")
        else:
            logger.warning("No data was collected.")
    
    def load_existing_data(self):
        """Load existing data from master CSV file if available"""
        master_file = "unegui_data_master.csv"
        try:
            if os.path.exists(master_file):
                df = pd.read_csv(master_file, encoding='utf-8-sig')
                logger.info(f"Loaded existing data from {master_file}: {len(df)} records")
                return df
            else:
                # Fallback to most recent daily file if master doesn't exist
                files = [f for f in os.listdir('.') if f.startswith('unegui_data_') and f.endswith('.csv') 
                        and not f == master_file]
                if not files:
                    return pd.DataFrame()
                
                latest_file = max(files)
                df = pd.read_csv(latest_file, encoding='utf-8-sig')
                logger.info(f"Loaded existing data from {latest_file}: {len(df)} records")
                return df
        except Exception as e:
            logger.warning(f"Could not load existing data: {str(e)}")
            return pd.DataFrame()
    
    def save_data(self, all_data):
        """Save data to CSV file"""
        df = pd.DataFrame(all_data)
        # Remove duplicates based on ad_id
        df = df.drop_duplicates(subset=['ad_id'])
        
        # Save to daily file for record keeping
        daily_file = f"unegui_data_{date.today().strftime('%Y%m%d')}.csv"
        df.to_csv(daily_file, index=False, encoding='utf-8-sig')
        
        # Also save to a master file for Streamlit
        master_file = "unegui_data_master.csv"
        df.to_csv(master_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"Data saved to {daily_file} and {master_file}: {len(df)} records")

def main():
    """Main entry point"""
    try:
        # Configuration
        base_url = "https://www.unegui.mn/l-hdlh/l-hdlh-treesllne/oron-suuts/"
        max_pages = 90
        
        # Create and run scraper
        start_time = datetime.now()
        logger.info(f"Starting scraper at {start_time}")
        
        scraper = UneguiScraper(base_url, max_pages)
        scraper.run()
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Scraping completed in {duration}")
        
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
