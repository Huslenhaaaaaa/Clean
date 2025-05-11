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
import re
from pathlib import Path
from urllib.parse import urljoin
import json
import shutil

class UneguiScraper:
    """
    Unified scraper for Unegui.mn apartment listings.
    Handles both sales and rental listings with a single CSV output per scraper type.
    """
    
    def __init__(self, base_url, listing_type, max_pages=90, max_workers=5):
        """
        Initialize the scraper with configuration parameters
        
        Args:
            base_url: The base URL to scrape
            listing_type: Type of listing ("sale" or "rental")
            max_pages: Maximum number of pages to scrape
            max_workers: Maximum number of concurrent workers
        """
        self.base_url = base_url
        self.listing_type = listing_type
        self.max_pages = max_pages
        self.max_workers = max_workers
        
        # Constants
        self.BASE_DELAY = 1.5  # Base delay between requests
        self.JITTER = 0.5  # Random jitter to add to delays
        self.MAX_RETRIES = 3  # Maximum number of retries for failed requests
        
        # Set up directories
        self.data_dir = Path("unegui_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Files based on listing type
        self.listing_dir = self.data_dir / listing_type
        self.listing_dir.mkdir(exist_ok=True)
        
        self.csv_file = self.data_dir / f"unegui_{listing_type}_data.csv"
        self.cache_file = self.listing_dir / f"scraped_{listing_type}_urls.json"
        self.log_file = self.listing_dir / f"{listing_type}_scraper_{date.today().strftime('%Y%m%d')}.log"
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Set up session
        self.session = self._create_session()
        
        # Load previously scraped URLs
        self.scraped_urls = self._load_scraped_urls()
        
    def _setup_logging(self):
        """Set up logging configuration"""
        logger = logging.getLogger(f"unegui_{self.listing_type}")
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _create_session(self):
        """Create and configure requests session with appropriate headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive',
        })
        return session
    
    def _load_scraped_urls(self):
        """Load previously scraped URLs from cache file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.warning(f"Error loading {self.cache_file}, creating new cache")
                return {}
        return {}
    
    def _save_scraped_url(self, url, data=None):
        """Save URL to cache dictionary after successful scraping"""
        self.scraped_urls[url] = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
    
    def _save_cache(self):
        """Save the URL cache to disk"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_urls, f)
        self.logger.info(f"Cache saved with {len(self.scraped_urls)} URLs")
    
    def _make_request(self, url, retry_count=0):
        """Make an HTTP request with retry logic"""
        try:
            # Add randomized delay to be respectful to the server
            time.sleep(self.BASE_DELAY + random.random() * self.JITTER)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except (requests.RequestException, requests.Timeout) as e:
            if retry_count < self.MAX_RETRIES:
                backoff_time = (2 ** retry_count) + random.random()
                self.logger.warning(f"Request failed for {url}: {str(e)}. Retrying in {backoff_time:.2f} seconds...")
                time.sleep(backoff_time)
                return self._make_request(url, retry_count + 1)
            else:
                self.logger.error(f"Failed to retrieve {url} after {self.MAX_RETRIES} attempts: {str(e)}")
                return None
    
    def _generate_ad_id(self, url):
        """Generate a unique ID for each ad based on URL"""
        # Try to extract numeric ID from URL first
        match = re.search(r'(\d+)/?$', url)
        if match:
            return match.group(1)
        # Fallback to hash
        return hashlib.md5(url.encode()).hexdigest()
    
    def _scrape_page(self, page_num):
        """Scrape all ad links from a single page"""
        url = self.base_url if page_num == 1 else f"{self.base_url}?page={page_num}"
        self.logger.info(f"Scraping page {page_num}...")
        
        response = self._make_request(url)
        if not response:
            return []
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all ad links based on the class "mask"
            ad_links = soup.find_all('a', class_='mask')
            
            # Extract the href attributes (the links)
            links = [urljoin("https://www.unegui.mn", link['href']) 
                    for link in ad_links if 'href' in link.attrs]
            
            self.logger.info(f"Found {len(links)} links on page {page_num}")
            return links
        except Exception as e:
            self.logger.error(f"Error parsing page {url}: {str(e)}")
            return []
    
    def _extract_address(self, soup):
        """Extract address information"""
        address = soup.find('span', itemprop="address")
        district, location = 'N/A', 'N/A'
        
        if address and '—' in address.text:
            parts = address.text.split('—')
            district = parts[0].strip()
            location = parts[1].strip()
        
        return district, location
    
    def _extract_price(self, soup):
        """Extract price information"""
        price_meta = soup.find('meta', {'itemprop': 'price'})
        if price_meta:
            price = price_meta.get('content', 'N/A')
            # Convert to integer if possible (remove .00)
            try:
                price = str(int(float(price)))
            except (ValueError, TypeError):
                pass
            return price
        return 'N/A'
    
    def _get_element_text(self, soup, element_type, **attrs):
        """Generic method to extract text from elements"""
        element = soup.find(element_type, attrs)
        return element.text.strip() if element else 'N/A'
    
    def _get_value_chars(self, soup, key):
        """Extract value from elements with class='value-chars'"""
        element = soup.find('span', string=lambda x: x and key in x)
        if element:
            value_chars = element.find_next('a', class_='value-chars')
            if value_chars:
                return value_chars.text.strip()
        return 'N/A'
    
    def _get_text_value(self, soup, key):
        """Extract value from next span after key"""
        element = soup.find('span', string=lambda x: x and key in x)
        if element:
            next_span = element.find_next('span')
            if next_span and not next_span.find('a', class_='value-chars'):
                return next_span.text.strip()
        return 'N/A'
    
    def _scrape_ad(self, url):
        """Scrape detailed information from a single ad page"""
        # Check if URL has already been scraped
        if url in self.scraped_urls:
            cache_entry = self.scraped_urls[url]
            if "data" in cache_entry and cache_entry["data"]:
                self.logger.debug(f"Using cached data for: {url}")
                return cache_entry["data"]
            else:
                self.logger.debug(f"Skipping previously failed URL: {url}")
                return None
        
        response = self._make_request(url)
        if not response:
            self._save_scraped_url(url)  # Mark as attempted but failed
            return None
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Initialize ad data with basic structure
            district, location = self._extract_address(soup)
            
            ad_data = {
                'ad_id': self._generate_ad_id(url),
                'link': url,
                'listing_type': self.listing_type,
                'Scraped_date': date.today().strftime("%d/%m/%Y"),
                'Дүүрэг': district,
                'Байршил': location,
                'Үнэ': self._extract_price(soup),
            }
            
            # Extract common property values
            property_fields = {
                'Шал': 'Шал:',
                'Тагт': 'Тагт:',
                'Гараж': 'Гараж:',
                'Цонх': 'Цонх:',
                'Хаалга': 'Хаалга:',
                'Цонхнытоо': 'Цонхны тоо:',
                'Барилгынявц': 'Барилгын явц',
                'Ашиглалтандорсонон': 'Ашиглалтанд орсон он:',
                'Барилгындавхар': 'Барилгын давхар:',
                'Талбай': 'Талбай:',
                'Хэдэндавхарт': 'Хэдэн давхарт:',
                'Лизингээравахболомж': 'Лизингээр авах боломж:'
            }
            
            # Process text values
            for field, key in property_fields.items():
                if "тоо" in key or "давхар" in key or "Талбай" in key:
                    ad_data[field] = self._get_value_chars(soup, key)
                else:
                    ad_data[field] = self._get_text_value(soup, key)
            
            # Extract views count
            views_element = soup.find('span', class_='counter-views')
            ad_data['Үзсэн'] = views_element.text.strip().replace(' ', '') if views_element else 'N/A'
            
            # Extract room count
            location_div = soup.find('div', class_='wrap js-single-item__location')
            if location_div and location_div.find_all('span'):
                ad_data['ӨрөөнийТоо'] = location_div.find_all('span')[-1].text.strip()
            else:
                ad_data['ӨрөөнийТоо'] = 'N/A'
            
            # Extract title
            ad_data['Зарыг гарчиг'] = self._get_element_text(soup, 'h1', class_='title-announcement')
            
            # Extract description
            desc_element = soup.find('div', class_='announcement-description')
            if desc_element:
                # Clean up description text
                text = desc_element.text.strip()
                # Replace multiple newlines with a single space
                text = re.sub(r'\n+', ' ', text)
                # Replace multiple spaces with a single space
                text = re.sub(r'\s+', ' ', text)
                ad_data['Зарын тайлбар'] = text
            else:
                ad_data['Зарын тайлбар'] = 'N/A'
            
            # Extract posted date
            date_element = soup.find('span', class_='date-meta')
            if date_element:
                ad_data['Нийтэлсэн'] = date_element.text.strip().replace('Нийтэлсэн: ', '')
            else:
                ad_data['Нийтэлсэн'] = 'N/A'
            
            # Save to cache
            self._save_scraped_url(url, ad_data)
            
            return ad_data
        except Exception as e:
            self.logger.error(f"Error scraping ad {url}: {str(e)}", exc_info=True)
            self._save_scraped_url(url)  # Mark as attempted but failed
            return None
    
    def _scrape_ads_parallel(self, links):
        """Scrape multiple ads in parallel"""
        results = []
        
        # Use ThreadPoolExecutor for parallel scraping
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all scraping tasks
            future_to_url = {executor.submit(self._scrape_ad, url): url for url in links}
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_url), 1):
                url = future_to_url[future]
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                    
                    # Log progress
                    if i % 10 == 0 or i == len(links):
                        self.logger.info(f"Progress: {i}/{len(links)} ads processed")
                
                except Exception as e:
                    self.logger.error(f"Error processing {url}: {str(e)}")
        
        return results
    
    def _load_existing_data(self):
        """Load existing data from the CSV file if available"""
        try:
            if self.csv_file.exists():
                df = pd.read_csv(self.csv_file, encoding='utf-8-sig')
                self.logger.info(f"Loaded existing data from {self.csv_file}: {len(df)} records")
                return df
            else:
                self.logger.info(f"No existing data file found at {self.csv_file}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.warning(f"Could not load existing data: {str(e)}")
            return pd.DataFrame()
    
    def _save_data(self, all_data):
        """Save data to a single CSV file with temporary backup"""
        if not all_data:
            self.logger.warning("No data to save.")
            return
            
        df = pd.DataFrame(all_data)
        
        # Handle duplicates by keeping newest version
        if 'ad_id' in df.columns:
            df = df.drop_duplicates(subset=['ad_id'], keep='last')
        
        backup_path = self.csv_file.with_suffix('.csv.bak')
        
        # If main file exists, create a backup before writing new data
        if self.csv_file.exists():
            try:
                shutil.copy2(self.csv_file, backup_path)
                self.logger.info(f"Created backup of existing data file at {backup_path}")
            except Exception as e:
                self.logger.error(f"Failed to create backup: {str(e)}")
        
        # Save the updated data to the main file
        try:
            df.to_csv(self.csv_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"Data saved to {self.csv_file}: {len(df)} records")
        except Exception as e:
            self.logger.error(f"Failed to save data: {str(e)}")
            
            # If saving failed and we have a backup, try to restore it
            if backup_path.exists():
                try:
                    shutil.copy2(backup_path, self.csv_file)
                    self.logger.info("Restored backup after save failure")
                except Exception as restore_err:
                    self.logger.error(f"Failed to restore backup: {str(restore_err)}")
    
    def run(self):
        """Main scraping process"""
        start_time = datetime.now()
        self.logger.info(f"Starting {self.listing_type} apartment scraper at {start_time}")
        
        all_data = []
        existing_data = self._load_existing_data()
        
        # Start with existing data if available
        if not existing_data.empty:
            all_data = existing_data.to_dict('records')
            self.logger.info(f"Loaded {len(all_data)} existing records")
        
        # Create a set of existing ad_ids for quick lookup
        existing_ids = {item['ad_id'] for item in all_data} if all_data else set()
        
        # Scrape links from all pages
        all_links = []
        
        # Use ThreadPoolExecutor for parallel page scraping
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_page = {executor.submit(self._scrape_page, page_num): page_num 
                             for page_num in range(1, self.max_pages + 1)}
            
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    links = future.result()
                    if links:
                        all_links.extend(links)
                except Exception as e:
                    self.logger.error(f"Error scraping page {page_num}: {str(e)}")
        
        # Remove duplicates and already scraped urls that we have data for
        unique_links = []
        for link in all_links:
            if link in self.scraped_urls:
                cache_entry = self.scraped_urls[link]
                # Only add if we don't have data for this URL
                if not cache_entry.get("data"):
                    unique_links.append(link)
            else:
                unique_links.append(link)
                
        self.logger.info(f"Found {len(unique_links)} new links to scrape out of {len(all_links)} total links")
        
        # Track if we've made any updates to data
        updates_made = False
        
        # Process in smaller batches for better memory management
        batch_size = 50
        for i in range(0, len(unique_links), batch_size):
            batch_links = unique_links[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(unique_links)-1)//batch_size + 1} ({len(batch_links)} links)")
            
            # Scrape batch of ads in parallel
            batch_data = self._scrape_ads_parallel(batch_links)
            
            # Filter out ads that already exist in our dataset
            new_batch_data = [item for item in batch_data if item and item['ad_id'] not in existing_ids]
            
            # Update the existing_ids set with new ad_ids
            for item in new_batch_data:
                existing_ids.add(item['ad_id'])
            
            # Add new data to all_data
            if new_batch_data:
                all_data.extend(new_batch_data)
                updates_made = True
                
                # Save after each batch if we have new data
                self._save_data(all_data)
                self._save_cache()  # Save the URL cache periodically
                self.logger.info(f"Saved batch data. Total records: {len(all_data)}")
        
        # If no updates were made, log it but don't create a new file
        if not updates_made:
            self.logger.info("No new data was found. Main CSV file remains unchanged.")
        
        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info(f"Scraping completed in {duration}. Total ads collected: {len(all_data)}")
        
        return all_data


def main():
    """Main entry point for both scraper types"""
    try:
        # Define scraper configurations
        scrapers = [
            {
                "name": "sales",
                "base_url": "https://www.unegui.mn/l-hdlh/l-hdlh-zarna/oron-suuts-zarna/",
                "max_pages": 200
            },
            {
                "name": "rental",
                "base_url": "https://www.unegui.mn/l-hdlh/l-hdlh-treesllne/oron-suuts/",
                "max_pages": 90
            }
        ]
        
        # Run each scraper
        for config in scrapers:
            print(f"\n{'='*50}")
            print(f"Starting {config['name']} scraper")
            print(f"{'='*50}\n")
            
            scraper = UneguiScraper(
                base_url=config["base_url"],
                listing_type=config["name"],
                max_pages=config["max_pages"]
            )
            scraper.run()
            
            print(f"\n{'='*50}")
            print(f"Completed {config['name']} scraper")
            print(f"{'='*50}\n")
        
    except Exception as e:
        logging.critical(f"Critical error in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
