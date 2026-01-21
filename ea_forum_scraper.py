"""
EA Sports FC Forum Scraper
Scrapes posts from the last 7 days for sentiment analysis
"""

import json
import time
import re
import copy
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ForumPost:
    """Data class representing a forum post"""
    title: str
    url: str
    body_preview: str
    author: str
    author_rank: str
    timestamp: datetime
    timestamp_relative: str
    category: str
    views: int
    likes: int
    comments: int
    
    def to_dict(self):
        """Convert to dictionary with ISO format timestamp"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        return d


class EAForumScraper:
    """Scraper for EA Sports FC Forums"""
    
    BASE_URL = "https://forums.ea.com"
    CATEGORY_URL = "https://forums.ea.com/category/ea-sports-fc-en"
    
    def __init__(self, headless: bool = True):
        """
        Initialize the scraper with Selenium WebDriver
        
        Args:
            headless: Run browser in headless mode (no GUI)
        """
        self.headless = headless
        self.driver = None
        self.posts: List[ForumPost] = []
        
    def _setup_driver(self):
        """Configure and return Chrome WebDriver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(10)
        
    def _close_driver(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def _parse_relative_time(self, relative_time: str, title_timestamp: str = None) -> datetime:
        """
        Parse relative time string (e.g., '23 minutes ago') to datetime
        Also tries to parse the title attribute for exact timestamp
        
        Args:
            relative_time: Relative time string like '23 minutes ago'
            title_timestamp: Optional exact timestamp from title attribute
            
        Returns:
            datetime object
        """
        now = datetime.now()
        
        # Try parsing the title timestamp first (e.g., "January 21, 2026 at 8:36 AM")
        if title_timestamp:
            try:
                return date_parser.parse(title_timestamp)
            except (ValueError, TypeError):
                pass
        
        # Parse relative time
        relative_time = relative_time.lower().strip()
        
        # Patterns for relative time
        patterns = [
            (r'(\d+)\s*second', 'seconds'),
            (r'(\d+)\s*minute', 'minutes'),
            (r'(\d+)\s*hour', 'hours'),
            (r'(\d+)\s*day', 'days'),
            (r'(\d+)\s*week', 'weeks'),
            (r'(\d+)\s*month', 'months'),
            (r'(\d+)\s*year', 'years'),
        ]
        
        for pattern, unit in patterns:
            match = re.search(pattern, relative_time)
            if match:
                value = int(match.group(1))
                if unit == 'seconds':
                    return now - timedelta(seconds=value)
                elif unit == 'minutes':
                    return now - timedelta(minutes=value)
                elif unit == 'hours':
                    return now - timedelta(hours=value)
                elif unit == 'days':
                    return now - timedelta(days=value)
                elif unit == 'weeks':
                    return now - timedelta(weeks=value)
                elif unit == 'months':
                    return now - relativedelta(months=value)
                elif unit == 'years':
                    return now - relativedelta(years=value)
        
        # If "just now" or similar
        if 'just now' in relative_time or 'moment' in relative_time:
            return now
            
        # Try parsing as absolute date
        try:
            return date_parser.parse(relative_time)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse time: {relative_time}")
            return now
    
    def _parse_count(self, count_text: str) -> int:
        """Parse count text to integer, handling K/M suffixes"""
        if not count_text:
            return 0
        # Remove common label text and clean up
        count_text = count_text.strip().upper()
        count_text = re.sub(r'[^0-9KM.]', '', count_text)  # Keep only digits, K, M, and decimal
        
        if not count_text:
            return 0
            
        try:
            if 'K' in count_text:
                return int(float(count_text.replace('K', '')) * 1000)
            elif 'M' in count_text:
                return int(float(count_text.replace('M', '')) * 1000000)
            return int(float(count_text))
        except (ValueError, TypeError):
            return 0
    
    def _parse_post_element(self, post_element) -> Optional[ForumPost]:
        """
        Parse a single post element and extract data
        
        Args:
            post_element: BeautifulSoup element for a post
            
        Returns:
            ForumPost object or None if parsing fails
        """
        try:
            # Make a copy to avoid modifying the original when we decompose elements
            post_element = copy.deepcopy(post_element)
            # Extract title and URL
            title_link = post_element.select_one('h4[data-testid="MessageSubject"] a[data-testid="MessageLink"]')
            if not title_link:
                title_link = post_element.select_one('a[data-testid="MessageLink"]')
            
            title = title_link.get('aria-label', '') or title_link.get_text(strip=True) if title_link else ''
            url = self.BASE_URL + title_link.get('href', '') if title_link else ''
            
            # Extract body preview
            body_span = post_element.select_one('span.styles_lia-g-message-body__LkV7_')
            body_preview = body_span.get_text(strip=True) if body_span else ''
            
            # Extract author
            author_link = post_element.select_one('a[data-testid="userLink"]')
            author = author_link.get_text(strip=True) if author_link else ''
            
            # Extract author rank
            rank_span = post_element.select_one('div[data-testid="userRank"] span.UserRankLabel_lia-rank-label__epEUI')
            author_rank = rank_span.get_text(strip=True) if rank_span else ''
            
            # Extract timestamp
            time_element = post_element.select_one('span[data-testid="messageTime"] span')
            timestamp_relative = time_element.get_text(strip=True) if time_element else ''
            timestamp_title = time_element.get('title', '') if time_element else ''
            timestamp = self._parse_relative_time(timestamp_relative, timestamp_title)
            
            # Extract category
            category_link = post_element.select_one('a[data-testid="nodeLink"]')
            category = ''
            if category_link:
                # Get the text that's not in sr-only span
                category_span = category_link.select_one('span[aria-hidden="true"]')
                category = category_span.get_text(strip=True) if category_span else category_link.get_text(strip=True)
            
            # Extract counts (views, likes, comments)
            views = 0
            likes = 0
            comments = 0
            
            # Find the footer section containing metrics
            footer_right = post_element.select_one('div.MessageViewInline_lia-footer-right__mxWEA')
            
            view_count = post_element.select_one('div[data-testid="ViewCount"]')
            if view_count:
                # Get text content, excluding sr-only spans
                for sr_only in view_count.select('span.styles_sr-only__NOnjB'):
                    sr_only.decompose()
                count_text = view_count.get_text(strip=True)
                views = self._parse_count(count_text)
                logger.debug(f"Views raw: '{count_text}' -> {views}")
            
            kudos_count = post_element.select_one('div[data-testid="kudosCount"]')
            if kudos_count:
                for sr_only in kudos_count.select('span.styles_sr-only__NOnjB'):
                    sr_only.decompose()
                count_text = kudos_count.get_text(strip=True)
                likes = self._parse_count(count_text)
                logger.debug(f"Likes raw: '{count_text}' -> {likes}")
            
            replies_count = post_element.select_one('div[data-testid="messageRepliesCount"]')
            if replies_count:
                for sr_only in replies_count.select('span.styles_sr-only__NOnjB'):
                    sr_only.decompose()
                count_text = replies_count.get_text(strip=True)
                comments = self._parse_count(count_text)
                logger.debug(f"Comments raw: '{count_text}' -> {comments}")
            
            return ForumPost(
                title=title,
                url=url,
                body_preview=body_preview,
                author=author,
                author_rank=author_rank,
                timestamp=timestamp,
                timestamp_relative=timestamp_relative,
                category=category,
                views=views,
                likes=likes,
                comments=comments
            )
            
        except Exception as e:
            logger.error(f"Error parsing post element: {e}")
            return None
    
    def _click_show_more(self, max_clicks: int = 50, days: int = 7):
        """
        Click the "Show More" button to load more posts until we have posts spanning the date range
        
        Args:
            max_clicks: Maximum number of button clicks
            days: Number of days to look back - stop when we find posts older than this
        """
        clicks = 0
        cutoff_date = datetime.now() - timedelta(days=days)
        
        while clicks < max_clicks:
            try:
                # Scroll to bottom first to ensure button is visible
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                
                # Find and click the "Show More" button
                show_more_btn = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-testid="PagerLoadMore.Button"]'))
                )
                
                # Scroll the button into view and click
                self.driver.execute_script("arguments[0].scrollIntoView(true);", show_more_btn)
                time.sleep(0.5)
                show_more_btn.click()
                clicks += 1
                
                logger.info(f"Clicked 'Show More' button ({clicks}/{max_clicks})")
                
                # Wait for new content to load
                time.sleep(2)
                
                # Check if we've loaded posts older than our date range
                # Parse the current page to check the oldest post
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                post_elements = soup.select('li.PaneledItemList_lia-panel-list-item__bV87f')
                
                if post_elements:
                    # Check the last post's timestamp
                    last_post = post_elements[-1]
                    time_element = last_post.select_one('span[data-testid="messageTime"] span')
                    if time_element:
                        timestamp_relative = time_element.get_text(strip=True)
                        timestamp_title = time_element.get('title', '')
                        oldest_timestamp = self._parse_relative_time(timestamp_relative, timestamp_title)
                        
                        logger.info(f"Oldest post so far: {timestamp_relative} ({oldest_timestamp})")
                        
                        # If oldest post is older than our cutoff, we have enough
                        if oldest_timestamp < cutoff_date:
                            logger.info(f"Found posts older than {days} days, stopping load")
                            break
                
                # Log progress every 10 clicks
                if clicks % 10 == 0:
                    logger.info(f"Loaded {len(post_elements)} posts so far...")
                    
            except TimeoutException:
                logger.info("No more 'Show More' button found - reached end of posts")
                break
            except Exception as e:
                logger.warning(f"Error clicking 'Show More': {e}")
                # Try scrolling and retry once
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                clicks += 1
                if clicks >= max_clicks:
                    break
        
        logger.info(f"Finished loading after {clicks} clicks")
    
    def _is_within_date_range(self, post_date: datetime, days: int = 7) -> bool:
        """Check if post is within the specified date range"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return post_date >= cutoff_date
    
    def scrape(self, days: int = 7, max_pages: int = 20) -> List[ForumPost]:
        """
        Scrape forum posts from the last N days
        
        Args:
            days: Number of days to look back (default 7)
            max_pages: Maximum number of page scrolls/loads
            
        Returns:
            List of ForumPost objects
        """
        logger.info(f"Starting scrape for posts from the last {days} days...")
        
        try:
            self._setup_driver()
            
            # Navigate to the forum category
            logger.info(f"Navigating to {self.CATEGORY_URL}")
            self.driver.get(self.CATEGORY_URL)
            
            # Wait for posts to load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'li.PaneledItemList_lia-panel-list-item__bV87f'))
            )
            
            # Additional wait for dynamic content (metrics, etc.)
            time.sleep(3)
            
            # Wait for metrics to be visible
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-testid="ViewCount"]'))
                )
            except TimeoutException:
                logger.warning("Metrics elements not found, continuing anyway...")
            
            # Click "Show More" button to load more posts
            logger.info("Loading more posts by clicking 'Show More' button...")
            self._click_show_more(max_clicks=max_pages, days=days)
            
            # Parse the page
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Find all post elements
            post_elements = soup.select('li.PaneledItemList_lia-panel-list-item__bV87f')
            logger.info(f"Found {len(post_elements)} post elements")
            
            # Parse each post
            posts_within_range = []
            posts_outside_range = 0
            
            for element in post_elements:
                post = self._parse_post_element(element)
                if post:
                    if self._is_within_date_range(post.timestamp, days):
                        posts_within_range.append(post)
                    else:
                        posts_outside_range += 1
            
            self.posts = posts_within_range
            logger.info(f"Scraped {len(posts_within_range)} posts within last {days} days")
            logger.info(f"Skipped {posts_outside_range} posts outside date range")
            
            return self.posts
            
        except TimeoutException:
            logger.error("Timeout waiting for page to load")
            return []
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            raise
        finally:
            self._close_driver()
    
    def save_to_json(self, filepath: str):
        """Save scraped posts to JSON file"""
        data = {
            'scraped_at': datetime.now().isoformat(),
            'total_posts': len(self.posts),
            'posts': [post.to_dict() for post in self.posts]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.posts)} posts to {filepath}")
    
    def save_to_csv(self, filepath: str):
        """Save scraped posts to CSV file"""
        import csv
        
        if not self.posts:
            logger.warning("No posts to save")
            return
        
        fieldnames = ['title', 'url', 'body_preview', 'author', 'author_rank', 
                      'timestamp', 'timestamp_relative', 'category', 'views', 'likes', 'comments']
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for post in self.posts:
                writer.writerow(post.to_dict())
        
        logger.info(f"Saved {len(self.posts)} posts to {filepath}")


def main():
    """Main entry point"""
    scraper = EAForumScraper(headless=True)
    
    # Scrape posts from last 7 days
    posts = scraper.scrape(days=7, max_pages=100)
    
    if posts:
        # Save results
        scraper.save_to_json('ea_forum_posts.json')
        scraper.save_to_csv('ea_forum_posts.csv')
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SCRAPING SUMMARY")
        print(f"{'='*60}")
        print(f"Total posts scraped: {len(posts)}")
        print(f"\nSample posts:")
        for post in posts[:5]:
            print(f"\n- {post.title}")
            print(f"  Author: {post.author} | {post.timestamp_relative}")
            print(f"  Category: {post.category}")
            print(f"  Stats: {post.views} views, {post.likes} likes, {post.comments} comments")
    else:
        print("No posts were scraped.")


if __name__ == "__main__":
    main()