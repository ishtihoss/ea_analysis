#!/usr/bin/env python3
"""
Runner script for EA Forum Scraper
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Scrape EA Sports FC Forums')
    parser.add_argument('--days', type=int, default=7, help='Number of days to look back (default: 7)')
    parser.add_argument('--output', type=str, default='ea_forum_posts', help='Output filename (without extension)')
    parser.add_argument('--format', choices=['json', 'csv', 'both'], default='both', help='Output format')
    parser.add_argument('--headless', action='store_true', default=True, help='Run browser in headless mode')
    parser.add_argument('--no-headless', dest='headless', action='store_false', help='Show browser window')
    parser.add_argument('--max-pages', type=int, default=100, help='Maximum "Show More" button clicks')
    
    args = parser.parse_args()
    
    # Import here to avoid issues if dependencies aren't installed
    try:
        from ea_forum_scraper import EAForumScraper
    except ImportError as e:
        print(f"Error importing scraper: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Run scraper
    print(f"Scraping EA Sports FC forums for posts from the last {args.days} days...")
    
    scraper = EAForumScraper(headless=args.headless)
    posts = scraper.scrape(days=args.days, max_pages=args.max_pages)
    
    if not posts:
        print("No posts found. The forum might be empty or there was an error.")
        sys.exit(1)
    
    # Save results
    output_dir = Path('.')
    
    if args.format in ['json', 'both']:
        json_path = output_dir / f"{args.output}.json"
        scraper.save_to_json(str(json_path))
        print(f"Saved JSON to: {json_path}")
    
    if args.format in ['csv', 'both']:
        csv_path = output_dir / f"{args.output}.csv"
        scraper.save_to_csv(str(csv_path))
        print(f"Saved CSV to: {csv_path}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Successfully scraped {len(posts)} posts")
    print(f"{'='*50}")
    
    # Show sample for verification
    if posts:
        print("\nMost recent posts:")
        for i, post in enumerate(posts[:5], 1):
            print(f"\n{i}. {post.title}")
            print(f"   URL: {post.url}")
            print(f"   Author: {post.author} ({post.author_rank})")
            print(f"   Time: {post.timestamp_relative} ({post.timestamp})")
            print(f"   Category: {post.category}")
            print(f"   Content Preview: {post.body_preview[:100]}..." if len(post.body_preview) > 100 else f"   Content Preview: {post.body_preview}")
            print(f"   Stats: 👁 {post.views} views | 👍 {post.likes} likes | 💬 {post.comments} comments")


if __name__ == "__main__":
    main()