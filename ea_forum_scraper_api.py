"""
EA Sports FC Forum Scraper - API Version
Attempts to use the Khoros/Lithium API endpoints that may be available
"""

import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EAForumAPIClient:
    """
    Client for EA Forums using potential Khoros API endpoints
    """
    
    BASE_URL = "https://forums.ea.com"
    
    # Common Khoros/Lithium API endpoints to try
    API_ENDPOINTS = {
        'liql_v2': '/api/2.0/search',  # LiQL v2 endpoint
        'messages': '/api/2.0/messages',
        'community_v1': '/restapi/vc/categories/id/ea-sports-fc-en/messages',
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        })
    
    def try_liql_query(self, days: int = 7) -> Optional[Dict]:
        """
        Try to query using LiQL (Lithium Query Language)
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        
        # LiQL query to get messages from the category
        liql_query = f"""
        SELECT 
            id, subject, body, post_time, view_count, kudos.sum(weight),
            author.login, author.rank.name, conversation.style, board.id
        FROM messages 
        WHERE board.category.id = 'ea-sports-fc-en' 
        AND post_time >= {cutoff_str}
        ORDER BY post_time DESC
        LIMIT 500
        """
        
        # Try different endpoint formats
        endpoints_to_try = [
            f"{self.BASE_URL}/api/2.0/search?q={requests.utils.quote(liql_query)}",
            f"{self.BASE_URL}/restapi/vc/search?q={requests.utils.quote(liql_query)}&restapi.format=json",
        ]
        
        for endpoint in endpoints_to_try:
            try:
                logger.info(f"Trying endpoint: {endpoint[:100]}...")
                response = self.session.get(endpoint, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data or 'items' in data:
                        logger.info("Successfully retrieved data via API!")
                        return data
                        
            except Exception as e:
                logger.debug(f"Endpoint failed: {e}")
                continue
        
        return None
    
    def try_graphql_endpoint(self) -> Optional[Dict]:
        """
        Try GraphQL endpoint if available (newer Khoros implementations)
        """
        graphql_url = f"{self.BASE_URL}/api/graphql"
        
        query = """
        query GetMessages($categoryId: String!, $first: Int!) {
            messages(categoryId: $categoryId, first: $first, sort: POST_TIME_DESC) {
                edges {
                    node {
                        id
                        subject
                        body
                        postTime
                        viewCount
                        kudosSumWeight
                        author {
                            login
                            rank {
                                name
                            }
                        }
                        board {
                            id
                            title
                        }
                    }
                }
            }
        }
        """
        
        try:
            response = self.session.post(
                graphql_url,
                json={
                    'query': query,
                    'variables': {
                        'categoryId': 'ea-sports-fc-en',
                        'first': 100
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and not data.get('errors'):
                    logger.info("Successfully retrieved data via GraphQL!")
                    return data
                    
        except Exception as e:
            logger.debug(f"GraphQL endpoint failed: {e}")
        
        return None
    
    def check_available_endpoints(self) -> Dict[str, bool]:
        """
        Check which API endpoints are available
        """
        results = {}
        
        test_endpoints = [
            '/api/2.0/',
            '/restapi/vc/',
            '/api/graphql',
            '/t5/ea-sports-fc-en/ct-p/ea-sports-fc-en.json',
        ]
        
        for endpoint in test_endpoints:
            url = f"{self.BASE_URL}{endpoint}"
            try:
                response = self.session.get(url, timeout=10)
                results[endpoint] = response.status_code < 500
                logger.info(f"{endpoint}: Status {response.status_code}")
            except Exception as e:
                results[endpoint] = False
                logger.info(f"{endpoint}: Failed - {e}")
        
        return results


def main():
    """Test API endpoints"""
    client = EAForumAPIClient()
    
    print("Checking available API endpoints...")
    endpoints = client.check_available_endpoints()
    
    print("\nEndpoint availability:")
    for endpoint, available in endpoints.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {endpoint}: {status}")
    
    print("\nTrying LiQL query...")
    result = client.try_liql_query(days=7)
    
    if result:
        print("API data retrieved successfully!")
        print(json.dumps(result, indent=2)[:1000])
    else:
        print("API endpoints not accessible. Use the Selenium-based scraper instead.")


if __name__ == "__main__":
    main()