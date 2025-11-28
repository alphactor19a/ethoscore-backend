"""
Article processing module for extracting content from URLs and handling manual text input.

This module provides functionality to:
- Validate and sanitize URLs
- Extract article content from URLs using newspaper4k
- Process manual text input (title and body)
- Handle errors gracefully for failed requests and parsing
"""

import logging
import re
from urllib.parse import urlparse, urlunparse
from typing import Tuple, Optional, Dict, Any

try:
    import newspaper
    from newspaper import Article
except Exception as e:
    import traceback
    traceback.print_exc()
    raise ImportError("newspaper4k is required. Install it with: pip install newspaper4k") from e


# Set up logging
logger = logging.getLogger(__name__)

class ArticleProcessingError(Exception):
    """Custom exception for article processing errors"""
    pass

class URLValidationError(ArticleProcessingError):
    """Exception raised for invalid URLs"""
    pass

class ArticleExtractionError(ArticleProcessingError):
    """Exception raised when article extraction fails"""
    pass

def validate_url(url: str) -> str:
    """
    Validate and sanitize a URL.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        str: The validated and sanitized URL
        
    Raises:
        URLValidationError: If the URL is invalid
    """
    if not url or not isinstance(url, str):
        raise URLValidationError("URL must be a non-empty string")
    
    # Strip whitespace
    url = url.strip()
    
    if not url:
        raise URLValidationError("URL cannot be empty")
    
    # Add http:// if no scheme is provided
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Parse the URL to validate it
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            raise URLValidationError("Invalid URL: missing domain")
        
        # Basic domain validation
        domain_pattern = re.compile(
            r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
        )
        if not domain_pattern.match(parsed.netloc.split(':')[0]):
            raise URLValidationError("Invalid URL: malformed domain")
            
        return url
        
    except Exception as e:
        raise URLValidationError(f"Invalid URL format: {str(e)}")

def extract_article_from_url(url: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Extract article title, body, and metadata from a URL using newspaper4k.
    
    Args:
        url (str): The URL to extract content from
        
    Returns:
        Tuple[str, str, Dict[str, Any]]: (title, body, metadata)
        
    Raises:
        ArticleExtractionError: If extraction fails
    """
    try:
        # Validate the URL first
        validated_url = validate_url(url)
        logger.info(f"Extracting article from URL: {validated_url}")
        
        # Create article object and extract content
        article = Article(validated_url)
        article.download()
        article.parse()
        
        # Get title and text
        title = article.title or ""
        body = article.text or ""
        
        # Validate that we got some content
        if not title and not body:
            raise ArticleExtractionError("No content could be extracted from the URL")
        
        if not title:
            logger.warning("No title found in article")
            title = "Untitled Article"
        
        if not body:
            raise ArticleExtractionError("No article body content found")
        
        # Clean up the content
        title = title.strip()
        body = body.strip()
        
        # Extract additional metadata
        authors = article.authors if hasattr(article, 'authors') else []
        publish_date = article.publish_date.isoformat() if hasattr(article, 'publish_date') and article.publish_date else None
        source_url = article.source_url if hasattr(article, 'source_url') else validated_url
        domain = urlparse(source_url).netloc if source_url else urlparse(validated_url).netloc
        
        logger.info(f"Successfully extracted article: '{title[:50]}...' ({len(body)} characters)")
        metadata = {
            "authors": authors,
            "publish_date": publish_date,
            "source": domain,
            "source_url": source_url
        }
        return title, body, metadata
        
    except URLValidationError:
        # Re-raise URL validation errors
        raise
    except Exception as e:
        error_msg = f"Failed to extract article from URL: {str(e)}"
        logger.error(error_msg)
        raise ArticleExtractionError(error_msg)

def process_manual_input(title: str, body: str) -> Tuple[str, str]:
    """
    Process and validate manual text input.
    
    Args:
        title (str): The article title
        body (str): The article body
        
    Returns:
        Tuple[str, str]: A tuple containing (cleaned_title, cleaned_body)
        
    Raises:
        ArticleProcessingError: If input validation fails
    """
    # Validate inputs
    if not isinstance(title, str) or not isinstance(body, str):
        raise ArticleProcessingError("Title and body must be strings")
    
    # Clean up the inputs
    title = title.strip()
    body = body.strip()
    
    # Validate that we have content
    if not title and not body:
        raise ArticleProcessingError("Both title and body cannot be empty")
    
    if not body:
        raise ArticleProcessingError("Article body cannot be empty")
    
    if not title:
        logger.warning("No title provided, using default")
        title = "Untitled Article"
    
    # Basic length validation
    if len(body) < 10:
        raise ArticleProcessingError("Article body is too short (minimum 10 characters)")
    
    if len(title) > 500:
        logger.warning("Title is very long, truncating")
        title = title[:500] + "..."
    
    logger.info(f"Processed manual input: '{title[:50]}...' ({len(body)} characters)")
    return title, body

def get_article_info(title: str, body: str) -> Dict[str, Any]:
    """
    Get basic information about an article.
    
    Args:
        title (str): The article title
        body (str): The article body
        
    Returns:
        Dict[str, Any]: Article information including word count, character count, etc.
    """
    word_count = len(body.split())
    char_count = len(body)
    title_length = len(title)
    
    # Estimate reading time (average 200 words per minute)
    reading_time_minutes = max(1, round(word_count / 200))
    
    return {
        "title_length": title_length,
        "word_count": word_count,
        "character_count": char_count,
        "estimated_reading_time_minutes": reading_time_minutes
    }

# Main processing functions for the Streamlit app
def process_url_input(url: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Main function to process URL input for the Streamlit app.
    
    Args:
        url (str): The URL to process
        
    Returns:
        Tuple[str, str, Dict[str, Any]]: (title, body, article_info)
        
    Raises:
        ArticleProcessingError: If processing fails
    """
    title, body, metadata = extract_article_from_url(url)
    article_info = get_article_info(title, body)
    article_info.update(metadata)
    return title, body, article_info

def process_manual_text_input(title: str, body: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Main function to process manual text input for the Streamlit app.
    
    Args:
        title (str): The article title
        body (str): The article body
        
    Returns:
        Tuple[str, str, Dict[str, Any]]: (cleaned_title, cleaned_body, article_info)
        
    Raises:
        ArticleProcessingError: If processing fails
    """
    clean_title, clean_body = process_manual_input(title, body)
    article_info = get_article_info(clean_title, clean_body)
    # For manual input, set metadata fields to None/empty
    article_info.update({
        "authors": [],
        "publish_date": None,
        "source": None,
        "source_url": None
    })
    return clean_title, clean_body, article_info 