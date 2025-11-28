"""
Batch processing module for the Article Framing Analyzer.

This module provides functionality to process multiple URLs in batch mode,
with comprehensive error handling, progress tracking, and result aggregation.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import time

from src.article_processor import process_url_input, ArticleProcessingError, URLValidationError, ArticleExtractionError
from src.model_analyzer import ArticleFramingAnalyzer, ModelInferenceError

logger = logging.getLogger(__name__)

@dataclass
class BatchResult:
    """Result for a single URL in batch processing."""
    url: str
    success: bool
    title: Optional[str] = None
    body: Optional[str] = None
    body_length: Optional[int] = None
    word_count: Optional[int] = None
    reading_time: Optional[float] = None
    ordinal_analysis: Optional[Dict[str, Any]] = None
    classification_analysis: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    timestamp: Optional[str] = None
    # CSV metadata fields
    csv_title: Optional[str] = None
    publish_date: Optional[str] = None
    media_name: Optional[str] = None
    csv_id: Optional[str] = None
    source_file: Optional[str] = None

class BatchProcessor:
    """
    Batch processor for analyzing multiple URLs.
    
    Features:
    - Concurrent processing for efficiency
    - Comprehensive error handling
    - Progress tracking
    - Result aggregation and export
    """
    
    def __init__(self, analyzer: ArticleFramingAnalyzer, max_concurrent: int = 3):
        """
        Initialize the batch processor.
        
        Args:
            analyzer: Initialized ArticleFramingAnalyzer instance
            max_concurrent: Maximum number of concurrent URL processing tasks
        """
        self.analyzer = analyzer
        self.max_concurrent = max_concurrent
        self.results: List[BatchResult] = []
        
    async def process_url_async(self, url: str, csv_metadata: Optional[Dict[str, str]] = None) -> BatchResult:
        """
        Process a single URL asynchronously.
        
        Args:
            url: URL to process
            csv_metadata: Optional metadata from CSV (title, publish_date, media_name, id)
            
        Returns:
            BatchResult with processing results
        """
        start_time = time.time()
        
        try:
            # Extract article content
            title, body, article_info = process_url_input(url)
            
            # Analyze with models
            analysis_results = self.analyzer.analyze_article(title, body)
            
            processing_time = time.time() - start_time
            
            # Create result with optional CSV metadata
            result = BatchResult(
                url=url,
                success=True,
                title=title,
                body=body,
                body_length=len(body),
                word_count=article_info.get('word_count'),
                reading_time=article_info.get('reading_time'),
                ordinal_analysis=analysis_results['ordinal_analysis'],
                classification_analysis=analysis_results['classification_analysis'],
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
            # Add CSV metadata if provided
            if csv_metadata:
                result.csv_title = csv_metadata.get('title')
                result.publish_date = csv_metadata.get('publish_date')
                result.media_name = csv_metadata.get('media_name')
                result.csv_id = csv_metadata.get('id')
            
            return result
            
        except URLValidationError as e:
            result = BatchResult(
                url=url,
                success=False,
                error_message=f"Invalid URL: {str(e)}",
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            if csv_metadata:
                result.csv_title = csv_metadata.get('title')
                result.publish_date = csv_metadata.get('publish_date')
                result.media_name = csv_metadata.get('media_name')
                result.csv_id = csv_metadata.get('id')
            return result
        except ArticleExtractionError as e:
            result = BatchResult(
                url=url,
                success=False,
                error_message=f"Failed to extract article: {str(e)}",
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            if csv_metadata:
                result.csv_title = csv_metadata.get('title')
                result.publish_date = csv_metadata.get('publish_date')
                result.media_name = csv_metadata.get('media_name')
                result.csv_id = csv_metadata.get('id')
            return result
        except ModelInferenceError as e:
            result = BatchResult(
                url=url,
                success=False,
                error_message=f"Model analysis failed: {str(e)}",
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            if csv_metadata:
                result.csv_title = csv_metadata.get('title')
                result.publish_date = csv_metadata.get('publish_date')
                result.media_name = csv_metadata.get('media_name')
                result.csv_id = csv_metadata.get('id')
            return result
        except Exception as e:
            result = BatchResult(
                url=url,
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            if csv_metadata:
                result.csv_title = csv_metadata.get('title')
                result.publish_date = csv_metadata.get('publish_date')
                result.media_name = csv_metadata.get('media_name')
                result.csv_id = csv_metadata.get('id')
            return result
    
    async def process_csv_data(self, csv_content: str) -> List[BatchResult]:
        """
        Process CSV data in the format: title,url,publish_date,media_name,id
        
        Args:
            csv_content: CSV string with the specified format
            
        Returns:
            List of BatchResult objects
        """
        # Parse CSV data
        csv_records = parse_csv_input(csv_content)
        
        if not csv_records:
            return []
        
        # Extract URLs and metadata
        urls_with_metadata = []
        for record in csv_records:
            urls_with_metadata.append((record['url'], record))
        
        # Process with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(url_and_metadata: Tuple[str, Dict[str, str]]) -> BatchResult:
            url, metadata = url_and_metadata
            async with semaphore:
                return await self.process_url_async(url, metadata)
        
        tasks = [process_with_semaphore(item) for item in urls_with_metadata]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred during gathering
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                url, metadata = urls_with_metadata[i]
                processed_results.append(BatchResult(
                    url=url,
                    success=False,
                    error_message=f"Processing error: {str(result)}",
                    timestamp=datetime.now().isoformat(),
                    csv_title=metadata.get('title'),
                    publish_date=metadata.get('publish_date'),
                    media_name=metadata.get('media_name'),
                    csv_id=metadata.get('id')
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def process_csv_data_sync(self, csv_content: str) -> List[BatchResult]:
        """
        Process CSV data synchronously (for use in Streamlit).
        
        Args:
            csv_content: CSV string with the specified format
            
        Returns:
            List of BatchResult objects
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_csv_data(csv_content))
        finally:
            loop.close()

    async def process_urls_batch(self, urls: List[str]) -> List[BatchResult]:
        """
        Process multiple URLs in batch with concurrency control.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of BatchResult objects
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(url: str) -> BatchResult:
            async with semaphore:
                return await self.process_url_async(url)
        
        tasks = [process_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred during gathering
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(BatchResult(
                    url=urls[i],
                    success=False,
                    error_message=f"Processing error: {str(result)}",
                    timestamp=datetime.now().isoformat()
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def process_urls_sync(self, urls: List[str]) -> List[BatchResult]:
        """
        Process URLs synchronously (for use in Streamlit).
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of BatchResult objects
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_urls_batch(urls))
        finally:
            loop.close()
    
    async def process_multiple_csv_files(self, csv_files: List[Tuple[str, str]]) -> List[BatchResult]:
        """
        Process multiple CSV files in batch.
        
        Args:
            csv_files: List of tuples (filename, csv_content)
            
        Returns:
            List of BatchResult objects from all CSV files
        """
        all_results = []
        
        for filename, csv_content in csv_files:
            try:
                # Parse CSV data
                csv_records = parse_csv_input(csv_content)
                
                if not csv_records:
                    continue
                
                # Add filename to metadata for tracking
                for record in csv_records:
                    record['source_file'] = filename
                
                # Extract URLs and metadata
                urls_with_metadata = []
                for record in csv_records:
                    urls_with_metadata.append((record['url'], record))
                
                # Process with concurrency control
                semaphore = asyncio.Semaphore(self.max_concurrent)
                
                async def process_with_semaphore(url_and_metadata: Tuple[str, Dict[str, str]]) -> BatchResult:
                    url, metadata = url_and_metadata
                    async with semaphore:
                        return await self.process_url_async(url, metadata)
                
                tasks = [process_with_semaphore(item) for item in urls_with_metadata]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle any exceptions that occurred during gathering
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        url, metadata = urls_with_metadata[i]
                        all_results.append(BatchResult(
                            url=url,
                            success=False,
                            error_message=f"Processing error: {str(result)}",
                            timestamp=datetime.now().isoformat(),
                            csv_title=metadata.get('title'),
                            publish_date=metadata.get('publish_date'),
                            media_name=metadata.get('media_name'),
                            csv_id=metadata.get('id'),
                            source_file=metadata.get('source_file')
                        ))
                    else:
                        # Add source file to existing result
                        result.source_file = filename
                        all_results.append(result)
                        
            except Exception as e:
                # If CSV parsing fails for a file, create error results
                error_result = BatchResult(
                    url=f"FILE_ERROR_{filename}",
                    success=False,
                    error_message=f"Failed to process CSV file {filename}: {str(e)}",
                    timestamp=datetime.now().isoformat(),
                    source_file=filename
                )
                all_results.append(error_result)
        
        return all_results
    
    def process_multiple_csv_files_sync(self, csv_files: List[Tuple[str, str]]) -> List[BatchResult]:
        """
        Process multiple CSV files synchronously (for use in Streamlit).
        
        Args:
            csv_files: List of tuples (filename, csv_content)
            
        Returns:
            List of BatchResult objects from all CSV files
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_multiple_csv_files(csv_files))
        finally:
            loop.close()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for batch processing results.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {}
        
        total_urls = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total_urls - successful
        
        # Calculate average processing time
        processing_times = [r.processing_time for r in self.results if r.processing_time]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Analyze successful results
        if successful > 0:
            successful_results = [r for r in self.results if r.success]
            
            # Ordinal model statistics
            ordinal_scores = [r.ordinal_analysis['normalized_score'] for r in successful_results]
            avg_ordinal_score = sum(ordinal_scores) / len(ordinal_scores)
            
            # Classification distribution
            class_counts = {}
            for r in successful_results:
                label = r.classification_analysis['predicted_label']
                class_counts[label] = class_counts.get(label, 0) + 1
            
            # Confidence statistics
            ordinal_confidences = [r.ordinal_analysis['confidence'] for r in successful_results]
            avg_ordinal_confidence = sum(ordinal_confidences) / len(ordinal_confidences)
            
            class_confidences = [r.classification_analysis['confidence'] for r in successful_results]
            avg_class_confidence = sum(class_confidences) / len(class_confidences)
            
            return {
                'total_urls': total_urls,
                'successful': successful,
                'failed': failed,
                'success_rate': successful / total_urls,
                'avg_processing_time': avg_processing_time,
                'avg_ordinal_score': avg_ordinal_score,
                'avg_ordinal_confidence': avg_ordinal_confidence,
                'avg_class_confidence': avg_class_confidence,
                'class_distribution': class_counts,
                'min_ordinal_score': min(ordinal_scores),
                'max_ordinal_score': max(ordinal_scores)
            }
        else:
            return {
                'total_urls': total_urls,
                'successful': 0,
                'failed': failed,
                'success_rate': 0,
                'avg_processing_time': avg_processing_time
            }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame for easy analysis.
        
        Returns:
            DataFrame with all results
        """
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            row = {
                'url': result.url,
                'success': result.success,
                'title': result.title,
                'body_length': result.body_length,
                'word_count': result.word_count,
                'reading_time': result.reading_time,
                'processing_time': result.processing_time,
                'timestamp': result.timestamp,
                'error_message': result.error_message
            }
            
            if result.success and result.ordinal_analysis:
                row.update({
                    'ordinal_predicted_label': result.ordinal_analysis['predicted_label'],
                    'ordinal_confidence': result.ordinal_analysis['confidence'],
                    'ordinal_raw_score': result.ordinal_analysis['scale_score'],
                    'ordinal_normalized_score': result.ordinal_analysis['normalized_score'],
                    'ordinal_interpretation': result.ordinal_analysis['score_interpretation'],
                    'ordinal_confidence_level': result.ordinal_analysis['confidence_level'],
                    'neutral_probability': result.ordinal_analysis['all_probabilities'].get('Neutral', 0),
                    'loaded_probability': result.ordinal_analysis['all_probabilities'].get('Loaded', 0),
                    'alarmist_probability': result.ordinal_analysis['all_probabilities'].get('Alarmist', 0)
                })
            
            if result.success and result.classification_analysis:
                row.update({
                    'classification_predicted_label': result.classification_analysis['predicted_label'],
                    'classification_confidence': result.classification_analysis['confidence'],
                    'class_neutral_probability': result.classification_analysis['all_probabilities'].get('Neutral', 0),
                    'class_loaded_probability': result.classification_analysis['all_probabilities'].get('Loaded', 0),
                    'class_alarmist_probability': result.classification_analysis['all_probabilities'].get('Alarmist', 0)
                })
            
            if result.csv_title:
                row['csv_title'] = result.csv_title
            if result.publish_date:
                row['publish_date'] = result.publish_date
            if result.media_name:
                row['media_name'] = result.media_name
            if result.csv_id:
                row['csv_id'] = result.csv_id
            if result.source_file:
                row['source_file'] = result.source_file
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def export_results(self, format: str = 'csv', filename: Optional[str] = None) -> str:
        """
        Export results to various formats.
        
        Args:
            format: Export format ('csv', 'excel', 'json')
            filename: Optional filename (without extension)
            
        Returns:
            Generated filename
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_analysis_results_{timestamp}"
        
        df = self.to_dataframe()
        
        if format.lower() == 'csv':
            full_filename = f"{filename}.csv"
            df.to_csv(full_filename, index=False)
        elif format.lower() == 'excel':
            full_filename = f"{filename}.xlsx"
            df.to_excel(full_filename, index=False)
        elif format.lower() == 'json':
            full_filename = f"{filename}.json"
            df.to_json(full_filename, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return full_filename

def parse_csv_input(csv_content: str) -> List[Dict[str, str]]:
    """
    Parse CSV input in the format: title,url,publish_date,media_name,id
    
    Args:
        csv_content: String containing CSV data
        
    Returns:
        List of dictionaries with parsed data
    """
    if not csv_content.strip():
        return []
    
    import io
    import pandas as pd
    
    try:
        # Read CSV from string
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Validate required columns
        required_columns = ['title', 'url', 'publish_date', 'media_name', 'id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert to list of dictionaries
        records = []
        for _, row in df.iterrows():
            record = {
                'title': str(row['title']),
                'url': str(row['url']),
                'publish_date': str(row['publish_date']),
                'media_name': str(row['media_name']),
                'id': str(row['id'])
            }
            records.append(record)
        
        return records
        
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {str(e)}")

def parse_url_input(url_input: str) -> List[str]:
    """
    Parse URL input from various formats.
    
    Supports:
    - One URL per line
    - Comma-separated URLs
    - URLs separated by semicolons
    - Markdown links [text](url)
    - URLs with extra characters
    - CSV format with columns: title,url,publish_date,media_name,id
    
    Args:
        url_input: String containing URLs in various formats
        
    Returns:
        List of cleaned URLs
    """
    if not url_input.strip():
        return []
    
    import re
    
    # Check if input looks like CSV format
    lines = url_input.strip().split('\n')
    if len(lines) > 0:
        first_line = lines[0].strip()
        # Check if first line contains CSV headers
        if 'title' in first_line.lower() and 'url' in first_line.lower() and 'publish_date' in first_line.lower():
            try:
                # Parse as CSV and extract URLs
                csv_records = parse_csv_input(url_input)
                urls = [record['url'] for record in csv_records]
                return urls
            except ValueError:
                # If CSV parsing fails, fall back to regular URL parsing
                pass
    
    # Original URL parsing logic
    urls = []
    for line in url_input.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Split by comma or semicolon
        for url in line.replace(';', ',').split(','):
            url = url.strip()
            if not url:
                continue
                
            # Clean the URL
            cleaned_url = clean_url(url)
            if cleaned_url and cleaned_url not in urls:
                urls.append(cleaned_url)
    
    return urls

def clean_url(url: str) -> str:
    """
    Clean and extract URL from various formats.
    
    Args:
        url: Raw URL string that might be malformed
        
    Returns:
        Cleaned URL string
    """
    import re
    
    # Remove markdown link format [text](url)
    markdown_pattern = r'\[([^\]]*)\]\(([^)]+)\)'
    markdown_match = re.search(markdown_pattern, url)
    if markdown_match:
        url = markdown_match.group(2)
    
    # Remove extra characters and whitespace
    url = url.strip()
    
    # Handle malformed URLs that end with ](https://...
    # This pattern matches URLs that got corrupted with markdown syntax
    malformed_pattern = r'(https?://[^\]\s]+)\]\s*\(https?://[^\)]*\)'
    malformed_match = re.search(malformed_pattern, url)
    if malformed_match:
        url = malformed_match.group(1)
    
    # Remove trailing punctuation and brackets
    url = re.sub(r'[\]\)\}\s]+$', '', url)
    
    # Remove leading punctuation and brackets
    url = re.sub(r'^[\[\(\{\s]+', '', url)
    
    # Ensure URL starts with http:// or https://
    if not url.startswith(('http://', 'https://')):
        # Try to find a URL pattern in the string
        url_pattern = re.search(r'https?://[^\s\]\)\}\]]+', url)
        if url_pattern:
            url = url_pattern.group(0)
        else:
            return ""  # No valid URL found
    
    # Remove any remaining markdown or extra characters
    url = re.sub(r'[\]\)\}\s]+$', '', url)
    
    # Final cleanup - remove any remaining malformed parts
    # Look for URLs that end with ]( and remove everything after
    url = re.sub(r'\]\s*\([^\)]*$', '', url)
    url = re.sub(r'\]\s*\([^\)]*\)', '', url)
    
    return url.strip()

def validate_urls(urls: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate a list of URLs.
    
    Args:
        urls: List of URLs to validate
        
    Returns:
        Tuple of (valid_urls, invalid_urls)
    """
    import re
    
    valid_urls = []
    invalid_urls = []
    
    # Basic URL pattern
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    for url in urls:
        if url_pattern.match(url):
            valid_urls.append(url)
        else:
            invalid_urls.append(url)
    
    return valid_urls, invalid_urls 