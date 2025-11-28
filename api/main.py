"""
FastAPI backend for EthoScore Article Analysis
"""

import os
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import our model analyzer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_analyzer import ArticleFramingAnalyzer, ModelLoadingError, ModelInferenceError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global analyzer instance
analyzer: Optional[ArticleFramingAnalyzer] = None
dataset_loaded = False

def download_file_from_google_drive(file_id: str, destination: str) -> bool:
    """
    Download a file from Google Drive, handling large file virus scan warnings.
    
    Args:
        file_id: Google Drive file ID
        destination: Local file path to save to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading file from Google Drive: {file_id} -> {destination}")
        
        # Google Drive download URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Handle large file virus scan warning
        # Google shows a warning page for files >100MB that contains a confirmation token
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        # If we got a token, we need to make another request with confirmation
        if token:
            logger.info("Large file detected, bypassing virus scan warning...")
            params = {'confirm': token, 'id': file_id}
            response = session.get(url, params=params, stream=True)
        
        # Also check for the UUID token in the response content (newer Google Drive behavior)
        if not token and response.status_code == 200:
            # Check if response is HTML (virus scan page) instead of binary file
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                # Parse the response to find the download link with confirmation
                text_content = response.text
                if 'download_warning' in text_content or 'virus' in text_content.lower():
                    # Try to extract the confirm token from HTML
                    import re
                    match = re.search(r'confirm=([^&"]+)', text_content)
                    if match:
                        token = match.group(1)
                        logger.info(f"Extracted confirmation token from HTML: {token[:20]}...")
                        params = {'confirm': token, 'id': file_id}
                        response = session.get(url, params=params, stream=True)
        
        # Check if successful
        if response.status_code != 200:
            logger.error(f"Failed to download file. Status code: {response.status_code}")
            return False
        
        # Verify we're getting binary content, not HTML error page
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type and destination.endswith('.safetensors'):
            logger.error(f"Received HTML instead of binary file. Download may have failed.")
            return False
        
        # Save file in chunks
        downloaded_size = 0
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
        
        file_size = os.path.getsize(destination)
        logger.info(f"Downloaded {destination} ({file_size / (1024*1024):.2f} MB)")
        
        # Verify file is not too small (likely an error page)
        if destination.endswith('.safetensors') and file_size < 1000000:  # Less than 1MB
            logger.error(f"Downloaded file is suspiciously small ({file_size} bytes). May be an error page.")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return False

def download_models() -> bool:
    """
    Download model files from Google Drive if they don't exist locally.
    Uses environment variables for Google Drive file IDs or URLs.
    
    Supported env variable formats:
    - ORDINAL_MODEL_ID / ORDINAL_MODEL_URL / MODEL_URL_ORDINAL_MODEL_BEST_CHECKPOINT_SAFETENSORS
    - CLASS_3_MODEL_ID / CLASS_3_MODEL_URL / MODEL_URL_3CLASS_MODEL_BEST_CHECKPOINT_SAFETENSORS
    - DATASET_ID / DATASET_URL / MODEL_URL_DATASET_FRAMING_ANNOTATIONS_LLAMA_3_3_70B_INSTRUCT_TURBO_CSV
    
    Returns:
        bool: True if all models are available, False otherwise
    """
    try:
        base_dir = Path(__file__).parent.parent
        
        models_to_download = [
            {
                'name': 'ordinal_model_best_checkpoint.safetensors',
                'env_vars': [
                    'ORDINAL_MODEL_ID',
                    'ORDINAL_MODEL_URL',
                    'MODEL_URL_ORDINAL_MODEL_BEST_CHECKPOINT_SAFETENSORS'
                ],
            },
            {
                'name': '3class_model_best_checkpoint.safetensors',
                'env_vars': [
                    'CLASS_3_MODEL_ID',
                    'CLASS_3_MODEL_URL',
                    'MODEL_URL_3CLASS_MODEL_BEST_CHECKPOINT_SAFETENSORS'
                ],
            },
            {
                'name': 'Dataset-framing_annotations-Llama-3.3-70B-Instruct-Turbo.csv',
                'env_vars': [
                    'DATASET_ID',
                    'DATASET_URL',
                    'MODEL_URL_DATASET_FRAMING_ANNOTATIONS_LLAMA_3_3_70B_INSTRUCT_TURBO_CSV'
                ],
            }
        ]
        
        all_successful = True
        
        for model_info in models_to_download:
            file_path = base_dir / model_info['name']
            
            # Skip if file already exists and is not empty
            if file_path.exists() and file_path.stat().st_size > 1000:
                logger.info(f"Model file already exists: {model_info['name']}")
                continue
            
            # Try all possible environment variable names
            file_id = None
            file_url = None
            
            for env_var in model_info['env_vars']:
                value = os.getenv(env_var)
                if value:
                    logger.info(f"Found environment variable {env_var} for {model_info['name']}")
                    if 'http' in value or 'drive.google.com' in value:
                        file_url = value
                    else:
                        file_id = value
                    break
            
            if file_id:
                logger.info(f"Downloading {model_info['name']} using file ID: {file_id[:20]}...")
                success = download_file_from_google_drive(file_id, str(file_path))
                if not success:
                    logger.error(f"Failed to download {model_info['name']}")
                    all_successful = False
            elif file_url:
                # Extract file ID from Google Drive URL
                extracted_id = None
                
                # Handle different URL formats
                if 'drive.google.com' in file_url:
                    # Format 1: https://drive.google.com/file/d/FILE_ID/view
                    if '/d/' in file_url:
                        extracted_id = file_url.split('/d/')[1].split('/')[0]
                    # Format 2: https://drive.google.com/uc?export=download&id=FILE_ID
                    elif 'id=' in file_url:
                        extracted_id = file_url.split('id=')[1].split('&')[0]
                
                if extracted_id:
                    logger.info(f"Extracted file ID from URL: {extracted_id[:20]}...")
                    success = download_file_from_google_drive(extracted_id, str(file_path))
                    if not success:
                        logger.error(f"Failed to download {model_info['name']}")
                        all_successful = False
                else:
                    logger.error(f"Could not parse Google Drive URL: {file_url}")
                    all_successful = False
            else:
                logger.warning(f"No environment variable found for {model_info['name']} "
                             f"(tried {', '.join(model_info['env_vars'])})")
                all_successful = False
        
        return all_successful
        
    except Exception as e:
        logger.error(f"Error in download_models: {str(e)}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global analyzer, dataset_loaded
    
    logger.info("[startup] Starting EthoScore API...")
    
    # Try to download models
    logger.info("[startup] Checking for model files...")
    models_available = download_models()
    
    if not models_available:
        logger.warning("[startup] WARNING: Could not download model files. Models may not work.")
    
    # Try to initialize models
    try:
        base_dir = Path(__file__).parent.parent
        ordinal_path = base_dir / "ordinal_model_best_checkpoint.safetensors"
        class_path = base_dir / "3class_model_best_checkpoint.safetensors"
        dataset_path = base_dir / "Dataset-framing_annotations-Llama-3.3-70B-Instruct-Turbo.csv"
        
        if ordinal_path.exists() and class_path.exists():
            logger.info("[startup] Initializing models...")
            analyzer = ArticleFramingAnalyzer(
                ordinal_checkpoint=str(ordinal_path),
                class_checkpoint=str(class_path),
                device_map="auto"
            )
            analyzer.initialize_models()
            logger.info("[startup] Models initialized successfully!")
        else:
            logger.error(f"[startup] Model files not found: ordinal={ordinal_path.exists()}, class={class_path.exists()}")
            
        # Check dataset
        if dataset_path.exists():
            dataset_loaded = True
            logger.info("[startup] Dataset file found")
        else:
            logger.warning("[startup] Dataset file not found")
            
    except Exception as e:
        logger.error(f"[startup] Model loading failed: {str(e)}")
    
    logger.info("[startup] Server startup complete")
    
    yield
    
    logger.info("[shutdown] Shutting down EthoScore API...")

# Create FastAPI app
app = FastAPI(
    title="EthoScore API",
    description="Article Framing Analysis API",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class AnalyzeRequest(BaseModel):
    title: str = Field(..., description="Article title")
    body: str = Field(..., description="Article body text")

class AnalyzeResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class TopicRequest(BaseModel):
    topic: str = Field(..., description="Topic to explore")

class HealthResponse(BaseModel):
    ok: bool
    models: Dict[str, bool]
    dataset_loaded: bool

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "EthoScore API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global analyzer, dataset_loaded
    
    return HealthResponse(
        ok=analyzer is not None and analyzer.is_initialized,
        models={
            "is_initialized": analyzer is not None and analyzer.is_initialized
        },
        dataset_loaded=dataset_loaded
    )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_article(request: AnalyzeRequest):
    """Analyze an article for framing bias"""
    global analyzer
    
    if analyzer is None or not analyzer.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Models not initialized. Please check server logs and ensure model files are available."
        )
    
    try:
        result = analyzer.analyze_article(request.title, request.body)
        return AnalyzeResponse(success=True, data=result)
    except ModelInferenceError as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/explore/topic")
async def explore_topic(request: TopicRequest):
    """Explore articles related to a topic"""
    global analyzer, dataset_loaded
    
    if analyzer is None or not analyzer.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Models not initialized"
        )
    
    if not dataset_loaded:
        raise HTTPException(
            status_code=503,
            detail="Dataset not loaded"
        )
    
    # TODO: Implement topic exploration logic
    # For now, return a placeholder response
    return {
        "success": True,
        "topic": request.topic,
        "message": "Topic exploration not yet implemented"
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about loaded models"""
    global analyzer
    
    if analyzer is None:
        return {
            "initialized": False,
            "error": "Models not initialized"
        }
    
    return analyzer.get_model_info()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
