"""
Model analyzer module for the Article Framing Analyzer.

This module provides functionality to:
- Initialize and load both ML models (3-class softmax and ordinal)
- Analyze articles using both models
- Format model outputs for display
- Handle model loading and inference errors
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
import torch

# Import the custom model classes
from news_clf.clf_transformers import (
    PretrainedModelForOrdinalSequenceClassification,
    PretrainedModelForUnorderedSequenceClassification
)

# Set up logging
logger = logging.getLogger(__name__)

class ModelAnalyzerError(Exception):
    """Custom exception for model analyzer errors"""
    pass

class ModelLoadingError(ModelAnalyzerError):
    """Exception raised when model loading fails"""
    pass

class ModelInferenceError(ModelAnalyzerError):
    """Exception raised when model inference fails"""
    pass

class ArticleFramingAnalyzer:
    """
    Main class for analyzing article framing using both ordinal and 3-class models.
    """
    
    def __init__(self, 
                 ordinal_checkpoint: str = "ordinal_model_best_checkpoint.safetensors",
                 class_checkpoint: str = "3class_model_best_checkpoint.safetensors",
                 device_map: str = "auto"):
        """
        Initialize the Article Framing Analyzer with both models.
        
        Args:
            ordinal_checkpoint (str): Path to the ordinal model checkpoint
            class_checkpoint (str): Path to the 3-class model checkpoint
            device_map (str): Device mapping for model loading
        """
        self.ordinal_checkpoint = ordinal_checkpoint
        self.class_checkpoint = class_checkpoint
        self.device_map = device_map
        
        # Model instances
        self.ordinal_model = None
        self.class_model = None
        
        # Model metadata
        self.class_labels = ['Neutral', 'Loaded', 'Alarmist']
        self.is_initialized = False
        
        logger.info("ArticleFramingAnalyzer initialized")
    
    def initialize_models(self) -> None:
        """
        Initialize both models with proper device mapping.
        
        Raises:
            ModelLoadingError: If model loading fails
        """
        try:
            logger.info("Initializing models...")
            
            # Check if checkpoint files exist
            if not os.path.exists(self.ordinal_checkpoint):
                raise ModelLoadingError(f"Ordinal model checkpoint not found: {self.ordinal_checkpoint}")
            
            if not os.path.exists(self.class_checkpoint):
                raise ModelLoadingError(f"3-class model checkpoint not found: {self.class_checkpoint}")
            
            # Initialize ordinal model
            logger.info("Loading ordinal model...")
            self.ordinal_model = PretrainedModelForOrdinalSequenceClassification(
                model_id='microsoft/deberta-v3-xsmall',
                num_classes=3,
                device_map=self.device_map,
                checkpoint=self.ordinal_checkpoint,
                class_labels=self.class_labels
            )
            
            # Initialize 3-class model
            logger.info("Loading 3-class model...")
            self.class_model = PretrainedModelForUnorderedSequenceClassification(
                model_id='microsoft/deberta-v3-xsmall',
                num_classes=3,
                device_map=self.device_map,
                checkpoint=self.class_checkpoint,
                class_labels=self.class_labels
            )
            
            self.is_initialized = True
            logger.info("Models successfully initialized")
            
        except Exception as e:
            error_msg = f"Failed to initialize models: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadingError(error_msg)
    
    def analyze_article(self, title: str, body: str) -> Dict[str, Any]:
        """
        Analyze an article using both models.
        
        Args:
            title (str): Article title
            body (str): Article body text
            
        Returns:
            Dict[str, Any]: Analysis results containing predictions from both models
            
        Raises:
            ModelInferenceError: If model inference fails
        """
        if not self.is_initialized:
            raise ModelInferenceError("Models not initialized. Call initialize_models() first.")
        
        try:
            logger.info(f"Analyzing article: '{title[:50]}...'")
            
            # Get predictions from both models
            ordinal_result = self._get_ordinal_prediction(title, body)
            class_result = self._get_class_prediction(title, body)
            
            # Combine results
            analysis_result = {
                'title': title,
                'body_length': len(body),
                'ordinal_analysis': ordinal_result,
                'classification_analysis': class_result,
                'timestamp': self._get_timestamp()
            }
            
            logger.info("Article analysis completed successfully")
            return analysis_result
            
        except Exception as e:
            error_msg = f"Failed to analyze article: {str(e)}"
            logger.error(error_msg)
            raise ModelInferenceError(error_msg)
    
    def _get_ordinal_prediction(self, title: str, body: str) -> Dict[str, Any]:
        """
        Get prediction from the ordinal model.
        
        Args:
            title (str): Article title
            body (str): Article body text
            
        Returns:
            Dict[str, Any]: Ordinal model prediction results
        """
        try:
            # Get raw outputs for detailed analysis
            raw_outputs = self.ordinal_model.classify_article(title, body, return_raw_outputs=True)
            
            # Extract key information
            predicted_class = int(raw_outputs.class_probabilities.argmax(dim=-1).cpu()[0])
            cls_probability = float(raw_outputs.class_probabilities.squeeze(0).cpu()[predicted_class])
            scale_score = float(raw_outputs.article_score.squeeze(0).cpu())
            predicted_label = self.class_labels[predicted_class]
            
            # Get all class probabilities
            all_probabilities = raw_outputs.class_probabilities.squeeze(0).cpu().tolist()
            
            return {
                'model_type': 'ordinal',
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': cls_probability,
                'scale_score': scale_score,
                'all_probabilities': {
                    label: prob for label, prob in zip(self.class_labels, all_probabilities)
                },
                'summary': f'{predicted_label} (P={cls_probability:.3f} with scale score {scale_score:.2f})'
            }
            
        except Exception as e:
            raise ModelInferenceError(f"Ordinal model inference failed: {str(e)}")
    
    def _get_class_prediction(self, title: str, body: str) -> Dict[str, Any]:
        """
        Get prediction from the 3-class model.
        
        Args:
            title (str): Article title
            body (str): Article body text
            
        Returns:
            Dict[str, Any]: 3-class model prediction results
        """
        try:
            # Get raw outputs for detailed analysis
            raw_outputs = self.class_model.classify_article(title, body, return_raw_outputs=True)
            
            # Extract key information
            logits = raw_outputs.logits
            probabilities = logits.softmax(axis=-1)
            
            predicted_class = int(probabilities.argmax(dim=-1).cpu()[0])
            cls_probability = float(probabilities.squeeze(0).cpu()[predicted_class])
            predicted_label = self.class_labels[predicted_class]
            
            # Get all class probabilities
            all_probabilities = probabilities.squeeze(0).cpu().tolist()
            
            return {
                'model_type': '3-class',
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': cls_probability,
                'all_probabilities': {
                    label: prob for label, prob in zip(self.class_labels, all_probabilities)
                },
                'summary': f'{predicted_label} ({cls_probability:.3f})'
            }
            
        except Exception as e:
            raise ModelInferenceError(f"3-class model inference failed: {str(e)}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for analysis results."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded models.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'ordinal_checkpoint': self.ordinal_checkpoint,
            'class_checkpoint': self.class_checkpoint,
            'device_map': self.device_map,
            'class_labels': self.class_labels,
            'is_initialized': self.is_initialized,
            'ordinal_model_loaded': self.ordinal_model is not None,
            'class_model_loaded': self.class_model is not None
        }

# Convenience functions for easy integration

def create_analyzer(ordinal_checkpoint: str = "ordinal_model_best_checkpoint.safetensors",
                   class_checkpoint: str = "3class_model_best_checkpoint.safetensors",
                   device_map: str = "auto") -> ArticleFramingAnalyzer:
    """
    Create and initialize an ArticleFramingAnalyzer instance.
    
    Args:
        ordinal_checkpoint (str): Path to the ordinal model checkpoint
        class_checkpoint (str): Path to the 3-class model checkpoint
        device_map (str): Device mapping for model loading
        
    Returns:
        ArticleFramingAnalyzer: Initialized analyzer instance
        
    Raises:
        ModelLoadingError: If model initialization fails
    """
    analyzer = ArticleFramingAnalyzer(
        ordinal_checkpoint=ordinal_checkpoint,
        class_checkpoint=class_checkpoint,
        device_map=device_map
    )
    analyzer.initialize_models()
    return analyzer

def analyze_article_text(title: str, body: str, 
                        analyzer: Optional[ArticleFramingAnalyzer] = None) -> Dict[str, Any]:
    """
    Analyze article text using the framing analyzer.
    
    Args:
        title (str): Article title
        body (str): Article body text
        analyzer (Optional[ArticleFramingAnalyzer]): Pre-initialized analyzer instance
        
    Returns:
        Dict[str, Any]: Analysis results
        
    Raises:
        ModelInferenceError: If analysis fails
    """
    if analyzer is None:
        analyzer = create_analyzer()
    
    return analyzer.analyze_article(title, body) 