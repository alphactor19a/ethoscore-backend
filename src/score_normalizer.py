"""
Score normalization module for the Article Framing Analyzer.

This module provides functionality to normalize the ordinal model's scale scores
from their unbounded range to a standardized 0-100 scale for research purposes.
The normalization preserves the ordinal relationships and class boundaries
while providing a more interpretable scale for analysis.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class NormalizationConfig:
    """Configuration for score normalization."""
    # Expected score ranges based on training data analysis
    neutral_range: Tuple[float, float] = (-10.0, 2.0)      # Neutral articles typically score -10 to +2
    loaded_range: Tuple[float, float] = (1.0, 6.0)         # Loaded articles typically score +1 to +6
    alarmist_range: Tuple[float, float] = (5.0, 15.0)      # Alarmist articles typically score +5 to +15+
    
    # Key transition points
    neutral_loaded_threshold: float = -0.5    # Crossover point between Neutral and Loaded
    loaded_alarmist_threshold: float = 6.0    # Crossover point between Loaded and Alarmist
    
    # Normalized score ranges (0-100 scale)
    normalized_neutral_range: Tuple[int, int] = (0, 30)     # 0-30 for Neutral
    normalized_loaded_range: Tuple[int, int] = (25, 65)     # 25-65 for Loaded  
    normalized_alarmist_range: Tuple[int, int] = (60, 100)  # 60-100 for Alarmist
    
    # Smoothing parameters for continuous mapping
    smoothing_factor: float = 0.1  # Controls smoothness of transitions between ranges

class ScoreNormalizer:
    """
    Normalizes ordinal model scale scores to a standardized 0-100 range.
    
    The normalization process:
    1. Maps the unbounded scale scores to meaningful ranges based on training data
    2. Preserves ordinal relationships and class boundaries
    3. Provides interpretable scores for research analysis
    4. Handles edge cases and outliers gracefully
    """
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize the score normalizer.
        
        Args:
            config: Normalization configuration. If None, uses default config.
        """
        self.config = config or NormalizationConfig()
        logger.info("ScoreNormalizer initialized with configuration")
        
    def normalize_score(self, raw_score: float) -> Dict[str, Any]:
        """
        Normalize a raw scale score to 0-100 range.
        
        Args:
            raw_score: Raw scale score from the ordinal model
            
        Returns:
            Dictionary containing normalized score and metadata
        """
        try:
            # Step 1: Determine the class region for the score
            class_region = self._determine_class_region(raw_score)
            
            # Step 2: Apply region-specific normalization
            normalized_score = self._apply_region_normalization(raw_score, class_region)
            
            # Step 3: Apply smoothing for scores near boundaries
            smoothed_score = self._apply_boundary_smoothing(raw_score, normalized_score, class_region)
            
            # Step 4: Clamp to valid range
            final_score = np.clip(smoothed_score, 0, 100)
            
            # Step 5: Generate interpretation
            interpretation = self._generate_interpretation(final_score, raw_score, class_region)
            
            return {
                'raw_score': raw_score,
                'normalized_score': final_score,
                'class_region': class_region,
                'interpretation': interpretation,
                'confidence_level': self._calculate_confidence_level(raw_score, class_region)
            }
            
        except Exception as e:
            logger.error(f"Error normalizing score {raw_score}: {e}")
            return self._fallback_normalization(raw_score)
    
    def _determine_class_region(self, raw_score: float) -> str:
        """
        Determine which class region the raw score belongs to.
        
        Args:
            raw_score: Raw scale score
            
        Returns:
            Class region: 'neutral', 'loaded', or 'alarmist'
        """
        if raw_score <= self.config.neutral_loaded_threshold:
            return 'neutral'
        elif raw_score <= self.config.loaded_alarmist_threshold:
            return 'loaded'
        else:
            return 'alarmist'
    
    def _apply_region_normalization(self, raw_score: float, class_region: str) -> float:
        """
        Apply region-specific normalization based on the class region.
        
        Args:
            raw_score: Raw scale score
            class_region: Determined class region
            
        Returns:
            Normalized score in 0-100 range
        """
        if class_region == 'neutral':
            # Map neutral range to 0-30
            source_range = self.config.neutral_range
            target_range = self.config.normalized_neutral_range
        elif class_region == 'loaded':
            # Map loaded range to 25-65
            source_range = self.config.loaded_range
            target_range = self.config.normalized_loaded_range
        else:  # alarmist
            # Map alarmist range to 60-100
            source_range = self.config.alarmist_range
            target_range = self.config.normalized_alarmist_range
        
        # Linear interpolation within the region
        source_min, source_max = source_range
        target_min, target_max = target_range
        
        # Handle edge cases where score is outside the expected range
        if raw_score < source_min:
            normalized = target_min
        elif raw_score > source_max:
            normalized = target_max
        else:
            # Linear interpolation
            normalized = target_min + (raw_score - source_min) * (target_max - target_min) / (source_max - source_min)
        
        return normalized
    
    def _apply_boundary_smoothing(self, raw_score: float, normalized_score: float, class_region: str) -> float:
        """
        Apply smoothing for scores near class boundaries to ensure smooth transitions.
        
        Args:
            raw_score: Raw scale score
            normalized_score: Current normalized score
            class_region: Current class region
            
        Returns:
            Smoothed normalized score
        """
        # Calculate distance to nearest boundary
        if class_region == 'neutral':
            boundary_distance = abs(raw_score - self.config.neutral_loaded_threshold)
        elif class_region == 'loaded':
            # Distance to nearest boundary (neutral or alarmist)
            dist_to_neutral = abs(raw_score - self.config.neutral_loaded_threshold)
            dist_to_alarmist = abs(raw_score - self.config.loaded_alarmist_threshold)
            boundary_distance = min(dist_to_neutral, dist_to_alarmist)
        else:  # alarmist
            boundary_distance = abs(raw_score - self.config.loaded_alarmist_threshold)
        
        # Apply smoothing if close to boundary
        smoothing_threshold = 1.0  # Apply smoothing within 1 unit of boundary
        if boundary_distance < smoothing_threshold:
            smoothing_factor = (1 - boundary_distance / smoothing_threshold) * self.config.smoothing_factor
            # Slightly adjust the score towards the boundary
            if raw_score < self.config.neutral_loaded_threshold:
                normalized_score += smoothing_factor * 5  # Move towards loaded region
            elif raw_score > self.config.loaded_alarmist_threshold:
                normalized_score -= smoothing_factor * 5  # Move towards loaded region
        
        return normalized_score
    
    def _generate_interpretation(self, normalized_score: float, raw_score: float, class_region: str) -> str:
        """
        Generate human-readable interpretation of the normalized score.
        
        Args:
            normalized_score: Normalized score (0-100)
            raw_score: Raw scale score
            class_region: Class region
            
        Returns:
            Interpretation string
        """
        if normalized_score <= 30:
            intensity = "Very Low" if normalized_score <= 10 else "Low" if normalized_score <= 20 else "Moderately Low"
        elif normalized_score <= 65:
            intensity = "Moderate" if normalized_score <= 45 else "Moderately High" if normalized_score <= 55 else "High"
        else:
            intensity = "Very High" if normalized_score >= 90 else "High" if normalized_score >= 80 else "Moderately High"
        
        return f"{intensity} emotional intensity ({normalized_score:.1f}/100) - {class_region.title()} framing detected"
    
    def _calculate_confidence_level(self, raw_score: float, class_region: str) -> str:
        """
        Calculate confidence level based on distance from class boundaries.
        
        Args:
            raw_score: Raw scale score
            class_region: Class region
            
        Returns:
            Confidence level: 'High', 'Medium', or 'Low'
        """
        if class_region == 'neutral':
            distance = abs(raw_score - self.config.neutral_loaded_threshold)
        elif class_region == 'loaded':
            dist_to_neutral = abs(raw_score - self.config.neutral_loaded_threshold)
            dist_to_alarmist = abs(raw_score - self.config.loaded_alarmist_threshold)
            distance = min(dist_to_neutral, dist_to_alarmist)
        else:  # alarmist
            distance = abs(raw_score - self.config.loaded_alarmist_threshold)
        
        if distance > 3.0:
            return "High"
        elif distance > 1.0:
            return "Medium"
        else:
            return "Low"
    
    def _fallback_normalization(self, raw_score: float) -> Dict[str, Any]:
        """
        Fallback normalization for edge cases or errors.
        
        Args:
            raw_score: Raw scale score
            
        Returns:
            Fallback normalization result
        """
        # Simple linear mapping as fallback
        normalized_score = np.clip((raw_score + 10) * 5, 0, 100)  # Rough mapping
        
        return {
            'raw_score': raw_score,
            'normalized_score': normalized_score,
            'class_region': 'unknown',
            'interpretation': f"Fallback normalization: {normalized_score:.1f}/100",
            'confidence_level': 'Low'
        }
    
    def get_normalization_info(self) -> Dict[str, Any]:
        """
        Get information about the normalization configuration.
        
        Returns:
            Dictionary with normalization configuration details
        """
        return {
            'config': {
                'neutral_range': self.config.neutral_range,
                'loaded_range': self.config.loaded_range,
                'alarmist_range': self.config.alarmist_range,
                'neutral_loaded_threshold': self.config.neutral_loaded_threshold,
                'loaded_alarmist_threshold': self.config.loaded_alarmist_threshold,
                'normalized_ranges': {
                    'neutral': self.config.normalized_neutral_range,
                    'loaded': self.config.normalized_loaded_range,
                    'alarmist': self.config.normalized_alarmist_range
                }
            },
            'description': {
                'purpose': 'Convert ordinal model scale scores to standardized 0-100 range',
                'methodology': 'Region-based normalization with boundary smoothing',
                'preserves': 'Ordinal relationships and class boundaries',
                'interpretation': 'Higher scores indicate greater emotional intensity/alarmist framing'
            }
        }

# Global normalizer instance
_default_normalizer = None

def get_normalizer() -> ScoreNormalizer:
    """Get the default score normalizer instance."""
    global _default_normalizer
    if _default_normalizer is None:
        _default_normalizer = ScoreNormalizer()
    return _default_normalizer

def normalize_score(raw_score: float) -> Dict[str, Any]:
    """
    Convenience function to normalize a score using the default normalizer.
    
    Args:
        raw_score: Raw scale score from the ordinal model
        
    Returns:
        Normalization result dictionary
    """
    return get_normalizer().normalize_score(raw_score) 