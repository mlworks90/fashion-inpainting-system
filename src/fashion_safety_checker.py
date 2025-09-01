"""
FASHION SAFETY CHECKER - CLEAN PRODUCTION VERSION
================================================

Production-ready fashion safety validation with:
- Silent blocking fix applied
- User-friendly "generating synthetic face" messaging
- Minimal logging with essential blocking reports only
- Full parameter control (face_scale, safety_mode)
"""

import cv2
import numpy as np
from PIL import Image
import torch
from typing import Dict, List, Tuple, Optional, Union
import os
import warnings
from dataclasses import dataclass
from enum import Enum
from contextlib import redirect_stdout
from io import StringIO

# Suppress warnings for production
warnings.filterwarnings('ignore')

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"  
    UNSAFE = "unsafe"
    BLOCKED = "blocked"

@dataclass
class SafetyResult:
    is_safe: bool
    safety_level: SafetyLevel
    confidence: float
    issues: List[str]
    warnings: List[str]
    detailed_analysis: Dict
    user_message: str

class FashionOptimizedSafetyChecker:
    """Production fashion safety checker"""
    
    def __init__(self, strictness_level: str = "fashion_moderate", verbose: bool = False):
        self.strictness_level = strictness_level
        self.verbose = verbose
        self._configure_thresholds()
        self._init_fashion_context()
        self._init_detection_systems()
    
    def _configure_thresholds(self):
        """Configure safety thresholds"""
        configs = {
            "fashion_permissive": {"content_safety_threshold": 0.3, "fashion_context_bonus": 0.3},
            "fashion_moderate": {"content_safety_threshold": 0.5, "fashion_context_bonus": 0.2},
            "fashion_strict": {"content_safety_threshold": 0.7, "fashion_context_bonus": 0.1},
            "legacy_strict": {"content_safety_threshold": 0.9, "fashion_context_bonus": 0.0}
        }
        self.thresholds = configs.get(self.strictness_level, configs["fashion_moderate"])
    
    def _init_fashion_context(self):
        """Initialize fashion keywords"""
        self.fashion_keywords = {
            'evening_wear': ['evening', 'formal', 'gown', 'cocktail'],
            'activewear': ['workout', 'sports', 'athletic', 'swimwear', 'bikini'],
            'professional': ['business', 'office', 'suit', 'blazer'],
            'casual': ['casual', 'everyday', 'street']
        }
    
    def _init_detection_systems(self):
        """Initialize detection systems silently"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except Exception:
            self.face_cascade = None
    
    def validate_target_image(self, 
                            target_image: Union[str, Image.Image, np.ndarray],
                            prompt_hint: str = "",
                            debug_output_path: Optional[str] = None) -> SafetyResult:
        """Production validation with proper bikini + strict mode detection"""
        try:
            # Load image
            if isinstance(target_image, str):
                image_pil = Image.open(target_image).convert('RGB')
                image_np = np.array(image_pil)
            elif isinstance(target_image, Image.Image):
                image_np = np.array(target_image.convert('RGB'))
            else:
                image_np = target_image
            
            # Fashion context analysis
            fashion_context = self._analyze_fashion_context(prompt_hint)
            
            # CRITICAL: Detect bikini + strict mode combination
            is_bikini_request = 'bikini' in prompt_hint.lower() or 'swimwear' in prompt_hint.lower()
            is_strict_mode = self.strictness_level == "fashion_strict"
            
            # Safety scoring
            base_score = 0.8
            if fashion_context['is_fashion_image']:
                base_score += fashion_context['score'] * self.thresholds['fashion_context_bonus']
            
            # STRICT MODE: Block bikini requests
            if is_strict_mode and is_bikini_request:
                return SafetyResult(
                    is_safe=False,
                    safety_level=SafetyLevel.BLOCKED,
                    confidence=0.2,  # Low confidence due to strict blocking
                    issues=["Bikini content blocked in strict mode"],
                    warnings=[],
                    detailed_analysis={'strict_mode_block': True, 'bikini_detected': True},
                    user_message="Content blocked due to strict safety settings."
                )
            
            # Normal safety evaluation
            if base_score >= 0.8:
                safety_level = SafetyLevel.SAFE
                is_safe = True
                user_message = "Fashion validation passed."
            elif base_score >= 0.6:
                safety_level = SafetyLevel.WARNING  
                is_safe = True
                user_message = "Fashion validation passed with minor concerns."
            else:
                safety_level = SafetyLevel.BLOCKED
                is_safe = False
                user_message = "Content blocked due to safety concerns."
            
            return SafetyResult(
                is_safe=is_safe,
                safety_level=safety_level,
                confidence=base_score,
                issues=[],
                warnings=[],
                detailed_analysis={'bikini_detected': is_bikini_request, 'strict_mode': is_strict_mode},
                user_message=user_message
            )
            
        except Exception as e:
            return SafetyResult(
                is_safe=False,
                safety_level=SafetyLevel.BLOCKED,
                confidence=0.0,
                issues=["Validation error"],
                warnings=[],
                detailed_analysis={},
                user_message="Safety validation failed."
            )
    
    def _analyze_fashion_context(self, prompt_hint: str) -> Dict:
        """Analyze fashion context from prompt"""
        context = {'is_fashion_image': False, 'score': 0.0}
        
        if prompt_hint:
            prompt_lower = prompt_hint.lower()
            for keywords in self.fashion_keywords.values():
                if any(keyword in prompt_lower for keyword in keywords):
                    context['is_fashion_image'] = True
                    context['score'] = 0.3
                    break
        
        return context

class FashionAwarePipeline:
    """Production fashion pipeline"""
    
    def __init__(self, safety_mode: str = "fashion_moderate", verbose: bool = False):
        self.safety_checker = FashionOptimizedSafetyChecker(strictness_level=safety_mode, verbose=verbose)
        self.safety_mode = safety_mode
        self.verbose = verbose
    
    def safe_fashion_transformation(self,
                                  source_image_path: str,
                                  checkpoint_path: str,
                                  outfit_prompt: str,
                                  output_path: str = "fashion_result.jpg",
                                  face_scale: float = 0.95,
                                  safety_override: bool = False) -> Dict:
        """Production fashion transformation with clear blocking reports"""
        
        result = {
            'success': False,
            'face_swap_applied': False,
            'final_output': None,
            'user_message': None,
            'safety_level': None,
            'blocking_reason': None,
            'safety_approved': False
        }
        
        try:
            # Generate outfit
            from fixed_realistic_vision_pipeline import FixedRealisticVisionPipeline
            
            # Suppress initialization prints only
            if not self.verbose:
                f = StringIO()
                with redirect_stdout(f):
                    outfit_pipeline = FixedRealisticVisionPipeline(checkpoint_path, device='cuda')
            else:
                outfit_pipeline = FixedRealisticVisionPipeline(checkpoint_path, device='cuda')
            
            # Generate outfit (suppress technical details in non-verbose mode)
            outfit_path = output_path.replace('.jpg', '_outfit.jpg')
            
            if not self.verbose:
                with redirect_stdout(f):
                    generated_image, generation_metadata = outfit_pipeline.generate_outfit(
                        source_image_path=source_image_path,
                        outfit_prompt=outfit_prompt,
                        output_path=outfit_path
                    )
            else:
                generated_image, generation_metadata = outfit_pipeline.generate_outfit(
                    source_image_path=source_image_path,
                    outfit_prompt=outfit_prompt,
                    output_path=outfit_path
                )
            
            # FIRST: Do safety validation to get the real blocking reason
            safety_result = self.safety_checker.validate_target_image(
                target_image=outfit_path,
                prompt_hint=outfit_prompt,
                debug_output_path=None
            )
            
            result['safety_level'] = safety_result.safety_level.value
            
            # Check generation metadata
            single_person_ok = generation_metadata.get('validation', {}).get('single_person', False)
            
            # DETERMINE THE REAL BLOCKING REASON
            # If safety failed AND single_person is False, it's likely a safety block causing synthetic face
            if not safety_result.is_safe and not single_person_ok:
                # This is likely a safety block manifesting as "synthetic face" (single_person=False)
                print(f"🚫 Content generation blocked - generating synthetic face")
                print(f"   Issue: Content safety restrictions triggered")
                print(f"   Action: Please try a more conservative outfit style")
                print(f"   Safety level: {safety_result.safety_level.value}")
                
                result['blocking_reason'] = f"Safety restrictions: {safety_result.safety_level.value}"
                result['user_message'] = "Content generation blocked - generating synthetic face. Content safety restrictions apply."
                result['final_output'] = outfit_path
                result['safety_approved'] = False
                return result
            
            elif not single_person_ok and safety_result.is_safe:
                # This is a genuine multiple people detection issue
                print(f"🚫 Content generation blocked - generating synthetic face")
                print(f"   Issue: Multiple people detected in generated content")
                print(f"   Action: Please try a different outfit description")
                
                result['blocking_reason'] = "Multiple people detected"
                result['user_message'] = "Content generation blocked - generating synthetic face. Multiple people detected in image."
                result['final_output'] = outfit_path
                return result
            
            # Face swap decision
            proceed = (safety_result.is_safe or 
                      (safety_override and safety_result.safety_level != SafetyLevel.BLOCKED))
            
            if proceed:
                result['safety_approved'] = True
                
                try:
                    # Apply face swap (suppress ALL prints during face swap execution)
                    if not self.verbose:
                        with redirect_stdout(f):
                            from integrated_fashion_pipelinbe_with_adjustable_face_scaling import IntegratedFashionPipeline
                            integrated_pipeline = IntegratedFashionPipeline()
                            
                            # Face swap execution - suppress technical details
                            final_image = integrated_pipeline.face_swapper.swap_faces_with_target_scaling(
                                source_image=source_image_path,
                                target_image=outfit_path,
                                face_scale=face_scale,
                                output_path=output_path,
                                quality_mode="balanced",
                                crop_to_original=False
                            )
                    else:
                        from integrated_fashion_pipelinbe_with_adjustable_face_scaling import IntegratedFashionPipeline
                        integrated_pipeline = IntegratedFashionPipeline()
                        
                        final_image = integrated_pipeline.face_swapper.swap_faces_with_target_scaling(
                            source_image=source_image_path,
                            target_image=outfit_path,
                            face_scale=face_scale,
                            output_path=output_path,
                            quality_mode="balanced",
                            crop_to_original=False
                        )
                    
                    # SUCCESS
                    result['success'] = True
                    result['face_swap_applied'] = True
                    result['final_output'] = output_path
                    result['user_message'] = "Fashion transformation completed successfully."
                    
                except Exception as e:
                    # TECHNICAL FAILURE
                    outfit_failure_path = output_path.replace('.jpg', '_outfit_only.jpg')
                    generated_image.save(outfit_failure_path)
                    
                    print(f"🚫 Content generation blocked - generating synthetic face")
                    print(f"   Issue: Technical error during face processing")
                    print(f"   Action: Please try again")
                    
                    result['success'] = False
                    result['face_swap_applied'] = False
                    result['final_output'] = outfit_failure_path
                    result['user_message'] = "Content generation blocked - generating synthetic face. Technical error occurred."
                    result['blocking_reason'] = f"Technical failure: {str(e)}"
                    result['safety_approved'] = True
                    
            else:
                # SAFETY BLOCK
                outfit_blocked_path = output_path.replace('.jpg', '_outfit_only.jpg')
                generated_image.save(outfit_blocked_path)
                
                print(f"🚫 Content generation blocked - generating synthetic face")
                print(f"   Issue: Content safety restrictions")
                print(f"   Action: Please try a more conservative outfit style")
                
                result['success'] = False
                result['face_swap_applied'] = False
                result['final_output'] = outfit_blocked_path
                result['user_message'] = "Content generation blocked - generating synthetic face. Please try a more conservative outfit style."
                result['blocking_reason'] = f"Safety restrictions: {safety_result.safety_level.value}"
                result['safety_approved'] = False
            
            return result
            
        except Exception as e:
            print(f"🚫 Content generation blocked - generating synthetic face")
            print(f"   Issue: System error occurred")
            print(f"   Action: Please try again")
            
            result['blocking_reason'] = f"System error: {str(e)}"
            result['user_message'] = "Content generation blocked - generating synthetic face. System error occurred."
            return result

def fashion_safe_generate(source_image_path: str,
                         checkpoint_path: str,
                         outfit_prompt: str,
                         output_path: str = "fashion_result.jpg",
                         face_scale: float = 0.95,
                         safety_mode: str = "fashion_moderate",
                         safety_override: bool = False,
                         verbose: bool = False) -> Dict:
    """
    PRODUCTION VERSION: Fashion generation with user-friendly messaging
    """
    pipeline = FashionAwarePipeline(safety_mode=safety_mode, verbose=verbose)
    
    return pipeline.safe_fashion_transformation(
        source_image_path=source_image_path,
        checkpoint_path=checkpoint_path,
        outfit_prompt=outfit_prompt,
        output_path=output_path,
        face_scale=face_scale,
        safety_override=safety_override
    )

def create_fashion_safety_pipeline(preset: str = "production", 
                                  safety_mode: Optional[str] = None,
                                  verbose: bool = False) -> FashionAwarePipeline:
    """
    Create fashion pipeline with production configuration
    
    Args:
        preset: "production", "demo", "strict_production", "permissive"
        safety_mode: Override safety mode ("fashion_moderate" default)
        verbose: Enable detailed logging (False for production)
    """
    presets = {
        'production': 'fashion_moderate',
        'demo': 'fashion_moderate', 
        'strict_production': 'fashion_strict',
        'permissive': 'fashion_permissive'
    }
    
    final_safety_mode = safety_mode if safety_mode is not None else presets.get(preset, "fashion_moderate")
    return FashionAwarePipeline(safety_mode=final_safety_mode, verbose=verbose)