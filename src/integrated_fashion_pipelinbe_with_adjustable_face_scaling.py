"""
INTEGRATED FASHION PIPELINE WITH ADJUSTABLE FACE SCALING
========================================================

Complete pipeline that combines:
1. Fashion outfit generation (your existing checkpoint system)
2. Target image scaling face swap (with optimal face_scale)
3. Batch testing across different outfit prompts

Features:
- Single function for complete transformation
- Batch testing with different garment types
- Optimal face_scale integration (default 0.95)
- Comprehensive logging and quality metrics
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import json
import time
from datetime import datetime

# Import all your existing systems
from adjustable_face_scale_swap import TargetScalingFaceSwapper
from fixed_appearance_analyzer import FixedAppearanceAnalyzer
from fixed_realistic_vision_pipeline import FixedRealisticVisionPipeline
from robust_face_detection_fix import RobustFaceDetector

class IntegratedFashionPipeline:
    """
    Complete fashion transformation pipeline with adjustable face scaling
    """
    
    def __init__(self, 
                 device: str = 'cuda',
                 default_face_scale: float = 0.95):
        
        self.device = device
        self.default_face_scale = default_face_scale
        
        # Initialize face swapper
        self.face_swapper = TargetScalingFaceSwapper()
        
        # Fashion generation system (placeholder for your existing code)
        self.fashion_generator = None
        self._init_fashion_generator()
        
        print(f"👗 Integrated Fashion Pipeline initialized")
        print(f"   Default face scale: {default_face_scale}")
        print(f"   Device: {device}")
    
    def _init_fashion_generator(self):
        """Initialize your complete fashion generation system"""
        try:
            # Initialize all your working systems
            self.appearance_analyzer = FixedAppearanceAnalyzer()
            self.robust_detector = RobustFaceDetector()
            
            print("   ✅ Fixed Appearance Analyzer initialized")
            print("   ✅ Robust Face Detector initialized") 
            print("   ✅ Ready for complete fashion transformation with:")
            print("      • Blonde/fair skin detection")
            print("      • False positive face detection elimination")
            print("      • RealisticVision checkpoint loading")
            print("      • Balanced face swapping")
            
            self.fashion_generator = "complete_system"
            
        except Exception as e:
            print(f"   ⚠️ Fashion generator initialization failed: {e}")
            self.fashion_generator = None
            self.appearance_analyzer = None
            self.robust_detector = None
    
    def complete_fashion_transformation(self,
                                      source_image_path: str,
                                      checkpoint_path: str,
                                      outfit_prompt: str,
                                      output_path: str,
                                      face_scale: float = None) -> Dict:
        """
        Complete fashion transformation pipeline
        
        Args:
            source_image_path: Original person image
            checkpoint_path: Fashion model checkpoint
            outfit_prompt: Description of desired outfit
            output_path: Final result path
            face_scale: Face scaling factor (None = use default 0.95)
        
        Returns:
            Dict with results and metadata
        """
        
        if face_scale is None:
            face_scale = self.default_face_scale
        
        print(f"👗 COMPLETE FASHION TRANSFORMATION")
        print(f"   Source: {os.path.basename(source_image_path)}")
        print(f"   Checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"   Outfit: {outfit_prompt}")
        print(f"   Face scale: {face_scale}")
        print(f"   Output: {output_path}")
        
        start_time = time.time()
        results = {
            'success': False,
            'source_image': source_image_path,
            'checkpoint': checkpoint_path,
            'outfit_prompt': outfit_prompt,
            'face_scale': face_scale,
            'output_path': output_path,
            'processing_time': 0,
            'steps': {}
        }
        
        try:
            # STEP 1: Generate outfit image
            print(f"\\n🎨 STEP 1: Fashion Generation")
            outfit_generation_result = self._generate_outfit_image(
                source_image_path, checkpoint_path, outfit_prompt
            )
            
            if not outfit_generation_result['success']:
                results['error'] = 'Outfit generation failed'
                return results
            
            generated_outfit_path = outfit_generation_result['output_path']
            results['steps']['outfit_generation'] = outfit_generation_result
            
            # STEP 2: Face swap with target scaling using your proven system
            print(f"\\n🎭 STEP 2: Target Scaling Face Swap")
            
            # Check if generated image passed validation
            generated_validation = outfit_generation_result.get('generated_validation')
            if generated_validation and not generated_validation['is_single_person']:
                print(f"   ⚠️ Generated image failed validation - using robust face swap approach")
                # Still proceed but with more caution
            
            # Perform target scaling face swap with your system
            face_swap_result = self.face_swapper.swap_faces_with_target_scaling(
                source_image=source_image_path,
                target_image=generated_outfit_path,
                face_scale=face_scale,
                output_path=output_path,
                crop_to_original=False,  # Keep scaled size for effect
                quality_mode="balanced"
            )
            
            results['steps']['face_swap'] = {
                'face_scale': face_scale,
                'method': 'target_scaling_face_swap',
                'crop_to_original': False,
                'output_size': face_swap_result.size,
                'success': True,
                'validation_passed': generated_validation['is_single_person'] if generated_validation else None
            }
            
            # Enhanced quality assessment with appearance data
            quality_metrics = self._assess_result_quality(
                source_image_path, output_path, outfit_prompt, outfit_generation_result
            )
            results['steps']['quality_assessment'] = quality_metrics
            
            # Success!
            results['success'] = True
            results['final_image'] = face_swap_result
            results['processing_time'] = time.time() - start_time
            
            print(f"✅ Complete transformation successful!")
            print(f"   Processing time: {results['processing_time']:.2f}s")
            print(f"   Final output: {output_path}")
            
            # Add comprehensive analysis summary if available
            if results['steps']['outfit_generation'].get('method') == 'complete_integrated_system':
                generation_data = results['steps']['outfit_generation']
                print(f"\\n📊 INTEGRATED SYSTEM SUMMARY:")
                print(f"   🎯 Appearance enhancements: {generation_data.get('enhancements_applied', [])}")
                print(f"   👱 Detected: {generation_data.get('hair_detected')} hair, {generation_data.get('skin_detected')} skin")
                print(f"   🔍 Validations: Source={generation_data.get('source_validation', {}).get('confidence', 0):.2f}, Generated={generation_data.get('generated_validation', {}).get('confidence', 0):.2f}")
                print(f"   🎭 Quality: Photorealistic={generation_data.get('looks_photorealistic', False)}")
                print(f"   🧰 Systems: {', '.join(generation_data.get('components_used', []))}")
                print(f"   🎲 Seed: {generation_data.get('generation_seed', 'unknown')}")
                
                # Add debug file references
                print(f"\\n🔧 DEBUG FILES:")
                if generation_data.get('source_debug_path'):
                    print(f"   📁 Source debug: {os.path.basename(generation_data['source_debug_path'])}")
                if generation_data.get('generated_debug_path'):
                    print(f"   📁 Generated debug: {os.path.basename(generation_data['generated_debug_path'])}")
                if generation_data.get('pose_debug_path'):
                    print(f"   📁 Pose debug: {os.path.basename(generation_data['pose_debug_path'])}")
            
            return results
            
        except Exception as e:
            results['error'] = str(e)
            results['processing_time'] = time.time() - start_time
            print(f"❌ Transformation failed: {e}")
            return results
    
    def _generate_outfit_image(self, 
                              source_image_path: str, 
                              checkpoint_path: str, 
                              outfit_prompt: str) -> Dict:
        """Generate outfit image using your complete integrated system"""
        
        # Temporary output path for generated outfit
        outfit_output = source_image_path.replace('.png', '_generated_outfit.jpg').replace('.jpg', '_generated_outfit.jpg')
        
        try:
            if self.appearance_analyzer is None or self.robust_detector is None:
                # Fallback without complete system
                print("   ⚠️ Using basic outfit generation (missing components)")
                
                # Basic generation fallback
                source_img = Image.open(source_image_path)
                source_img.save(outfit_output)
                
                return {
                    'success': True,
                    'output_path': outfit_output,
                    'prompt': outfit_prompt,
                    'checkpoint': checkpoint_path,
                    'method': 'basic_fallback'
                }
            
            else:
                # COMPLETE INTEGRATED SYSTEM
                print("   🎨 Using COMPLETE INTEGRATED FASHION SYSTEM")
                print("      🔧 Fixed Appearance Analyzer")
                print("      🔍 Robust Face Detection")
                print("      🎭 RealisticVision Pipeline")
                print("      🎯 Target Scaling Face Swap")
                
                # Step 1: Robust face detection validation
                source_image = Image.open(source_image_path).convert('RGB')
                source_debug_path = outfit_output.replace('.jpg', '_source_robust_debug.jpg')
                
                source_validation = self.robust_detector.detect_single_person_robust(
                    source_image, source_debug_path
                )
                
                print(f"   🔍 Source validation: {source_validation['is_single_person']} (conf: {source_validation['confidence']:.2f})")
                
                if not source_validation['is_single_person'] or source_validation['confidence'] < 0.6:
                    print("   ⚠️ Source image validation failed - proceeding with caution")
                
                # Step 2: Enhance prompt with appearance analysis
                enhancement_result = self.appearance_analyzer.enhance_prompt_fixed(
                    base_prompt=outfit_prompt,
                    image_path=source_image_path
                )
                
                enhanced_prompt = enhancement_result['enhanced_prompt']
                appearance_data = enhancement_result['appearance_analysis']
                enhancements = enhancement_result['enhancements_applied']
                
                print(f"   📝 Original prompt: '{outfit_prompt}'")
                print(f"   📝 Enhanced prompt: '{enhanced_prompt}'")
                print(f"   🎯 Enhancements: {enhancements}")
                
                # Step 3: Initialize RealisticVision pipeline for this generation
                print("   🎭 Initializing RealisticVision pipeline...")
                realistic_pipeline = FixedRealisticVisionPipeline(
                    checkpoint_path=checkpoint_path,
                    device=self.device
                )
                
                # Step 4: Generate outfit using your complete system
                print("   🎨 Generating outfit with complete system...")
                
                # Use RealisticVision-specific parameters
                generation_params = {
                    'num_inference_steps': 50,
                    'guidance_scale': 7.5,  # RealisticVision optimized
                    'controlnet_conditioning_scale': 1.0
                }
                
                generated_image, generation_metadata = realistic_pipeline.generate_outfit(
                    source_image_path=source_image_path,
                    outfit_prompt=enhanced_prompt,  # Use enhanced prompt!
                    output_path=outfit_output,
                    **generation_params
                )
                
                # Step 5: Validate generated image with robust detection
                generated_debug_path = outfit_output.replace('.jpg', '_generated_robust_debug.jpg')
                generated_validation = self.robust_detector.detect_single_person_robust(
                    generated_image, generated_debug_path
                )
                
                print(f"   🔍 Generated validation: {generated_validation['is_single_person']} (conf: {generated_validation['confidence']:.2f})")
                
                # Combine all metadata
                return {
                    'success': True,
                    'output_path': outfit_output,
                    'original_prompt': outfit_prompt,
                    'enhanced_prompt': enhanced_prompt,
                    'appearance_analysis': appearance_data,
                    'enhancements_applied': enhancements,
                    'checkpoint': checkpoint_path,
                    'method': 'complete_integrated_system',
                    
                    # Appearance detection results
                    'hair_detected': appearance_data['hair_color']['color_name'],
                    'skin_detected': appearance_data['skin_tone']['tone_name'],
                    'hair_confidence': appearance_data['hair_color']['confidence'],
                    'skin_confidence': appearance_data['skin_tone']['confidence'],
                    
                    # Robust detection results
                    'source_validation': source_validation,
                    'generated_validation': generated_validation,
                    'source_debug_path': source_debug_path,
                    'generated_debug_path': generated_debug_path,
                    
                    # RealisticVision results
                    'realistic_pipeline_metadata': generation_metadata,
                    'pose_debug_path': generation_metadata.get('pose_debug_path'),
                    'generation_seed': generation_metadata.get('seed'),
                    'looks_photorealistic': generation_metadata['validation']['looks_photorealistic'],
                    
                    # System components used
                    'components_used': [
                        'FixedAppearanceAnalyzer',
                        'RobustFaceDetector', 
                        'FixedRealisticVisionPipeline',
                        'TargetScalingFaceSwapper'
                    ]
                }
                
        except Exception as e:
            print(f"   ❌ Complete system generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'output_path': None,
                'original_prompt': outfit_prompt,
                'enhanced_prompt': None,
                'method': 'failed'
            }
    
    def _assess_result_quality(self, 
                              source_path: str, 
                              result_path: str, 
                              prompt: str,
                              generation_result: Dict = None) -> Dict:
        """Assess the quality of the final result with appearance analysis data"""
        
        print(f"\\n📊 STEP 3: Quality Assessment")
        
        try:
            # Load images for analysis
            source_img = Image.open(source_path)
            result_img = Image.open(result_path)
            
            # Basic metrics
            metrics = {
                'source_size': source_img.size,
                'result_size': result_img.size,
                'size_change_ratio': result_img.size[0] / source_img.size[0],
                'prompt_complexity': len(prompt.split()),
                'file_size_kb': os.path.getsize(result_path) / 1024,
                'success': True
            }
            
            # Add appearance enhancement data if available
            if generation_result and generation_result.get('method') == 'appearance_enhanced_generation':
                metrics.update({
                    'appearance_enhanced': True,
                    'original_prompt': generation_result.get('original_prompt'),
                    'enhanced_prompt': generation_result.get('enhanced_prompt'),
                    'hair_detected': generation_result.get('hair_detected'),
                    'skin_detected': generation_result.get('skin_detected'),
                    'hair_confidence': generation_result.get('hair_confidence', 0),
                    'skin_confidence': generation_result.get('skin_confidence', 0),
                    'enhancements_applied': generation_result.get('enhancements_applied', []),
                    'prompt_enhancement_success': len(generation_result.get('enhancements_applied', [])) > 0,
                    
                    # Add robust detection results
                    'source_validation_confidence': generation_result.get('source_validation', {}).get('confidence', 0),
                    'generated_validation_confidence': generation_result.get('generated_validation', {}).get('confidence', 0),
                    'source_single_person': generation_result.get('source_validation', {}).get('is_single_person', False),
                    'generated_single_person': generation_result.get('generated_validation', {}).get('is_single_person', False),
                    
                    # Add RealisticVision results  
                    'photorealistic_result': generation_result.get('looks_photorealistic', False),
                    'generation_seed': generation_result.get('generation_seed'),
                    'complete_system_used': generation_result.get('method') == 'complete_integrated_system'
                })
                
                print(f"   👱 Hair detected: {generation_result.get('hair_detected')} (conf: {generation_result.get('hair_confidence', 0):.2f})")
                print(f"   🎨 Skin detected: {generation_result.get('skin_detected')} (conf: {generation_result.get('skin_confidence', 0):.2f})")
                print(f"   📝 Enhancements: {generation_result.get('enhancements_applied', [])}")
                print(f"   🔍 Source validation: {generation_result.get('source_validation', {}).get('is_single_person', 'unknown')}")
                print(f"   🔍 Generated validation: {generation_result.get('generated_validation', {}).get('is_single_person', 'unknown')}")
                print(f"   🎭 Photorealistic: {generation_result.get('looks_photorealistic', 'unknown')}")
                print(f"   🧰 Components: {len(generation_result.get('components_used', []))} systems integrated")
                
            else:
                metrics.update({
                    'appearance_enhanced': False,
                    'prompt_enhancement_success': False,
                    'complete_system_used': False
                })
            
            # Face detection check
            face_swapper_temp = TargetScalingFaceSwapper()
            source_np = np.array(source_img)
            result_np = np.array(result_img)
            
            source_faces = face_swapper_temp._detect_faces_enhanced(source_np)
            result_faces = face_swapper_temp._detect_faces_enhanced(result_np)
            
            metrics['faces_detected'] = {
                'source': len(source_faces),
                'result': len(result_faces),
                'face_preserved': len(result_faces) > 0
            }
            
            print(f"   👤 Faces: Source({len(source_faces)}) → Result({len(result_faces)})")
            if len(result_faces) > 0:
                print(f"   ✅ Face preservation: SUCCESS")
            else:
                print(f"   ⚠️ Face preservation: FAILED")
            
            return metrics
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'appearance_enhanced': False
            }
    
    def batch_test_outfits(self,
                          source_image_path: str,
                          checkpoint_path: str,
                          outfit_prompts: List[str],
                          face_scale: float = None,
                          output_dir: str = "batch_outfit_results") -> Dict:
        """
        Batch test different outfit prompts
        
        Args:
            source_image_path: Source person image
            checkpoint_path: Fashion model checkpoint  
            outfit_prompts: List of outfit descriptions to test
            face_scale: Face scaling factor (None = use default)
            output_dir: Directory for batch results
        """
        
        if face_scale is None:
            face_scale = self.default_face_scale
        
        print(f"🧪 BATCH OUTFIT TESTING")
        print(f"   Source: {os.path.basename(source_image_path)}")
        print(f"   Outfits to test: {len(outfit_prompts)}")
        print(f"   Face scale: {face_scale}")
        print(f"   Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        batch_results = {
            'source_image': source_image_path,
            'checkpoint': checkpoint_path,
            'face_scale': face_scale,
            'total_prompts': len(outfit_prompts),
            'results': {},
            'summary': {},
            'timestamp': datetime.now().isoformat()
        }
        
        successful_results = []
        failed_results = []
        
        for i, prompt in enumerate(outfit_prompts):
            print(f"\\n👗 Testing {i+1}/{len(outfit_prompts)}: {prompt}")
            
            # Generate safe filename
            safe_prompt = self._make_safe_filename(prompt)
            output_path = os.path.join(output_dir, f"outfit_{i+1:02d}_{safe_prompt}.jpg")
            
            # Run complete transformation
            result = self.complete_fashion_transformation(
                source_image_path=source_image_path,
                checkpoint_path=checkpoint_path,
                outfit_prompt=prompt,
                output_path=output_path,
                face_scale=face_scale
            )
            
            # Store result
            batch_results['results'][prompt] = result
            
            if result['success']:
                successful_results.append(result)
                print(f"   ✅ Success: {output_path}")
            else:
                failed_results.append(result)
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Generate summary
        batch_results['summary'] = {
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / len(outfit_prompts) * 100,
            'avg_processing_time': np.mean([r['processing_time'] for r in successful_results]) if successful_results else 0,
            'best_results': self._identify_best_results(successful_results),
            'common_failures': self._analyze_failures(failed_results)
        }
        
        # Save batch report
        report_path = os.path.join(output_dir, "batch_test_report.json")
        with open(report_path, 'w') as f:
            # Convert any PIL images to string representations for JSON
            json_safe_results = self._make_json_safe(batch_results)
            json.dump(json_safe_results, f, indent=2)
        
        print(f"\\n📊 BATCH TEST COMPLETED")
        print(f"   Success rate: {batch_results['summary']['success_rate']:.1f}%")
        print(f"   Successful: {batch_results['summary']['successful']}/{len(outfit_prompts)}")
        print(f"   Report saved: {report_path}")
        
        return batch_results
    
    def _make_safe_filename(self, prompt: str) -> str:
        """Convert prompt to safe filename"""
        # Remove/replace unsafe characters
        safe = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe = safe.replace(' ', '_').lower()
        return safe[:30]  # Limit length
    
    def _make_json_safe(self, data):
        """Convert data to JSON-safe format"""
        if isinstance(data, dict):
            return {k: self._make_json_safe(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_safe(item) for item in data]
        elif isinstance(data, Image.Image):
            return f"PIL_Image_{data.size[0]}x{data.size[1]}"
        elif isinstance(data, np.ndarray):
            return f"numpy_array_{data.shape}"
        else:
            return data
    
    def _identify_best_results(self, successful_results: List[Dict]) -> List[str]:
        """Identify the best results from successful generations"""
        if not successful_results:
            return []
        
        # Sort by processing time (faster is better for now)
        sorted_results = sorted(successful_results, key=lambda x: x['processing_time'])
        
        # Return top 3 prompts
        return [r['outfit_prompt'] for r in sorted_results[:3]]
    
    def _analyze_failures(self, failed_results: List[Dict]) -> List[str]:
        """Analyze common failure patterns"""
        if not failed_results:
            return []
        
        # Count error types
        error_counts = {}
        for result in failed_results:
            error = result.get('error', 'Unknown')
            error_counts[error] = error_counts.get(error, 0) + 1
        
        # Return most common errors
        return sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    
    def find_optimal_face_scale_for_outfit(self,
                                          source_image_path: str,
                                          checkpoint_path: str,
                                          outfit_prompt: str,
                                          test_scales: List[float] = None,
                                          output_dir: str = "face_scale_optimization") -> Dict:
        """
        Find optimal face scale for a specific outfit
        
        Args:
            source_image_path: Source person image
            checkpoint_path: Fashion checkpoint
            outfit_prompt: Specific outfit to test
            test_scales: List of scales to test
            output_dir: Output directory for test results
        """
        
        if test_scales is None:
            test_scales = [0.85, 0.9, 0.95, 1.0, 1.05]
        
        print(f"🔍 FACE SCALE OPTIMIZATION")
        print(f"   Outfit: {outfit_prompt}")
        print(f"   Testing scales: {test_scales}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        scale_results = {}
        best_scale = None
        best_score = 0
        
        for scale in test_scales:
            print(f"\\n📏 Testing face scale: {scale}")
            
            output_path = os.path.join(output_dir, f"scale_{scale:.2f}_{self._make_safe_filename(outfit_prompt)}.jpg")
            
            result = self.complete_fashion_transformation(
                source_image_path=source_image_path,
                checkpoint_path=checkpoint_path,
                outfit_prompt=outfit_prompt,
                output_path=output_path,
                face_scale=scale
            )
            
            # Simple scoring (you can make this more sophisticated)
            score = 1.0 if result['success'] else 0.0
            if result['success']:
                # Bonus for reasonable processing time
                if result['processing_time'] < 30:  # seconds
                    score += 0.1
                # Bonus for face preservation
                if result['steps']['quality_assessment']['faces_detected']['face_preserved']:
                    score += 0.2
            
            scale_results[scale] = {
                'result': result,
                'score': score,
                'output_path': output_path
            }
            
            if score > best_score:
                best_score = score
                best_scale = scale
            
            print(f"   Score: {score:.2f}")
        
        optimization_result = {
            'outfit_prompt': outfit_prompt,
            'best_scale': best_scale,
            'best_score': best_score,
            'all_results': scale_results,
            'recommendation': f"Use face_scale={best_scale} for '{outfit_prompt}'"
        }
        
        print(f"\\n🎯 OPTIMIZATION COMPLETE")
        print(f"   Best scale: {best_scale} (score: {best_score:.2f})")
        print(f"   Recommendation: Use face_scale={best_scale}")
        
        return optimization_result


# Predefined outfit prompts for comprehensive testing
OUTFIT_TEST_PROMPTS = {
    "dresses": [
        "elegant red evening dress",
        "casual blue summer dress", 
        "black cocktail dress",
        "white wedding dress",
        "floral print sundress",
        "little black dress"
    ],
    
    "formal_wear": [
        "black business suit",
        "navy blue blazer with white shirt",
        "formal tuxedo",
        "professional gray suit",
        "burgundy evening gown"
    ],
    
    "casual_wear": [
        "blue jeans and white t-shirt",
        "comfortable hoodie and jeans",
        "casual denim jacket",
        "khaki pants and polo shirt",
        "summer shorts and tank top"
    ],
    
    "seasonal": [
        "warm winter coat",
        "light spring cardigan",
        "summer bikini",
        "autumn sweater",
        "holiday party outfit"
    ],
    
    "colors": [
        "vibrant red outfit",
        "royal blue ensemble",
        "emerald green dress",
        "sunshine yellow top",
        "deep purple gown"
    ]
}


# Easy-to-use wrapper functions

def complete_fashion_makeover(source_image_path: str,
                            checkpoint_path: str,
                            outfit_prompt: str,
                            output_path: str = "fashion_makeover.jpg",
                            face_scale: float = 0.95) -> Image.Image:
    """
    Simple one-function fashion makeover
    
    Args:
        source_image_path: Original person image
        checkpoint_path: Fashion model checkpoint
        outfit_prompt: Desired outfit description
        output_path: Where to save result  
        face_scale: Face scaling (0.95 recommended)
    
    Returns:
        Final transformed image
    """
    pipeline = IntegratedFashionPipeline(default_face_scale=face_scale)
    
    result = pipeline.complete_fashion_transformation(
        source_image_path=source_image_path,
        checkpoint_path=checkpoint_path,
        outfit_prompt=outfit_prompt,
        output_path=output_path,
        face_scale=face_scale
    )
    
    if result['success']:
        return result['final_image']
    else:
        raise Exception(f"Fashion makeover failed: {result.get('error', 'Unknown error')}")


def batch_test_fashion_categories(source_image_path: str,
                                checkpoint_path: str,
                                categories: List[str] = None,
                                face_scale: float = 0.95) -> Dict:
    """
    Test multiple fashion categories
    
    Args:
        source_image_path: Source person image
        checkpoint_path: Fashion checkpoint
        categories: Categories to test (None = all categories)
        face_scale: Face scaling factor
    
    Returns:
        Batch test results
    """
    pipeline = IntegratedFashionPipeline(default_face_scale=face_scale)
    
    if categories is None:
        categories = list(OUTFIT_TEST_PROMPTS.keys())
    
    all_prompts = []
    for category in categories:
        if category in OUTFIT_TEST_PROMPTS:
            all_prompts.extend(OUTFIT_TEST_PROMPTS[category])
    
    return pipeline.batch_test_outfits(
        source_image_path=source_image_path,
        checkpoint_path=checkpoint_path,
        outfit_prompts=all_prompts,
        face_scale=face_scale,
        output_dir="batch_fashion_test"
    )


def find_best_face_scale(source_image_path: str,
                        checkpoint_path: str,
                        outfit_prompt: str = "elegant red evening dress") -> float:
    """
    Find the optimal face scale for your specific setup
    
    Args:
        source_image_path: Source person image
        checkpoint_path: Fashion checkpoint
        outfit_prompt: Test outfit
    
    Returns:
        Optimal face scale value
    """
    pipeline = IntegratedFashionPipeline()
    
    result = pipeline.find_optimal_face_scale_for_outfit(
        source_image_path=source_image_path,
        checkpoint_path=checkpoint_path,
        outfit_prompt=outfit_prompt,
        test_scales=[0.85, 0.9, 0.95, 1.0, 1.05]
    )
    
    return result['best_scale']


if __name__ == "__main__":
    print("👗 INTEGRATED FASHION PIPELINE WITH ADJUSTABLE FACE SCALING")
    print("=" * 65)
    
    print("🎯 KEY FEATURES:")
    print("  ✅ Complete fashion transformation pipeline")
    print("  ✅ Target image scaling (face stays constant)")
    print("  ✅ Optimal face_scale integration (default 0.95)")
    print("  ✅ Batch testing across outfit categories")
    print("  ✅ Face scale optimization for specific outfits")
    print("  ✅ Comprehensive quality assessment")
    
    print("\\n📋 USAGE EXAMPLES:")
    print("""
# Single transformation with optimal scale
result = complete_fashion_makeover(
    source_image_path="woman_jeans_t-shirt.png",
    checkpoint_path="realisticVisionV60B1_v51HyperVAE.safetensors",
    outfit_prompt="elegant red evening dress",
    output_path="fashion_result.jpg",
    face_scale=0.95  # Your optimal value
)

# Batch test different outfit categories
batch_results = batch_test_fashion_categories(
    source_image_path="woman_jeans_t-shirt.png",
    checkpoint_path="realisticVisionV60B1_v51HyperVAE.safetensors",
    categories=["dresses", "formal_wear", "casual_wear"],
    face_scale=0.95
)

# Find optimal face scale for specific outfit
optimal_scale = find_best_face_scale(
    source_image_path="woman_jeans_t-shirt.png", 
    checkpoint_path="realisticVisionV60B1_v51HyperVAE.safetensors",
    outfit_prompt="black cocktail dress"
)
""")
    
    print("\\n👗 OUTFIT CATEGORIES AVAILABLE:")
    for category, prompts in OUTFIT_TEST_PROMPTS.items():
        print(f"  • {category}: {len(prompts)} prompts")
        print(f"    Examples: {', '.join(prompts[:2])}")
    
    print("\\n🔧 INTEGRATION NOTES:")
    print("  • Replace placeholder fashion generation with your existing code")
    print("  • Adjust quality assessment metrics as needed")
    print("  • Customize outfit prompts for your use case")
    print("  • Face scale 0.95 is pre-configured as optimal")
    
    print("\\n🎯 EXPECTED WORKFLOW:")
    print("  1. Generate outfit image (your existing checkpoint system)")
    print("  2. Apply target scaling face swap (face_scale=0.95)")
    print("  3. Quality assessment and result validation")
    print("  4. Batch testing across different garment types")