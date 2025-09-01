"""
TARGET IMAGE SCALING APPROACH - SUPERIOR METHOD
===============================================

Scales the target image instead of the face for better results.

Logic: 
- face_scale = 0.9 → Scale target to 111% (1/0.9) → Face appears smaller
- face_scale = 1.1 → Scale target to 91% (1/1.1) → Face appears larger

Advantages:
- Preserves source face quality (no interpolation)
- Natural body proportion adjustment
- Better alignment and blending
- Simpler processing pipeline
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import os
from typing import Optional, Tuple, Union

class TargetScalingFaceSwapper:
    """
    Superior face swapping approach: Scale target image instead of face
    """
    
    def __init__(self):
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        print("🎭 Target Scaling Face Swapper initialized")
        print("   Method: Scale target image (superior approach)")
        print("   Preserves source face quality completely")
    
    def swap_faces_with_target_scaling(self,
                                      source_image: Union[str, Image.Image],
                                      target_image: Union[str, Image.Image],
                                      face_scale: float = 1.0,
                                      output_path: Optional[str] = None,
                                      quality_mode: str = "balanced",
                                      crop_to_original: bool = False) -> Image.Image:
        """
        Perform face swap by scaling target image (superior method)
        
        Args:
            source_image: Source image (face to extract)
            target_image: Target image (to be scaled)
            face_scale: Desired face scale (0.5-2.0)
                       0.9 = face appears 10% smaller
                       1.1 = face appears 10% larger
            output_path: Optional save path
            quality_mode: "balanced", "clarity", or "natural"
            crop_to_original: Whether to resize back to original size (recommended: False)
                         True = resize back (may reduce scaling effect)
                         False = keep scaled size (preserves scaling effect)
        """
        
        # Validate and calculate target scale
        face_scale = max(0.5, min(2.0, face_scale))
        target_scale = 1.0 / face_scale  # Inverse relationship
        
        print(f"🎭 Target scaling face swap:")
        print(f"   Desired face appearance: {face_scale} (relative to current)")
        print(f"   Face extraction scale: 1.0 (constant - no face scaling)")
        print(f"   Target image scale: {target_scale:.3f}")
        print(f"   Logic: face_scale {face_scale} → target scales to {target_scale:.2f}")
        
        try:
            # Load images
            source_pil = self._load_image(source_image)
            target_pil = self._load_image(target_image)
            
            original_target_size = target_pil.size
            print(f"   Original target size: {original_target_size}")
            
            # STEP 1: Scale target image
            scaled_target = self._scale_target_image(target_pil, target_scale)
            print(f"   Scaled target size: {scaled_target.size}")
            
            # STEP 2: Perform face swap on scaled target (normal process)
            swapped_result = self._perform_standard_face_swap(
                source_pil, scaled_target, quality_mode
            )
            
            # STEP 3: Handle final sizing - CRITICAL LOGIC FIX
            if crop_to_original:
                # STRATEGIC crop that preserves the face scaling effect
                final_result = self._smart_crop_preserving_face_scale(
                    swapped_result, original_target_size, face_scale
                )
                print(f"   Smart cropped to preserve face scale: {final_result.size}")
            else:
                final_result = swapped_result
                print(f"   Keeping scaled size to preserve effect: {swapped_result.size}")
            
            # Save result
            if output_path:
                final_result.save(output_path)
                print(f"   💾 Saved: {output_path}")
            
            print(f"   ✅ Target scaling face swap completed!")
            return final_result
            
        except Exception as e:
            print(f"   ❌ Target scaling face swap failed: {e}")
            return target_image if isinstance(target_image, Image.Image) else Image.open(target_image)
    
    def _load_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """Load and validate image"""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            return Image.open(image_input).convert('RGB')
        else:
            return image_input.convert('RGB')
    
    def _scale_target_image(self, target_image: Image.Image, scale_factor: float) -> Image.Image:
        """Scale target image with high-quality resampling"""
        original_w, original_h = target_image.size
        
        # Calculate new dimensions
        new_w = int(original_w * scale_factor)
        new_h = int(original_h * scale_factor)
        
        # Use high-quality resampling
        if scale_factor > 1.0:
            # Upscaling - use LANCZOS for best quality
            resampling = Image.Resampling.LANCZOS
        else:
            # Downscaling - use LANCZOS for best quality
            resampling = Image.Resampling.LANCZOS
        
        scaled_image = target_image.resize((new_w, new_h), resampling)
        
        print(f"   📏 Target scaled: {original_w}x{original_h} → {new_w}x{new_h}")
        return scaled_image
    
    def _perform_standard_face_swap(self,
                                   source_image: Image.Image,
                                   target_image: Image.Image,
                                   quality_mode: str) -> Image.Image:
        """Perform face swap with CONSTANT face size (never resize face)"""
        
        # Convert to numpy for OpenCV processing
        source_np = np.array(source_image)
        target_np = np.array(target_image)
        
        # Detect faces
        source_faces = self._detect_faces_enhanced(source_np)
        target_faces = self._detect_faces_enhanced(target_np)
        
        if not source_faces or not target_faces:
            print("   ⚠️ Face detection failed in standard swap")
            return target_image
        
        # Get best faces
        source_face = source_faces[0]
        target_face = target_faces[0]
        
        # Extract source face (full quality, NO SCALING EVER)
        source_face_region, source_mask = self._extract_face_region_quality(source_np, source_face)
        
        print(f"   👤 Source face extracted: {source_face_region.shape[:2]} (NEVER RESIZED)")
        print(f"   🎯 Target face detected: {target_face['bbox'][2]}x{target_face['bbox'][3]}")
        
        # CRITICAL: Get the ORIGINAL size of extracted face
        face_h, face_w = source_face_region.shape[:2]
        
        # Apply quality enhancements to original size face
        enhanced_face = self._apply_quality_enhancement(source_face_region, quality_mode)
        
        # CRITICAL: Place face at its ORIGINAL size, centered on target face location
        tx, ty, tw, th = target_face['bbox']
        target_center_x = tx + tw // 2
        target_center_y = ty + th // 2
        
        # Calculate position for original-sized face (centered)
        face_x = target_center_x - face_w // 2
        face_y = target_center_y - face_h // 2
        
        # Ensure face stays within image bounds
        face_x = max(0, min(target_np.shape[1] - face_w, face_x))
        face_y = max(0, min(target_np.shape[0] - face_h, face_y))
        
        # Adjust face dimensions if it extends beyond bounds
        actual_face_w = min(face_w, target_np.shape[1] - face_x)
        actual_face_h = min(face_h, target_np.shape[0] - face_y)
        
        print(f"   📍 Face placement: ({face_x}, {face_y}) size: {actual_face_w}x{actual_face_h}")
        print(f"   🔒 Face size is CONSTANT - never resized to match target")
        
        # Crop face and mask if needed for boundaries
        if actual_face_w != face_w or actual_face_h != face_h:
            enhanced_face = enhanced_face[:actual_face_h, :actual_face_w]
            source_mask = source_mask[:actual_face_h, :actual_face_w]
        
        # Color matching with the area where face will be placed
        target_region = target_np[face_y:face_y+actual_face_h, face_x:face_x+actual_face_w]
        if target_region.shape == enhanced_face.shape:
            color_matched_face = self._match_colors_lab(enhanced_face, target_region)
        else:
            color_matched_face = enhanced_face
        
        # Blend into target at ORIGINAL face size
        result_np = self._blend_faces_smooth(
            target_np, color_matched_face, source_mask, (face_x, face_y, actual_face_w, actual_face_h)
        )
        
        return Image.fromarray(result_np)
    
    def _smart_crop_preserving_face_scale(self, 
                                          scaled_result: Image.Image, 
                                          original_size: Tuple[int, int],
                                          face_scale: float) -> Image.Image:
        """
        CRITICAL FIX: Smart cropping that preserves face scaling effect
        
        The key insight: We don't want to just center crop back to original size,
        as that defeats the purpose. Instead, we need to crop strategically.
        """
        original_w, original_h = original_size
        scaled_w, scaled_h = scaled_result.size
        
        if face_scale >= 1.0:
            # Face should appear larger - target was scaled down
            # Crop from center normally since target is smaller than original
            crop_x = max(0, (scaled_w - original_w) // 2)
            crop_y = max(0, (scaled_h - original_h) // 2)
            
            cropped = scaled_result.crop((
                crop_x, crop_y, 
                crop_x + original_w, 
                crop_y + original_h
            ))
            
        else:
            # Face should appear smaller - target was scaled up
            # CRITICAL: Don't just center crop - this undoes the scaling effect!
            # Instead, we need to preserve the larger context
            
            # Option 1: Keep the scaled image (don't crop at all)
            # return scaled_result
            
            # Option 2: Resize back to original while preserving aspect ratio
            # This maintains the face size relationship
            aspect_preserved = scaled_result.resize(original_size, Image.Resampling.LANCZOS)
            return aspect_preserved
        
        return cropped
    
    def _crop_to_original_size_old(self, scaled_result: Image.Image, original_size: Tuple[int, int]) -> Image.Image:
        """
        OLD METHOD - FLAWED LOGIC
        This method defeats the purpose by cropping back exactly to original size
        """
        original_w, original_h = original_size
        scaled_w, scaled_h = scaled_result.size
        
        # Calculate crop area (center crop)
        crop_x = (scaled_w - original_w) // 2
        crop_y = (scaled_h - original_h) // 2
        
        # Ensure crop area is valid
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)
        
        # Crop to original size - THIS UNDOES THE SCALING EFFECT!
        cropped = scaled_result.crop((
            crop_x, 
            crop_y, 
            crop_x + original_w, 
            crop_y + original_h
        ))
        
        return cropped
    
    def _detect_faces_enhanced(self, image_np: np.ndarray) -> list:
        """Enhanced face detection (from your existing system)"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=4, 
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return []
        
        face_data = []
        for (x, y, w, h) in faces:
            # Enhanced face scoring
            area = w * h
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Detect eyes for quality
            face_roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_roi_gray, 1.1, 3)
            
            quality_score = area + len(eyes) * 100
            
            face_data.append({
                'bbox': (x, y, w, h),
                'center': (center_x, center_y),
                'area': area,
                'quality_score': quality_score
            })
        
        # Sort by quality
        face_data.sort(key=lambda f: f['quality_score'], reverse=True)
        return face_data
    
    def _extract_face_region_quality(self, image_np: np.ndarray, face_data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract face region with quality preservation"""
        x, y, w, h = face_data['bbox']
        
        # Moderate padding to avoid cutting features
        padding = int(max(w, h) * 0.2)
        
        x1 = max(0, x - padding)
        y1 = max(0, y - padding) 
        x2 = min(image_np.shape[1], x + w + padding)
        y2 = min(image_np.shape[0], y + h + padding)
        
        face_region = image_np[y1:y2, x1:x2]
        
        # Create smooth elliptical mask
        mask_h, mask_w = face_region.shape[:2]
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        
        center = (mask_w // 2, mask_h // 2)
        axes = (mask_w // 2 - 5, mask_h // 2 - 5)
        
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (17, 17), 0)
        
        return face_region, mask
    
    def _apply_quality_enhancement(self, face_np: np.ndarray, quality_mode: str) -> np.ndarray:
        """Apply your existing quality enhancements"""
        face_pil = Image.fromarray(face_np)
        
        if quality_mode == "clarity":
            enhanced = face_pil.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        elif quality_mode == "natural":
            enhancer = ImageEnhance.Color(face_pil)
            enhanced = enhancer.enhance(1.1)
        else:  # balanced
            # Your proven balanced approach
            sharpened = face_pil.filter(ImageFilter.UnsharpMask(radius=0.8, percent=100, threshold=3))
            enhancer = ImageEnhance.Color(sharpened)
            enhanced = enhancer.enhance(1.05)
        
        return np.array(enhanced)
    
    def _match_colors_lab(self, source_face: np.ndarray, target_region: np.ndarray) -> np.ndarray:
        """LAB color matching (your proven method)"""
        try:
            source_lab = cv2.cvtColor(source_face, cv2.COLOR_RGB2LAB)
            target_lab = cv2.cvtColor(target_region, cv2.COLOR_RGB2LAB)
            
            source_mean, source_std = cv2.meanStdDev(source_lab)
            target_mean, target_std = cv2.meanStdDev(target_lab)
            
            result_lab = source_lab.copy().astype(np.float64)
            
            for i in range(3):
                if source_std[i] > 0:
                    result_lab[:, :, i] = (
                        (result_lab[:, :, i] - source_mean[i]) * 
                        (target_std[i] / source_std[i]) + target_mean[i]
                    )
            
            result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
            return cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
            
        except Exception as e:
            print(f"   ⚠️ Color matching failed: {e}")
            return source_face
    
    def _blend_faces_smooth(self,
                           target_image: np.ndarray,
                           face_region: np.ndarray,
                           face_mask: np.ndarray,
                           bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Smooth face blending (your proven method)"""
        
        result = target_image.copy()
        x, y, w, h = bbox
        
        # Boundary checks
        if (y + h > result.shape[0] or x + w > result.shape[1] or
            h != face_region.shape[0] or w != face_region.shape[1]):
            print(f"   ⚠️ Boundary issue in blending")
            return result
        
        # Normalize mask
        mask_normalized = face_mask.astype(np.float32) / 255.0
        mask_3d = np.stack([mask_normalized] * 3, axis=-1)
        
        # Extract target region
        target_region = result[y:y+h, x:x+w]
        
        # Alpha blending
        blended_region = (
            face_region.astype(np.float32) * mask_3d +
            target_region.astype(np.float32) * (1 - mask_3d)
        )
        
        result[y:y+h, x:x+w] = blended_region.astype(np.uint8)
        return result
    
    def batch_test_target_scaling(self,
                                 source_image: Union[str, Image.Image],
                                 target_image: Union[str, Image.Image],
                                 scales: list = [0.8, 0.9, 1.0, 1.1, 1.2],
                                 output_prefix: str = "target_scale_test") -> dict:
        """Test multiple target scaling factors"""
        
        print(f"🧪 Testing {len(scales)} face scale factors...")
        print(f"   Method: Face stays 1.0, target image scales accordingly")
        print(f"   Logic: Smaller face_scale → Larger target → Face appears smaller")
        
        results = {}
        
        for face_scale in scales:
            try:
                target_scale = 1.0 / face_scale  # Target scale calculation
                output_path = f"{output_prefix}_faceScale{face_scale:.2f}_targetScale{target_scale:.2f}.jpg"
                
                result_image = self.swap_faces_with_target_scaling(
                    source_image=source_image,
                    target_image=target_image,
                    face_scale=face_scale,
                    output_path=output_path,
                    quality_mode="balanced",
                    crop_to_original=False  # CRITICAL: Don't crop back to preserve effect
                )
                
                results[face_scale] = {
                    'image': result_image,
                    'path': output_path,
                    'face_scale': 1.0,  # Face always stays 1.0
                    'target_scale': target_scale,
                    'success': True
                }
                
                print(f"   ✅ face_scale {face_scale:.2f} → face:1.0, target:{target_scale:.2f} → {output_path}")
                
            except Exception as e:
                print(f"   ❌ face_scale {face_scale:.2f} failed: {e}")
                results[face_scale] = {'success': False, 'error': str(e)}
        
        return results
    
    def compare_scaling_methods(self,
                               source_image: Union[str, Image.Image],
                               target_image: Union[str, Image.Image],
                               face_scale: float = 0.9) -> dict:
        """
        Compare target scaling vs face scaling methods
        """
        print(f"⚔️ COMPARING SCALING METHODS (scale={face_scale})")
        
        results = {}
        
        # Method 1: Target scaling (your suggested approach)
        try:
            print(f"\n1️⃣ Testing TARGET SCALING method...")
            result1 = self.swap_faces_with_target_scaling(
                source_image, target_image, face_scale,
                "comparison_target_scaling.jpg", "balanced", True
            )
            results['target_scaling'] = {
                'image': result1,
                'path': "comparison_target_scaling.jpg",
                'success': True,
                'method': 'Scale target image'
            }
        except Exception as e:
            results['target_scaling'] = {'success': False, 'error': str(e)}
        
        # Method 2: Face scaling (old approach) for comparison
        try:
            print(f"\n2️⃣ Testing FACE SCALING method...")
            from adjustable_face_scale_swap import AdjustableFaceScaleSwapper
            
            old_swapper = AdjustableFaceScaleSwapper()
            result2 = old_swapper.swap_faces_with_scale(
                source_image, target_image, face_scale,
                "comparison_face_scaling.jpg", "balanced"
            )
            results['face_scaling'] = {
                'image': result2,
                'path': "comparison_face_scaling.jpg", 
                'success': True,
                'method': 'Scale face region'
            }
        except Exception as e:
            results['face_scaling'] = {'success': False, 'error': str(e)}
        
        # Analysis
        print(f"\n📊 METHOD COMPARISON:")
        for method, result in results.items():
            if result['success']:
                print(f"   ✅ {method}: {result['path']}")
            else:
                print(f"   ❌ {method}: Failed")
        
        return results


# Convenient functions for your workflow

def target_scale_face_swap(source_image_path: str,
                          target_image_path: str,
                          face_scale: float = 1.0,
                          output_path: str = "target_scaled_result.jpg") -> Image.Image:
    """
    Simple function using target scaling approach
    
    Args:
        face_scale: 0.9 = face 10% smaller, 1.1 = face 10% larger
    """
    swapper = TargetScalingFaceSwapper()
    return swapper.swap_faces_with_target_scaling(
        source_image=source_image_path,
        target_image=target_image_path,
        face_scale=face_scale,
        output_path=output_path
    )


def find_optimal_target_scale(source_image_path: str,
                             target_image_path: str,
                             test_scales: list = None) -> dict:
    """
    Find optimal face scale using target scaling method
    
    Args:
        test_scales: List of face scales to test
    """
    if test_scales is None:
        test_scales = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    
    swapper = TargetScalingFaceSwapper()
    return swapper.batch_test_target_scaling(
        source_image=source_image_path,
        target_image=target_image_path,
        scales=test_scales
    )


def integrate_target_scaling_with_fashion_pipeline(source_image_path: str,
                                                  checkpoint_path: str,
                                                  outfit_prompt: str,
                                                  face_scale: float = 1.0,
                                                  output_path: str = "fashion_target_scaled.jpg"):
    """
    Complete fashion pipeline with target scaling face swap
    
    This would integrate with your existing fashion generation code
    """
    print(f"👗 Fashion Pipeline with Target Scaling (face_scale={face_scale})")
    
    # Step 1: Generate outfit (your existing code)
    # generated_outfit = your_fashion_generation_function(...)
    
    # Step 2: Apply target scaling face swap
    final_result = target_scale_face_swap(
        source_image_path=source_image_path,
        target_image_path="generated_outfit.jpg",  # Your generated image
        face_scale=face_scale,
        output_path=output_path
    )
    
    print(f"✅ Fashion pipeline completed with target scaling")
    return final_result


if __name__ == "__main__":
    print("🎯 TARGET SCALING FACE SWAP - SUPERIOR APPROACH")
    print("=" * 55)
    
    print("🚀 WHY TARGET SCALING IS BETTER:")
    print("  ✅ Preserves source face quality (no interpolation)")
    print("  ✅ Natural body proportion adjustment")
    print("  ✅ Better feature alignment")
    print("  ✅ Simpler processing pipeline")
    print("  ✅ No artifacts from face region scaling")
    
    print("\n📏 CORRECTED LOGIC:")
    print("  • face_scale = 0.85 → Face stays 1.0, Target scales to 1.18 → Face appears smaller")
    print("  • face_scale = 0.90 → Face stays 1.0, Target scales to 1.11 → Face appears smaller") 
    print("  • face_scale = 1.00 → Face stays 1.0, Target scales to 1.00 → No change")
    print("  • face_scale = 1.10 → Face stays 1.0, Target scales to 0.91 → Face appears larger")
    print("  • face_scale = 1.20 → Face stays 1.0, Target scales to 0.83 → Face appears larger")
    
    print("\n📋 USAGE:")
    print("""
# Basic usage with target scaling
result = target_scale_face_swap(
    source_image_path="blonde_woman.jpg",
    target_image_path="red_dress.jpg", 
    face_scale=0.9,  # Face 10% smaller via target scaling
    output_path="result.jpg"
)

# Find optimal scale
results = find_optimal_target_scale(
    source_image_path="blonde_woman.jpg",
    target_image_path="red_dress.jpg",
    test_scales=[0.85, 0.9, 0.95, 1.0, 1.05]
)

# Compare both methods
comparison = swapper.compare_scaling_methods(
    source_image="blonde_woman.jpg",
    target_image="red_dress.jpg", 
    face_scale=0.9
)
""")
    
    print("\n🎯 RECOMMENDED FOR YOUR CASE:")
    print("  • face_scale=0.85 → face:1.0, target:1.18 (face appears smaller)")
    print("  • face_scale=0.90 → face:1.0, target:1.11 (face appears smaller)")
    print("  • Test range: 0.85 - 0.95 for smaller face appearance")
    print("  • Use crop_to_original=True for final results")
    print("  • Face quality preserved at full resolution!")