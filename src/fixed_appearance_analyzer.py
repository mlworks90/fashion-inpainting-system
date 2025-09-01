"""
FIXED APPEARANCE ANALYZER - SPECIFIC FIXES FOR YOUR ISSUES
=========================================================

Addresses the specific problems from your test:
1. Better blonde detection (was detecting light_brown instead)
2. Fixed hair conflict detection (false positive with "red evening dress")
3. Fixed division by zero in skin analysis
4. Lower confidence thresholds for application
5. More aggressive blonde classification
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, List
import os

class FixedAppearanceAnalyzer:
    """
    Fixed analyzer addressing your specific detection issues
    """
    
    def __init__(self):
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # FIXED hair color ranges - more aggressive blonde detection
        self.hair_colors = {
            'platinum_blonde': {
                'brightness_min': 210,
                'terms': ['platinum blonde', 'very light blonde'],
                'rgb_ranges': [(210, 255), (200, 255), (180, 220)]
            },
            'blonde': {
                'brightness_min': 170,
                'terms': ['blonde', 'golden blonde', 'light blonde'],
                'rgb_ranges': [(170, 220), (150, 210), (120, 180)]
            },
            'light_blonde': {
                'brightness_min': 140,
                'terms': ['light blonde', 'dirty blonde'],
                'rgb_ranges': [(140, 180), (130, 170), (100, 140)]
            },
            'light_brown': {
                'brightness_min': 100,
                'terms': ['light brown', 'ash brown'],
                'rgb_ranges': [(100, 140), (90, 130), (70, 110)]
            },
            'brown': {
                'brightness_min': 70,
                'terms': ['brown', 'chestnut brown'],
                'rgb_ranges': [(70, 110), (60, 100), (40, 80)]
            },
            'dark_brown': {
                'brightness_min': 40,
                'terms': ['dark brown', 'chocolate brown'],
                'rgb_ranges': [(40, 80), (30, 60), (20, 50)]
            },
            'black': {
                'brightness_min': 0,
                'terms': ['black', 'jet black'],
                'rgb_ranges': [(0, 50), (0, 40), (0, 35)]
            }
        }
        
        # FIXED skin tone ranges - more aggressive fair skin detection
        self.skin_tones = {
            'very_fair': {
                'brightness_min': 200,
                'terms': ['very fair skin', 'porcelain skin'],
                'rgb_ranges': [(200, 255), (190, 245), (180, 235)]
            },
            'fair': {
                'brightness_min': 170,
                'terms': ['fair skin', 'light skin'],
                'rgb_ranges': [(170, 220), (160, 210), (150, 200)]
            },
            'light_medium': {
                'brightness_min': 140,
                'terms': ['light medium skin'],
                'rgb_ranges': [(140, 180), (130, 170), (120, 160)]
            },
            'medium': {
                'brightness_min': 110,
                'terms': ['medium skin'],
                'rgb_ranges': [(110, 150), (100, 140), (90, 130)]
            },
            'medium_dark': {
                'brightness_min': 80,
                'terms': ['medium dark skin'],
                'rgb_ranges': [(80, 120), (70, 110), (60, 100)]
            },
            'dark': {
                'brightness_min': 50,
                'terms': ['dark skin'],
                'rgb_ranges': [(50, 90), (45, 85), (40, 80)]
            }
        }
        
        print("🔧 Fixed Appearance Analyzer initialized")
        print("   Fixes: Blonde detection + Conflict detection + Division by zero")
    
    def analyze_appearance_fixed(self, image_path: str) -> Dict:
        """
        Fixed appearance analysis addressing your specific issues
        """
        print(f"🔧 Fixed appearance analysis: {os.path.basename(image_path)}")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Detect face
            face_bbox = self._detect_main_face(image)
            if face_bbox is None:
                print("   ⚠️ No face detected")
                return self._default_result()
            
            # FIXED hair analysis
            hair_result = self._analyze_hair_fixed(image, face_bbox)
            
            # FIXED skin analysis
            skin_result = self._analyze_skin_fixed(image, face_bbox)
            
            # Combine results
            combined_result = {
                'hair_color': hair_result,
                'skin_tone': skin_result,
                'combined_prompt_addition': f"{hair_result['prompt_addition']}, {skin_result['prompt_addition']}",
                'overall_confidence': (hair_result['confidence'] + skin_result['confidence']) / 2,
                'success': True
            }
            
            print(f"   ✅ Hair: {hair_result['color_name']} (conf: {hair_result['confidence']:.2f})")
            print(f"   ✅ Skin: {skin_result['tone_name']} (conf: {skin_result['confidence']:.2f})")
            print(f"   🎯 Combined: '{combined_result['combined_prompt_addition']}'")
            
            return combined_result
            
        except Exception as e:
            print(f"   ⚠️ Fixed analysis failed: {e}")
            return self._default_result()
    
    def _detect_main_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Simple face detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        
        if len(faces) == 0:
            return None
        
        # Return largest face
        return tuple(max(faces, key=lambda x: x[2] * x[3]))
    
    def _analyze_hair_fixed(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict:
        """
        FIXED hair analysis with aggressive blonde detection
        """
        fx, fy, fw, fh = face_bbox
        h, w = image.shape[:2]
        
        # Define hair region (above and around face)
        hair_top = max(0, fy - int(fh * 0.4))
        hair_bottom = fy + int(fh * 0.1)
        hair_left = max(0, fx - int(fw * 0.1))
        hair_right = min(w, fx + fw + int(fw * 0.1))
        
        if hair_bottom <= hair_top or hair_right <= hair_left:
            return self._default_hair_result()
        
        # Extract hair region
        hair_region = image[hair_top:hair_bottom, hair_left:hair_right]
        
        if hair_region.size == 0:
            return self._default_hair_result()
        
        # Convert to RGB
        hair_rgb = cv2.cvtColor(hair_region, cv2.COLOR_BGR2RGB)
        
        # Get average color (simple but effective)
        hair_pixels = hair_rgb.reshape(-1, 3)
        
        # Filter out very dark (shadows) and very bright (highlights) pixels
        brightness = np.mean(hair_pixels, axis=1)
        valid_mask = (brightness > 40) & (brightness < 220)
        
        if valid_mask.sum() < 10:
            filtered_pixels = hair_pixels
        else:
            filtered_pixels = hair_pixels[valid_mask]
        
        # Calculate average color
        avg_hair_color = np.mean(filtered_pixels, axis=0).astype(int)
        
        print(f"   🔬 Hair RGB: {avg_hair_color}")
        
        # FIXED: Aggressive blonde classification
        hair_result = self._classify_hair_fixed(avg_hair_color)
        
        return hair_result
    
    def _classify_hair_fixed(self, rgb_color: np.ndarray) -> Dict:
        """
        FIXED hair classification with aggressive blonde detection
        """
        r, g, b = rgb_color
        brightness = (r + g + b) / 3
        
        print(f"   🔬 Hair brightness: {brightness:.1f}")
        
        # AGGRESSIVE blonde detection
        if brightness > 140:  # Lowered threshold
            # Additional blonde checks
            blue_ratio = b / max(1, (r + g) / 2)  # Avoid division by zero
            rg_diff = abs(r - g)
            
            print(f"   🔬 Blue ratio: {blue_ratio:.2f}, RG diff: {rg_diff}")
            
            # Blonde characteristics: low blue ratio, similar R&G
            if blue_ratio < 1.1 and rg_diff < 30:
                if brightness > 180:
                    color_name = 'blonde'
                    confidence = 0.9
                elif brightness > 160:
                    color_name = 'blonde'
                    confidence = 0.85
                else:
                    color_name = 'light_blonde'
                    confidence = 0.8
                
                print(f"   🎯 BLONDE DETECTED: {color_name}")
                
                return {
                    'color_name': color_name,
                    'confidence': confidence,
                    'rgb_values': tuple(rgb_color),
                    'prompt_addition': self.hair_colors[color_name]['terms'][0],
                    'detection_method': 'aggressive_blonde_detection'
                }
        
        # Non-blonde classification
        for color_name, color_info in self.hair_colors.items():
            if color_name in ['platinum_blonde', 'blonde', 'light_blonde']:
                continue
            
            if brightness >= color_info['brightness_min']:
                return {
                    'color_name': color_name,
                    'confidence': 0.7,
                    'rgb_values': tuple(rgb_color),
                    'prompt_addition': color_info['terms'][0],
                    'detection_method': 'brightness_classification'
                }
        
        # Default fallback
        return self._default_hair_result()
    
    def _analyze_skin_fixed(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict:
        """
        FIXED skin analysis with division by zero protection
        """
        fx, fy, fw, fh = face_bbox
        
        # Define skin regions (forehead and cheeks)
        regions = [
            # Forehead
            (fx + int(fw * 0.2), fy + int(fh * 0.1), int(fw * 0.6), int(fh * 0.2)),
            # Left cheek  
            (fx + int(fw * 0.1), fy + int(fh * 0.4), int(fw * 0.25), int(fh * 0.2)),
            # Right cheek
            (fx + int(fw * 0.65), fy + int(fh * 0.4), int(fw * 0.25), int(fh * 0.2))
        ]
        
        skin_samples = []
        
        for rx, ry, rw, rh in regions:
            if rw <= 0 or rh <= 0:
                continue
                
            # Extract region
            region = image[ry:ry+rh, rx:rx+rw]
            if region.size == 0:
                continue
            
            # Convert to RGB
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            region_pixels = region_rgb.reshape(-1, 3)
            
            # FIXED: Safe filtering with division by zero protection
            brightness = np.mean(region_pixels, axis=1)
            valid_mask = (brightness > 70) & (brightness < 230)
            
            if valid_mask.sum() > 5:
                filtered_pixels = region_pixels[valid_mask]
                avg_color = np.mean(filtered_pixels, axis=0)
                skin_samples.append(avg_color)
        
        if not skin_samples:
            return self._default_skin_result()
        
        # Average all samples
        avg_skin_color = np.mean(skin_samples, axis=0).astype(int)
        
        print(f"   🔬 Skin RGB: {avg_skin_color}")
        
        # FIXED skin classification
        skin_result = self._classify_skin_fixed(avg_skin_color)
        
        return skin_result
    
    def _classify_skin_fixed(self, rgb_color: np.ndarray) -> Dict:
        """
        FIXED skin classification with aggressive fair skin detection
        """
        r, g, b = rgb_color
        brightness = (r + g + b) / 3
        
        print(f"   🔬 Skin brightness: {brightness:.1f}")
        
        # AGGRESSIVE fair skin detection
        if brightness > 160 and min(r, g, b) > 140:  # Lowered thresholds
            if brightness > 190:
                tone_name = 'very_fair'
                confidence = 0.9
            else:
                tone_name = 'fair'
                confidence = 0.85
            
            print(f"   🎯 FAIR SKIN DETECTED: {tone_name}")
            
            return {
                'tone_name': tone_name,
                'confidence': confidence,
                'rgb_values': tuple(rgb_color),
                'prompt_addition': self.skin_tones[tone_name]['terms'][0],
                'detection_method': 'aggressive_fair_detection'
            }
        
        # Non-fair classification
        for tone_name, tone_info in self.skin_tones.items():
            if tone_name in ['very_fair', 'fair']:
                continue
            
            if brightness >= tone_info['brightness_min']:
                return {
                    'tone_name': tone_name,
                    'confidence': 0.7,
                    'rgb_values': tuple(rgb_color),
                    'prompt_addition': tone_info['terms'][0],
                    'detection_method': 'brightness_classification'
                }
        
        return self._default_skin_result()
    
    def enhance_prompt_fixed(self, base_prompt: str, image_path: str) -> Dict:
        """
        FIXED prompt enhancement with proper conflict detection
        """
        print(f"🔧 Fixed prompt enhancement...")
        
        # Analyze appearance
        appearance = self.analyze_appearance_fixed(image_path)
        
        if not appearance['success']:
            return {
                'enhanced_prompt': base_prompt,
                'appearance_analysis': appearance,
                'enhancements_applied': []
            }
        
        # FIXED conflict detection - more specific keywords
        prompt_lower = base_prompt.lower()
        
        # Hair conflict: only actual hair color words
        hair_conflicts = ['blonde', 'brunette', 'brown hair', 'black hair', 'red hair', 'auburn', 'platinum']
        has_hair_conflict = any(conflict in prompt_lower for conflict in hair_conflicts)
        
        # Skin conflict: only actual skin tone words
        skin_conflicts = ['fair skin', 'dark skin', 'pale', 'tan skin', 'light skin', 'medium skin']
        has_skin_conflict = any(conflict in prompt_lower for conflict in skin_conflicts)
        
        print(f"   🔍 Hair conflict: {has_hair_conflict}")
        print(f"   🔍 Skin conflict: {has_skin_conflict}")
        
        enhancements_applied = []
        enhanced_prompt = base_prompt
        
        # Add hair color if no conflict and decent confidence
        if not has_hair_conflict and appearance['hair_color']['confidence'] > 0.6:
            hair_addition = appearance['hair_color']['prompt_addition']
            enhanced_prompt += f", {hair_addition}"
            enhancements_applied.append('hair_color')
            print(f"   💇 Added hair: {hair_addition}")
        
        # Add skin tone if no conflict and decent confidence  
        if not has_skin_conflict and appearance['skin_tone']['confidence'] > 0.5:
            skin_addition = appearance['skin_tone']['prompt_addition']
            enhanced_prompt += f", {skin_addition}"
            enhancements_applied.append('skin_tone')
            print(f"   🎨 Added skin: {skin_addition}")
        
        return {
            'enhanced_prompt': enhanced_prompt,
            'appearance_analysis': appearance,
            'enhancements_applied': enhancements_applied
        }
    
    def _default_hair_result(self) -> Dict:
        return {
            'color_name': 'brown',
            'confidence': 0.3,
            'rgb_values': (120, 100, 80),
            'prompt_addition': 'brown hair',
            'detection_method': 'default'
        }
    
    def _default_skin_result(self) -> Dict:
        return {
            'tone_name': 'medium',
            'confidence': 0.3,
            'rgb_values': (180, 160, 140),
            'prompt_addition': 'medium skin',
            'detection_method': 'default'
        }
    
    def _default_result(self) -> Dict:
        return {
            'hair_color': self._default_hair_result(),
            'skin_tone': self._default_skin_result(),
            'combined_prompt_addition': 'natural appearance',
            'overall_confidence': 0.3,
            'success': False
        }


def test_fixed_appearance_analysis(image_path: str,
                                 checkpoint_path: str = None,
                                 outfit_prompt: str = "red evening dress"):
    """
    Test the FIXED appearance analysis system
    """
    print(f"🔧 TESTING FIXED APPEARANCE ANALYSIS")
    print(f"   Image: {os.path.basename(image_path)}")
    print(f"   Fixes: Blonde detection + Conflict detection + Division errors")
    
    # Initialize fixed analyzer
    analyzer = FixedAppearanceAnalyzer()
    
    # Test fixed analysis with the actual prompt
    result = analyzer.enhance_prompt_fixed(outfit_prompt, image_path)
    
    print(f"\n📊 FIXED ANALYSIS RESULTS:")
    print(f"   Original prompt: '{outfit_prompt}'")
    print(f"   Enhanced prompt: '{result['enhanced_prompt']}'")
    print(f"   Enhancements applied: {result['enhancements_applied']}")
    
    appearance = result['appearance_analysis']
    print(f"\n🔍 DETECTION DETAILS:")
    print(f"   Hair: {appearance['hair_color']['color_name']} (conf: {appearance['hair_color']['confidence']:.2f})")
    print(f"   Hair method: {appearance['hair_color'].get('detection_method', 'unknown')}")
    print(f"   Skin: {appearance['skin_tone']['tone_name']} (conf: {appearance['skin_tone']['confidence']:.2f})")
    print(f"   Skin method: {appearance['skin_tone'].get('detection_method', 'unknown')}")
    
    # Test with other prompts to verify conflict detection works
    test_prompts = [
        "red evening dress",  # Should add both hair and skin
        "blonde woman in red dress",  # Should skip hair, add skin  
        "fair skinned woman in dress",  # Should add hair, skip skin
        "brunette with pale skin in dress"  # Should skip both
    ]
    
    print(f"\n🧪 CONFLICT DETECTION TESTS:")
    for test_prompt in test_prompts:
        test_result = analyzer.enhance_prompt_fixed(test_prompt, image_path)
        print(f"   '{test_prompt}' → {test_result['enhancements_applied']}")
    
    # If checkpoint provided, test full generation
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n🎨 TESTING FULL GENERATION WITH FIXED ANALYSIS...")
        
        try:
            from robust_face_detection_fix import fix_false_positive_detection
            
            result_image, metadata = fix_false_positive_detection(
                source_image_path=image_path,
                checkpoint_path=checkpoint_path,
                outfit_prompt=result['enhanced_prompt'],
                output_path="fixed_appearance_test.jpg"
            )
            
            # Add analysis to metadata
            metadata['fixed_appearance_analysis'] = appearance
            metadata['fixed_enhancements'] = result['enhancements_applied'] 
            metadata['original_prompt'] = outfit_prompt
            metadata['fixed_enhanced_prompt'] = result['enhanced_prompt']
            
            print(f"   ✅ Generation completed with FIXED appearance matching!")
            print(f"   Output: fixed_appearance_test.jpg")
            return result_image, metadata
            
        except Exception as e:
            print(f"   ⚠️ Full generation test failed: {e}")
    
    return result


def debug_blonde_detection(image_path: str):
    """
    Debug why blonde detection isn't working
    """
    print(f"🔍 DEBUGGING BLONDE DETECTION FOR: {os.path.basename(image_path)}")
    
    analyzer = FixedAppearanceAnalyzer()
    
    # Load image and detect face
    image = cv2.imread(image_path)
    face_bbox = analyzer._detect_main_face(image)
    
    if face_bbox is None:
        print("   ❌ No face detected")
        return
    
    fx, fy, fw, fh = face_bbox
    h, w = image.shape[:2]
    
    # Extract hair regions
    hair_top = max(0, fy - int(fh * 0.4))
    hair_bottom = fy + int(fh * 0.1)
    hair_left = max(0, fx - int(fw * 0.1))
    hair_right = min(w, fx + fw + int(fw * 0.1))
    
    hair_region = image[hair_top:hair_bottom, hair_left:hair_right]
    hair_rgb = cv2.cvtColor(hair_region, cv2.COLOR_BGR2RGB)
    
    # Sample analysis
    hair_pixels = hair_rgb.reshape(-1, 3)
    brightness = np.mean(hair_pixels, axis=1)
    valid_mask = (brightness > 40) & (brightness < 220)
    filtered_pixels = hair_pixels[valid_mask] if valid_mask.sum() > 10 else hair_pixels
    avg_hair_color = np.mean(filtered_pixels, axis=0).astype(int)
    
    r, g, b = avg_hair_color
    overall_brightness = (r + g + b) / 3
    blue_ratio = b / max(1, (r + g) / 2)
    rg_diff = abs(r - g)
    
    print(f"   🔬 Hair region: {hair_region.shape}")
    print(f"   🔬 Average RGB: {avg_hair_color}")
    print(f"   🔬 Brightness: {overall_brightness:.1f}")
    print(f"   🔬 Blue ratio: {blue_ratio:.2f}")
    print(f"   🔬 R-G difference: {rg_diff}")
    print(f"   🔬 Blonde test: brightness > 140? {overall_brightness > 140}")
    print(f"   🔬 Blonde test: blue_ratio < 1.1? {blue_ratio < 1.1}")
    print(f"   🔬 Blonde test: rg_diff < 30? {rg_diff < 30}")
    
    # Save debug image
    debug_path = image_path.replace('.png', '_hair_debug.png').replace('.jpg', '_hair_debug.jpg')
    cv2.rectangle(image, (hair_left, hair_top), (hair_right, hair_bottom), (0, 255, 0), 2)
    cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
    cv2.imwrite(debug_path, image)
    print(f"   💾 Debug image saved: {debug_path}")


if __name__ == "__main__":
    print("🔧 FIXED APPEARANCE ANALYZER")
    print("="*45)
    
    print("\n🎯 SPECIFIC FIXES FOR YOUR ISSUES:")
    print("✅ Aggressive blonde detection (lowered brightness threshold)")
    print("✅ Fixed conflict detection (more specific keywords)")
    print("✅ Division by zero protection in skin analysis")
    print("✅ Lower confidence thresholds for application")
    print("✅ Debugging tools for blonde detection")
    
    print("\n🔬 BLONDE DETECTION LOGIC:")
    print("• Brightness > 140 (lowered from 170)")
    print("• Blue ratio < 1.1 (blonde has less blue)")
    print("• Red-Green difference < 30 (similar R&G in blonde)")
    print("• Minimum component check removed")
    
    print("\n🚫 CONFLICT DETECTION FIXES:")
    print("• Hair conflicts: Only actual hair words (blonde, brunette, etc.)")
    print("• 'red evening dress' will NOT trigger hair conflict")
    print("• More specific skin conflict detection")
    
    print("\n📋 USAGE:")
    print("""
# Test the fixed system
result = test_fixed_appearance_analysis(
    image_path="woman_jeans_t-shirt.png",
    checkpoint_path="realisticVisionV60B1_v51HyperVAE.safetensors"
)

# Debug blonde detection specifically  
debug_blonde_detection("woman_jeans_t-shirt.png")
""")
    
    print("\n🎯 EXPECTED IMPROVEMENTS:")
    print("• Should detect 'blonde' instead of 'light_brown'")
    print("• Should detect 'fair' instead of 'medium' skin")
    print("• Should ADD enhancements to 'red evening dress' prompt")
    print("• Should eliminate division by zero warnings")
    print("• Should show proper conflict detection logic")