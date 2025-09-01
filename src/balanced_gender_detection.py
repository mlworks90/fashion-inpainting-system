"""
BALANCED GENDER DETECTION FIX
=============================

ISSUE IDENTIFIED:
The current gender detection is heavily biased toward male detection:
- "red evening dress" with woman source → generates man in dress
- System defaults to male unless there are NO male indicators at all

PROBLEM IN CURRENT CODE:
- if male_score > 0.6: return male (reasonable)
- elif male_score > 0.3: return male (TOO AGGRESSIVE)
- else: return female (only as last resort)

SOLUTION:
- Balanced scoring system that considers both male AND female indicators
- Proper thresholds for both genders
- Better facial analysis that doesn't bias toward masculinity
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
import os


class BalancedGenderDetector:
    """
    BALANCED gender detection that works equally well for men and women
    
    Fixes the current bias toward male classification
    """
    
    def __init__(self):
        self.face_cascade = self._load_face_cascade()
        
        print("🔧 BALANCED Gender Detector initialized")
        print("   ✅ Equal consideration for male and female features")
        print("   ✅ Removes male bias from detection logic")
        print("   ✅ Better thresholds for both genders")
    
    def _load_face_cascade(self):
        """Load face cascade"""
        try:
            cascade_paths = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                'haarcascade_frontalface_default.xml'
            ]
            
            for path in cascade_paths:
                if os.path.exists(path):
                    return cv2.CascadeClassifier(path)
            
            return None
        except Exception as e:
            print(f"⚠️ Error loading face cascade: {e}")
            return None
    
    def detect_gender_balanced(self, image_path: str) -> Dict:
        """
        BALANCED gender detection from image
        
        Returns proper classification for both men and women
        """
        print(f"🔍 BALANCED gender detection: {os.path.basename(image_path)}")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Detect face
            face_bbox = self._detect_main_face(image)
            if face_bbox is None:
                print("   ⚠️ No face detected - using fallback analysis")
                return self._analyze_without_face(image)
            
            fx, fy, fw, fh = face_bbox
            print(f"   ✅ Face detected: {fw}x{fh} at ({fx}, {fy})")
            
            # Extract face region
            face_region = image[fy:fy+fh, fx:fx+fw]
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # BALANCED analysis - consider both male AND female indicators
            male_indicators = self._analyze_male_indicators(face_region, face_gray, fw, fh)
            female_indicators = self._analyze_female_indicators(face_region, face_gray, fw, fh)
            
            # Make balanced decision
            gender_result = self._make_balanced_gender_decision(male_indicators, female_indicators)
            
            print(f"   📊 Male indicators: {male_indicators['total_score']:.2f}")
            print(f"   📊 Female indicators: {female_indicators['total_score']:.2f}")
            print(f"   🎯 Final gender: {gender_result['gender']} (conf: {gender_result['confidence']:.2f})")
            
            return gender_result
            
        except Exception as e:
            print(f"   ❌ Gender detection failed: {e}")
            return {
                'gender': 'neutral',
                'confidence': 0.5,
                'method': 'error_fallback'
            }
    
    def _detect_main_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect main face in image"""
        if self.face_cascade is None:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        
        if len(faces) == 0:
            return None
        
        return tuple(max(faces, key=lambda x: x[2] * x[3]))
    
    def _analyze_male_indicators(self, face_region: np.ndarray, face_gray: np.ndarray, fw: int, fh: int) -> Dict:
        """
        Analyze indicators that suggest MALE gender
        
        More conservative than the current overly aggressive detection
        """
        male_score = 0.0
        indicators = {}
        
        # 1. Face width-to-height ratio (men often have wider faces)
        aspect_ratio = fw / fh
        indicators['aspect_ratio'] = aspect_ratio
        
        if aspect_ratio > 0.90:  # More conservative threshold (was 0.85)
            male_score += 0.2
            indicators['wide_face'] = True
        else:
            indicators['wide_face'] = False
        
        # 2. Facial hair detection (strong male indicator when present)
        facial_hair_result = self._detect_facial_hair_conservative(face_gray, fw, fh)
        indicators['facial_hair'] = facial_hair_result
        
        if facial_hair_result['detected'] and facial_hair_result['confidence'] > 0.7:
            male_score += 0.4  # Strong indicator
            print(f"   👨 Strong facial hair detected (conf: {facial_hair_result['confidence']:.2f})")
        elif facial_hair_result['detected']:
            male_score += 0.2  # Weak indicator
            print(f"   👨 Weak facial hair detected (conf: {facial_hair_result['confidence']:.2f})")
        
        # 3. Jawline sharpness (men often have more defined jawlines)
        jawline_result = self._analyze_jawline_sharpness(face_gray, fh)
        indicators['jawline'] = jawline_result
        
        if jawline_result['sharpness'] > 0.2:  # More conservative
            male_score += 0.15
        
        # 4. Eyebrow thickness (men often have thicker eyebrows)
        eyebrow_result = self._analyze_eyebrow_thickness(face_gray, fw, fh)
        indicators['eyebrows'] = eyebrow_result
        
        if eyebrow_result['thickness'] > 0.6:
            male_score += 0.1
        
        indicators['total_score'] = male_score
        
        return indicators
    
    def _analyze_female_indicators(self, face_region: np.ndarray, face_gray: np.ndarray, fw: int, fh: int) -> Dict:
        """
        Analyze indicators that suggest FEMALE gender
        
        NEW: The current system doesn't properly look for female indicators!
        """
        female_score = 0.0
        indicators = {}
        
        # 1. Face shape analysis (women often have more oval faces)
        aspect_ratio = fw / fh
        indicators['aspect_ratio'] = aspect_ratio
        
        if 0.75 <= aspect_ratio <= 0.85:  # More oval/narrow
            female_score += 0.2
            indicators['oval_face'] = True
        else:
            indicators['oval_face'] = False
        
        # 2. Skin smoothness (women often have smoother skin texture)
        smoothness_result = self._analyze_skin_smoothness(face_gray)
        indicators['skin_smoothness'] = smoothness_result
        
        if smoothness_result['smoothness'] > 0.6:
            female_score += 0.25
        elif smoothness_result['smoothness'] > 0.4:
            female_score += 0.15
        
        # 3. Eye makeup detection (subtle indicator)
        eye_makeup_result = self._detect_subtle_makeup(face_region, fw, fh)
        indicators['makeup'] = eye_makeup_result
        
        if eye_makeup_result['likely_makeup']:
            female_score += 0.2
        
        # 4. Hair length analysis (longer hair often indicates female)
        # This is done at image level, not face level
        hair_length_result = self._estimate_hair_length_from_face(face_region, fw, fh)
        indicators['hair_length'] = hair_length_result
        
        if hair_length_result['appears_long']:
            female_score += 0.15
        
        # 5. Facial feature delicacy (women often have more delicate features)
        delicacy_result = self._analyze_feature_delicacy(face_gray, fw, fh)
        indicators['feature_delicacy'] = delicacy_result
        
        if delicacy_result['delicate_score'] > 0.5:
            female_score += 0.1
        
        indicators['total_score'] = female_score
        
        return indicators
    
    def _detect_facial_hair_conservative(self, face_gray: np.ndarray, fw: int, fh: int) -> Dict:
        """
        CONSERVATIVE facial hair detection
        
        The current system is too aggressive - detecting shadows as facial hair
        """
        if fh < 60:  # Face too small for reliable detection
            return {'detected': False, 'confidence': 0.0, 'method': 'face_too_small'}
        
        # Focus on mustache and beard areas
        mustache_region = face_gray[int(fh*0.55):int(fh*0.75), int(fw*0.3):int(fw*0.7)]
        beard_region = face_gray[int(fh*0.7):int(fh*0.95), int(fw*0.2):int(fw*0.8)]
        
        facial_hair_detected = False
        confidence = 0.0
        
        # Mustache analysis
        if mustache_region.size > 0:
            mustache_mean = np.mean(mustache_region)
            mustache_std = np.std(mustache_region)
            dark_pixel_ratio = np.sum(mustache_region < mustache_mean - mustache_std) / mustache_region.size
            
            if dark_pixel_ratio > 0.25:  # More conservative (was 0.15)
                facial_hair_detected = True
                confidence += 0.4
        
        # Beard analysis
        if beard_region.size > 0:
            beard_mean = np.mean(beard_region)
            beard_std = np.std(beard_region)
            dark_pixel_ratio = np.sum(beard_region < beard_mean - beard_std) / beard_region.size
            
            if dark_pixel_ratio > 0.20:  # More conservative
                facial_hair_detected = True
                confidence += 0.6
        
        # Additional texture analysis for confirmation
        if facial_hair_detected:
            # Check for hair-like texture patterns
            combined_region = np.vstack([mustache_region, beard_region]) if mustache_region.size > 0 and beard_region.size > 0 else beard_region
            if combined_region.size > 0:
                texture_variance = cv2.Laplacian(combined_region, cv2.CV_64F).var()
                if texture_variance > 50:  # Hair has texture
                    confidence += 0.2
                else:
                    confidence *= 0.7  # Reduce confidence if no texture
        
        return {
            'detected': facial_hair_detected,
            'confidence': min(1.0, confidence),
            'method': 'conservative_analysis'
        }
    
    def _analyze_jawline_sharpness(self, face_gray: np.ndarray, fh: int) -> Dict:
        """Analyze jawline sharpness"""
        if fh < 60:
            return {'sharpness': 0.0}
        
        # Focus on jawline area
        jaw_region = face_gray[int(fh*0.75):, :]
        
        if jaw_region.size == 0:
            return {'sharpness': 0.0}
        
        # Edge detection for jawline sharpness
        edges = cv2.Canny(jaw_region, 50, 150)
        sharpness = np.mean(edges) / 255.0
        
        return {'sharpness': sharpness}
    
    def _analyze_eyebrow_thickness(self, face_gray: np.ndarray, fw: int, fh: int) -> Dict:
        """Analyze eyebrow thickness"""
        if fh < 60:
            return {'thickness': 0.0}
        
        # Eyebrow region
        eyebrow_region = face_gray[int(fh*0.25):int(fh*0.45), int(fw*0.2):int(fw*0.8)]
        
        if eyebrow_region.size == 0:
            return {'thickness': 0.0}
        
        # Look for dark horizontal structures (eyebrows)
        mean_brightness = np.mean(eyebrow_region)
        dark_threshold = mean_brightness - 20
        dark_pixels = np.sum(eyebrow_region < dark_threshold)
        thickness = dark_pixels / eyebrow_region.size
        
        return {'thickness': thickness}
    
    def _analyze_skin_smoothness(self, face_gray: np.ndarray) -> Dict:
        """Analyze skin texture smoothness"""
        # Use Laplacian variance to measure texture
        texture_variance = cv2.Laplacian(face_gray, cv2.CV_64F).var()
        
        # Lower variance = smoother skin
        # Normalize to 0-1 scale (rough approximation)
        smoothness = max(0, 1.0 - (texture_variance / 500.0))
        
        return {'smoothness': smoothness, 'texture_variance': texture_variance}
    
    def _detect_subtle_makeup(self, face_region: np.ndarray, fw: int, fh: int) -> Dict:
        """Detect subtle makeup indicators"""
        if len(face_region.shape) != 3 or fh < 60:
            return {'likely_makeup': False, 'confidence': 0.0}
        
        # Focus on eye area
        eye_region = face_region[int(fh*0.3):int(fh*0.55), int(fw*0.2):int(fw*0.8)]
        
        if eye_region.size == 0:
            return {'likely_makeup': False, 'confidence': 0.0}
        
        # Look for color enhancement around eyes
        eye_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
        
        # Check for enhanced colors (makeup often increases color saturation)
        saturation = np.std(eye_rgb, axis=2)
        high_saturation_ratio = np.sum(saturation > np.percentile(saturation, 80)) / saturation.size
        
        likely_makeup = high_saturation_ratio > 0.15
        confidence = min(1.0, high_saturation_ratio * 3)
        
        return {'likely_makeup': likely_makeup, 'confidence': confidence}
    
    def _estimate_hair_length_from_face(self, face_region: np.ndarray, fw: int, fh: int) -> Dict:
        """Estimate hair length from visible hair around face"""
        # This is a rough estimate based on hair visible around face edges
        
        # Check hair regions around face
        hair_regions = []
        
        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_region
        
        # Check top region for hair
        top_region = gray_face[:int(fh*0.2), :]
        if top_region.size > 0:
            hair_regions.append(top_region)
        
        # Check side regions
        left_region = gray_face[:, :int(fw*0.15)]
        right_region = gray_face[:, int(fw*0.85):]
        
        if left_region.size > 0:
            hair_regions.append(left_region)
        if right_region.size > 0:
            hair_regions.append(right_region)
        
        # Analyze for hair-like texture
        total_hair_indicators = 0
        total_regions = len(hair_regions)
        
        for region in hair_regions:
            if region.size > 10:  # Enough pixels to analyze
                texture_var = np.var(region)
                # Hair typically has more texture variation than skin
                if texture_var > 200:  # Has hair-like texture
                    total_hair_indicators += 1
        
        hair_ratio = total_hair_indicators / max(1, total_regions)
        appears_long = hair_ratio > 0.5
        
        return {
            'appears_long': appears_long,
            'hair_ratio': hair_ratio,
            'regions_analyzed': total_regions
        }
    
    def _analyze_feature_delicacy(self, face_gray: np.ndarray, fw: int, fh: int) -> Dict:
        """Analyze overall feature delicacy"""
        # Use edge detection to measure feature sharpness
        edges = cv2.Canny(face_gray, 30, 100)  # Lower thresholds for subtle features
        
        # Delicate features have softer, less harsh edges
        edge_intensity = np.mean(edges)
        
        # Lower edge intensity = more delicate features
        delicate_score = max(0, 1.0 - (edge_intensity / 50.0))
        
        return {'delicate_score': delicate_score, 'edge_intensity': edge_intensity}
    
    def _make_balanced_gender_decision(self, male_indicators: Dict, female_indicators: Dict) -> Dict:
        """
        BALANCED gender decision based on both male AND female indicators
        
        FIXES the current bias toward male classification
        """
        male_score = male_indicators['total_score']
        female_score = female_indicators['total_score']
        
        print(f"   🔄 Gender scoring: Male={male_score:.2f}, Female={female_score:.2f}")
        
        # Clear male indicators (high confidence)
        if male_score > 0.7 and male_score > female_score + 0.3:
            return {
                'gender': 'male',
                'confidence': min(0.95, 0.6 + male_score),
                'method': 'strong_male_indicators',
                'male_score': male_score,
                'female_score': female_score
            }
        
        # Clear female indicators (high confidence)
        elif female_score > 0.7 and female_score > male_score + 0.3:
            return {
                'gender': 'female',
                'confidence': min(0.95, 0.6 + female_score),
                'method': 'strong_female_indicators',
                'male_score': male_score,
                'female_score': female_score
            }
        
        # Moderate male indicators
        elif male_score > 0.5 and male_score > female_score + 0.2:
            return {
                'gender': 'male',
                'confidence': 0.75,
                'method': 'moderate_male_indicators',
                'male_score': male_score,
                'female_score': female_score
            }
        
        # Moderate female indicators  
        elif female_score > 0.5 and female_score > male_score + 0.2:
            return {
                'gender': 'female',
                'confidence': 0.75,
                'method': 'moderate_female_indicators',
                'male_score': male_score,
                'female_score': female_score
            }
        
        # Close scores - use slight preference but lower confidence
        elif male_score > female_score:
            return {
                'gender': 'male',
                'confidence': 0.6,
                'method': 'slight_male_preference',
                'male_score': male_score,
                'female_score': female_score
            }
        
        elif female_score > male_score:
            return {
                'gender': 'female',
                'confidence': 0.6,
                'method': 'slight_female_preference',
                'male_score': male_score,
                'female_score': female_score
            }
        
        # Equal scores - neutral
        else:
            return {
                'gender': 'neutral',
                'confidence': 0.5,
                'method': 'equal_indicators',
                'male_score': male_score,
                'female_score': female_score
            }
    
    def _analyze_without_face(self, image: np.ndarray) -> Dict:
        """Fallback analysis when face detection fails"""
        print("   📐 Fallback analysis (no face detected)")
        
        # Simple image-based heuristics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Hair length estimation from top region
        top_region = gray[:int(h*0.3), :]
        hair_variance = np.var(top_region) if top_region.size > 0 else 0
        
        # Very rough estimation
        if hair_variance > 400:  # High variance suggests longer/more complex hair
            return {
                'gender': 'female',
                'confidence': 0.6,
                'method': 'image_fallback_long_hair'
            }
        else:
            return {
                'gender': 'male',
                'confidence': 0.6,
                'method': 'image_fallback_short_hair'
            }


def create_balanced_enhancer_patch():
    """
    Integration patch to replace the biased gender detection
    """
    print("🔧 BALANCED GENDER DETECTION PATCH")
    print("="*35)
    
    print("\nISSUE IDENTIFIED:")
    print("   Current system is biased toward MALE detection")
    print("   'red evening dress' + woman image → man in dress")
    print("   Gender detection defaults to male unless NO male indicators")
    
    print("\nFIXES APPLIED:")
    print("   ✅ Balanced scoring (considers both male AND female indicators)")
    print("   ✅ Conservative facial hair detection (less false positives)")
    print("   ✅ Female indicator analysis (missing in current system)")
    print("   ✅ Proper decision thresholds for both genders")
    
    print("\nINTEGRATION:")
    print("""
# In your ImprovedUnifiedGenderAppearanceEnhancer class, replace:

def _analyze_gender_simple(self, image, face_bbox):
    # Current biased logic
    
# With:

def _analyze_gender_simple(self, image, face_bbox):
    \"\"\"Use balanced gender detection\"\"\"
    if not hasattr(self, 'balanced_detector'):
        self.balanced_detector = BalancedGenderDetector()
    
    # Convert face_bbox to image_path analysis (simplified for integration)
    # For full fix, extract face region and analyze directly
    
    # Placeholder logic - you'll need to adapt this to your specific interface
    # The key is using balanced scoring instead of male-biased scoring
    
    male_score = 0.0
    female_score = 0.0
    
    # Facial analysis here...
    # Use the balanced decision logic from BalancedGenderDetector
    
    if male_score > female_score + 0.3:
        return {'gender': 'male', 'confidence': 0.8}
    elif female_score > male_score + 0.3:
        return {'gender': 'female', 'confidence': 0.8}
    else:
        return {'gender': 'neutral', 'confidence': 0.6}
""")


def test_balanced_detection():
    """Test cases for balanced gender detection"""
    print("\n🧪 TESTING BALANCED GENDER DETECTION")
    print("="*40)
    
    test_cases = [
        {
            'description': 'Woman with long hair and smooth skin',
            'male_indicators': {'total_score': 0.1},
            'female_indicators': {'total_score': 0.8},
            'expected': 'female'
        },
        {
            'description': 'Man with facial hair and wide face',
            'male_indicators': {'total_score': 0.9},
            'female_indicators': {'total_score': 0.2},
            'expected': 'male'
        },
        {
            'description': 'Ambiguous features (current system would default to male)',
            'male_indicators': {'total_score': 0.4},
            'female_indicators': {'total_score': 0.5},
            'expected': 'female'  # Should properly detect female now
        }
    ]
    
    detector = BalancedGenderDetector()
    
    for case in test_cases:
        result = detector._make_balanced_gender_decision(
            case['male_indicators'],
            case['female_indicators']
        )
        
        passed = result['gender'] == case['expected']
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"{status} {case['description']}")
        print(f"   Male: {case['male_indicators']['total_score']:.1f}, "
              f"Female: {case['female_indicators']['total_score']:.1f}")
        print(f"   Result: {result['gender']} (expected: {case['expected']})")
        print(f"   Method: {result['method']}")
        print()


if __name__ == "__main__":
    print("🔧 BALANCED GENDER DETECTION FIX")
    print("="*35)
    
    print("\n❌ CURRENT PROBLEM:")
    print("System biased toward MALE detection")
    print("'red evening dress' + woman → man in dress")
    print("Defaults to male unless zero male indicators")
    
    print("\n✅ SOLUTION PROVIDED:")
    print("• Balanced scoring for both genders")
    print("• Conservative facial hair detection") 
    print("• Female indicator analysis (NEW)")
    print("• Proper decision thresholds")
    
    # Test the balanced detection
    test_balanced_detection()
    
    # Integration instructions
    create_balanced_enhancer_patch()
    
    print(f"\n🎯 EXPECTED FIX:")
    print("• Woman + 'red evening dress' → woman in dress ✅")
    print("• Man + 'business suit' → man in suit ✅") 
    print("• Equal consideration for both genders")
    print("• No more default-to-male bias")