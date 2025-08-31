# Fashion Inpainting System

🎨 **Advanced AI-powered fashion transformation system that preserves body pose and facial identity while generating new clothing styles.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co)

## 🚀 Key Features

- **Pose Preservation**: Advanced 25.3% pose coverage system maintains body structure and proportions
- **Facial Identity Protection**: Preserves original facial features and expressions
- **Safety-First Design**: Built-in content filtering and safety checks
- **Multiple Checkpoint Support**: Compatible with various Stable Diffusion checkpoints
- **Production Ready**: Comprehensive error handling and fallback systems

## 🎯 What This System Does

**Input**: Person wearing any outfit  
**Output**: Same person in a completely different outfit while maintaining:
- ✅ Exact facial identity
- ✅ Original body pose and proportions  
- ✅ Natural fabric draping and fit
- ✅ Appropriate content generation

## 🛡️ Safety & Ethical Use

### ⚠️ IMPORTANT USAGE RESTRICTIONS

This system is designed for **creative and artistic purposes only**. By using this software, you agree to:

**✅ ALLOWED USES:**
- Fashion design and visualization
- Creative artwork and artistic expression
- Educational and research purposes  
- Personal style exploration
- Commercial fashion applications (with proper licensing)

**❌ PROHIBITED USES:**
- Creating deceptive or misleading content
- Non-consensual image manipulation
- Identity theft or impersonation
- Harassment or bullying
- Creation of inappropriate content
- Any illegal or harmful activities

### 🔒 Built-in Safety Features

- **Content Filtering**: Automatic detection and prevention of inappropriate outputs
- **Identity Preservation**: System designed to change clothing only, not faces
- **Pose Validation**: Ensures generated content maintains appropriate poses
- **Quality Thresholds**: Filters out low-quality or distorted results

## 🏗️ System Architecture

### Core Components

1. **Pose Extraction System** (25.3% coverage)
   - OpenPose-based pose detection via controlnet_aux
   - 5-channel pose vectors (Body, Hands, Face, Feet, Skeleton)
   - Dilated regions for enhanced coverage

2. **Hand Exclusion Logic**
   - Prevents generation of extra hands/limbs
   - Conservative mask erosion with exclusion zones
   - Optimized for natural results

3. **Safety-Aware Generation**
   - Content filtering for appropriate results
   - Coverage analysis for generation scope
   - Adaptive prompting based on input analysis

4. **Checkpoint Compatibility**
   - Supports custom Stable Diffusion models
   - Automatic parameter optimization
   - Fashion-specific model recommendations

## 📋 Requirements

```bash
Python 3.8+
torch>=1.13.0
diffusers>=0.21.0
transformers>=4.21.0
controlnet_aux>=0.4.0
opencv-python>=4.6.0
pillow>=9.0.0
numpy>=1.21.0
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mlworks90/fashion-inpainting-system.git
cd fashion-inpainting-system

# Install dependencies
pip install -r requirements.txt

# Optional: Install with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Basic Usage

```python
from fashion_safety_checker import create_fashion_safety_pipeline
pipeline = create_fashion_safety_pipeline()

# Transform outfit
result = pipeline.safe_fashion_transformation(
    source_image_path="person_in_some_outfit.png",
    checkpoint_path="desired_outfit_style_checkpoint.safetensors",
    outfit_prompt="another_ouitfit",
    output_path="same_person_another_ouitfit.jpg",
    face_scale=0.90     # manmual face to body ratio adjustment 
)

if result['success']:
    print("✅ Fashion transformation completed")
else:    
    print(f"Blocking reason: {result['blocking_reason']}")
    print(f"User message: {result['user_message']}")
```

## 📊 Performance & Quality

- **Pose Preservation**: 25.3% coverage ensures accurate body structure
- **Face Identity**: >95% facial feature preservation
- **Generation Speed**: ~30-60 seconds per image (depending on hardware)
- **Memory Usage**: 8-12GB VRAM recommended
- **Success Rate**: >85% for well-posed input images

## 🔧 Configuration

### Safety Settings

```python
pipeline = create_fashion_safety_pipeline(safety_mode="legacy_strict") 
 # legacy_strict - highest safety restrictions
 # fashion_strict - conservative outfits only
 # fashion_moderate - default level. Suitable for most garment types except some swimwear.
 # fashion_permissive - most permissive mode. Be aware of inappropriate outputs possibility!
            

```



## 🧪 Examples

### Fashion Transformations

| Input | Target Prompt | Output |
|-------|---------------|--------|
| Casual wear | "elegant evening dress" | [Link to example] |
| Casual wear | "Business suit" | [Link to example] |
| Casual wear | "Business Costume" |  [Link to example] |

## 🏢 Commercial Use & Support

### Open Source License
This project is licensed under **Apache License 2.0**, allowing:
- ✅ Commercial use
- ✅ Modification and distribution
- ✅ Private use
- ✅ Patent grant

### Professional Services Available

For commercial deployments, we offer:
- **Custom model training** for specific fashion domains
- **API integration** and cloud deployment
- **Performance optimization** for production environments
- **Priority support** and SLA guarantees
- **Custom safety filtering** for brand-specific requirements

Contact: [mlworks90@gmail.com](mailto:mlworks90@gmail.com)

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api_reference.md)  
- [Safety Guidelines](docs/safety_guidelines.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Commercial Licensing](docs/commercial_licensing.md)

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

### Development Setup

```bash
# Clone repository
git clone https://github.com/mlworks90/fashion-inpainting-system.git
cd fashion-inpainting-system

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## 🙏 Acknowledgments

This system builds upon excellent open-source projects:
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) by CompVis
- [ControlNet](https://github.com/lllyasviel/ControlNet) by lllyasviel
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [controlnet_aux](https://github.com/patrickvonplaten/controlnet_aux) for OpenPose processing

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## ⚖️ Legal & Safety Disclaimers

- Users are responsible for ensuring appropriate use and obtaining necessary consents
- This software is provided "as is" without warranty
- Not intended for creating deceptive or harmful content
- Users must comply with applicable laws and regulations
- Commercial users should review terms and consider professional support

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/mlworks90/fashion-inpainting-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mlworks90/fashion-inpainting-system/discussions)
- **Commercial Inquiries**: [your-email@domain.com](mailto:mlworks90@gmailo.com)
- **Documentation**: [Project Wiki](https://github.com/mlworks90/fashion-inpainting-system/wiki)

---

**Made with ❤️ for the AI and Fashion communities**
