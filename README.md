# AI Style Transfer Studio

A powerful toolkit for generating artwork in various artistic styles using AI. This project allows you to train custom artistic styles and generate new artwork based on text prompts.

## Features

- Generate artwork in 5 different artistic styles:
  - **Watercolor**: Soft, flowing, painterly aesthetic
  - **Cyberpunk**: Neon, futuristic, high-tech visual elements
  - **Van Gogh**: Swirling brush strokes, vibrant colors
  - **Ukiyo-e**: Traditional Japanese woodblock print aesthetic
  - **Retro**: 80s/90s nostalgic color palette and design

- Custom "Harmony Loss" function that enhances:
  - Color harmony relationships based on color theory
  - Luminance balancing for better light distribution
  - Focal contrast enhancement for areas of interest
  - Edge coherence for cleaner transitions

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ai-style-transfer-studio.git
cd ai-style-transfer-studio
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training Custom Style Embeddings

The repository comes with synthetic images for training. You can train style embeddings for all available styles with:

```bash
python train_all_styles.py
```

Or train an individual style with:

```bash
python src/train_style.py watercolor --steps 3000 --lr 1e-4 --seed 42
```

### 2. Generating Artwork

Generate an image using a specific style:

```bash
python test_style_transfer.py "a serene mountain landscape with a lake" --style watercolor
```

Generate a comparison of all available styles:

```bash
python test_style_transfer.py "a portrait of a woman in traditional clothing" --compare
```

### 3. Project Structure

- `src/train_style.py`: Script for training style embeddings
- `src/generate_art.py`: Core module for generating artwork
- `style_images/`: Contains training images organized by style
- `style_embeddings/`: Contains trained style embeddings
- `outputs/`: Generated artwork will be saved here

## How the Harmony Loss Works

The Harmony Loss function enhances the aesthetic quality of generated images through:

1. **Color Harmony in YUV Space**: Works in a color space closer to human perception to optimize complementary color relationships.

2. **Luminance Balancing**: Creates a balanced distribution of lights and darks, ensuring proper tonal variation.

3. **Focal Contrast Enhancement**: Identifies important areas of the image and enhances their contrast for better visual impact.

4. **Edge Coherence**: Improves the clarity of edges and boundaries while reducing noise.

## Examples

Generated examples will appear in the `outputs` directory after running the generation scripts.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Based on Stable Diffusion by Runway and Stability AI
- Implements techniques inspired by color theory and art composition
