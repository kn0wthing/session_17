# Guided Diffusion with Custom Loss Functions

This project demonstrates how to use custom loss functions with Stable Diffusion to guide image generation. It includes implementations for:

1. **Edge Sharpening Loss** - Makes the edges in the generated image more defined and crisp
2. **Subject Changing Loss** - Gradually shifts the subject of an image to a different concept
3. **Blue Pixel Loss** - A simple example that increases the blue component in generated images

## Requirements

Install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install Pillow numpy tqdm
```

## Usage

The main script `src/generate_images.py` provides a command-line interface for generating images with different loss functions:

```bash
# Generate an image with edge sharpening
python src/generate_images.py --prompt "Portrait of a woman wearing a red hat" --mode edge

# Generate an image with subject changing
python src/generate_images.py --prompt "A golden retriever" --mode subject --target_subject "A wolf in a forest"

# Generate an image with blue emphasis
python src/generate_images.py --prompt "A sunset over mountains" --mode blue
```

### Command-line arguments

- `--prompt`: Text prompt for image generation
- `--mode`: Loss function to use: "edge", "subject", or "blue"
- `--target_subject`: Target subject for subject changing mode
- `--steps`: Number of diffusion steps (default: 50)
- `--scale`: Scale for the loss function (defaults depend on mode)
- `--output_dir`: Directory to save intermediate results
- `--seed`: Random seed for reproducibility (default: 42)
- `--model_id`: Hugging Face model ID to use (default: "runwayml/stable-diffusion-v1-5")

## How It Works

This project uses the concept of classifier-free guidance combined with custom loss functions applied in the latent space during the diffusion process:

1. At each diffusion step, we predict the denoised image (x0) based on the current noisy latents
2. We apply our custom loss function to this predicted image
3. We calculate the gradient of the loss with respect to the latents
4. We update the latents to reduce this loss, steering the generation process

## Examples

### Edge Sharpening

The edge sharpening loss uses Sobel filters to detect edges and maximizes their magnitude. This encourages the diffusion process to generate images with more defined edges.

### Subject Changing

The subject changing loss uses CLIP to compute the similarity between the generated image and two text prompts: the original prompt and the target subject. It then steers the generation to maximize similarity with the target while minimizing similarity with the original.

### Blue Pixel Emphasis

A simple example loss that promotes pixels with high blue channel values, resulting in images with a blue tint.

## Customization

You can create your own custom loss functions by extending the `GuidedDiffusion` class in `src/guided_diffusion.py`. Implement a new loss function and a corresponding generation method, following the patterns in the existing code. 