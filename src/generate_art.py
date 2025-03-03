"""
Enhanced Art Generator with Harmony Loss function.
This script is optimized for Google Colab with GPU support.
"""

import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import os
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

class ArtGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", style_embeddings_dir="style_embeddings"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize the pipeline
        print("Initializing Stable Diffusion model...")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.pipeline = self.pipeline.to(self.device)
        
        # Define available styles
        self.style_names = ["watercolor", "cyberpunk", "vangogh", "ukiyoe", "retro"]
        
        # Define seeds for each style for reproducibility
        self.style_seeds = {
            "watercolor": 42,
            "cyberpunk": 123,
            "vangogh": 456,
            "ukiyoe": 789,
            "retro": 1024
        }
        
        # Load style embeddings
        self.style_tokens = {}
        for style_name in self.style_names:
            try:
                token = self._load_style_embedding(
                    os.path.join(style_embeddings_dir, f"{style_name}.bin")
                )
                self.style_tokens[style_name] = token
                print(f"✓ Loaded {style_name} style")
            except Exception as e:
                print(f"Error loading {style_name} style: {e}")
        
        if not self.style_tokens:
            raise ValueError("No style embeddings could be loaded")
    
    def _load_style_embedding(self, embedding_path):
        """Load and process the style embedding."""
        try:
            # Load the embedding
            loaded_embeds = torch.load(embedding_path, map_location="cpu")
            trained_token = list(loaded_embeds.keys())[0]
            embeds = loaded_embeds[trained_token]
            
            # Get the expected dimension from the text encoder
            expected_dim = self.pipeline.text_encoder.get_input_embeddings().weight.shape[1]
            current_dim = embeds.shape[0]
            
            # Resize embeddings if dimensions don't match
            if current_dim != expected_dim:
                print(f"Resizing embedding from {current_dim} to {expected_dim}")
                if current_dim > expected_dim:
                    embeds = embeds[:expected_dim]
                else:
                    embeds = torch.cat([embeds, torch.zeros(expected_dim - current_dim)], dim=0)
            
            # Reshape to match expected dimensions
            embeds = embeds.unsqueeze(0)
            
            # Cast to dtype of text_encoder
            dtype = self.pipeline.text_encoder.get_input_embeddings().weight.dtype
            embeds = embeds.to(dtype)
            
            # Add the token in tokenizer
            self.pipeline.tokenizer.add_tokens(trained_token)
            
            # Resize the token embeddings
            self.pipeline.text_encoder.resize_token_embeddings(len(self.pipeline.tokenizer))
            
            # Get the id for the token and assign the embeds
            token_id = self.pipeline.tokenizer.convert_tokens_to_ids(trained_token)
            self.pipeline.text_encoder.get_input_embeddings().weight.data[token_id] = embeds[0]
            
            return trained_token
            
        except Exception as e:
            print(f"Error loading style embedding: {str(e)}")
            raise
    
    def _harmony_loss(self, i, t, latents):
        """Advanced Harmony Loss for enhanced color relationships.
        
        This creates a more aesthetically pleasing image by:
        1. Enhancing color harmony based on color theory principles
        2. Improving contrast in focal areas
        3. Balancing tonal values across the image
        
        It's designed to create more visually striking and balanced images.
        """
        if i % 5 == 0:  # Apply enhancement every 5 steps
            try:
                # Create a copy that requires gradients
                latents_copy = latents.detach().clone()
                latents_copy.requires_grad_(True)
                
                # Compute harmony loss (combines multiple aesthetic aspects)
                loss = self._calculate_harmony_score(latents_copy)
                
                # Compute gradients
                if loss.requires_grad:
                    grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=latents_copy,
                        allow_unused=True,
                        retain_graph=False
                    )[0]
                    
                    if grads is not None:
                        # Apply gradients with a stronger effect than before
                        return latents - 0.15 * grads.detach()  # Increased from 0.1
            
            except Exception as e:
                print(f"Error in harmony enhancement: {e}")
            
        return latents
    
    def _calculate_harmony_score(self, images):
        """Calculate the harmony score based on multiple aesthetic factors."""
        # Ensure we're working with gradients
        if not images.requires_grad:
            images = images.detach().requires_grad_(True)
        
        # Convert to float32 and normalize
        images = images.float() / 2 + 0.5
        
        # Extract color channels
        red = images[:,0:1]
        green = images[:,1:2]
        blue = images[:,2:3]
        
        # 1. Calculate color harmony based on complementary color theory
        # This optimizes for pleasing color relationships based on the color wheel
        
        # YUV color space components (closer to human perception)
        # Y (luminance), U and V (chrominance)
        y = 0.299 * red + 0.587 * green + 0.114 * blue
        u = 0.492 * (blue - y)
        v = 0.877 * (red - y)
        
        # Complementary color harmony score
        # This encourages colors that are aesthetically pleasing together
        # Higher UV plane distances for interesting color contrasts
        color_harmony = ((u.pow(2) + v.pow(2)).sqrt() * 3.0).mean()
        
        # 2. Luminance harmony - balanced distribution of lights and darks
        # Encourages a balanced histogram of luminance values
        lum_mean = y.mean()
        lum_std = y.std()
        lum_harmony = (
            # Penalize luminance means too far from 0.5 (middle gray)
            torch.abs(lum_mean - 0.5) * 0.5 + 
            # But reward good standard deviation (contrast)
            torch.exp(-((lum_std - 0.2).pow(2) * 10))
        )
        
        # 3. Focal contrast - encourages areas of high local contrast
        focal_contrast = self._calculate_focal_contrast(y)
        
        # 4. Edge coherence - encourage clean edges
        soft_edges = self._calculate_edge_coherence(y)
        
        # Combine the components with weights
        harmony_score = (
            color_harmony * 2.0 +   # Color relationships
            lum_harmony * 1.0 +     # Luminance balance
            focal_contrast * 1.5 +  # Visual interest through contrast
            soft_edges * 0.8        # Edge aesthetics
        )
        
        return harmony_score
    
    def _calculate_focal_contrast(self, luminance):
        """Calculate focal contrast to create visual interest."""
        # Apply average pooling to get local areas
        avg_pool = torch.nn.functional.avg_pool2d(luminance, kernel_size=7, stride=3, padding=3)
        
        # Calculate local contrast in these regions
        local_mean = torch.nn.functional.avg_pool2d(luminance, kernel_size=3, stride=1, padding=1)
        local_var = torch.nn.functional.avg_pool2d(luminance.pow(2), kernel_size=3, stride=1, padding=1) - local_mean.pow(2)
        local_contrast = local_var.sqrt()
        
        # Create a score that rewards areas of high contrast
        return local_contrast.mean() * 5.0
    
    def _calculate_edge_coherence(self, luminance):
        """Calculate edge coherence for pleasing edges."""
        # Simple approximation of a Sobel filter using gradients
        dx = luminance[:,:,:,1:] - luminance[:,:,:,:-1]
        dy = luminance[:,:,1:,:] - luminance[:,:,:-1,:]
        
        # Edge magnitude
        edge_magnitude = (dx.pow(2)[:,:,:,:-1] + dy.pow(2)[:,:,:-1,:]).sqrt()
        
        # We want clean, coherent edges - not noise
        # This rewards stronger, cleaner edges
        return edge_magnitude.mean() * 3.0
    
    def generate_artwork(self, prompt, style_name="watercolor", output_dir="outputs", use_harmony_loss=True):
        """Generate artwork with the given prompt and style."""
        try:
            # Verify style name
            if style_name not in self.style_tokens:
                raise ValueError(f"Style '{style_name}' not found. Available styles: {', '.join(self.style_tokens.keys())}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            style_output_dir = os.path.join(output_dir, style_name)
            os.makedirs(style_output_dir, exist_ok=True)
            
            # Add style token to prompt
            full_prompt = f"{prompt}, {self.style_tokens[style_name]}"
            
            # Set seed for reproducibility
            generator_seed = self.style_seeds.get(style_name, 42)
            torch.manual_seed(generator_seed)
            if self.device == "cuda":
                torch.cuda.manual_seed(generator_seed)
            
            # Create generators
            generator = torch.Generator(self.device).manual_seed(generator_seed)
            
            print(f"Generating {style_name} style artwork...")
            print(f"Prompt: {prompt}")
            print(f"Seed: {generator_seed}")
            
            # Generate base image (without custom loss)
            with autocast(self.device):
                base_image = self.pipeline(
                    full_prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    generator=generator
                ).images[0]
            
            # Generate enhanced image (with harmony loss)
            if use_harmony_loss:
                print("Applying harmony enhancement...")
                with autocast(self.device):
                    enhanced_image = self.pipeline(
                        full_prompt,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        callback=self._harmony_loss,
                        callback_steps=5,
                        generator=torch.Generator(self.device).manual_seed(generator_seed)
                    ).images[0]
            else:
                enhanced_image = base_image
            
            # Save the images
            base_path = os.path.join(style_output_dir, f"{style_name}_base.png")
            enhanced_path = os.path.join(style_output_dir, f"{style_name}_harmony.png")
            
            base_image.save(base_path)
            enhanced_image.save(enhanced_path)
            
            print(f"Images saved to {style_output_dir}")
            return base_image, enhanced_image
            
        except Exception as e:
            print(f"Error generating artwork: {str(e)}")
            raise
    
    def generate_comparison(self, prompt, output_dir="outputs"):
        """Generate artwork with all available styles for comparison."""
        results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate artwork for each style
        for style_name in self.style_tokens.keys():
            print(f"\n===== Processing {style_name.upper()} style =====")
            try:
                base_img, enhanced_img = self.generate_artwork(prompt, style_name, output_dir)
                results[style_name] = (base_img, enhanced_img)
            except Exception as e:
                print(f"Error with {style_name} style: {e}")
        
        # Create a comparison grid
        self._create_comparison_grid(results, prompt, output_dir)
        
        return results
    
    def _create_comparison_grid(self, results, prompt, output_dir):
        """Create a visual comparison grid of all styles."""
        if not results:
            print("No results to display")
            return
        
        # Determine grid size
        n_styles = len(results)
        if n_styles <= 3:
            grid_rows, grid_cols = 1, n_styles
        else:
            grid_rows = (n_styles + 2) // 3
            grid_cols = min(3, n_styles)
        
        # Create figure for base images
        plt.figure(figsize=(grid_cols * 5, grid_rows * 5))
        plt.suptitle(f"Base Images for: \"{prompt}\"", fontsize=16)
        
        # Plot base images
        for i, (style, (base_img, _)) in enumerate(results.items()):
            plt.subplot(grid_rows, grid_cols, i+1)
            plt.imshow(np.array(base_img))
            plt.title(f"{style.title()} Style (Seed: {self.style_seeds[style]})")
            plt.axis('off')
        
        # Save base comparison
        base_comparison_path = os.path.join(output_dir, "base_comparison.png")
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.savefig(base_comparison_path)
        
        # Create figure for harmony enhanced images
        plt.figure(figsize=(grid_cols * 5, grid_rows * 5))
        plt.suptitle(f"Harmony Enhanced Images for: \"{prompt}\"", fontsize=16)
        
        # Plot enhanced images
        for i, (style, (_, enhanced_img)) in enumerate(results.items()):
            plt.subplot(grid_rows, grid_cols, i+1)
            plt.imshow(np.array(enhanced_img))
            plt.title(f"{style.title()} Style with Harmony Loss")
            plt.axis('off')
        
        # Save enhanced comparison
        enhanced_comparison_path = os.path.join(output_dir, "harmony_comparison.png")
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.savefig(enhanced_comparison_path)
        
        print(f"\nComparison grids saved to:")
        print(f"Base styles: {base_comparison_path}")
        print(f"Harmony enhanced: {enhanced_comparison_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate artwork using multiple styles')
    parser.add_argument('prompt', type=str, help='Text prompt for image generation')
    parser.add_argument('--style', type=str, default=None, 
                       help='Specific style to use (watercolor, cyberpunk, vangogh, ukiyoe, retro). If not specified, generates all styles.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save generated images')
    parser.add_argument('--no_harmony', action='store_true', help='Disable harmony enhancement')
    args = parser.parse_args()
    
    # Initialize the generator
    generator = ArtGenerator()
    
    if args.style:
        # Generate artwork with specific style
        generator.generate_artwork(
            args.prompt, 
            args.style, 
            args.output_dir,
            use_harmony_loss=not args.no_harmony
        )
    else:
        # Generate comparison with all styles
        generator.generate_comparison(args.prompt, args.output_dir)

if __name__ == "__main__":
    main() 