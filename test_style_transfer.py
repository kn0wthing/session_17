import os
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Check if embeddings are ready
def check_embeddings():
    embedding_dir = Path("style_embeddings")
    styles = ["watercolor", "cyberpunk", "vangogh", "ukiyoe", "retro"]
    
    available_styles = []
    for style in styles:
        embedding_path = embedding_dir / f"{style}.bin"
        if embedding_path.exists():
            available_styles.append(style)
    
    return available_styles

def generate_image(prompt, style, output_dir="outputs"):
    """Generate an image using the specified style."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Import inside function to avoid loading the model if no styles are available
    from src.generate_art import ArtGenerator
    
    # Initialize the generator
    generator = ArtGenerator()
    
    # Generate the artwork
    output_path = generator.generate_artwork(prompt, style_name=style, output_dir=output_dir)
    
    return output_path

def generate_comparison(prompt, output_dir="outputs"):
    """Generate a comparison of all available styles."""
    # Import inside function to avoid loading the model if no styles are available
    from src.generate_art import ArtGenerator
    
    # Initialize the generator
    generator = ArtGenerator()
    
    # Generate the comparison
    output_path = generator.generate_comparison(prompt, output_dir=output_dir)
    
    return output_path

if __name__ == "__main__":
    # Check which embeddings are available
    available_styles = check_embeddings()
    
    if not available_styles:
        print("No style embeddings found. Please wait for the training to complete.")
        exit(1)
    
    print(f"Available styles: {', '.join(available_styles)}")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test style transfer with trained embeddings")
    parser.add_argument("prompt", type=str, help="Prompt for image generation")
    parser.add_argument("--style", type=str, choices=available_styles, default=available_styles[0],
                        help=f"Style to use. Available: {', '.join(available_styles)}")
    parser.add_argument("--compare", action="store_true", help="Generate a comparison of all styles")
    
    args = parser.parse_args()
    
    # Generate image(s)
    if args.compare:
        print(f"Generating comparison for prompt: \"{args.prompt}\"...")
        output_path = generate_comparison(args.prompt)
        print(f"Comparison saved to: {output_path}")
    else:
        print(f"Generating image with style {args.style} for prompt: \"{args.prompt}\"...")
        output_path = generate_image(args.prompt, args.style)
        print(f"Image saved to: {output_path}")
        
    print("Done!") 