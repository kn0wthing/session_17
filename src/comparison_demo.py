import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from guided_diffusion import GuidedDiffusion

def generate_comparison(
    diffusion,
    prompt,
    loss_type="edge",
    target_subject=None,
    num_steps=50,
    scale=None,
    seed=42,
    height=512,
    width=512,
):
    """Generate two images with the same seed - one with loss function, one without"""
    
    # Set default scale based on loss type
    if scale is None:
        if loss_type == "edge":
            scale = 50
        elif loss_type == "subject":
            scale = 500
        else:  # blue
            scale = 200
    
    # Function to generate without any loss function applied
    def generate_without_loss(seed):
        """Generate image using just the diffusion model without custom loss"""
        # Set seed
        torch.manual_seed(seed)
        # Use the pipe's generate method directly
        result = diffusion.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=7.5,
            generator=torch.Generator(device=diffusion.device).manual_seed(seed)
        ).images[0]
        return result
    
    # Generate the image without loss guidance
    print(f"Generating image WITHOUT {loss_type} loss guidance...")
    img_without_loss = generate_without_loss(seed)
    
    # Generate the image with the specified loss function
    print(f"Generating image WITH {loss_type} loss guidance...")
    if loss_type == "edge":
        img_with_loss = diffusion.generate_with_edge_sharpening(
            prompt=prompt,
            num_inference_steps=num_steps,
            sharpening_scale=scale,
            seed=seed,
            height=height,
            width=width,
            output_dir=f"comparison_{loss_type}",
            save_steps=False
        )
    elif loss_type == "subject":
        if target_subject is None:
            raise ValueError("Target subject must be provided for subject changing loss")
        img_with_loss = diffusion.generate_with_subject_changing(
            prompt=prompt,
            target_subject=target_subject,
            num_inference_steps=num_steps,
            subject_change_scale=scale,
            seed=seed,
            height=height,
            width=width,
            output_dir=f"comparison_{loss_type}",
            save_steps=False
        )
    elif loss_type == "blue":
        img_with_loss = diffusion.generate_with_blue(
            prompt=prompt,
            num_inference_steps=num_steps,
            blue_scale=scale,
            seed=seed,
            height=height,
            width=width,
            output_dir=f"comparison_{loss_type}",
            save_steps=False
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return img_without_loss, img_with_loss

def display_comparison(img_without_loss, img_with_loss, title="", loss_type=""):
    """Display two images side by side for comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.imshow(np.array(img_without_loss))
    ax1.set_title("Without Loss Function", fontsize=16)
    ax1.axis('off')
    
    ax2.imshow(np.array(img_with_loss))
    ax2.set_title(f"With {loss_type.title()} Loss", fontsize=16)
    ax2.axis('off')
    
    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()
    
    # Save the comparison
    comparison = Image.new('RGB', (img_without_loss.width + img_with_loss.width, max(img_without_loss.height, img_with_loss.height)))
    comparison.paste(img_without_loss, (0, 0))
    comparison.paste(img_with_loss, (img_without_loss.width, 0))
    comparison.save(f"comparison_{loss_type}.png")
    print(f"Comparison saved to comparison_{loss_type}.png")

def main():
    # Initialize diffusion model
    diffusion = GuidedDiffusion()
    
    # Edge Sharpening Comparison
    print("\n=== Edge Sharpening Comparison ===")
    prompt = "Portrait of a woman wearing a red hat"
    img_without_loss, img_with_loss = generate_comparison(
        diffusion,
        prompt=prompt,
        loss_type="edge",
        num_steps=50,
        scale=50,
        seed=42
    )
    display_comparison(
        img_without_loss, 
        img_with_loss, 
        title=f'"{prompt}"', 
        loss_type="edge sharpening"
    )
    
    # Subject Changing Comparison
    print("\n=== Subject Changing Comparison ===")
    prompt = "A golden retriever sitting in a garden"
    target_subject = "A wolf in a snowy forest"
    img_without_loss, img_with_loss = generate_comparison(
        diffusion,
        prompt=prompt,
        loss_type="subject",
        target_subject=target_subject,
        num_steps=50,
        scale=500,
        seed=42
    )
    display_comparison(
        img_without_loss, 
        img_with_loss, 
        title=f'"{prompt}" → "{target_subject}"', 
        loss_type="subject changing"
    )
    
    # Blue Pixel Emphasis Comparison
    print("\n=== Blue Pixel Emphasis Comparison ===")
    prompt = "A sunset over mountains"
    img_without_loss, img_with_loss = generate_comparison(
        diffusion,
        prompt=prompt,
        loss_type="blue",
        num_steps=50,
        scale=200,
        seed=42
    )
    display_comparison(
        img_without_loss, 
        img_with_loss, 
        title=f'"{prompt}"', 
        loss_type="blue pixel"
    )

if __name__ == "__main__":
    main() 