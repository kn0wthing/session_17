import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from torch import autocast
import os
from pathlib import Path
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_scheduler
from PIL import Image
import numpy as np
import glob
import argparse
import tqdm

def train_style(
    style_name,
    image_paths,
    placeholder_token="<new-style>",
    initializer_token="art",
    num_training_steps=3000,
    learning_rate=1e-4,
    train_batch_size=1,
    gradient_accumulation_steps=4,
    output_dir="style_embeddings"
):
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16" if torch.cuda.is_available() else "no"
    )
    
    device = accelerator.device
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the tokenizer and text encoder
    model_id = "runwayml/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    
    # Add the placeholder token to the tokenizer
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    
    # Resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
    
    # Move text encoder to device
    text_encoder = text_encoder.to(device)
    
    # Create image projection layer
    # Load and process training images
    train_images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512))
        train_images.append(np.array(image))
    
    train_images = torch.tensor(np.stack(train_images)).permute(0, 3, 1, 2) / 255.0
    train_images = train_images.to(device, dtype=torch.float32)
    
    # Create image projection layer
    image_features = train_images.mean(dim=[2, 3])  # Average pool spatial dimensions
    embedding_dim = text_encoder.get_input_embeddings().weight.shape[1]  # Should be 768
    
    # Create image projection as a proper module
    image_projection = torch.nn.Linear(image_features.shape[1], embedding_dim)
    image_projection = image_projection.to(device)
    
    # Create the optimizer with all trainable parameters
    params_to_optimize = [
        {"params": [token_embeds[placeholder_token_id]], "lr": learning_rate},
        {"params": image_projection.parameters(), "lr": learning_rate}
    ]
    
    optimizer = torch.optim.AdamW(params_to_optimize)
    
    # Prepare everything with accelerator
    text_encoder, image_projection, optimizer = accelerator.prepare(
        text_encoder, image_projection, optimizer
    )
    
    # Training loop
    progress_bar = tqdm.tqdm(range(num_training_steps), desc="Training")
    for step in range(num_training_steps):
        text_encoder.train()
        
        with accelerator.accumulate(text_encoder):
            # Forward pass
            loss = train_batch(text_encoder, train_images, placeholder_token, tokenizer, image_projection, device)
            
            # Backward pass
            accelerator.backward(loss)
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
        
        # Log progress
        if step % 100 == 0 or step == num_training_steps - 1:
            print(f"Step {step}: Loss {loss.item()}")
        
        progress_bar.update(1)
    
    # Save the trained embeddings
    accelerator.wait_for_everyone()
    
    # Get the unwrapped model
    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
    
    # Get the trained token embedding
    learned_embeds = unwrapped_text_encoder.get_input_embeddings().weight[placeholder_token_id].detach().cpu()
    learned_embeds_dict = {placeholder_token: learned_embeds}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the embedding
    output_path = Path(output_dir) / f"{style_name}.bin"
    torch.save(learned_embeds_dict, output_path)
    print(f"Saved style embedding to {output_path}")

def train_batch(text_encoder, images, placeholder_token, tokenizer, image_projection, device):
    # Create positive and negative prompts
    positive_prompt = f"a photo in {placeholder_token} style"
    negative_prompt = "low quality, blurry, distorted"
    
    # Tokenize both prompts
    pos_input = tokenizer(
        [positive_prompt] * len(images),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    neg_input = tokenizer(
        [negative_prompt] * len(images),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Get embeddings for both prompts
    with torch.set_grad_enabled(True):
        pos_embeddings = text_encoder(pos_input.input_ids)[0]
        neg_embeddings = text_encoder(neg_input.input_ids)[0]
        
        # Calculate contrastive loss
        # Push positive embeddings closer to image features while pushing negative embeddings away
        image_features = images.mean(dim=[2, 3])  # Average pool spatial dimensions
        
        # Project and normalize image features
        projected_image_features = image_projection(image_features)
        projected_image_features = projected_image_features / projected_image_features.norm(dim=-1, keepdim=True)
        
        # Get mean embeddings for positive and negative prompts
        pos_embeddings = pos_embeddings.mean(dim=1)  # Average across sequence length
        neg_embeddings = neg_embeddings.mean(dim=1)  # Average across sequence length
        
        # Normalize text embeddings
        pos_embeddings = pos_embeddings / pos_embeddings.norm(dim=-1, keepdim=True)
        neg_embeddings = neg_embeddings / neg_embeddings.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        pos_similarity = (projected_image_features * pos_embeddings).sum(dim=-1)
        neg_similarity = (projected_image_features * neg_embeddings).sum(dim=-1)
        
        # Contrastive loss with margin
        margin = 0.2
        contrastive_loss = torch.relu(neg_similarity - pos_similarity + margin).mean()
        
        # Add L2 regularization to prevent embeddings from growing too large
        l2_reg = 0.01 * (pos_embeddings.pow(2).sum() + neg_embeddings.pow(2).sum())
        
        total_loss = contrastive_loss + l2_reg
        
        return total_loss

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a specific style for Stable Diffusion')
    parser.add_argument('style_name', type=str, help='Name of the style to train (e.g., dhoni, mickey_mouse, balloon, lion_king, rose_flower)')
    parser.add_argument('--steps', type=int, default=3000, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    
    styles_to_train = {
        "dhoni": ["training_images/dhoni/*.jpg", "training_images/dhoni/*.png"],
        "mickey_mouse": ["training_images/mickey_mouse/*.jpg", "training_images/mickey_mouse/*.png"],
        "balloon": ["training_images/balloon/*.jpg", "training_images/balloon/*.png"],
        "lion_king": ["training_images/lion_king/*.jpg", "training_images/lion_king/*.png"],
        "rose_flower": ["training_images/rose_flower/*.jpg", "training_images/rose_flower/*.png"]
    }
    
    if args.style_name not in styles_to_train:
        print(f"Error: Style '{args.style_name}' not found. Available styles: {', '.join(styles_to_train.keys())}")
        exit(1)
    
    print(f"\nTraining {args.style_name} style...")
    all_image_paths = []
    for pattern in styles_to_train[args.style_name]:
        all_image_paths.extend(glob.glob(pattern))
    
    if not all_image_paths:
        print(f"Warning: No images found for {args.style_name} style")
        exit(1)
    
    print(f"Found {len(all_image_paths)} images for training")
        
    train_style(
        style_name=args.style_name,
        image_paths=all_image_paths,
        placeholder_token=f"<{args.style_name}-style>",
        num_training_steps=args.steps,
        learning_rate=args.lr
    )