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
    accelerator = Accelerator()
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Load the tokenizer and text encoder
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
    
    # Create the optimizer
    optimizer = torch.optim.AdamW(
        [token_embeds[placeholder_token_id]],
        lr=learning_rate
    )
    
    # Load and process training images
    train_images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512))
        train_images.append(np.array(image))
    train_images = torch.tensor(np.stack(train_images)).permute(0, 3, 1, 2) / 255.0
    
    # Training loop
    for step in range(num_training_steps):
        with accelerator.accumulate():
            loss = train_batch(text_encoder, train_images, placeholder_token, tokenizer)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        
        if step % 100 == 0:
            print(f"Step {step}: Loss {loss.item()}")
    
    # Save the trained embeddings
    learned_embeds = token_embeds[placeholder_token_id].detach().cpu()
    learned_embeds_dict = {placeholder_token: learned_embeds}
    
    output_path = Path(output_dir) / f"{style_name}.bin"
    torch.save(learned_embeds_dict, output_path)
    print(f"Saved style embedding to {output_path}")

def train_batch(text_encoder, images, placeholder_token, tokenizer):
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
    )
    
    neg_input = tokenizer(
        [negative_prompt] * len(images),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    # Get embeddings for both prompts
    pos_embeddings = text_encoder(pos_input.input_ids)[0]
    neg_embeddings = text_encoder(neg_input.input_ids)[0]
    
    # Calculate contrastive loss
    # Push positive embeddings closer to image features while pushing negative embeddings away
    image_features = images.mean(dim=[2, 3])  # Average pool spatial dimensions
    
    # Project image features to match text embedding dimension
    embedding_dim = pos_embeddings.shape[-1]  # Should be 768
    if not hasattr(text_encoder, 'image_projection'):
        text_encoder.image_projection = torch.nn.Linear(image_features.shape[-1], embedding_dim)
        text_encoder.image_projection = text_encoder.image_projection.to(images.device)
    
    # Project and normalize image features
    image_features = text_encoder.image_projection(image_features)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Get mean embeddings for positive and negative prompts
    pos_embeddings = pos_embeddings.mean(dim=1)  # Average across sequence length
    neg_embeddings = neg_embeddings.mean(dim=1)  # Average across sequence length
    
    # Normalize text embeddings
    pos_embeddings = pos_embeddings / pos_embeddings.norm(dim=-1, keepdim=True)
    neg_embeddings = neg_embeddings / neg_embeddings.norm(dim=-1, keepdim=True)
    
    # Compute similarities
    pos_similarity = (image_features * pos_embeddings).sum(dim=-1)
    neg_similarity = (image_features * neg_embeddings).sum(dim=-1)
    
    # Contrastive loss with margin
    margin = 0.2
    loss = torch.relu(neg_similarity - pos_similarity + margin).mean()
    
    # Add L2 regularization to prevent embeddings from growing too large
    l2_reg = 0.01 * (pos_embeddings.pow(2).sum() + neg_embeddings.pow(2).sum())
    
    total_loss = loss + l2_reg
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
        
    train_style(
        style_name=args.style_name,
        image_paths=all_image_paths,
        placeholder_token=f"<{args.style_name}-style>",
        num_training_steps=args.steps,
        learning_rate=args.lr
    )