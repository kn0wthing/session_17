# -*- coding: utf-8 -*-
"""
# Diffusion Model Loss Function Comparison

Compare images generated with and without custom loss functions applied.
This notebook is optimized for Google Colab's memory constraints.

## Setup
"""

# @title Install Dependencies
!pip install torch torchvision diffusers transformers Pillow matplotlib

# @title Import Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gc
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel

# @title Memory Optimization Functions
def clear_memory():
  """Clear CUDA memory cache"""
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# @title Memory-Optimized Generation Class
class MemoryOptimizedDiffusion:
  def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
    """Initialize the diffusion model with memory optimizations"""
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.dtype = torch.float16 if self.device == "cuda" else torch.float32
    
    print(f"Using {self.device} with {self.dtype}")
    
    # Load model with memory optimizations
    self.pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=self.dtype,
        safety_checker=None  # Disable safety checker to save memory
    ).to(self.device)
    
    # Use memory efficient attention
    if hasattr(self.pipe, "enable_attention_slicing"):
      self.pipe.enable_attention_slicing()
    
    # Use DDIM scheduler (more stable)
    self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
    
    # Extract components
    self.unet = self.pipe.unet
    self.vae = self.pipe.vae
    self.tokenizer = self.pipe.tokenizer
    self.text_encoder = self.pipe.text_encoder
    self.scheduler = self.pipe.scheduler
    
    # For subject changing loss
    print("Loading CLIP model...")
    self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
    self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
  
  def encode_prompt(self, prompt):
    """Encode text prompt to embeddings"""
    # Tokenize text
    text_input = self.tokenizer(
        [prompt],
        padding="max_length",
        max_length=self.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    # Get text embeddings
    with torch.no_grad():
      text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
    
    # Unconditional embeddings for classifier-free guidance
    uncond_input = self.tokenizer(
        [""],
        padding="max_length",
        max_length=text_input.input_ids.shape[-1],
        return_tensors="pt"
    )
    
    with torch.no_grad():
      uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
    
    # Concatenate embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings
  
  def generate_standard(self, prompt, seed=42, steps=30, height=512, width=512, guidance_scale=7.5):
    """Generate image using standard diffusion (no custom loss)"""
    clear_memory()
    
    # Set seed
    generator = torch.Generator(device=self.device).manual_seed(seed)
    
    # Generate the image
    with torch.no_grad():
      image = self.pipe(
          prompt=prompt,
          height=height,
          width=width,
          num_inference_steps=steps,
          guidance_scale=guidance_scale,
          generator=generator
      ).images[0]
    
    clear_memory()
    return image
  
  def blue_loss(self, images):
    """Loss function that emphasizes blue pixels"""
    return torch.abs(images[:, 2] - 0.9).mean()
  
  def edge_sharpening_loss(self, images):
    """Loss function that enhances edge details"""
    # Convert to grayscale
    gray = 0.2989 * images[:, 0] + 0.5870 * images[:, 1] + 0.1140 * images[:, 2]
    gray = gray.unsqueeze(1)
    
    # Sobel filters for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device).to(self.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device).to(self.dtype).view(1, 1, 3, 3)
    
    # Apply filters
    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)
    
    # Calculate edge magnitude
    edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
    
    # We want to maximize edge magnitude (negative for gradient descent)
    return -torch.mean(edge_magnitude) + 1e-5
    
  def subject_changing_loss(self, images, target_prompt, current_prompt=None):
    """
    Loss function that encourages image to match a different subject.
    Properly processes the decoded image for CLIP.
    """
    # Convert images tensor to format for CLIP processor
    # First move to CPU and convert to numpy (CLIP processor expects numpy array or PIL image)
    with torch.no_grad():
      # Ensure images are in range [0, 1]
      images_np = images.detach().cpu().numpy()
      # Convert to 0-255 uint8 format expected by CLIP
      images_np = (images_np * 255).astype(np.uint8)
      # Create PIL Images 
      pil_images = [Image.fromarray(img.transpose(1, 2, 0)) for img in images_np]
      
      # Process with CLIP
      clip_inputs = self.clip_processor(images=pil_images, return_tensors="pt").to(self.device)
      image_features = self.clip_model.get_image_features(**clip_inputs)
      image_features = image_features / image_features.norm(dim=-1, keepdim=True)
      
      # Process target prompt with CLIP
      target_inputs = self.clip_processor.tokenizer(
          [target_prompt],
          padding="max_length", 
          max_length=77,
          truncation=True, 
          return_tensors="pt"
      ).to(self.device)
      target_features = self.clip_model.get_text_features(**target_inputs)
      target_features = target_features / target_features.norm(dim=-1, keepdim=True)
      
      # Compute similarity
      target_similarity = (image_features * target_features).sum(dim=-1)
      
      # If we have a current prompt, we want to push away from it
      if current_prompt:
        current_inputs = self.clip_processor.tokenizer(
            [current_prompt],
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        current_features = self.clip_model.get_text_features(**current_inputs)
        current_features = current_features / current_features.norm(dim=-1, keepdim=True)
        
        # Compute current similarity
        current_similarity = (image_features * current_features).sum(dim=-1)
        
        # We want to maximize target similarity and minimize current similarity
        return -target_similarity.mean() + current_similarity.mean()
      
      # Just optimize for target similarity
      return -target_similarity.mean()
  
  def generate_with_custom_loss(
      self, 
      prompt, 
      loss_type="blue", 
      loss_scale=100, 
      seed=42, 
      steps=30, 
      height=512, 
      width=512, 
      guidance_scale=7.5,
      apply_every=5,  # Apply loss less frequently to save memory
      target_subject=None  # Only needed for subject_changing loss
  ):
    """Generate image with custom loss function applied during diffusion"""
    clear_memory()
    
    # Set seed
    generator = torch.Generator(device=self.device).manual_seed(seed)
    
    # Select loss function
    if loss_type == "blue":
      loss_fn = self.blue_loss
    elif loss_type == "edge":
      loss_fn = self.edge_sharpening_loss
    elif loss_type == "subject":
      if target_subject is None:
        raise ValueError("target_subject must be provided for subject changing loss")
      loss_fn = lambda img: self.subject_changing_loss(img, target_subject, prompt)
    else:
      raise ValueError(f"Unsupported loss type: {loss_type}")
    
    # Get text embeddings
    text_embeddings = self.encode_prompt(prompt)
    
    # Initialize scheduler
    self.scheduler.set_timesteps(steps)
    
    # Create initial random latents
    latents = torch.randn(
        (1, self.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=self.device,
        dtype=self.dtype
    )
    latents = latents * self.scheduler.init_noise_sigma
    
    # Diffusion loop
    for i, t in tqdm(enumerate(self.scheduler.timesteps)):
      # Free memory
      if i > 0:
        clear_memory()
      
      # Prepare latent input
      latent_model_input = torch.cat([latents] * 2)
      latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
      
      # Predict noise
      with torch.no_grad():
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
      
      # Apply classifier-free guidance
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
      
      # Apply loss function guidance (less frequently to save memory)
      # For subject changing, only start after some initial denoising
      should_apply_loss = i % apply_every == 0
      if loss_type == "subject":
        should_apply_loss = should_apply_loss and i > steps // 4
      
      if should_apply_loss:
        # Enable gradients
        latents = latents.detach().requires_grad_()
        
        # Get predicted denoised image (x0)
        pred_original_sample = self.scheduler.step(noise_pred, t, latents).pred_original_sample
        
        # Decode to image space
        with torch.no_grad():
          denoised_images = self.vae.decode((1 / 0.18215) * pred_original_sample).sample / 2 + 0.5
        
        # Calculate loss
        loss = loss_fn(denoised_images) * loss_scale
        
        # Get gradient
        grad = torch.autograd.grad(loss, latents)[0]
        
        # Update latents
        latents = latents.detach() - grad * 0.1
      
      # Step scheduler
      latents = self.scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode final image
    with torch.no_grad():
      latents = 1 / 0.18215 * latents
      image = self.vae.decode(latents).sample
      
    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")[0]
    image = Image.fromarray(image)
    
    clear_memory()
    return image
  
  def run_comparison(self, prompt, loss_type, loss_scale, target_subject=None, seed=42, steps=30, height=512, width=512):
    """Generate and compare images with and without the loss function"""
    print(f"Running comparison for '{prompt}' with {loss_type} loss (scale={loss_scale})")
    
    # Generate standard image
    print("Generating image WITHOUT custom loss...")
    image_standard = self.generate_standard(
        prompt=prompt,
        seed=seed,
        steps=steps,
        height=height,
        width=width
    )
    
    # Generate with custom loss
    print(f"Generating image WITH {loss_type} loss...")
    image_with_loss = self.generate_with_custom_loss(
        prompt=prompt,
        loss_type=loss_type,
        loss_scale=loss_scale,
        seed=seed,
        steps=steps,
        height=height,
        width=width,
        target_subject=target_subject
    )
    
    # Show comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    ax1.imshow(np.array(image_standard))
    ax1.set_title("Standard Generation", fontsize=14)
    ax1.axis('off')
    
    ax2.imshow(np.array(image_with_loss))
    ax2.set_title(f"With {loss_type.capitalize()} Loss", fontsize=14)
    ax2.axis('off')
    
    plt.suptitle(f'"{prompt}"', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Save comparison
    comparison = Image.new('RGB', (image_standard.width + image_with_loss.width, image_standard.height))
    comparison.paste(image_standard, (0, 0))
    comparison.paste(image_with_loss, (image_standard.width, 0))
    comparison_filename = f"comparison_{loss_type}_{seed}.png"
    comparison.save(comparison_filename)
    print(f"Comparison saved to {comparison_filename}")
    
    return image_standard, image_with_loss

# @title Run Examples
# @markdown Configure parameters for the comparison examples

diffusion = MemoryOptimizedDiffusion()

# @title Blue Emphasis Comparison
prompt = "A sunset over mountains" # @param {type:"string"}
loss_scale = 150 # @param {type:"slider", min:50, max:400, step:25}
steps = 30 # @param {type:"slider", min:20, max:50, step:5}
seed = 42 # @param {type:"integer"}

standard_image, blue_image = diffusion.run_comparison(
    prompt=prompt,
    loss_type="blue",
    loss_scale=loss_scale,
    seed=seed,
    steps=steps
)

# @title Edge Sharpening Comparison
prompt = "Portrait of a woman wearing a red hat" # @param {type:"string"}
loss_scale = 50 # @param {type:"slider", min:10, max:100, step:5}
steps = 30 # @param {type:"slider", min:20, max:50, step:5}
seed = 42 # @param {type:"integer"}

standard_image, edge_image = diffusion.run_comparison(
    prompt=prompt,
    loss_type="edge",
    loss_scale=loss_scale,
    seed=seed,
    steps=steps
)

# @title Subject Changing Comparison
original_prompt = "A golden retriever sitting in a garden" # @param {type:"string"}
target_subject = "A wolf in a snowy forest" # @param {type:"string"}
loss_scale = 300 # @param {type:"slider", min:100, max:800, step:50}
steps = 30 # @param {type:"slider", min:20, max:50, step:5}
seed = 42 # @param {type:"integer"}

standard_image, subject_image = diffusion.run_comparison(
    prompt=original_prompt,
    loss_type="subject",
    target_subject=target_subject,
    loss_scale=loss_scale,
    seed=seed,
    steps=steps
) 