# -*- coding: utf-8 -*-
"""
# Guided Diffusion with Custom Loss Functions

This notebook demonstrates how to use custom loss functions with Stable Diffusion models to guide image generation.

**Features:**
* Edge sharpening - Enhances edge details in the generated image
* Subject changing - Morphs the content toward a different target subject
* Blue pixel emphasis - Increases blue components in the generated image

## Setup and Installation
"""

# @title Install dependencies
# @markdown Run this cell to install all required dependencies
!pip install torch torchvision torchaudio
!pip install diffusers transformers accelerate
!pip install Pillow numpy tqdm

"""## Import Libraries"""

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import torchvision.transforms as T
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
import os
import argparse
import matplotlib.pyplot as plt
import random

"""## Helper Functions"""

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: make operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""## Guided Diffusion Implementation"""

class GuidedDiffusion:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load models
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id).to(self.device)
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.scheduler = self.pipe.scheduler
        
        # For subject changing loss
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # CLIP text tokenizer - use the proper class directly
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
    def set_timesteps(self, num_inference_steps):
        """Properly set up the scheduler timesteps for inference"""
        if hasattr(self.scheduler, "set_timesteps"):
            # Special handling for PNDM scheduler
            if hasattr(self.scheduler, "config") and hasattr(self.scheduler.config, "skip_prk_steps"):
                # Check if it's a PNDM scheduler
                if self.scheduler.__class__.__name__ == "PNDMScheduler":
                    print("Detected PNDM scheduler - replacing with DDIM for better compatibility")
                    # Import DDIM scheduler
                    from diffusers import DDIMScheduler
                    # Create a DDIM scheduler with same config settings
                    ddim_config = {
                        "num_train_timesteps": self.scheduler.config.num_train_timesteps,
                        "beta_start": self.scheduler.config.beta_start,
                        "beta_end": self.scheduler.config.beta_end,
                        "beta_schedule": self.scheduler.config.beta_schedule,
                        "clip_sample": self.scheduler.config.clip_sample,
                    }
                    self.scheduler = DDIMScheduler.from_config(ddim_config)
                    self.pipe.scheduler = self.scheduler
                    
            # Call the scheduler's set_timesteps to initialize internal parameters properly
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        else:
            # For older scheduler versions
            self.scheduler.num_inference_steps = num_inference_steps
        
        # Store the timesteps
        self.timesteps = self.scheduler.timesteps
        
        # Verify the scheduler is properly set up for our use case
        if hasattr(self.scheduler, "alphas_cumprod") and len(self.scheduler.alphas_cumprod) < num_inference_steps:
            print("Warning: Scheduler's alphas_cumprod length doesn't match num_inference_steps")
            print(f"Rebuilding scheduler with {num_inference_steps} steps...")
            # Reinitialize the scheduler to ensure proper arrays
            self.pipe.scheduler = type(self.scheduler).from_config(self.scheduler.config)
            self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
            self.scheduler = self.pipe.scheduler
            self.timesteps = self.scheduler.timesteps

    def latents_to_pil(self, latents):
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(img) for img in images]
        return pil_images
    
    def blue_loss(self, images):
        # Simple blue loss example - encourages the image to have more blue
        error = torch.abs(images[:, 2] - 0.9).mean()
        return error
    
    def edge_sharpening_loss(self, images):
        """
        A loss function that encourages sharper edges in the image.
        Uses a Sobel filter to detect edges and maximizes their magnitude.
        """
        # Convert images to grayscale for edge detection
        gray = 0.2989 * images[:, 0] + 0.5870 * images[:, 1] + 0.1140 * images[:, 2]
        gray = gray.unsqueeze(1)
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device).float().view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device).float().view(1, 1, 3, 3)
        
        # Apply Sobel filters
        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)
        
        # Calculate edge magnitude
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        
        # We want to maximize edge magnitude, so we negate the mean
        # Add a small epsilon to prevent division by zero
        return -torch.mean(edge_magnitude) + 1e-5
    
    def noop_loss(self, images):
        """
        A loss function that does nothing (returns a constant).
        Used to compare against actual loss functions with identical computation paths.
        """
        # Return a very small constant value that won't affect the generation
        # BUT we need to make sure it maintains the gradient path by using the input tensor
        return torch.sum(images * 0) + torch.tensor(1e-10, device=self.device, requires_grad=True)
    
    def subject_changing_loss(self, images, target_subject, current_subject=None, step_idx=0, total_steps=100):
        """
        Advanced subject changing loss that uses multiple techniques to transform the subject:
        1. Concept blending - using multiple target prompts related to the target subject
        2. Directional CLIP guidance - pushing away from source, towards target
        3. Progressive transformation - increasing strength over time
        4. Feature matching - targeting specific features of the target concept
        """
        # Clone the images to avoid modifying the original
        images_for_clip = images.detach().clone()
        
        # Convert to numpy for PIL conversion
        images_np = images_for_clip.cpu().numpy()
        # Scale to 0-255 range
        images_np = (images_np * 255).round().astype("uint8")
        # Convert to PIL - CLIP expects HWC format
        pil_images = [Image.fromarray(img.transpose(1, 2, 0)) for img in images_np]
        
        # Process images with CLIP
        clip_inputs = self.clip_processor(images=pil_images, return_tensors="pt").to(self.device)
        
        # Create attention map for the current subject (what we want to transform)
        if current_subject:
            # We need to detach pixel_values and set requires_grad to True
            pixel_values = clip_inputs.pixel_values.detach().clone()
            pixel_values.requires_grad_(True)
            
            # Process through CLIP model with gradients enabled
            with torch.enable_grad():
                # Get CLIP features
                image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Get current subject text features
                current_text = self.clip_tokenizer(
                    current_subject, 
                    padding="max_length", 
                    max_length=77, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(self.device)
                text_features = self.clip_model.get_text_features(**current_text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity and backprop to get attention map
                sim = (image_features * text_features).sum(dim=-1)
                sim.backward(retain_graph=True)
                
                # Create attention map from gradients
                current_attn_map = pixel_values.grad.abs().sum(dim=1, keepdim=True)
                
                # Normalize to [0, 1]
                current_attn_map = (current_attn_map - current_attn_map.min()) / (current_attn_map.max() - current_attn_map.min() + 1e-8)
                current_attn_map = current_attn_map.detach()
        else:
            # If no current subject is provided, use a uniform attention map
            current_attn_map = torch.ones_like(images[:, :1])
        
        # === TECHNIQUE 1: CONCEPT BLENDING ===
        # Use multiple prompts related to the target subject for richer guidance
        tiger_concept_prompts = [
            target_subject,                  # Base concept
            f"{target_subject} face",        # Face details
            f"{target_subject} fur pattern", # Fur/pattern
            f"{target_subject} stripes",     # Specific tiger feature
            f"{target_subject} in wild"      # Natural context
        ]
        
        # Calculate text features for all concept prompts
        concept_features = []
        for concept in tiger_concept_prompts:
            concept_text = self.clip_tokenizer(
                concept, 
                padding="max_length", 
                max_length=77, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                features = self.clip_model.get_text_features(**concept_text)
                features = features / features.norm(dim=-1, keepdim=True)
                concept_features.append(features)
        
        # === TECHNIQUE 2: DIRECTIONAL GUIDANCE ===
        # Get cat concept (what to move away from)
        if current_subject:
            cat_concept_prompts = [
                current_subject,
                f"{current_subject} face",
                f"{current_subject} fur",
                "domestic pet",
                "feline pet"
            ]
            
            # Calculate text features for cat concepts
            cat_features = []
            for concept in cat_concept_prompts:
                concept_text = self.clip_tokenizer(
                    concept, 
                    padding="max_length", 
                    max_length=77, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    features = self.clip_model.get_text_features(**concept_text)
                    features = features / features.norm(dim=-1, keepdim=True)
                    cat_features.append(features)
        
        # Create a differentiable path to the original image
        # Use the attention map to focus the guidance
        subject_mask = F.interpolate(current_attn_map, size=images.shape[2:], mode='bilinear')
        
        # Apply the mask with a minimum influence to ensure some guidance everywhere
        # This helps transform small details outside the main attention area
        min_influence = 0.3
        effective_mask = subject_mask.clamp(min=min_influence)
        
        # Apply mask to create a dummy op for gradient path
        masked_images = images * effective_mask
        dummy_op = (masked_images * 0.0).sum()
        
        # Calculate image features for similarity
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**clip_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # === TECHNIQUE 3: PROGRESSIVE TRANSFORMATION ===
        # Calculate progress factor - start gentle, then get stronger
        progress = step_idx / total_steps
        # S-curve for smooth progression (slower start, faster middle, slower end)
        s_curve_progress = 1.0 / (1.0 + np.exp(-12 * (progress - 0.5)))

        # Calculate loss with all the concepts
        target_loss = 0
        target_sim_total = 0
        
        # Weight each concept differently
        concept_weights = [1.0, 0.8, 0.7, 0.7, 0.5]  # Main concept has highest weight
        
        # Calculate weighted similarity to all target concepts
        for i, concept_feature in enumerate(concept_features):
            sim = (image_features * concept_feature).sum(dim=-1)
            target_sim_total += sim.item() * concept_weights[i]
            # Negative because we want to maximize similarity
            target_loss -= sim.mean() * concept_weights[i]
        
        # Add directional loss (push away from cat concepts)
        if current_subject:
            cat_weights = [0.8, 0.7, 0.5, 0.3, 0.3]  # Weights for cat concepts
            for i, cat_feature in enumerate(cat_features):
                # This is positive because we want to minimize similarity to cat
                target_loss += (image_features * cat_feature).sum(dim=-1).mean() * cat_weights[i]
        
        # Apply progressive scaling - increase effect over time
        scaled_loss = target_loss * (0.5 + 1.5 * s_curve_progress)
        
        # Add the dummy op to maintain gradient path
        final_loss = scaled_loss + dummy_op
        
        # For debugging: return average similarity across all target concepts
        avg_target_sim = target_sim_total / sum(concept_weights[:len(concept_features)])
        
        return final_loss, avg_target_sim
    
    def generate_with_edge_sharpening(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        sharpening_scale=50,
        apply_every=1,
        seed=None,
        output_dir="edge_sharp_steps",
        save_steps=True
    ):
        """Generate an image with edge sharpening guidance"""
        import os
        if save_steps and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set seed for reproducibility
        if seed is not None:
            set_seed(seed)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        else:
            generator = None
        
        # Prep text embeddings
        text_input = self.tokenizer(
            [prompt], 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        # Uncond embeddings for classifier-free guidance
        uncond_input = self.tokenizer(
            [""], 
            padding="max_length", 
            max_length=text_input.input_ids.shape[-1], 
            return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prep scheduler
        self.set_timesteps(num_inference_steps)
        
        # Prep latents
        latents = torch.randn(
            (1, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device
        )
        latents = latents * self.scheduler.init_noise_sigma
        
        # Loop
        for i, t in tqdm(enumerate(self.timesteps), total=len(self.timesteps)):
            # expand the latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            sigma = self.scheduler.sigmas[i] if hasattr(self.scheduler, "sigmas") else None
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Apply edge sharpening guidance
            if i % apply_every == 0:
                # Requires grad on the latents
                latents = latents.detach().requires_grad_()
                
                # Get the predicted x0
                if sigma is not None:
                    latents_x0 = latents - sigma * noise_pred
                else:
                    # Handle different scheduler output formats
                    step_output = self.scheduler.step(noise_pred, t, latents)
                    if hasattr(step_output, "pred_original_sample"):
                        latents_x0 = step_output.pred_original_sample
                    elif isinstance(step_output, dict) and "pred_original_sample" in step_output:
                        latents_x0 = step_output["pred_original_sample"]
                    else:
                        # Fallback: manually compute the predicted original sample
                        if hasattr(self.scheduler, "alphas_cumprod"):
                            alpha_prod_t = self.scheduler.alphas_cumprod[i] if i < len(self.scheduler.alphas_cumprod) else self.scheduler.alphas_cumprod[-1]
                            latents_x0 = latents - noise_pred * (1.0 / alpha_prod_t**0.5)
                        else:
                            # Very basic approximation if we can't find alphas_cumprod
                            latents_x0 = latents - noise_pred * (1.0 / 0.18215)
                
                # Decode to image space
                denoised_images = self.vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5  # range (0, 1)
                
                # Calculate sharpening loss
                loss = self.edge_sharpening_loss(denoised_images) * sharpening_scale
                
                # Occasionally print loss
                if i % 10 == 0:
                    print(f"Step {i}, Sharpening Loss: {loss.item()}")
                
                # Get gradient
                grad = torch.autograd.grad(loss, latents)[0]
                
                # Modify the latents based on the gradient
                if sigma is not None:
                    latents = latents.detach() - grad * sigma**2
                else:
                    latents = latents.detach() - grad * 0.1  # Smaller step size if sigma not available
            
            # Step with scheduler
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Save intermediate result
            if save_steps and output_dir and i % 5 == 0:
                images = self.latents_to_pil(latents)
                images[0].save(f"{output_dir}/{i:04d}.png")
        
        # Decode the final latents
        final_images = self.latents_to_pil(latents)
        if output_dir:
            final_images[0].save(f"{output_dir}/final.png")
        
        return final_images[0]
    
    def generate_with_subject_changing(
        self,
        prompt,
        target_subject,
        source_subject=None,  # Optional explicit source subject to replace
        height=512,
        width=512,
        num_inference_steps=150,  # More steps for better transformation
        guidance_scale=7.5,
        subject_change_scale=2000,  # Much higher scale for stronger effect
        apply_after_step=5,   # Start earlier for longer effect
        step_size_factor=10.0,  # Stronger gradient updates
        color_preserve_strength=0.3,  # How much to preserve original colors
        advanced_guidance=True,  # Enable advanced guidance techniques
        seed=None,
        output_dir="subject_change_steps",
        save_steps=True
    ):
        """Generate an image with advanced subject changing guidance"""
        import os
        if save_steps and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set seed for reproducibility
        if seed is not None:
            set_seed(seed)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        else:
            generator = None
            
        # If source subject is not specified, try to extract it from the prompt
        if source_subject is None:
            # Try to identify potential subjects in the prompt
            prompt_parts = prompt.lower().replace(',', ' ').split()
            common_subjects = ['dog', 'cat', 'person', 'man', 'woman', 'car', 'house', 
                            'bird', 'tree', 'mountain', 'river', 'ocean', 'building']
            
            for subject in common_subjects:
                if subject in prompt_parts:
                    source_subject = subject
                    break
        
        print(f"Generating image with advanced subject changing")
        print(f"Full prompt: '{prompt}'")
        if source_subject:
            print(f"Identified source subject: '{source_subject}'")
        print(f"Target subject: '{target_subject}'")
        print(f"Using subject_change_scale={subject_change_scale}, step_size_factor={step_size_factor}")
        
        # Prep text embeddings for the original prompt
        text_input = self.tokenizer(
            [prompt], 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        # Uncond embeddings for classifier-free guidance
        uncond_input = self.tokenizer(
            [""], 
            padding="max_length", 
            max_length=text_input.input_ids.shape[-1], 
            return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Combine original prompt with target subject for better coherence
        # This helps maintain the scene/context while changing the subject
        combined_prompt = f"{prompt} with {target_subject} instead of {source_subject}"
        combined_text_input = self.tokenizer(
            [combined_prompt], 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        with torch.no_grad():
            combined_text_embeddings = self.text_encoder(combined_text_input.input_ids.to(self.device))[0]
            combined_text_embeddings = torch.cat([uncond_embeddings, combined_text_embeddings])
        
        # Prep scheduler
        self.set_timesteps(num_inference_steps)
        
        # Prep latents
        latents = torch.randn(
            (1, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device
        )
        latents = latents * self.scheduler.init_noise_sigma
        
        # Track similarity over time
        similarities = []
        
        # Loop through diffusion steps
        for i, t in tqdm(enumerate(self.timesteps), total=len(self.timesteps)):
            # Adaptive text embedding - start with original, gradually move to combined
            # This creates a smoother transition and maintains scene coherence
            blend_factor = min(1.0, i / (num_inference_steps * 0.3))
            current_text_embeddings = text_embeddings * (1 - blend_factor) + combined_text_embeddings * blend_factor
            
            # Expand the latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            sigma = self.scheduler.sigmas[i] if hasattr(self.scheduler, "sigmas") else None
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=current_text_embeddings)["sample"]
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # Adaptive classifier-free guidance - stronger in the middle of the process
            adaptive_scale = guidance_scale * (1.0 + 0.5 * np.sin(np.pi * i / num_inference_steps))
            noise_pred = noise_pred_uncond + adaptive_scale * (noise_pred_text - noise_pred_uncond)
            
            # Apply subject changing guidance - start early for more effect
            if i > apply_after_step:
                # Requires grad on the latents
                latents = latents.detach().requires_grad_()
                
                # Get the predicted x0
                if sigma is not None:
                    latents_x0 = latents - sigma * noise_pred
                else:
                    # Handle different scheduler output formats
                    step_output = self.scheduler.step(noise_pred, t, latents)
                    if hasattr(step_output, "pred_original_sample"):
                        latents_x0 = step_output.pred_original_sample
                    elif isinstance(step_output, dict) and "pred_original_sample" in step_output:
                        latents_x0 = step_output["pred_original_sample"]
                    else:
                        # Fallback: manually compute the predicted original sample
                        if hasattr(self.scheduler, "alphas_cumprod"):
                            alpha_prod_t = self.scheduler.alphas_cumprod[i] if i < len(self.scheduler.alphas_cumprod) else self.scheduler.alphas_cumprod[-1]
                            latents_x0 = latents - noise_pred * (1.0 / alpha_prod_t**0.5)
                        else:
                            # Very basic approximation if we can't find alphas_cumprod
                            latents_x0 = latents - noise_pred * (1.0 / 0.18215)
                
                # Decode to image space
                denoised_images = self.vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5  # range (0, 1)
                
                # Calculate subject change loss with advanced guidance
                loss, similarity = self.subject_changing_loss(
                    denoised_images,
                    target_subject=target_subject,
                    current_subject=source_subject,
                    step_idx=i,
                    total_steps=num_inference_steps
                )
                
                # Scale the loss - use adaptive scaling that increases in middle steps
                progress = i / num_inference_steps
                adaptive_scale = 1.0
                if progress < 0.3:
                    # Ramp up
                    adaptive_scale = progress / 0.3
                elif progress > 0.7:
                    # Ramp down
                    adaptive_scale = (1.0 - progress) / 0.3
                
                scaled_loss = loss * subject_change_scale * adaptive_scale
                
                # Track similarity
                similarities.append(similarity)
                
                # Occasionally print loss and similarity
                if i % 10 == 0:
                    print(f"Step {i}, Similarity with '{target_subject}': {similarity:.4f}")
                
                # Get gradient
                grad = torch.autograd.grad(scaled_loss, latents)[0]
                
                # Apply stronger gradient update with step size scheduling
                # Smaller steps at beginning and end, larger in middle
                progress_factor = 4 * progress * (1 - progress)
                step_size = step_size_factor * (0.1 if sigma is None else sigma**2) * progress_factor
                latents = latents.detach() - grad * step_size
            
            # Step with scheduler
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Save intermediate result
            if save_steps and output_dir and i % 10 == 0:
                images = self.latents_to_pil(latents)
                images[0].save(f"{output_dir}/{i:04d}.png")
        
        # Save final image
        final_images = self.latents_to_pil(latents)
        if output_dir:
            final_images[0].save(f"{output_dir}/final.png")
            
            # Plot the similarity progression
            if len(similarities) > 0:
                plt.figure(figsize=(10, 5))
                plt.plot(similarities)
                plt.title(f"Similarity with target '{target_subject}' over time")
                plt.xlabel("Diffusion Step")
                plt.ylabel("CLIP Similarity")
                plt.grid(True)
                plt.savefig(f"{output_dir}/similarity_plot.png")
                plt.close()
        
        print(f"Finished subject change from '{prompt}' to '{target_subject}'")
        return final_images[0]
    
    def generate_with_blue(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        blue_scale=200,
        apply_every=5,
        seed=None,
        output_dir="blue_output",
        save_steps=True
    ):
        """Generate an image with blue pixel emphasis"""
        import os
        if save_steps and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set seed for reproducibility
        if seed is not None:
            set_seed(seed)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        else:
            generator = None
        
        # Prep text embeddings
        text_input = self.tokenizer(
            [prompt], 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        # Uncond embeddings for classifier-free guidance
        uncond_input = self.tokenizer(
            [""], 
            padding="max_length", 
            max_length=text_input.input_ids.shape[-1], 
            return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prep scheduler
        self.set_timesteps(num_inference_steps)
        
        # Prep latents
        latents = torch.randn(
            (1, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device
        )
        latents = latents * self.scheduler.init_noise_sigma
        
        # Loop
        for i, t in tqdm(enumerate(self.timesteps), total=len(self.timesteps)):
            # expand the latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            sigma = self.scheduler.sigmas[i] if hasattr(self.scheduler, "sigmas") else None
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Apply blue guidance
            if i % apply_every == 0:
                # Requires grad on the latents
                latents = latents.detach().requires_grad_()
                
                # Get the predicted x0
                if sigma is not None:
                    latents_x0 = latents - sigma * noise_pred
                else:
                    # Handle different scheduler output formats
                    step_output = self.scheduler.step(noise_pred, t, latents)
                    if hasattr(step_output, "pred_original_sample"):
                        latents_x0 = step_output.pred_original_sample
                    elif isinstance(step_output, dict) and "pred_original_sample" in step_output:
                        latents_x0 = step_output["pred_original_sample"]
                    else:
                        # Fallback: manually compute the predicted original sample
                        if hasattr(self.scheduler, "alphas_cumprod"):
                            alpha_prod_t = self.scheduler.alphas_cumprod[i] if i < len(self.scheduler.alphas_cumprod) else self.scheduler.alphas_cumprod[-1]
                            latents_x0 = latents - noise_pred * (1.0 / alpha_prod_t**0.5)
                        else:
                            # Very basic approximation if we can't find alphas_cumprod
                            latents_x0 = latents - noise_pred * (1.0 / 0.18215)
                
                # Decode to image space
                denoised_images = self.vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5  # range (0, 1)
                
                # Calculate blue loss
                loss = self.blue_loss(denoised_images) * blue_scale
                
                # Occasionally print loss
                if i % 10 == 0:
                    print(f"Step {i}, Blue Loss: {loss.item()}")
                
                # Get gradient
                grad = torch.autograd.grad(loss, latents)[0]
                
                # Modify the latents based on the gradient
                if sigma is not None:
                    latents = latents.detach() - grad * sigma**2
                else:
                    latents = latents.detach() - grad * 0.1  # Smaller step size if sigma not available
            
            # Step with scheduler
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Save intermediate result
            if save_steps and output_dir and i % 5 == 0:
                images = self.latents_to_pil(latents)
                images[0].save(f"{output_dir}/{i:04d}.png")
        
        # Decode the final latents
        final_images = self.latents_to_pil(latents)
        if output_dir:
            final_images[0].save(f"{output_dir}/final.png")
        
        return final_images[0]

    def generate_with_noop(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=None,
        output_dir="noop_steps",
        save_steps=True
    ):
        """Generate an image with a no-op loss function that follows the same code path as other losses"""
        import os
        if save_steps and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set seed for reproducibility
        if seed is not None:
            set_seed(seed)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        else:
            generator = None
        
        # Prep text embeddings
        text_input = self.tokenizer(
            [prompt], 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        # Uncond embeddings for classifier-free guidance
        uncond_input = self.tokenizer(
            [""], 
            padding="max_length", 
            max_length=text_input.input_ids.shape[-1], 
            return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prep scheduler
        self.set_timesteps(num_inference_steps)
        
        # Prep latents
        latents = torch.randn(
            (1, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device
        )
        latents = latents * self.scheduler.init_noise_sigma
        
        # Loop
        for i, t in tqdm(enumerate(self.timesteps), total=len(self.timesteps)):
            # expand the latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            sigma = self.scheduler.sigmas[i] if hasattr(self.scheduler, "sigmas") else None
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Apply the noop loss using the same computation path as real losses
            if i % 5 == 0:  # Apply every 5 steps like other losses
                # Requires grad on the latents
                latents = latents.detach().requires_grad_()
                
                # Get the predicted x0
                if sigma is not None:
                    latents_x0 = latents - sigma * noise_pred
                else:
                    # Handle different scheduler output formats
                    step_output = self.scheduler.step(noise_pred, t, latents)
                    if hasattr(step_output, "pred_original_sample"):
                        latents_x0 = step_output.pred_original_sample
                    elif isinstance(step_output, dict) and "pred_original_sample" in step_output:
                        latents_x0 = step_output["pred_original_sample"]
                    else:
                        # Fallback: manually compute the predicted original sample
                        if hasattr(self.scheduler, "alphas_cumprod"):
                            alpha_prod_t = self.scheduler.alphas_cumprod[i] if i < len(self.scheduler.alphas_cumprod) else self.scheduler.alphas_cumprod[-1]
                            latents_x0 = latents - noise_pred * (1.0 / alpha_prod_t**0.5)
                        else:
                            # Very basic approximation if we can't find alphas_cumprod
                            latents_x0 = latents - noise_pred * (1.0 / 0.18215)
                
                # Decode to image space
                denoised_images = self.vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5  # range (0, 1)
                
                # Calculate noop loss
                loss = self.noop_loss(denoised_images)
                
                # Occasionally print loss
                if i % 10 == 0:
                    print(f"Step {i}, Noop Loss: {loss.item()}")
                
                # Get gradient
                grad = torch.autograd.grad(loss, latents)[0]
                
                # Modify the latents based on the gradient
                if sigma is not None:
                    latents = latents.detach() - grad * sigma**2
                else:
                    latents = latents.detach() - grad * 0.1  # Smaller step size if sigma not available
            
            # Step with scheduler
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Save intermediate result
            if save_steps and output_dir and i % 5 == 0:
                images = self.latents_to_pil(latents)
                images[0].save(f"{output_dir}/{i:04d}.png")
        
        # Decode the final latents
        final_images = self.latents_to_pil(latents)
        if output_dir:
            final_images[0].save(f"{output_dir}/final.png")
        
        return final_images[0]

    def compare_noop_vs_edge(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        sharpening_scale=50,
        seed=42,
        output_dir="noop_vs_edge_comparison",
        save_comparison=True
    ):
        """
        Generate and compare images with no-op loss and edge sharpening loss.
        This allows seeing the exact difference the edge loss makes compared to the standard
        generation process with identical computation path.
        """
        import os
        # Create main output directory
        if save_comparison and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Create subdirectories for each generation type
        noop_dir = f"{output_dir}/noop" if save_comparison else None
        edge_dir = f"{output_dir}/edge" if save_comparison else None
        
        if noop_dir:
            os.makedirs(noop_dir, exist_ok=True)
        if edge_dir:
            os.makedirs(edge_dir, exist_ok=True)
            
        print(f"Generating image with NO-OP loss...")
        # Generate with no-op loss (follows same computation path but without actual effects)
        noop_image = self.generate_with_noop(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            output_dir=noop_dir,
            save_steps=False
        )
        
        print(f"Generating image with EDGE SHARPENING loss...")
        # Generate with edge sharpening loss
        edge_image = self.generate_with_edge_sharpening(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            sharpening_scale=sharpening_scale,
            seed=seed,
            output_dir=edge_dir,
            save_steps=False
        )
        
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1.imshow(np.array(noop_image))
        ax1.set_title("With No-Op Loss", fontsize=14)
        ax1.axis('off')
        
        ax2.imshow(np.array(edge_image))
        ax2.set_title(f"With Edge Sharpening Loss", fontsize=14)
        ax2.axis('off')
        
        plt.suptitle(f'"{prompt}"', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Save comparison image
        if save_comparison:
            comparison = Image.new('RGB', (noop_image.width + edge_image.width, noop_image.height))
            comparison.paste(noop_image, (0, 0))
            comparison.paste(edge_image, (noop_image.width, 0))
            comparison_filename = f"{output_dir}/comparison.png"
            comparison.save(comparison_filename)
            print(f"Comparison saved to {comparison_filename}")
        
        return noop_image, edge_image

"""## Run the demos

Let's demonstrate each of the loss functions with examples:
"""

# @title Edge Sharpening Demo
prompt = "Portrait of a woman wearing a red hat" # @param {type:"string"}
steps = 50 # @param {type:"slider", min:20, max:100, step:5}
sharpening_scale = 50 # @param {type:"slider", min:10, max:100, step:5}
show_intermediates = True # @param {type:"boolean"}

diffusion = GuidedDiffusion()  
output_dir = "edge_sharpening_output"

# Generate the image
image = diffusion.generate_with_edge_sharpening(
    prompt=prompt,
    num_inference_steps=steps,
    sharpening_scale=sharpening_scale,
    output_dir=output_dir,
    save_steps=show_intermediates
)

# Display the final image
plt.figure(figsize=(10, 10))
plt.imshow(np.array(image))
plt.title(f"Edge Sharpened: '{prompt}'")
plt.axis('off')
plt.show()

# @title Subject Changing Demo
original_prompt = "A golden retriever sitting in a garden" # @param {type:"string"}
target_subject = "A wolf in a snowy forest" # @param {type:"string"}
steps = 50 # @param {type:"slider", min:20, max:100, step:5}
subject_change_scale = 500 # @param {type:"slider", min:100, max:1000, step:50}
show_intermediates = True # @param {type:"boolean"}

diffusion = GuidedDiffusion()  
output_dir = "subject_change_output"

# Generate the image
image = diffusion.generate_with_subject_changing(
    prompt=original_prompt,
    target_subject=target_subject,
    num_inference_steps=steps,
    subject_change_scale=subject_change_scale,
    output_dir=output_dir,
    save_steps=show_intermediates
)

# Display the final image
plt.figure(figsize=(10, 10))
plt.imshow(np.array(image))
plt.title(f"Subject Changed: '{original_prompt}' → '{target_subject}'")
plt.axis('off')
plt.show()

# @title Blue Pixel Emphasis Demo
prompt = "A sunset over mountains" # @param {type:"string"}
steps = 50 # @param {type:"slider", min:20, max:100, step:5}
blue_scale = 200 # @param {type:"slider", min:50, max:400, step:25}
show_intermediates = True # @param {type:"boolean"}

diffusion = GuidedDiffusion()  
output_dir = "blue_output"

# Generate the image
image = diffusion.generate_with_blue(
    prompt=prompt,
    num_inference_steps=steps,
    blue_scale=blue_scale,
    output_dir=output_dir,
    save_steps=show_intermediates
)

# Display the final image
plt.figure(figsize=(10, 10))
plt.imshow(np.array(image))
plt.title(f"Blue Emphasis: '{prompt}'")
plt.axis('off')
plt.show()

"""## Visualize Intermediate Steps

You can visualize how the image evolves during the generation process:
"""

# @title Visualize Generation Process Animation
# @markdown Select which generation output to visualize:
output_dir = "edge_sharpening_output" # @param ["edge_sharpening_output", "subject_change_output", "blue_output"]
fps = 5 # @param {type:"slider", min:1, max:12, step:1}

from IPython.display import HTML
from matplotlib import animation
import matplotlib.pyplot as plt
import glob

# Get all images in sorted order
image_files = sorted(glob.glob(f"{output_dir}/*.png"))
if not image_files:
    print(f"No images found in {output_dir}! Please run one of the generation demos first.")
else:
    # Remove final.png if it exists
    if f"{output_dir}/final.png" in image_files:
        image_files.remove(f"{output_dir}/final.png")
    
    # Create animation
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    
    images = [plt.imshow(np.array(Image.open(img)), animated=True) for img in image_files]
    animation_title = plt.title(f"Generation Process: {output_dir}")
    
    ani = animation.ArtistAnimation(fig, [[img, animation_title] for img in images], 
                                   interval=1000/fps, blit=True)
    
    plt.close()  # Prevents duplicate display
    HTML(ani.to_jshtml())

"""## Create Your Own Custom Loss Function

You can extend the `GuidedDiffusion` class to create your own custom loss functions. Here's an example template:
"""

# @title Create a Custom Loss Function
# Example of a custom loss function

class CustomGuidedDiffusion(GuidedDiffusion):
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        super().__init__(model_id, device)
    
    def custom_loss(self, images, **kwargs):
        """
        Implement your custom loss function here.
        
        Args:
            images: The batch of images to apply the loss to
            **kwargs: Additional parameters for your loss
            
        Returns:
            loss: The computed loss value (lower is better)
        """
        # Example: A loss that encourages high contrast
        mean_brightness = images.mean(dim=[2, 3])  # Mean brightness per channel
        contrast = torch.var(images, dim=[2, 3])   # Variance per channel (proxy for contrast)
        
        # We want high contrast, so negative variance
        return -torch.mean(contrast)
    
    def generate_with_custom_loss(
        self,
        prompt,
        custom_param=1.0,  # Your custom parameter
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        custom_scale=100,  # Weight for your custom loss
        apply_every=1,
        seed=None,
        output_dir="custom_output",
        save_steps=True
    ):
        """Generate an image with your custom loss function"""
        import os
        if save_steps and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set seed for reproducibility
        if seed is not None:
            set_seed(seed)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        else:
            generator = None
        
        # Prep text embeddings (same as other methods)
        text_input = self.tokenizer(
            [prompt], 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        # Uncond embeddings for classifier-free guidance
        uncond_input = self.tokenizer(
            [""], 
            padding="max_length", 
            max_length=text_input.input_ids.shape[-1], 
            return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prep scheduler
        self.set_timesteps(num_inference_steps)
        
        # Prep latents
        latents = torch.randn(
            (1, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device
        )
        latents = latents * self.scheduler.init_noise_sigma
        
        # Loop through diffusion steps
        for i, t in tqdm(enumerate(self.timesteps), total=len(self.timesteps)):
            # expand the latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            sigma = self.scheduler.sigmas[i] if hasattr(self.scheduler, "sigmas") else None
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Apply custom guidance
            if i % apply_every == 0:
                # Requires grad on the latents
                latents = latents.detach().requires_grad_()
                
                # Get the predicted x0
                if sigma is not None:
                    latents_x0 = latents - sigma * noise_pred
                else:
                    # Handle different scheduler output formats
                    step_output = self.scheduler.step(noise_pred, t, latents)
                    if hasattr(step_output, "pred_original_sample"):
                        latents_x0 = step_output.pred_original_sample
                    elif isinstance(step_output, dict) and "pred_original_sample" in step_output:
                        latents_x0 = step_output["pred_original_sample"]
                    else:
                        # Fallback: manually compute the predicted original sample
                        if hasattr(self.scheduler, "alphas_cumprod"):
                            alpha_prod_t = self.scheduler.alphas_cumprod[i] if i < len(self.scheduler.alphas_cumprod) else self.scheduler.alphas_cumprod[-1]
                            latents_x0 = latents - noise_pred * (1.0 / alpha_prod_t**0.5)
                        else:
                            # Very basic approximation if we can't find alphas_cumprod
                            latents_x0 = latents - noise_pred * (1.0 / 0.18215)
                
                # Decode to image space
                denoised_images = self.vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5  # range (0, 1)
                
                # Calculate custom loss with your parameters
                loss = self.custom_loss(denoised_images, custom_param=custom_param) * custom_scale
                
                # Occasionally print loss
                if i % 10 == 0:
                    print(f"Step {i}, Custom Loss: {loss.item()}")
                
                # Get gradient
                grad = torch.autograd.grad(loss, latents)[0]
                
                # Modify the latents based on the gradient
                if sigma is not None:
                    latents = latents.detach() - grad * sigma**2
                else:
                    latents = latents.detach() - grad * 0.1  # Smaller step size if sigma not available
            
            # Step with scheduler
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Save intermediate result
            if save_steps and output_dir and i % 5 == 0:
                images = self.latents_to_pil(latents)
                images[0].save(f"{output_dir}/{i:04d}.png")
        
        # Decode the final latents
        final_images = self.latents_to_pil(latents)
        if output_dir:
            final_images[0].save(f"{output_dir}/final.png")
        
        return final_images[0]

# @title Run Custom Loss (Example)
# use_custom_loss = False  # @param {type:"boolean"}

# if use_custom_loss:
#     # Initialize custom diffusion
#     custom_diffusion = CustomGuidedDiffusion()
    
#     # Generate with custom loss
#     custom_image = custom_diffusion.generate_with_custom_loss(
#         prompt="A landscape with mountains and a lake",
#         custom_param=1.5,  # Your custom parameter
#         custom_scale=150,  # Weight for the loss
#         output_dir="custom_output"
#     )
    
#     # Display
#     plt.figure(figsize=(10, 10))
#     plt.imshow(np.array(custom_image))
#     plt.title("Generated with Custom Loss")
#     plt.axis('off')
#     plt.show()
# else:
#     print("Enable the checkbox to run the custom loss example")

"""## No-Op vs Edge Sharpening Comparison

Run this cell to see the difference between standard generation and edge sharpening loss.
"""

# @title Run No-Op vs Edge Sharpening Comparison
prompt = "A detailed portrait of a siberian husky" # @param {type:"string"}
sharpening_scale = 75 # @param {type:"slider", min:20, max:200, step:5}
steps = 50 # @param {type:"slider", min:20, max:100, step:5}
seed = 42 # @param {type:"integer"}

# Initialize the model
diffusion = GuidedDiffusion()

# Generate and compare images
noop_image, edge_image = diffusion.compare_noop_vs_edge(
    prompt=prompt,
    sharpening_scale=sharpening_scale,
    num_inference_steps=steps,
    seed=seed,
    output_dir="noop_vs_edge_comparison"
)

"""## Conclusion

This notebook demonstrated how to use custom loss functions with Stable Diffusion to guide image generation in specific ways. You can:

1. Enhance edge details using the edge sharpening loss
2. Transform subjects using the CLIP-based subject changing loss
3. Add color emphasis with the blue pixel loss
4. Create your own custom loss functions
5. Compare "no-op" path with losses to see exact differences

Experiment with different prompts, loss scales, and parameters to achieve your desired results!
""" 