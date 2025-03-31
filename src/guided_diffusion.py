import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from diffusers.utils import set_seed
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import torchvision.transforms as T
from transformers import CLIPImageProcessor, CLIPModel

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
        
    def set_timesteps(self, num_inference_steps):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
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
    
    def subject_changing_loss(self, images, target_prompt, current_prompt=None):
        """
        A loss function that encourages the image to match a different subject/prompt.
        Uses CLIP to measure text-image similarity and pushes towards the target prompt.
        """
        # Process the generated image with CLIP
        with torch.no_grad():
            # Convert to PIL and process with CLIP
            pil_images = self.latents_to_pil(images)
            clip_inputs = self.clip_processor(images=pil_images, return_tensors="pt").to(self.device)
            image_features = self.clip_model.get_image_features(**clip_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Process the target prompt with CLIP
            target_text_inputs = self.clip_processor.tokenizer(
                target_prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            target_text_features = self.clip_model.get_text_features(**target_text_inputs)
            target_text_features = target_text_features / target_text_features.norm(dim=-1, keepdim=True)
            
            # If we have a current prompt, we want to push away from it
            if current_prompt:
                current_text_inputs = self.clip_processor.tokenizer(
                    current_prompt,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                current_text_features = self.clip_model.get_text_features(**current_text_inputs)
                current_text_features = current_text_features / current_text_features.norm(dim=-1, keepdim=True)
                
                # Push away from current and towards target
                target_similarity = (image_features * target_text_features).sum(dim=-1)
                current_similarity = (image_features * current_text_features).sum(dim=-1)
                
                # We want to maximize target similarity and minimize current similarity
                return -target_similarity.mean() + current_similarity.mean()
            
            # Just optimize for target similarity
            target_similarity = (image_features * target_text_features).sum(dim=-1)
            return -target_similarity.mean()
    
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
            generator = torch.manual_seed(seed)
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
        for i, t in tqdm(enumerate(self.scheduler.timesteps), total=len(self.scheduler.timesteps)):
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
                    latents_x0 = self.scheduler.step(noise_pred, t, latents).pred_original_sample
                
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
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        subject_change_scale=500,
        apply_every=1,
        seed=None,
        output_dir="subject_change_steps",
        save_steps=True
    ):
        """Generate an image with subject changing guidance"""
        import os
        if save_steps and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set seed for reproducibility
        if seed is not None:
            set_seed(seed)
            generator = torch.manual_seed(seed)
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
        for i, t in tqdm(enumerate(self.scheduler.timesteps), total=len(self.scheduler.timesteps)):
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
            
            # Apply subject changing guidance - but only after some initial denoising
            # This helps establish the base image first
            if i % apply_every == 0 and i > num_inference_steps // 4:
                # Requires grad on the latents
                latents = latents.detach().requires_grad_()
                
                # Get the predicted x0
                if sigma is not None:
                    latents_x0 = latents - sigma * noise_pred
                else:
                    latents_x0 = self.scheduler.step(noise_pred, t, latents).pred_original_sample
                
                # Decode to image space
                denoised_images = self.vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5  # range (0, 1)
                
                # Calculate subject change loss
                loss = self.subject_changing_loss(
                    denoised_images, 
                    target_prompt=target_subject, 
                    current_prompt=prompt
                ) * subject_change_scale
                
                # Occasionally print loss
                if i % 10 == 0:
                    print(f"Step {i}, Subject Change Loss: {loss.item()}")
                
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