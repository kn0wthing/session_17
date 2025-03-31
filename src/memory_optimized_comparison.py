import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from transformers import CLIPProcessor, CLIPModel

class MemoryOptimizedComparison:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """Initialize the comparison class with memory optimizations"""
        # Use lower precision to reduce memory usage
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"Using {self.device} with {self.dtype}")
        
        # Load pipeline with memory optimizations
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,  # Disable safety checker to save memory
        ).to(self.device)
        
        # Use memory efficient attention if available
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        
        # Use DDIM scheduler which is more memory efficient
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Extract components
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        
        # Initialize CLIP for subject changing
        print("Loading CLIP model...")
        self.clip_model = None
        self.clip_processor = None
        
    def _load_clip_if_needed(self):
        """Lazy-load CLIP model to save memory when not using subject changing"""
        if self.clip_model is None:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    def clear_memory(self):
        """Clear GPU memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    
    def generate_without_loss(self, prompt, seed=42, steps=50, height=512, width=512):
        """Generate an image without custom loss function"""
        self.clear_memory()
        
        # Set seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate image
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=steps,
                generator=generator,
                height=height,
                width=width
            ).images[0]
        
        self.clear_memory()
        return image
    
    def encode_prompt(self, prompt):
        """Encode the prompt to text embeddings"""
        # Encode text
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
        
        # Concatenate
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings
    
    def blue_loss(self, images):
        """Loss function to emphasize blue pixels"""
        error = torch.abs(images[:, 2] - 0.9).mean()
        return error
    
    def edge_sharpening_loss(self, images):
        """Loss function to enhance edges"""
        import torch.nn.functional as F
        
        # Convert to grayscale
        gray = 0.2989 * images[:, 0] + 0.5870 * images[:, 1] + 0.1140 * images[:, 2]
        gray = gray.unsqueeze(1)
        
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device).to(self.dtype).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device).to(self.dtype).view(1, 1, 3, 3)
        
        # Detect edges
        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)
        
        # Calculate magnitude
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
        
        # We want to maximize edge magnitude
        return -torch.mean(edge_magnitude) + 1e-5
    
    def subject_changing_loss(self, images, target_prompt, current_prompt=None):
        """Loss function to change the subject of an image using CLIP"""
        self._load_clip_if_needed()
        
        # Process the image for CLIP
        with torch.no_grad():
            # Convert to numpy array
            images_np = images.detach().cpu().numpy()
            # Scale to 0-255 range
            images_np = (images_np * 255).astype(np.uint8)
            # Create PIL images - CLIP expects HWC format
            pil_images = [Image.fromarray(img.transpose(1, 2, 0)) for img in images_np]
            
            # Process images with CLIP
            clip_inputs = self.clip_processor(images=pil_images, return_tensors="pt").to(self.device)
            image_features = self.clip_model.get_image_features(**clip_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Process target prompt with CLIP
            target_text_inputs = self.clip_processor.tokenizer(
                target_prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            target_text_features = self.clip_model.get_text_features(**target_text_inputs)
            target_text_features = target_text_features / target_text_features.norm(dim=-1, keepdim=True)
            
            # Calculate target similarity
            target_similarity = (image_features * target_text_features).sum(dim=-1)
            
            # If we have a current prompt, also calculate its similarity
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
                
                # Calculate current similarity
                current_similarity = (image_features * current_text_features).sum(dim=-1)
                
                # We want to maximize target similarity and minimize current similarity
                return -target_similarity.mean() + current_similarity.mean()
            
            # If no current prompt, just maximize target similarity
            return -target_similarity.mean()
    
    def generate_with_loss(self, prompt, loss_type="blue", loss_scale=100, target_subject=None, seed=42, steps=50, height=512, width=512):
        """Generate an image with the specified loss function"""
        self.clear_memory()
        
        # Set seed
        torch.manual_seed(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Get text embeddings
        text_embeddings = self.encode_prompt(prompt)
        
        # Set up timesteps
        self.scheduler.set_timesteps(steps)
        
        # Create random latents
        latents = torch.randn(
            (1, self.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        latents = latents * self.scheduler.init_noise_sigma
        
        # Choose loss function
        if loss_type == "blue":
            loss_fn = self.blue_loss
        elif loss_type == "edge":
            loss_fn = self.edge_sharpening_loss
        elif loss_type == "subject":
            if target_subject is None:
                raise ValueError("target_subject must be provided for subject changing loss")
            # Load CLIP model if needed
            self._load_clip_if_needed()
            # Create a closure that includes the target subject
            loss_fn = lambda img: self.subject_changing_loss(img, target_subject, prompt)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # Loop through diffusion steps
        for i, t in enumerate(self.scheduler.timesteps):
            # Free memory from previous step
            if i > 0:
                self.clear_memory()
            
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # Apply classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)  # guidance_scale=7.5
            
            # Apply loss function (with reduced frequency for subject changing to save memory)
            should_apply_loss = i % 5 == 0  # Apply every 5 steps
            if loss_type == "subject":
                # For subject changing, only start after some initial denoising
                should_apply_loss = should_apply_loss and i > steps // 4
            
            if should_apply_loss:
                # Enable gradients for this step
                latents = latents.detach().requires_grad_()
                
                # Get predicted x0
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
            
            # Step with scheduler
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode the final image
        with torch.no_grad():
            # Scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents).sample
        
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")[0]
        image = Image.fromarray(image)
        
        self.clear_memory()
        return image
    
    def compare_with_without_loss(self, prompt, loss_type="blue", loss_scale=100, target_subject=None, seed=42, steps=30, height=512, width=512):
        """Generate and compare images with and without the loss function"""
        # For memory reasons, we'll generate these one at a time
        print(f"Generating image WITHOUT {loss_type} loss...")
        img_without_loss = self.generate_without_loss(
            prompt=prompt,
            seed=seed,
            steps=steps,
            height=height,
            width=width
        )
        
        print(f"Generating image WITH {loss_type} loss...")
        img_with_loss = self.generate_with_loss(
            prompt=prompt,
            loss_type=loss_type,
            loss_scale=loss_scale,
            target_subject=target_subject,
            seed=seed,
            steps=steps,
            height=height,
            width=width
        )
        
        # Display comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        ax1.imshow(np.array(img_without_loss))
        ax1.set_title("Without Loss Function", fontsize=14)
        ax1.axis('off')
        
        ax2.imshow(np.array(img_with_loss))
        ax2.set_title(f"With {loss_type.capitalize()} Loss", fontsize=14)
        ax2.axis('off')
        
        title = f'"{prompt}"'
        if loss_type == "subject":
            title += f' → "{target_subject}"'
            
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Save side-by-side comparison
        comparison = Image.new('RGB', (img_without_loss.width + img_with_loss.width, img_without_loss.height))
        comparison.paste(img_without_loss, (0, 0))
        comparison.paste(img_with_loss, (img_without_loss.width, 0))
        comparison_filename = f"comparison_{loss_type}.png"
        comparison.save(comparison_filename)
        print(f"Comparison saved to {comparison_filename}")
        
        return img_without_loss, img_with_loss

# Example usage (call this in a Colab cell)
# comparator = MemoryOptimizedComparison()
# comparator.compare_with_without_loss(
#     prompt="A sunset over mountains",
#     loss_type="blue",
#     loss_scale=150,
#     steps=30  # Fewer steps to save memory
# )
#
# comparator.compare_with_without_loss(
#     prompt="Portrait of a woman wearing a red hat",
#     loss_type="edge",
#     loss_scale=50,
#     steps=30  # Fewer steps to save memory
# )
#
# comparator.compare_with_without_loss(
#     prompt="A golden retriever sitting in a garden",
#     loss_type="subject",
#     target_subject="A wolf in a snowy forest",
#     loss_scale=300,
#     steps=30  # Fewer steps to save memory
# ) 