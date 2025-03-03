import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
from pathlib import Path
import traceback

class StyleTransfer:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.pipeline = None
        self.style_tokens = []
        self.styles = [
            "dhoni",
            "mickey_mouse",
            "balloon",
            "lion_king",
            "rose_flower"
        ]
        self.style_names = [
            "Dhoni Style",
            "Mickey Mouse Style",
            "Balloon Style",
            "Lion King Style",
            "Rose Flower Style"
        ]
        self.is_initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("NVIDIA GPU not found. Running on CPU (this will be slower)")

    def initialize_pipeline(self):
        if self.is_initialized:
            return
            
        try:
            print("Initializing Stable Diffusion model...")
            model_id = "runwayml/stable-diffusion-v1-5"
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            )
            self.pipeline = self.pipeline.to(self.device)
            
            # Load style embeddings from current directory
            current_dir = Path(__file__).parent.parent
            
            for style, style_name in zip(self.styles, self.style_names):
                style_path = current_dir / f"{style}.bin"
                if not style_path.exists():
                    raise FileNotFoundError(f"Style embedding not found: {style_path}")
                
                print(f"Loading style: {style_name}")
                token = self._load_style_embedding(str(style_path))
                self.style_tokens.append(token)
                print(f"✓ Loaded style: {style_name}")
            
            self.is_initialized = True
            print(f"Model initialization complete! Using device: {self.device}")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            print(traceback.format_exc())
            raise

    def _load_style_embedding(self, embedding_path, token=None):
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
        embeds = embeds.unsqueeze(0)  # Add batch dimension
        
        # Cast to dtype of text_encoder
        dtype = self.pipeline.text_encoder.get_input_embeddings().weight.dtype
        embeds = embeds.to(dtype)

        # Add the token in tokenizer
        token = token if token is not None else trained_token
        self.pipeline.tokenizer.add_tokens(token)
        
        # Resize the token embeddings
        self.pipeline.text_encoder.resize_token_embeddings(len(self.pipeline.tokenizer))
        
        # Get the id for the token and assign the embeds
        token_id = self.pipeline.tokenizer.convert_tokens_to_ids(token)
        self.pipeline.text_encoder.get_input_embeddings().weight.data[token_id] = embeds[0]
        return token

    def generate_artwork(self, prompt, selected_style):
        try:
            # Find the index of the selected style
            style_idx = self.style_names.index(selected_style)
            
            # Generate single image with selected style
            styled_prompt = f"{prompt}, {self.style_tokens[style_idx]}"
            
            # Set seed for reproducibility
            generator_seed = 42
            torch.manual_seed(generator_seed)
            if self.device == "cuda":
                torch.cuda.manual_seed(generator_seed)
            
            # Generate base image
            with autocast(self.device):
                base_image = self.pipeline(
                    styled_prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    generator=torch.Generator(self.device).manual_seed(generator_seed)
                ).images[0]
            
            # Generate same image with color enhancement
            with autocast(self.device):
                enhanced_image = self.pipeline(
                    styled_prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    callback=self._enhance_colors,
                    callback_steps=5,
                    generator=torch.Generator(self.device).manual_seed(generator_seed)
                ).images[0]
            
            return base_image, enhanced_image
            
        except Exception as e:
            print(f"Error in generate_artwork: {e}")
            raise

    def _enhance_colors(self, i, t, latents):
        if i % 5 == 0:  # Apply enhancement every 5 steps
            try:
                # Create a copy that requires gradients
                latents_copy = latents.detach().clone()
                latents_copy.requires_grad_(True)
                
                # Compute color distance loss
                loss = self._calculate_color_distance(latents_copy)
                
                # Compute gradients
                if loss.requires_grad:
                    grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=latents_copy,
                        allow_unused=True,
                        retain_graph=False
                    )[0]
                    
                    if grads is not None:
                        # Apply gradients to original latents
                        return latents - 0.1 * grads.detach()
            
            except Exception as e:
                print(f"Error in color enhancement: {e}")
            
        return latents

    def _calculate_color_distance(self, images):
        # Ensure we're working with gradients
        if not images.requires_grad:
            images = images.detach().requires_grad_(True)
        
        # Convert to float32 and normalize
        images = images.float() / 2 + 0.5
        
        # Get RGB channels
        red = images[:,0:1]
        green = images[:,1:2]
        blue = images[:,2:3]
        
        # Calculate color distances using L2 norm
        rg_distance = ((red - green) ** 2).mean()
        rb_distance = ((red - blue) ** 2).mean()
        gb_distance = ((green - blue) ** 2).mean()
        
        return (rg_distance + rb_distance + gb_distance) * 100  # Scale up the loss