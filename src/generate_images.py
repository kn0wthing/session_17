import torch
import argparse
from guided_diffusion import GuidedDiffusion

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with custom loss functions")
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Portrait of a woman wearing a red hat",
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["edge", "subject", "blue"], 
        default="edge",
        help="Loss function to use: edge sharpening, subject changing, or blue pixels"
    )
    parser.add_argument(
        "--target_subject", 
        type=str, 
        default="Portrait of a woman wearing a blue hat with flowers",
        help="Target subject for subject changing mode"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=50,
        help="Number of diffusion steps"
    )
    parser.add_argument(
        "--scale", 
        type=float, 
        default=None,
        help="Scale for the loss function (defaults to appropriate value for each mode)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save intermediate results (defaults to mode name)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model ID to use"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set default output directory based on mode if not specified
    if args.output_dir is None:
        args.output_dir = f"{args.mode}_output"
    
    # Set default scale based on mode if not specified
    if args.scale is None:
        if args.mode == "edge":
            args.scale = 50
        elif args.mode == "subject":
            args.scale = 500
        else:  # blue
            args.scale = 200
    
    print(f"Using {args.mode} mode with scale {args.scale}")
    print(f"Prompt: {args.prompt}")
    if args.mode == "subject":
        print(f"Target subject: {args.target_subject}")
    
    # Initialize the model
    diffusion = GuidedDiffusion(model_id=args.model_id)
    
    # Generate image based on selected mode
    if args.mode == "edge":
        image = diffusion.generate_with_edge_sharpening(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            sharpening_scale=args.scale,
            seed=args.seed,
            output_dir=args.output_dir
        )
        print(f"Generated image with edge sharpening. Saved to {args.output_dir}/final.png")
    
    elif args.mode == "subject":
        image = diffusion.generate_with_subject_changing(
            prompt=args.prompt,
            target_subject=args.target_subject,
            num_inference_steps=args.steps,
            subject_change_scale=args.scale,
            seed=args.seed,
            output_dir=args.output_dir
        )
        print(f"Generated image with subject changing. Saved to {args.output_dir}/final.png")
    
    else:  # blue mode - let's create a custom function
        # Custom blue generation function using the existing blue_loss from our class
        def generate_with_blue(diffusion, prompt, blue_scale=200, steps=50, seed=42, output_dir="blue_output"):
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Set up generation parameters
            height = 512
            width = 512
            guidance_scale = 7.5
            generator = torch.manual_seed(seed)
            
            # Prep text embeddings
            text_input = diffusion.tokenizer(
                [prompt], 
                padding="max_length", 
                max_length=diffusion.tokenizer.model_max_length, 
                truncation=True, 
                return_tensors="pt"
            )
            with torch.no_grad():
                text_embeddings = diffusion.text_encoder(text_input.input_ids.to(diffusion.device))[0]
            
            # Uncond embeddings for classifier-free guidance
            uncond_input = diffusion.tokenizer(
                [""], 
                padding="max_length", 
                max_length=text_input.input_ids.shape[-1], 
                return_tensors="pt"
            )
            with torch.no_grad():
                uncond_embeddings = diffusion.text_encoder(uncond_input.input_ids.to(diffusion.device))[0]
            
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
            # Prep scheduler
            diffusion.set_timesteps(steps)
            
            # Prep latents
            latents = torch.randn(
                (1, diffusion.unet.in_channels, height // 8, width // 8),
                generator=generator,
                device=diffusion.device
            )
            latents = latents * diffusion.scheduler.init_noise_sigma
            
            # Loop
            for i, t in enumerate(diffusion.scheduler.timesteps):
                # expand the latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2)
                sigma = diffusion.scheduler.sigmas[i] if hasattr(diffusion.scheduler, "sigmas") else None
                latent_model_input = diffusion.scheduler.scale_model_input(latent_model_input, t)
                
                # predict the noise residual
                with torch.no_grad():
                    noise_pred = diffusion.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
                
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Apply blue guidance every 5 steps
                if i % 5 == 0:
                    # Requires grad on the latents
                    latents = latents.detach().requires_grad_()
                    
                    # Get the predicted x0
                    if sigma is not None:
                        latents_x0 = latents - sigma * noise_pred
                    else:
                        latents_x0 = diffusion.scheduler.step(noise_pred, t, latents).pred_original_sample
                    
                    # Decode to image space
                    denoised_images = diffusion.vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5  # range (0, 1)
                    
                    # Calculate blue loss
                    loss = diffusion.blue_loss(denoised_images) * blue_scale
                    
                    # Occasionally print loss
                    if i % 10 == 0:
                        print(f"Step {i}, Blue Loss: {loss.item()}")
                    
                    # Get gradient
                    grad = torch.autograd.grad(loss, latents)[0]
                    
                    # Modify the latents based on the gradient
                    if sigma is not None:
                        latents = latents.detach() - grad * sigma**2
                    else:
                        latents = latents.detach() - grad * 0.1
                
                # Step with scheduler
                latents = diffusion.scheduler.step(noise_pred, t, latents).prev_sample
                
                # Save intermediate result
                if i % 5 == 0:
                    images = diffusion.latents_to_pil(latents)
                    images[0].save(f"{output_dir}/{i:04d}.png")
            
            # Decode the final latents
            final_images = diffusion.latents_to_pil(latents)
            final_images[0].save(f"{output_dir}/final.png")
            
            return final_images[0]
        
        # Generate with blue loss
        image = generate_with_blue(
            diffusion=diffusion,
            prompt=args.prompt,
            blue_scale=args.scale,
            steps=args.steps,
            seed=args.seed,
            output_dir=args.output_dir
        )
        print(f"Generated image with blue pixel emphasis. Saved to {args.output_dir}/final.png")

if __name__ == "__main__":
    main() 