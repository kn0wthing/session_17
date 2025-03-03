"""
Main module for the AI Style Transfer Studio.
Provides a command-line interface to run the application or train new styles.
"""
import argparse
import sys
from pathlib import Path


def run_app():
    """Run the Streamlit web application."""
    try:
        import streamlit.web.cli as stcli
        from utils.style_generator import StyleTransfer
        
        # Initialize the style generator
        StyleTransfer.get_instance().initialize_pipeline()
        
        # Get the path to app.py
        app_path = Path(__file__).parent / "app.py"
        
        # Run the Streamlit app
        sys.argv = ["streamlit", "run", str(app_path), "--server.headless", "true"]
        sys.exit(stcli.main())
    except Exception as e:
        print(f"Error running the application: {e}")
        sys.exit(1)


def train_style(style_name, steps, learning_rate):
    """Train a new style embedding.
    
    Args:
        style_name (str): Name of the style to train
        steps (int): Number of training steps
        learning_rate (float): Learning rate for training
    """
    try:
        from train_style import train_style as train_style_func
        import glob
        
        # Get the training images
        training_dir = Path(__file__).parent.parent / "training_images" / style_name
        if not training_dir.exists():
            print(f"Error: Training directory not found: {training_dir}")
            sys.exit(1)
            
        image_paths = []
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            image_paths.extend(glob.glob(str(training_dir / ext)))
        
        if not image_paths:
            print(f"Error: No images found in {training_dir}")
            sys.exit(1)
            
        print(f"Training {style_name} style with {len(image_paths)} images...")
        train_style_func(
            style_name=style_name,
            image_paths=image_paths,
            placeholder_token=f"<{style_name}-style>",
            num_training_steps=steps,
            learning_rate=learning_rate
        )
    except Exception as e:
        print(f"Error training style: {e}")
        sys.exit(1)


def main():
    """Main function to parse command-line arguments and run the appropriate function."""
    parser = argparse.ArgumentParser(description="AI Style Transfer Studio")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run app command
    run_parser = subparsers.add_parser("run", help="Run the web application")
    
    # Train style command
    train_parser = subparsers.add_parser("train", help="Train a new style")
    train_parser.add_argument("style_name", help="Name of the style to train")
    train_parser.add_argument("--steps", type=int, default=3000, help="Number of training steps")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_app()
    elif args.command == "train":
        train_style(args.style_name, args.steps, args.lr)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 