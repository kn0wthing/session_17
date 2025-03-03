import subprocess
import os
import time

# Ensure style_embeddings directory exists
os.makedirs("style_embeddings", exist_ok=True)

# Define styles and their corresponding parameters
styles = [
    {"name": "watercolor", "steps": 500, "lr": 1e-4, "seed": 42},
    {"name": "cyberpunk", "steps": 500, "lr": 1e-4, "seed": 123},
    {"name": "vangogh", "steps": 500, "lr": 1e-4, "seed": 456},
    {"name": "ukiyoe", "steps": 500, "lr": 1e-4, "seed": 789},
    {"name": "retro", "steps": 500, "lr": 1e-4, "seed": 1024}
]

# Train each style
for style in styles:
    print(f"\n{'='*80}")
    print(f"Training {style['name']} style")
    print(f"{'='*80}")
    
    # Build command
    cmd = [
        "python", 
        "src/train_style.py", 
        style["name"], 
        "--steps", str(style["steps"]), 
        "--lr", str(style["lr"]), 
        "--seed", str(style["seed"])
    ]
    
    # Execute command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error training {style['name']}: {e}")
    
    # Allow some time between training sessions
    print(f"Completed training {style['name']}")
    if style != styles[-1]:  # If not the last style
        print("Waiting 5 seconds before starting next style...")
        time.sleep(5)

print("\nAll styles have been trained successfully!")
print("You can now generate artwork using:")
print("python src/generate_art.py \"your prompt here\" --style watercolor") 