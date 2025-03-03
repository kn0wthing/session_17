# AI Style Transfer Studio Documentation

This directory contains documentation for the AI Style Transfer Studio project.

## Structure

- `usage.md`: Usage examples and tutorials for the application
- `api.md`: API documentation for the style transfer functionality
- `training.md`: Guide for training new style embeddings
- `development.md`: Development guidelines for contributors

## Quick Start

### Running the Application

```bash
# Activate the virtual environment
source venv/bin/activate

# Run the application
python -m src.main run
```

### Training a New Style

```bash
# Activate the virtual environment
source venv/bin/activate

# Train a new style
python -m src.main train new_style_name --steps 3000 --lr 1e-4
```

## Style Reference

| Style Name | Description | Best For |
|------------|-------------|----------|
| Dhoni Style | Sports action style based on cricket player MS Dhoni | Action scenes, sports photography |
| Mickey Mouse Style | Cartoon style based on Disney's Mickey Mouse | Whimsical scenes, cartoon characters |
| Balloon Style | Colorful festive style based on balloons | Celebrations, festive scenes |
| Lion King Style | Majestic animal style based on Lion King | Animal portraits, nature scenes |
| Rose Flower Style | Floral style based on roses | Romantic scenes, floral compositions |

## Building Documentation

Documentation can be built using tools like Sphinx or MkDocs. 