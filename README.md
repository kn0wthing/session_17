# AI Style Transfer Studio

A powerful Stable Diffusion based application that allows you to transform your text prompts into artistic masterpieces using various trained style models.

## Features

- **Multiple Artistic Styles**: Transform your prompts using various pre-trained styles:
  - Dhoni Style (Sports/Action)
  - Mickey Mouse Style (Cartoon/Whimsical)
  - Balloon Style (Festive/Celebratory)
  - Lion King Style (Majestic/Animal)
  - Rose Flower Style (Floral/Romantic)

- **Color Enhancement**: Advanced color processing technology that maximizes color channel separation for more vibrant and striking images

- **User-Friendly Interface**: Clean, modern web interface built with Streamlit

- **Style Training**: Custom script for training new style embeddings using your own image datasets

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have the following system requirements:
   - Python 3.7 or higher
   - CUDA-compatible GPU (recommended) or CPU
   - At least 8GB RAM (16GB recommended)

## Usage

### Running the Application

1. Start the web interface:
   ```bash
   python src/app.py
   ```

2. Open your web browser and navigate to the displayed URL

3. Enter your text prompt and select a style

4. Click "Generate Artwork" to create your masterpiece

### Training New Styles

1. Prepare your training images:
   - Place your style reference images in `training_images/<style_name>/`
   - Supported formats: JPG, PNG
   - Recommended: 3-5 high-quality images per style

2. Run the training script:
   ```bash
   python src/train_style.py <style_name> --steps 3000 --lr 1e-4
   ```

## Project Structure

```
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── train_style.py         # Style training script
│   └── utils/
│       ├── style_generator.py # Core style transfer logic
│       └── ui_components.py   # UI component definitions
├── training_images/           # Training image datasets
│   ├── dhoni/                 # Dhoni style reference images
│   ├── mickey_mouse/          # Mickey Mouse style reference images
│   ├── balloon/               # Balloon style reference images
│   ├── lion_king/             # Lion King style reference images
│   └── rose_flower/           # Rose Flower style reference images
├── docs/                      # Documentation
└── requirements.txt           # Project dependencies
```

## Technical Details

- Built on Stable Diffusion v1.5
- Uses textual inversion for style embedding
- Implements custom color enhancement using distance loss
- Supports both CPU and CUDA acceleration

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Google Colab Logs

```bash
!python src/train_style.py dhoni --steps 1000 --lr 1e-4
```

The above command generates the embedding file "dhoni.bin" in "style_embeddings" folder. 
All embedding files are generated based on the above command, by providing the desired style name.
Generated bin files are used in the HuggingFace App and not uploaded with this repo do avoid size constraints of 100 MB on GitHub.
