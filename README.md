# doro

A specialized GIF processing pipeline that automatically optimizes GIFs for looping and renames them using AI-powered image captioning.

## Features

- **Automatic Looping**: Uses ImageMagick to ensure GIFs are set to loop infinitely (`-loop 0`).
- **AI Auto-Renaming**: Uses the BLIP (Bootstrapping Language-Image Pre-training) model to analyze GIF content and automatically rename files with descriptive Chinese captions.
- **Automated Workflow**: Fully integrated with GitHub Actions for "hands-off" processing.
- **Gallery View**: A lightweight HTML frontend to showcase the processed GIF library.

## Workflow

The repository utilizes two main GitHub Action pipelines:

1. **Convert GIFs to Looping**:
   - Triggered when `.gif` files are pushed to the `动态/` directory.
   - Processes the images and saves the looping versions into the `loop/` directory.

2. **Rename GIFs with BLIP**:
   - Triggered manually via `workflow_dispatch`.
   - Uses `scripts/rename_blip.py` to:
     - Extract a frame from the GIF.
     - Generate an English caption using Salesforce's BLIP model.
     - Translate the caption to Chinese using the Helsinki-NLP MarianMT model.
     - Rename the file in the `loop/` directory based on the generated description.

## Technical Stack

- **Image Processing**: ImageMagick, OpenCV, Pillow.
- **AI/ML**: PyTorch, Hugging Face Transformers (BLIP, MarianMT).
- **CI/CD**: GitHub Actions.
- **Frontend**: HTML/JavaScript.

## Setup & Usage

### Adding New GIFs
1. Upload your GIF files to the `动态/` folder in the repository.
2. The **Convert GIFs to Looping** action will automatically trigger and place optimized versions in the `loop/` folder.

### Renaming GIFs
1. Go to the **Actions** tab in the GitHub repository.
2. Select the **"Rename GIFs with BLIP"** workflow.
3. Click **Run workflow**.

### Viewing the Gallery
Open `doro.html` in any browser to view the current collection of processed GIFs. Note that the `gifs` array in the HTML script may need to be updated to reflect the new AI-generated filenames.
