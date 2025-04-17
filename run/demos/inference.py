import hydra
from PIL import Image
import numpy as np
import textwrap
from glob import glob
import os
import sys

sys.path.append(os.getcwd())
from models.retinavlm_wrapper import load_from_api

def jpg2png(image_dir):
    """
    Convert all JPG/JPEG images in the directory to PNG format.
    Returns a list of paths to all PNG files (both converted and original).
    """
    # Find all jpg/jpeg files
    jpg_files = glob(os.path.join(image_dir, '*.jpg'))
    jpg_files.extend(glob(os.path.join(image_dir, '*.jpeg')))
    
    # Convert each jpg to png
    for jpg_path in jpg_files:
        png_path = os.path.splitext(jpg_path)[0] + '.png'
        print(f'Converting {jpg_path} to {png_path}')
        
        try:
            image = Image.open(jpg_path)
            # Preserve original format characteristics when converting
            image = image.convert('RGB')
            image.save(png_path, 'PNG')
            print(f'Successfully converted {jpg_path} to {png_path}')
        except Exception as e:
            print(f'Error converting {jpg_path}: {e}')
    
    # Return all png files in the directory (both converted and original)
    return glob(os.path.join(image_dir, '*.png'))

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def demo_inference(config):
    sample_image_dir = 'dataset/processed_images'
    png_files = jpg2png(sample_image_dir)
    # png_files = glob(os.path.join(sample_image_dir, '*.png'))

    instruction = textwrap.dedent(f'''Write an extensive report describing the OCT image and listing any visible biomarkers or other observations. Do not provide disease stage or patient referral recommendations yet.''')
    print(f'Running textual instruction: \"{instruction}\"')

    retinavlm = load_from_api(config).eval()

    for image_path in png_files:
        image = Image.open(image_path)
        print(f'Original image size: {image.size}')
        image = image.resize((455, 455))
        image = image.convert('RGB')
        image = np.array(image)
        print(f'On image: {image_path}')
        
        # Run inference
        output = retinavlm.forward(image, [instruction])
        
        # Print the response
        print(f'VLM response: \"{output[0]}\"')

if __name__ == "__main__":
    demo_inference()

