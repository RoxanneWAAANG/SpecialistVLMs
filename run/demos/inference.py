import hydra
from PIL import Image
import numpy as np
import textwrap
from glob import glob
import os
import sys

sys.path.append(os.getcwd())
from models.retinavlm_wrapper import load_from_api

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def demo_inference(config):
    sample_image_dir = 'dataset/processed_images'
    png_files = glob(os.path.join(sample_image_dir, '*.png'))

    instruction = textwrap.dedent(f'''Write an extensive report describing the OCT image and listing any visible biomarkers or other observations. Do not provide disease stage or patient referral recommendations yet.''')
    print(f'Running textual instruction: \"{instruction}\"')

    retinavlm = load_from_api(config).eval()

    for image_path in png_files:
        # image = np.array(Image.open(image_path))
        # print(f'On image: {image_path}')
        image = Image.open(image_path)
        
        # Resize the image to 192x192 and ensure it's in RGB format
        image = image.resize((192, 192))
        image = image.convert('RGB')  # Ensure 3 channels (RGB)
        
        # Convert the image to a NumPy array
        image_np = np.array(image)
        
        # Ensure that the image has the correct shape (C, 192, 192)
        if image_np.shape[2] != 3:
            print(f"Warning: Image {image_path} does not have 3 channels, converting.")
            image_np = np.stack([image_np] * 3, axis=-1)  # Convert grayscale to RGB if necessary
        
        print(f'On image: {image_path}')
        
        # Run inference
        output = retinavlm.forward(image_np, [instruction])
        
        # Print the response
        print(f'VLM response: \"{output[0]}\"')

if __name__ == "__main__":
    demo_inference()

