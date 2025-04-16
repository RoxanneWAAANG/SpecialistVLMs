from run.vision_language_pretraining import MiniGPT4Module
import hydra
import torch
import sys
from PIL import Image
import numpy as np
from huggingface_hub import login, HfApi
import scipy
import textwrap
from transformers import PreTrainedModel, PretrainedConfig

class RetinaVLMConfig(PretrainedConfig):
    model_type = "RetinaVLM"
    def __init__(self, torch_dtype="float32", **kwargs):
        super().__init__()
        self.torch_dtype = torch_dtype
        self.__dict__.update(kwargs)

class RetinaVLM(PreTrainedModel):
    config_class = RetinaVLMConfig

    def __init__(self, config, device=None):
        hf_config = RetinaVLMConfig()
        print(hf_config)
        super().__init__(hf_config)

        # Handle device properly - don't set as attribute
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store as a private attribute to avoid conflict with parent class
        self._device_type = device
        
        # Initialize model
        self.model = self._initialize_model(config, device)
    
    def _initialize_model(self, config, device):
        try:
            print(f"Creating model on {device}...")
            # Create model
            model = MiniGPT4Module(config, device=device).model
            
            # After initialization, move to desired device
            if device != "cpu":
                model = model.to(device)
            
            return model.eval()
        except Exception as e:
            print(f"Error during model initialization: {e}")
            # Fallback to basic initialization
            return MiniGPT4Module(config, device=device).model.eval()

    def convert_any_image_to_normalized_tensor(self, image_input):
        # Convert input to numpy array if it's a PIL Image
        if isinstance(image_input, Image.Image):
            image_input = np.array(image_input)
            if image_input.ndim == 3:  # If it has channels
                if image_input.shape[2] == 4:  # If RGBA, drop alpha channel
                    image_input = image_input[:, :, :3]
                # Permute from W x H x C to C x W x H for numpy processing
                image_input = np.transpose(image_input, (2, 0, 1))

        # Convert input to numpy array if it's a PyTorch tensor, assuming it is already C x W x H
        elif isinstance(image_input, torch.Tensor):
            # Make sure it's on CPU and convert to numpy
            image_input = image_input.cpu().numpy()

        # Check if the input is now a numpy array
        elif not isinstance(image_input, np.ndarray):
            raise TypeError("Unsupported image type. Ensure input is a PIL Image, NumPy array, or PyTorch tensor.")

        # If input is a 2D grayscale numpy array, add a channel dimension
        if image_input.ndim == 2:
            image_input = image_input[np.newaxis, :, :]

        # Resize the image if not already 192x192
        if image_input.shape[1] != 192 or image_input.shape[2] != 192:
            # Resize using scipy to maintain the channel-first format
            zoom_factors = [1, 192 / image_input.shape[1], 192 / image_input.shape[2]]
            image_input = scipy.ndimage.zoom(image_input, zoom_factors, order=1)  # Bilinear interpolation

        # Normalize the pixel values to 0-1 if the dtype indicates they are in the range 0-255
        if not np.issubdtype(image_input.dtype, np.floating):
            image_input = image_input.astype(np.float32) / 255.0

        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(image_input)

        # Move tensor to same device as model
        img_tensor = img_tensor.to(self._device_type)

        return img_tensor

    def forward(self, images, queries, max_new_tokens=750):
        answer_preambles = [''] * len(images)
        if isinstance(images, torch.Tensor) and len(images.shape) == 2:
            images = [images]
        elif isinstance(images, list) and len(images) == 1 and isinstance(images[0], np.ndarray) and len(images[0].shape) == 2:
            images = [images[0]]
        
        # Process images and ensure they're on the right device
        processed_images = []
        for image in images:
            # Convert and move to correct device
            processed_img = self.convert_any_image_to_normalized_tensor(image)
            processed_images.append(processed_img)
        
        # Stack images and ensure on right device
        images_tensor = torch.stack(processed_images, dim=0)
        images_tensor = images_tensor.to(self._device_type)
        
        # Now query the model
        outputs, samples = self.model.query(
            images_tensor, 
            queries, 
            answer_preamble=answer_preambles, 
            max_new_tokens=max_new_tokens, 
            output_only=True, 
            return_samples=True
        )
        return outputs

def load_retinavlm_specialist_from_hf(config):
    print("Loading model with safe initialization...")
    
    # Step 1: Define configuration
    try:
        rvlm_config = RetinaVLMConfig.from_pretrained(
            "RobbieHolland/RetinaVLM", 
            subfolder="RetinaVLM-Specialist"
        )
        if hasattr(config, 'to_dict'):
            rvlm_config.update(config.to_dict())
        else:
            rvlm_config.update(config)
        
        if hasattr(rvlm_config, 'model') and hasattr(rvlm_config.model, 'checkpoint_path'):
            rvlm_config.model.checkpoint_path = None
    except Exception as e:
        print(f"Error loading config: {e}")
        rvlm_config = RetinaVLMConfig()
    
    # Step 2: Create model with explicit GPU initialization first
    try:
        # Initialize model on GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = RetinaVLM(rvlm_config, device=device)
        return model.eval()
    
    except Exception as e:
        print(f"Error during model creation: {e}")
        print("Trying alternative loading method...")
        
        # Try loading transformers directly
        try:
            # First check if the package is up-to-date
            print("Checking if transformers package needs to be updated...")
            import subprocess
            subprocess.run(['pip', 'install', '--upgrade', 'transformers'])
            
            print("Attempting to load with device_map='auto'...")
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                "RobbieHolland/RetinaVLM",
                subfolder="RetinaVLM-Specialist",
                device_map="auto",
                torch_dtype=torch.float32
            )
            return model.eval()
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            
            # Try installing from source and loading again
            try:
                print("Attempting to install transformers from source...")
                subprocess.run(['pip', 'install', 'git+https://github.com/huggingface/transformers.git'])
                
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    "RobbieHolland/RetinaVLM",
                    subfolder="RetinaVLM-Specialist",
                    device_map="auto",
                    torch_dtype=torch.float32
                )
                return model.eval()
            except Exception as e3:
                print(f"Source installation and loading failed: {e3}")
                raise RuntimeError("Failed to initialize model")

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def load_from_api(config):
    model = load_retinavlm_specialist_from_hf(config)
    return model

if __name__ == "__main__":
    # Run your preferred function
    load_from_api()