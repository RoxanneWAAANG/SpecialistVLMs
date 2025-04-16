from run.vision_language_pretraining import MiniGPT4Module
import hydra
import torch
import sys
from PIL import Image
import numpy as np
from huggingface_hub import login, HfApi
import scipy
import textwrap
from transformers import PreTrainedModel, PretrainedConfig, LogitsProcessorList, StoppingCriteriaList
from transformers import RepetitionPenaltyLogitsProcessor, NoRepeatNGramLogitsProcessor

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
        
        # Improved generation parameters to prevent repetitions
        generation_params = {
            'max_new_tokens': max_new_tokens,
            'num_beams': 3,
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'repetition_penalty': 1.2,
            'no_repeat_ngram_size': 3,
            'output_only': True,
            'return_samples': True
        }
        
        # If the model includes a custom query method, use it
        if hasattr(self.model, 'query'):
            try:
                outputs, samples = self.model.query(
                    images_tensor, 
                    queries, 
                    answer_preamble=answer_preambles,
                    **generation_params
                )
                return outputs
            except Exception as e:
                print(f"Error using model.query: {e}")
                print("Trying alternative generation method...")
        
        # Alternative generation method in case the default one fails
        try:
            # Create a direct prompt to enhance generation quality
            prompts = []
            for i, query in enumerate(queries):
                # Format with clear instructions
                enhanced_prompt = f"[Image Analysis Instruction]\n{query}\n\n[Detailed Analysis]:"
                prompts.append(enhanced_prompt)
            
            # Get logits processors to improve text quality
            logits_processor = LogitsProcessorList([
                RepetitionPenaltyLogitsProcessor(penalty=1.2),
                NoRepeatNGramLogitsProcessor(3)
            ])
            
            # Extract the language model from the architecture
            if hasattr(self.model, 'llama_model'):
                lm = self.model.llama_model
            elif hasattr(self.model, 'language_model'):
                lm = self.model.language_model
            else:
                raise AttributeError("Cannot find language model in the architecture")
            
            # Process image through vision encoder (assuming standard architecture)
            image_embeds = self.model.encode_image(images_tensor)
            
            # Generate text
            outputs = []
            for i, prompt in enumerate(prompts):
                text_input = self.model.tokenizer(prompt, return_tensors="pt").to(self._device_type)
                combined_embeds = self.model.get_combined_embeddings(text_input, image_embeds[i:i+1])
                
                output_tokens = lm.generate(
                    inputs_embeds=combined_embeds,
                    max_length=max_new_tokens,
                    num_beams=3,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    logits_processor=logits_processor
                )
                
                output_text = self.model.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                
                # Extract just the analysis part
                if "[Detailed Analysis]:" in output_text:
                    output_text = output_text.split("[Detailed Analysis]:")[1].strip()
                
                outputs.append(output_text)
            
            return outputs
            
        except Exception as e:
            print(f"Alternative generation also failed: {e}")
            print("Falling back to basic generation...")
            
            # Last resort: simplified generation
            outputs = []
            for query in queries:
                basic_output = f"OCT Image Analysis:\n\nUnable to generate a proper analysis due to model generation issues. Please check model configuration and ensure proper weight loading."
                outputs.append(basic_output)
            
            return outputs

def load_retinavlm_specialist_from_hf(config):
    print("Loading model with safe initialization...")
    
    # Step 1: Define configuration
    try:
        # Try to load config from repository
        rvlm_config = RetinaVLMConfig.from_pretrained(
            "RobbieHolland/RetinaVLM", 
            subfolder="RetinaVLM-Specialist"
        )
        
        # Update with user config
        if hasattr(config, 'to_dict'):
            rvlm_config.update(config.to_dict())
        else:
            rvlm_config.update(config)
        
        # Ensure checkpoint path is cleared to avoid conflicts
        if hasattr(rvlm_config, 'model') and hasattr(rvlm_config.model, 'checkpoint_path'):
            rvlm_config.model.checkpoint_path = None
            
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration instead")
        rvlm_config = RetinaVLMConfig()
        
        # Still update with user config
        if hasattr(config, 'to_dict'):
            rvlm_config.update(config.to_dict())
        else:
            rvlm_config.update(config)
    
    # Add generation parameters to config to improve output quality
    rvlm_config.generation_config = {
        'temperature': 0.7,
        'top_p': 0.9,
        'repetition_penalty': 1.2,
        'no_repeat_ngram_size': 3,
        'num_beams': 3,
        'do_sample': True
    }
    
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
            # Update transformers if needed
            print("Checking if transformers package needs to be updated...")
            import subprocess
            try:
                subprocess.run(['pip', 'install', '--upgrade', 'transformers'], check=True)
                print("Transformers updated successfully")
            except subprocess.CalledProcessError:
                print("Failed to update transformers package")
            
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
                subprocess.run(['pip', 'install', 'git+https://github.com/huggingface/transformers.git'], check=True)
                
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
def demo_inference(config):
    import os
    from pathlib import Path
    
    # Get or create the image path
    try:
        # First, try to get the image path from the config
        image_path = config.get('input_image_path', None)
        
        # If not found, look in default locations
        if not image_path or not os.path.exists(image_path):
            # Try to find a sample image in the dataset directory
            dataset_dir = Path(config.get('images_for_figures_dir', 'dataset/processed_images'))
            
            # Look for any image file
            image_files = list(dataset_dir.glob('*.jpg')) + list(dataset_dir.glob('*.png'))
            
            if image_files:
                image_path = str(image_files[0])
                print(f"Using sample image: {image_path}")
            else:
                raise FileNotFoundError("No sample images found in dataset directory")
        
        # Load the image
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        print(f"Loading image from {image_path}")
        image = Image.open(image_path)
        
        # Prepare query
        query = "Write an extensive report describing the OCT image and listing any visible biomarkers or other observations. Do not provide disease stage or patient referral recommendations yet."
        
        # Load model
        print("Loading model...")
        retinavlm = load_retinavlm_specialist_from_hf(config).eval()
        
        # Run inference
        print("Running inference...")
        output = retinavlm.forward([image], [query])
        
        # Print result
        print("\n==== RESULT ====\n")
        print(output[0])
        print("\n================\n")
        
    except Exception as e:
        print(f"Error in demo inference: {e}")
        import traceback
        traceback.print_exc()

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def load_from_api(config):
    model = load_retinavlm_specialist_from_hf(config)
    return model

if __name__ == "__main__":
    # Run the demo inference
    demo_inference()