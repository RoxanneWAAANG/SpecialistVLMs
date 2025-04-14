import hydra
import torch
import sys
import os
from pathlib import Path
import warnings

# Set environment variable to show full error trace
os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def debug_inference(config):
    print("Starting debug inference...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    print(f"Config paths: {config.paths}")
    print(f"Pretrained model dir: {config.pretrained_model_dir}")
    
    # Check if the model directory exists
    if hasattr(config, 'pretrained_model_dir'):
        model_dir = Path(config.pretrained_model_dir)
        print(f"Model dir exists: {model_dir.exists()}")
        if model_dir.exists():
            print(f"Model dir contents: {list(model_dir.iterdir())}")
    
    # Suppress certain warnings that might be noise
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    try:
        # First initialize RetinaVLMConfig
        from models.retinavlm_wrapper import RetinaVLMConfig
        print("Initializing RetinaVLMConfig...")
        rvlm_config = RetinaVLMConfig()
        print("RetinaVLMConfig initialized successfully")
        
        # Then try to initialize MiniGPT4Module without moving to device
        from run.vision_language_pretraining import MiniGPT4Module
        print("Initializing MiniGPT4Module...")
        module = MiniGPT4Module(config, device=None)
        print("MiniGPT4Module initialized successfully")
        
        # Try loading the model
        from models.retinavlm_wrapper import load_retinavlm_specialist_from_hf
        print("Loading model from HF...")
        retinavlm = load_retinavlm_specialist_from_hf(config)
        print("Model loaded successfully!")
        
        # Success! Run a simple inference to verify it works
        import numpy as np
        dummy_image = np.random.rand(3, 192, 192).astype(np.float32)
        dummy_image = torch.from_numpy(dummy_image)
        
        query = "Describe this image briefly."
        print(f"Running inference with query: {query}")
        
        with torch.no_grad():
            output = retinavlm.forward([dummy_image], [query], max_new_tokens=20)
        
        print(f"Inference output: {output[0]}")
        print("Debug completed successfully!")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")
        print(f"Traceback: {traceback.format_exc()}")
    
    return "Debug completed"

if __name__ == "__main__":
    debug_inference()