import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch
import streamlit as st # Added Streamlit import for caching resource

class TextGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Replace '/' with '_' for directory names to avoid path issues
        self.model_path = f"models/{model_name.replace('/', '__')}" 
        
        # This part of init should not be directly cached by st.cache_resource on the class,
        # but the function that instantiates this class should be cached.
        self._load_model()
    
    def _load_model(self):
        """Load model from local storage or download if not present."""
        # Check if model exists locally
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # Use torch.bfloat16 for better memory/speed if supported by GPU
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        else:
            print(f"Downloading {self.model_name} to {self.model_path}")
            os.makedirs(self.model_path, exist_ok=True)
            
            # Download and save
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
            
            # Save to local directory
            self.tokenizer.save_pretrained(self.model_path)
            self.model.save_pretrained(self.model_path)
            print(f"Model saved to {self.model_path}")
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.to("cuda")
            print(f"Model {self.model_name} moved to GPU.")
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id


    def generate_stream(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7, **kwargs):
        """Generate text with token streaming using a Transformers model."""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move inputs to GPU if model is on GPU
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Generation parameters (default for Transformers)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True, # Always use do_sample if temperature is not 1.0
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id, # Explicitly set pad_token_id
        )
        
        # Add optional parameters from kwargs (passed from Streamlit UI)
        # Ensure only valid parameters for model.generate are passed
        valid_generate_args = [
            'top_p', 'top_k', 'repetition_penalty', 'num_beams', 
            'no_repeat_ngram_size', 'num_return_sequences', 'early_stopping',
            'length_penalty', 'bad_words_ids', 'force_words_ids',
            'decoder_start_token_id', 'diversity_penalty', 'encoder_repetition_penalty',
            'min_length', 'max_length' # Max_length is often max_new_tokens + prompt_length
        ]
        
        for k, v in kwargs.items():
            if k in valid_generate_args and v is not None:
                generation_kwargs[k] = v
        
        # Handle do_sample = False explicitly if temperature is 1.0 or if explicitly set to False
        if temperature == 1.0 and 'do_sample' not in kwargs:
             generation_kwargs['do_sample'] = False
        elif 'do_sample' in kwargs: # Respect explicit do_sample setting
             generation_kwargs['do_sample'] = kwargs['do_sample']

        # Ensure num_beams is at least 1 for non-greedy decoding
        if 'num_beams' in generation_kwargs and generation_kwargs['num_beams'] > 1:
            generation_kwargs['do_sample'] = True # Beam search usually combined with sampling
            # For beam search, we might want to disable top_k/top_p or use with caution
            # if 'top_k' in generation_kwargs: del generation_kwargs['top_k']
            # if 'top_p' in generation_kwargs: del generation_kwargs['top_p']
        
        # Run generation in separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens as they come
        for token in streamer:
            yield token
        
        thread.join() # Wait for the generation thread to complete
