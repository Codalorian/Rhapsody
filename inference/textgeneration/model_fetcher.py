import requests
from typing import Dict, List, Tuple
import re
import streamlit as st # Import st for caching

class ModelFetcher:
    def __init__(self):
        self.hf_api_url = "https://huggingface.co/api/models"
        self.company_patterns = {
            "OpenAI": ["openai-community", "openai"],
            "Meta": ["meta-llama", "facebook"],
            "Google": ["google", "google-bert"],
            "Microsoft": ["microsoft"],
            "Anthropic": ["anthropic"],
            "EleutherAI": ["EleutherAI"],
            "Stability AI": ["stabilityai"],
            "Mistral AI": ["mistralai"],
            "BigScience": ["bigscience"],
            "Salesforce": ["Salesforce"],
            "Cohere": ["cohere"],
            "AI21 Labs": ["ai21labs"],
            "Technology Innovation Institute": ["tiiuae"],
            "Together": ["togethercomputer"],
            "Databricks": ["databricks"],
            "01.AI": ["01-ai"],
            "Qwen": ["Qwen"],
            "DeepSeek": ["deepseek"],
            "HuggingFace": ["HuggingFace", "hf-internal-testing"], # Add HuggingFace itself
        }
        
    def get_model_size(self, model_id: str, model_info: dict) -> str:
        """Extract model size from model ID or info"""
        # Common size patterns
        size_patterns = [
            (r'(\d+\.?\d*)[Bb]', lambda x: f"{float(x)}B"), # e.g., 7B, 1.3B
            (r'(\d+\.?\d*)[Mm]', lambda x: f"{int(float(x))}M"), # e.g., 125M, 350M
            (r'(\d+)[Kk]', lambda x: f"{int(x)/1000}M"), # e.g., 70K -> 0.07M
        ]
        
        # Check model ID for size
        for pattern, formatter in size_patterns:
            match = re.search(pattern, model_id)
            if match:
                size = match.group(1)
                return formatter(size)
        
        # Check for common size indicators
        size_keywords = {
            "small": "Small",
            "base": "Base",
            "medium": "Medium",
            "large": "Large",
            "xl": "XL",
            "xxl": "XXL",
            "tiny": "Tiny",
            "mini": "Mini",
            "pico": "Pico",
            "nano": "Nano",
        }
        
        model_id_lower = model_id.lower()
        for keyword, size in size_keywords.items():
            if keyword in model_id_lower:
                return size
                
        # Fallback if no clear size number or keyword is found but it's a known model type often with a size suffix
        if any(kw in model_id_lower for kw in ["gpt2", "opt", "phi", "flan-t5", "bert"]):
            return "Unknown (Common)"
        
        return "Unknown"
    
    def get_company(self, model_id: str) -> str:
        """Determine company from model ID"""
        model_owner = model_id.split("/")[0] if "/" in model_id else ""
        
        for company, patterns in self.company_patterns.items():
            for pattern in patterns:
                if pattern.lower() == model_owner.lower(): # Exact match for owner
                    return company
                # For more general patterns, ensure it's at the start or clearly separated
                if re.match(r'^' + re.escape(pattern).lower() + r'($|/|-)', model_owner.lower()):
                    return company
        
        return "Miscellaneous"
    
    @st.cache_data(ttl=3600*24) # Cache for 24 hours to avoid frequent API calls
    def fetch_text_generation_models(_self, limit: int = 1000) -> Dict[str, Dict[str, List[str]]]:
        """Fetch text generation models from Hugging Face"""
        # Using _self because this is a cached method of the class
        try:
            params = {
                "filter": "text-generation",
                "sort": "downloads",
                "direction": "-1",
                "limit": limit
            }
            
            response = requests.get(_self.hf_api_url, params=params)
            response.raise_for_status()
            models = response.json()
            
            organized_models = {}
            
            for model in models:
                model_id = model.get("modelId", "")
                if not model_id:
                    continue
                
                if model.get("private", False):
                    continue
                
                company = _self.get_company(model_id)
                size = _self.get_model_size(model_id, model)
                
                if company not in organized_models:
                    organized_models[company] = {}
                
                if size not in organized_models[company]:
                    organized_models[company][size] = []
                
                # Only add if not already present (can happen with different size inferences)
                if model_id not in organized_models[company][size]:
                    organized_models[company][size].append(model_id)
            
            # Sort models within each size group alphabetically
            for company in organized_models:
                for size in organized_models[company]:
                    organized_models[company][size].sort()

            return organized_models
            
        except Exception as e:
            st.error(f"Error fetching Hugging Face models: {e}. Using fallback models.")
            return _self._get_fallback_models()
    
    # Store descriptions for popular models locally, as HF API doesn't provide them for all
    # These will be the descriptions for Transformers models too.
    def get_model_description(self, model_id: str, company: str, size: str) -> str:
        if model_id in MODEL_DESCRIPTIONS_TRANSFORMERS:
            return MODEL_DESCRIPTIONS_TRANSFORMERS[model_id]
        
        if company == "Miscellaneous":
            return f"A text generation model from an independent creator with {size} parameters."
        return f"A text generation model from {company} with {size} parameters."

    def get_popular_models(self) -> List[str]:
        # Return a curated list of models. This could also be fetched or manually maintained.
        return [
            "gpt2",
            "gpt2-medium",
            "microsoft/phi-2",
            "EleutherAI/gpt-neo-125M",
            "distilgpt2",
            "google/flan-t5-small",
            "facebook/opt-125m",
            "mistralai/Mistral-7B-v0.1" # Example of a larger model that might be popular
        ]
    
    def _get_fallback_models(self) -> Dict[str, Dict[str, List[str]]]:
        """Fallback models in case HF API call fails."""
        return {
            "OpenAI": {
                "Small": ["gpt2", "distilgpt2"],
                "Medium": ["gpt2-medium"],
                "Large": ["gpt2-large"],
                "XL": ["gpt2-xl"]
            },
            "EleutherAI": {
                "Small": ["EleutherAI/gpt-neo-125M"],
                "Medium": ["EleutherAI/gpt-neo-1.3B"]
            },
            "Microsoft": {
                "Small": ["microsoft/phi-1_5"]
            },
            "Google": {
                "Small": ["google/flan-t5-small"]
            },
            "Miscellaneous": {
                "Unknown": ["facebook/opt-125m"]
            }
        }

# Predefined descriptions for some common Transformers models
MODEL_DESCRIPTIONS_TRANSFORMERS = {
    "gpt2": "A generative pre-trained transformer model by OpenAI, known for coherent text generation.",
    "gpt2-medium": "A larger version of GPT-2, offering improved generation quality.",
    "gpt2-large": "An even larger GPT-2 model, capable of more complex and nuanced text.",
    "gpt2-xl": "The largest GPT-2 model, known for high-quality, long-form generation.",
    "distilgpt2": "A distilled version of GPT-2, faster and smaller while retaining much of its performance.",
    "microsoft/phi-2": "A small, high-quality language model from Microsoft, excelling in common sense reasoning and language understanding.",
    "EleutherAI/gpt-neo-125M": "A small, open-source GPT-style model from EleutherAI, good for basic text generation.",
    "EleutherAI/gpt-neo-1.3B": "A larger GPT-Neo model offering more robust language capabilities.",
    "google/flan-t5-small": "A smaller variant of Google's FLAN-T5, fine-tuned for instruction following and diverse tasks.",
    "google/flan-t5-base": "A base version of Google's FLAN-T5, providing strong performance across various NLP tasks.",
    "facebook/opt-125m": "A foundational language model from Facebook, part of the Open Pre-trained Transformer family.",
    "facebook/opt-350m": "A mid-sized model in the OPT family, balancing size and performance.",
    "mistralai/Mistral-7B-v0.1": "A powerful 7B parameter model from Mistral AI, known for strong performance and efficiency."
}

