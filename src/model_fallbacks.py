class ModelFallbacks:
    @staticmethod
    def get_available_models():
        """List of fallback models if primary fails"""
        return [
            "microsoft/DialoGPT-small",   # Start with smaller model for stability
            "distilgpt2",                 # Smallest option  
            "gpt2",                       # Most compatible
            "microsoft/DialoGPT-medium",  # Larger model (last resort)
        ]
    
    @staticmethod
    def test_model_availability(model_name):
        """Test if model can be loaded"""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return True
        except Exception as e:
            print(f"Model {model_name} not available: {e}")
            return False
    
    @staticmethod
    def select_best_model():
        """Select the best available model"""
        models = ModelFallbacks.get_available_models()
        
        for model in models:
            if ModelFallbacks.test_model_availability(model):
                print(f"Selected model: {model}")
                return model
        
        raise Exception("No compatible models available")