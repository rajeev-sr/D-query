from src.data_validator import DataValidator
from src.model_trainer import ModelTrainer
import os

def integration():
    print("=== INTEGRATION TEST ===")
    
    # Check if we have labeled data
    if not os.path.exists("/home/rajeev-kumar/Desktop/D-query/data/processed/training_data.csv"):
        print("No training dataset found. Run Day 1 first.")
        return False
    
    # Validate dataset
    validator = DataValidator("/home/rajeev-kumar/Desktop/D-query/data/processed/training_data.csv")
    stats = validator.validate_dataset()
    
    print(f"\nDataset Stats:")
    print(f"- Total samples: {stats['total_samples']}")
    print(f"- Labeled samples: {stats['labeled_samples']}")
    
    # Check if we need more labeling
    if stats['labeled_samples'] < 20:
        print(f"\nWarning: Only {stats['labeled_samples']} labeled samples.")
        print("Recommend labeling at least 20-50 samples for better results.")
        print("Run: streamlit run src/data_labeler.py")
        
        # Still proceed with available data for testing
        if stats['labeled_samples'] < 5:
            print("Need at least 5 samples to proceed")
            return False
    
    # Prepare training data
    print("\nPreparing training format...")
    training_file = validator.prepare_training_format()
    
    # Test model loading
    print("\nTesting model setup...")
    from src.model_fallbacks import ModelFallbacks
    
    try:
        model_name = ModelFallbacks.select_best_model()
        trainer = ModelTrainer(model_name)
        
        if trainer.setup_model_and_tokenizer():
            print("Model setup successful")
            trainer.setup_lora_config()
            
            # Test dataset preparation
            dataset = trainer.prepare_dataset(training_file)
            if dataset:
                print("Dataset preparation successful")
                print(f"Ready to train with {len(dataset)} samples")
            else:
                print("Dataset preparation failed")
                return False
        else:
            print("Model setup failed")
            return False
            
    except Exception as e:
        print(f"Model setup error: {e}")
        return False

    print("\nDAY 2 COMPLETE - Ready for training!")
    return True

if __name__ == "__main__":
    integration()