# train_model.py
from src.robust_trainer import RobustTrainer
from src.data_validator import DataValidator
import os

def main():
    print("=== STARTING MODEL TRAINING ===")
    
    # Check prerequisites
    if not os.path.exists("data/processed/training_data.jsonl"):
        print("Training data not found. Running data preparation...")
        validator = DataValidator("data/processed/training_data.csv")
        validator.prepare_training_format()
    
    # Initialize trainer
    from src.model_fallbacks import ModelFallbacks
    model_name = ModelFallbacks.select_best_model()
    
    trainer = RobustTrainer(model_name)
    
    # Setup model
    print("Setting up model...")
    if not trainer.setup_model_and_tokenizer():
        print("Failed to setup model")
        return False
    
    trainer.setup_lora_config()
    
    # Prepare dataset
    print("Loading dataset...")
    dataset = trainer.prepare_dataset("data/processed/training_data.jsonl")
    if not dataset:
        print("Failed to load dataset")
        return False

    print(f"Dataset ready: {len(dataset)} samples")
    
    # Start training
    print("Starting training process...")
    success = trainer.train_with_monitoring(dataset)
    
    if success:
        print("\nTRAINING COMPLETED SUCCESSFULLY!")
        print("Model saved to: models/fine_tuned/")
        print("Training logs: models/training_logs.json")
        print("Training curve: models/training_curve.png")
    else:
        print("\nTRAINING FAILED")
        return False
    
    return True

if __name__ == "__main__":
    main()