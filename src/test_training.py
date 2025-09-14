from src.model_trainer import ModelTrainer
from src.model_fallbacks import ModelFallbacks
from src.data_validator import DataValidator

def test_training_pipeline():
    print("=== TESTING TRAINING PIPELINE ===")
    
    # Step 1: Validate data
    print("1. Validating training data...")
    validator = DataValidator("data/processed/training_dataset.csv")
    stats = validator.validate_dataset()
    
    if not stats['sufficient_data']:
        print("Insufficient training data. Please label more emails.")
        return False
    
    training_file = validator.prepare_training_format()
    
    # Step 2: Select model
    print("2. Selecting compatible model...")
    try:
        model_name = ModelFallbacks.select_best_model()
    except Exception as e:
        print(f"No compatible models: {e}")
        return False
    
    # Step 3: Setup trainer
    print("3. Setting up trainer...")
    trainer = ModelTrainer(model_name)
    
    if not trainer.setup_model_and_tokenizer():
        return False
    
    trainer.setup_lora_config()
    
    # Step 4: Prepare data
    print("4. Preparing dataset...")
    dataset = trainer.prepare_dataset(training_file)
    
    if dataset is None:
        return False
    
    print(f"Training pipeline ready!")
    print(f"Dataset: {len(dataset)} samples")
    print(f"Model: {model_name}")
    
    # Optional: Start training (comment out if you want to test first)
    # print("5. Starting training...")
    # success = trainer.train_model(dataset)
    # return success
    
    return True

if __name__ == "__main__":
    test_training_pipeline()