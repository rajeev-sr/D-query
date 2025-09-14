# test_validation.py
from src.data_validator import DataValidator

validator = DataValidator("/home/rajeev-kumar/Desktop/D-query/data/processed/training_data.csv")
stats = validator.validate_dataset()

if stats['sufficient_data']:
    print("Dataset ready for fine-tuning")
    training_file = validator.prepare_training_format()
    print(f"Training data saved to: {training_file}")
else:
    print("Need more labeled data")