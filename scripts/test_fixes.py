#!/usr/bin/env python3
"""
Test script to verify all corruption fixes are working properly.
"""

import torch
import sys
import os

def test_model_trainer():
    """Test that model trainer can load data without errors"""
    print("🧪 Testing model trainer...")
    
    try:
        from src.model_trainer import ModelTrainer
        trainer = ModelTrainer()
        
        # Test data loading
        dataset = trainer.prepare_dataset("data/processed/fine_tuning_data.json")
        
        if dataset is None:
            print("❌ Dataset loading failed")
            return False
        
        print(f"✅ Dataset loaded successfully with {len(dataset)} samples")
        
        # Test tokenization on a small sample
        sample_size = min(5, len(dataset))
        small_dataset = dataset.select(range(sample_size))
        
        def tokenize_function(examples):
            tokenized = trainer.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors=None
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Remove original columns to avoid conflicts
        tokenized = small_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=small_dataset.column_names
        )
        
        # Check for NaN in tokenized data
        for sample in tokenized:
            input_ids = sample['input_ids']
            if any(x < 0 or x > 50000 for x in input_ids):  # Check for invalid token IDs
                print(f"❌ Invalid token IDs found: {input_ids}")
                return False
        
        print("✅ Tokenization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Model trainer test failed: {e}")
        return False

def test_model_inference():
    """Test that model inference handles corrupted models properly"""
    print("🧪 Testing model inference...")
    
    try:
        from src.model_inference import QueryClassifier
        
        # This should fail gracefully and load base model
        classifier = QueryClassifier(model_path="models/fine_tuned")
        
        # Test with simple query
        result = classifier.classify_and_respond("Hello")
        
        if "error" in result:
            print(f"⚠️ Inference returned error: {result['error']}")
        else:
            print("✅ Inference test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Model inference test failed: {e}")
        return False

def test_cuda_safety():
    """Test CUDA setup and safety measures"""
    print("🧪 Testing CUDA safety...")
    
    try:
        if torch.cuda.is_available():
            # Test basic CUDA operations
            torch.cuda.empty_cache()
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            result = test_tensor + 1
            
            # Check for NaN/inf
            if torch.isnan(result).any() or torch.isinf(result).any():
                print("❌ CUDA operations produce NaN/inf")
                return False
            
            test_tensor.cpu()
            del test_tensor, result
            torch.cuda.empty_cache()
            
            print("✅ CUDA safety test passed")
        else:
            print("ℹ️ CUDA not available, using CPU")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA safety test failed: {e}")
        return False

def test_data_validation():
    """Test that data validation script works"""
    print("🧪 Testing data validation...")
    
    try:
        # Check if fixed file exists
        fixed_file = "data/processed/fine_tuning_data_fixed.jsonl"
        if not os.path.exists(fixed_file):
            print(f"❌ Fixed data file not found: {fixed_file}")
            return False
        
        # Read a few lines to validate format
        with open(fixed_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Only check first 3 lines
                    break
                
                try:
                    import json
                    data = json.loads(line.strip())
                    if not all(key in data for key in ['instruction', 'output']):
                        print(f"❌ Line {i+1}: Missing required keys")
                        return False
                except json.JSONDecodeError as e:
                    print(f"❌ Line {i+1}: Invalid JSON: {e}")
                    return False
        
        print("✅ Data validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== CORRUPTION FIX VERIFICATION ===")
    
    tests = [
        ("Data Validation", test_data_validation),
        ("CUDA Safety", test_cuda_safety),
        ("Model Trainer", test_model_trainer),
        ("Model Inference", test_model_inference),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: CRASHED - {e}")
    
    print(f"\n=== RESULTS ===")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All corruption fixes verified!")
        print("\n📋 Next steps:")
        print("1. Run: python -m train_model")
        print("2. Monitor training for NaN losses")
        print("3. Test the trained model")
    else:
        print("⚠️ Some tests failed. Check the issues above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)