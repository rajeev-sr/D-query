#!/usr/bin/env python3
"""
Data validation and corruption fix script for D-query fine-tuning.
This script identifies and fixes common data issues that cause model corruption.
"""
import json
import re
import os

def validate_and_fix_training_data(input_file, output_file):
    """
    Validate and fix training data JSON format and content.
    """
    print(f"Validating and fixing: {input_file}")
    
    fixed_data = []
    issues_found = []
    
    try:
        # Try reading as JSONL first
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Check if it's malformed JSON (separate objects without newlines)
        if content.startswith('{') and not content.startswith('['):
            # Split by '}\n{' pattern
            objects = re.split(r'}\s*\n\s*{', content)
            
            # Fix the first and last objects
            if len(objects) > 1:
                objects[0] = objects[0] + '}'
                objects[-1] = '{' + objects[-1]
                
                # Fix middle objects
                for i in range(1, len(objects)-1):
                    objects[i] = '{' + objects[i] + '}'
            
            # Parse each object
            for i, obj_str in enumerate(objects):
                try:
                    obj = json.loads(obj_str)
                    
                    # Validate required fields
                    if not obj.get('instruction'):
                        issues_found.append(f"Line {i+1}: Missing instruction")
                        continue
                    if not obj.get('output'):
                        issues_found.append(f"Line {i+1}: Missing output")
                        continue
                    
                    # Clean and validate content
                    instruction = obj['instruction'].strip()
                    output = obj['output'].strip()
                    
                    # Remove problematic characters
                    instruction = re.sub(r'[^\w\s\n\r\.\,\?\!\:\;\-\(\)\[\]\"\'\/\\]', '', instruction)
                    output = re.sub(r'[^\w\s\n\r\.\,\?\!\:\;\-\(\)\[\]\"\'\/\\]', '', output)
                    
                    # Check for excessive length
                    if len(instruction) > 2000:
                        instruction = instruction[:2000] + "..."
                        issues_found.append(f"Line {i+1}: Instruction truncated")
                    
                    if len(output) > 2000:
                        output = output[:2000] + "..."
                        issues_found.append(f"Line {i+1}: Output truncated")
                    
                    # Ensure non-empty after cleaning
                    if instruction and output:
                        fixed_data.append({
                            "instruction": instruction,
                            "input": obj.get("input", ""),
                            "output": output
                        })
                    else:
                        issues_found.append(f"Line {i+1}: Empty after cleaning")
                        
                except json.JSONDecodeError as e:
                    issues_found.append(f"Line {i+1}: JSON parse error: {e}")
                except Exception as e:
                    issues_found.append(f"Line {i+1}: Unexpected error: {e}")
    
    except Exception as e:
        print(f"Failed to read input file: {e}")
        return False
    
    # Report issues
    if issues_found:
        print(f"Found {len(issues_found)} issues:")
        for issue in issues_found[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues_found) > 10:
            print(f"  ... and {len(issues_found) - 10} more issues")
    
    # Save fixed data
    if fixed_data:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in fixed_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"✅ Fixed data saved to {output_file}")
            print(f"✅ {len(fixed_data)} valid samples saved")
            return True
            
        except Exception as e:
            print(f"Failed to save fixed data: {e}")
            return False
    else:
        print("❌ No valid data found after cleaning")
        return False

def validate_model_directory(model_path):
    """
    Check if the model directory contains suspicious files that could indicate corruption.
    """
    print(f"Validating model directory: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ Model directory does not exist")
        return False
    
    required_files = ['config.json', 'tokenizer_config.json']
    model_files = ['pytorch_model.bin', 'model.safetensors', 'adapter_model.bin']
    
    issues = []
    
    # Check required files
    for file in required_files:
        path = os.path.join(model_path, file)
        if not os.path.exists(path):
            issues.append(f"Missing required file: {file}")
        else:
            # Check file size
            size = os.path.getsize(path)
            if size == 0:
                issues.append(f"Empty file: {file}")
            elif size < 100:  # Very small config files are suspicious
                issues.append(f"Suspiciously small file: {file} ({size} bytes)")
    
    # Check model files
    model_file_found = False
    for file in model_files:
        path = os.path.join(model_path, file)
        if os.path.exists(path):
            model_file_found = True
            size = os.path.getsize(path)
            if size == 0:
                issues.append(f"Empty model file: {file}")
            elif size < 1000000:  # Less than 1MB is suspicious for model weights
                issues.append(f"Suspiciously small model file: {file} ({size} bytes)")
    
    if not model_file_found:
        issues.append("No model weight files found")
    
    if issues:
        print("❌ Model validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Model directory looks valid")
        return True

def main():
    """Main validation and fixing routine"""
    print("=== D-QUERY DATA CORRUPTION FIX ===")
    
    # Fix training data
    input_file = "data/processed/fine_tuning_data.json"
    output_file = "data/processed/fine_tuning_data_fixed.jsonl"
    
    if os.path.exists(input_file):
        if validate_and_fix_training_data(input_file, output_file):
            print(f"✅ Training data fixed and saved to {output_file}")
            print(f"💡 Use {output_file} for training instead of {input_file}")
        else:
            print("❌ Failed to fix training data")
    else:
        print(f"❌ Training data file not found: {input_file}")
    
    # Validate model
    model_path = "models/fine_tuned"
    if os.path.exists(model_path):
        if not validate_model_directory(model_path):
            print("❌ Model appears corrupted. Consider deleting and retraining.")
            print("💡 To delete: rm -rf models/fine_tuned")
    else:
        print("ℹ️ No existing model found (this is ok for new training)")
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. Delete corrupted model: rm -rf models/fine_tuned")
    print("2. Use fixed training data: data/processed/fine_tuning_data_fixed.jsonl")
    print("3. Retrain with safer parameters (lower learning rate, no fp16)")
    print("4. Monitor training for NaN losses")

if __name__ == "__main__":
    main()