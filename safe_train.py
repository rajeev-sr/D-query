#!/usr/bin/env python3
"""
Ultra-safe training script with aggressive safety measures
"""
import torch
import json
import os
from src.model_trainer import ModelTrainer
from src.model_fallbacks import ModelFallbacks

class UltraSafeTrainer(ModelTrainer):
    def __init__(self, model_name="microsoft/DialoGPT-small"):  # Use smaller model
        super().__init__(model_name)
        self.max_loss_threshold = 10.0  # Stop if loss exceeds this
        self.nan_tolerance = 0  # Stop immediately on NaN
        
    def _ultra_safe_training_args(self, dataset, output_dir):
        """Ultra-conservative training arguments"""
        from transformers import TrainingArguments
        
        # Very small batch and ultra-low learning rate
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,  # Just 1 epoch
            per_device_train_batch_size=1,  # Minimum batch size
            gradient_accumulation_steps=4,
            learning_rate=1e-6,  # Ultra-low learning rate
            warmup_steps=10,
            logging_steps=5,  # Log very frequently
            save_steps=20,  # Save very frequently
            save_total_limit=5,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=False,
            bf16=False,
            max_grad_norm=0.1,  # Very strict gradient clipping
            adam_epsilon=1e-8,
            weight_decay=0.001,
            lr_scheduler_type="constant",  # No learning rate decay
            report_to=None,
            eval_steps=None,
            evaluation_strategy="no",
        )
    
    def train_ultra_safe(self, dataset, output_dir="models/fine_tuned"):
        """Ultra-safe training with aggressive monitoring"""
        print("=== ULTRA-SAFE TRAINING MODE ===")
        
        # Extra data validation
        print("Validating training data...")
        if not self._validate_dataset_deeply(dataset):
            print("❌ Dataset validation failed")
            return False
        
        training_args = self._ultra_safe_training_args(dataset, output_dir)
        
        try:
            from transformers import Trainer, DataCollatorForLanguageModeling
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal LM, not masked LM
            )
            
            # Custom trainer with loss monitoring
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                callbacks=[self._get_safety_callback()],
            )
            
            print("Starting ultra-safe training...")
            trainer.train()
            
            # Validate model after training
            if self._validate_final_model():
                trainer.save_model()
                self.tokenizer.save_pretrained(output_dir)
                print("✅ Ultra-safe training completed successfully!")
                return True
            else:
                print("❌ Model validation failed after training")
                return False
                
        except Exception as e:
            print(f"❌ Ultra-safe training failed: {e}")
            return False
    
    def _validate_dataset_deeply(self, dataset):
        """Deep validation of dataset"""
        print(f"Validating {len(dataset)} samples...")
        
        for i, sample in enumerate(dataset):
            # Check for NaN or inf in input_ids
            input_ids = torch.tensor(sample['input_ids'])
            if torch.isnan(input_ids.float()).any() or torch.isinf(input_ids.float()).any():
                print(f"❌ Invalid values in sample {i}")
                return False
            
            # Check sequence length
            if len(input_ids) > 256:  # Very conservative length
                print(f"❌ Sample {i} too long: {len(input_ids)} tokens")
                return False
                
        print("✅ Dataset validation passed")
        return True
    
    def _get_safety_callback(self):
        """Callback to monitor training and stop on anomalies"""
        from transformers import TrainerCallback
        
        class SafetyCallback(TrainerCallback):
            def __init__(self, max_loss=10.0):
                self.max_loss = max_loss
                
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if logs and 'loss' in logs:
                    loss = logs['loss']
                    print(f"Step {state.global_step}: Loss = {loss:.4f}")
                    
                    # Check for problematic loss
                    if loss > self.max_loss or torch.isnan(torch.tensor(loss)):
                        print(f"❌ Stopping: Loss {loss} exceeds threshold {self.max_loss}")
                        control.should_training_stop = True
                        
                    # Check gradient norm
                    if 'grad_norm' in logs:
                        grad_norm = logs['grad_norm']
                        if torch.isnan(torch.tensor(grad_norm)):
                            print("❌ Stopping: NaN gradient detected")
                            control.should_training_stop = True
                            
        return SafetyCallback(self.max_loss_threshold)
    
    def _validate_final_model(self):
        """Validate model produces reasonable outputs"""
        print("Validating final model...")
        
        test_inputs = [
            "Hello",
            "What is your name?",
            "How are you today?",
        ]
        
        for test_input in test_inputs:
            try:
                inputs = self.tokenizer(
                    test_input,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=50
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Check for NaN or inf
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print(f"❌ Model produces invalid values for input: {test_input}")
                        return False
                        
            except Exception as e:
                print(f"❌ Model validation failed for '{test_input}': {e}")
                return False
        
        print("✅ Model validation passed")
        return True


def main():
    print("=== ULTRA-SAFE TRAINING APPROACH ===")
    
    # Use the smallest model for maximum stability
    trainer = UltraSafeTrainer("microsoft/DialoGPT-small")
    
    # Setup model
    if not trainer.setup_model_and_tokenizer():
        print("Failed to setup model")
        return False
    
    trainer.setup_lora_config()
    
    # Load cleaned dataset
    dataset = trainer.prepare_dataset("data/processed/training_data_cleaned.jsonl")
    if not dataset:
        print("Failed to load dataset")
        return False
    
    # Take only a small subset for ultra-safe training
    small_dataset = dataset.select(range(min(100, len(dataset))))
    print(f"Using small subset: {len(small_dataset)} samples")
    
    # Train with ultra-safe parameters
    success = trainer.train_ultra_safe(small_dataset)
    
    if success:
        print("\n🎉 ULTRA-SAFE TRAINING COMPLETED!")
        print("Model saved to: models/fine_tuned/")
    else:
        print("\n💥 ULTRA-SAFE TRAINING FAILED")
        
    return success


if __name__ == "__main__":
    main()