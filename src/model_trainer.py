
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import pandas as pd

class ModelTrainer:
    def __init__(self, base_model_name="microsoft/DialoGPT-medium"):
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.model = None
        self.peft_model = None
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with proper configuration"""
        print(f"Loading model: {self.base_model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # Add special tokens if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            print("Model and tokenizer loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def setup_lora_config(self):
        """Setup LoRA configuration"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn"]  # Changed for DialoGPT
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        print("LoRA configuration applied")
        
        self.peft_model.print_trainable_parameters()
        return self.peft_model
    
    def prepare_dataset(self, json_file_path):
        """Fixed dataset preparation"""
        try:
            # Load data
            with open(json_file_path, 'r') as f:
                data = [json.loads(line) for line in f]
            
            print(f"Loaded {len(data)} training examples")
            
            # Prepare texts for training
            texts = []
            for item in data:
                # Create a simple format for DialoGPT
                instruction = item.get('instruction', '')
                output = item.get('output', '')
                
                # Simple conversation format
                text = f"<|endoftext|>User: {instruction}\nAssistant: {output}<|endoftext|>"
                texts.append(text)
            
            # Create dataset
            dataset = Dataset.from_dict({"text": texts})
            
            print(f"Dataset created with {len(dataset)} samples")
            
            # Tokenize function - FIXED VERSION
            def tokenize_function(examples):
                # Tokenize all texts at once
                tokenized = self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=256,  # Reduced max length
                    return_tensors=None  # Important: Don't return tensors yet
                )
                
                # Labels are the same as input_ids for language modeling
                tokenized["labels"] = tokenized["input_ids"].copy()
                
                return tokenized
            
            # Apply tokenization
            tokenized_dataset = dataset.map(
                tokenize_function, 
                batched=True,
                remove_columns=dataset.column_names  # Remove original columns
            )
            
            print(f"Dataset tokenized successfully")
            return tokenized_dataset
            
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            print(f"Error details: {type(e).__name__}: {str(e)}")
            return None
    
    def train_model(self, dataset, output_dir="models/fine_tuned"):
        """Fixed training method"""
        if not self.peft_model:
            print("LoRA model not initialized")
            return False
        
        try:
            # Data collator for language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                return_tensors="pt"
            )
            
            # Training arguments - more conservative
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=2,  # Reduced epochs
                per_device_train_batch_size=1,  # Start with batch size 1
                gradient_accumulation_steps=4,
                learning_rate=5e-5,
                warmup_steps=50,
                logging_steps=5,
                save_steps=100,
                save_total_limit=2,
                prediction_loss_only=True,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                fp16=torch.cuda.is_available(),  # Use fp16 if GPU available
                report_to=None  # Disable wandb
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )
            
            print("Starting fixed training...")
            
            # Train the model
            trainer.train()
            
            # Save the model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)

            print(f"Model saved to {output_dir}")
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            print(f"Error type: {type(e).__name__}")
            return False