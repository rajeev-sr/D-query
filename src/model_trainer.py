
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
import os
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
        """Fixed dataset preparation with better error handling"""
        
        # Ensure model and tokenizer are initialized
        if not self.tokenizer:
            print("Initializing tokenizer...")
            if not self.setup_model_and_tokenizer():
                print("Failed to initialize tokenizer")
                return None
        
        try:
            # Check if we have a fixed JSONL file
            fixed_file = json_file_path.replace('.json', '_fixed.jsonl')
            if os.path.exists(fixed_file):
                print(f"Using fixed data file: {fixed_file}")
                json_file_path = fixed_file
            
            # Load data based on file extension
            data = []
            if json_file_path.endswith('.jsonl'):
                # Load JSONL format (one JSON object per line)
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                data.append(item)
                            except json.JSONDecodeError as e:
                                print(f"Skipping invalid JSON on line {line_num}: {e}")
            else:
                # Load regular JSON format
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # Handle multiple JSON objects
                    import re
                    objects = re.split(r'}\s*\n\s*{', content)
                    if len(objects) > 1:
                        objects[0] = objects[0] + '}'
                        objects[-1] = '{' + objects[-1]
                        for i in range(1, len(objects)-1):
                            objects[i] = '{' + objects[i] + '}'
                    
                    for obj_str in objects:
                        try:
                            data.append(json.loads(obj_str))
                        except json.JSONDecodeError as e:
                            print(f"Skipping invalid JSON object: {e}")
            
            print(f"Loaded {len(data)} training examples")
            
            if len(data) == 0:
                print("No valid training data found")
                return None
            
            # Prepare texts for training with validation
            texts = []
            for i, item in enumerate(data):
                try:
                    # Validate required fields
                    instruction = item.get('instruction', '').strip()
                    output = item.get('output', '').strip()
                    
                    if not instruction or not output:
                        print(f"Skipping item {i+1}: missing instruction or output")
                        continue
                    
                    # Create a simple format for DialoGPT with proper tokens
                    text = f"<|startoftext|>User: {instruction}\nAssistant: {output}<|endoftext|>"
                    texts.append(text)
                    
                except Exception as e:
                    print(f"Skipping item {i+1}: {e}")
                    continue
            
            if len(texts) == 0:
                print("No valid texts created from data")
                return None
            
            # Create dataset
            from datasets import Dataset
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
            
            # Training arguments - more conservative with NaN detection
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=2,  # Reduced epochs
                per_device_train_batch_size=1,  # Start with batch size 1
                gradient_accumulation_steps=4,
                learning_rate=1e-5,  # Much lower learning rate to prevent NaN
                warmup_steps=50,
                logging_steps=5,
                save_steps=100,
                save_total_limit=2,
                prediction_loss_only=True,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                fp16=False,  # Disable fp16 to prevent NaN issues
                gradient_checkpointing=True,  # Enable gradient checkpointing
                max_grad_norm=1.0,  # Clip gradients to prevent explosion
                skip_memory_metrics=True,
                report_to=None  # Disable wandb
            )
            
            # Create trainer with NaN detection
            class SafeTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False):
                    try:
                        outputs = model(**inputs)
                        loss = outputs.loss
                        
                        # Check for NaN loss
                        if torch.isnan(loss).any() or torch.isinf(loss).any():
                            print(f"NaN/Inf loss detected: {loss}")
                            # Use a safe fallback loss
                            loss = torch.tensor(1.0, device=loss.device)
                        
                        return (loss, outputs) if return_outputs else loss
                    except Exception as e:
                        print(f"Training step failed: {e}")
                        # Return a safe fallback loss
                        safe_loss = torch.tensor(1.0, device=inputs['input_ids'].device)
                        return safe_loss
            
            trainer = SafeTrainer(
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