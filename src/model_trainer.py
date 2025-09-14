import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import json

class ModelTrainer:
    def __init__(self, base_model_name="microsoft/DialoGPT-medium"):
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.model = None
        self.peft_model = None
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        print(f"Loading model: {self.base_model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Model and tokenizer loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def setup_lora_config(self):
        """Setup LoRA configuration"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"],
            # target_modules=["q_proj", "v_proj"]  # Target attention layers
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        print("LoRA configuration applied")
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        return self.peft_model
    
    def prepare_dataset(self, json_file_path):
        """Prepare dataset for training"""
        try:
            dataset = load_dataset('json', data_files=json_file_path, split='train')
            
            def tokenize_function(examples):
                # Combine instruction and output for training
                texts = []
                for instruction, output in zip(examples['instruction'], examples['output']):
                    text = f"<|startoftext|>{instruction}\n\n{output}<|endoftext|>"
                    texts.append(text)
                
                tokenized = self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # For causal LM, labels are the same as input_ids
                tokenized["labels"] = tokenized["input_ids"].clone()
                return tokenized
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            print(f"Dataset prepared: {len(tokenized_dataset)} samples")
            
            return tokenized_dataset
            
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            return None
    
    def train_model(self, dataset, output_dir="models/fine_tuned"):
        """Train the model with LoRA"""
        if not self.peft_model:
            print("LoRA model not initialized")
            return False
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        try:
            print("Starting training...")
            trainer.train()
            
            # Save the model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            print(f"Model saved to {output_dir}")
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            return False