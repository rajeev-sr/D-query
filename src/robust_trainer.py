import torch
from src.model_trainer import ModelTrainer
from src.training_monitor import TrainingMonitor
import os
import gc
import psutil

class RobustTrainer(ModelTrainer):
    def _retry_with_smaller_batch(self, dataset, output_dir):
        """Retry training with a smaller batch size and increased gradient accumulation."""
        print("Retrying training with smaller batch size...")
        # Get current training args and reduce batch size
        training_args = self._get_optimal_training_args(dataset, output_dir)
        # Set minimum batch size and maximum gradient accumulation
        training_args.per_device_train_batch_size = 1
        training_args.gradient_accumulation_steps = max(16, training_args.gradient_accumulation_steps * 2)
        try:
            return self._execute_training(dataset, training_args, output_dir)
        except Exception as e:
            print(f"Retry with smaller batch also failed: {e}")
            return False
    def __init__(self, base_model_name="microsoft/DialoGPT-medium"):
        super().__init__(base_model_name)
        self.monitor = TrainingMonitor()
    
    def train_with_monitoring(self, dataset, output_dir="models/fine_tuned"):
        """Enhanced training with monitoring and error recovery"""
        
        # Pre-training checks
        print("Pre-training checks...")
        if not self._check_system_requirements():
            return False
        
        if not self._check_dataset_quality(dataset):
            return False
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Adjust training parameters based on system
        training_args = self._get_optimal_training_args(dataset, output_dir)
        
        try:
            return self._execute_training(dataset, training_args, output_dir)
        
        except Exception as e:
            print(f"Training failed: {e}")
            return self._handle_training_failure(e, dataset, output_dir)
    
    def _check_system_requirements(self):
        """Check if system can handle training"""
        # Check available memory
        available_ram = psutil.virtual_memory().available / (1024**3)
        if available_ram < 2.0:
            print(f"Low RAM: {available_ram:.1f} GB available")
            print("Consider closing other applications")
        
        # Check disk space
        disk_usage = psutil.disk_usage('.')
        available_disk = disk_usage.free / (1024**3)
        if available_disk < 5.0:
            print(f"Low disk space: {available_disk:.1f} GB available")
            return False
        
        return True
    
    def _check_dataset_quality(self, dataset):
        """Validate dataset before training"""
        if len(dataset) < 5:
            print("Dataset too small (minimum 5 samples required)")
            return False
        
        if len(dataset) > 1000:
            print(f"Large dataset ({len(dataset)} samples) - training may take long")

        return True
    
    def _get_optimal_training_args(self, dataset, output_dir):
        """Get optimal training arguments based on system and data"""
        
        # Determine batch size based on available memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory > 10:
                batch_size = 4
                gradient_accumulation = 2
            elif gpu_memory > 6:
                batch_size = 2
                gradient_accumulation = 4
            else:
                batch_size = 1
                gradient_accumulation = 8
        else:
            batch_size = 1
            gradient_accumulation = 8
        
        # Determine epochs based on dataset size
        if len(dataset) < 20:
            epochs = 5
        elif len(dataset) < 100:
            epochs = 3
        else:
            epochs = 2
        
        from transformers import TrainingArguments
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=5e-5,
            warmup_steps=min(100, len(dataset) // 10),
            logging_steps=max(1, len(dataset) // 20),
            save_steps=max(50, len(dataset) // 4),
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None,  # Disable wandb/tensorboard
        )
    
    def _execute_training(self, dataset, training_args, output_dir):
        """Execute the actual training"""
        from transformers import Trainer, DataCollatorForLanguageModeling
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Custom trainer with monitoring
        class MonitoredTrainer(Trainer):
            def __init__(self, monitor, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.monitor = monitor
            
            def log(self, logs, *args, **kwargs):
                super().log(logs, *args, **kwargs)
                if 'train_loss' in logs:
                    self.monitor.log_progress(
                        self.state.global_step,
                        logs['train_loss'],
                        logs.get('learning_rate')
                    )
        
        # Initialize trainer
        trainer = MonitoredTrainer(
            monitor=self.monitor,
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save monitoring data
        self.monitor.save_logs()
        self.monitor.plot_training_curve()
        
        print(f"Training completed! Model saved to {output_dir}")
        return True
    
    def _handle_training_failure(self, error, dataset, output_dir):
        """Handle training failures with recovery options"""
        print(f"\nAttempting recovery from error: {error}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Try with smaller model if memory error
        if "memory" in str(error).lower() or "cuda" in str(error).lower():
            print("Memory error detected - trying with smaller model...")
            return self._try_smaller_model(dataset, output_dir)
        
        # Try with smaller batch size
        if "batch" in str(error).lower():
            print("Batch size error - reducing batch size...")
            return self._retry_with_smaller_batch(dataset, output_dir)
        
        return False
    
    def _try_smaller_model(self, dataset, output_dir):
        """Fallback to smaller model"""
        smaller_models = ["distilgpt2", "gpt2"]
        
        for model_name in smaller_models:
            try:
                print(f"Trying smaller model: {model_name}")
                
                # Reinitialize with smaller model
                self.base_model_name = model_name
                if self.setup_model_and_tokenizer():
                    self.setup_lora_config()
                    
                    # Retry training
                    return self.train_with_monitoring(dataset, output_dir)
                    
            except Exception as e:
                print(f"{model_name} also failed: {e}")
                continue
        
        return False