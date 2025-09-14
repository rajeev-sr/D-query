import pandas as pd
from collections import Counter

class DataValidator:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
    
    def validate_dataset(self):
        """Validate training dataset quality"""
        print("=== DATASET VALIDATION ===")
        
        # Basic stats
        total_rows = len(self.df)
        labeled_rows = self.df['category'].notna().sum()
        response_rows = self.df['response'].notna().sum()
        
        print(f"Total emails: {total_rows}")
        print(f"Labeled emails: {labeled_rows} ({labeled_rows/total_rows*100:.1f}%)")
        print(f"With responses: {response_rows} ({response_rows/total_rows*100:.1f}%)")
        
        # Category distribution
        category_counts = Counter(self.df['category'].dropna())
        print(f"\nCategory distribution:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")
        
        # Check minimum samples per category
        min_samples = 10
        insufficient_categories = [cat for cat, count in category_counts.items() if count < min_samples]
        
        if insufficient_categories:
            print(f"\nWarning: Categories with < {min_samples} samples: {insufficient_categories}")
            print("Consider collecting more data for these categories.")
        
        # Response length analysis
        response_lengths = self.df['response'].dropna().apply(len)
        print(f"\nResponse lengths:")
        print(f"  Average: {response_lengths.mean():.1f} characters")
        print(f"  Min: {response_lengths.min()}")
        print(f"  Max: {response_lengths.max()}")
        
        return {
            'total_samples': total_rows,
            'labeled_samples': labeled_rows,
            'category_distribution': dict(category_counts),
            'sufficient_data': len(insufficient_categories) == 0
        }
    
    def prepare_training_format(self):
        """Convert to fine-tuning format"""
        # Filter only labeled data
        labeled_df = self.df.dropna(subset=['category', 'response'])
        
        training_data = []
        for _, row in labeled_df.iterrows():
            # Format for instruction-following
            instruction = f"You are an institute assistant. Classify and respond to this student query.\n\nQuery: {row['query']}"
            
            output = f"Category: {row['category']}\nResponse: {row['response']}"
            
            training_data.append({
                'instruction': instruction,
                'input': '',
                'output': output
            })
        
        # Save training format
        training_df = pd.DataFrame(training_data)
        output_path = self.csv_path.replace('.csv', '.jsonl')
        training_df.to_json(output_path, orient='records', lines=True)
        
        print(f"Prepared {len(training_data)} samples for fine-tuning")
        print(f"Saved to: {output_path}")
        return output_path