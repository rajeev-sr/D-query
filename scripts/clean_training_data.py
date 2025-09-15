#!/usr/bin/env python3
"""
Data cleaning script to fix training data issues causing gradient explosion
"""
import json
import re
from typing import List, Dict

class TrainingDataCleaner:
    def __init__(self, max_total_length: int = 1024, max_output_length: int = 512):
        self.max_total_length = max_total_length
        self.max_output_length = max_output_length
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Fix common formatting issues
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
        
        return text
    
    def truncate_response(self, response: str, max_length: int) -> str:
        """Intelligently truncate response while preserving structure"""
        if len(response) <= max_length:
            return response
        
        # Try to truncate at sentence boundary
        sentences = response.split('. ')
        truncated = ""
        for sentence in sentences:
            if len(truncated + sentence + '. ') <= max_length - 20:  # Leave room for closing
                truncated += sentence + '. '
            else:
                break
        
        if truncated:
            # Add proper closing
            truncated = truncated.strip()
            if not truncated.endswith(('.', '!', '?')):
                truncated += '.'
            return truncated
        else:
            # Fallback: hard truncate with ellipsis
            return response[:max_length-3] + '...'
    
    def clean_training_item(self, item: Dict) -> Dict:
        """Clean a single training item"""
        cleaned = {
            'instruction': self.clean_text(item.get('instruction', '')),
            'input': item.get('input', ''),
            'output': self.clean_text(item.get('output', ''))
        }
        
        # Check if output is too long
        if len(cleaned['output']) > self.max_output_length:
            cleaned['output'] = self.truncate_response(cleaned['output'], self.max_output_length)
        
        # Check total length
        total_length = len(cleaned['instruction']) + len(cleaned['output'])
        if total_length > self.max_total_length:
            # Reduce output further if needed
            available_for_output = self.max_total_length - len(cleaned['instruction']) - 50
            if available_for_output > 0:
                cleaned['output'] = self.truncate_response(cleaned['output'], available_for_output)
        
        return cleaned
    
    def clean_dataset(self, input_file: str, output_file: str) -> None:
        """Clean entire dataset"""
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON on line {line_num}: {e}")
        
        print(f"Loaded {len(data)} training examples")
        
        # Clean data
        cleaned_data = []
        skipped = 0
        
        for i, item in enumerate(data):
            try:
                cleaned_item = self.clean_training_item(item)
                
                # Validate cleaned item
                if (len(cleaned_item['instruction']) > 0 and 
                    len(cleaned_item['output']) > 0 and
                    len(cleaned_item['instruction'] + cleaned_item['output']) <= self.max_total_length):
                    cleaned_data.append(cleaned_item)
                else:
                    skipped += 1
                    print(f"Skipped item {i}: invalid after cleaning")
                    
            except Exception as e:
                skipped += 1
                print(f"Error cleaning item {i}: {e}")
        
        # Save cleaned data
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Cleaned dataset saved to {output_file}")
        print(f"Final samples: {len(cleaned_data)}")
        print(f"Skipped samples: {skipped}")
        
        # Report statistics
        self.report_statistics(cleaned_data)
    
    def report_statistics(self, data: List[Dict]) -> None:
        """Report statistics on cleaned data"""
        instruction_lengths = [len(item['instruction']) for item in data]
        output_lengths = [len(item['output']) for item in data]
        total_lengths = [len(item['instruction'] + item['output']) for item in data]
        
        import numpy as np
        
        print(f"\n=== CLEANED DATA STATISTICS ===")
        print(f"Total samples: {len(data)}")
        
        print(f"\nInstruction lengths:")
        print(f"  Mean: {np.mean(instruction_lengths):.0f} chars")
        print(f"  Max: {np.max(instruction_lengths)} chars")
        
        print(f"\nOutput lengths:")
        print(f"  Mean: {np.mean(output_lengths):.0f} chars")
        print(f"  Max: {np.max(output_lengths)} chars")
        
        print(f"\nTotal sequence lengths:")
        print(f"  Mean: {np.mean(total_lengths):.0f} chars")
        print(f"  Max: {np.max(total_lengths)} chars")
        
        # Validate all sequences are within limits
        over_limit = sum(1 for length in total_lengths if length > self.max_total_length)
        print(f"\nSequences over {self.max_total_length} chars: {over_limit}")


def main():
    cleaner = TrainingDataCleaner(max_total_length=1024, max_output_length=400)
    
    input_file = "data/processed/training_data.jsonl"
    output_file = "data/processed/training_data_cleaned.jsonl"
    
    print("=== CLEANING TRAINING DATA ===")
    cleaner.clean_dataset(input_file, output_file)
    print("=== CLEANING COMPLETE ===")


if __name__ == "__main__":
    main()