# src/data_labeler.py
import pandas as pd
import streamlit as st
from typing import Dict, List

class DataLabeler:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
    
    def create_labeling_interface(self):
        """Create Streamlit interface for data labeling"""
        st.title("Email Data Labeling Interface")
        
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0
        
        current_idx = st.session_state.current_index
        total_emails = len(self.df)
        
        if current_idx >= total_emails:
            st.success("All emails labeled!")
            return
        
        email = self.df.iloc[current_idx]
        
        # Display email
        st.subheader(f"Email {current_idx + 1} of {total_emails}")
        st.text_area("Subject + Body:", value=email['query'], height=200)
        st.write(f"**From:** {email['sender']}")
        
        # Labeling controls
        col1, col2 = st.columns(2)
        
        with col1:
            category = st.selectbox(
                "Category:",
                ['academic', 'administrative', 'technical', 'complex', 'spam'],
                index=0 if pd.isna(email['category']) else 
                      ['academic', 'administrative', 'technical', 'complex', 'spam'].index(email['category'])
            )
        
        # Response template
        response = st.text_area(
            "Sample Response (how you would typically respond):",
            value=email['response'] if not pd.isna(email['response']) else "",
            height=150
        )
        
        # Navigation buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Previous") and current_idx > 0:
                st.session_state.current_index -= 1
                st.rerun()
        
        with col2:
            if st.button("Save & Next"):
                self.df.iloc[current_idx, self.df.columns.get_loc('category')] = category
                self.df.iloc[current_idx, self.df.columns.get_loc('response')] = response
                
                self.df.to_csv(self.csv_path, index=False)
                st.session_state.current_index += 1
                st.rerun()
        
        with col3:
            if st.button("Skip"):
                st.session_state.current_index += 1
                st.rerun()
        
        # Progress bar
        progress = current_idx / total_emails
        st.progress(progress)
        st.write(f"Progress: {current_idx}/{total_emails} ({progress*100:.1f}%)")

# Run labeling interface
def run_labeler():
    labeler = DataLabeler("data/processed/training_dataset.csv")
    labeler.create_labeling_interface()

if __name__ == "__main__":
    run_labeler()