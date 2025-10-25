#!/usr/bin/env python3
"""
Generate 5 interview conversations with API limit handling
"""

from generator import generate_dataset

if __name__ == "__main__":
    print("Starting generation of 5 interview conversations...")
    print("Progress will be saved after each conversation.")
    print("-" * 50)
    
    conversations = generate_dataset(
        num_conversations=5,
        output_file="interview_dataset_5.json"
    )
    
    print(f"\nGeneration complete!")
    print(f"Successfully generated {len(conversations)} conversations")
    print(f"Dataset saved to: interview_dataset_5.json")
