#!/usr/bin/env python3
"""
Script đơn giản để regenerate data từ scratch
Usage: python regenerate_data.py
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    print("🔄 Regenerating Scratch Knowledge Graph data...")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please run: export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        # Import and run the main extraction
        from extract_entities_relationships import build_kg
        
        print("📊 Starting extraction...")
        build_kg()
        
        print("✅ Data regeneration completed!")
        print("📁 Output files:")
        
        output_dir = Path("output")
        for file in output_dir.glob("*.parquet"):
            print(f"  - {file.name}")
            
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return

if __name__ == "__main__":
    main()

