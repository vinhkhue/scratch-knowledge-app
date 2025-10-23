# Scratch Knowledge Graph Data Extraction

## Files

- `extract_entities_relationships.py` - Main extraction script với LLM
- `regenerate_data.py` - Simple script để regenerate data
- `input/` - Input text files (Blocks.txt, scratch_clean.txt)
- `output/` - Generated parquet files

## Usage

```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Extract data
python extract_entities_relationships.py

# Or use simple script
python regenerate_data.py
```

## Output Files

- `entities.parquet` - Extracted entities (blocks + concepts)
- `relationships.parquet` - Relationships between entities  
- `documents.parquet` - Source documents
- `text_units.parquet` - Text chunks
- `communities.parquet` - Entity communities
- `community_reports.parquet` - Community summaries

## Features

- ✅ LLM-based extraction với GPT-4o-mini + GPT-4o
- ✅ Rate limit handling với retry logic
- ✅ File-specific prompts cho Blocks.txt vs Tin học 8
- ✅ Cross-file relationship generation
- ✅ Fuzzy entity matching
- ✅ Improved relationship descriptions

