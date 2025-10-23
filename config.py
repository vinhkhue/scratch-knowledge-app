"""
Configuration module for Scratch Knowledge Graph App
"""
import os
from pathlib import Path

# Application Settings
APP_TITLE = "Scratch Knowledge Graph"
APP_DESCRIPTION = "Hệ thống truy vấn kiến thức lập trình Scratch lớp 8"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# GraphRAG Configuration
GRAPHRAG_ROOT_DIR = Path(__file__).parent / "data" / "scratch_index"
GRAPHRAG_OUTPUT_DIR = GRAPHRAG_ROOT_DIR / "output"
GRAPHRAG_INPUT_DIR = GRAPHRAG_ROOT_DIR / "input"
GRAPHRAG_CONFIG_FILE = GRAPHRAG_ROOT_DIR / "settings.yaml"

# Parquet file paths
ENTITIES_FILE = GRAPHRAG_OUTPUT_DIR / "entities.parquet"
RELATIONSHIPS_FILE = GRAPHRAG_OUTPUT_DIR / "relationships.parquet"
COMMUNITIES_FILE = GRAPHRAG_OUTPUT_DIR / "communities.parquet"
COMMUNITY_REPORTS_FILE = GRAPHRAG_OUTPUT_DIR / "community_reports.parquet"
TEXT_UNITS_FILE = GRAPHRAG_OUTPUT_DIR / "text_units.parquet"

# Query Configuration
DEFAULT_COMMUNITY_LEVEL = 2
DEFAULT_RESPONSE_TYPE = "Multiple Paragraphs"
MIN_RESPONSE_LENGTH = 50  # Minimum length to consider response valid

# Graph Visualization Configuration
MAX_GRAPH_NODES = 50  # Limit nodes for visualization
GRAPH_LAYOUT = "spring"  # NetworkX layout algorithm
GRAPH_FIGSIZE = (12, 8)  # Matplotlib figure size

# Web Search Configuration
WEB_SEARCH_TIMEOUT = 30  # Timeout for web search in seconds
WEB_SEARCH_MAX_RESULTS = 5  # Maximum number of search results



