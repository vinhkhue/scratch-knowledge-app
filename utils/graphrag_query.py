"""
GraphRAG Query Module for Scratch Knowledge Graph App
Enhanced with LLM integration for better responses
"""
import pandas as pd
import asyncio
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging
import openai
import json

from config import (
    GRAPHRAG_OUTPUT_DIR,
    GRAPHRAG_CONFIG_FILE,
    ENTITIES_FILE,
    RELATIONSHIPS_FILE,
    COMMUNITIES_FILE,
    COMMUNITY_REPORTS_FILE,
    TEXT_UNITS_FILE,
    DEFAULT_COMMUNITY_LEVEL,
    DEFAULT_RESPONSE_TYPE,
    MIN_RESPONSE_LENGTH,
    OPENAI_API_KEY
)

logger = logging.getLogger(__name__)

class GraphRAGQuery:
    """GraphRAG query handler"""
    
    def __init__(self):
        self.entities = pd.DataFrame()
        self.relationships = pd.DataFrame()
        self.communities = pd.DataFrame()
        self.community_reports = pd.DataFrame()
        self.text_units = pd.DataFrame()
        self.config = None
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self._load_data()
    
    def _load_data(self):
        """Load parquet files and config"""
        try:
            # Load parquet files if they exist
            if ENTITIES_FILE.exists():
                self.entities = pd.read_parquet(ENTITIES_FILE)
                logger.info(f"Loaded {len(self.entities)} entities")
            
            if RELATIONSHIPS_FILE.exists():
                self.relationships = pd.read_parquet(RELATIONSHIPS_FILE)
                logger.info(f"Loaded {len(self.relationships)} relationships")
            
            if COMMUNITIES_FILE.exists():
                self.communities = pd.read_parquet(COMMUNITIES_FILE)
                logger.info(f"Loaded {len(self.communities)} communities")
            
            if COMMUNITY_REPORTS_FILE.exists():
                self.community_reports = pd.read_parquet(COMMUNITY_REPORTS_FILE)
                logger.info(f"Loaded {len(self.community_reports)} community reports")
            
            if TEXT_UNITS_FILE.exists():
                self.text_units = pd.read_parquet(TEXT_UNITS_FILE)
                logger.info(f"Loaded {len(self.text_units)} text units")
            
            # Load config
            if GRAPHRAG_CONFIG_FILE.exists():
                try:
                    from graphrag.config import GraphRagConfig
                    self.config = GraphRagConfig.from_yaml(str(GRAPHRAG_CONFIG_FILE))
                except ImportError:
                    logger.warning("GraphRAG not available, using mock config")
                    self.config = None
            
        except Exception as e:
            logger.error(f"Error loading GraphRAG data: {e}")
            # Don't create mock data, keep empty DataFrames
            logger.info("Using empty DataFrames - no mock data")
    
    
    async def local_search(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Perform local search using GraphRAG"""
        try:
            if self.config is None:
                # Use enhanced search with real data
                return await self._enhanced_search_with_llm(query, "local")
            
            # Import GraphRAG API
            from graphrag.api import local_search
            
            response, context_data = await local_search(
                config=self.config,
                entities=self.entities,
                communities=self.communities,
                community_reports=self.community_reports,
                text_units=self.text_units,
                relationships=self.relationships,
                covariates=None,
                community_level=DEFAULT_COMMUNITY_LEVEL,
                response_type=DEFAULT_RESPONSE_TYPE,
                query=query
            )
            
            return response, context_data
            
        except Exception as e:
            logger.error(f"Error in local search: {e}")
            return await self._enhanced_search_with_llm(query, "local")
    
    async def global_search(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Perform global search using GraphRAG"""
        try:
            if self.config is None:
                # Use enhanced search with real data
                return await self._enhanced_search_with_llm(query, "global")
            
            # Import GraphRAG API
            from graphrag.api import global_search
            
            response, context_data = await global_search(
                config=self.config,
                entities=self.entities,
                communities=self.communities,
                community_reports=self.community_reports,
                community_level=DEFAULT_COMMUNITY_LEVEL,
                dynamic_community_selection=True,
                response_type=DEFAULT_RESPONSE_TYPE,
                query=query
            )
            
            return response, context_data
            
        except Exception as e:
            logger.error(f"Error in global search: {e}")
            return await self._enhanced_search_with_llm(query, "global")
    
    async def _enhanced_search_with_llm(self, query: str, search_type: str) -> Tuple[str, Dict[str, Any]]:
        """Enhanced search using LLM to generate intelligent responses"""
        query_lower = query.lower()
        
        # Find relevant entities - improved to include related entities
        relevant_entities = []
        for _, entity in self.entities.iterrows():
            if any(keyword in entity['title'].lower() or keyword in entity['description'].lower() 
                   for keyword in query_lower.split()):
                relevant_entities.append(entity)
        
        # Find additional entities through relationships
        relevant_entity_ids = set(entity['id'] for entity in relevant_entities)
        
        # Find entities that are connected to relevant entities through relationships
        for _, rel in self.relationships.iterrows():
            if rel['source'] in relevant_entity_ids or rel['target'] in relevant_entity_ids:
                # Add the other entity in the relationship
                if rel['source'] not in relevant_entity_ids:
                    source_entity = self.entities[self.entities['id'] == rel['source']]
                    if not source_entity.empty:
                        relevant_entities.append(source_entity.iloc[0])
                        relevant_entity_ids.add(rel['source'])
                
                if rel['target'] not in relevant_entity_ids:
                    target_entity = self.entities[self.entities['id'] == rel['target']]
                    if not target_entity.empty:
                        relevant_entities.append(target_entity.iloc[0])
                        relevant_entity_ids.add(rel['target'])
        
        # Find relevant relationships - improved logic
        relevant_relationships = []
        
        # Find relationships involving these entities
        for _, rel in self.relationships.iterrows():
            # Check if relationship involves relevant entities
            if (rel['source'] in relevant_entity_ids or 
                rel['target'] in relevant_entity_ids or
                any(keyword in rel['description'].lower() for keyword in query_lower.split())):
                relevant_relationships.append(rel)
        
        # Find relevant text units
        relevant_text_units = []
        for _, text_unit in self.text_units.iterrows():
            if any(keyword in text_unit['text'].lower() for keyword in query_lower.split()):
                relevant_text_units.append(text_unit)
        
        # Prepare context for LLM
        context_info = {
            'entities': [{'title': e['title'], 'description': e['description']} for e in relevant_entities],
            'relationships': [{'description': r['description']} for r in relevant_relationships],
            'text_chunks': [{'text': t['text'][:500]} for t in relevant_text_units[:3]]  # Limit text length
        }
        
        # Generate intelligent response using LLM
        try:
            response = await self._generate_llm_response(query, context_info)
            
            # Check if LLM requests web search
            if "activate_web_search_tool" in response.lower():
                logger.info("LLM requested web search - returning special response")
                return "activate_web_search_tool", {
                    'entities': pd.DataFrame(relevant_entities) if relevant_entities else pd.DataFrame(),
                    'relationships': pd.DataFrame(relevant_relationships) if relevant_relationships else pd.DataFrame(),
                    'text_units': pd.DataFrame(relevant_text_units) if relevant_text_units else pd.DataFrame(),
                    'search_type': search_type
                }
                
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            response = self._fallback_response(query, relevant_entities, relevant_relationships)
        
        # Create context data for visualization
        context_data = {
            'entities': pd.DataFrame(relevant_entities) if relevant_entities else pd.DataFrame(),
            'relationships': pd.DataFrame(relevant_relationships) if relevant_relationships else pd.DataFrame(),
            'text_units': pd.DataFrame(relevant_text_units) if relevant_text_units else pd.DataFrame(),
            'search_type': search_type
        }
        
        return response, context_data
    
    async def _generate_llm_response(self, query: str, context_info: Dict[str, Any]) -> str:
        """Generate intelligent response using OpenAI with web search capability"""
        
        # Prepare context string
        context_str = ""
        if context_info['entities']:
            context_str += "Các khái niệm liên quan:\n"
            for entity in context_info['entities']:
                context_str += f"- {entity['title']}: {entity['description']}\n"
        
        if context_info['relationships']:
            context_str += "\nMối quan hệ:\n"
            for rel in context_info['relationships']:
                context_str += f"- {rel['description']}\n"
        
        if context_info['text_chunks']:
            context_str += "\nNội dung liên quan:\n"
            for chunk in context_info['text_chunks']:
                context_str += f"- {chunk['text']}\n"
        
        # Create prompt for LLM with web search instruction
        prompt = f"""Bạn là một chuyên gia về lập trình Scratch, hãy trả lời câu hỏi của học sinh một cách chi tiết và dễ hiểu.

Câu hỏi: {query}

Thông tin từ cơ sở dữ liệu kiến thức:
{context_str}

QUAN TRỌNG: Nếu câu hỏi về:
- Phiên bản mới nhất của Scratch (sau 2023)
- Thông tin cập nhật gần đây
- Tính năng mới được thêm vào
- Thông tin không có trong dữ liệu hiện tại

Thì KHÔNG trả lời chi tiết, chỉ nói ngắn gọn rằng không có thông tin và kết thúc bằng: "activate_web_search_tool"

Hãy trả lời:
1. Giải thích chi tiết và dễ hiểu
2. Đưa ra ví dụ cụ thể nếu có thể
3. Sử dụng ngôn ngữ phù hợp với học sinh lớp 8
4. Nếu có liên quan đến các khái niệm khác, hãy giải thích mối liên hệ
5. Kết thúc bằng lời khuyên hoặc gợi ý thực hành
6. Nếu thông tin không đủ (đặc biệt về phiên bản mới), chỉ nói ngắn gọn và kết thúc bằng "activate_web_search_tool"

Trả lời bằng tiếng Việt, ngắn gọn nhưng đầy đủ thông tin."""

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là một giáo viên tin học chuyên về Scratch, có kinh nghiệm dạy học sinh lớp 8."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def _fallback_response(self, query: str, entities: list, relationships: list) -> str:
        """Fallback response when LLM fails"""
        if entities:
            response = f"Dựa trên kiến thức về Scratch, đây là câu trả lời cho câu hỏi '{query}':\n\n"
            for entity in entities[:3]:
                response += f"• **{entity['title']}**: {entity['description']}\n"
            
            if relationships:
                response += "\n**Mối quan hệ liên quan:**\n"
                for rel in relationships[:2]:
                    response += f"• {rel['description']}\n"
        else:
            response = f"Không tìm thấy thông tin cụ thể về '{query}' trong cơ sở dữ liệu kiến thức Scratch."
        
        return response
    
    def is_response_valid(self, response: str) -> bool:
        """Check if response is valid and meaningful"""
        if not response or len(response.strip()) < MIN_RESPONSE_LENGTH:
            return False
        
        # Check for common error patterns
        error_patterns = [
            "không tìm thấy",
            "không có thông tin",
            "error",
            "exception",
            "failed"
        ]
        
        response_lower = response.lower()
        for pattern in error_patterns:
            if pattern in response_lower:
                return False
        
        return True
    
    def should_use_web_search(self, graph_stats: dict, entities_count: int) -> bool:
        """Quyết định có nên dùng web search dựa trên kết quả graph"""
        
        # Nếu không có mối quan hệ nào
        if graph_stats.get('edges', 0) == 0:
            logger.info("No edges found - switching to web search")
            return True
        
        # Nếu mật độ quá thấp (< 0.01) - adjusted threshold for smaller graphs
        if graph_stats.get('density', 0) < 0.01:
            logger.info(f"Low density ({graph_stats.get('density', 0):.3f}) - switching to web search")
            return True
        
        # Nếu quá ít entities (< 2) - giảm threshold
        if entities_count < 2:
            logger.info(f"Too few entities ({entities_count}) - switching to web search")
            return True
        
        # Nếu tất cả nodes đều riêng lẻ (connected_components = entities_count)
        if graph_stats.get('connected_components', 0) == entities_count:
            logger.info("All nodes are isolated - switching to web search")
            return True
        
        logger.info("Graph quality is good - using GraphRAG")
        return False

# Global instance
graphrag_query = GraphRAGQuery()
