"""
Main Streamlit App for Scratch Knowledge Graph
"""
import streamlit as st
import asyncio
import logging
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import APP_TITLE, APP_DESCRIPTION
from utils.graphrag_query import graphrag_query
from utils.web_search import web_search_tool
from utils.graph_viz import graph_visualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def display_header():
    """Display app header and description"""
    st.title(f"🧩 {APP_TITLE}")
    st.markdown(f"**{APP_DESCRIPTION}**")
    st.markdown("---")
    
    # Add info about the app
    with st.expander("ℹ️ Giới thiệu về ứng dụng"):
        st.markdown("""
        **Scratch Knowledge Graph** là hệ thống truy vấn kiến thức thông minh được xây dựng dựa trên:
        
        - **GraphRAG**: Sử dụng Knowledge Graph để tìm kiếm và trả lời câu hỏi
        - **Web Search**: Tự động tìm kiếm thông tin bổ sung khi cần thiết
        - **Graph Visualization**: Hiển thị mối quan hệ giữa các khái niệm
        
        **Cách sử dụng:**
        1. Nhập câu hỏi về lập trình Scratch vào ô bên dưới
        2. Hệ thống sẽ tìm kiếm trong cơ sở dữ liệu kiến thức
        3. Nếu không tìm thấy, sẽ tự động tìm kiếm trên web
        4. Kết quả sẽ hiển thị kèm graph minh họa (nếu có)
        """)

def display_sidebar():
    """Display sidebar with additional options"""
    st.sidebar.title("⚙️ Cài đặt")
    
    # Search options
    st.sidebar.subheader("🔍 Tùy chọn tìm kiếm")
    search_type = st.sidebar.selectbox(
        "Loại tìm kiếm GraphRAG:",
        ["Tự động", "Local Search", "Global Search"],
        help="Local Search: Tìm kiếm cục bộ trong các thực thể liên quan\nGlobal Search: Tìm kiếm toàn cục trong tất cả báo cáo cộng đồng"
    )
    
    # Graph options
    st.sidebar.subheader("📊 Tùy chọn Graph")
    show_graph = st.sidebar.checkbox("Hiển thị graph", value=True)
    max_nodes = st.sidebar.slider("Số nút tối đa", 10, 100, 50)
    
    # Advanced options
    with st.sidebar.expander("🔧 Tùy chọn nâng cao"):
        min_response_length = st.number_input(
            "Độ dài tối thiểu câu trả lời", 
            min_value=10, max_value=200, value=50
        )
        enable_web_search = st.checkbox("Bật tìm kiếm web", value=True)
    
    return {
        'search_type': search_type,
        'show_graph': show_graph,
        'max_nodes': max_nodes,
        'min_response_length': min_response_length,
        'enable_web_search': enable_web_search
    }

async def process_query(query: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Process user query and return results"""
    results = {
        'query': query,
        'source': None,
        'response': '',
        'context_data': None,
        'graph_stats': None,
        'error': None
    }
    
    try:
        # First, try GraphRAG search
        st.info("🔍 Đang tìm kiếm trong cơ sở dữ liệu kiến thức...")
        
        if options['search_type'] == "Local Search":
            response, context_data = await graphrag_query.local_search(query)
        elif options['search_type'] == "Global Search":
            response, context_data = await graphrag_query.global_search(query)
        else:  # Auto
            # Try local search first, then global if needed
            response, context_data = await graphrag_query.local_search(query)
            if not graphrag_query.is_response_valid(response):
                response, context_data = await graphrag_query.global_search(query)
        
        # Check if LLM requested web search
        if response == "activate_web_search_tool":
            logger.info("LLM requested web search - switching to web search")
            
            if options['enable_web_search']:
                st.warning("⚠️ LLM phát hiện thông tin không đủ. Đang tìm kiếm thông tin mới nhất trên web...")
                
                web_response = await web_search_tool.search(query)
                results['source'] = 'Web Search Tool (LLM Requested)'
                results['response'] = web_response
                results['graph_stats'] = {'nodes': 0, 'edges': 0, 'density': 0, 'connected_components': 0}
                results['context_data'] = context_data
            else:
                results['source'] = 'GraphRAG'
                results['response'] = "Thông tin không đủ trong cơ sở dữ liệu. Web search bị tắt."
                results['graph_stats'] = {'nodes': 0, 'edges': 0, 'density': 0, 'connected_components': 0}
                results['context_data'] = context_data
        
        # Check if GraphRAG response is valid
        elif graphrag_query.is_response_valid(response):
            # Tính toán graph stats trước
            graph_stats = graph_visualizer.get_graph_stats(
                context_data['entities'], 
                context_data['relationships']
            )
            
            # Kiểm tra graph quality để quyết định có dùng web search không
            if graphrag_query.should_use_web_search(graph_stats, len(context_data['entities'])):
                # Graph quality kém - chuyển sang web search
                logger.info(f"Graph quality check: edges={graph_stats.get('edges', 0)}, density={graph_stats.get('density', 0):.3f}, entities={len(context_data['entities'])}")
                logger.info("Switching to web search due to poor graph quality")
                
                if options['enable_web_search']:
                    st.warning("⚠️ Chất lượng graph thấp (mật độ thấp, ít mối quan hệ). Đang tìm kiếm trên web...")
                    
                    web_response = await web_search_tool.search(query)
                    results['source'] = 'Web Search Tool'
                    results['response'] = web_response
                    results['graph_stats'] = graph_stats
                    results['context_data'] = context_data
                else:
                    results['source'] = 'GraphRAG'
                    results['response'] = "Chất lượng graph thấp. Không tìm thấy thông tin phù hợp."
                    results['graph_stats'] = graph_stats
                    results['context_data'] = context_data
            else:
                # Graph quality tốt - dùng GraphRAG
                logger.info(f"Graph quality check: edges={graph_stats.get('edges', 0)}, density={graph_stats.get('density', 0):.3f}, entities={len(context_data['entities'])}")
                logger.info("Using GraphRAG due to good graph quality")
                
                results['source'] = 'GraphRAG Knowledge Graph'
                results['response'] = response
                results['graph_stats'] = graph_stats
                results['context_data'] = context_data
        else:
            # GraphRAG didn't find good results, try web search
            if options['enable_web_search']:
                st.warning("⚠️ Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu. Đang tìm kiếm trên web...")
                
                web_response = await web_search_tool.search(query)
                results['source'] = 'Web Search Tool'
                results['response'] = web_response
                results['graph_stats'] = {'nodes': 0, 'edges': 0, 'density': 0, 'connected_components': 0}
                results['context_data'] = context_data
            else:
                results['source'] = 'GraphRAG'
                results['response'] = "Không tìm thấy thông tin phù hợp cho câu hỏi này trong cơ sở dữ liệu kiến thức."
                results['graph_stats'] = {'nodes': 0, 'edges': 0, 'density': 0, 'connected_components': 0}
                results['context_data'] = context_data
                results['error'] = "No results found"
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        results['error'] = str(e)
        results['response'] = f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"
    
    return results

def display_results(results: Dict[str, Any], options: Dict[str, Any]):
    """Display query results"""
    if results['error']:
        st.error(f"❌ Lỗi: {results['error']}")
        return
    
    # Display response
    st.subheader("💬 Câu trả lời")
    
    # Source indicator
    if results['source'] == 'GraphRAG Knowledge Graph':
        st.success("✅ **Nguồn: GraphRAG Knowledge Graph**")
    elif results['source'] == 'Web Search Tool':
        st.info("🌐 **Nguồn: Web Search Tool**")
        
        # Hiển thị lý do chuyển sang web search nếu có
        if results.get('graph_stats'):
            
            graph_stats = results['graph_stats']
            reasons = []
            
            if graph_stats.get('edges', 0) == 0:
                reasons.append("• Không có mối quan hệ nào giữa các khái niệm")
            if graph_stats.get('density', 0) < 0.05:
                reasons.append(f"• Mật độ graph quá thấp ({graph_stats.get('density', 0):.3f})")
            if len(results['context_data']['entities']) < 2:
                reasons.append(f"• Quá ít entities ({len(results['context_data']['entities'])})")
            if graph_stats.get('connected_components', 0) == len(results['context_data']['entities']):
                reasons.append("• Tất cả nodes đều riêng lẻ")
            st.info("🔍 **Lý do chuyển sang Web Search:** " + ", ".join(reasons))
            
    
    # Response content
    st.markdown(results['response'])
    
    # Display graph if available and requested
    if (results['source'] in ['GraphRAG', 'GraphRAG Knowledge Graph'] and 
        results['context_data'] and 
        options['show_graph'] and
        'entities' in results['context_data'] and 
        'relationships' in results['context_data']):
        
        entities = results['context_data']['entities']
        relationships = results['context_data']['relationships']
        
        if not entities.empty and not relationships.empty:
            st.subheader("📊 Graph Minh họa")
            
            # Graph statistics
            if results['graph_stats']:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Số nút", results['graph_stats']['nodes'])
                with col2:
                    st.metric("Số cạnh", results['graph_stats']['edges'])
                with col3:
                    st.metric("Mật độ", f"{results['graph_stats']['density']:.3f}")
                with col4:
                    st.metric("Thành phần liên thông", results['graph_stats']['connected_components'])
            
            # Create and display graph
            try:
                fig = graph_visualizer.visualize(
                    entities, 
                    relationships, 
                    f"Knowledge Graph cho: {results['query']}"
                )
                st.pyplot(fig)
                
                # Display entity details
                with st.expander("📋 Chi tiết các thực thể"):
                    for _, entity in entities.iterrows():
                        st.markdown(f"**{entity.get('title', entity.get('name', entity['id']))}**")
                        st.markdown(f"*{entity.get('description', 'Không có mô tả')}*")
                        st.markdown("---")
                        
            except Exception as e:
                st.error(f"Lỗi khi tạo graph: {e}")
        else:
            st.info("Không có dữ liệu graph để hiển thị")

def main():
    """Main app function"""
    setup_page_config()
    display_header()
    
    # Get options from sidebar
    options = display_sidebar()
    
    # Query input
    st.subheader("❓ Đặt câu hỏi về Scratch")
    
    # Example questions
    with st.expander("💡 Câu hỏi mẫu"):
        example_questions = [
            "Scratch là gì?",
            "Khối lệnh trong Scratch có những loại nào?",
            "Sprite hoạt động như thế nào?",
            "Cách tạo vòng lặp trong Scratch?",
            "Phiên bản mới nhất của Scratch tới ngày hôm nay là gì?"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(f"💭 {question}", key=f"example_{i}"):
                st.session_state.query_input = question
    
    # Query input
    query = st.text_input(
        "Nhập câu hỏi của bạn:",
        placeholder="Ví dụ: Scratch là gì? Khối lệnh có những loại nào?",
        key="query_input"
    )
    
    # Process query button
    if st.button("🔍 Tìm kiếm", type="primary") or query:
        if query.strip():
            # Process query asynchronously
            with st.spinner("Đang xử lý câu hỏi..."):
                results = asyncio.run(process_query(query, options))
            
            # Display results
            display_results(results, options)
        else:
            st.warning("Vui lòng nhập câu hỏi!")

if __name__ == "__main__":
    main()



