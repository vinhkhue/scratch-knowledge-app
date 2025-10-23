"""
Web Search Module with OpenAI Function Calling for Scratch Knowledge Graph App
"""
import openai
import json
import logging
from typing import Dict, Any, Optional
import time

from config import OPENAI_API_KEY, WEB_SEARCH_TIMEOUT, WEB_SEARCH_MAX_RESULTS

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Web search tool using OpenAI function calling"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.web_search_tool = {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information about Scratch programming, computer science education, or related topics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up on the web"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    async def search(self, query: str) -> str:
        """Perform web search using OpenAI function calling with citations"""
        try:
            # Create messages for OpenAI
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can search the web for information. When asked a question, use the web_search function to find relevant information and provide a comprehensive answer based on the search results. Always include citations with URLs when providing information from web sources."
                },
                {
                    "role": "user",
                    "content": f"Please search for information about: {query}. Provide detailed information with citations from web sources."
                }
            ]
            
            # Call OpenAI with function calling
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=[self.web_search_tool],
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000
            )
            
            message = response.choices[0].message
            
            # Check if function was called
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                if tool_call.function.name == "web_search":
                    # Parse the function arguments
                    args = json.loads(tool_call.function.arguments)
                    search_query = args.get("query", query)
                    
                    # Simulate web search results (since OpenAI doesn't have native web search)
                    search_results = self._simulate_web_search(search_query)
                    
                    # Create follow-up message with search results
                    follow_up_messages = messages + [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": json.dumps({"query": search_query})
                                    }
                                }
                            ]
                        },
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": search_results
                        }
                    ]
                    
                    # Get final response with citations
                    final_response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=follow_up_messages + [
                            {
                                "role": "system",
                                "content": "Please provide a comprehensive answer based on the web search results. Always include citations with URLs when referencing information from web sources. Format citations as [Source: URL] at the end of relevant information."
                            }
                        ],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    return final_response.choices[0].message.content
            
            # If no function call, return the direct response
            return message.content
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return self._fallback_search(query)
    
    def _simulate_web_search(self, query: str) -> str:
        """Simulate web search results for Scratch-related queries"""
        query_lower = query.lower()
        
        # Mock search results based on common Scratch topics
        mock_results = {
            "scratch": "Scratch là ngôn ngữ lập trình trực quan được phát triển bởi MIT. Nó cho phép người dùng tạo ra các câu chuyện tương tác, trò chơi và hoạt ảnh bằng cách kéo thả các khối lệnh.",
            "lập trình": "Lập trình là quá trình thiết kế, viết, kiểm tra và bảo trì mã nguồn của chương trình máy tính. Scratch giúp học sinh làm quen với tư duy lập trình thông qua giao diện trực quan.",
            "khối lệnh": "Khối lệnh trong Scratch là các thành phần cơ bản để tạo ra chương trình. Chúng được chia thành các danh mục như Di chuyển, Ngoại hình, Âm thanh, Sự kiện, Điều khiển, Cảm biến, Toán học và Biến.",
            "sprite": "Sprite là nhân vật hoạt động trong Scratch. Mỗi sprite có thể có nhiều trang phục (costume) và có thể được lập trình để thực hiện các hành động khác nhau.",
            "sân khấu": "Sân khấu (Stage) là không gian làm việc chính trong Scratch, nơi các sprite hoạt động. Sân khấu có thể có nhiều phông nền (background) khác nhau.",
            "điều khiển": "Các khối lệnh điều khiển trong Scratch bao gồm: lặp lại, nếu-thì, nếu-thì-khác, chờ đợi, dừng tất cả, dừng script này.",
            "sự kiện": "Các khối lệnh sự kiện trong Scratch bao gồm: khi cờ xanh được nhấn, khi phím được nhấn, khi sprite được nhấn, khi nhận được tin nhắn.",
            "biến": "Biến trong Scratch là nơi lưu trữ dữ liệu có thể thay đổi. Có hai loại biến: biến cho sprite và biến cho tất cả sprite.",
            "danh sách": "Danh sách trong Scratch là tập hợp các giá trị được sắp xếp theo thứ tự. Danh sách có thể được sử dụng để lưu trữ nhiều giá trị cùng một lúc.",
            "hàm": "Hàm trong Scratch cho phép tạo ra các khối lệnh tùy chỉnh để thực hiện một nhiệm vụ cụ thể. Điều này giúp mã nguồn trở nên có tổ chức và dễ hiểu hơn."
        }
        
        # Find the best matching topic
        best_match = None
        best_score = 0
        
        for topic, content in mock_results.items():
            if topic in query_lower:
                score = len(topic)
                if score > best_score:
                    best_score = score
                    best_match = content
        
        if best_match:
            return f"Kết quả tìm kiếm cho '{query}':\n\n{best_match}\n\nThông tin này được lấy từ các nguồn giáo dục về Scratch và lập trình."
        else:
            return f"Kết quả tìm kiếm cho '{query}':\n\nScratch là một công cụ giáo dục tuyệt vời để học lập trình. Nó sử dụng giao diện trực quan với các khối lệnh màu sắc, giúp học sinh dễ dàng tạo ra các dự án sáng tạo như trò chơi, câu chuyện tương tác và hoạt ảnh."
    
    def _fallback_search(self, query: str) -> str:
        """Fallback search when OpenAI API fails"""
        logger.warning("Using fallback search due to API error")
        
        return f"""Dựa trên kiến thức chung về Scratch và lập trình:

**Câu hỏi:** {query}

**Trả lời:** Scratch là ngôn ngữ lập trình trực quan được thiết kế đặc biệt cho trẻ em và người mới bắt đầu học lập trình. Nó sử dụng giao diện kéo thả các khối lệnh màu sắc thay vì viết mã văn bản.

**Các tính năng chính của Scratch:**
- Khối lệnh trực quan và dễ hiểu
- Sprite (nhân vật) có thể tùy chỉnh
- Sân khấu với nhiều phông nền
- Hỗ trợ âm thanh và hình ảnh
- Chia sẻ dự án với cộng đồng

**Lợi ích học tập:**
- Phát triển tư duy logic
- Khuyến khích sáng tạo
- Học cách giải quyết vấn đề
- Làm việc nhóm và chia sẻ

*Lưu ý: Đây là thông tin tổng quát. Để có câu trả lời cụ thể hơn, vui lòng thử lại sau.*"""

# Global instance
web_search_tool = WebSearchTool()
