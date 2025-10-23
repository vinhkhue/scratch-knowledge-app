# Scratch Knowledge Graph - Streamlit App

## 🧩 Giới thiệu

**Scratch Knowledge Graph** là ứng dụng web được xây dựng bằng Streamlit để truy vấn kiến thức về lập trình Scratch lớp 8. Ứng dụng sử dụng GraphRAG (Graph-based Retrieval Augmented Generation) kết hợp với web search để cung cấp câu trả lời chính xác và minh họa bằng graph visualization.

## ✨ Tính năng chính

- 🔍 **GraphRAG Query**: Tìm kiếm thông minh trong knowledge graph
- 🌐 **Web Search Fallback**: Tự động tìm kiếm web khi không có kết quả
- 📊 **Graph Visualization**: Hiển thị mối quan hệ giữa các khái niệm
- 🎯 **Source Attribution**: Ghi rõ nguồn câu trả lời (GraphRAG hoặc Web Search)
- ⚙️ **Customizable Options**: Tùy chỉnh loại tìm kiếm và hiển thị

## 🚀 Cài đặt và chạy local

### Yêu cầu hệ thống
- Python 3.10+
- OpenAI API Key

### Cài đặt

1. **Clone repository hoặc download source code**
```bash
cd /path/to/scratch-knowledge-app
```

2. **Cài đặt dependencies**
```bash
pip install -r requirements.txt
```

3. **Cấu hình API Key**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

4. **Chạy ứng dụng**
```bash
streamlit run app.py
```

5. **Mở trình duyệt**
Truy cập: `http://localhost:8501`

## ☁️ Deploy lên Streamlit Cloud

### Bước 1: Chuẩn bị repository
1. Tạo GitHub repository
2. Upload toàn bộ source code lên repository
3. Đảm bảo có file `requirements.txt` và `.streamlit/config.toml`

### Bước 2: Deploy trên Streamlit Cloud
1. Truy cập [Streamlit Cloud](https://share.streamlit.io/)
2. Đăng nhập bằng GitHub account
3. Click "New app"
4. Chọn repository và branch
5. Đặt tên app (ví dụ: `scratch-knowledge-graph`)

### Bước 3: Cấu hình Secrets
Trong Streamlit Cloud dashboard, thêm secrets:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

**Cách thêm secrets:**
1. Vào app dashboard trên Streamlit Cloud
2. Click "Settings" → "Secrets"
3. Thêm key-value pairs như trên

### Bước 4: Deploy
1. Click "Deploy"
2. Đợi quá trình build hoàn tất
3. Truy cập URL được cung cấp

## 📁 Cấu trúc project

```
scratch-knowledge-app/
├── app.py                      # Main Streamlit app
├── config.py                   # Configuration constants
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── .gitignore                 # Git ignore file
├── README.md                  # This file
├── utils/
│   ├── __init__.py
│   ├── graphrag_query.py      # GraphRAG query logic
│   ├── web_search.py          # OpenAI web search with function calling
│   └── graph_viz.py           # Graph visualization
└── data/
    └── scratch_index/         # GraphRAG data directory
        ├── input/             # Input text files
        ├── output/            # Generated parquet files
        └── settings.yaml      # GraphRAG configuration
```

## 🔧 Cấu hình

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key (required)

### Streamlit Configuration
File `.streamlit/config.toml` chứa:
- Theme settings
- Server configuration
- Browser settings

### GraphRAG Configuration
File `data/scratch_index/settings.yaml` chứa:
- Model settings (OpenAI)
- Input/output paths
- Workflow configuration

## 🎯 Cách sử dụng

### 1. Đặt câu hỏi
- Nhập câu hỏi về Scratch vào ô text input
- Sử dụng các câu hỏi mẫu có sẵn
- Click "Tìm kiếm" hoặc nhấn Enter

### 2. Xem kết quả
- **Nguồn GraphRAG**: Câu trả lời từ knowledge graph + graph visualization
- **Nguồn Web Search**: Câu trả lời từ web search khi GraphRAG không tìm thấy

### 3. Tùy chỉnh
- **Sidebar**: Chọn loại tìm kiếm, hiển thị graph, số nút tối đa
- **Advanced**: Độ dài tối thiểu câu trả lời, bật/tắt web search

## 📊 Graph Visualization

Khi sử dụng GraphRAG, ứng dụng sẽ hiển thị:
- **Nodes**: Các thực thể/khái niệm (màu xanh lá)
- **Edges**: Mối quan hệ giữa các thực thể (màu xám)
- **Statistics**: Số nút, số cạnh, mật độ graph
- **Details**: Chi tiết các thực thể được tìm thấy

## 🔍 Loại tìm kiếm

### Local Search
- Tìm kiếm trong các thực thể và mối quan hệ cụ thể
- Phù hợp cho câu hỏi về khái niệm cụ thể
- Nhanh và chính xác

### Global Search
- Tìm kiếm trong tất cả báo cáo cộng đồng
- Phù hợp cho câu hỏi tổng quát
- Toàn diện nhưng chậm hơn

### Auto (Mặc định)
- Thử Local Search trước
- Nếu không có kết quả tốt, chuyển sang Global Search
- Cân bằng tốt nhất giữa tốc độ và độ chính xác

## 🛠️ Troubleshooting

### Lỗi thường gặp

1. **"Module not found"**
   - Kiểm tra đã cài đặt đầy đủ dependencies: `pip install -r requirements.txt`

2. **"OpenAI API Error"**
   - Kiểm tra API key: `echo $OPENAI_API_KEY`
   - Đảm bảo có credit trong OpenAI account

3. **"GraphRAG data not found"**
   - Ứng dụng sẽ sử dụng mock data để test
   - Để có dữ liệu thật, cần chạy GraphRAG indexing

4. **"Graph visualization error"**
   - Kiểm tra matplotlib backend
   - Thử giảm số nút tối đa trong sidebar

### Performance Tips

1. **Giảm số nút graph**: Đặt max_nodes = 20-30 cho graph nhỏ hơn
2. **Tắt graph**: Uncheck "Hiển thị graph" nếu không cần
3. **Sử dụng Local Search**: Nhanh hơn Global Search

## 📝 Development

### Thêm tính năng mới
1. Tạo module trong `utils/`
2. Import và sử dụng trong `app.py`
3. Update `requirements.txt` nếu cần dependencies mới

### Debug
- Enable logging: Set `logging.basicConfig(level=logging.DEBUG)`
- Check Streamlit logs trong terminal
- Sử dụng `st.write()` để debug

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📞 Support

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra phần Troubleshooting
2. Tạo issue trên GitHub
3. Mô tả chi tiết lỗi và steps để reproduce

---

**Made with ❤️ for Scratch education**



