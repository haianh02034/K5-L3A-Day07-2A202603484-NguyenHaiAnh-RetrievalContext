# Mã nguồn Day 7 — Agentic RAG

Thư mục chứa toàn bộ mã nguồn triển khai hệ thống Agentic RAG cho Day 7:

- `retrievers.py`: Triển khai 3 phương pháp retrieval độc lập:
  1. `DenseRetriever`: Kết nối ChromaDB Day 6 (`strategy_b`) với embedding model `paraphrase-multilingual-MiniLM-L12-v2` (384-d, cosine similarity), hỗ trợ metadata filter.
  2. `LexicalRetriever`: Thuật toán TF-IDF với tokenizer tiếng Việt.
  3. `HybridRetriever`: Thuật toán kết hợp Reciprocal Rank Fusion (RRF) giữa Dense và Lexical.
- `agent.py`: `OpenRouterController` điều phối Agent thông minh với LLM `openai/gpt-4o-mini`, quản lý hội thoại đa lượt, kiểm soát tool call `search`, chèn citation `[chunk-id]`, và tự động từ chối khi thiếu bằng chứng.
- `prepare_data.py`: Trích xuất 65 chunk từ ChromaDB Day 6, nạp 20 câu hỏi có đáp án và tạo thêm 4 câu hỏi thiếu evidence (tổng cộng 24 câu benchmark).
- `benchmark_retrieval.py`: Benchmark đo lường Hit@1, Evidence Hit@3, Context Evidence Rate, thời gian và chi phí cho các cấu hình retrieval trên dev và test.
- `run_agent_eval.py`: Chạy toàn bộ Agent thật trên dev và test cho Baseline và Cấu hình đã chọn, lưu vết traces chi tiết.
- `generate_reviews.py`: Tự động audit, chấm nhãn review thủ công và tổng hợp kết quả chuẩn hóa qua `starter/agent_metrics.py`.
