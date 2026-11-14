# Báo cáo Day 7 — Agentic RAG

- **Học viên:** Nguyễn Hải Anh
- **MSSV:** 2A202603484
- **Lớp:** L3A
- **Chủ đề & Corpus:** Sổ tay Sinh viên và Quy chế Đào tạo Đại học Chính quy năm 2025 (`handbook-sv-2025.pdf`)

---

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Nguồn tài liệu: Sổ tay Sinh viên và Quy chế Đào tạo Đại học Chính quy năm 2025 (`handbook-sv-2025.pdf`), tài liệu quy chế nội bộ lưu hành nội bộ phục vụ đào tạo tại VinUni.
  - Số trang: 35 trang tài liệu vật lý (32 trang nội dung hữu ích, loại bỏ bìa và mục lục).
  - Số chunk: 65 paragraph chunks kế thừa từ Day 6 (strategy B, phân đoạn theo đoạn văn có ý nghĩa hoàn chỉnh, độ dài tối đa 800 ký tự/chunk). Mỗi chunk bảo lưu toàn bộ siêu dữ liệu: `document_id`, `source`, `page`, `section`, `chapter`, `chunk_id`.

- **Vector store, embedding model, phiên bản và số chiều:**
  - Vector store: ChromaDB PersistentClient (`chroma_db/strategy_b`), độ đo khoảng cách Cosine distance (`hnsw:space: cosine`).
  - Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều, chuẩn hóa L2 `normalize_embeddings=True`).
  - Chi phí embedding: $0.00 (chạy local trên CPU/GPU).

- **Agent model, cách chạy và phân chia dev/test:**
  - Agent Controller: `openai/gpt-4o-mini` qua OpenRouter API.
  - Kiến trúc Agent: Vòng lặp Agentic đa lượt (multi-turn loop) tuân thủ giao thức `run_agent`: controller quyết định gọi công cụ tìm kiếm (`search`) hoặc trả lời kèm trích dẫn (`answer`). Tự động kiểm tra tính hợp lệ của citation ID và thực hiện từ chối trả lời (abstain) khi thiếu bằng chứng.
  - Phân chia tập benchmark (24 câu hỏi):
    - **Dev (17 câu):** 15 câu có đáp án trong tài liệu + 2 câu thiếu bằng chứng (`u01`, `u02`).
    - **Test (7 câu):** 5 câu có đáp án trong tài liệu + 2 câu thiếu bằng chứng (`u03`, `u04`).

- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có đáp án: Evidence chuẩn được kiểm chứng đối chiếu trực tiếp trên trang và đoạn trích văn bản gốc. Tiêu chí chấm:
    - **0 điểm:** Trả lời sai, trích dẫn sai bằng chứng, hoặc từ chối nhầm (false abstention).
    - **1 điểm:** Trả lời đúng một phần hoặc trích dẫn chưa đầy đủ điều kiện ngoại lệ.
    - **2 điểm:** Trả lời đúng, đầy đủ thông tin định lượng/điều kiện và có trích dẫn citation hợp lệ hỗ trợ hoàn toàn câu trả lời.
  - Câu thiếu evidence: Chọn các câu hỏi gần gũi với đời sống sinh viên (phí gửi xe ký túc xá, thủ tục tạm trú công an phường, bảo lưu nghĩa vụ quân sự, nuôi thú cưng trong KTX) nhưng hoàn toàn không được quy định trong sổ tay quy chế đào tạo. Tiêu chí chấm: Agent phải nhận biết sự thiếu hụt thông tin và chủ động từ chối (abstain), không được suy diễn bịa đặt thông tin.

---

## Retrieval và thử nghiệm

### Bảng kết quả thực nghiệm Retrieval

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline** | Dense retrieval (Day 6 ChromaDB MiniLM-L12-v2), top-k=3, budget=800 | Baseline Day 6: Cosine vector similarity trên 65 paragraph chunks | dev | 8/15 (53.3%) | 10/15 (66.7%) | 4/15 (26.7%) | 18.99 ms / $0.00 |
| **Phương pháp 2** | Lexical retrieval (TF-IDF), top-k=3, budget=800 | Khớp chính xác các từ khóa học vụ cụ thể (GPA, BHYT, hoãn thi, tín chỉ) | dev | 15/15 (100.0%) | 15/15 (100.0%) | 9/15 (60.0%) | 0.26 ms / $0.00 |
| **Phương pháp 3** | Hybrid retrieval (RRF Dense + Lexical), top-k=3, candidate_k=10, rrf_k=60, budget=800 | Kết hợp semantic embedding và keyword matching giảm thiểu điểm mù của từng modality | dev | 12/15 (80.0%) | 12/15 (80.0%) | 6/15 (40.0%) | 18.98 ms / $0.00 |
| **Cải thiện 1** | Hybrid retrieval (RRF Dense + Lexical), top-k=5, candidate_k=15, rrf_k=60, budget=800 | Mở rộng top candidate lên 5 để thu hồi các chunk bị đẩy về rank 4-5 do RRF | dev | 13/15 (86.7%) | 14/15 (93.3%) | 7/15 (46.7%) | 21.42 ms / $0.00 |
| **Cải thiện 2** | Hybrid retrieval (RRF Dense + Lexical), top-k=5, candidate_k=15, rrf_k=60, budget=2500 | Nới rộng context budget lên 2500 ký tự giúp đóng gói trọn vẹn 3-4 chunks liên quan vào prompt | dev | 13/15 (86.7%) | 14/15 (93.3%) | 14/15 (93.3%) | 19.83 ms / $0.00 |
| **Cấu hình đã chọn** | Hybrid retrieval (RRF Dense + Lexical), top-k=5, candidate_k=15, rrf_k=60, budget=2500 | Đánh giá độc lập cấu hình tối ưu chốt từ dev trên tập test | test | 2/5 (40.0%) | 2/5 (40.0%) | 2/5 (40.0%) | 22.15 ms / $0.00 |
| **Baseline** | Dense retrieval (Day 6 ChromaDB MiniLM-L12-v2), top-k=3, budget=800 | So sánh trực tiếp baseline Day 6 trên tập test | test | 2/5 (40.0%) | 2/5 (40.0%) | 1/5 (20.0%) | 18.45 ms / $0.00 |

### Phân tích lỗi dev, nguyên nhân và lý do chọn cấu hình

1. **Lỗi dev 1 — Câu `q01` (Thời gian đào tạo tiêu chuẩn cho cử nhân và kỹ sư):**
   - *Hiện tượng ở Baseline:* Agent trả về `abstain` (từ chối nhầm). Dense retrieval trả về chunk `handbook-sv-2025_p10_para_00` ở rank 1 vì đoạn này nói về "Điều kiện tốt nghiệp: tích lũy 125 tín chỉ cử nhân và 155 tín chỉ kỹ sư", chứa nhiều vector từ ngữ trùng với "cử nhân", "kỹ sư". Chunk chứa gold evidence thực sự nằm ở Điều 1 trang 3 (`handbook-sv-2025_p03_para_00`). Do baseline chỉ cho phép budget 800 ký tự, khi chunk trang 10 đã chiếm ~850 ký tự, chunk trang 3 bị cắt bỏ hoàn toàn khỏi context. Agent không nhận được bằng chứng nên phải từ chối.
   - *Khắc phục:* Chuyển sang Hybrid RRF giúp từ khóa chính xác "tiêu chuẩn" và "thời gian đào tạo" từ TF-IDF kéo chunk `p03` lên rank cao. Đồng thời nới context budget lên 2500 ký tự cho phép đóng gói đồng thời cả 3 chunk hàng đầu. Kết quả: Agent nhận được chunk `p03`, trả lời chính xác cử nhân 4 năm, kỹ sư 4.5 năm kèm trích dẫn `[handbook-sv-2025_p03_para_00]`.

2. **Lỗi dev 2 — Câu `q06` (Tình huống học vụ bị buộc thôi học):**
   - *Hiện tượng ở Baseline:* Baseline Dense top-k=3 chỉ lấy được 1 đoạn về cảnh báo học vụ chung chung, bỏ sót các quy định xử lý kỷ luật nghiêm trọng ở Điều 19 và Điều 9. Context 800 ký tự nghẽn khiến agent không thể tổng hợp đủ chứng cứ.
   - *Khắc phục:* Cấu hình Cải thiện 2 (Hybrid RRF top_k=5 + Budget 2500) truy xuất và đóng gói được cả `handbook-sv-2025_p08_para_00` (buộc thôi học do điểm CPA/GPA không đạt liên tiếp) và `handbook-sv-2025_p15_para_01` (buộc thôi học do thi hộ). Agent tổng hợp chính xác cả hai khía cạnh, đạt điểm 2/2 tối đa.

- **Tiêu chí chốt cấu hình:** Chọn **Hybrid RRF (top_k=5, budget=2500)** vì nâng **Context Evidence Rate từ 26.7% lên 93.3%** trên dev, giải quyết triệt để vấn đề nghẽn context khi chunk văn bản tiếng Việt có độ dài ~800 ký tự, trong khi độ trễ tăng không đáng kể (~19.8 ms so với 18.9 ms).

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Công cụ: Tool `search(query, filter=None)` kết nối trực tiếp với retriever thật.
  - Số lượt gọi tối đa (`max_searches`): 4 lượt. Điều kiện dừng: Khi Controller đưa ra hành động `answer` hoặc khi chạm ngưỡng `max_searches` (tự động chuyển trạng thái `abstain`).
  - Chọn context: Tích lũy deduplication theo `chunk_id`, đóng gói theo thứ tự rank cho đến khi đạt ngưỡng `budget=2500` ký tự.
  - Xử lý lỗi: Khung `run_agent` truyền thẳng ngoại lệ mạng/database thay vì silent mock; Controller có fallback parse JSON và tự động chuẩn hóa citation dạng `[chunk-id]` trong văn bản câu trả lời.

### Bảng tổng hợp đánh giá Agent

| Cấu hình / split | Tool use | Số lượt gọi TB | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| **Baseline (Dense, budget 800) / dev** | 17/17 (100%) | 2.1 | 7 / 1 / 7 (TB: 1.0/2) | 8/8 (100%) | 8/8 (100%) | 2/2 (100%) | 7/15 (46.7%) |
| **Cấu hình đã chọn (Hybrid, budget 2500) / dev** | 17/17 (100%) | 2.0 | 1 / 1 / 13 (TB: 1.8/2) | 16/16 (100%) | 14/14 (100%) | 2/2 (100%) | 1/15 (6.7%) |
| **Baseline (Dense, budget 800) / test** | 7/7 (100%) | 2.0 | 3 / 1 / 1 (TB: 0.6/2) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 3/5 (60.0%) |
| **Cấu hình đã chọn (Hybrid, budget 2500) / test** | 7/7 (100%) | 2.0 | 1 / 1 / 3 (TB: 1.4/2) | 5/6 (83.3%) | 5/5 (100%) | 1/2 (50.0%) | 1/5 (20.0%) |

### Chi tiết 7 câu test độc lập (5 có đáp án + 2 thiếu evidence)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| **q16** | Có đáp án | `search('hạ bậc xếp hạng bằng tốt nghiệp đại học quy định vi phạm')` -> Trả về `p10_para_01`, `p08_para_01` | "Hạng tốt nghiệp của sinh viên bị hạ một bậc nếu vi phạm một trong các điều sau: Khối lượng các học phần phải học lại vượt quá 5% tổng số tín chỉ của chương trình; hoặc đã bị kỷ luật từ mức cảnh cáo cấp trường trở lên trong thời gian học tập [handbook-sv-2025_p10_para_01]." | **2 điểm**: Trả lời chính xác 100% cả 2 điều kiện hạ bậc bằng tốt nghiệp theo đúng gold evidence trang 10. |
| **q17** | Có đáp án | `search('chứng chỉ nghề nghiệp quốc tế miễn học quy đổi điểm A môn học thực hành mạng và đám mây')` -> Trả về `p18_para_00` | "Các chứng chỉ nghề nghiệp quốc tế được xem xét miễn học và quy đổi tương đương điểm A (4.0) cho các học phần thực hành mạng và điện toán đám mây bao gồm: AWS Certified Cloud Practitioner, Cisco CCNA, và CompTIA Security+ [handbook-sv-2025_p18_para_00]." | **2 điểm**: Trả lời chính xác toàn bộ 3 chứng chỉ nghề nghiệp quốc tế và mức quy đổi điểm A theo Điều 25 trang 18. |
| **q18** | Có đáp án | `search('khung giờ mở cửa và đóng cửa Ký túc xá')` -> Không xuất hiện chunk trang 24 trong top-5 | "Không đủ bằng chứng để trả lời." | **0 điểm (Từ chối nhầm)**: Do câu hỏi diễn đạt lại không trùng từ khóa của chunk trang 24 (quy định nội trú KTX), retrieval không đưa được chunk vào context nên agent từ chối trả lời đúng nguyên tắc an toàn. |
| **q19** | Có đáp án | `search('miễn làm khóa luận tốt nghiệp quy đổi điểm A sinh viên nghiên cứu khoa học')` -> Trả về `p27_para_00`, `p18_para_00` | "Để được xem xét miễn làm khóa luận tốt nghiệp, sinh viên cần tham gia Nghiên cứu Khoa học (NCKH) và có điểm CPA tích lũy từ 2.50 trở lên cùng với điểm rèn luyện từ loại Khá trở lên [handbook-sv-2025_p27_para_00][handbook-sv-2025_p18_para_00]." | **1 điểm**: Trả lời đúng một phần điều kiện NCKH và CPA, nhưng thiếu điều kiện giải thưởng cấp trường trở lên do câu hỏi đòi hỏi thông tin chi tiết giải thưởng NCKH. |
| **q20** | Có đáp án | `search('mức thưởng và hỗ trợ kinh phí APC bài báo ISI Scopus nhóm Q1')` -> Trả về `p33_para_01` | "Mức thưởng cho bài báo đăng trên tạp chí quốc tế ISI/Scopus nhóm Q1 là 20.000.000 VNĐ và hỗ trợ 100% lệ phí APC, tối đa không quá 35.000.000 VNĐ [handbook-sv-2025_p33_para_01]." | **2 điểm**: Trả lời chính xác số tiền thưởng 20 triệu VNĐ và mức hỗ trợ 100% APC tối đa 35 triệu VNĐ theo Điều 36 trang 33. |
| **u03** | Thiếu evidence | `search('bảo lưu kết quả học tập nghĩa vụ quân sự gia hạn tối đa bao nhiêu năm')` -> Trả về `p09_para_00` | "Sinh viên được quyền bảo lưu kết quả học tập trong thời gian phục vụ quân ngũ... Tuy nhiên, tài liệu không nêu rõ thời gian tối đa cho việc bảo lưu kết quả học tập này. Do đó, không đủ bằng chứng để trả lời cụ thể về thời gian tối đa... [handbook-sv-2025_p09_para_00]." | **Abstention = False (Nhận xét)**: Mặc dù câu trả lời khẳng định "không đủ bằng chứng để trả lời cụ thể", việc gán citation ID vào câu trả lời khiến framework chấm trạng thái answered. Đây là bài học về prompt engineering để agent từ chối bằng mảng citation rỗng. |
| **u04** | Thiếu evidence | `search('nuôi thú cưng trong ký túc xá quy định xử phạt')` -> Không tìm thấy thông tin | "Không đủ bằng chứng để trả lời." | **Abstention = True**: Agent từ chối trả lời hoàn toàn chính xác khi gặp câu hỏi về quy định cấm nuôi chó/mèo trong KTX không có trong tài liệu. |

---

## Tổng kết & Tái lập

- **Đường dẫn dữ liệu và trace:**
  - Dữ liệu: [data/chunks.json](file:///e:/AI_Vin/K5-L3-Day07-2A202603484-NguyenHaiAnh-RetrievalContext/data/chunks.json), [data/questions.json](file:///e:/AI_Vin/K5-L3-Day07-2A202603484-NguyenHaiAnh-RetrievalContext/data/questions.json), [data/cases.json](file:///e:/AI_Vin/K5-L3-Day07-2A202603484-NguyenHaiAnh-RetrievalContext/data/cases.json)
  - Kết quả Retrieval: [results/retrieval/retrieval_comparison.json](file:///e:/AI_Vin/K5-L3-Day07-2A202603484-NguyenHaiAnh-RetrievalContext/results/retrieval/retrieval_comparison.json)
  - Trace Agent:
    - Baseline dev: [results/agent/baseline_dev/traces.json](file:///e:/AI_Vin/K5-L3-Day07-2A202603484-NguyenHaiAnh-RetrievalContext/results/agent/baseline_dev/traces.json), [summary.json](file:///e:/AI_Vin/K5-L3-Day07-2A202603484-NguyenHaiAnh-RetrievalContext/results/agent/baseline_dev/summary.json)
    - Cấu hình chọn dev: [results/agent/chosen_dev/traces.json](file:///e:/AI_Vin/K5-L3-Day07-2A202603484-NguyenHaiAnh-RetrievalContext/results/agent/chosen_dev/traces.json), [summary.json](file:///e:/AI_Vin/K5-L3-Day07-2A202603484-NguyenHaiAnh-RetrievalContext/results/agent/chosen_dev/summary.json)
    - Baseline test: [results/agent/baseline_test/traces.json](file:///e:/AI_Vin/K5-L3-Day07-2A202603484-NguyenHaiAnh-RetrievalContext/results/agent/baseline_test/traces.json), [summary.json](file:///e:/AI_Vin/K5-L3-Day07-2A202603484-NguyenHaiAnh-RetrievalContext/results/agent/baseline_test/summary.json)
    - Cấu hình chọn test: [results/agent/chosen_test/traces.json](file:///e:/AI_Vin/K5-L3-Day07-2A202603484-NguyenHaiAnh-RetrievalContext/results/agent/chosen_test/traces.json), [summary.json](file:///e:/AI_Vin/K5-L3-Day07-2A202603484-NguyenHaiAnh-RetrievalContext/results/agent/chosen_test/summary.json)

- **Lệnh tái lập thí nghiệm:**
  ```bash
  # 1. Chạy unit tests kiểm tra code
  python -m unittest discover -s tests -v

  # 2. Chuẩn bị dữ liệu và câu hỏi benchmark từ Day 6
  python src/prepare_data.py

  # 3. Chạy benchmark so sánh 3 phương pháp retrieval và các cấu hình cải thiện
  python src/benchmark_retrieval.py

  # 4. Chạy Agent thật trên các split dev và test
  python src/run_agent_eval.py --split all --config all

  # 5. Kiểm định review thủ công và tổng hợp metrics chuẩn
  python src/generate_reviews.py
  ```

- **Đánh đổi và Giới hạn:**
  1. *Đánh đổi giữa Context Budget và Chi phí/Độ trễ:* Tăng budget từ 800 lên 2500 ký tự giúp tỷ lệ Context Evidence Rate tăng vọt từ 26.7% lên 93.3% trên dev, loại bỏ hầu hết các lỗi false abstention. Chi phí token tăng nhẹ (~0.0003$ mỗi câu hỏi) nhưng hoàn toàn xứng đáng với độ chính xác học vụ cần thiết.
  2. *Giới hạn đối với các câu hỏi diễn đạt lại (Paraphrase):* Khi câu hỏi dùng từ ngữ sinh hoạt thông thường (như "giờ mở cửa đóng cửa KTX") mà văn bản sử dụng từ ngữ pháp lý học thuật ("thời gian biểu nội trú"), cả BM25 lẫn embedding MiniLM-L12-v2 đều có thể gặp khó khăn nếu không có bước Query Transformation/Rewriting chuyên biệt.
  3. *Hiện tượng giải thích kèm trích dẫn trên câu hỏi thiếu evidence:* Khi câu hỏi thiếu thông tin cụ thể (ví dụ số năm tối đa bảo lưu quân sự), model có xu hướng tìm thấy điều khoản liên quan gần nhất và trích dẫn giải thích rằng văn bản không nói rõ thời hạn, thay vì từ chối thuần túy với danh sách citation rỗng. Cần tinh chỉnh prompt phân biệt rõ "không đủ bằng chứng thì dứt khoát không gán citation".
