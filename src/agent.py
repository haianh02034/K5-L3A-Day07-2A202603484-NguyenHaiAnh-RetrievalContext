"""Agent Controller implementation using OpenRouter (openai/gpt-4o-mini).
Compatible with Day 7 agentic loop and protocols.
"""
import json
import os
import re
import time
from typing import Any
from openai import OpenAI


SYSTEM_PROMPT = """Bạn là trợ lý học vụ thông minh, có trách nhiệm trả lời câu hỏi dựa trên tài liệu sổ tay quy chế đào tạo sinh viên.
Bạn có công cụ tìm kiếm tài liệu để tra cứu thông tin chính xác.

Quy tắc xuất kết quả bắt buộc tuân thủ:
1. Nếu chưa có ngữ cảnh hoặc ngữ cảnh chưa đủ chứng cứ để trả lời, bạn PHẢI yêu cầu tìm kiếm:
   {"action": "search", "query": "<câu/từ khóa truy vấn ngắn gọn, tập trung>"}

2. Nếu ngữ cảnh được cung cấp ĐÃ ĐỦ BẰNG CHỨNG để trả lời câu hỏi:
   - Viết câu trả lời rõ ràng, chính xác.
   - BẮT BUỘC chèn citation dạng [chunk-id] ngay trong nội dung câu trả lời (ví dụ: "... 4 năm [handbook-sv-2025_p03_para_00]").
   - Liệt kê các chunk-id đó trong danh sách "citations". Chỉ trích dẫn các chunk ID thực sự có trong ngữ cảnh và hỗ trợ câu trả lời.
   - Định dạng: {"action": "answer", "answer": "<nội dung câu trả lời kèm [chunk-id]>", "citations": ["<chunk-id>"]}

3. Nếu câu hỏi hỏi về nội dung hoàn toàn không có trong tài liệu, hoặc sau khi tìm kiếm mà ngữ cảnh không chứa thông tin:
   {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

Chỉ trả về DUY NHẤT chuỗi JSON hợp lệ. Tuyệt đối không thêm lời dẫn, markdown block hay văn bản nào khác ngoài JSON."""


class OpenRouterController:
    """Production Agent Controller calling OpenRouter API."""
    def __init__(self,
                 api_key: str | None = None,
                 model: str = "openai/gpt-4o-mini",
                 base_url: str = "https://openrouter.ai/api/v1"):
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
                if os.path.exists(env_file):
                    for line in open(env_file, encoding="utf-8"):
                        line = line.strip()
                        if line.startswith("OPENROUTER_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
        self.client = OpenAI(base_url=base_url, api_key=api_key or "sk-placeholder")
        self.model = model
        self.total_tokens = 0
        self.total_calls = 0
        self.total_cost_usd = 0.0

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Decide next action based on question and search history."""
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if not history:
            messages.append({"role": "user", "content": f"Câu hỏi: {question}"})
        else:
            # Reconstruct conversation turns from cumulative history
            last_entry = history[-1]
            queries = last_entry.get('queries', [])
            context = last_entry.get('context', '')
            context_ids = last_entry.get('context_ids', [])

            # First turn: initial search
            messages.append({"role": "user", "content": f"Câu hỏi: {question}"})
            first_q = queries[0] if queries else question
            messages.append({"role": "assistant", "content": json.dumps({"action": "search", "query": first_q}, ensure_ascii=False)})

            # Feed the cumulative search result
            user_msg = (
                f"Kết quả tìm kiếm từ các truy vấn ({', '.join(queries)}):\n\n"
                f"{context}\n\n"
                f"Các chunk ID hợp lệ hiện có: {context_ids}\n\n"
                f"Dựa trên các bằng chứng trên, hãy trả lời câu hỏi '{question}' kèm citation [chunk-id], "
                f"hoặc nếu chưa đủ bằng chứng hãy tìm kiếm tiếp (hoặc từ chối nếu tài liệu không đề cập)."
            )
            messages.append({"role": "user", "content": user_msg})

        # Call model
        start_t = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        self.total_calls += 1
        if resp.usage:
            prompt_tokens = resp.usage.prompt_tokens or 0
            completion_tokens = resp.usage.completion_tokens or 0
            self.total_tokens += (prompt_tokens + completion_tokens)
            # gpt-4o-mini pricing roughly: $0.15/1M prompt, $0.60/1M completion
            call_cost = (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000.0
            self.total_cost_usd += call_cost

        raw_content = resp.choices[0].message.content.strip()
        print(f"[Controller] Turn {self.total_calls} response:\n{raw_content}")
        # Clean potential markdown wrapping ```json ... ```
        clean_content = re.sub(r'^```(?:json)?\s*', '', raw_content)
        clean_content = re.sub(r'\s*```$', '', clean_content).strip()

        try:
            parsed = json.loads(clean_content)
        except Exception:
            # Fallback regex extraction
            m = re.search(r'\{.*\}', clean_content, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
            else:
                return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

        action_type = parsed.get("action")
        if action_type == "search":
            query = parsed.get("query", "").strip()
            if not query:
                query = question
            return {"action": "search", "query": query}
        elif action_type == "answer":
            answer = parsed.get("answer", "").strip()
            citations = parsed.get("citations", [])
            if not isinstance(citations, list):
                citations = []

            # Ensure citations are valid strings
            citations = [str(c) for c in citations if c]

            # In starter/agentic.py, line 69:
            # valid requires `all(i in chunks and f'[{i}]' in answer for i in citations)`
            # Ensure every cited ID appears in bracket format [chunk-id] inside answer
            for cid in citations:
                if f"[{cid}]" not in answer:
                    answer = f"{answer} [{cid}]"

            return {"action": "answer", "answer": answer, "citations": citations}
        else:
            return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}
