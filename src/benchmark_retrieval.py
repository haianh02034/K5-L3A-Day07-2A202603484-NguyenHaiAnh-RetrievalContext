"""Benchmark retrieval methods and configurations on dev and test splits.
Evaluates:
- Hit@1
- Evidence Hit@3
- Context Evidence Rate
- Average context characters
- Latency (median ms)
- Service cost (USD)
"""
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'starter'))

from baseline import hit, pack_context
from src.retrievers import DenseRetriever, LexicalRetriever, HybridRetriever


def evaluate_retriever(retriever_fn, questions: list[dict[str, Any]], top_k: int = 3, budget: int = 800) -> dict[str, Any]:
    traces = []
    times = []
    for q in questions:
        start_t = time.perf_counter()
        results = retriever_fn(q['question'], filters=q.get('filter'), top_k=top_k)
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - start_t) * 1000
        times.append(elapsed_ms)

        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]
        h1 = bool(marked and marked[0]['hit'])
        h3 = any(c['hit'] for c in marked[:3])
        ctx_hit = any(c['hit'] and c['id'] in ids for c in marked)

        traces.append({
            'query_id': q['id'],
            'question': q['question'],
            'top_k': top_k,
            'budget': budget,
            'hit1': h1,
            'hit3': h3,
            'context_hit': ctx_hit,
            'context_ids': ids,
            'context_chars': len(context),
            'elapsed_ms': elapsed_ms,
            'results': marked
        })

    n = len(questions)
    hit1_n = sum(t['hit1'] for t in traces)
    hit3_n = sum(t['hit3'] for t in traces)
    ctx_n = sum(t['context_hit'] for t in traces)

    return {
        'n': n,
        'hit1_n': hit1_n,
        'hit1_pct': round(hit1_n / n * 100, 1) if n else 0.0,
        'hit3_n': hit3_n,
        'hit3_pct': round(hit3_n / n * 100, 1) if n else 0.0,
        'context_hit_n': ctx_n,
        'context_hit_pct': round(ctx_n / n * 100, 1) if n else 0.0,
        'avg_context_chars': round(statistics.mean(t['context_chars'] for t in traces), 1) if traces else 0,
        'median_ms': round(statistics.median(times), 2) if times else 0,
        'cost_usd': 0.0,
        'traces': traces
    }


def main():
    root = Path(__file__).resolve().parent.parent
    chunks_path = root / 'data' / 'chunks.json'
    questions_path = root / 'data' / 'questions.json'
    results_dir = root / 'results' / 'retrieval'
    results_dir.mkdir(parents=True, exist_ok=True)

    chunks = json.load(open(chunks_path, encoding='utf-8'))
    all_questions = json.load(open(questions_path, encoding='utf-8'))
    dev_questions = [q for q in all_questions if q['split'] == 'dev']
    test_questions = [q for q in all_questions if q['split'] == 'test']

    print(f"Loaded {len(chunks)} chunks, {len(dev_questions)} dev questions, {len(test_questions)} test questions.")

    dense = DenseRetriever()
    lexical = LexicalRetriever(chunks)
    hybrid = HybridRetriever(dense, lexical)

    benchmarks = [
        # 1. Baseline on dev: Dense Retrieval (Day 6 vector store), top-k=3, budget=800
        {
            'name': 'Baseline (Dense)',
            'method': 'Dense retrieval (Day 6 ChromaDB MiniLM-L12-v2)',
            'config': 'top_k=3, budget=800, no query rewrite',
            'hypothesis': 'Baseline Day 6: Cosine vector similarity trên 65 paragraph chunks',
            'split': 'dev',
            'questions': dev_questions,
            'top_k': 3,
            'budget': 800,
            'fn': lambda q, filters, top_k: dense.search(q, filters=filters, top_k=top_k)
        },
        # 2. Method 2 on dev: Lexical Retrieval (TF-IDF), top-k=3, budget=800
        {
            'name': 'Phương pháp 2 (Lexical TF-IDF)',
            'method': 'Lexical retrieval (TF-IDF)',
            'config': 'top_k=3, budget=800',
            'hypothesis': 'Khớp chính xác các từ khóa học vụ cụ thể (GPA, BHYT, hoãn thi, tín chỉ)',
            'split': 'dev',
            'questions': dev_questions,
            'top_k': 3,
            'budget': 800,
            'fn': lambda q, filters, top_k: lexical.search(q, filters=filters, top_k=top_k)
        },
        # 3. Method 3 on dev: Hybrid Retrieval (RRF), top-k=3, budget=800
        {
            'name': 'Phương pháp 3 (Hybrid RRF)',
            'method': 'Hybrid retrieval (RRF Dense + Lexical)',
            'config': 'top_k=3, candidate_k=10, rrf_k=60, budget=800',
            'hypothesis': 'Kết hợp semantic embedding và keyword matching giảm thiểu điểm mù của từng modality',
            'split': 'dev',
            'questions': dev_questions,
            'top_k': 3,
            'budget': 800,
            'fn': lambda q, filters, top_k: hybrid.search(q, filters=filters, top_k=top_k, candidate_k=10)
        },
        # 4. Improvement 1 on dev: Hybrid RRF with top_k=5, budget=800
        {
            'name': 'Cải thiện 1 (Hybrid RRF top_k=5)',
            'method': 'Hybrid retrieval (RRF Dense + Lexical)',
            'config': 'top_k=5, candidate_k=15, rrf_k=60, budget=800',
            'hypothesis': 'Mở rộng top candidate lên 5 để thu hồi các chunk bị đẩy về rank 4-5 do RRF',
            'split': 'dev',
            'questions': dev_questions,
            'top_k': 5,
            'budget': 800,
            'fn': lambda q, filters, top_k: hybrid.search(q, filters=filters, top_k=top_k, candidate_k=15)
        },
        # 5. Improvement 2 on dev: Hybrid RRF with top_k=5, budget=2500
        {
            'name': 'Cải thiện 2 (Hybrid RRF top_k=5 + Budget=2500)',
            'method': 'Hybrid retrieval (RRF Dense + Lexical)',
            'config': 'top_k=5, candidate_k=15, rrf_k=60, budget=2500',
            'hypothesis': 'Nới rộng context budget lên 2500 ký tự giúp đóng gói trọn vẹn 3-4 chunks liên quan vào prompt',
            'split': 'dev',
            'questions': dev_questions,
            'top_k': 5,
            'budget': 2500,
            'fn': lambda q, filters, top_k: hybrid.search(q, filters=filters, top_k=top_k, candidate_k=15)
        },
        # 6. Chosen Configuration on test: Hybrid RRF top_k=5, budget=2500
        {
            'name': 'Cấu hình đã chọn (Hybrid RRF top_k=5 + Budget=2500)',
            'method': 'Hybrid retrieval (RRF Dense + Lexical)',
            'config': 'top_k=5, candidate_k=15, rrf_k=60, budget=2500',
            'hypothesis': 'Đánh giá độc lập cấu hình tối ưu chốt từ dev trên tập test',
            'split': 'test',
            'questions': test_questions,
            'top_k': 5,
            'budget': 2500,
            'fn': lambda q, filters, top_k: hybrid.search(q, filters=filters, top_k=top_k, candidate_k=15)
        },
        # 7. Baseline on test: Dense Retrieval, top-k=3, budget=800
        {
            'name': 'Baseline (Dense) trên test',
            'method': 'Dense retrieval (Day 6 ChromaDB MiniLM-L12-v2)',
            'config': 'top_k=3, budget=800',
            'hypothesis': 'So sánh trực tiếp baseline Day 6 trên tập test',
            'split': 'test',
            'questions': test_questions,
            'top_k': 3,
            'budget': 800,
            'fn': lambda q, filters, top_k: dense.search(q, filters=filters, top_k=top_k)
        }
    ]

    all_results = []
    print("\n" + "="*80)
    print(f"{'Cấu hình':<30} | {'Split':<5} | {'Hit@1':<10} | {'Hit@3':<10} | {'Ctx Evidence':<14} | {'Latency':<8}")
    print("="*80)

    for b in benchmarks:
        eval_res = evaluate_retriever(b['fn'], b['questions'], top_k=b['top_k'], budget=b['budget'])
        h1_str = f"{eval_res['hit1_n']}/{eval_res['n']} ({eval_res['hit1_pct']}%)"
        h3_str = f"{eval_res['hit3_n']}/{eval_res['n']} ({eval_res['hit3_pct']}%)"
        ctx_str = f"{eval_res['context_hit_n']}/{eval_res['n']} ({eval_res['context_hit_pct']}%)"
        lat_str = f"{eval_res['median_ms']} ms"

        print(f"{b['name']:<30} | {b['split']:<5} | {h1_str:<10} | {h3_str:<10} | {ctx_str:<14} | {lat_str:<8}")

        entry = {
            'name': b['name'],
            'method': b['method'],
            'config': b['config'],
            'hypothesis': b['hypothesis'],
            'split': b['split'],
            'metrics': {
                'n': eval_res['n'],
                'hit1': h1_str,
                'hit3': h3_str,
                'context_evidence_rate': ctx_str,
                'median_ms': eval_res['median_ms'],
                'cost_usd': eval_res['cost_usd']
            },
            'traces': eval_res['traces']
        }
        all_results.append(entry)

    with open(results_dir / 'retrieval_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("="*80)
    print(f"Detailed retrieval benchmark results saved to {results_dir / 'retrieval_comparison.json'}")

if __name__ == '__main__':
    main()
