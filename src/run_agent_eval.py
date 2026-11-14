"""Run agent evaluation across benchmark splits (dev / test).
Saves traces compatible with starter/agent_metrics.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'starter'))

from agentic import run_agent
from src.retrievers import DenseRetriever, LexicalRetriever, HybridRetriever
from src.agent import OpenRouterController


def run_evaluation(config_name: str, retriever, controller: OpenRouterController,
                   cases: list[dict], questions_map: dict,
                   output_dir: Path, top_k: int = 5, budget: int = 2500, max_searches: int = 4):
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    
    print(f"\nRunning {config_name} on {len(cases)} cases (top_k={top_k}, budget={budget})...")
    for i, c in enumerate(cases):
        qid = c['id']
        question = c['question']
        # Extract filters if provided in question metadata
        q_meta = questions_map.get(qid, {})
        filters = q_meta.get('filter')
        
        start_t = time.perf_counter()
        try:
            res = run_agent(
                question=question,
                controller=controller,
                retriever=retriever,
                filters=filters,
                max_searches=max_searches,
                top_k=top_k,
                budget=budget
            )
            elapsed = time.perf_counter() - start_t
            trace = {
                'query_id': qid,
                'split': c['split'],
                'question': question,
                'status': res['status'],
                'answer': res['answer'],
                'citations': res['citations'],
                'events': res['events'],
                'elapsed_seconds': round(elapsed, 2)
            }
        except Exception as e:
            elapsed = time.perf_counter() - start_t
            trace = {
                'query_id': qid,
                'split': c['split'],
                'question': question,
                'status': 'error',
                'error_message': str(e),
                'answer': '',
                'citations': [],
                'events': [],
                'elapsed_seconds': round(elapsed, 2)
            }
        
        traces.append(trace)
        print(f"[{i+1}/{len(cases)}] {qid}: status={trace['status']}, citations={trace['citations']}, elapsed={trace['elapsed_seconds']}s")

    trace_file = output_dir / 'traces.json'
    with open(trace_file, 'w', encoding='utf-8') as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)

    print(f"Traces saved to {trace_file}")
    return traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('dev', 'test', 'all'), default='all')
    parser.add_argument('--config', choices=('baseline', 'chosen', 'all'), default='all')
    args = parser.parse_args()

    chunks = json.load(open(root / 'data' / 'chunks.json', encoding='utf-8'))
    all_cases = json.load(open(root / 'data' / 'cases.json', encoding='utf-8'))
    all_questions = json.load(open(root / 'data' / 'questions.json', encoding='utf-8'))
    questions_map = {q['id']: q for q in all_questions}

    dense = DenseRetriever()
    lexical = LexicalRetriever(chunks)
    hybrid = HybridRetriever(dense, lexical)
    controller = OpenRouterController()

    splits = ['dev', 'test'] if args.split == 'all' else [args.split]
    configs = ['baseline', 'chosen'] if args.config == 'all' else [args.config]

    for s in splits:
        split_cases = [c for c in all_cases if c['split'] == s]
        for cfg in configs:
            if cfg == 'baseline':
                # Baseline: Dense Retriever, top_k=3, budget=800
                out_dir = root / 'results' / 'agent' / f"{cfg}_{s}"
                run_evaluation(
                    config_name=f"Baseline (Dense, top_k=3, budget=800) [{s}]",
                    retriever=dense,
                    controller=controller,
                    cases=split_cases,
                    questions_map=questions_map,
                    output_dir=out_dir,
                    top_k=3,
                    budget=800
                )
            elif cfg == 'chosen':
                # Chosen: Hybrid RRF, top_k=5, budget=2500
                out_dir = root / 'results' / 'agent' / f"{cfg}_{s}"
                run_evaluation(
                    config_name=f"Chosen (Hybrid RRF, top_k=5, budget=2500) [{s}]",
                    retriever=hybrid,
                    controller=controller,
                    cases=split_cases,
                    questions_map=questions_map,
                    output_dir=out_dir,
                    top_k=5,
                    budget=2500
                )


if __name__ == '__main__':
    main()
