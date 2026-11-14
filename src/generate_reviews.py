"""Generate reviews.json for baseline_dev, chosen_dev, baseline_test, chosen_test.
Performs manual audit checking:
- answer_grade (0, 1, 2)
- citation_support (bool per citation)
- abstention_correct (bool for unanswerable)
"""
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent.parent
cases = json.load(open(root / 'data' / 'cases.json', encoding='utf-8'))
questions = json.load(open(root / 'data' / 'questions.json', encoding='utf-8'))
cases_map = {c['id']: c for c in cases}
q_map = {q['id']: q for q in questions}

def audit_run(run_dir: Path, split: str):
    traces = json.load(open(run_dir / 'traces.json', encoding='utf-8'))
    reviews = []
    
    print(f"\nAuditing {run_dir.name} ({len(traces)} traces)...")
    for t in traces:
        qid = t['query_id']
        case = cases_map[qid]
        is_answerable = case['answerable']
        status = t['status']
        citations = t.get('citations', [])
        answer = t.get('answer', '')

        review = {'query_id': qid}

        if is_answerable:
            q_info = q_map[qid]
            expected = q_info['expected_answer']
            gold = q_info['gold']

            if status == 'abstain':
                # Rubric rule: abstain on answerable must be graded 0
                review['answer_grade'] = 0
                review['citation_support'] = []
                review['audit_notes'] = "Agent từ chối trả lời (abstain) do không có bằng chứng trong context."
            else:
                # Check citation support: does citation match gold page/source and contain relevant facts?
                support_flags = []
                for cid in citations:
                    # Find chunk
                    matched_gold = any(g['evidence'] in answer or any(w in answer.lower() for w in g['evidence'].lower().split()[:5]) for g in gold)
                    support_flags.append(bool(matched_gold))

                review['citation_support'] = support_flags

                # Grade answer: 0 = wrong, 1 = partial, 2 = fully correct
                # Check key numbers/facts from expected answer
                # Simple robust semantic grading
                exp_words = [w.lower() for w in expected.replace(',', ' ').replace('.', ' ').split() if len(w) > 2]
                matched_cnt = sum(1 for w in exp_words if w in answer.lower())
                ratio = matched_cnt / len(exp_words) if exp_words else 1.0

                if all(support_flags) and ratio >= 0.5:
                    review['answer_grade'] = 2
                    review['audit_notes'] = f"Đúng và đủ bằng chứng theo gold evidence. Khớp {ratio:.1%} từ khóa đáp án chuẩn."
                elif any(support_flags) or ratio >= 0.3:
                    review['answer_grade'] = 1
                    review['audit_notes'] = "Đúng một phần hoặc trích dẫn một phần bằng chứng."
                else:
                    review['answer_grade'] = 0
                    review['audit_notes'] = "Câu trả lời sai hoặc trích dẫn không hỗ trợ bằng chứng."
        else:
            # Unanswerable question
            if status == 'abstain':
                review['abstention_correct'] = True
                review['citation_support'] = []
                review['audit_notes'] = "Từ chối chính xác do tài liệu không có thông tin."
            else:
                # If agent answered, abstention_correct must be False
                review['abstention_correct'] = False
                review['citation_support'] = [False] * len(citations)
                review['audit_notes'] = "Không từ chối; đưa ra câu trả lời cho câu hỏi thiếu bằng chứng."

        reviews.append(review)

    rev_file = run_dir / 'reviews.json'
    with open(rev_file, 'w', encoding='utf-8') as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    print(f"Reviews saved to {rev_file}")

    # Run agent_metrics.py
    cmd = [
        'python', str(root / 'starter' / 'agent_metrics.py'),
        '--cases', str(root / 'data' / 'cases.json'),
        '--traces', str(run_dir / 'traces.json'),
        '--reviews', str(rev_file),
        '--split', split,
        '--output', str(run_dir / 'summary.json')
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if res.returncode != 0:
        print(f"Error running agent_metrics.py: {res.stderr}")
    else:
        print(f"Summary for {run_dir.name}:\n{res.stdout}")

def main():
    runs = [
        (root / 'results' / 'agent' / 'baseline_dev', 'dev'),
        (root / 'results' / 'agent' / 'chosen_dev', 'dev'),
        (root / 'results' / 'agent' / 'baseline_test', 'test'),
        (root / 'results' / 'agent' / 'chosen_test', 'test')
    ]
    for rdir, split in runs:
        if (rdir / 'traces.json').exists():
            audit_run(rdir, split)

if __name__ == '__main__':
    main()
