import json
from pathlib import Path
import chromadb
import re

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load chunks from Day 6 ChromaDB (strategy_b - paragraph chunks, best semantic coherence)
    chroma_path = r'E:\AI_Vin\K5-L3A-Day06-2A202603484-DataIngestionEngineering\chroma_db'
    client = chromadb.PersistentClient(chroma_path)
    col = client.get_collection('strategy_b')
    res = col.get()

    chunks = []
    for i in range(len(res['ids'])):
        meta = dict(res['metadatas'][i])
        doc_id = meta.get('document_id', 'handbook-sv-2025')
        src = meta.get('source', 'data/handbook_sv_2025.pdf').replace('\\', '/')
        page = int(meta.get('page', 0))
        chunks.append({
            'id': res['ids'][i],
            'text': res['documents'][i],
            'source_id': doc_id,
            'source': src,
            'page': page,
            'metadata': meta
        })

    # Sort chunks by id
    chunks.sort(key=lambda c: c['id'])

    # 2. Load questions from Day 6 benchmark
    day6_q_path = r'E:\AI_Vin\K5-L3A-Day06-2A202603484-DataIngestionEngineering\benchmark\questions.json'
    day6_questions = json.load(open(day6_q_path, encoding='utf-8'))

    questions = []
    cases = []

    for q in day6_questions:
        gold_page = q['gold_page']
        ev_quote = q['evidence_quote']
        page_chunks = [c for c in chunks if c['page'] == gold_page]
        
        if q['id'] == 'q09':
            gold_list = [
                {'source_id': q['gold_source_id'], 'page': gold_page, 'evidence': 'Sinh viên được chấp thuận hoãn thi sẽ được nhận điểm chữ I (Chưa hoàn thành) trên hệ thống.'},
                {'source_id': q['gold_source_id'], 'page': gold_page, 'evidence': 'để chuyển điểm I thành điểm chính thức.'}
            ]
        else:
            norm_ev = ' '.join(ev_quote.split())
            best_chunk = None
            best_exact_quote = None
            words = norm_ev.split()
            pattern = r'\s+'.join(re.escape(w) for w in words)
            for c in page_chunks:
                m = re.search(pattern, c['text'])
                if m:
                    best_chunk = c
                    best_exact_quote = m.group(0)
                    break
            if not best_exact_quote:
                raise ValueError(f"Could not find exact quote for {q['id']}")
            gold_list = [{'source_id': q['gold_source_id'], 'page': gold_page, 'evidence': best_exact_quote}]

        item = {
            'id': q['id'],
            'split': q['split'],
            'question': q['question'],
            'expected_answer': q['expected_answer'],
            'gold': gold_list
        }
        if 'filter' in q and q['filter']:
            item['filter'] = q['filter']
        questions.append(item)

        cases.append({
            'id': q['id'],
            'split': q['split'],
            'question': q['question'],
            'answerable': True,
            'expected_behavior': 'search_and_answer',
            'expected_answer': q['expected_answer']
        })

    # 3. Add unanswerable questions (2 for dev, 2 for test)
    unanswerable = [
        {
            'id': 'u01',
            'split': 'dev',
            'question': 'Mức phí gửi xe máy và xe đạp điện theo tháng tại tầng hầm ký túc xá là bao nhiêu tiền?',
            'answerable': False,
            'expected_behavior': 'search_and_abstain',
            'absence_reason': 'Tài liệu sổ tay sinh viên quy định quy chế đào tạo, học phí học phần và học bổng; không có điều khoản quy định mức phí trông giữ xe ký túc xá.'
        },
        {
            'id': 'u02',
            'split': 'dev',
            'question': 'Quy trình đăng ký tạm trú cho sinh viên quốc tế tại công an phường được hướng dẫn như thế nào?',
            'answerable': False,
            'expected_behavior': 'search_and_abstain',
            'absence_reason': 'Tài liệu tập trung vào quy chế học vụ, điểm số và khen thưởng; không có quy trình thủ tục hành chính đăng ký tạm trú tại công an phường.'
        },
        {
            'id': 'u03',
            'split': 'test',
            'question': 'Quy định về việc bảo lưu kết quả học tập để thực hiện nghĩa vụ quân sự được gia hạn tối đa bao nhiêu năm?',
            'answerable': False,
            'expected_behavior': 'search_and_abstain',
            'absence_reason': 'Điều 1 quy định tạm dừng học tập có phép không vượt quá 2 lần thời gian thiết kế tiêu chuẩn nói chung, không có điều khoản quy định số năm gia hạn riêng cho nghĩa vụ quân sự.'
        },
        {
            'id': 'u04',
            'split': 'test',
            'question': 'Sinh viên có được phép nuôi thú cưng (chó, mèo) trong phòng ký túc xá không và hình thức xử phạt nếu vi phạm?',
            'answerable': False,
            'expected_behavior': 'search_and_abstain',
            'absence_reason': 'Tài liệu là sổ tay quy chế đào tạo đại học chính quy, không có quy chế nội trú ký túc xá hoặc quy định về việc cấm nuôi vật nuôi.'
        }
    ]

    cases.extend(unanswerable)

    # Save files
    with open(data_dir / 'chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    with open(data_dir / 'questions.json', 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    with open(data_dir / 'cases.json', 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"Data prepared successfully:")
    print(f"  - chunks: {len(chunks)}")
    print(f"  - answerable questions: {len(questions)} (dev: {sum(1 for q in questions if q['split'] == 'dev')}, test: {sum(1 for q in questions if q['split'] == 'test')})")
    print(f"  - cases: {len(cases)} (dev: {sum(1 for c in cases if c['split'] == 'dev')}, test: {sum(1 for c in cases if c['split'] == 'test')})")

if __name__ == '__main__':
    main()
