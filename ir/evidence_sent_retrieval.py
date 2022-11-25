# 검색모델에서 받은 keyword 및 evidence sentence를 이용해서
# 기계독해 모델이 사용할 filtered context 만드는 코드
# 검색모델이 만든 "test_0_context.json" 파일을 열어서 "test_0_context.json" 파일을 만듦

import json
import os
import os.path as path
from tqdm import tqdm

import config
from src.lucene.searcher import Searcher

root_dir = path.abspath(path.join(__file__, "../../../"))
read_dir = os.path.join(root_dir, "data")
read_file = "test_0_context.json"
read_path = os.path.join(read_dir, read_file)

s = Searcher()
print(config.DOCUMENT_PATH)

with open(read_path, "r", encoding="utf-8", errors="ignore") as rf:
    all_text = json.load(rf, strict=False)

    cnt = 0
    examples = []
    sentence_idxs = []
    correct = [0]*10
    for data in tqdm(all_text):
        question = data["question"]
        keyword = data["keyword_text"]
        evidence_sent = data["evidence_sent"][0]
        answer = data["answer"]
        q_id = data["question_id"].split("-")[0].replace("_", "")
        # 쿼리로 검색
        try:
            sentences = s.search(question)
        except:
            print(keyword)

        candidates = [e.document.replace("\n", " ") for e in sentences[:100] if keyword in e.document and q_id in e.document_name]
        sentences = [e for e in sentences[:100] if keyword in e.document and q_id in e.document_name]

        sentence_idx = []
        # 검색된 문장들 중 top 10 후보
        for e in sentences[:10]:
            if keyword in e.document and q_id in e.document_name:
                temp = e.title.split("_")[1]

                sentence_idx.append(int(e.title.split("_")[1]))
                sentence_idx.append(int(e.title.split("_")[1]) + 1)
                sentence_idx.append(int(e.title.split("_")[1]) + 2)
        sentence_idxs.append(sentence_idx)

        result = candidates
        for top_n in range(10):
            # if evidence_sent in " ".join(result[:top_n+1]):  # evidence 문장 검색
            #     correct[top_n] += 1
            if answer in " ".join(result[:top_n + 1]):  # answer 검색
                correct[top_n] += 1

    top_n_recall = [cor / (len(all_text) + 1e-10) for cor in correct]
    print("\n".join(["top_{}_recall: {}".format(e + 1, top_n_recall[e]) for e in range(10)]))

    # 검색된 문장들로 filtered context 만들기
    for data, sentence_idx in tqdm(zip(all_text, sentence_idxs)):
        evidence_sent = data["evidence_sent"][0]
        retrieved_context = set()

        if not sentence_idx or len(sentence_idx) < 3:
            result_context = data["splited_sent"]
        else:
            for idx in sentence_idx:
                if idx in range(len(data["splited_sent"])):
                    retrieved_context.add((data["splited_sent"][idx], idx))

            # 원본 내용 순서대로 검색된 문장 정렬
            retrieved_context = list(retrieved_context)
            sorted_context = sorted(retrieved_context, key=lambda x:x[1])
            result_context = [c[0] for c in sorted_context]

        example = {"title": data["title"], "answer": data["answer"], "question_id": data["question_id"],
                   "question": data["question"], "context": result_context, "keyword_text": data["keyword_text"],
                   "evidence_sent": data["evidence_sent"], "labels": data["labels"],
                   "keyword_pred": data["keyword_pred"]}

        examples.append(example)
        temp = examples

    # 파일로 filtered context 저장
    write_file = "test_0_context.json"
    write_path = os.path.join(read_dir, write_file)
    with open(write_path, "w", encoding="utf-8") as wf:
        json.dump(examples, wf, ensure_ascii=False, indent="\t")
        print()
        print("__context.json created!")

