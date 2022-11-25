# raw 데이터에서 keyword 및 evidence sentence Gold를 찾는 코드
# 기계독해 모델 학습시 Gold 사용

from transformers import AutoTokenizer
import json
import os
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


class QA_processor:
    def __init__(self, lm_name, max_length=100):
        cache_dir = "cache_dir/"
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name, cache_dir=cache_dir)

    def de_preprocessing(self, read_path, write_path):
        def checker(feature):
            context = feature["context"]
            keywords = feature["keywords"]
            questions = feature["questions"]

        with open(read_path, "r", encoding="utf-8", errors='ignore') as rf:
            all_text = json.load(rf, strict=False)
            # data = all_text["data"]

            data = []
            for text in all_text:
                data += text["data"]

            features = []
            for example in tqdm(data):
                context = example["context"]
                title = example["title"]

                for qa in example["qas"]:
                    feature = []
                    temp_context = example["context"]
                    question = qa["question"]
                    answer = qa["answer"]["answer_text"]
                    keyword_text = qa["keyword"]["keyword_text"]
                    keyword_start = qa["keyword"]["keyword_start"]  # 컨텍스트 내 키워드 시작 음절 위치
                    keyword_end = keyword_start + len(keyword_text) - 1
                    question_id = qa["id"]

                    if keyword_text[0] != temp_context[keyword_start]:
                        continue

                    # 컨텍스트 내 키워드 마스킹
                    masked_context = temp_context[:keyword_start] + "[MASK]" + temp_context[keyword_end+1:]

                    # 컨텍스트를 문장 단위로 분리
                    splited_sent = []
                    sentences = masked_context.split("\n\n")
                    for sentence in sentences:
                        splited_sent += sent_tokenize(sentence)

                    # 컨텍스트에서 evidence sentence 찾기
                    evidence_sent = []
                    labels = []
                    ori_splited_sent = []
                    for sent in splited_sent:
                        if "[MASK]" in sent:
                            sent = sent.replace("[MASK]", keyword_text)
                            evidence_sent.append(sent)
                            ori_splited_sent.append(sent)
                            labels.append(1)

                        else:
                            labels.append(0)
                            ori_splited_sent.append(sent)

                    feature = {"title": title, "answer": answer, "question_id": question_id, "question": question, "splited_sent": ori_splited_sent, "keyword_text": keyword_text, "evidence_sent": evidence_sent, "labels": labels}
                    features.append(feature)

        with open(write_path, "w", encoding="utf-8") as wf:
            json.dump(features, wf, ensure_ascii=False, indent="\t")

        return features



if __name__ == "__main__":
    processor = QA_processor("klue/roberta-large")
    root_dir = "../"  # QA

    read_dir = os.path.join(root_dir, "data", "train")
    read_files = os.listdir(read_dir)

    write_dir = os.path.join(root_dir, "preproc")
    write_files = ["train_04_combined_pre.json"]

    for read_file, write_file in zip(read_files, write_files):
        read_path = os.path.join(read_dir, read_file)
        write_path = os.path.join(write_dir, write_file)

        processor.de_preprocessing(read_path=read_path, write_path=write_path)