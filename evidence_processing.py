# 검색모델에서 넘겨주거나 raw data에서 추출한 evidence 파일을
# 다시 기계독해 모델 학습에 이용하기 위해서 전처리 하는 코드

import json
import os
import logging
import argparse
from attrdict import AttrDict
from transformers import AutoTokenizer
from src.functions.utils import init_logger


tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base", use_fast=False, do_lower_case=False)


def get_keyword_label(question, keyword_text, max_length, tokenizer):
    '''
    # 데이터에 키워드가 있을 때 키워드로 레이블을 만듦
    # Query : 한국 정부에서 늘어나는 폐기물 발생량 을 줄이기 위해 수립한 것은
    # Label : 0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1 ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

    :param question: json 파일에서 각 데이터의 질문
    :param keyword_text: 질문에서의 키워드
    :param max_length: 질문 최대 길이
    :param tokenizer: tokenizer
    :return: keyword_label 만든 후에 다시 파일 저장
    '''
    question_tokens = tokenizer.tokenize(question, max_length=max_length, padding="max_length", truncation=True)     # , add_prefix_space=True
    tokenized_keyword = tokenizer.tokenize(keyword_text)

    # keyword label 생성
    keyword_label = [0]
    temp_tokenized_keyword = tokenized_keyword
    for (i, tok) in enumerate(question_tokens):
        if tok in temp_tokenized_keyword:
            keyword_label.append(1)
            temp_tokenized_keyword = temp_tokenized_keyword[1:]
        else:
            keyword_label.append(0)

    keyword_label = keyword_label + [0] * (max_length - len(keyword_label))     # padding

    return keyword_label


def create_evidence_file(evidence_pred, output_filename, output_dir):

    outputs = []
    with open(os.path.join(output_dir, evidence_pred), "r", encoding="utf-8") as f:
        evidences = json.load(f)
        for e in evidences:
            question = e["question"]
            keyword = e["keyword_text"]
            # keyword_labels = [l-1 for l in e["keyword_pred"]] + ([0]*(64-len(e["keyword_pred"])))      # silver
            keyword_labels = get_keyword_label(question=question, keyword_text=keyword, max_length=64,
                                               tokenizer=tokenizer)      # keyword_labels gold
            e["keyword_labels"] = keyword_labels
            # e["evidence_labels"] = list(map(int, data["labels"]))  # silver
            e = {k: v for k, v in e.items() if k not in ["title","answer","splited_sent","labels", "keyword_pred"]}
            outputs.append(e)

    with open(os.path.join(output_dir, output_filename), "w") as writer:
        writer.write(json.dumps(outputs, indent=4, ensure_ascii=False) + "\n")

    print(f"[save] {output_filename}")


def main(cli_args):

    args = AttrDict(vars(cli_args))
    logger = logging.getLogger(__name__)

    init_logger()

    if args.do_preprocessing:
        create_evidence_file(args.pre_evidence_filename, args.evidence_output_file, args.evidence_dir)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    # IR 모델의 예측 evidnece file
    cli_parser.add_argument("--evidence_dir", type=str, default="./data/evidence")
    cli_parser.add_argument("--pre_evidence_filename", type=str, default="val_0_pre.json")     # demo_0_pre.json
    cli_parser.add_argument("--evidence_output_file", type=str, default="val_0_evidence.json")    # demo_0_evidence.json

    # mode
    cli_parser.add_argument("--do_preprocessing", type=bool, default=True)

    cli_args = cli_parser.parse_args()
    main(cli_args)
