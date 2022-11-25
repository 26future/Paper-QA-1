import logging
import random
import torch
import numpy as np
import os

from src.functions.processor_plus import SquadV1Processor, squad_convert_examples_to_features

logger = logging.getLogger(__name__)

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()


def load_and_cache_dataset(args, tokenizer, evaluate=False, output_examples=False, dataset_num=None):
    """
    Load Dataset or Create Dataset
    :param args:
    :param tokenizer:
    :param evaluate:
    :param output_examples:
    :param dataset_num: train dataset split 해서 사용
    :return:
    """
    val_num = str(args.predict_file).split(".")[0]     # validation 용
    cached_evidences_file = os.path.join(
        args.data_dir,
        "cached_{}_{}".format(
            "dev" if evaluate else "train",
            str(dataset_num) if dataset_num is not None else val_num,     # 파일명에 dataset_num 추가
        ),
    )

    # Init evidences from cache if it exists
    if os.path.exists(cached_evidences_file):
        logger.info("Loading features from cached file %s", cached_evidences_file)
        features_and_dataset = torch.load(cached_evidences_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )

    else:
        logger.info("Creating examples from dataset file at %s", args.data_dir)
        logger.info("Creating evidences from evidence file at %s", args.evidence_dir)
        processor = SquadV1Processor()

        if evaluate:
            predict_filename = str(args.predict_file).split(".")[0]
            if args.filtered_context:
                examples = processor.get_dev_examples(data_dir=os.path.join(args.data_dir, "val"),
                                                      evidence_dir=args.evidence_dir,
                                                      input_filename=args.predict_file,
                                                      evidence_filename=f"{predict_filename}_evidence.json",
                                                      filtered_context=True)
            else:
                examples = processor.get_dev_examples(data_dir=os.path.join(args.data_dir, "val"),
                                                      evidence_dir=args.evidence_dir,
                                                      input_filename=args.predict_file,
                                                      evidence_filename=f"{predict_filename}_evidence.json")
        else:
            train_filename = str(args.train_file).split(".")[0]     # train
            examples = processor.get_train_examples(data_dir=os.path.join(args.data_dir, "train"),
                                                    evidence_dir=args.evidence_dir,
                                                    input_filename=f"{train_filename}_{dataset_num}.json",      # splited data
                                                    evidence_filename= f"{train_filename}_{dataset_num}_evidence.json")

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        logger.info("Saving features into cached file %s", cached_evidences_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_evidences_file)

    if output_examples:
        return dataset, examples, features
    return dataset


# 데모용
def load_examples(args, tokenizer, evaluate=True, output_examples=False, input_dict=None):

    processor = SquadV1Processor()

    examples = processor.get_example_from_input(input_dict, args.max_query_length, tokenizer)

    features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

    if output_examples:
        return dataset, examples, features
    return dataset