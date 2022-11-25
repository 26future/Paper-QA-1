from transformers import AutoConfig
from transformers import AutoTokenizer

from src.model.models import PaperQuestionAnswering

CONFIG = {
    "paper_qa":AutoConfig,
}

TOKENIZER = {
    "paper_qa":AutoTokenizer,
}

QUESTION_ANSWERING_MODEL = {
    "paper_qa":PaperQuestionAnswering,
}
