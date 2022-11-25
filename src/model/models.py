
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import RobertaPreTrainedModel

from src.model.roberta_model import RobertaModel      # 키워드 임베딩 + evidence 임베딩 추가 모델


class PaperQuestionAnswering(RobertaPreTrainedModel):
    def __init__(self, config):
        super(PaperQuestionAnswering, self).__init__(config)

        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.roberta = RobertaModel(config)     # 임베딩 추가
        self.bi_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                             num_layers=1, batch_first=True, bidirectional=True)     # dropout=0.2
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            keyword_labels=None,
            evidence_labels=None,
            start_positions=None,
            end_positions=None,
    ):

        outputs = self.roberta(
            input_ids = input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            keyword_labels=keyword_labels,
            evidence_labels=evidence_labels,
        )

        sequence_output = outputs[0]
        gru_output, _ = self.bi_gru(sequence_output)
        logits = self.qa_outputs(gru_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        outputs = (start_logits, end_logits,)

        if start_positions is not None and end_positions is not None:

            loss_fct = CrossEntropyLoss()

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss ) / 2

            # outputs : (total_loss, start_logits, end_logits)
            outputs = (total_loss,) + outputs

        return outputs

