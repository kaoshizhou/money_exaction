from transformers import BertModel
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.BERT_NAME = 'hfl/chinese-roberta-wwm-ext'
        self.bert = BertModel.from_pretrained(self.BERT_NAME)
        for p in self.parameters():
            p.requires_grad = False
        self.cls = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        hidden_state = self.bert(**inputs).last_hidden_state
        outputs = self.cls(hidden_state)
        outputs = self.softmax(outputs)
        return outputs