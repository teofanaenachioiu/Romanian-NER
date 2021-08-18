import torch
from transformers import BertTokenizer
import numpy as np

TAG_VALUES = ['I-PERIOD',
              'B-PERSON',
              'I-ORDINAL',
              'I-GPE',
              'B-QUANTITY',
              'I-WORK_OF_ART',
              'B-FACILITY',
              'I-QUANTITY',
              'B-EVENT',
              'B-ORDINAL',
              'I-NAT_REL_POL',
              'I-NUMERIC_VALUE',
              'B-PRODUCT',
              'O',
              'B-GPE',
              'I-LANGUAGE',
              'B-LANGUAGE',
              'I-LOC',
              'I-PRODUCT',
              'I-EVENT',
              'B-LOC',
              'I-ORGANIZATION',
              'B-NUMERIC_VALUE',
              'B-DATETIME',
              'B-PERIOD',
              'B-WORK_OF_ART',
              'B-ORGANIZATION',
              'I-FACILITY',
              'I-MONEY',
              'B-NAT_REL_POL',
              'I-PERSON',
              'B-MONEY',
              'I-DATETIME']


class NER_Model:
    def __init__(self):
        self.saved_model = torch.load('model_fine_tuned', map_location=torch.device('cpu'))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',
                                                       do_lower_case=False)

    def _construct_text(self, tokens, labels):
        result = ""
        for token, label in zip(tokens, labels):
            result += f"<span " \
                      f"foreground=\'{'red' if label != 'O' else 'white'}\'>" \
                      f"{token} {'[' + label + ']' if label != 'O' else ''} " \
                      f"</span>"
        return result

    def predict(self, test_sentence):
        tokenized_sentence = self.tokenizer.encode(test_sentence)
        input_ids = torch.tensor([tokenized_sentence])

        with torch.no_grad():
            output = self.saved_model(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []

        for token, label_idx in zip(tokens, label_indices[0]):
            if token == '[CLS]' or token == "[SEP]":
                continue
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(TAG_VALUES[label_idx])
                new_tokens.append(token)

        return self._construct_text(new_tokens, new_labels)
