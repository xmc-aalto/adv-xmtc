import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ruamel.yaml import YAML
from pathlib import Path
from .models import AttentionRNN, MultiLabelDataset
from nltk.tokenize import word_tokenize


yaml = YAML(typ='safe')

class AxmlTokenizer(object):

    def __init__(self, data_path):
        self.vocab = self._get_vocab(data_path)
    
    def _get_vocab(self, data_path):
        vocab_path = os.path.join(data_path, 'vocab.npy')
        vocab = {word: idx for idx, word in enumerate(np.load(vocab_path, allow_pickle=True))}
        return vocab

    def tokenize(self, sentence: str, sep='/SEP/', unknown='/UNK/'):
        # We added a /SEP/ symbol between titles and descriptions such as Amazon datasets.
        return [token.lower() if token != sep and token != unknown else token for token in word_tokenize(sentence)
                if len(re.sub(r'[^\w]', '', token)) > 0]
    

    def _truncate_text(self, texts, max_len=500, padding_idx=0, unknown_idx=1):
        if max_len is None:
            return texts
        texts = np.asarray([list(x[:max_len]) + [padding_idx] * (max_len - len(x)) for x in texts])
        texts[(texts == padding_idx).all(axis=1), 0] = unknown_idx
        return texts


    def convert_to_binary(self, texts, max_len=None, pad='<PAD>', unknown='<UNK>'):
        vocab = self.vocab
        texts = np.asarray([[vocab.get(word, vocab[unknown]) for word in row]
                        for row in texts])

        return self._truncate_text(texts, max_len, vocab[pad], vocab[unknown])
    

    def convert_tokens_to_string(self, tokens):
        out_string = ' '.join(tokens).strip()
        return out_string



class AttentionXmlModel(object):

    def __init__(self,  model_path):
        self.cnfg_args = yaml.load(Path(os.path.join(model_path,'confg_axml.yaml')))
        self.model = nn.DataParallel(AttentionRNN(**self.cnfg_args).cuda())
        self._load_model(os.path.join(model_path, 'AttentionXML'))
        self.tokenizer = AxmlTokenizer(model_path)


    def _load_model(self, model_path):
        self.model.module.load_state_dict(torch.load(model_path))


    def _predict_step(self, data_x: torch.Tensor, return_logits, top):
        self.model.eval()
        with torch.no_grad():
            scores = self.model(data_x)
            if not return_logits:
                preds = torch.topk(scores.cpu(), top)[1]
            else:
                preds = scores.cpu()
            return preds


    def predict(self, list_seq: list, labels: list, max_length: int, return_logits=True, top=5, batch_size=12): # labels !!!!!!!
        texts = []
        for seq in list_seq:
            texts.append(self.tokenizer.tokenize(seq))
        texts = self.tokenizer.convert_to_binary(texts, max_length)
        test_loader = DataLoader(MultiLabelDataset(texts), batch_size, num_workers=4)
        preds = [self._predict_step(data_x, return_logits, top) for data_x in test_loader]
        loss = 1 # TODO: compute loss
        return loss, preds
