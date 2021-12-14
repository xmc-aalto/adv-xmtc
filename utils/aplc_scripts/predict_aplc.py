import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from .models import (XLNetConfig, XLNetForMultiLabelSequenceClassification, 
                                 WEIGHTS_NAME, XLNetTokenizer)

from tqdm import tqdm

MODEL_CLASSES = {
    'xlnet': (XLNetConfig, XLNetForMultiLabelSequenceClassification, XLNetTokenizer)
}


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class AplcXlnetModel(object):

    config_class, model_class, tokenizer_class = MODEL_CLASSES['xlnet']

    def __init__(self, model_path):
        self.tokenizer = self.tokenizer_class.from_pretrained(model_path)
        self.model = self.model_class.from_pretrained(model_path).to('cuda')
        self.model.reweighting_factors = [None] * 4
        # tgt_model.to('cuda')
    
    def predict(self, list_seq: list, labels: list, max_length: int, return_logits=True, top=5, batch_size=12, ver=False):

        all_input_ids = [] 
        all_masks = [] 
        all_segs = []
        for i, seq in enumerate(list_seq):
            feature_ids = self._convert_feature_to_ids(seq, max_length, self.tokenizer,
                                                       # xlnet has a cls token at the end
                                                       cls_token_at_end=True,
                                                       cls_token=self.tokenizer.cls_token,
                                                       sep_token=self.tokenizer.sep_token,
                                                       cls_token_segment_id=2,
                                                       # pad on the left for xlnet
                                                       pad_on_left=True,
                                                       pad_token_segment_id=4)

            all_input_ids.append(feature_ids.input_ids)
            all_masks.append(feature_ids.input_mask)
            all_segs.append(feature_ids.segment_ids)
        
        seqs = torch.tensor(all_input_ids, dtype=torch.long, device='cuda')
        masks = torch.tensor(all_masks, dtype=torch.long, device='cuda')
        segs = torch.tensor(all_segs, dtype=torch.long, device='cuda')
        labels = torch.tensor(labels, dtype=torch.long, device='cuda')

        eval_data = TensorDataset(seqs, masks, segs, labels)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
        
        loss_all = []
        preds = []

        with torch.no_grad():
            self.model.eval()
            if ver:
                eval_dataloader = tqdm(eval_dataloader, desc='prediction')
            for batch in eval_dataloader:

                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels':         batch[3]
                         }

                outputs = self.model(**inputs)
                loss, logits = outputs[:2]

                loss_all.append(loss.item())
                if not return_logits:
                    preds.append(torch.topk(logits.cpu(), top)[1])
                else:
                    preds.append(logits.cpu())
        return loss_all, preds


    def _convert_feature_to_ids(self, example, max_seq_length,
                            tokenizer, cls_token_at_end=False,
                            pad_on_left=False, cls_token='[CLS]',
                            sep_token='[SEP]', pad_token=0,
                            sequence_a_segment_id=0, sequence_b_segment_id=1,
                            cls_token_segment_id=1, pad_token_segment_id=0,
                            mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """

        tokens_a = tokenizer.tokenize(example)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1]
                        * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + \
                ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        feature_ids = InputFeatures(input_ids=input_ids,
                                    input_mask=input_mask,
                                    segment_ids=segment_ids)

        return feature_ids
