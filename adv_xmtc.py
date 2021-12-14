# The script is adapted from https://github.com/LinyangLee/BERT-Attack/blob/master/bertattack.py
# written by Linyang Li


import warnings
import os

import torch
import torch.nn as nn
import json
from transformers import BertConfig, BertTokenizer
from transformers import BertForSequenceClassification, BertForMaskedLM

from utils import processors

import copy
import argparse
import numpy as np

from utils import AplcXlnetModel
from utils import AttentionXmlModel


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

TGT_MODEL_TYPES = ['axml', 'aplc']

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
filter_words = set(filter_words)


def get_sim_embed(embed_path, sim_path):
    id2word = {}
    word2id = {}

    with open(embed_path, 'r', encoding='utf-8') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in id2word:
                id2word[len(id2word)] = word
                word2id[word] = len(id2word) - 1

    cos_sim = np.load(sim_path)
    return cos_sim, word2id, id2word


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class Feature(object):
    def __init__(self, seq_a, label, label_str, target_label, target_label_str):
        self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []

        self.label_str = label_str
        self.target_label = target_label
        self.target_label_str = target_label_str
        self.adv_label = label
        self.adv_label_str = label_str
        self.pred_label = label
        self.pred_label_str = label_str


def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '')

    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = [word]
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys

def _special_char_axml(texts):
    for i, text in enumerate(texts):
        text = text.replace('[SEP]', '/SEP/')
        text = text.replace('[UNK]', '/UNK/')
        texts[i] = text
    return texts

def _get_masked(words):
    len_text = len(words)
    masked_words = []
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    # list of words
    return masked_words


def get_important_scores(words, tgt_model, target_label, orig_label, orig_probs, attack_type, tgt_type, batch_size, max_length):

    masked_words = _get_masked(words)
    # list of text of masked words
    texts = [' '.join(words) for words in masked_words]

    if tgt_type == 'axml':
        texts = _special_char_axml(texts)
    _, leave_1_probs = tgt_model.predict(texts, [orig_label for i in range(len(texts))], max_length)
    leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
    leave_1_probs = torch.sigmoid(leave_1_probs)

    orig_prob = orig_probs[target_label]

    if attack_type == 'A_pos':
        import_scores = ((orig_prob - leave_1_probs[:, target_label])).data.cpu().numpy()

    if attack_type == 'A_neg':
        import_scores = (leave_1_probs[:, target_label] - orig_prob).data.cpu().numpy()

    return import_scores


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k

    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N*L, -1),
                 all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words


def attack(feature, tgt_model, mlm_model, tokenizer_mlm, \
           k, label_rev_map, attack_type, tgt_type, change_threshold, \
           batch_size, max_length=512, cos_mat=None, w2i={}, \
           i2w={}, use_bpe=1, threshold_pred_score=0.3, top=5):
    # MLM-process
    words, sub_words, keys = _tokenize(feature.seq, tokenizer_mlm)


    _, logits = tgt_model.predict([feature.seq], [feature.label], max_length)
    logits = logits[0]

    orig_probs = torch.sigmoid(logits).squeeze()
    _, pred_label = torch.sort(logits, descending=True)
    pred_label = pred_label.squeeze().tolist()


    target_label = feature.target_label
    if attack_type == 'A_pos':
        invalid_attack = target_label not in feature.label or target_label not in pred_label[:top]
    elif attack_type == 'A_neg':
        invalid_attack = target_label in feature.label or target_label in pred_label[:top]
    else:
        print('Unrecognized attack type')
        feature.success = -1
        return feature


    if invalid_attack:
        print('invalid attack')
        feature.success = -1
        return feature

    feature.pred_label = pred_label[:10]
    feature.pred_label_str = [label_rev_map[label] for label in feature.pred_label]

    feature.adv_label = pred_label[:10]
    feature.adv_label_str = [label_rev_map[label] for label in feature.pred_label]

    sub_words = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]']
    input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
    word_predictions = mlm_model(input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab
    word_pred_scores_all, word_predictions = torch.topk(
        word_predictions, k, -1)  # seq-len k

    word_predictions = word_predictions[1:len(sub_words) + 1, :]
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]


    important_scores = get_important_scores(words, tgt_model, target_label, feature.label, orig_probs,
                                            attack_type, tgt_type, batch_size, max_length)

    feature.query += int(len(words))
    list_of_index = sorted(enumerate(important_scores),
                           key=lambda x: x[1], reverse=True)
    final_words = copy.deepcopy(words)

    for top_index in list_of_index:
        if feature.change > int(change_threshold * (len(words))):
            feature.success = 1  # exceed
            return feature

        tgt_word = words[top_index[0]]
        if tgt_word in filter_words:
            continue
        if keys[top_index[0]][0] > max_length - 2:
            continue

        substitutes = word_predictions[keys[top_index[0]]
                                       [0]:keys[top_index[0]][1]]
        word_pred_scores = word_pred_scores_all[keys[top_index[0]]
                                                [0]:keys[top_index[0]][1]]

        substitutes = get_substitues(
            substitutes, tokenizer_mlm, mlm_model, use_bpe, word_pred_scores, threshold_pred_score)

        most_gap = 0.0
        candidate = None

        for substitute_ in substitutes:
            substitute = substitute_

            if substitute == tgt_word:
                continue  # filter out original word
            if '##' in substitute:
                continue  # filter out sub-word

            if substitute in filter_words:
                continue
            if substitute in w2i and tgt_word in w2i:
                if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                    continue
            temp_replace = final_words
            temp_replace[top_index[0]] = substitute
            temp_text = tgt_model.tokenizer.convert_tokens_to_string(temp_replace)

            if tgt_type == 'axml':
                temp_text = _special_char_axml([temp_text])[0]
            _, temp_logits = tgt_model.predict([temp_text], [feature.label], max_length)
            temp_logits = temp_logits[0]

            feature.query += 1
            temp_prob = torch.sigmoid(temp_logits).squeeze()
            _, temp_label = torch.topk(temp_prob, top)

            if attack_type == 'A_pos':
                terminate_condition = target_label not in temp_label
            if attack_type == 'A_neg':
                terminate_condition = target_label in temp_label

            if terminate_condition:
                feature.change += 1
                final_words[top_index[0]] = substitute
                feature.changes.append(
                    [keys[top_index[0]][0], substitute, tgt_word])
                feature.final_adverse = temp_text
                feature.success = 4

                _, adv_label = torch.sort(temp_prob, descending=True)
                adv_label = adv_label.squeeze().tolist()

                feature.adv_label = adv_label[:10]
                feature.adv_label_str = [label_rev_map[label] for label in feature.adv_label]

                return feature

            else:
                if attack_type == 'A_both' or attack_type == 'A_neg':
                    gap_pos = 0
                    gap_neg = temp_prob[target_label] - orig_probs[target_label]

                if attack_type == 'A_pos':
                    gap_pos = orig_probs[target_label] - temp_prob[target_label]
                    gap_neg = 0

                gap = gap_pos + gap_neg

                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

        if most_gap > 0:
            feature.change += 1
            feature.changes.append(
                [keys[top_index[0]][0], candidate, tgt_word])
            # current_prob = current_prob - most_gap
            final_words[top_index[0]] = candidate

    feature.final_adverse = (tgt_model.tokenizer.convert_tokens_to_string(final_words))

    
    _, adv_label = torch.sort(temp_prob, descending=True)
    adv_label = adv_label.squeeze().tolist()

    feature.adv_label = adv_label[:10]
    feature.adv_label_str = [label_rev_map[label] for label in feature.adv_label]
    feature.success = 2
    return feature


def evaluate(features):  # mohamm: read it and change it for multilabel
    do_use = 1
    use = None
    sim_thres = 0
    # evaluate with USE

    if do_use == 1:
        cache_path = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'
        import tensorflow as tf
        import tensorflow_hub as hub

        class USE(object):
            def __init__(self, cache_path):
                super(USE, self).__init__()

                self.embed = hub.Module(cache_path)
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session()
                self.build_graph()
                self.sess.run(
                    [tf.global_variables_initializer(), tf.tables_initializer()])

            def build_graph(self):
                self.sts_input1 = tf.placeholder(tf.string, shape=(None))
                self.sts_input2 = tf.placeholder(tf.string, shape=(None))

                sts_encode1 = tf.nn.l2_normalize(
                    self.embed(self.sts_input1), axis=1)
                sts_encode2 = tf.nn.l2_normalize(
                    self.embed(self.sts_input2), axis=1)
                self.cosine_similarities = tf.reduce_sum(
                    tf.multiply(sts_encode1, sts_encode2), axis=1)
                clip_cosine_similarities = tf.clip_by_value(
                    self.cosine_similarities, -1.0, 1.0)
                self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

            def semantic_sim(self, sents1, sents2):
                sents1 = [s.lower() for s in sents1]
                sents2 = [s.lower() for s in sents2]
                scores = self.sess.run(
                    [self.sim_scores],
                    feed_dict={
                        self.sts_input1: sents1,
                        self.sts_input2: sents2,
                    })
                return scores[0]

        use = USE(cache_path)

    acc = 0
    origin_success = 0
    total = 0
    total_q = 0
    total_change = 0
    total_word = 0

    for feat in features:
        if feat.success != 2:
            total += 1
            total_change += feat.change
            total_word += len(feat.seq.split(' '))

            if do_use == 1:
                    sim = float(use.semantic_sim([feat.seq], [feat.final_adverse]))
                    feat.sim = sim

            if feat.success > 2:
                acc += 1

                if feat.success == 3:
                    origin_success += 1

    suc = float(acc / total)

    change_rate = float(total_change / total_word)

    origin_acc = 1 - origin_success / total
    after_atk = 1 - suc

    print('acc/aft-atk-acc {:.6f}/ {:.6f}, change-rate {:.4f}'.format(
        origin_acc, after_atk, change_rate))

    eval_results = {'origin_acc':origin_acc, 'after_atk':after_atk, 'change_rate':change_rate}

    return eval_results


def dump_features(features, output, origin_acc, after_atk, change_rate):
    outputs = []

    for feature in features:
        changes = []
        list(map(changes.extend, feature.changes))
        changes = list(map(str, changes))
        changes = ", ".join(changes)
        outputs.append({'true label': ', '.join(feature.label_str),
                        'pred label': ', '.join(feature.pred_label_str),
                        'adv label': ', '.join(feature.adv_label_str),
                        'target label': feature.target_label_str,
                        'success': feature.success,
                        'sim': feature.sim,
                        'change': feature.change,
                        'num_word': len(feature.seq.split(' ')),
                        'query': feature.query,
                        'changes': changes,
                        'seq_a': feature.seq,
                        'adv': feature.final_adverse
                        })
    outputs.append({'original acc': origin_acc,
                    'attack acc': after_atk,
                    'change_rate': change_rate
                    })
    output_json = output
    json.dump(outputs, open(output_json, 'w'), indent=2)

    print('finished dump')


def run_attack():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--mlm_path", type=str)
    parser.add_argument("--tgt_path", type=str)

    parser.add_argument("--output_path", type=str)
    parser.add_argument("--use_sim_mat", type=int,
                        help='whether use cosine_similarity to filter out atonyms')
    parser.add_argument("--use_bpe", type=int)
    parser.add_argument("--k", type=int, default=48, help='number of candidates')
    parser.add_argument("--threshold_pred_score", type=float, default=0.0)

    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--pos_label", type=int)
    parser.add_argument("--top", type=int, default=5, help="threshold for prediction")
    parser.add_argument("--dev_name", type=str, default="dev.csv", help="name of the evaluation file")
    parser.add_argument("--attack_type", type=str, help="it can be A_both, A_pos, or A_neg")
    parser.add_argument("--change_threshold", type=float, default=0.5,
                        help="determines the rate of allowed changes. The lower, the less changes.")
    parser.add_argument("--tgt_type", type=str, help="type of attacked classifier (axml or aplc)")
    parser.add_argument("--pos_samples", type=int, default=100, help="number of tageted samples for A_pos")

    args = parser.parse_args()
    task_name = args.task_name.lower()
    mlm_path = str(args.mlm_path)
    tgt_path = str(args.tgt_path)
    output_path = str(args.output_path)
    use_bpe = args.use_bpe
    k = args.k
    max_seq_length = args.max_seq_length
    threshold_pred_score = args.threshold_pred_score
    top = args.top
    attack_type = args.attack_type
    change_threshold = args.change_threshold
    tgt_type = args.tgt_type
    data_dir = args.data_dir
    pos_samples = args.pos_samples
    dev_name = args.dev_name

    if tgt_type not in TGT_MODEL_TYPES:
        raise ValueError(F'Unknown model type: {tgt_type}')

    print('start process')

    tokenizer_mlm = BertTokenizer.from_pretrained(
        mlm_path, do_lower_case=False)

    config_atk = BertConfig.from_pretrained(mlm_path)
    mlm_model = BertForMaskedLM.from_pretrained(mlm_path, config=config_atk)
    mlm_model.to('cuda')

    tgt_model = AplcXlnetModel(tgt_path) if tgt_type == 'aplc' else AttentionXmlModel(tgt_path)

    processor = processors[task_name]()

    label_list = processor.get_labels(os.path.dirname(data_dir))
    label_map = {label: i for i, label in enumerate(label_list)}

    label_rev_map = {i: label for label, i in label_map.items()}


    examples = processor.get_dev_examples(data_dir, dev_name=dev_name)

    print('loading sim-embed')

    if args.use_sim_mat == 1:
        cos_mat, w2i, i2w = get_sim_embed(
            'data_defense/counter-fitted-vectors.txt', 'data_defense/cos_sim_counter_fitting.npy')
    else:
        cos_mat, w2i, i2w = None, {}, {}

    print('finish get-sim-embed')
    features_output = []

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with torch.no_grad():
        for index, example in enumerate(examples):
            
            label = []
            for item in example.label:
                try:
                    label.append(label_map[item])
                except KeyError:
                    print(F'label "{item}" not found')
            target_label_str = str(example.target_label)
            target_label = label_map[target_label_str]
            feat = Feature(example.text_a, label, example.label, target_label, target_label_str)

            feat = attack(feat, tgt_model, mlm_model, tokenizer_mlm, k, label_rev_map, \
                          attack_type, tgt_type, change_threshold, batch_size=32, \
                          max_length=max_seq_length, cos_mat=cos_mat, w2i=w2i, i2w=i2w, \
                          use_bpe=use_bpe, threshold_pred_score=threshold_pred_score, top=top)

            if feat.success > -1:
                features_output.append(feat)
            if (index + 1) % 5 == 0:
                print(F'Running evalution for {index+1} samples')
                eval_results = evaluate(features_output)
                if features_output != []:
                    dump_features(features_output, output_path, **eval_results)
            if index + 1 == pos_samples and attack_type == 'A_pos':
                break

    print(F'Running final evalution')
    eval_results = evaluate(features_output)
    if features_output != []:
        dump_features(features_output, output_path, **eval_results)


if __name__ == '__main__':
    run_attack()
