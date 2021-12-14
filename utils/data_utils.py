
''' Selecting targeted samples and labels for each attack (A_pos & A_neg) '''
import os
import copy
import numpy as np
import pandas as pd
import argparse
import torch
from aplc_scripts import AplcXlnetModel
from axml_scripts import AttentionXmlModel




def A_pos_sample_selector(test_data, labels_all, labels_freq_dict, labels_freq, pred_labels, data_path, bin_size=50):


    # selecting samples which are classifed correctly for each frequency and each label
    selected_data = {freq: pd.DataFrame([], columns=['id', 'text', 'label', 'target_label']) for freq in labels_freq}
    for pred, data in zip(pred_labels, test_data.iterrows()):
        data = data[1]
        true_labels_index = [np.where(label==labels_all)[0][0] for label in data['label'].split(',') if label in labels_all]

        selected_labels =  [labels_all[label] for label in pred if label in true_labels_index]

        for target_label in selected_labels:
            data['target_label'] = target_label
            freq = labels_freq_dict[target_label]
            data_copy = copy.deepcopy(data)
            selected_data[freq] = selected_data[freq].append(data_copy)
        

    
    # split frequencies into different bins and write them
    selected_freqs = []
    selected_labels = []
    selected_data_bins = pd.DataFrame([], columns=['id', 'text', 'label', 'target_label'])
    selected_labels_bins = {} # store selected labels for each bin (will be used in A_neg_sample_selector)
    
    for i, freq in enumerate(selected_data.keys()):
        selected_freqs.append(freq)
        target_labels = np.unique(np.array(selected_data[freq]['target_label'])).tolist()
        selected_labels.extend(target_labels)
        selected_labels_len = len(selected_labels)
        if selected_labels_len >= bin_size or i==len(selected_data)-1:
            for f in selected_freqs:
                selected_data_bins = selected_data_bins.append(selected_data[f])
            selected_data_bins = selected_data_bins.sample(frac=1) # random shuffling
            start_freq = str(selected_freqs[0])
            end_freq = str(selected_freqs[-1])
            write_path = os.path.join(data_path, F'freq_{start_freq}_{end_freq}_pos.csv')
            selected_data_bins.to_csv(write_path, index=False)

            bin_key = F'{start_freq}_{end_freq}'
            selected_labels_bins[bin_key] = selected_labels

            selected_freqs = []
            selected_labels = []
            selected_data_bins = pd.DataFrame([], columns=['id', 'text', 'label', 'target_label'])


    return selected_labels_bins





def A_neg_sample_selector(test_data, labels_all, labels_sp, labels_freq, pred_labels, data_path, max_samples=10):

    selected_data = {freq: pd.DataFrame([], columns=['id', 'text', 'label', 'target_label']) for freq in labels_sp.keys()}
    for freq in labels_sp.keys():
        if len(labels_sp[freq]) > 0:
            random_idx = np.arange(len(test_data)) # random shuffling
            np.random.shuffle(random_idx)
            pred_labels = pred_labels[random_idx, :]
            test_data = test_data.loc[random_idx, :]
            for pred, data in zip(pred_labels, test_data.iterrows()):
                data = data[1]

                target_labels_list = list(labels_sp[freq])
                # target label is selected randomly from the specified frequency
                target_label = target_labels_list[int(np.random.randint(len(labels_sp[freq]), size=1))]
                true_labels_index = [np.where(label==labels_all)[0][0] for label in data['label'].split(',') if label in labels_all]
                target_label_index = np.where(target_label==labels_all)[0][0]

                if target_label_index not in true_labels_index and target_label_index not in pred:
                    data['target_label'] = target_label
                    data_copy = copy.deepcopy(data)
                    selected_data[freq] = selected_data[freq].append(data_copy)
                if len(selected_data[freq]) >= max_samples:
                    break
    
    for freq in selected_data.keys():
        if not selected_data[freq].empty:
            write_path = os.path.join(data_path, F'freq_{freq}_neg.csv')
            selected_data[freq].to_csv(write_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_label', type=int, default=30, help='number of positive labels (for APLC_XLNet)')
    parser.add_argument('--top', type=int, default=5, help='determines number of predicted labels')
    parser.add_argument('--max_neg_samples', type=int, default=10, help='maximum number of targeted samples for each label in A_neg')
    parser.add_argument('--max_seq_length', type=int, default=512, help='500 for axml and 512 for aplc')
    parser.add_argument('--data_path', type=str, help='data path to store the selected samples for each bin')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_type', type=str, help='aplc or axml')
    parser.add_argument('--bin_size', type=int, default=50, help='minimum number of labels per bin')

    args = parser.parse_args()
    pos_label = args.pos_label
    top = args.top
    max_neg_samples = args.max_neg_samples
    data_path = args.data_path
    model_path = args.model_path
    model_type = args.model_type
    max_seq_length = args.max_seq_length
    bin_size = args.bin_size

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    label_path = os.path.join(os.path.dirname(data_path), 'labels.txt')
    train_path = os.path.join(os.path.dirname(data_path), 'train.csv')
    dev_path = os.path.join(os.path.dirname(data_path), 'dev.csv')

    with open(label_path, 'r') as f:
        labels = f.read().split("\n")
    labels = np.array(labels)

    # computing the frequency of labels
    print('computing the frequency of labels')
    train_data = pd.read_csv(train_path)
    labels_unique = []
    labels_freq = np.array([])
    for i, row in train_data.iterrows():
        sample_labels = row['label'].split(',')
        labels_unique.extend(sample_labels)
    
    labels_unique, labels_freq= np.unique(labels_unique, return_counts=True)
    labels_freq_dict = dict(zip(labels_unique, labels_freq))
    labels_freq = np.unique(labels_freq)

    diff_labels = np.setdiff1d(labels, np.array(list(labels_freq_dict.keys()))) # set labels with zero frequency
    for label in diff_labels:
        labels_freq_dict[label] = 0


    # selecting all samples for prediction
    print('selecting all samples for prediction')
    test_data = pd.read_csv(dev_path).dropna().reset_index(drop=True)

    selected_texts = list(test_data['text'])
    selected_labels = list(test_data['label'])

    for i, sample_labels in enumerate(selected_labels):
        sample_labels_index = [np.where(labels==label)[0][0] if label in labels else
                               labels.size for label in sample_labels.split(',')]
        sample_labels_index.extend([sample_labels_index[0] for i in range(pos_label - len(sample_labels_index))])
        selected_labels[i] = sample_labels_index

    print('evaluating all the test samples')
    model = AplcXlnetModel(model_path) if model_type == 'aplc' else AttentionXmlModel(model_path, data_path)
    _, pred_labels = model.predict(selected_texts, selected_labels, max_seq_length, return_logits=False, top=top, batch_size=12, ver=True)
    pred_labels = torch.cat(pred_labels, dim=0)

    print('selecting samples for A_pos')
    selected_labels_pos = A_pos_sample_selector(test_data, labels, labels_freq_dict, labels_freq, pred_labels, data_path, bin_size)
    print('selecting samples for A_neg')
    A_neg_sample_selector(test_data, labels, selected_labels_pos, labels_freq, pred_labels, data_path, max_neg_samples)

if __name__ == '__main__':
    main()
