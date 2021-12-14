import os
import pandas as pd
import numpy as np



def preprocess(data_path, label_path, write_path):
    with open(data_path) as f:
        data = f.read().split('\n')
    data = data[:-1]

    with open(label_path) as f:
        labels = f.read().split('\n')
    labels = labels[:-1]


    new_data = []
    for i, (sample, label) in enumerate(zip(data, labels)):
        if sample == '':
            continue
        # data_csv = data_csv.append(pd.DataFrame({'id': [i], 'text': sample, 'label':label}))
        label = ','.join(label.split(' '))
        new_data.append([i, sample, label])
    
    data_csv = pd.DataFrame(new_data, columns=['id', 'text', 'label'])

    data_csv.to_csv(write_path, index=False)


data_dir = 'data_defense/data/axml/Wiki10'

data_path = os.path.join(data_dir, 'train_texts.txt')
label_path = os.path.join(data_dir, 'train_labels.txt')
write_path = os.path.join(data_dir, 'train.csv')

preprocess(data_path, label_path, write_path)


data_path = os.path.join(data_dir, 'test_texts.txt')
label_path = os.path.join(data_dir, 'test_labels.txt')
write_path = os.path.join(data_dir, 'dev.csv')

preprocess(data_path, label_path, write_path)

label_map_np_path = os.path.join(data_dir, 'labels_binarizer')
labels_map_txt_path = os.path.join(data_dir, 'labels.txt')

labels_all = np.load(label_map_np_path, allow_pickle=True)
with open(labels_map_txt_path, 'w') as f:
    f.write('\n'.join(labels_all))
