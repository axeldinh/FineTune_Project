from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np

######################################################################################
###### DATASETS

class single_sentence_set(Dataset):
    """
    Creates a dataset given an encoding and labels, returns a tuple (encoding, label)
    Parameters:
        - sentences (list(str)): list of the sentences to encode
        - labels (list(int)): list of labels
        - tokenizer: tokenizer to use for the encoding
        - max_length (int): maximum length of the 
    """
    def __init__(self, sentences, labels, tokenizer, max_length = None):

        self.encodings = []
        self.labels = []

        if max_length is not None:
            
            for sentence, label in zip(sentences, labels):
                
                encoding = tokenizer(sentence)
                if len(encoding['input_ids']) <= max_length:
                    self.encodings.append(encoding)
                    self.labels.append(label)
        else:
            self.encodings = [tokenizer(x) for x in sentences]
            self.labels = labels
        
    def __getitem__(self, i):
        return (self.encodings[i],
               self.labels[i])
        
    def __len__(self):
        return len(self.encodings)
    
class double_sentences_set(Dataset):
    """
    Creates a dataset given an encoding and labels, returns a tuple (encoding, label)
    Parameters:
        - sentences1 (list(str)): list of the first sentences to encode
        - sentences2 (list(str)): list of the second sentences to encode
        - labels (list(int)): list of labels
        - tokenizer: tokenizer to use for the encoding
        - max_length (int): maximum length of the 
    """
    def __init__(self, sentences1, sentences2, labels, tokenizer, max_length=None):

        self.encodings = []
        self.labels = []

        if max_length is not None:
            for sentence1, (sentence2, label) in zip(sentences1, zip(sentences2, labels)):
                encoding = tokenizer(sentence1, sentence2)
                if len(encoding['input_ids']) <= max_length:
                    self.encodings.append(encoding)
                    self.labels.append(label)
        else:
            self.encodings = [tokenizer(x, y) for x, y in zip(sentences1, sentences2)]
            self.labels = labels
        
    def __getitem__(self, i):
        return (self.encodings[i],
                self.labels[i])
        
    def __len__(self):
        return len(self.encodings)

def load_dataset(dataset_name, tokenizer, TRAIN_PATH, TEST_PATH, train_frac = 0.8, subsample_train_size = None, subsample_test_size = None, max_length = None):
    """
    Load a dataset.
    Parameters:
        -dataset_name (str): dataset to load, can be 'CoLA', 'SST-2', 'MRPC', 'QNLI', 'QQP', 'RTE'.
        -tokenizer: tokenizer for the discretization of the dataset.
        -TRAIN_PATH (str)
        -TEST_PATH (str)
        -subsample (bool): Set to True to only retrieve 1000 samples.
        -subsample_size (int): size of the subsample
        -max_length (int): maximum size of the sentences to retrieve.
    Returns:
        -train_dataset: dataset which returns a tuple (encoding, label)
        -val_dataset: same but for the validation dataset
        -test_dataset: same but for the test dataset
    """
    
    if dataset_name == 'CoLA':
        df_train = pd.read_csv(TRAIN_PATH, sep = '\t', header = None)
        df_test = pd.read_csv(TEST_PATH, sep = '\t', header = None)
        
        len_train = len(df_train)
        len_test = len(df_test)
        
        train_idx = np.random.permutation(range(len_train))[:subsample_train_size]
        test_idx = np.random.permutation(range(len_test))[:subsample_test_size]
        
        sentences_train = list(df_train[3].iloc[train_idx])
        labels_train = list(df_train[1].iloc[train_idx])
        sentences_test = list(df_test[3].iloc[test_idx])
        labels_test = list(df_test[1].iloc[test_idx])

        train_dataset = single_sentence_set(sentences_train, labels_train, tokenizer, max_length)
        test_dataset = single_sentence_set(sentences_test, labels_test, tokenizer, max_length)
        
    elif dataset_name == 'SST-2':
        
        df_train = pd.read_csv(TRAIN_PATH, sep = '\t')
        df_test = pd.read_csv(TEST_PATH, sep = '\t')
        
        len_train = len(df_train)
        len_test = len(df_test)
        
        train_idx = np.random.permutation(range(len_train))[:subsample_train_size]
        test_idx = np.random.permutation(range(len_test))[:subsample_test_size]
        
        sentences_train = list(df_train['sentence'].iloc[train_idx])
        labels_train = list(df_train['label'].iloc[train_idx])
        sentences_test = list(df_test['sentence'].iloc[test_idx])
        labels_test = list(df_test['label'].iloc[test_idx])

        train_dataset = single_sentence_set(sentences_train, labels_train, tokenizer, max_length)
        test_dataset = single_sentence_set(sentences_test, labels_test, tokenizer, max_length)
        
    elif dataset_name == 'MRPC':
        
        df_train = pd.read_csv(TRAIN_PATH, sep = '\t', error_bad_lines=False)
        df_train.dropna(inplace = True)

        df_test = pd.read_csv(TEST_PATH, sep = '\t', error_bad_lines=False)
        df_test.dropna(inplace = True)
        
        len_train = len(df_train)
        len_test = len(df_test)
        
        train_idx = np.random.permutation(range(len_train))[:subsample_train_size]
        test_idx = np.random.permutation(range(len_test))[:subsample_test_size]
        
        sentences1_train = list(df_train['#1 String'].iloc[train_idx])
        sentences2_train = list(df_train['#2 String'].iloc[train_idx])
        labels_train = list(df_train['Quality'].iloc[train_idx])
        sentences1_test = list(df_test['#1 String'].iloc[test_idx])
        sentences2_test = list(df_test['#2 String'].iloc[test_idx])
        labels_test = list(df_test['Quality'].iloc[test_idx])
        
        train_dataset = double_sentences_set(sentences1_train, sentences2_train, labels_train, tokenizer, max_length)
        test_dataset = double_sentences_set(sentences1_test, sentences2_test, labels_test, tokenizer, max_length)
        
    elif dataset_name == 'QNLI':
        
        df_train = pd.read_csv(TRAIN_PATH, sep = '\t', error_bad_lines=False, index_col = 'index')
        df_train.dropna(inplace = True)
        df_test = pd.read_csv(TEST_PATH, sep = '\t', error_bad_lines=False, index_col = 'index')
        df_test.dropna(inplace = True)
        
        len_train = len(df_train)
        len_test = len(df_test)
        
        train_idx = np.random.permutation(range(len_train))[:subsample_train_size]
        test_idx = np.random.permutation(range(len_test))[:subsample_test_size]
        
        sentences1_train = list(df_train['question'].iloc[train_idx])
        sentences2_train = list(df_train['sentence'].iloc[train_idx])
        labels_train = list(df_train['label'].apply(lambda x: 0 if 'not' in x else 1).iloc[train_idx])
        sentences1_test = list(df_test['question'].iloc[test_idx])
        sentences2_test = list(df_test['sentence'].iloc[test_idx])
        labels_test = list(df_test['label'].apply(lambda x: 0 if 'not' in x else 1).iloc[test_idx])

        train_dataset = double_sentences_set(sentences1_train, sentences2_train, labels_train, tokenizer, max_length)
        test_dataset = double_sentences_set(sentences1_test, sentences2_test, labels_test, tokenizer, max_length)
        
    elif dataset_name == 'QQP':
        df_train = pd.read_csv(TRAIN_PATH, sep = '\t', error_bad_lines=False)
        df_train.dropna(inplace = True)
        df_test = pd.read_csv(TEST_PATH, sep = '\t', error_bad_lines=False)
        df_test.dropna(inplace = True)
        
        len_train = len(df_train)
        len_test = len(df_test)
        
        train_idx = np.random.permutation(range(len_train))[:subsample_train_size]
        test_idx = np.random.permutation(range(len_test))[:subsample_test_size]
        
        sentences1_train = list(df_train['question1'].iloc[train_idx])
        sentences2_train = list(df_train['question2'].iloc[train_idx])
        labels_train = list(df_train['is_duplicate'].iloc[train_idx])
        sentences1_test = list(df_test['question1'].iloc[test_idx])
        sentences2_test = list(df_test['question2'].iloc[test_idx])
        labels_test = list(df_test['is_duplicate'].iloc[test_idx])

        train_dataset = double_sentences_set(sentences1_train, sentences2_train, labels_train, tokenizer, max_length)
        test_dataset = double_sentences_set(sentences1_test, sentences2_test, labels_test, tokenizer, max_length)
        
    elif dataset_name == 'RTE':
        df_train = pd.read_csv(TRAIN_PATH, sep = '\t', error_bad_lines=False, index_col = 'index')
        df_train.dropna(inplace = True)
        df_test = pd.read_csv(TEST_PATH, sep = '\t', error_bad_lines=False, index_col = 'index')
        df_test.dropna(inplace = True)
        
        len_train = len(df_train)
        len_test = len(df_test)
        
        train_idx = np.random.permutation(range(len_train))[:subsample_train_size]
        test_idx = np.random.permutation(range(len_test))[:subsample_test_size]
        
        sentences1_train = list(df_train['sentence1'].iloc[train_idx])
        sentences2_train = list(df_train['sentence2'].iloc[train_idx])
        labels_train = list(df_train['label'].apply(lambda x: 0 if 'not' in x else 1).iloc[train_idx])
        sentences1_test = list(df_test['sentence1'].iloc[test_idx])
        sentences2_test = list(df_test['sentence2'].iloc[test_idx])
        labels_test = list(df_test['label'].apply(lambda x: 0 if 'not' in x else 1).iloc[test_idx])

        train_dataset = double_sentences_set(sentences1_train, sentences2_train, labels_train, tokenizer, max_length)
        test_dataset = double_sentences_set(sentences1_test, sentences2_test, labels_test, tokenizer, max_length)
        
    train_size = int(train_frac * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    [train_dataset, val_dataset] = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
    return train_dataset, val_dataset, test_dataset

def get_dataloader(dataset_name, batch_size, tokenizer, TRAIN_PATH, TEST_PATH, train_frac = 0.8,
                   subsample_train_size = None, subsample_test_size = None, max_length = None):
    """
    Returns the dataloaders of the dataset, with the appropriate collate function
    Parameters:
        -dataset_name (str): name of the dataset to load, can be 'CoLA', 'SST-2', 'MRPC', 'QNLI', 'QQP', 'RTE'.
        -batch_size (int): batch_size to use in the dataloader
        -tokenizer: tokenizer to encode the sentences
        -TRAIN_PATH (str)
        -TEST_PATH (str)
        -subsample_train_size (int): if not None, the training dataloader will only contain subsample_train_size samples
        -subsample_val_size (int): if not None, the validation dataloader will only contain subsample_val_size samples
        -max_length (int): maximum length of the sentences to encode (character-wise, e.g len('transformers') == 12)
    """
    
    train_dataset, val_dataset, test_dataset = load_dataset(dataset_name, tokenizer, TRAIN_PATH, TEST_PATH,
                                                            train_frac, subsample_train_size,
                                                            subsample_test_size, max_length)
    
    def collate(data):
    
        batch_encoding = {}

        for key in data[0][0].keys():
            batch_encoding[key] = [x[0][key] for x in data]

        labels = [x[1] for x in data]

        return (tokenizer.pad(batch_encoding, padding = True, return_tensors = 'pt'), torch.tensor(labels))
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle = True, collate_fn = collate)
    val_loader = DataLoader(val_dataset, batch_size, shuffle = True, collate_fn = collate)
    test_loader = DataLoader(test_dataset, batch_size, shuffle = True, collate_fn = collate)
    
    return train_loader, val_loader, test_loader