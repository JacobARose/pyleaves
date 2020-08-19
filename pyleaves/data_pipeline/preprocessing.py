'''
Functions for preprocessing/transforming data between extraction from the database and input to the model.
'''
from boltons.dictutils import OneToOne
import copy
import numpy as np
import pandas as pd
import dataset
import os
import json
from pyleaves.utils import ensure_dir_exists


def encode_labels_str2int(data, y_col='family'):
    '''
    Create 'label' column in data_df that features integer values corresponding to text labels contained in y_col.

    Arguments:
        data: dataset.util.ResultIter, Should be the returned result from loading data from the leavesdb database (e.g. data = leavesdb.db_query.load_data(db)).
        y_col: str, name of the columns containing text labels for each sample in data.
    Returns:
        data_df: pd.DataFrame, Contains 3 columns, one for paths, one for str labels, and one for int labels.
    '''
    data = data.sort_values(by=y_col) #pd.DataFrame(data)
    data['label'] = pd.Categorical(data[y_col])
    data['label'] = data['label'].cat.codes
    return data

encode_labels = encode_labels_str2int #For backwards compatibility

def generate_encoding_map(data, text_label_col='family', int_label_col='label'):
    '''
    Returns a dictionary mapping integer labels to their corresponding text label
    {0:'Annonaceae',
    ...
    19:'Passifloraceae'}
    '''
    #Review the below TODO, may be unecessary because similar process is used in encode_labels_str2int
    #TODO Potentially filter for unique text_label_col values instead, ensures the int representation remains the same in the case of changes.
    int_labels, label_indices = np.unique(data[int_label_col], return_index=True)
    text_labels = data[text_label_col].iloc[label_indices].values

    return {int_label:text_label for int_label, text_label in zip(int_labels, text_labels)}


def load_label_encodings_from_file(filepath):
#     import pdb; pdb.set_trace()
    assert os.path.isfile(filepath)
    if filepath.endswith('json'):
        with open(filepath, 'r') as file:
            data = json.load(file)

    elif filepath.endswith('csv'):
        data = pd.read_csv(filepath, index_col=0).squeeze().to_dict()
#         data = data.rename(columns={'Unnamed: 0':'text_label','0':'int_label'})
    return dict(data)

def save_label_encodings_to_file(encoding_dict, filepath):
    base_dir = os.path.dirname(filepath)
    ensure_dir_exists(base_dir)
    if filepath.endswith('json'):
        with open(filepath, 'w') as file:
            json.dump(encoding_dict, file)

    elif filepath.endswith('csv'):
        data = pd.DataFrame(list(encoding_dict.values()),index=list(encoding_dict.keys()))
        data.to_csv(filepath)


class LabelEncoder:

    def __init__(self, labels=[], reserved_mappings = {}, filepath=None):
        '''
        Arguments:
            labels=[], list(strings):
                A list of potentially non-unique strings representing categorical labels
            reserved_mappings={}, dict({str:int}):
                a dictionary mapping of text to integer numbers
            filepath

        '''
        self.num_classes=0
        self._encodings = OneToOne()
        if len(reserved_mappings)>0:
            reserved_mappings = list(reserved_mappings)
            self.merge_labels(reserved_mappings)
        if len(labels)>0:
            self.merge_labels(labels)
        if filepath is not None:
#             if len(self)>0:
            self.merge_labels(self.load_labels(filepath))



    def filter(self, data_df, text_label_col='family', int_label_col=None):
        '''
        Filter a dataframe to include only rows corresponding to labels in the encoder. Useful for preprocessing a target domain dataset for a model trained on source domain labels.
        '''
        int_whitelist= list(self.get_encodings().inv)
        text_whitelist=list(self.get_encodings())
        if int_label_col:
            data = data_df[data_df[int_label_col].isin(int_whitelist)]
        else:
            data = data_df[data_df[text_label_col].isin(text_whitelist)]
        return data

    def transform(self, labels):
        return [self._encodings[l] for l in list(labels)]

    def inv_transform(self, encoded_labels):
        return [self._encodings.inv[l] for l in list(encoded_labels)]

    def merge_labels(self,labels=[]):
        '''
        Labels can be list, or a dict where the keys are str
        Iterates through labels or unique values that dont already exist in encoder.
        '''
        labels = list(labels)
        for l in np.unique(labels):
            if l not in self._encodings.keys():
                self._encodings.update({l:self.num_classes})
                self.num_classes += 1

    def load_labels(self, filepath):
        return load_label_encodings_from_file(filepath)

    def save_labels(self, filepath):
        save_label_encodings_to_file(self.get_encodings(), filepath)

    def get_encodings(self):
        return copy.deepcopy(self._encodings)

    def __len__(self):
        return len(self.get_encodings())

    def __repr__(self):
        return json.dumps(self.get_encodings(), indent=2)



def load_all_data(db, x_col='path', y_col='family'):
    '''
    Function to load x_col and y_col for each row in db from all datasets
    '''
#     paths_labels = list(db['dataset'].distinct(x_col, y_col, 'dataset'))
    data = pd.DataFrame(db['dataset'].distinct(x_col, y_col, 'dataset'))
    return data
















def one_hot_encode_labels(labels):
    '''
    Arguments:
        labels, list(int): list of labels encoded in scalar integers

    Returns:
        encoded_labels, np.array: numpy array with shape = (num_samples, num_classes) and elements of value 0 or 1.
    '''
    num_samples = len(labels)
    num_classes = np.max(labels)+1

    encoded_labels = np.zeros((num_samples, num_classes))

    for i, label in enumerate(labels):
        encoded_labels[i,label] = 1

    return encoded_labels

def one_hot_decode_labels(one_hot_labels):
    '''
    Arguments:
        one_hot_labels, np.array: one_hot_encoded labels with features on axis=1 and samples on axis=0
    Returns:
        np.array: shape=(num_samples,1), integer values indicating label
    '''

    return np.argmax(one_hot_labels, axis=1).reshape(-1,1)


def get_class_counts(data_df, y_col='label', verbose=True):
    labels, label_counts = np.unique(data_df[y_col], return_counts=True)
    if verbose:
        print('label : count')
        for label, count in zip(labels, label_counts):
            print(label,' : ', count)
    return labels, label_counts

def filter_low_count_labels(data_df, threshold=2, y_col='family', verbose = True):
    '''
    Function for omitting samples that belong to a class with a population size below the threshold. Used primarily for omitting classes with only 1 sample.
    '''
    text_labels, label_counts = np.unique(data_df[y_col], return_counts=True)
    filtered_label_idx = np.where(label_counts >= threshold)[0]
    filtered_text_labels = text_labels[filtered_label_idx]
    filtered_data = data_df[data_df[y_col].isin(filtered_text_labels)]
    if verbose:
        print(f'Selecting only samples that belong to a class with population >= {threshold} samples')
        print(f'Previous num_classes = {len(label_counts)}, new num_classes = {len(filtered_text_labels)}')
        print(f'Previous data_df.shape = {data_df.shape}, new data_df.shape = {filtered_data.shape}')
    return filtered_data


#     data_df = encode_labels(data_df, y_col=y_col)
#     labels, label_counts = np.unique(data_df['label'], return_counts=True)
#     filtered_labels = np.where(label_counts >= threshold)[0]
#     filtered_data = data_df[data_df['label'].isin(filtered_labels)]
#     if verbose:
#         print(f"filter_low_count_labels(data_df, threshold={threshold}, y_col={y_col}, verbose = {verbose})")
#         print(f'Selecting only samples that belong to a class with population >= {threshold} samples')
#         print(f'Previous num_classes = {len(label_counts)}, new num_classes = {len(filtered_labels)}')
#         print(f'Previous data_df.shape = {data_df.shape}, new data_df.shape = {filtered_data.shape}')
#     return filtered_data
