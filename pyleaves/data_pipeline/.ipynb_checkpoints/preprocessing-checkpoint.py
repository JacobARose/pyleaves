'''
Functions for preprocessing/transforming data between extraction from the database and input to the model.
'''

import numpy as np
import pandas as pd
import dataset



def encode_labels_str2int(data, y_col='family'):
    '''
    Create 'label' column in data_df that features integer values corresponding to text labels contained in y_col.

    Arguments:
        data: dataset.util.ResultIter, Should be the returned result from loading data from the leavesdb database (e.g. data = leavesdb.db_query.load_data(db)).
        y_col: str, name of the columns containing text labels for each sample in data.
    Returns:
        data_df: pd.DataFrame, Contains 3 columns, one for paths, one for str labels, and one for int labels.
    '''
    data = pd.DataFrame(data)
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
    int_labels, label_indices = np.unique(data[int_label_col], return_index=True)
    text_labels = data[text_label_col].iloc[label_indices].values

    return {int_label:text_label for int_label, text_label in zip(int_labels, text_labels)}



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


def get_class_counts(data_df, verbose=True):
    labels, label_counts = np.unique(data_df['label'], return_counts=True)
    if verbose:
        print('label : count')
        for label, count in zip(labels, label_counts):
            print(label,' : ', count)
    return labels, label_counts

def filter_low_count_labels(data_df, threshold=2, y_col='family', verbose = True):
    '''
    Function for omitting samples that belong to a class with a population size below the threshold. Used primarily for omitting classes with only 1 sample.
    '''
    data_df = encode_labels(data_df, y_col=y_col)
    labels, label_counts = np.unique(data_df['label'], return_counts=True)
    filtered_labels = np.where(label_counts >= threshold)[0]
    filtered_data = data_df[data_df['label'].isin(filtered_labels)]
    if verbose:
        print(f'Selecting only samples that belong to a class with population >= {threshold} samples')
        print(f'Previous num_classes = {len(label_counts)}, new num_classes = {len(filtered_labels)}')
        print(f'Previous data_df.shape = {data_df.shape}, new data_df.shape = {filtered_data.shape}')
    return filtered_data
