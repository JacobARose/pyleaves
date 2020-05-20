# @Author: Jacob A Rose
# @Date:   Tue, March 31st 2020, 12:34 am
# @Email:  jacobrose@brown.edu
# @Filename: base_dataset.py


'''
Script for defining base class BaseDataset for managing information about a particular subset or collection of datasets during preparation for a particular experiment.

'''
from boltons.dictutils import OneToOne
from collections import OrderedDict
import dataset
import json
import numpy as np
import os
import pandas as pd
import random
from stuf import stuf
from toolz.itertoolz import frequencies
from pyleaves import leavesdb
import pyleaves





class BaseDataset(object):

    __version__ = '1.1'

    def __init__(self, name='', src_db=pyleaves.DATABASE_PATH):
        """
        Base class meant to be subclassed for unique named datasets. Implements some property setters/getters for maintaining consistency
        of data and filters (like min class count threshold).


        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>> dataset = BaseDataset()

        >>> leaves_dataset = LeavesDataset()
        ... fossil_dataset = FossilDataset()
        ... pnas_dataset = PNASDataset()
        ... pnas_fossil_data = pnas_data+fossil_data
        ... pnas_leaves_data = pnas_data+leaves_data


        >>>

        """
        self.name = name
        self.columns = ['path','family']
        if src_db:
            self.local_db = leavesdb.init_local_db(src_db = src_db, verbose=False)
        self._threshold = 0
        self._data = pd.DataFrame(columns=self.columns)

    def load_from_db(self, x_col='path', y_col='family', all_cols=False):
        """
        Load a dataframe from the SQLite db with 2 columns, paths and labels.
        Subclasses should use this function in their __init__ method to instantiate self._data

        -set all_cols=True in order to ignore x_col and y_col and instead load all columns in table

        Returns
        -------
        pd.DataFrame
            Description of returned object.

        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

        """
        db = dataset.connect(f"sqlite:///{self.local_db}", row_type=stuf)
        if all_cols:
            data = pd.DataFrame(db['dataset'].all())
        else:
            data = pd.DataFrame(leavesdb.db_query.load_data(db=db, x_col=x_col, y_col=y_col, dataset=self.name))
        return data

    def load_from_csv(self, filepath):
        """Load a dataframe from a CSV file with 2 columns, paths and labels.

        Returns
        -------
        pd.DataFrame
            Description of returned object.

        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

        """
        data = pd.read_csv(filepath, drop_index=True)
        return data

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name='', threshold=0):
        new_dataset = cls(name=name, src_db=None)
        new_dataset._threshold = threshold
        new_dataset.data = df
        return new_dataset

    def exclude_rare_classes(self, threshold):
        """
        Uses helper function filter_low_count_labels to keep only classes with the number of samples equal to or greater than threshold.

        Updates the self._data dataframe in place

        Parameters
        ----------
        threshold : int
            Keep classes with num_samples >= threshold

        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

        """
        self._data = pyleaves.data_pipeline.preprocessing.filter_low_count_labels(self.data,threshold,'family',verbose=False)
        self._threshold = threshold

    def merge_with(self, other):
        #TODO: combine this with __add__ method, since it's just a wrapper
        assert issubclass(type(other), BaseDataset)

        merged_dataset = BaseDataset()
        #Keep highest threshold between the 2 instances
        merged_dataset._threshold = max([self.threshold,other.threshold])
        #Use the Base class setter method for self.data to concatenate the 2 instances' dataframes then performing a round of
        #filtering out duplicates and class thresholding
        merged_dataset.data = pd.concat({self.name:self.data,
                                other.name:other.data})
        merged_dataset.name = '+'.join([self.name, other.name])

        return merged_dataset

    def __add__(self, other):
        return self.merge_with(other)

    def __eq__(self, other):
        if self.name != other.name:
            return False
        elif self.threshold != other.threshold:
            return False
        elif not np.all(self.data == other.data):
            return False
        elif not np.all(self.classes == other.classes):
            return False
        return True

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data: pd.DataFrame):
        # import pdb; pdb.set_trace()
        self._data = new_data.drop_duplicates(subset='path')
        self.exclude_rare_classes(self.threshold)
        if len(self._data) != len(new_data):
            print(f'dropped {len(new_data)-len(self._data)} duplicate rows')

    @property
    def class_counts(self):
        '''
        Returns
        -------
        dict
            mapping {class_name:class_count} values
        '''
        y_col = self.columns[1]
        return frequencies(self.data[y_col])

    @property
    def classes(self):
        '''
        Returns
        -------
        list
            Sorted list of class names
        '''
        return sorted(self.class_counts.keys())
        # return pyleaves.data_pipeline.preprocessing.get_class_counts(self.data,'family', verbose=False)[0]

    @property
    def num_samples(self):
        return len(self.data)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def threshold(self):
        return self._threshold

    def __repr__(self):
        return f'''{self.name}:
    num_samples: {self.num_samples}
    num_classes: {self.num_classes}
    class_count_threshold: {self.threshold}
        '''


    def select_data_by_source_dataset(self, source_name):
        """
        Returns a pd.DataFrame containing rows from data that originate from the dataset indicated by source_name.

        data must be the result of at least one addition of 2 or more datasets.
        e.g. pnas_dataset + leaves_dataset.
        The output of this addition results in a new dataframe with a multiIndex, where index level 0 is the
        dataset source name and index level 1 is the row number within the original dataset

        Parameters
        ----------
        data : pd.DataFrame
            Should be extracted from a subclass of BaseDataset, by accessing the .data property, after combining
            2 or more datasets
        source_name : str
            Should refer to one of the 2 or more datasets used to construct data

        Returns
        -------
        pd.DataFrame
            Contains only rows belonging to source_name's dataset, same columns and index levels as self.data previously

        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

        """
        idx = np.where(self.data.index.get_level_values(0)==source_name)[0]

        return self.data.iloc[idx,:]

    def leave_one_class_out(self, class_name: str):
        """
        LEAVE-ONE-OUT EXPERIMENT helper function

        Returns a tuple with length==2. The first item is a DataFrame where every row comes from self.data, but does not
        belong to the class indicated by class_name. The second item is a DataFrame containing all the rows that do
        belong to class_name.

        Parameters
        ----------
        class_name : str
            The class to be separated out

        Returns
        -------
        tuple(pd.DataFrame, pd.DataFrame)
            tuple corresponding to (included classes, excluded class)

        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

        """
        label_col = self.columns[1]

        include = self.data[self.data[label_col]!=class_name]
        exclude = self.data[self.data[label_col]==class_name]

        return (BaseDataset.from_dataframe(include, threshold=self.threshold),
                BaseDataset.from_dataframe(exclude, threshold=self.threshold))

        # assert include.shape[0]+exclude.shape[0]==self.data.shape[0]
        # return (include, exclude)

    def enforce_class_whitelist(self, class_names: list):
        """
        Similar task as leave_one_class_out, but opposite approach. User provides a list of classes to include, while
        the rest are excluded.

        Useful for limiting a dataset to only classes that exist in another

        Parameters
        ----------
        class_names : list
            The classes to be kept

        Returns
        -------
        tuple(pd.DataFrame, pd.DataFrame)
            tuple corresponding to (included classes, excluded class)

        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

        """

        label_col = self.columns[1]

        idx = self.data[label_col].isin(class_names)

        include = self.data[idx]
        exclude = self.data[~idx]

        return (BaseDataset.from_dataframe(include, threshold=self.threshold),
                BaseDataset.from_dataframe(exclude, threshold=self.threshold))

        # assert include.shape[0]+exclude.shape[0]==self.data.shape[0]
        # return (include, exclude)


#############################################################################################




class LabelEncoder:

    fname = 'label_encoder.json'

    def __init__(self, labels):
        self.classes = tuple(np.unique(sorted(labels)))
        self._encoder = OneToOne(enumerate(self.classes)).inv
        self.fname = 'label_encoder.json'

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def encoder(self):
        return self._encoder

    @encoder.setter
    def encoder(self, data):
        if type(data)==list:
            self._encoder = OneToOne(enumerate(data)).inv
        elif type(data) in [dict, OrderedDict]:
            self._encoder = OneToOne(data)
        else:
            assert False


    @property
    def decoder(self):
        return self.encoder.inv


    def encode(self, labels):
        '''str->int'''
        return [self.encoder[l] for l in list(labels)]

    def decode(self, labels):
        '''int->str'''
        return [self.decoder[l] for l in labels]


    @classmethod
    def load_config(cls, label_dir):
        with open(os.path.join(label_dir, cls.fname), 'r') as file:
            data = json.load(file)
        # cls(**data)
        loaded = cls(list(data.keys()))
        loaded.encoder = data
        return loaded

    def save_config(self, out_dir):
        with open(os.path.join(out_dir, self.fname), 'w') as file:
            json.dump(self.encoder, file)






def partition_data(data, partitions=OrderedDict({'train':0.5,'test':0.5})):
    '''
    Split data into named partitions by fraction

    Example:
    --------
    #split_data will be a dict with the same keys as partitions, and the values will be the corresponding samples from data.
    #i.e. 'train' will get the first 40% of samples, 'val' the next 10%, and 'test' the last 50%

    >> split_data = partition_data(data, partitions=OrderedDict({'train':0.4,'val':0.1,'test':0.5}))
    '''
    num_rows = len(data)
    output={}
    taken = 0.0
    for k,v in partitions.items():
        idx = (int(taken*num_rows),int((taken+v)*num_rows))
        print(k, v, idx)
        output.update({k:data[idx[0]:idx[1]]})
        taken+=v
    assert taken <= 1.0
    return output

def preprocess_data(dataset, encoder, config):
    """
    Function to perform 4 preprocessing steps:
        1. Exclude classes below minimum threshold defined in config.threshold
        2. Exclude all classes that are not referenced in encoder.classes
        3. Encode and normalize data into (path: str, label: int) tuples
        4. Partition data samples into fractional splits defined in config.data_splits_meta

    Parameters
    ----------
    dataset : BaseDataset
        Any instance of BaseDataset or its subclasses
    encoder : LabelEncoder
        Description of parameter `encoder`.
    config : Namespace or stuf.stuf
        Config object containing the attributes/properties:
            config.threshold
            config.data_splits_meta

    Returns
    -------
    dict
        Dictionary mapping from keys defined in config.data_splits_meta.keys(), to lists of tuples representing each sample.

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>> dataset = LeavesDataset()
    ... encoder = LabelEncoder(dataset.data.family)
    ... data_splits = preprocess_data(dataset, encoder, config)

    """

    dataset.exclude_rare_classes(threshold=config.threshold)

    dataset, _ = dataset.enforce_class_whitelist(class_names=encoder.classes)

    x = list(dataset.data['path'].values)#.reshape((-1,1))
    y = np.array(encoder.encode(dataset.data['family']))

    # import pdb;pdb.set_trace()
    shuffled_data = list(zip(x,y))
    random.shuffle(shuffled_data)

    return partition_data(data=shuffled_data,
                          partitions=OrderedDict(config.data_splits_meta)
                          )

def calculate_class_counts(y_data : list):
    labels, label_counts = np.unique(y_data, return_counts=True)
    if type(labels[0])!=str:
        labels = [int(label) for label in labels]
    label_counts = [int(count) for count in label_counts]
    return {label: count for label,count in zip(labels, label_counts)}

def calculate_class_weights(y_data : list):
    """
    Calculate class weights as w[i] = <total # of samples>/(<total # of classes>*<class[i] count>)

    Parameters
    ----------
    y_data : list
        List of y labels to be counted per class for calculating class weights

    Returns
    -------
    dict
        Contains key:value pairs corresponding to unique class labels: corresponding weights
        e.g. {0:1.0,
              1:2.344,
              2:5.456}

    """

    # labels, label_counts = np.unique(y_data, return_counts=True)

    class_counts_dict = calculate_class_counts(y_data)

    total = sum(class_counts_dict.values())
    num_classes = len(class_counts_dict)

    calc_weight = lambda count: total / (num_classes * count)

    class_weights = {label:calc_weight(count) for label, count in class_counts_dict.items()}

    # class_weights = {k: v/np.min(list(class_weights.values())) for k,v in class_weights.items()}
    # class_weights = {k: v/np.max(list(class_weights.values())) for k,v in class_weights.items()}
    return class_weights
    # total = sum(label_counts)
    # num_classes = len(labels)

    # class_weights = {}
    # for label, c in zip(labels,label_counts):
    #     if type(label) != str:
    #         label = int(label)
    #     class_weights[label] = total / (num_classes * c)
    # return class_weights
