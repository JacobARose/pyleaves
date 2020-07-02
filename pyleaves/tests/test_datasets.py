# @Author: Jacob A Rose
# @Date:   Wed, April 8th 2020, 9:11 pm
# @Email:  jacobrose@brown.edu
# @Filename: test_datasets.py


'''
Functions for testing the valid functionality of classes and functions defined within pyleaves/datasets submodule.
'''

from stuf import stuf

from typing import List, Dict

from pyleaves.datasets import base_dataset, leaves_dataset, fossil_dataset, pnas_dataset


# TODO add tests for all BaseDataset methods. Particularly including creation, alteration, and combination methods





# metadata_source = {
#                     'Leaves':
#
# }


def test_base_dataset():

    # dataset = base_dataset.BaseDataset()


    fossil_data = fossil_dataset.FossilDataset()
    leaves_data = leaves_dataset.LeavesDataset()

    added_data = (leaves_data+fossil_data)
    for class_name in added_data.classes:
        include, exclude = added_data.leave_one_class_out(class_name)
        assert include.shape[0]+exclude.shape[0]==added_data.data.shape[0]

def test_leaves_dataset():

    dataset = leaves_dataset.LeavesDataset()

    assert dataset.name = "Leaves"
    assert dataset.num_samples == 26953
    assert dataset.num_classes == 376
    assert dataset._threshold == 0
