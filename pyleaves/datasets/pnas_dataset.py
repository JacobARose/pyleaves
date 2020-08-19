# @Author: Jacob A Rose
# @Date:   Wed, April 8th 2020, 9:18 pm
# @Email:  jacobrose@brown.edu
# @Filename: pnas_dataset.py


'''
Script for defining base class BaseDataset for managing information about a particular subset or collection of datasets during preparation for a particular experiment.

'''
import pyleaves
from pyleaves import leavesdb
from pyleaves.datasets.base_dataset import BaseDataset
from pyleaves.tests.test_utils import MetaData


class PNASDataset(BaseDataset):

    def __init__(self, src_db=pyleaves.DATABASE_PATH, all_cols=False):
        super().__init__(name='PNAS', src_db=src_db)
        self._data = self.load_from_db(all_cols=all_cols)
        self.__class__.metadata = METADATA



METADATA =  MetaData(
            name="PNAS",
            num_samples=5314,
            num_classes=19,
            threshold=0,
            class_distribution={
'Anacardiaceae': 123,
'Annonaceae': 350,
'Apocynaceae': 244,
'Betulaceae': 138,
'Celastraceae': 130,
'Combretaceae': 191,
'Ericaceae': 184,
'Fabaceae': 776,
'Fagaceae': 184,
'Lauraceae': 311,
'Malvaceae': 192,
'Melastomataceae': 193,
'Myrtaceae': 434,
'Passifloraceae': 134,
'Phyllanthaceae': 365,
'Rosaceae': 207,
'Rubiaceae': 449,
'Salicaceae': 455,
'Sapindaceae': 254
}
)
