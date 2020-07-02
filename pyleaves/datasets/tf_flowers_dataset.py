# @Author: Jacob A Rose
# @Date:   Wed, April 8th 2020, 9:17 pm
# @Email:  jacobrose@brown.edu
# @Filename: fossil_dataset.py


'''
Script for defining base class BaseDataset for managing information about a particular subset or collection of datasets during preparation for a particular experiment.

'''
import pyleaves
from pyleaves import leavesdb
from pyleaves.datasets.base_dataset import BaseDataset
from pyleaves.tests.test_utils import MetaData



class TFFlowersDataset(BaseDataset):

    def __init__(self, src_db=None):
        super().__init__(name='tf_flowers', src_db=src_db)
        self._data = self.load_from_db()
        self.__class__.metadata = METADATA


METADATA =  MetaData(
            name="Fossil",
            num_samples=6122,
            num_classes=27,
            threshold=0,
            class_distribution={
'Adoxaceae': 33,
'Anacardiaceae': 229,
'Araceae': 2,
'Araliaceae': 4,
'Berberidaceae': 26,
'Betulaceae': 74,
'Cupressaceae': 254,
'Dryopteridaceae': 64,
'Fabaceae': 144,
'Fagaceae': 746,
'Grossulariaceae': 3,
'Hydrangeaceae': 1,
'II. IDs, families uncertain': 943,
'Juglandaceae': 63,
'Lauraceae': 6,
'Meliaceae': 37,
'Myrtaceae': 21,
'New_Fossil_Dataset': 1,
'Pinaceae': 66,
'Rhamnaceae': 12,
'Rosaceae': 429,
'Salicaceae': 159,
'Sapindaceae': 234,
'Taxaceae': 9,
'Ulmaceae': 1048,
'Unidentified': 1492,
'Vitaceae': 22
}
)
