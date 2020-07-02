# @Author: Jacob A Rose
# @Date:   Tue, March 31st 2020, 12:34 am
# @Email:  jacobrose@brown.edu
# @Filename: __init__.py


from os.path import dirname, join
#Location of repo database resource files
RESOURCES_DIR = join(dirname(__file__),'leavesdb','resources')
#Full dir+filename of SQLite database db file
DATABASE_PATH_v1 = join(RESOURCES_DIR,'leavesdb.db')
DATABASE_PATH = DATABASE_PATH_v1 #join(RESOURCES_DIR,'leavesdb-v1_1.db')
EXPERIMENTS_DB = join(RESOURCES_DIR,'experiments.db')

# def test_scope():
#     import pyleaves as pl
#     print(dir(pl))
#     type(pl)
# print('root init')
# test_scope()


from .configs import config
from . import analysis
from . import base
from . import configs

from . import datasets
from . import data_pipeline
from . import leavesdb
from . import models
# import pdb;pdb.set_trace()
from . import utils
