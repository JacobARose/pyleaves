# @Author: Jacob A Rose
# @Date:   Tue, March 31st 2020, 12:36 am
# @Email:  jacobrose@brown.edu
# @Filename: utils.py


'''

'''

# import pdb;pdb.set_trace();print(__file__)
from distutils.version import StrictVersion
import numpy as np
import os
import itertools
from collections import defaultdict, OrderedDict
import gpustat
from pprint import pprint
import random
from typing import List

def setGPU(only_return=False):
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
    pairs = list(zip(ids, ratios))
    random.shuffle(pairs)
    bestGPU = min(pairs, key=lambda x: x[1])[0]

    print(f'setGPU: GPU:memory ratios initially visible to setGPU:')
    pprint(pairs)
    if only_return:
        return bestGPU
    print(f"setGPU: Setting GPU to: {bestGPU}")
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)
    return bestGPU


def set_tf_config(seed: int=None):
    import tensorflow as tf
    assert using_tensorflow2()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # import pdb; pdb.set_trace()
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        if gpus:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except:
        print('setting memory growth failed, continuing anyway.')

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def using_tensorflow2() -> bool:
    import tensorflow as tf
    return StrictVersion(tf.__version__).version >= StrictVersion('2.0.0').version



# def create_session(target='', gpus='0,1,2,3', timeout_sec=10):
#     '''Create an intractive TensorFlow session.
#     Helper function that creates TF session that uses growing GPU memory
#     allocation and opration timeout. 'allow_growth' flag prevents TF
#     from allocating the whole GPU memory an once, which is useful
#     when having multiple python sessions sharing the same GPU.
#     '''
#     try:
#         import tensorflow as tf
#     except:
#         pass
#     tf.reset_default_graph()
#     # graph = tf.Graph()
#     config = tf.ConfigProto()

#     # config.gpu_options.visible_device_list= gpus
#     config.gpu_options.allow_growth = True
#     config.gpu_options.per_process_gpu_memory_fraction = 0.9
#     config.operation_timeout_in_ms = int(timeout_sec*1000)
#     return tf.Session(target=target, graph=tf.get_default_graph(), config=config)
#     # return tf.InteractiveSession(target=target, graph=graph, config=config)



#####################################################

def ensure_dir_exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    if os.path.isdir(dir_path):
        return True
    else:
        return False

def set_random_seed(seed_value: int):
    import random
    import numpy as np
    import tensorflow as tf
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)


def get_visible_devices(device_type='GPU'):
    '''
    device_type can be either 'GPU' or 'CPU'
    '''
    # from tensorflow.config import experimental_list_devices
    # gpus = [dev for dev in experimental_list_devices() if 'GPU:' in dev]
    # return gpus
    pass

def set_visible_gpus(gpu_ids=[0]):
    print('DEPRECATED FUNCTION set_visible_gpus()')
    return None
    # from tensorflow.config import experimental, experimental_list_devices
#     import pdb; pdb.set_trace()
#     gpus = experimental.get_visible_devices('GPU')
    # gpus = [dev for dev in experimental_list_devices() if 'GPU:' in dev]

    # if None:#gpus:
#         print(gpus)
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids).strip('[]')
#         gpu_objects = [gpus[i] for i in gpu_ids]

#         experimental.set_memory_growth(*gpu_objects, True)
#         experimental.set_visible_devices(gpu_objects, 'GPU')
#         experimental.set_memory_growth(*gpu_objects, True)

        # logical_gpus = experimental.get_visible_devices('GPU')
#         experimental.set_memory_growth(*logical_gpus, True)
    #     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    #     print(logical_gpus)
    # else:
    #     print("No visible GPUs found")


##############################################################
## HYPERPARAMETER TUNING UTILS


def process_hparam_arg(arg):
    if type(arg)!=list:
        arg = [arg]
    return arg


def process_hparam_args(args = {}, search_params=[]):

    for k, v in args.__dict__.items():
        if k in search_params or search_params==[]:
            args.__dict__[k] = process_hparam_arg(v)
    return args








class HyperParameters(defaultdict):

    def __init__(self):

        self.search_params=['model_name', 'source_datasets','target_datasets', 'base_learning_rate','batch_size']
        self.all = OrderedDict()
        self.all['model_name'] = ['resnet_50_v2', 'vgg16']
        self.all['dataset_name'] = ['PNAS','Leaves','Fossil']
        self.all['source_datasets'] = ['PNAS+Leaves']
        self.all['target_datasets'] = ['Fossil']


        self.all['base_learning_rate'] = [1e-4, 1e-5]
        self.all['batch_size'] = [64]
        self._a = defaultdict(list)

    def parse_arg(self, k, v):
        if v == 'all':
            return self.all[k]
        if type(v)!=list:
            return [v]
        else:
            return v

    def parse_args(self, args):
        parsed = {}
        for key in self.search_params:
            if key in args.__dict__.keys():
                parsed[key] = self.parse_arg(key, args.__dict__[key])
        return parsed

    def initialize_iterator(self, args, shuffle=False):
        '''
        Function returns an iterator that iterates through every permutation of hyperparameters
        '''
#         args.run_names = ['_'.join(names) for names in args.dataset_names]
        #########################################
#         regularizer = {args.regularizations:args.r_params}
#         self = HyperParameters()

        hparams = self.parse_args(args)

#         hparams = []
#         for p in [list(itertools.product([k],v)) for k, v in parsed_args]:
#             print(p)
# #             hparams.extend(p)
#             hparams.append(p)
#         hparam_sampler = list(itertools.product(hparams))


        hparams_labeled = OrderedDict()
        for k, v in hparams.items():
            hparams_labeled[k] = list(itertools.product([k],v))

        hparam_sampler = list(
                itertools.product(
                                *list(
                                    hparams_labeled.values()
                                    )
                                )
                            )



#         hparam_sampler = list(
#                 itertools.product(*list(hparams.values()))
#         )
        import random
        print('Initializing HParam search through a total of ', len(hparam_sampler),'individual permutations.')
        print('#'*20)
        print('#'*20)
        if shuffle:
            random.shuffle(hparam_sampler)

        return hparam_sampler




















def __check_if_hdf5(path=''):
    _ext = ['h5','hdf5']

    for e in _ext:
        if path.endswith(e):
            return True
    return False

def __check_if_json(path=''):
    return path.endswith('json')


def validate_filepath(path, file_type='json'):
    if file_type == 'json':
        return __check_if_json(path)
    if file_type in ['h5','hdf5']:
        return __check_if_hdf5(path)
    return False
# import pdb;pdb.set_trace();print(__file__)
