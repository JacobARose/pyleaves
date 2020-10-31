'''

Logging utils for working with mlflow

'''

import os
import pandas as pd
import wandb




def load_Leaves_Minus_PNAS_test_dataset():
    run = wandb.init(reinit=True)
    with run:
        artifact = run.use_artifact('jrose/uncategorized/Leaves-PNAS_test:v2', type='dataset')
        artifact_dir = artifact.download()
        print(artifact_dir)
        train_df = pd.read_csv(os.path.join(artifact_dir,'train.csv'),index_col='id')
        test_df = pd.read_csv(os.path.join(artifact_dir,'test.csv'),index_col='id')
        pnas_train_df = pd.read_csv(os.path.join(artifact_dir,'PNAS_train.csv'),index_col='id')
    
    return train_df, test_df, pnas_train_df

def load_Leaves_Minus_PNAS_dataset():
    run = wandb.init(reinit=True)
    # with run:
    artifact = run.use_artifact('jrose/uncategorized/Leaves-PNAS:v1', type='dataset')
    artifact_dir = artifact.download()
    print(artifact_dir)
    train_df = pd.read_csv(os.path.join(artifact_dir,'train.csv'),index_col='id')
    test_df = pd.read_csv(os.path.join(artifact_dir,'test.csv'),index_col='id')
    
    return train_df, test_df



def load_train_test_artifact(artifact_uri='jrose/uncategorized/Leaves-PNAS:v1', run=None):
    if run is None:
        run = wandb.init(reinit=True)
    
    # with run:
    artifact = run.use_artifact(artifact_uri, type='dataset')
    artifact_dir = artifact.download()
    print('artifact_dir =',artifact_dir)
    train_df = pd.read_csv(os.path.join(artifact_dir,'train.csv'),index_col='id')
    test_df = pd.read_csv(os.path.join(artifact_dir,'test.csv'),index_col='id')
    return train_df, test_df




def load_dataset_from_artifact(dataset_name='Fossil', threshold=4, test_size=0.3, version='latest', artifact_name=None, run=None):
    train_size = 1 - test_size
    if artifact_name:
        pass
    elif dataset_name=='Fossil':
        artifact_name = f'{dataset_name}_{threshold}_{int(train_size*100)}-{int(100*test_size)}:{version}'
    elif dataset_name=='PNAS':
        artifact_name = f'{dataset_name}_family_{threshold}_50-50:{version}'
    elif dataset_name=='Leaves':
        artifact_name = f'{dataset_name}_family_{threshold}_{int(train_size*100)}-{int(100*test_size)}:{version}'
    elif dataset_name=='Leaves-PNAS':
        artifact_name = f'{dataset_name}_{int(train_size*100)}-{int(100*test_size)}:{version}'
    elif dataset_name in ['Leaves_in_PNAS', 'PNAS_in_Leaves']:
        artifact_name = f'{dataset_name}_{int(train_size*100)}-{int(100*test_size)}:{version}'
            
    artifact_uri = f'brown-serre-lab/paleoai-project/{artifact_name}'
    return load_train_test_artifact(artifact_uri=artifact_uri, run=run)









# def load_Leaves_Minus_PNAS_test_dataset():
#     run = wandb.init()

#     artifact = run.use_artifact('jrose/uncategorized/Leaves-PNAS_test:v2', type='dataset')
#     artifact_dir = artifact.download()
#     print(artifact_dir)
#     train_df = pd.read_csv(os.path.join(artifact_dir,'train.csv'),index_col='id')
#     test_df = pd.read_csv(os.path.join(artifact_dir,'test.csv'),index_col='id')
#     pnas_train_df = pd.read_csv(os.path.join(artifact_dir,'PNAS_train.csv'),index_col='id')
    
#     return train_df, test_df, pnas_train_df

# def load_Leaves_Minus_PNAS_dataset():
#     run = wandb.init()

#     artifact = run.use_artifact('jrose/uncategorized/Leaves-PNAS:v1', type='dataset')
#     artifact_dir = artifact.download()
#     print(artifact_dir)
#     train_df = pd.read_csv(os.path.join(artifact_dir,'train.csv'),index_col='id')
#     test_df = pd.read_csv(os.path.join(artifact_dir,'test.csv'),index_col='id')
    
#     return train_df, test_df





def init_new_run(project, run_name, job_type):
    run = wandb.init(project=project, name=run_name, job_type=job_type)
    return run

# def create_dataset_artifact(run,name):
#     artifact = wandb.Artifact(name,type='dataset')
#     artifact.add_dir('data/custom/images')
#     artifact.add_dir('data/custom/labels')
#     artifact.add_file('data/custom/valid.txt')
#     artifact.add_file('data/custom/train.txt')
#     run.use_artifact(artifact)


# def create_model_artifact(path,run,name):
#     artifact = wandb.Artifact(name,type='model')
#     artifact.add_file(path)
#     run.log_artifact(artifact)




















# def mlflow_log_params_dict(params_dict: dict, prefix='', sep=',', max_recursion=3):
    
#     assert max_recursion>=len(prefix.split(sep))
#     for k,v in params_dict.items():
#         if type(v) is dict:
#             mlflow_log_params_dict(params_dict=v, prefix='_'.join(prefix+k), sep=sep, max_recursion=max_recursion)
#         else:
#             mlflow.log_param(k,v)


# def mlflow_log_history(history, history_name=''):
#     '''
#     Log full metric history per each epoch
    
#     Arguments:
#         history:
#             history object returned from tensorflow training callback
#         history_name:
#             optional str to append to each metric label
    
#     '''
        
#     epochs = history.epoch    
#     if len(history_name)>0: 
#         history_name+='_'
        
#     for k,v in history.history.items():
#         for epoch in epochs:
#             mlflow.log_metric(history_name+k, v[epoch], epoch)


# def mlflow_log_best_history(history):
#     '''
#     Log individual best metric values along full history
    
#     Arguments:
#         history:
#             history object returned from tensorflow training callback
    
#     '''
    
#     #Types of metrics according to whether we want to log the max or min
#     monitor_max = ['accuracy', 'precision', 'recall']
#     monitor_min = ['loss']
        
#     logs = {}
#     for k, v in history.history.items():
#         #Extract the max or min metric value from run according to metric label
#         for m in monitor_max:
#             if m in k:
#                 log = ('max_' + k, max(v))
#         for m in monitor_min:
#             if m in k:
#                 log = ('min_' + k, min(v))
            
#         logs.update({log[0]:log[1]})
#         mlflow.log_metric(**log)
# #         mlflow.log_metric(*log)
            
#     return logs