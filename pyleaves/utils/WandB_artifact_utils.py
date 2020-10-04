'''

Logging utils for working with mlflow

'''



import wandb





def init_new_run(project, run_name, job_type):
    run = wandb.init(project=project, name=run_name, job_type=job_type)
    return run

def create_dataset_artifact(run,name):
    artifact = wandb.Artifact(name,type='dataset')
    artifact.add_dir('data/custom/images')
    artifact.add_dir('data/custom/labels')
    artifact.add_file('data/custom/valid.txt')
    artifact.add_file('data/custom/train.txt')
    run.use_artifact(artifact)


def create_model_artifact(path,run,name):
    artifact = wandb.Artifact(name,type='model')
    artifact.add_file(path)
    run.log_artifact(artifact)




















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