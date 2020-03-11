'''

Logging utils for working with mlflow

'''



import mlflow


# def mlflow_log_params_dict(params_dict: dict):
    
#     for k,v in params_dict.items():
#         mlflow.log_para


def mlflow_log_history(history, history_name=''):
    '''
    Log full metric history per each epoch
    
    Arguments:
        history:
            history object returned from tensorflow training callback
        history_name:
            optional str to append to each metric label
    
    '''
        
    epochs = history.epoch    
    if len(history_name)>0: 
        history_name+='_'
        
    for k,v in history.history.items():
        for epoch in epochs:
            mlflow.log_metric(history_name+k, v[epoch], epoch)


def mlflow_log_best_history(history):
    '''
    Log individual best metric values along full history
    
    Arguments:
        history:
            history object returned from tensorflow training callback
    
    '''
    
    #Types of metrics according to whether we want to log the max or min
    monitor_max = ['accuracy', 'precision', 'recall']
    monitor_min = ['loss']
        
    logs = {}
    for k, v in history.history.items():
        #Extract the max or min metric value from run according to metric label
        for m in monitor_max:
            if m in k:
                log = ('max_' + k, max(v))
        for m in monitor_min:
            if m in k:
                log = ('min_' + k, min(v))
            
        logs.update({log[0]:log[1]})
        mlflow.log_metric(*log)
            
    return logs