'''

Logging utils for working with mlflow

'''



import mlflow


# def mlflow_log_params_dict(params_dict: dict):
    
#     for k,v in params_dict.items():
#         mlflow.log_para


def mlflow_log_history(history):
    '''
    Log full metric history per each epoch
    
    Arguments:
        history:
            history object returned from tensorflow training callback
    
    '''
        
    epochs = history.epoch    
    
    for k,v in history.history.items():
        for epoch in epochs:
            mlflow.log_metric(k, v[epoch], epoch)


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