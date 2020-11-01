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






from wandb import util, Image, Error, termwarn

# assums X represents images and y_true/y_pred are logits for each class
def image_categorizer_dataframe(x, y_true, y_pred, labels, example_ids=None):
    np = util.get_module('numpy', required='dataframes require numpy')
    pd = util.get_module('pandas', required='dataframes require pandas')

    x, y_true, y_pred, labels = np.array(x), np.array(y_true), np.array(y_pred), np.array(labels)

    # If there is only one output value of true_prob, convert to 2 class false_prob, true_prob
    if y_true[0].shape[-1] == 1 and y_pred[0].shape[-1] == 1:
        y_true = np.concatenate((1-y_true, y_true), axis=-1)
        y_pred = np.concatenate((1-y_pred, y_pred), axis=-1)

    if x.shape[0] != y_true.shape[0]:
        termwarn('Sample count mismatch: x(%d) != y_true(%d). skipping evaluation' % (x.shape[0], y_true.shape[0]))
        return
    if x.shape[0] != y_pred.shape[0]:
        termwarn('Sample count mismatch: x(%d) != y_pred(%d). skipping evaluation' % (x.shape[0], y_pred.shape[0]))
        return
    if y_true.shape[-1] != len(labels):
        termwarn('Label count mismatch: y_true(%d) != labels(%d). skipping evaluation' % (y_true.shape[-1], len(labels)))
        return
    if y_pred.shape[-1] != len(labels):
        termwarn('Label count mismatch: y_pred(%d) != labels(%d). skipping evaluation' % (y_pred.shape[-1], len(labels)))
        return

    class_preds = []
    for i in range(len(labels)):
        class_preds.append(y_pred[:,i])

    images = [Image(img) for img in x]
    true_class = labels[y_true.argmax(axis=-1)]
    true_prob = y_pred[np.arange(y_pred.shape[0]), y_true.argmax(axis=-1)]
    pred_class = labels[y_pred.argmax(axis=-1)]
    pred_prob = y_pred[np.arange(y_pred.shape[0]), y_pred.argmax(axis=-1)]
    correct = true_class == pred_class

    if example_ids is None:
        example_ids = ['example_' + str(i) for i in range(len(x))]

    dfMap = {
        'wandb_example_id': example_ids,
        'image': images,
        'true_class': true_class,
        'true_prob': true_prob,
        'pred_class': pred_class,
        'pred_prob': pred_prob,
        'correct': correct,
    }

    for i in range(len(labels)):
        dfMap['prob_{}'.format(labels[i])] = class_preds[i]

    all_columns = [
        'wandb_example_id',
        'image',
        'true_class',
        'true_prob',
        'pred_class',
        'pred_prob',
        'correct',
    ] + ['prob_{}'.format(l) for l in labels]

    return pd.DataFrame(dfMap, columns=all_columns)



import tensorflow as tf

class WandBImagePredictionCallback(tf.keras.callbacks.Callback):

    def __init__(self, validation_data = None, class_labels=None, example_ids=None, generator=None, validation_steps=None):
        super().__init__(self)

        if isinstance(validation_data, tuple):
            x, y = validation_data
        else:
            x, y = None, None
        self.x = x
        self.y = y
        self.class_labels = class_labels
        self.example_ids = example_ids
        self.generator = generator
        self.validation_steps = validation_steps


    def on_train_end(self, logs=None):
        if self.log_evaluation:
            wandb.run.summary["results"] = self._log_dataframe()
        pass


    def _log_dataframe(self):
        x, y_true, y_pred = None, None, None

        if self.generator:
            if not self.validation_steps:
                wandb.termwarn(
                    "when using a generator for validation data with dataframes, you must pass validation_steps. skipping"
                )
                return None

            for i in range(self.validation_steps):
                bx, by_true = next(self.generator)
                by_pred = self.model.predict(bx)
                if x is None:
                    x, y_true, y_pred = bx, by_true, by_pred
                else:
                    x, y_true, y_pred = (
                        np.append(x, bx, axis=0),
                        np.append(y_true, by_true, axis=0),
                        np.append(y_pred, by_pred, axis=0),
                    )
        else:
            x, y_true = self.x, self.y
            y_pred = self.model.predict(x)
        try:
            return image_categorizer_dataframe(
                                    x=x, y_true=y_true, y_pred=y_pred, labels=self.class_labels, example_ids=self.example_ids
                                    )
        except:
            print('WARNING: Fix WandB prediction callback')
            return None







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





# def init_new_run(project, run_name, job_type):
#     run = wandb.init(project=project, name=run_name, job_type=job_type)
#     return run

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