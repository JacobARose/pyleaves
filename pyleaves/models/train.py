import tensorflow as tf 
import logging
import argparse
import json
import glob,os 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pyleaves.models.keras_models import *
from sklearn.model_selection  import StratifiedKFold
from imgaug import augmenters as iaa
import cv2
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report
from tensorpack.dataflow import * 
from tensorpack.dataflow.parallel import PrefetchDataZMQ
from stuf import stuf
import dataset
import random
import subprocess



to_categorical = tf.keras.utils.to_categorical
CSVLogger = tf.compat.v1.keras.callbacks.CSVLogger 




sometimes = lambda aug: iaa.Sometimes(0.5, aug)
fewtimes = lambda aug: iaa.Sometimes(0.1, aug)
seq = iaa.Sequential([fewtimes(
                        iaa.OneOf([
                             iaa.EdgeDetect(alpha=(0, 0.7)),
                                iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        ])),
                        fewtimes(iaa.Affine(scale=(0.8, 1.5))),
                    sometimes(iaa.Affine(rotate=(179,181)))
                ])

def save_json(data,name):
    with open(name, 'w') as outfile:
        json.dump(data, outfile)

##______ metrics_______##

def top3_acc(labels, logits): 
    return tf.keras.metrics.top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=3)

def top5_acc(labels, logits): 
    return tf.keras.metrics.top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=5)


    
    
def load_data(db, x_col='path', y_col='family', dataset='Fossil'):
	'''
	General data loader function with flexibility for all query types.
	
	Arguments:
		db: dataset.database.Database, Must be an open connection to a database
		x_col: str, Inputs column. Should usually be the column containing filepaths for each sample
		y_col: str, Labels column. Can be any of {'family','genus','species'}
		dataset: str, Can be any dataset name that is contained in db
	
	Return:
		paths_labels: dataset.util.ResultIter,  

	'''
	paths_labels = db['dataset'].distinct(x_col, y_col, dataset=dataset)
	return paths_labels

def encode_labels(data, y_col='family'):
	'''
	Create 'label' column in data_df that features integer values corresponding to text labels contained in y_col.
	
	Arguments:
		data: dataset.util.ResultIter, Should be the returned result from loading data from the leavesdb database (e.g. data = leavesdb.db_query.load_data(db)).
		y_col: str, name of the columns containing text labels for each sample in data.
	Returns:
		data_df: pd.DataFrame, Contains 3 columns, one for paths, one for str labels, and one for int labels.
	'''
	data = pd.DataFrame(data)
	data['label'] = pd.Categorical(data[y_col])
	data['label'] = data['label'].cat.codes
	return data

def parse_function(filename, label,channels=3,img_size = (229,229)):
    img = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(img, channels=channels)#, dtype=tf.float32)
    img = tf.image.resize(img, img_size)
    return img, label #{'image':img, 'label':label}

# def train_preprocess(img, label):
#     img = tf.image.resize(img, img_size)
#     return {'image':img, 'label':label}
    

def get_tf_dataset(filenames, labels,batch_size=50):
    data = tf.data.Dataset.from_tensor_slices((filenames, labels))
    data = data.shuffle(len(filenames))
    data = data.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     data = data.map(train_preprocess, num_parallel_calls=4)
    data = data.batch(batch_size)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)
    data = data.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0'))
    return data



class Dataflow(RNGDataFlow):
    def __init__(self,paths,labels,size=(229,229)):
        self.paths = paths
        self.labels = labels
        self.tuple_label = [[paths[i],labels[i]] for i in range(len(labels))]
        random.shuffle(self.tuple_label)
        self.size = size
    def __iter__(self):
        j = random.randint(1,200)
        for i   in range(len(self.labels)):
            idx = (i+j)%len(self.labels)
            p,l = self.tuple_label[idx][0],self.tuple_label[idx][1]
            #print(p)
            try:
                image = cv2.resize(cv2.imread(p),(229,229))
                
                #print(p)
                #image = image.astype(np.float64)/255.0
                #cv2.normalize(image,image,0,255,cv.NORM_MINMAX)
                #image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #image = image[:,:,np.newaxis]
                rn = random.randint(1,200)
                if rn%21==0:
                    cv2.imwrite('randomimg_PNAS.jpeg',image)
                yield [image,l]

            except:
                print('problem with image %s'%p)
                continue 


# class AttentionLogger(tf.keras.callbacks.Callback):
#         def __init__(self, val_data, logsdir):
#             super(AttentionLogger, self).__init__()
#             self.logsdir = logsdir  # where the event files will be written 
#             self.validation_data = val_data # validation data generator
#             self.writer = tf.summary.FileWriter(self.logsdir)  # creating the summary writer

#         @tfmpl.figure_tensor
#         def attention_matplotlib(self, gen_images): 
#             '''
#             Creates a matplotlib figure and writes it to tensorboard using tf-matplotlib
#             gen_images: The image tensor of shape (batchsize,width,height,channels) you want to write to tensorboard
#             '''  
#             r, c = 5,5  # want to write 25 images as a 5x5 matplotlib subplot in TBD (tensorboard)
#             figs = tfmpl.create_figures(1, figsize=(15,15))
#             cnt = 0
#             for idx, f in enumerate(figs):
#                 for i in range(r):
#                     for j in range(c):    
#                         ax = f.add_subplot(r,c,cnt+1)
#                         ax.set_yticklabels([])
#                         ax.set_xticklabels([])
#                         ax.imshow(gen_images[cnt])  # writes the image at index cnt to the 5x5 grid
#                         cnt+=1
#                 f.tight_layout()
#             return figs

#         def on_train_begin(self, logs=None):  # when the training begins (run only once)
#                 image_summary = [] # creating a list of summaries needed (can be scalar, images, histograms etc)
#                 for index in range(len(self.model.output)):  # self.model is accessible within callback
#                     img_sum = tf.summary.image('img{}'.format(index), self.attention_matplotlib(self.model.output[index]))                    
#                     image_summary.append(img_sum)
#                 self.total_summary = tf.summary.merge(image_summary)

#         def on_epoch_end(self, epoch, logs = None):   # at the end of each epoch run this
#             logs = logs or {} 
#             x,y = next(self.validation_data)  # get data from the generator
#             # get the backend session and sun the merged summary with appropriate feed_dict
#             sess_run_summary = K.get_session().run(self.total_summary, feed_dict = {self.model.input: x['encoder_input']})
#             self.writer.add_summary(sess_run_summary, global_step =epoch)  #finally write the summary!

def log_confusion_matrix(epoch, logs):

    # Use the model to predict the values from the validation dataset.
    test_pred = model.predict_classes(test_images)
 
    con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=test_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
 
    con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes, 
                     columns = classes)
 
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
  
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
 
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
 
    image = tf.expand_dims(image, 0)
  
     # Log the confusion matrix as an image summary.
    with file_writer.as_default():
        tf.summary.image("Confusion Matrix", image, step=epoch)

def decay(epoch):
  if epoch < 10:
    return 1e-3
  elif epoch >= 10  and epoch < 20:
    return 1e-4
  else:
    return 1e-5


def get_callbacks(weights_best,logs_dir):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_best, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', save_freq=100)
    tfboard = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)
    lrs =  tf.keras.callbacks.LearningRateScheduler(decay)
    csv = CSVLogger(os.path.join(logs_dir,'training.log.csv'))
    #callback_image = AttentionLogger(logsdir=logs_dir, val_data=val_generator)
    early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    return [checkpoint,tfboard,lrs,early,csv]

def configure(gpu):

    print(gpu)
    if gpu != 'all':
        print(gpu)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
        os.environ["CUDA_VISIBLE_DEVICES"]="%d"%gpu   
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = fraction
    #session = tf.Session(config=config)

    return 

def gen(ds,batchsize=20):
    while True: 
        images,labels = [None]*batchsize, [None]*batchsize
        images, labels = next(ds.get_data())
        yield images,labels
        
def logging_configuration (output_folder):
    os.makedirs(output_folder,exist_ok=True)
    weights_best = os.path.join(output_folder,"weights.best.h5")
    logs_dir = os.path.join(output_folder,'logs')
    os.makedirs(logs_dir,exist_ok=True)
    return weights_best,logs_dir

def save_split(X_train,X_test,y_train,y_test,output_folder):

    train =[]
    for x,y in zip(X_train,y_train):
        train.append([x,y])
    traindf = pd.DataFrame(train)
    save_name = os.path.join(output_folder,'train.csv')
    traindf.to_csv(save_name)
    test =[]
    for x,y in zip(X_test,y_test):
        test.append([x,y])
    testdf = pd.DataFrame(test)
    save_name = os.path.join(output_folder,'test.csv')
    testdf.to_csv(save_name)
    return



def train_cross_validation_model(model,X,y,output_folder,splits,resolution,batch_size=20,epochs=50):
    
    skf = StratifiedKFold(n_splits=splits,random_state=42,shuffle=True)
    params = {
          'batch_size': batch_size,
          'input_shape':(229,229,3),
          'size':(229,229),
          'shuffle': True}
    split=1
      
  
    print(y)
    for train_idx,test_idx in skf.split(X,y):
        y = to_categorical(y)
        print('Starting Split : %02d'%split)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx],y[test_idx]

        output_log = os.path.join(output_folder,'split_%03d'%split)
        weights_best ,logs_dir= logging_configuration(output_log)
        save_split(X_train,X_test,y_train,y_test,output_log)
        ds = Dataflow(X_train,y_train,size=(229,229))
        dsm = MultiProcessRunner(ds,num_prefetch=50, num_proc=15)
        ds1 = BatchData(dsm, 50)
        train_gen = gen(ds1)


        callbacks_list = get_callbacks(weights_best,logs_dir)
        if True:
            History=model.fit_generator(train_gen,callbacks=
            callbacks_list,epochs=epochs,steps_per_epoch=len(y_train)/10)
        
        else:
            train_data = get_tf_dataset(filenames = X_train, labels = y_train)
            validation_data = get_tf_dataset(filenames = X_test, labels = y_test)
            History=model.fit(train_data,callbacks=callbacks_list, epochs=epochs,steps_per_epoch=len(y)/100)
        X_test_img = np.array([cv2.resize(cv2.imread(im),(224,224)) for im in X_test],dtype=np.float16)
        y_pred = model.predict_classes(X_test_img)
        test = np.argmax(y_test,axis=1)
        report = classification_report(test,y_pred)
        print(report)
        Historydf= pd.DataFrame(History.history)
        history_file = os.path.join(output_log,'history.csv')
        Historydf.to_csv(history_file)
        report_file = os.path.join(output_log,'report.json')
        save_json(report,report_file)
        split+=1



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default ='/home/irodri15/Lab/leafs/data/processed/full_dataset_processed.csv', help='input file with names')
    parser.add_argument('--output_folder', type=str, default='SAVING', help='how to save this training')
    parser.add_argument('--gpu',default =1, help= 'what gpu to use, if "all" try to allocate on every gpu'  )
    parser.add_argument('--gpu_fraction', type=float, default =0.9, help= 'how much memory of the gpu to use' )
    parser.add_argument('--pre_trained_weights',type=str,default= None,help='Pre_trained weights ')
    parser.add_argument('--resolution',default=768,help='resolution if "all" will use all the resolutions available')
    parser.add_argument('--splits',type=int,default=10,help='how many splits use for evaluation')
    

    args = parser.parse_args()
    fraction = float(args.gpu_fraction)
    gpu = int(args.gpu)
    path= args.path
    output = args.output_folder
    output_folder =args.output_folder
    weights = args.pre_trained_weights 
    splits = args.splits 

    configure(gpu,fraction)
    
    Data=LeafData(path)
    if resolution == 'all':
        Data.multiple_resolution()    
    else:
        Data.single_resolution(resolution)
    X,y = Data.X, Data.Y  
    
    classes =len(np.unique(y))
    