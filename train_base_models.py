'''
Example cmd line args:

>> python train_base_models.py --dataset_name Fossil --base resnet50 --output_folder ~/experiment_data/Fossil --gpu 1 --batchsize 128 -l 10

'''

import tensorflow as tf
from pyleaves.data_pipeline.preprocessing import filter_low_count_labels, one_hot_encode_labels, one_hot_decode_labels
from pyleaves import leavesdb
from pyleaves.models.keras_models import *
from pyleaves.models.train import *
import logging
import argparse
import dataset
import os
from stuf import stuf

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/media/data_cifs/irodri15/data/processed/full_dataset_processed.csv', help='input file with names')
    parser.add_argument('--dataset_name', type=str, default='PNAS', help='Name of dataset to load from database', choices=['PNAS','Fossil','Leaves','plant_village'])
    parser.add_argument('--output_folder', type=str, default='SAVING', help='how to save this training')
    parser.add_argument('--gpu',default =1, help='what gpu to use, if "all" try to allocate on every gpu')
    parser.add_argument('--gpu_fraction', type=float, default=0.9, help='how much memory of the gpu to use')
    parser.add_argument('--pre_trained_weights',type=str,default=None, help='Pre_trained weights ')
    parser.add_argument('--resolution',default=768,help='resolution if "all" will use all the resolutions available')
    parser.add_argument('--splits',type=int,default=10,help='how many splits use for evaluation')
    parser.add_argument('--base', type=str,default='resnet101',choices=['resnet101','resnet50','xception','vgg','shallow'])
    parser.add_argument('--batchsize', type=int,default=50)
    parser.add_argument('--epochs', type=int,default=100)
    parser.add_argument('--base_learning_rate', type=float, default=0.0001, help= 'initial learning rate to decay from' )
    parser.add_argument('-l','--low_class_count_threshold',type=int, default=10, help='Minimum number of samples to allow per class')
    args = parser.parse_args()
    fraction = float(args.gpu_fraction)
    gpu = int(args.gpu)
    path= args.path
    dataset_name = args.dataset_name
    output = args.output_folder
    output_folder = args.output_folder
    weights = args.pre_trained_weights
    splits = args.splits
    base = args.base
    resolution = args.resolution
    batch_size = args.batchsize
    epochs = args.epochs
    base_learning_rate = args.base_learning_rate
    low_class_count_thresh = args.low_class_count_threshold
    
    configure(gpu)

    #Data=LeafData(path)
    #if resolution == 'all':
    #    Data.multiple_resolution()
    #else:
    #    Data.single_resolution(resolution)
    #X,y, lu = Data.X, Data.Y,Data.lookup_table


    local_db = os.path.expanduser(r'~/pyleaves/pyleaves/leavesdb/resources/leavesdb.db')#leavesdb.init_local_db()
    db = dataset.connect(f'sqlite:///{local_db}', row_type=stuf)
#     db = dataset.connect('sqlite:////home/irodri15/Code/leavesdb/leavesdb.db',row_type=stuf)
    datasets=db['dataset']
    data= load_data(db,x_col='path', y_col='family', dataset=dataset_name)
    
    data_df = encode_labels(data)
    data_df = filter_low_count_labels(data_df, threshold=low_class_count_thresh, verbose = True)
    data_df = encode_labels(data_df)
    

    X = data_df['path'].values
    y = data_df['label'].values

    num_classes = len(np.unique(y))
    print('num_classes =', num_classes)
    if 'resnet101'==base:
        base_model = resnet_101_v2_base(num_classes=num_classes,frozen_layers=(0,-2))
    elif 'resnet50'==base:
        base_model = resnet_50_v2_base(num_classes=num_classes,frozen_layers=(0,-2))
    elif 'xception'==base:
        base_model = xception_base(num_classes=num_classes,frozen_layers=(0,-2))
    elif 'vgg'==base:
        base_model = vgg16_base()
    elif 'shallow' ==base:
        base_model = shallow()

    output_folder = os.path.join(output_folder,base)

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    if base != 'shallow':
        conv1 = tf.keras.layers.Dense(2048,activation='relu')
        conv2 = tf.keras.layers.Dense(512,activation='relu')
        prediction_layer = tf.keras.layers.Dense(num_classes,activation='softmax')
        model = tf.keras.Sequential([
            base_model,
            global_average_layer,conv1,conv2,
            prediction_layer
            ])
    else:
        prediction_layer = tf.keras.layers.Dense(num_classes,activation='softmax')
        model = tf.keras.Sequential([
            base_model,
            prediction_layer
            ])

#     base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy',top3_acc,top5_acc])



    train_cross_validation_model(model,X,y,output_folder,splits,resolution,batch_size,epochs)
