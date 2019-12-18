"""
Created on Tue Dec 17 03:23:32 2019

script: pyleaves/pyleaves/train/train.py

@author: JacobARose
"""
from pyleaves.data_pipeline.preprocessing import encode_labels, filter_low_count_labels, one_hot_encode_labels #, one_hot_decode_labels
from pyleaves.data_pipeline.tf_data_loaders import DatasetBuilder
from pyleaves.leavesdb.db_query import get_label_encodings as _get_label_encodings, load_from_db
from pyleaves.leavesdb.tf_utils.tf_utils import train_val_test_split as _train_val_test_split
from pyleaves.models.keras_models import build_model #vgg16_base, xception_base, resnet_50_v2_base, resnet_101_v2_base, shallow

class Experiment:

    def __init__(self,
                 model_name='shallow',
                 dataset_name='PNAS',
                 data_root_dir='/media/data/jacob',
                 output_dir='/media/data/jacob/experiments'
                 input_shape=(224,224),
                 low_class_count_thresh=0,
                 frozen_layers=(0,-4),
                 base_learning_rate=0.001,
                 batch_size=64,
                 seed=17):

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_root_dir = data_root_dir
        self.output_folder = os.path.join(output_folder,model_name)

        self.input_shape = input_shape
        self.low_class_count_thresh = low_class_count_thresh
        self.frozen_layers = frozen_layers
        self.base_learning_rate = base_learning_rate

        self.batch_size = batch_size
        self.seed = seed

        self.load_metadata_from_db()
        self.format_loaded_metadata()
        self.data_subsets = self.train_val_test_split()

        self.dataset_builder = DatasetBuilder(root_dir=self.data_root_dir,
                                              batch_size=self.batch_size,
                                              seed=self.seed)
        self.tfrecord_splits = self.dataset_builder.collect_subsets(self.data_root_dir)

        self.label_encodings = get_label_encodings()

    def load_metadata_from_db(self):
        self.metadata = load_from_db(dataset_name=self.dataset_name)
        return self.metadata
    def format_loaded_metadata(self,
                               metadata=self.metadata,
                               low_class_count_thresh=self.low_class_count_thresh,
                               verbose=self.verbose):
        data_df = encode_labels(metadata)
        data_df = filter_low_count_labels(data_df, low_class_count_thresh=low_class_count_thresh, verbose = verbose)
        data_df = encode_labels(data_df) #Re-encode numeric labels after removing sub-threshold classes so that max(labels) == len(labels)
        paths = data_df['path'].values.reshape((-1,1))
        labels = data_df['label'].values
        self.x = paths
        self.y = labels
        return self.x, self.y

    def train_val_test_split(self,
                             x=self.x,
                             y=self.y,
                             test_size=self.test_size,
                             val_size=self.val_size,
                             verbose=self.verbose):

        self.data_subsets = _train_val_test_split(x, y, test_size=test_size, val_size=val_size, verbose=verbose)
        return self.data_subsets

    def get_label_encodings(self, dataset=self.dataset_name,
                            low_count_thresh=self.low_class_count_thresh):

        self.label_encodings = _get_label_encodings(dataset=dataset, low_count_thresh=low_count_thresh)
        self.num_classes = len(self.label_encodings)
        return self.label_encodings


    def build_model(self,
                    model_name=self.model_name,
                    num_classes=self.num_classes,
                    frozen_layers=self.frozen_layers,
                    input_shape=self.input_shape,
                    base_learning_rate=self.base_learning_rate):
        return build_model(name=model_name,
                           num_classes=num_classes,
                           frozen_layers=frozen_layers,
                           input_shape=input_shape,
                           base_learning_rate=base_learning_rate)


    def train_model(self, model, X, Y, num_epochs=50):
        '''

        '''






def run_experiment():

    experiment = Experiment(model_name='shallow',
                            dataset_name='PNAS',
                            data_root_dir='/media/data/jacob',
                            output_dir='/media/data/jacob/experiments'
                            input_shape=(224,224),
                            low_class_count_thresh=0,
                            frozen_layers=(0,-4),
                            base_learning_rate=0.001))

    model = experiment.build_model(name='shallow',
                                   num_classes=10000,
                                   frozen_layers=(0,-4),
                                   input_shape=(224,224,3),
                                   base_learning_rate=0.001)

    train_data = experiment.dataset_builder.get_dataset(subset='train')
    val_data = experiment.dataset_builder.get_dataset(subset='val')
    test_data = experiment.dataset_builder.get_dataset(subset='test')

    model.fit()




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

        print('Starting Split : %02d'%split)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = to_categorical(y[train_idx]),to_categorical(y[test_idx])

        output_log = os.path.join(output_folder,'split_%03d'%split)
        weights_best ,logs_dir= logging_configuration(output_log)
        save_split(X_train,X_test,y_train,y_test,output_log)
        ds = Dataflow(X_train,y_train,size=(229,229))
        dsm = MultiProcessRunner(ds,num_prefetch=batch_size, num_proc=5)
        ds1 = BatchData(dsm, batch_size)
        train_gen = gen(ds1)


        callbacks_list = get_callbacks(weights_best,logs_dir)
        if True:
            History=model.fit_generator(train_gen,callbacks=
            callbacks_list,epochs=epochs,steps_per_epoch=len(y_train))

        else:
            train_data = get_tf_dataset(filenames = X_train, labels = y_train)
            validation_data = get_tf_dataset(filenames = X_test, labels = y_test)
            History=model.fit(train_data,callbacks=callbacks_list, epochs=epochs,steps_per_epoch=len(y)/100)
        X_test_img = np.array([cv2.resize(cv2.imread(im),(224,224)) for im in X_test],dtype=np.float16)
        y_pred = model.predict_classes(X_test_img)
        test = np.argmax(y_test,axis=1)
        report = classification_report(test,y_pred,output_dict=True)
        df = pd.DataFrame(report).transpose()
        print(report)
        Historydf= pd.DataFrame(History.history)
        history_file = os.path.join(output_log,'history.csv')
        Historydf.to_csv(history_file)
        report_file = os.path.join(output_log,'report.csv')
        df.to_csv(report_file)
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
