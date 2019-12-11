import tensorflow as tf
from tensorflow.keras import backend as K

def reset_keras_session():
    
    K.clear_session()
    K.get_session().close()
    tf.reset_default_graph()
    
    tf_config=tf.ConfigProto(log_device_placement=True)
    tf_config.gpu_options.per_process_gpu_memory_fraction=0.9
    tf_config.gpu_options.allocator_type = 'BFC'
    tf_config.gpu_options.allow_growth = True
    #tf_config.allow_soft_placement = True
    sess = tf.Session(graph=tf.get_default_graph(), config=tf_config)
    K.set_session(sess)

