"""tensorflow summary util"""
import tensorflow as tf


class BalancedAccuracyMetric(tf.keras.metrics.Metric):
    '''
    Accuracy metric that returns the macro-averaged accuracy metric.

    i.e. For each class, calculate the ratio of correct_predictions/total_predictions. Then, the macro-averaged accuracy is simply the average of these individual accuracies.
    
    '''

    def __init__(self, num_classes, name='balanced_accuracy', **kwargs):
        super().__init__(name=name,**kwargs)
        self.num_classes=num_classes
        self.total_cm = self.add_weight("total_cm", shape=(num_classes,num_classes), initializer="zeros")
        
    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))
            
    def update_state(self, y_true, y_pred,sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true,y_pred))
        return self.total_cm
        
    def result(self):
        return self.balanced_accuracy()
    
    def confusion_matrix(self,y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_true=tf.argmax(y_true,1)
        y_pred=tf.argmax(y_pred,1)
        cm=tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)
        return cm

    def balanced_accuracy(self):
        diag = tf.linalg.diag_part(self.total_cm)
        return tf.reduce_sum(diag/tf.reduce_sum(self.total_cm, axis=1))/self.num_classes




def mean_summary(var):
    """mean scalar summary
    :type var: tensorflow.Variable
    :param var: variable to add summary
    """
    with tf.name_scope(var.name.split(":")[0]):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)


def stddev_summary(var):
    """stddev scalar summary
    :type var: tensorflow.Variable
    :param var: variable to add summary
    """
    with tf.name_scope(var.name.split(":")[0]):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)


def histogram_summary(var):
    """histogram summary
    :type var: tensorflow.Variable
    :param var: variable to add summary
    """
    with tf.name_scope(var.name.split(":")[0]):
        tf.summary.histogram('histogram', var)


def max_summary(var):
    """max scalar summary
    :type var: tensorflow.Variable
    :param var: variable to add summary
    """
    with tf.name_scope(var.name.split(":")[0]):
        tf.summary.scalar("max", tf.reduce_max(var))


def min_summary(var):
    """min summary
    :type var: tensorflow.Variable
    :param var: variable to add summary
    """
    with tf.name_scope(var.name.split(":")[0]):
        tf.summary.scalar("min", tf.reduce_min(var))


def summary_loss(var):
    """loss summary
    loss's scalar and histogram summary
    :type var: tensorflow.Variable
    :param var: variable to summary
    """
    with tf.name_scope(var.name.split(":")[0]):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)


def summary_image(var, max_outputs=0):
    """image summary
    :type var: tensorflow.Variable
    :type max_outputs: int
    :param var: variable to summary
    :param max_outputs: max output to summary image
    """
    with tf.name_scope(var.name.split(":")[0]):
        tf.summary.image("image", var, max_outputs=max_outputs)