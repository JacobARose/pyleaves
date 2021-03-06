# @Author: Jacob A Rose
# @Date:   Tue, March 31st 2020, 12:36 am
# @Email:  jacobrose@brown.edu
# @Filename: test_utils.py


'''
Benchmarking and test utils for testing ETL pipeline performance.

Based on official guide located at:
'''
import itertools
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from stuf import stuf
import time
from typing import Dict
<<<<<<< HEAD
<<<<<<< HEAD
from textwrap import wrap
=======
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
=======
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
import tensorflow as tf
if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    tf.executing_eagerly()


import pyleaves

<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
=======

>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
AUTOTUNE = tf.data.experimental.AUTOTUNE



class MetaData:

    def __init__(self,
                name :str,
                num_samples : int,
                num_classes : int,
                threshold : int,
                class_distribution : Dict[str,int]): # -> stuf
        '''
        Helper function for defining default parameters associated with a particular dataset. Generates a reference object with the expected metadata values to test if data is as expected.
        '''

        self.name=name
        self.num_samples=num_samples
        self.num_classes=num_classes
        self.threshold=threshold
        self.class_distribution = OrderedDict(class_distribution)
        # self.class_names = list(self.class_distribution.keys())
        # self.class_counts = list(self.class_distribution.values())

    @property
    def class_distribution(self):
        return self._class_distribution

    @class_distribution.setter
    def class_distribution(self, class_distribution):
        self._class_distribution = OrderedDict(class_distribution)

    @property
    def class_names(self):
        return list(self.class_distribution.keys())

    @property
    def class_counts(self):
        return list(self.class_distribution.values())


    @classmethod
<<<<<<< HEAD
<<<<<<< HEAD
    def from_Dataset(cls, data): #: pyleaves.datasets.BaseDataset):
=======
    def from_Dataset(cls, data ): #: pyleaves.datasets.BaseDataset):
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
=======
    def from_Dataset(cls, data ): #: pyleaves.datasets.BaseDataset):
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
        return cls(name = data.name,
                   num_samples = data.num_samples,
                   num_classes = data.num_classes,
                   threshold = data.threshold,
                   class_distribution = data.class_counts)

    def plot_class_distribution(self,
                                class_names=None,
                                sort_by='count',
                                ascending=False,
                                figsize=(10,10),
<<<<<<< HEAD
<<<<<<< HEAD
                                ax=None,
                                plot_minmax=False):
=======
                                ax=None):
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
=======
                                ax=None):
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
        if class_names is None:
            class_names = self.class_names
            class_counts = self.class_counts
        else:
            # Only plot user provided subset of classes
            keep_class = {'class_names':[], 'class_counts':[]}
            for i, class_i in enumerate(self.class_names):
                if class_i in class_names:
                    keep_class['class_names'].append(class_i)
                    keep_class['class_counts'].append(self.class_counts[i])
            class_names = keep_class['class_names']
            class_counts = keep_class['class_counts']


        data = pd.DataFrame({'label':class_names,
                             'count':class_counts})
        data = data.sort_values(by=sort_by, ascending=ascending)
        x = 'label'
        y = 'count'

<<<<<<< HEAD
<<<<<<< HEAD
        if ax:
            fig = plt.gcf()
        else:
            fig, ax = plt.subplots(1,1,figsize=figsize)
        plt.sca(ax)
        ax = sns.barplot(x=x,y=y,data=data, ax=ax)

        plt.sca(ax)
        plt.axhline(y=data[y].min(), ls='--', label=f'min={data[y].min()}')
        plt.axhline(y=data[y].max(), ls='--', label=f'max={data[y].max()}')
        plt.legend()
=======
=======
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
        if not ax:
            fig, ax = plt.subplots(1,1,figsize=figsize)
        ax = sns.barplot(x=x,y=y,data=data, ax=ax)

<<<<<<< HEAD
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
=======
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
        N = sum(class_counts)
        M = len(class_names)
        ax.set_title(f'{self.name} (N={N},M={M})')
        for xlabel in ax.get_xticklabels():
            xlabel.set_rotation(90)
<<<<<<< HEAD
<<<<<<< HEAD
        return fig, ax

    def plot_class_percentiles(self,
                               class_names=None,
                               sort_by='count',
                               ascending=False,
                               figsize=(10,10),
                               ax=None,
                               plot_minmax=False):
        name = self.name
        if class_names is None:
            class_names = self.class_names
            class_counts = self.class_counts
        else:
            # Only plot user provided subset of classes
            keep_class = {'class_names':[], 'class_counts':[]}
            for i, class_i in enumerate(self.class_names):
                if class_i in class_names:
                    keep_class['class_names'].append(class_i)
                    keep_class['class_counts'].append(self.class_counts[i])
            class_names = keep_class['class_names']
            class_counts = keep_class['class_counts']

        data = pd.DataFrame({'label':class_names,
                             'count':class_counts})
        data = data.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
        x = 'label'
        y = 'count'

        mincount = 0
        maxcount = data[y].max()
        thresholds = np.linspace(mincount, maxcount, dtype=np.int32)

        data_counts = data['count']
        CDF = []
        for threshold in thresholds:
            percent_included = (data_counts[data_counts>threshold].sum() / data_counts.sum())*100
            CDF.append((threshold, percent_included))
        CDF = pd.DataFrame(CDF, columns = ['threshold','cumulative frequency (%)'])

        if ax:
            fig = plt.gcf()
        else:
            fig, ax = plt.subplots(1,1,figsize=figsize)
        plt.sca(ax)
        ax = sns.lineplot(x='threshold', y='cumulative frequency (%)', data=CDF, markers=True,ax=ax)

        plt.axhline(y=100, ls='--', color='k', lw=1.75)
        plt.axhline(y=0, ls='--', color='k', lw=1.75)
        ax.set_title('\n'.join(wrap(f'{name} - % of samples belonging to classes where (images/class) >= threshold',60)))
        ax.set_xlim(left=0, right=maxcount)
        plt.tight_layout()

        return fig, ax





























#         data = pd.DataFrame({'label':class_names,
#                              'count':class_counts})
#         data = data.sort_values(by=sort_by, ascending=ascending)
#         x = 'label'
#         y = 'count'
#
#         if not ax:
#             fig, ax = plt.subplots(1,1,figsize=figsize)
#         ax = sns.barplot(x=x,y=y,data=data, ax=ax)
#
#         plt.axhline(y=data[y].min(), ls='--', label=f'min={data[y].min()}')
#         plt.axhline(y=data[y].max(), ls='--', label=f'max={data[y].max()}')
#         plt.legend()
#         N = sum(class_counts)
#         M = len(class_names)
#         ax.set_title(f'{self.name} (N={N},M={M})')
#         for xlabel in ax.get_xticklabels():
#             xlabel.set_rotation(90)
#         return fig, ax
#
#
#
#     def __repr__(self):
#         distr = {k:v for k,v in self.class_distribution.items()}
#         return \
# f'''name: {self.name}
# num_samples: {self.num_samples}
# num_classes: {self.num_classes}
# class_count_threshold: {self.threshold}
# class_distribution: {distr}
#         '''
=======
=======
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
        return ax

    def __repr__(self):
        distr = {k:v for k,v in self.class_distribution.items()}
        return \
f'''name: {self.name}
num_samples: {self.num_samples}
num_classes: {self.num_classes}
class_count_threshold: {self.threshold}
class_distribution: {distr}
        '''
<<<<<<< HEAD
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
=======
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e





def metadata_template(name :str,
                      num_samples : int,
                      num_classes : int,
                      threshold : int,
                      class_distribution : Dict[str,int]): # -> stuf
    '''
    Helper function for defining default parameters associated with a particular dataset. Generates a reference object with the expected metadata values to test if data is as expected.
    '''

    return stuf(name=name,
                num_samples=num_samples,
                num_classes=num_classes,
                threshold=threshold,
                class_distribution=class_distribution)






















def timeit(ds, batch_size=None, steps=1000):
    print(f'Initiating timing run for {steps} steps:')
    start = time.time()
    it = iter(ds)
    if batch_size==None:
        batch_size=next(it)[0].numpy().shape[0]
    for i in range(steps):
        batch = next(it)
        if i%10 == 0:
            print(f'{100*i//steps:d}% : ','.'*(i//10),end='\r')
    print('\n')
    end = time.time()
    duration = end-start

    result = {'image_rate':batch_size*steps/duration,
              'batch_size':batch_size,
             'num_batches':steps,
             'total_time':duration}

    print('Completed timing run.')
    print(f"{steps} batches: {duration:0.2f} s, batch_size: {batch_size}")
    print(f"{result['image_rate']:0.2f} Images/s")

    return result

def benchmark(dataset, num_epochs=2):
    '''Wrapper for benchmarking tf.data.Dataset performance.'''
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    total_time = time.perf_counter() - start_time
#     tf.print(f"Execution time: {total_time}")
#     tf.print(f"Execution time per Epoch: {total_time}")
    print('Finished')
    print(f"Execution time: {total_time}")
    print(f"Execution time per Epoch: {total_time}")


def fast_benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for _ in tf.data.Dataset.range(num_epochs):
        for _ in dataset:
            pass
    tf.print("Execution time:", time.perf_counter() - start_time)

def increment(x):
    return x+1

def timelined_benchmark(dataset, num_epochs=2):
    # Initialize accumulators
    steps_acc = tf.zeros([0, 1], dtype=tf.dtypes.string)
    times_acc = tf.zeros([0, 2], dtype=tf.dtypes.float32)
    values_acc = tf.zeros([0, 3], dtype=tf.dtypes.int32)

    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        epoch_enter = time.perf_counter()
        for (steps, times, values) in dataset:
            # Record dataset preparation informations
            steps_acc = tf.concat((steps_acc, steps), axis=0)
            times_acc = tf.concat((times_acc, times), axis=0)
            values_acc = tf.concat((values_acc, values), axis=0)

            # Simulate training time
            train_enter = time.perf_counter()
            time.sleep(0.01)
            train_elapsed = time.perf_counter() - train_enter

            # Record training informations
            steps_acc = tf.concat((steps_acc, [["Train"]]), axis=0)
            times_acc = tf.concat((times_acc, [(train_enter, train_elapsed)]), axis=0)
            values_acc = tf.concat((values_acc, [values[-1]]), axis=0)

        epoch_elapsed = time.perf_counter() - epoch_enter
        # Record epoch informations
        steps_acc = tf.concat((steps_acc, [["Epoch"]]), axis=0)
        times_acc = tf.concat((times_acc, [(epoch_enter, epoch_elapsed)]), axis=0)
        values_acc = tf.concat((values_acc, [[-1, epoch_num, -1]]), axis=0)
        time.sleep(0.001)

    tf.print("Execution time:", time.perf_counter() - start_time)
    return {"steps": steps_acc, "times": times_acc, "values": values_acc}


class ArtificialDataset(tf.data.Dataset):
    def _generator(num_samples):
        # Opening the file
        time.sleep(0.03)

        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)

            yield (sample_idx,)

    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.int64,
            output_shapes=(1,),
            args=(num_samples,)
            )

class TimeMeasuredDataset(tf.data.Dataset):
    # OUTPUT: (steps, timings, counters)
    OUTPUT_TYPES = (tf.dtypes.string, tf.dtypes.float32, tf.dtypes.int32)
    OUTPUT_SHAPES = ((2, 1), (2, 2), (2, 3))

    _INSTANCES_COUNTER = itertools.count()  # Number of datasets generated
    _EPOCHS_COUNTER = defaultdict(itertools.count)  # Number of epochs done for each dataset

    def _generator(instance_idx, num_samples):
        '''This dataset provides samples of shape [[2, 1], [2, 2], [2, 3]] and of type [tf.dtypes.string, tf.dtypes.float32, tf.dtypes.int32]. Each sample is:

            (
              [("Open"), ("Read")],
              [(t0, d), (t0, d)],
              [(i, e, -1), (i, e, s)]
            )

            Where:

                Open and Read are steps identifiers
                t0 is the timestamp when the corresponding step started
                d is the time spent in the corresponding step
                i is the instance index
                e is the epoch index (number of times the dataset has been iterated)
                s is the sample index

        '''

        epoch_idx = next(TimeMeasuredDataset._EPOCHS_COUNTER[instance_idx])

        # Opening the file
        open_enter = time.perf_counter()
        time.sleep(0.03)
        open_elapsed = time.perf_counter() - open_enter

        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            read_enter = time.perf_counter()
            time.sleep(0.015)
            read_elapsed = time.perf_counter() - read_enter

            yield (
                [("Open",), ("Read",)],
                [(open_enter, open_elapsed), (read_enter, read_elapsed)],
                [(instance_idx, epoch_idx, -1), (instance_idx, epoch_idx, sample_idx)]
            )
            open_enter, open_elapsed = -1., -1.  # Negative values will be filtered


    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=cls.OUTPUT_TYPES,
            output_shapes=cls.OUTPUT_SHAPES,
            args=(next(cls._INSTANCES_COUNTER), num_samples)
        )




def draw_timeline(timeline, title, width=0.5, annotate=False, save=False):
    # Remove invalid entries (negative times, or empty steps) from the timelines
    invalid_mask = np.logical_and(timeline['times'] > 0, timeline['steps'] != b'')[:,0]
    steps = timeline['steps'][invalid_mask].numpy()
    times = timeline['times'][invalid_mask].numpy()
    values = timeline['values'][invalid_mask].numpy()

    # Get a set of different steps, ordered by the first time they are encountered
    step_ids, indices = np.stack(np.unique(steps, return_index=True))
    step_ids = step_ids[np.argsort(indices)]

    # Shift the starting time to 0 and compute the maximal time value
    min_time = times[:,0].min()
    times[:,0] = (times[:,0] - min_time)
    end = max(width, (times[:,0]+times[:,1]).max() + 0.01)

    cmap = mpl.cm.get_cmap("plasma")
    plt.close()
    fig, axs = plt.subplots(len(step_ids), sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle(title)
    fig.set_size_inches(17.0, len(step_ids))
    plt.xlim(-0.01, end)

    for i, step in enumerate(step_ids):
        step_name = step.decode()
        ax = axs[i]
        ax.set_ylabel(step_name)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("time (s)")
        ax.set_xticklabels([])
        ax.grid(which="both", axis="x", color="k", linestyle=":")

        # Get timings and annotation for the given step
        entries_mask = np.squeeze(steps==step)
        serie = np.unique(times[entries_mask], axis=0)
        annotations = values[entries_mask]

        ax.broken_barh(serie, (0, 1), color=cmap(i / len(step_ids)), linewidth=1, alpha=0.66)
        if annotate:
            for j, (start, width) in enumerate(serie):
                annotation = "\n".join([f"{l}: {v}" for l,v in zip(("i", "e", "s"), annotations[j])])
                ax.text(start + 0.001 + (0.001 * (j % 2)), 0.55 - (0.1 * (j % 2)), annotation,
                        horizontalalignment='left', verticalalignment='center')
    if save:
        plt.savefig(title.lower().translate(str.maketrans(" ", "_")) + ".svg")




def map_decorator(func):
    '''
    Use wrappers for mapped function
    - To run mapped function in an eager context, you have to wrap them inside a tf.py_function call.
        '''
    def wrapper(steps, times, values):
        # Use a tf.py_function to prevent auto-graph from compiling the method
        return tf.py_function(
            func,
            inp=(steps, times, values),
            Tout=(steps.dtype, times.dtype, values.dtype)
        )
    return wrapper


_batch_map_num_items = 50
def dataset_generator_fun(*args):
    return TimeMeasuredDataset(num_samples=_batch_map_num_items)




@map_decorator
def naive_map(steps, times, values):
    map_enter = time.perf_counter()
    time.sleep(0.001)  # Time consuming step
    time.sleep(0.0001)  # Memory consuming step
    map_elapsed = time.perf_counter() - map_enter

    return (
        tf.concat((steps, [["Map"]]), axis=0),
        tf.concat((times, [[map_enter, map_elapsed]]), axis=0),
        tf.concat((values, [values[-1]]), axis=0)
    )



@map_decorator
def time_consuming_map(steps, times, values):
    map_enter = time.perf_counter()
    time.sleep(0.001 * tf.cast(values.shape[0], tf.float32))  # Time consuming step
    map_elapsed = time.perf_counter() - map_enter

    return (
        tf.concat((steps, tf.tile([[["1st map"]]], [steps.shape[0], 1, 1])), axis=1),
        tf.concat((times, tf.tile([[[map_enter, map_elapsed]]], [times.shape[0], 1, 1])), axis=1),
        tf.concat((values, tf.tile([[values[:][-1][0]]], [values.shape[0], 1, 1])), axis=1)
    )


@map_decorator
def memory_consuming_map(steps, times, values):
    map_enter = time.perf_counter()
    time.sleep(0.0001 * tf.cast(values.shape[0], tf.float32))  # Memory consuming step
    map_elapsed = time.perf_counter() - map_enter

    # Use tf.tile to handle batch dimension
    return (
        tf.concat((steps, tf.tile([[["2nd map"]]], [steps.shape[0], 1, 1])), axis=1),
        tf.concat((times, tf.tile([[[map_enter, map_elapsed]]], [times.shape[0], 1, 1])), axis=1),
        tf.concat((values, tf.tile([[values[:][-1][0]]], [values.shape[0], 1, 1])), axis=1)
    )


if __name__ == '__main__':

    naive_dataset = tf.data.Dataset.range(2) \
        .flat_map(dataset_generator_fun) \
        .map(naive_map) \
        .batch(_batch_map_num_items, drop_remainder=True) \
        .unbatch()

    # Parallelize data reading
    # Vectorize your mapped function
    # Parallelize map transformation
    # Cache data
    # Reduce memory usage
    optimized_dataset = tf.data.Dataset.range(2) \
        .interleave(dataset_generator_fun,num_parallel_calls=AUTOTUNE) \
        .batch(_batch_map_num_items,drop_remainder=True) \
        .map(time_consuming_map,num_parallel_calls=AUTOTUNE) \
        .cache() \
        .map(memory_consuming_map,num_parallel_calls=AUTOTUNE) \
        .prefetch(AUTOTUNE) \
        .unbatch()

    fast_dataset = tf.data.Dataset.range(10000)

    ####################################################

    naive_timeline = timelined_benchmark(naive_dataset,5)

    optimized_timeline = timelined_benchmark(optimized_dataset,5)




    benchmark(ArtificialDataset())

    # Fast Benchmarking

    ## Scalar Mapping
    fast_benchmark(fast_dataset \
                    # Apply function one item at a time
                    .map(increment) \
                    # Batch
                    .batch(256)
                    )
    ## Vectorized mapping
    fast_benchmark(fast_dataset \
                    .batch(256)
                    # Apply function on a batch of items
                    # The tf.Tensor.__add__ method already handle batches
                    .map(increment) \
                    )



    draw_timeline(naive_timeline, "Naive", 15)

    draw_timeline(optimized_timeline, "Optimized", 15)
