# @Author: Jacob A Rose
# @Date:   Thu, August 22nd 2020, 11:24 pm
# @Email:  jacobrose@brown.edu
# @Filename: multiprocessing_utils.py


import os
import cloudpickle
from multiprocessing import Pool

class RunAsCUDASubprocess:

    def __init__(self, num_gpus=0, memory_fraction=0.8):
        """
        Decorator to transparently launch Tensorflow code in a subprocess to ensure GPU memory is freed afterwards

        Args:
            num_gpus (int, optional): Defaults to 0.
            memory_fraction (float, optional): Minimum available memory necessary for a gpu to be treated as free. Defaults to 0.8.
        """        
        self._num_gpus = num_gpus
        self._memory_fraction = memory_fraction

    @staticmethod
    def _subprocess_code(num_gpus, memory_fraction, fn, args):
        # set the env vars inside the subprocess so that we don't alter the parent env
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see tensorflow issue #152
        try:
            import py3nvml
            num_grabbed = py3nvml.grab_gpus(num_gpus, gpu_fraction=memory_fraction)
        except:
            # either CUDA is not installed on the system or py3nvml is not installed (which probably means the env
            # does not have CUDA-enabled packages). Either way, block the visible devices to be sure.
            num_grabbed = 0
            os.environ['CUDA_VISIBLE_DEVICES'] = ""

        assert num_grabbed == num_gpus, 'Could not grab {} GPU devices with {}% memory available'.format(
            num_gpus,
            memory_fraction * 100)
        if os.environ['CUDA_VISIBLE_DEVICES'] == "":
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # see tensorflow issues: #16284, #2175

        # using cloudpickle because it is more flexible about what functions it will
        # pickle (lambda functions, notebook code, etc.)
        return cloudpickle.loads(fn)(*args)

    def __call__(self, f):
        def wrapped_f(*args):
            with Pool(1) as p:
                return p.apply(RunAsCUDASubprocess._subprocess_code, (self._num_gpus, self._memory_fraction, cloudpickle.dumps(f), args))

        return wrapped_f




import concurrent
import itertools


def perform_concurrent_tasks(perform_func, tasks_to_do: list, max_processes: int=4):
    # Schedule the first N futures.  We don't want to schedule them all
    # at once, to avoid consuming excessive amounts of memory.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(perform_func, task): task
            for task in itertools.islice(tasks_to_do, max_processes)
        }

        while futures:
            # Wait for the next future to complete.
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for done_future in done:
                original_task = futures.pop(done_future)
                print(f"The outcome of {original_task} is {done_future.result()}")

            # Schedule the next set of futures.  We don't want more than N futures
            # in the pool at a time, to keep memory consumption down.
            for task in itertools.islice(tasks_to_do, len(done)):
                done_future = executor.submit(perform_func, task)
                futures[done_future] = task

    return futures
