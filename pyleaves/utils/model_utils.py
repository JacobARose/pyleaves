# @Author: Jacob A Rose
# @Date:   Wed, July 22nd 2020, 10:41 pm
# @Email:  jacobrose@brown.edu
# @Filename: model_utils.py


import os


from tensorflow.python.distribute import distributed_file_utils
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import mode_keys
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.framework import ops
from tensorflow.python.eager import context

from pyleaves.utils.checkpoint_management_utils import CheckpointManager

CKPT_SAVED_EPOCH = '_ckpt_saved_epoch'
CKPT_SAVED_EPOCH_UNUSED_VALUE = -1

class WorkerTrainingState(object):
    """
    Base on official tensorflow.python.keras.distribute.worker_training_state
    in tensorflow version 2.3+

    Training state management class.
    This class provides apis for backing up and restoring the training state.
    This allows model and epoch information to be saved periodically and restore
    for fault-tolerance, also known as preemption-recovery purpose.
    """

    def __init__(self, model, checkpoint_dir):
        self._model = model

        # The epoch at which the checkpoint is saved. Used for fault-tolerance.
        # GPU device only has int64 dtype registered VarHandleOp.
        self._ckpt_saved_epoch = variables.Variable(
                    initial_value=constant_op.constant(CKPT_SAVED_EPOCH_UNUSED_VALUE, dtype=dtypes.int64),
                    name='ckpt_saved_epoch')

        # Variable initialization.
        K.set_value(self._ckpt_saved_epoch, CKPT_SAVED_EPOCH_UNUSED_VALUE)

        # _ckpt_saved_epoch gets tracked and is included in the checkpoint file
        # when backing up.
        checkpoint = trackable_util.Checkpoint(
                model=self._model, ckpt_saved_epoch=self._ckpt_saved_epoch)

        # If this is single-worker training, checkpoint_dir are the same for
        # write_checkpoint_manager and read_checkpoint_manager.
        #
        # If this is multi-worker training, and this worker should not
        # save checkpoint, we replace the write_checkpoint_manager's checkpoint_dir
        # with a temp filepath, so it writes to a file that will be removed at the
        # end of back_up() call. This is necessary because the SyncOnReadVariable
        # needs to be synced across all the workers in order to be read, and all
        # workers need to perform `save()`.
        # But all workers should restore from the same checkpoint_dir as passed in
        # read_checkpoint_manager.
        self.write_checkpoint_dir = distributed_file_utils.write_dirpath(
                                                    checkpoint_dir, None)#self._model.distribute_strategy)
        self.write_checkpoint_manager = CheckpointManager(checkpoint,
                                                          directory=self.write_checkpoint_dir,
                                                          max_to_keep=1,
                                                          checkpoint_interval=1)
        if self.write_checkpoint_dir == checkpoint_dir:
            self.read_checkpoint_manager = self.write_checkpoint_manager
        else:
            self.read_checkpoint_manager = CheckpointManager(checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=1)

    def back_up(self, epoch):
        """Back up the current state of training into a checkpoint file.
        Arguments:
          epoch: The current epoch information to be saved.
        """
        K.set_value(self._ckpt_saved_epoch, epoch)
        # Save the model plus CKPT_SAVED_EPOCH variable.
        if self.write_checkpoint_manager.save():
            distributed_file_utils.remove_temp_dirpath(
                                  self.write_checkpoint_manager.directory,
                                  None) #self._model.distribute_strategy)

    def restore(self):
        """Restore the training state from the backed up checkpoint file.
        Returns:
          True if the training state is successfully restored. False if the training
          state doesn't need to be restored, or error occurred so it can't.
        """
        # For multi-worker training, it should not restore a model in certain
        # worker setting (e.g. non-chief worker in ParameterServerStrategy).
        # pylint: disable=protected-access
        if self._model._in_multi_worker_mode() and not multi_worker_util.should_load_checkpoint():
            return
        self.read_checkpoint_manager.restore_or_initialize()

    def delete_backup(self):
        """Delete the backup directories.
        Delete the backup directories which should not exist after `fit()`
        successfully finishes.
        """
        for pathname in file_io.get_matching_files(self.write_checkpoint_manager._prefix + '*'):
            _delete_file_or_dir(pathname)
        for pathname in file_io.get_matching_files(os.path.join(self.write_checkpoint_manager.directory, 'checkpoint')):
            _delete_file_or_dir(pathname)

    def maybe_load_initial_epoch_from_ckpt(self, initial_epoch, mode):
        """Maybe load initial epoch from ckpt considering possible worker recovery.
        When `_ckpt_saved_epoch` attribute exists and is not
        `CKPT_SAVED_EPOCH_UNUSED_VALUE`, this is under multi-worker training setting
        and indicates the worker is recovering from previous failure. In this case,
        infer `initial_epoch` from `self._ckpt_saved_epoch` to continue previous
        unfinished training from certain epoch.
        Arguments:
        initial_epoch: The original initial_epoch user passes in in `fit()`.
        mode: The mode for running `model.fit()`.
        Returns:
        If the training is recovering from previous failure under multi-worker
        training setting, return the epoch the training is supposed to continue
        at. Otherwise, return the `initial_epoch` the user passes in.
        """

        epoch = K.eval(self._ckpt_saved_epoch)
        if mode == mode_keys.ModeKeys.TRAIN and epoch >= 0:
            # The most recently saved epoch is one epoch prior to the epoch it
            # failed at, so return the value of 'self._ckpt_saved_epoch' plus one.
            return epoch + 1
        return initial_epoch


def _evaluate(tensor):
    """Returns the numpy value of a tensor."""
    if context.executing_eagerly():
        return tensor.numpy()
    return ops.get_default_session().run(tensor)




def _delete_file_or_dir(pathname):
    if file_io.is_directory(pathname):
        file_io.delete_recursively(pathname)
        print(f'Deleted backup dir {pathname}')
    elif file_io.file_exists(pathname):
        file_io.delete_file(pathname)
        print(f'Deleted backup file {pathname}')


































#
# class CheckpointManagerCustom(checkpoint_management.CheckpointManager):
#
#     '''
#     Subclassing CheckpointManager in Tensorflow.__version__==2.1 to include a method
#     that was added post __version__==2.3+
#     '''
#     # assert tf.__version__=='2.1.0'
#
#     @property
#     def directory(self):
#         return self._directory
#
#
#     def restore_or_initialize(self):
#         """Restore items in `checkpoint` from the latest checkpoint file.
#         This method will first try to restore from the most recent checkpoint in
#         `directory`. If no checkpoints exist in `directory`, and `init_fn` is
#         specified, this method will call `init_fn` to do customized
#         initialization. This can be used to support initialization from pretrained
#         models.
#         Note that unlike `tf.train.Checkpoint.restore()`, this method doesn't return
#         a load status object that users can run assertions on
#         (e.g. assert_consumed()). Thus to run assertions, users should directly use
#         `tf.train.Checkpoint.restore()` method.
#         Returns:
#           The restored checkpoint path if the lastest checkpoint is found and
#           restored. Otherwise None.
#         """
#         if self._latest_checkpoint is not None:
#             self._checkpoint.restore(self._latest_checkpoint)
#             if self._checkpoint_interval is not None:
#                 self._last_checkpoint_step = _evaluate(self._step_counter)
#             return self._latest_checkpoint
#
#         # if self._init_fn is not None:
#         #     self._init_fn()
#         return None
