from functools import reduce
import tensorflow as tf
import os
import threading
import time
from advantage.utils.decorator_utils import parameterized
from advantage.utils.tf_utils import build_init_uninit_op

""" Checkpointing system
"""

class CheckpointError(Exception):
    """ Exception related to the checkpoint
    system.
    """
    pass

class CheckpointThread(threading.Thread):
    """ Thread for triggering the checkpoint action.
    """
    def __init__(self,
                 checkpoint_event,
                 checkpoint_sleep_cond,
                 checkpoint,
                 checkpoint_freq_sec):
        self._checkpoint_event = checkpoint_event

        self._checkpoint_sleep_cond = checkpoint_sleep_cond

        # callable attr that actually saves checkpoint
        self._checkpoint = checkpoint

        self._checkpoint_freq_sec = checkpoint_freq_sec

        super().__init__(target=self._checkpoint, group=None)

    def run(self):
        """ Saves checkpoint every `_checkpoint_freq_sec`
        """

        while self._checkpoint_event.is_set():
            self._checkpoint_sleep_cond.acquire()
            self._checkpoint_sleep_cond.wait(self._checkpoint_freq_sec)
            if self._checkpoint_event.is_set():
                self._checkpoint()
            self._checkpoint_sleep_cond.release()

        self._checkpoint_sleep_cond.acquire()
        # one last checkpoint to save model
        self._checkpoint()
        self._checkpoint_sleep_cond.release()

@parameterized
def checkpointable(cls, **exclude):
    """ Marks a class as one which is part of a checkpoint.
    This means it contains a native TF layer or some attribute
    which itself is marked as checkpointable. This allows
    the components of the model to control which parts
    arg checkpointed.
        Args:
            chain_head: usually a `LearningModel`, but represents
                the one class that should collect all the variables
                into a `tf.train.Saver` actually start the
                `CheckpointThread`.
            exclude: names a attributes whose scopes we exclude
                from a checkpoint.
        Returns:
            cls decorator
    """
    # pylint: disable=too-many-instance-attributes
    # reason-disabled: all attrs are needed
    class Checkpointable:
        """ Represents a class whose instances can
        be the components of a checkpoint.
        """

        def __init__(self, *args, **kwargs):
            self._wrapped = cls(*args, **kwargs)
            self._wrapped.restore_session = None

            self.__class__.__name__ = self._wrapped.__class__.__name__
            self.__class__.__doc__ = self._wrapped.__class__.__doc__

            self._adv_checkpointable = True

            self._checkpoint_thread = None

            self._checkpoint_thread_event = None
            self._checkpoint_thread_sleep_cond = None

            self._tf_saver = None

            self.checkpoint_dir_path = None
            self.checkpoint_file_prefix = None
            self.checkpoint_freq_sec = None
            self._tf_global_step = None

            self._info_log_frequency = None

        def __getattr__(self, attr):
            return getattr(self._wrapped, attr)

        @property
        def checkpoint_lock(self):
            """ property for accessing lock
            for stopping checkpointing
            """
            return self._checkpoint_thread_sleep_cond

        @property
        def tf_global_step(self):
            """ property for `_tf_global_step`
            """
            return self._tf_global_step

        @property
        def adv_checkpointable(self):
            """ property for `_adv_checkpointable`
            """
            return self._adv_checkpointable

        @property
        def info_log_frequency(self):
            """ property for info_log_frequency
            """
            return self._info_log_frequency

        @info_log_frequency.setter
        def info_log_frequency(self, freq):
            """ setter for propagating attr to wrapped
            cls
            """
            self._wrapped.info_log_frequency = freq
            self._info_log_frequency = freq

        def _checkpoint(self):
            """ Tells TF to save a model variables to checkpoint proto
                    Raises:
                        CheckpointError: this `cls` is not the chain head or
                            some step for setting up checkpoints was not met
            """
            if not hasattr(self, "_tf_saver"):
                raise CheckpointError("This class is not the highest in \
                    the checkpoint chain and must call `set_up`.")

            if not self.restore_session:
                raise CheckpointError("Instance must set `restore_session` \
                    to specify which session to restore variables to.")

            if not self.checkpoint_dir_path:
                raise CheckpointError("Instance must set `checkpoint_dir_path`")

            if not self.checkpoint_file_prefix:
                raise CheckpointError("Instance must set `checkpoint_file_prefix`")

            if not self._tf_global_step:
                raise CheckpointError("Instance must call `set_up`")

            file_prefix = self.checkpoint_file_prefix

            ckpt_full_path = os.path.join(self.checkpoint_dir_path,
                                          file_prefix)
            with self.model_scope():
                step = self.restore_session.run(self._tf_global_step)

                tf.logging.warn("Saving checkpoint for %s-%d" % (file_prefix, step))
                self._tf_saver.save(self.restore_session,
                                    ckpt_full_path,
                                    global_step=self._tf_global_step)
                tf.logging.info("Checkpoint saved")


        def _try_restore(self, var_list):
            """ Restores from checkpoint into session.
                    Args:
                        var_list: list of global variables
                    Raises:
                        CheckpointError: not chain head
            """
            if not self.restore_session:
                raise CheckpointError("Instance must set `restore_session` \
                    to specify which session to restore variables to.")

            if not self.checkpoint_dir_path:
                raise CheckpointError("Instance must set `checkpoint_dir_path`")

            if not self._tf_saver:
                raise CheckpointError("Instance must call `set_up` \
                    before calling `restore`.")

            latest_ckpt = tf.train.latest_checkpoint(self.checkpoint_dir_path)

            tf.logging.warn("Starting restore procedures")

            with self.model_scope():
                if not latest_ckpt:
                    tf.logging.warn("No latest checkpoint found. Not restoring.")
                else:
                    try:
                        self._tf_saver.restore(self.restore_session,
                                               latest_ckpt)
                        tf.logging.warn("Restored from %s." % latest_ckpt)
                    except Exception:
                        raise CheckpointError("Failed to restore from %s." % latest_ckpt)

                init_op = build_init_uninit_op(self.restore_session, var_list)

                if init_op:
                    try:
                        tf.logging.warn("Found uninitalized variables")
                        self.restore_session.run(init_op)
                        tf.logging.info("Successfully initialized")
                    except Exception:
                        raise CheckpointError("Failed to initialize uninitalized varables")

            tf.logging.info("Restore procedures completed")

        def export_model(self, export_dir_path):
            """ Exports model to protobuf
                    Args:
                        export_path : directory to export to
            """
            raise NotImplementedError("Need to implement")

        def start_checkpoint_system(self):
            """ Starts up the action of checkpointing the model.
            """

            self._checkpoint_thread_event = threading.Event()
            self._checkpoint_thread_event.set()
            self._checkpoint_thread_sleep_cond = threading.Condition(threading.Lock())

            self._checkpoint_thread = CheckpointThread(self._checkpoint_thread_event,
                                                       self._checkpoint_thread_sleep_cond,
                                                       self._checkpoint,
                                                       self.checkpoint_freq_sec)
            self._checkpoint_thread.start()

        def stop_checkpoint_system(self):
            """ Stops the checkpoint system with one file save.
            """
            self._checkpoint_thread_event.clear()

            self._checkpoint_thread_sleep_cond.acquire()
            self._checkpoint_thread_sleep_cond.notify()
            self._checkpoint_thread_sleep_cond.release()

            self._checkpoint_thread.join()

        def set_up_train(self):
            """ Replaces `wrapped`'s `set_up_train` attr.
            Runs `set_up_train` method of `wrapped` class.
            Also setups checkpoint system.
            """
            with self.model_scope():
                # this is where the global_step is created
                global_step = tf.train.get_or_create_global_step()
                self._tf_global_step = global_step

            self._wrapped.set_up_train()

            with self.model_scope():

                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                             scope=self.name_scope)


                self._tf_saver = tf.train.Saver(var_list)

            self._try_restore(var_list)

        def set_up(self):
            """ Replaces `wrapped`'s `set_up`'s attr.
            Runs `set_up` method of `wrapped` class and
            restores variables.
            """
            self._wrapped.set_up()

            with self.model_scope():

                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                             scope=self.name_scope)

                self._tf_saver = tf.train.Saver(var_list)

            self._try_restore(var_list)

    return Checkpointable
