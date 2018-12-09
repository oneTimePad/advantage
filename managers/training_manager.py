from abc import ABCMeta, abstractmethod
from functools import reduce
import tensorflow as tf
import threading
from advantage.checkpoint import CheckpointError
from advantage.utils.proto_parsers import parse_hooks
from advantage.builders import build_model, build_environment
from advantage.utils.tf_utils import get_or_create_improve_step, create_improve_step_update_op
from advantage.loggers.logger import Logger
import advantage.loggers as loggers

""" All necessary classes for running the training process
for an RL model
"""

class Train:
    """ Context Manager for managing
    training.
    """
    def __init__(self,
                 model,
                 improve_for_steps,
                 checkpoint_dir_path,
                 checkpoint_file_prefix,
                 checkpoint_freq_sec,
                 config,
                 hooks,
                 stopper):
        self._training_manager = TrainingManager(model,
                                                 improve_for_steps,
                                                 checkpoint_dir_path,
                                                 checkpoint_file_prefix,
                                                 checkpoint_freq_sec,
                                                 config,
                                                 hooks,
                                                 stopper)
    def __enter__(self):
        self._training_manager.set_up()
        return self._training_manager

    # credit to https://tinyurl.com/exit-best-practices
    def __exit__(self, exception_type, exception_value, traceback):
        """ Handles training ending gracefully. Attempts to save
        model to checkpoint
        """
        show_traceback = False
        if exception_type:
            if exception_type is KeyboardInterrupt:
                tf.logging.warn("Training ended by user")
            else:
                tf.logging.error("Found Exception %s" % exception_type)
                tf.logging.error("Exception value %s" % exception_value)
                tf.logging.warn("We will still try to save the model!")
                show_traceback = True
        else:
            tf.logging.warn("Training has completed.")

        # all shutdown procedures will run sequentially in `shutdown`
        # if one fails we will still continue
        # if an `unknown` exception occurs, we can't continue
        # hence `fatal_exception`
        try:
            tf.logging.warn("Starting shutdown proceedures")
            self._training_manager.shutdown()
            tf.logging.info("Shutdown has complete")
            return not show_traceback
        except Exception as fatal_exception:
            # something terribly wrong
            tf.logging.fatal("Failed to shutdown properly!.")
            tf.logging.fatal("Received fatal exception while attempting to shutdown!")
            tf.logging.fatal("Shutdown procedure can't continue!")
            raise fatal_exception

    @classmethod
    def from_config(cls, config, stopper, env=None):
        """ Builds `Train` object from config
                Args:
                    config: the training configuration
                    stopper: stop training if async
                    env: optional `Environment` object

                Returns:
                    `Train` object
        """
        if not env:
            env = build_environment(config.environment)

        model = build_model(config.model, env, True)
        return cls(model,
                   config.improve_for_steps,
                   config.checkpoint_dir_path,
                   config.checkpoint_file_prefix,
                   config.checkpoint_freq_sec,
                   config,
                   parse_hooks(None),
                   stopper)


class TrainingManager:
    """ Manages the Training of an RL Model
    """
    def __init__(self,
                 model,
                 improve_for_steps,
                 checkpoint_dir_path,
                 checkpoint_file_prefix,
                 checkpoint_freq_sec,
                 config,
                 hooks,
                 stopper):
        """
            Args:
                model: constructed model
                run_for: number of steps to run for
                checkpoint_dir_path: location to checkpoint directory
                    will make if doesn't exist
                checkpoint_modulo: period to checkpoint the model
                hooks: list of TrainHook objects
        """
        self._model = model

        self._model.checkpoint_dir_path = checkpoint_dir_path
        self._model.checkpoint_file_prefix = checkpoint_file_prefix
        self._model.checkpoint_freq_sec = checkpoint_freq_sec

        self._improve_for_steps = improve_for_steps

        self._config = config

        self._before_train_hooks = TrainHookRunTime.filter_before_train(hooks)
        self._during_train_hooks = TrainHookRunTime.filter_during_train(hooks)
        self._after_train_hooks = TrainHookRunTime.filter_after_train(hooks)

        self._stopper = stopper

        self._tf_increment_improve_step = None
        self._tf_improve_step = None

        self._thread_event = None
        self._thread_sleep_cond = None
        self._thread_lock = threading.Lock()

        self._logger = None
        self._logger_freq_sec = config.info_log_frequency

        self._logger_thread_started = False
        self._checkpoint_thread_started = False

    @property
    def improve_step_value(self):
        """ property for fetching runtime
        value of `improve_step`
        """
        return self._model.restore_session.run(self._tf_improve_step)

    @loggers.value("Model has completed %d improvement steps",
                   loggers.LogVarType.INSTANCE_ATTR,
                   "improve_step_value")
    def _increment_improve_step(self):
        self._model.restore_session.run(self._tf_increment_improve_step)


    def set_up(self):
        """ builds all necessary dependencies for training
        """
        self._tf_improve_step = get_or_create_improve_step(self._model.model_scope) # created here

        self._tf_increment_improve_step = create_improve_step_update_op(self._model.model_scope,
                                                                        self._tf_improve_step)

        self._model.set_up_train()

        TrainHook.set_up_hooks(self._before_train_hooks)
        TrainHook.set_up_hooks(self._during_train_hooks)
        TrainHook.set_up_hooks(self._after_train_hooks)

    def shutdown(self):
        """ Peforms any necessary shutdown procedures
        """
        self._thread_event.clear()

        self._thread_sleep_cond.acquire()
        self._thread_sleep_cond.notifyAll()
        self._thread_sleep_cond.release()

        if self._logger_thread_started:
            self._logger.join()

        if self._checkpoint_thread_started:
            self._model.checkpoint_join()

        self._model.clean()


    def _startup(self):
        """ Performs start up procedures
        """

        thread_event = threading.Event()
        thread_sleep_cond = threading.Condition(self._thread_lock)
        thread_event.set()
        self._thread_event = thread_event
        self._thread_sleep_cond = thread_sleep_cond

        self._logger = Logger(thread_event,
                              thread_sleep_cond,
                              self._logger_freq_sec)

        self._logger.start()

        self._logger_thread_started = True

        try:
            self._model.start_checkpoint_system(thread_event,
                                                thread_sleep_cond)
            self._checkpoint_thread_started = True

        except AttributeError:
            tf.logging.error("Checkpointing system couldn't be started."
                             " This is because the model class isn't decorated"
                             " with `checkpointable`")
            raise Exception("Failed to start checkpoint system")
        except CheckpointError:
            tf.logging.error("Checkpointing system couldn't be started!"
                             " Checkpointing System wasn't setup properly!")
            raise Exception("Failed to start checkpoint system")

    def train_model(self):
        """ Runs the actually training process.
            The model's act_iteration evaluates the agent(s)
            for a specified number of steps. Then a improvement
            step is made based on the evaluation of the agents.
            This runs until a specified termination point.
        """
        TrainHook.run_hooks(self._before_train_hooks)

        self._startup()

        stopper = self._stopper

        tf.logging.set_verbosity(tf.logging.INFO)

        tf.summary.FileWriter(self._model.checkpoint_dir_path, self._model.graph)

        while self.improve_step_value < self._improve_for_steps and not stopper.should_stop:
            with self._thread_lock:
                info_dict = self._model.act_iteration()

                if self._model.improve_iteration(info_dict):
                    self._increment_improve_step()

                TrainHook.run_hooks(self._during_train_hooks)

        TrainHook.run_hooks(self._after_train_hooks)

class TrainHookRunTime:
    """ Contain constants for specifying when hooks should run
    """
    BEFORE_TRAIN = "before_train"
    DURING_TRAIN = "during_train"
    AFTER_TRAIN = "after_train"

    def __new__(cls):
        raise NotImplementedError("Can't instantiate")

    @staticmethod
    def filter_before_train(hooks):
        """ filters from list hooks with whenToRun of BEFORE_TRAIN
        """
        return filter(lambda x: x.whenToRun == TrainHookRunTime.BEFORE_TRAIN, hooks)

    @staticmethod
    def filter_during_train(hooks):
        """ filters from list hooks with whenToRun of DURING_TRAIN
        """
        return filter(lambda x: x.whenToRun == TrainHookRunTime.DURING_TRAIN, hooks)

    @staticmethod
    def filter_after_train(hooks):
        """ filters from list hooks with whenToRun of AFTER_TRAIN
        """
        return filter(lambda x: x.whenToRun == TrainHookRunTime.AFTER_TRAIN, hooks)


class TrainHook(metaclass=ABCMeta):
    """ Interface for hook ran at specified time
    during training.
    """
    @abstractmethod
    def set_up(self):
        """ setup all necessary ops for hook
        """
        raise NotImplementedError()

    @abstractmethod
    def run(self, *args, **kwargs):
        """ Runs hook
        """
        raise NotImplementedError()

    @staticmethod
    def set_up_hooks(hooks):
        """ Sets up list of TrainHooks
        """
        map(lambda x: x.set_up(), hooks)

    @staticmethod
    def run_hooks(hooks):
        """ Runs list of Train Hooks
        """
        map(lambda x: x.run(), hooks)
