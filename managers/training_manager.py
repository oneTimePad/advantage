from abc import ABCMeta, abstractmethod
import tensorflow as tf
from advantage.checkpoint import CheckpointError
from advantage.utils.proto_parsers import parse_hooks
from advantage.builders import build_model, build_environment

""" All necessary classes for running the training process
for an RL model
"""

class Train:
    """ Context Manager for managing
    training.
    """
    def __init__(self,
                 model,
                 run_for_steps,
                 checkpoint_dir_path,
                 checkpoint_file_prefix,
                 checkpoint_freq_sec,
                 hooks,
                 stopper):
        self._training_manager = TrainingManager(model,
                                                 run_for_steps,
                                                 checkpoint_dir_path,
                                                 checkpoint_file_prefix,
                                                 checkpoint_freq_sec,
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
        if exception_type:
            if isinstance(exception_type, KeyboardInterrupt):
                print("Training ended by user")
            else:
                print("Found Exception %s" % exception_type)
                print("Exception value %s" % exception_value)
                print("We will still try to save the model!")
        else:
            print("Training has completed.")

        # all shutdown procedures will run sequentially in `shutdown`
        # if one fails we will still continue
        # if an `unknown` exception occurs, we can't continue
        # hence `fatal_exception`
        try:
            print("Starting shutdown proceedures")
            self._training_manager.shutdown()
            print("Shutdown has complete")
        except Exception as fatal_exception:
            # something terribly wrong
            print("Failed to shutdown properly!.")
            print("Received fatal exception while attempting to shutdown!")
            print("Shutdown procedure can't continue!")
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
                   config.run_for_steps,
                   config.checkpoint_dir_path,
                   config.checkpoint_file_prefix,
                   config.checkpoint_freq_sec,
                   parse_hooks(None),
                   stopper)


class TrainingManager:
    """ Manages the Training of an RL Model
    """
    def __init__(self,
                 model,
                 run_for_steps,
                 checkpoint_dir_path,
                 checkpoint_file_prefix,
                 checkpoint_freq_sec,
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

        self._run_for_steps = run_for_steps

        self._before_train_hooks = TrainHookRunTime.filter_before_train(hooks)
        self._during_train_hooks = TrainHookRunTime.filter_during_train(hooks)
        self._after_train_hooks = TrainHookRunTime.filter_after_train(hooks)

        self._stopper = stopper

    def set_up(self):
        """ builds all necessary dependencies for training
        """
        self._model.set_up_train()

        TrainHook.set_up_hooks(self._before_train_hooks)
        TrainHook.set_up_hooks(self._during_train_hooks)
        TrainHook.set_up_hooks(self._after_train_hooks)

    def shutdown(self):
        """ Peforms any necessary shutdown procedures
        """
        try:
            tf.logging.warn("Saving model to checkpoint. Please wait...")
            self._model.stop_checkpoint_system() # stop checkpointing
            self._model.clean()
            tf.logging.warn("Model has been successfully saved to checkpoint.")
        except AttributeError:
            tf.logging.error("Checkpoint couldn't be saved. This is"
                             " because the model class isn't decorated with `checkpointable`")
        except CheckpointError:
            tf.logging.error("Checkpoint couldn't be saved!"
                             " Checkpointing System wasn't setup properly!")

    def train_model(self):
        """ Runs the actually training process.
            The model's act_iteration evaluates the agent(s)
            for a specified number of steps. Then a improvement
            step is made based on the evaluation of the agents.
            This runs until a specified termination point.
        """
        TrainHook.run_hooks(self._before_train_hooks)
        try:
            self._model.start_checkpoint_system()
        except AttributeError:
            tf.logging.error("Checkpointing system couldn't be started. "
                             "This is because the model class isn't decorated"
                             " with `checkpointable`")
            raise Exception("Failed to start checkpoint system")
        except CheckpointError:
            tf.logging.error("Checkpointing system couldn't be started!"
                             " Checkpointing System wasn't setup properly!")
            raise Exception("Failed to start checkpoint system")
        stopper = self._stopper
        while self._model.steps < self._run_for_steps and not stopper.should_stop:

            info_dict = self._model.act_iteration()
            total_steps_after = self._model.steps
            tf.logging.info("Model completed a total of %d steps.", total_steps_after)

            tf.logging.info("Proceeding with training iteration.")
            self._model.train_iteration(info_dict)
            tf.logging.info("Training iteration completed.")

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
