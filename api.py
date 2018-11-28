import multiprocessing
import threading
from advantage.managers import Train, Infer
from advantage.environments import Environment
from advantage.utils.proto_parsers import parse_obj_from_file
from advantage.protos import config_pb2
from advantage.exception import AdvantageError


""" The main module exposed to clients of the Advantage framework
"""

class _AsyncTrainManager:
    """ Controlling Asynchronous training
    """
    def __init__(self, signal, runner=None):
        self._signal = signal
        self.runner = runner


    def stop(self):
        """ Tell training to stop
        """
        if hasattr(self._signal, "set"):
            self._signal.set()
        else:
            self._signal = False

    @property
    def is_training(self):
        """ property for determing if training
        """
        return not self.should_stop

    @property
    def should_stop(self):
        """ property for determing if training should
        stop
        """
        if hasattr(self._signal, "is_set"):
            return self._signal.is_set()
        return self._signal

    def wait(self, timeout=None):
        """ Wait for training to end
                Args:
                    timeout: training timeout
        """
        if self.runner:
            self.runner.wait(timeout)

class Agent:
    """ This represents a Reinforcement Learning
    agent. This is different than the concept of `agent`
    internal to the framework. But represents the term
    `agent` in typical RL literature
    """

    def __init__(self, config):
        if not isinstance(config, config_pb2.Config):
            raise AdvantageError("config is not of type `config_pb2`")

        self._config = config
        self._infer_manager = None
        self._env = None
        self._async_train_manager = None
        self._checkpoint_file_prefix = config.checkpoint_file_prefix
        self._checkpoint_dir_path = config.checkpoint_dir_path

    @property
    def config(self):
        """ property for `_config`
        """
        return self._config

    @config.setter
    def config(self, configuration):
        """ setter for `_config`
        """
        if not isinstance(configuration, config_pb2.Config):
            raise AdvantageError("config is not of type `config_pb2`")
        self._config = configuration

    @property
    def checkpoint_file_prefix(self):
        """ property for `_checkpoint_file_prefix`
        """
        return self._checkpoint_file_prefix

    @checkpoint_file_prefix.setter
    def checkpoint_file_prefix(self, file_prefix):
        """setter for `_checkpoint_file_prefix`
        """
        self._config.checkpoint_file_prefix = file_prefix

    @property
    def checkpoint_dir_path(self):
        """ property for `_checkpoint_dir_path`
        """
        return self._checkpoint_dir_path

    @checkpoint_dir_path.setter
    def checkpoint_dir_path(self, dir_path):
        """setter for `_checkpoint_file_prefix`
        """
        self._config.checkpoint_dir_path = dir_path

    def attach_env(self, environment):
        """ Attaches an `Environment`
        to this agent at runtime. This allows the
        agent to utilizes different
        sources for the environment it
        acts in. However the environment
        must have the correct input shapes
        that the agent expects

            Args:
                env : specified `Environment`

            Raises:
                AdvantageError: env doesn't implement `Environment`
        """
        if not isinstance(environment, Environment):
            raise AdvantageError("environment must implement `Environment` interface")
        self._env = environment

    def train(self, async_proc=False, async_thread=False):
        """ Constructs `Train` which is used
        to manage `Model` training
            Args:
                async_proc: whether to train asynchronously in process
                async_thread: whether to train asynchronously in thread
        """
        def _train(signal):
            nonlocal self
            with Train.from_config(self._config, signal, env=self._env) as manager:
                manager.train_model()

            self._async_train_manager.stop()
            self._async_train_manager = None

        if async_proc and async_thread:
            raise AdvantageError("`async_proc and `async_thread` cannot both be `True`")

        # running training in separate process
        if async_proc:
            async_train_manager = _AsyncTrainManager(multiprocessing.Event())
            proc = multiprocessing.Process(target=_train, args=(async_train_manager,))
            async_train_manager.runner = proc
            proc.start()
            self._async_train_manager = async_train_manager
            return

        # run training in separate thread
        if async_thread:
            async_train_manager = _AsyncTrainManager(threading.Event())
            thread = threading.Thread(target=_train, args=(async_train_manager,))
            async_train_manager.runner = thread
            thread.start()
            self._async_train_manager = async_train_manager
            return

        # synchronous training
        self._async_train_manager = _AsyncTrainManager(False)
        _train(self._async_train_manager)

    def stop_training(self):
        """ Stop training. usually used if asynchronous
        """
        if self._async_train_manager and self._async_train_manager.is_training:
            self._async_train_manager.stop()
            self._async_train_manager = None

    def wait_on_training(self, timeout):
        """ Wait on training to finish
                Args:
                    timeout: wait timeout
        """
        if self._async_train_manager and self._async_train_manager.is_training:
            self._async_train_manager.wait(timeout)

    def infer(self):
        """ Constructs `Infer` context manager which is used
        to manager `Model` inference
        """
        self._async_train_manager = None
        return Infer.from_config(self._config, env=self._env)

    @classmethod
    def from_config(cls, config_file_path):
        """ Builds `Agent` from a configuration
        file
            Args:
                config_file_path: path to train config

            Returns:
                `Agent` instance
        """
        config = parse_obj_from_file(config_file_path,
                                     config_pb2.Config)
        self = cls(config)
        return self
