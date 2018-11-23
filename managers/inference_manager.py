import tensorflow as tf
from advantage.builders import build_model, build_environment

""" Contains the Inference manager for running a
Model in an Environment
"""

class Infer:
    """ Context Manager for Inference
    """


    def __init__(self, model, env):
        self._model = model
        self._env = env
        self._set_up = False


    def __enter__(self):
        self._model.set_up()

        return self

    # credit to https://tinyurl.com/exit-best-practices
    def __exit__(self, exception_type, exception_value, traceback, shutdown=True):
        if isinstance(exception_type, KeyboardInterrupt):
            tf.logging.warn("Inference ended by user")
            shutdown = True
        elif isinstance(exception_type, Exception):
            tf.logging.fatal("Found Exception %s" % exception_type)
            tf.logging.fatal("Exception value %s" % exception_value)
            tf.logging.fatal("Exception traceback %s" % traceback)

        if shutdown:
            self.shutdown()


    def shutdown(self):
        """ Cleans up model
        """
        self._model.clean()

    def as_default(self):
        """ Allows `Infer` to be used
        multiple times without shutdowning
        down at end of `with`
        """
        infer = self
        class _InferProxy:
            """ Wraps `Infer` to allow caller
            to use `Infer` multiple times in `with`
            without shutdowning down
            """
            def __init__(self):
                self.__class__.__name__ = infer.__class__.__name__
                self.__class__.__doc__ = infer.__class__.__doc__

            def __enter__(self):
                nonlocal infer

                return infer

            def __exit__(self, *args):
                nonlocal infer
                infer.__exit__(*args, shutdown=False)

        return _InferProxy()


    def run_trajectory(self, run_through=False):
        """ Runs a trajectory in the environment
        with a model
            Args:
                run_through: whether to run through
                    until termination without yielding
        """
        if run_through:
            return self._model.run_trajectory_through()
        return self._model.run_trajectory()

    @classmethod
    def from_config(cls, config, env=None):
        """ Builds `Infer` object from config
                Args:
                    config: the training configuration
                    env: optional `Environment` object

                Returns:
                    `Infer` object
        """
        if not env:
            env = build_environment(config.environment)
        model = build_model(config.model, env, False)
        model.checkpoint_dir_path = config.checkpoint_dir_path
        model.checkpoint_file_prefix = config.checkpoint_file_prefix

        return cls(model, env)
