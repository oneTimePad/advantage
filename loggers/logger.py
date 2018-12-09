from datetime import datetime
import threading
import tensorflow as tf

""" Contains the acutal logger used that calls all the loggers
"""

# pylint: disable=too-few-public-methods
# reason-disable: dataclass
class LogElement:
    """ Represents an attr to log
    """
    def __init__(self,
                 string,
                 var_name,
                 stdout,
                 tensorboard):
        self._string = string
        self.var_name = var_name
        self.var = None
        self._stdout = stdout
        self._tensorboard = tensorboard

    def __str__(self):
        if self.var:
            return self._string % self.var
        return ""

    @property
    def to_stdout(self):
        """ property for determing
        to log to stdout
        """
        return self._stdout

    @property
    def to_tensorboard(self):
        """ property for determing
        to log to tensorboard
        """
        return self._tensorboard

# pylint: disable=too-few-public-methods
# reason-disable: dataclass
class LogVarType:
    """ Variable types for logging
    `Data class`
    """
    RETURNED_DICT = "returned_dict" # whether the variable is returned by function
    RETURNED_VALUE = "returned_value"
    INSTANCE_ATTR = "instance_attr" # or is an instance attribute of the method


def make_log_step(scope):
    """ Makes log_step variable
    counter
        Args:
            scope: model ScopeWrap

        Returns:
            update op for log_step
    """
    with scope(graph_only=True):
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            log_step = tf.get_variable("log_step",
                                       shape=(),
                                       initializer=tf.zeros_initializer(),
                                       trainable=False,
                                       dtype=tf.int32)
            return log_step, tf.assign(log_step, log_step + 1, name="increment_log_step")

class Logger(threading.Thread):
    """ The logging manager that
    logs LogElement's in loggers periodically.
    LogElement's are added via decorators
    in log_decorators
    """
    loggers = {}
    _loggers_unique_id = 0

    def __init__(self,
                 scope,
                 logging_event,
                 logging_sleep_cond,
                 logging_freq_sec,
                 file_writer):
        self._scope = scope
        self.session = None
        self._logging_event = logging_event
        self._logging_sleep_cond = logging_sleep_cond
        self._logging_freq_sec = logging_freq_sec
        log_step, log_step_update = make_log_step(scope)
        self._log_step = log_step
        self._log_step_update = log_step_update
        self._file_writer = file_writer

        super().__init__(target=self._log, group=None)

    @property
    def writer(self):
        """ property for `_file_writer`
        """
        return self._file_writer

    def _log(self):
        """ logs periodically
        """
        while self._logging_event.is_set():
            self._logging_sleep_cond.acquire()

            self._logging_sleep_cond.wait(self._logging_freq_sec)
            # since we have the lock `update_vars`
            # can't be called (i.e. trainer won't call act_iteration)
            if self._logging_event.is_set():
                step = self.session.run(self._log_step)
                print("%d ------------%s------------" % (step, datetime.now()))
                for log in self.loggers.values():
                    if log.to_stdout:
                        log_str = str(log)
                        if log_str:
                            tf.logging.info(" " + log_str)

                    if log.to_tensorboard and log.var:
                        self._file_writer.add_summary(log.var, step)
                self._file_writer.flush()

            self.session.run(self._log_step_update)
            self._logging_sleep_cond.release()

    @classmethod
    def add_logger(cls, log_element):
        """ Adds logger to dict of
        loggers

            Args:
                log_element: LogElement to add

            Returns:
                correct `var_name`
        """

        if not log_element.var_name:
            log_element.var_name = str(cls._loggers_unique_id)
            cls._loggers_unique_id += 1

        var_name = log_element.var_name
        cls.loggers.update({var_name: log_element})

        return var_name


    @classmethod
    def update_var(cls, var_name, new_value):
        """ Updates the var in the LogElement
        specified by var_name in `loggers`
        """
        cls.loggers[var_name].var = new_value

    @classmethod
    def get_var(cls, var_name):
        """ Gets the var in the LogElement
        specified by var_name in `loggers`

            Args:
                cls: Logger
                var_name: key for LogElement
                    in `loggers`

            Returns:
                var from LogElement
        """
        return cls.loggers[var_name].var
