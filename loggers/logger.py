import threading
import tensorflow as tf

""" Contains the acutal logger used that calls all the loggers
"""

# pylint: disable=too-few-public-methods
# reason-disable: dataclass
class LogElement:
    """ Represents an attr to log
    """
    def __init__(self, string, var_name):
        self._string = string
        self._var_name = var_name
        self.var = None

    def __str__(self):
        if self.var:
            return self._string % self.var
        return ""


# pylint: disable=too-few-public-methods
# reason-disable: dataclass
class LogVarType:
    """ Variable types for logging
    `Data class`
    """
    RETURNED_DICT = "returned_dict" # whether the variable is returned by function
    RETURNED_VALUE = "returned_value"
    INSTANCE_ATTR = "instance_attr" # or is an instance attribute of the method


class Logger(threading.Thread):
    """ The logging manager that
    logs LogElement's in loggers periodically.
    LogElement's are added via decorators
    in log_decorators
    """
    loggers = {}

    def __init__(self,
                 logging_event,
                 logging_sleep_cond,
                 logging_freq_sec):

        self._logging_event = logging_event
        self._logging_sleep_cond = logging_sleep_cond
        self._logging_freq_sec = logging_freq_sec

        super().__init__(target=self._log, group=None)

    def _log(self):
        """ logs periodically
        """
        while self._logging_event.is_set():
            self._logging_sleep_cond.acquire()

            self._logging_sleep_cond.wait(self._logging_freq_sec)
            # since we have the lock `update_vars`
            # can't be called (i.e. trainer won't call act_iteration)
            if self._logging_event.is_set():
                for log in self.loggers.values():
                    if str(log):
                        tf.logging.info(log)
            self._logging_sleep_cond.release()

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
