import types
from advantage.loggers.logger import Logger, LogVarType, LogElement
from advantage.utils.decorator_utils import parameterized

""" Decorators to enable logging during training/inference
of specific attributes. These can occur on the call of an attr
and even be put in TensorFlow variables to be logged to Tensorboard
or stdout
"""

def extract_return(decorated,
                   ret,
                   var_type,
                   var_name,
                   dec_first_arg):
    """ Extracts attr to log from return
    of `decorated` func. The return value is
    pass in as `ret`

        Args:
            ret: return value from `decorated`
            var_type: type specified by LogVarType cls attrs
            var_name: name of variable (for dict and instance var)
            dec_first_arg: first argument passed to `decorated` if any
                for instance var logging only. This represents `self`

        Returns:
            value fetched from `ret`

        Raises:
            AttributeError on looking for an unknown instance attr
                of dec_first_arg (which is supposed to be `self`)
    """
    if var_type == LogVarType.RETURNED_DICT:
        value = ret[var_name]
    elif var_type == LogVarType.RETURNED_VALUE:
        value = ret
    elif var_type == LogVarType.INSTANCE_ATTR:
        value = getattr(dec_first_arg, var_name, None)

        if not value:
            raise AttributeError("`logger` decorator of %s"
                                 " specifies `LogVarType.INSTANCE_ATTR`"
                                 " but instance doesn't have attribute %s"
                                 % (decorated.__name__, var_name))
    else:
        raise ValueError("var_type must be value attribute of `LogVarType`")

    return value

def _log_as_generator(gen, log, dec_first_arg):
    """ Used track values returned
    from a generator. Used for real-time
    tracking.

        Args:
            gen: generator
            func: logging function

        Yields: generator yielded value
    """
    for value in gen:
        log(value, dec_first_arg)
        yield value


@parameterized
def _logger(logger,
            func,
            log_string,
            var_type,
            var_name,
            when):
    """ Marks a decorator as logger decorator
    This must decorate the `wrapping` function of
    the decorator NOT the decorator itself

        Args:
            logger: the `wrapping` function of the decorator
            func: the function/method that logger is wrapping
                (i.e.) first arg passed to the decorator
                for the `wrapping` function
            log_string: format string to print while logging
            var_type: LogVarType
            when: tuple(LogVarType, var_name, predicit_func):
                used to determing when to call `logger`
    """

    log_element = LogElement(log_string,
                             var_name)

    Logger.loggers.update({var_name: log_element})

    def extract_and_track(value, first_arg):

        extracted_value = extract_return(func,
                                         value,
                                         var_type,
                                         var_name,
                                         first_arg)

        if when:
            when_var_type, when_var_name, predicate = when
            when_value = extract_return(func,
                                        value,
                                        when_var_type,
                                        when_var_name,
                                        first_arg)
            if predicate(when_value):
                logger(extracted_value)

    def wrap_func(*args, **kwargs):

        ret = func(*args, **kwargs)

        dec_first_arg = args[0] if args else None

        if isinstance(ret, types.GeneratorType):
            return _log_as_generator(ret, extract_and_track, dec_first_arg)

        extract_and_track(ret, dec_first_arg)
        return ret

    return wrap_func

@parameterized
def avg(func,
        log_string,
        var_type,
        var_name,
        when=(),
        tensorboard=False):
    """ Decorator for tracking averages

        Args:
            func: decorated method
            log_string: format string for logging
            var_type: LogVarType
            var_name: tracked attr name
            when: tuple(LogVarType, var_name, predicit_func)
                used to determine when to call `avg_logger`
            tensorboard: whether to log to tensorboard
    """

    count = 0

    @_logger(func, log_string, var_type, var_name, when)
    def avg_logger(ret_value):
        nonlocal count

        count += 1

        cur_var = Logger.get_var(var_name)
        if cur_var:
            new_var = cur_var + (ret_value - cur_var) / count
        else:
            new_var = ret_value

        Logger.update_var(var_name, new_var)

    return avg_logger
