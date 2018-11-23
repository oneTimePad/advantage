from inspect import isclass
from functools import wraps

""" Utilities for decorators
"""

def parameterized(decorator):
    """ Defines a decorator as one which (might) takes parameters
            Args:
                decorator: a decorator

            Returns:
                decorator with optional argument
                capabilities
    """
    @wraps(decorator)
    def param_decorator(*args, **kwargs):
        if len(args) == 1 and not kwargs and (callable(args[0]) or isclass(args[0])):
            return decorator(args[0])

        return lambda wrapped: decorator(wrapped, *args, **kwargs)

    return param_decorator
