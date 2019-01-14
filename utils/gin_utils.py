from advantage.utils.decorator_utils import parameterized
import gin

""" Utilities for gin configuration
"""

def gin_bind(cls, parameter, value, name_scope=""):
    """ Shortcut for using gin.bind_parameter
            Args:
                cls: class running from
                parameter: gin param name
                value: parameter value
                name_scope: optional higher name_scope
    """
    param_str = "{0}/{1}/{2}".format(name_scope, cls.name_scope, parameter)
    gin.bind_parameter(param_str, value)

def gin_bind_init(cls, parameter, value, name_scope=""):
    """ Shortcut for using gin.bind_parameter for classes
            Args:
                cls: class running from
                parameter: gin param name
                value: parameter value
                name_scope: optional higher name_scope
    """
    param_str = "{0}/{1}/{2}.{3}".format(name_scope, cls.name_scope, cls.__name__, parameter)
    gin.bind_parameter(param_str, value)

@parameterized
def gin_classmethod(func, *args, **kwargs):
    """ Enables gin configuration for classmethods
            Args:
                func: classmethod to configure
            Returns
                decorated classmethod
    """
    gin_func = gin.configurable(func, *args, **kwargs)
    clsmth = classmethod(gin_func)
    clsmth.__name__ = func.__name__
    return clsmth
