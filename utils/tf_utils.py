from functools import reduce
from contextlib import ExitStack
import tensorflow as tf
import os
import itertools

""" TensorFlow related utils
"""


def build_init_uninit_op(session, variables):
    """
        Initializes all uninitalized in list `variables`
            Args:
                session: session to initialize in
                variables: list of variables to check
            Returns:
                initialization op
    """
    # credit to https://tinyurl.com/initialize-unitialized-vars
    is_not_initialized = session.run([tf.is_variable_initialized(var) for var in variables])
    not_initialized_vars = [v for (v, f) in zip(variables, is_not_initialized) if not f]

    return tf.variables_initializer(not_initialized_vars) if not_initialized_vars else None


def strip_and_replace_scope(name_scope, variable, suffix_start):
    """ Strip highest scope of  and replace with self's var_scope.name as
    new highest scope.
            Args:
                name_scope: new upper scope
                variable: variable string name
                suffix_start: start of suffix to keep

            Returns:
                new fully-qualified variable name
    """
    combine = lambda l: reduce(lambda x, y: os.path.join(x, y), l)

    suffix = combine(variable.split("/")[suffix_start:])

    return combine([name_scope, suffix])

class ScopeWrap:
    """ Wraps the graph.as_default
    and variable_scope context managers
    into one that can be used to add
    new variables to the graph and scope
    """
    def __init__(self, graph, name, reuse):
        self._scope = None
        self._name = name
        self._graph = graph
        self._reuse = reuse

    def __call__(self):
        """ Combines graph and var_scope into one
        Context Manager
        """
        ctx = ExitStack()
        ctx.enter_context(self._graph.as_default())
        if not self._scope:
            self._scope = ctx.enter_context(tf.variable_scope(self._name,
                                                              reuse=self._reuse))

        else:
            ctx.enter_context(tf.variable_scope(self._scope, auxiliary_name_scope=False))
            ctx.enter_context(tf.name_scope(self._scope.original_name_scope))

        return ctx

    @property
    def name_scope(self):
        """ Returns name_scope
        """
        return self._name

    @property
    def graph(self):
        """ property for `_graph`
        """
        return self._graph

    @property
    def reuse(self):
        """ property for `_reuse`
        """
        return self._reuse

    @classmethod
    def build(cls, upper_scope, new_scope):
        """ Creates ScopeWrap by combing
        an upper scope with the scope str
        `new_scope`
        """
        name_scope = os.path.join(upper_scope.name_scope,
                                  new_scope)
        return cls(upper_scope.graph,
                   name_scope,
                   upper_scope.reuse)
