import tensorflow as tf
from advantage.agents.policies import ProbabilisticPolicy

""" Manages policy regularizers
"""

class Regularizer:
    """ Represents an RLFunction regularizer
    """
    def __init__(self, loss, weight):
        self._loss = loss
        self._weight = weight

    def __add__(self, other):
        return other + self._weight * self._loss

    @property
    def loss(self):
        """ property for `_loss`
        """
        return self._loss

    @property
    def weight(self):
        """ property for `_weight`
        """
        return self.weight


class L2Norm(Regularizer):
    """ Represents L2 Norm for
    `rl_func`
    """
    def __init__(self, rl_func, weight):

        loss = tf.nn.l2_loss(rl_func.trainable_parameters)

        super().__init__(self, loss, weight)

    def __call__(self, session, states):
        return session.run(self._loss, feed_dict={"state": states})

class Entropy(Regularizer):
    """ Represents Entropy of policy.
    Only defined for ProbabilisticPolicy
    """
    def __init__(self, rl_func, weight):

        if not isinstance(rl_func, ProbabilisticPolicy):
            raise ValueError("Entropy regularizer expected rl_func"
                             "to be of type ProbabilisticPolicy")

        self._rl_func = rl_func
        loss = rl_func.liklehood
        super().__init__(self, loss, weight)

    def __call__(self, session, states, actions):
        return self._rl_func.eval_liklehood(states, actions)


class KL(Regularizer):
    """ Represents the KL Divergence
    between `p` and `q`
    """
    def __init__(self, p, q, weight):
        loss = -p * Entropy(p, weight) + p * Entropy(q, weight)

        super().__init__(self, loss, weight)

class KLImprove(KL):
    """ Represents the KL Divergence
    between the old and new policy
    for regularizing policy improvement
    """
    def __init__(self, policy, weight):
        super().__init__(policy, policy.old)
