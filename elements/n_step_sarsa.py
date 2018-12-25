import numpy as np
from .sarsa import Sarsa, np_attr
from advantage.elements.base.element import Element

""" Sarsa that accounts for the
n-step reward
"""

@Element.element
class NStepSarsa(Sarsa):
    """ NStepSarsa is an element
    similar to Sarsa that also accounts
    for the N-Step discounted reward
    """
    n_step_return = np_attr(np.float32)

    @staticmethod
    def proto_name_to_attr_dict():
        """ Creates dict for mapping key entries
        in protobuf file to actual attr names
        """

        return {"normalizeNStepReturn" : "n_step_return",
                **super(NStepSarsa, NStepSarsa).proto_name_to_attr_dict()}


    @classmethod
    def make_element(cls,
                     state,
                     action,
                     reward,
                     done,
                     next_state,
                     n_step_return):

        """ Makes NStepSarsa.
                Args:
                    the attr values

                Returns:
                    NStepSarsa
        """
        return cls(state=state,
                   action=action,
                   reward=reward,
                   done=done,
                   next_state=next_state,
                   n_step_return=n_step_return)


    @classmethod
    def make_element_from_env(cls, env_dict):
        """ Makes NStepSarsa from env_dict
        returned by Environment.step

            Args:
                env_dict: dict returned by Environment.step act_in_env using an Environment

            Returns:
                NStepSarsa
        """
        sarsa_kwargs = super().parse_env(env_dict)
        n_step_return = np.copy(sarsa_kwargs["reward"])

        return cls(n_step_return=n_step_return,
                   **sarsa_kwargs)
