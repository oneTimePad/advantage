from .element import Element, NumpyElementMixin, NormalizingElementMixin
import numpy as np

np_attr = NumpyElementMixin.np_attr
@Element.element
class Sarsa(Element, NumpyElementMixin, NormalizingElementMixin):
    """ Sarsa is an Element that contains
            state: agent previous state
            action: agent action taken
            reward: reward presented by env from action
            done: action resulted in game over
            next_state: the next state that action + state_transition led the agent to
            next_action: the next action chosen by the agent in this new state
    """
    state = np_attr(np.float32)
    action = np_attr(np.float32)
    reward = np_attr(np.float32)
    done = np_attr(np.bool)
    next_state = np_attr(np.float32)
    next_action = np_attr(np.float32)

    @classmethod
    def make_element(cls, state,
                action,
                reward,
                done,
                next_state,
                next_action=np.array([0.0], dtype=np.float32)):
        """ Makes Sarsa. next_action is defaulted to zero because it is rarely used.
                Args:
                    the attr values

                Returns:
                    Sarsa
        """
        return cls(state=state, action=action,
                    reward=reward,
                    done=done,
                    next_state=next_state,
                    next_action=next_action)

    @classmethod
    def make_element_zero(cls, state_col_dim,
                    action_col_dim,
                    reward_col_dim,
                    next_state_col_dim,
                    next_action_col_dim=1):
        """ Makes a 'zeroed' out Sarsa. Again next_action has default.
                Args:
                    the column dimensions of the numpy arrays
                        (done is always just a one element array)
                Returns:
                    Sarsa
        """
        return cls(state=np.zeros((state_col_dim,), dtype=np.float32),
                    action=np.zeros((action_col_dim,), dtype=np.float32),
                    reward=np.zeros((reward_col_dim,), dtype=np.float32),
                    done=np.array([False]),
                    next_state=np.zeros((next_state_col_dim,), dtype=np.float32),
                    next_action=np.zeros((next_action_col_dim,), dtype=np.float32))
