from google.protobuf.json_format import MessageToDict
import numpy as np
from advantage.elements.base.element import Element, NumpyElementMixin, NormalizingElementMixin

np_attr = NumpyElementMixin.np_attr

@Element.element
class Sarsa(Element, NumpyElementMixin, NormalizingElementMixin):
    """ Sarsa is an Element that contains
            state: agent previous state
            action: agent action taken
            reward: reward presented by env from action
            done: action resulted in game over
            next_state: the next state that action + state_transition led the agent to
            n_step_reward : possible bootstrapped discounted n-step reward from
                `state`
    """
    state = np_attr(np.float32)
    action = np_attr(np.float32)
    reward = np_attr(np.float32)
    done = np_attr(np.bool)
    next_state = np_attr(np.float32)

    @staticmethod
    def parse_env(env_dict):
        """ Parses environment state dictionary
        into the attributes to make a Sarsa

            Args:
                env_dict: return of Environment.step

            Return:
                dict containing kwargs to make Sarsa
        """

        state = env_dict["state"].astype(np.float32) if isinstance(env_dict["state"], np.ndarray) else np.array([env_dict["state"]], dtype=np.float32)
        action = env_dict["action"].astype(np.float32) if isinstance(env_dict["action"], np.ndarray) else np.array([env_dict["action"]], dtype=np.float32)
        reward = env_dict["reward"].astype(np.float32) if isinstance(env_dict["reward"], np.ndarray) else np.array([env_dict["reward"]], dtype=np.float32)
        done = np.array([env_dict["done"]], dtype=np.bool)
        next_state = env_dict["next_state"].astype(np.float32) if isinstance(env_dict["next_state"], np.ndarray) else np.array([env_dict["next_state"]], dtype=np.float32)

        return {"state": state, "action": action, "reward": reward, "done": done, "next_state": next_state}

    @staticmethod
    def proto_name_to_attr_dict():
        """ Creates dict for mapping key entries
        in protobuf file to actual attr names
        """
        return {"normalizeState" : "state",
                "normalizeAction" : "action",
                "normalizeReward" : "reward"}

    @classmethod
    def normalize_list_from_config(cls, config):
        """ Construct list of attrs to normalize
        from protobuf config
        """
        sarsa_as_dict = MessageToDict(config)
        attr_mapping = cls.proto_name_to_attr_dict()
        return [v for k, v in attr_mapping.items() if k not in sarsa_as_dict.keys()]

    @classmethod
    def make_element(cls,
                     state,
                     action,
                     reward,
                     done,
                     next_state):
        """ Makes Sarsa.
                Args:
                    the attr values

                Returns:
                    Sarsa
        """
        return cls(state=state,
                   action=action,
                   reward=reward,
                   done=done,
                   next_state=next_state)

    @classmethod
    def make_element_from_env(cls, env_dict):
        """Makes a Sarsa element from env_dict returned by from env_dict
                returned by Environment.step
                Args:
                    env_dict: dict returned by Environment.step act_in_env using an Environment

                Returns:
                    Sarsa
        """

        return cls.make_element(**cls.parse_env(env_dict))

    @classmethod
    def make_element_zero(cls, state_col_dim,
                          action_col_dim,
                          reward_col_dim,
                          next_state_col_dim):
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
                   n_step_reward=np.zeros((reward_col_dim,), dtype=np.float32))
