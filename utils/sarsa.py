import attr
import numpy as np
from kwargs_only import kwargs_only

@attr.s
class Sarsa(object):
    """ Sarsa object to put in experience replay buffer """
    state = attr.ib()
    action = attr.ib()
    reward = attr.ib()
    done = attr.ib()
    next_state = attr.ib()
    next_action = attr.ib()

    @classmethod
    def are_compatible(cls, a, b):
        """Determines if a and b's attributes are compatible"""
        result = True
        for a_attr, b_attr in zip(cls.unzip(a), cls.unzip(b)):
            result = result and type(a_attr) == type(b_attr) and ( (isinstance(a_attr, np.ndarray) and a_attr.shape == b_attr.shape) \
                or not isinstance(a_attr, np.ndarray))

            if not result:
                return False

        return True

    @classmethod
    def unzip(cls, sarsa):
        return sarsa.state, sarsa.action, sarsa.reward, sarsa.done, sarsa.next_state, sarsa.next_action


    @classmethod
    def make(cls, state=0.0,
                action=0.0,
                reward=0.0,
                done=False,
                next_state=0.0,
                next_action=0.0):
        """ Makes Sarsa with default args (suggested method to create) """
        return cls(state, action,
                    reward,
                    done,
                    next_state,
                    next_action)

    @classmethod
    def zero_initialize(cls, state_size=None,
                        action_size=None,
                        reward_size=None,
                        #done_size=None,
                        next_state_size=None,
                        next_action_size=None):

        return cls(np.zeros((state_size), dtype=np.float32) if state_size is not None else 0.0,
                    np.zeros((action_size,), dtype=np.float32) if action_size is not None else 0.0,
                    np.zeros((reward_size,), dtype=np.float32) if reward_size is not None else 0.0,
                    False,#np.zeros((done_size,), dtype=np.float32) if done_size is not None else 0.0,
                    np.zeros((next_state_size,), dtype=np.float32) if next_state_size is not None else 0.0,
                    np.zeros((next_action_size,), dtype=np.float32) if next_action_size is not None else 0.0)

    @classmethod
    def split_list_to_np(cls, sarsas_list):
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        next_actions = []
        for s in sarsas_list:
            states.append(s.state)
            actions.append(s.action)
            rewards.append(s.reward)
            dones.append(s.done)
            next_states.append(s.next_state)
            next_actions.append(s.next_action)

        return tuple(map(np.vstack, [states, actions, rewards, dones, next_states, next_actions]))


    @classmethod
    def make_normalization_stats(cls, state_size=None,
                                    action_size=None,
                                    reward_size=None,
                                    #done_size=None,
                                    next_state_size=None,
                                    next_action_size=None):
        """ Creates stats for normalizing Sarsa
                Returns:
                    running_sum, running_sum_of_squares
        """
        return tuple([cls.zero_initialize(state_size, action_size,
                                    reward_size,
                                    #done_size,
                                    next_state_size,
                                    next_action_size)] * 2)

    @classmethod
    def update_normalization_stats(cls, stats, new_sarsa):
        """ Updates the normalization stats in stats args

                Args:
                    stats: tuple of normalization stats created by make_normalization_stats
                    new_sarsa: new sarsa element

                Returns:
                    updated stats

                Raises:
                    ValueError: bad arguments
        """
        sums, sums_sqr = stats
        return cls(state=sums.state + new_sarsa.state,
                    action=sums.action + new_sarsa.action,
                    reward=sums.reward + new_sarsa.reward,
                    done=None,
                    next_state=sums.next_state + new_sarsa.next_state,
                    next_action=sums.next_action + new_sarsa.next_action), \
                cls(state=sums_sqr.state + new_sarsa.state ** 2,
                    action=sums_sqr.action + new_sarsa.action ** 2,
                    reward=sums_sqr.reward + new_sarsa.reward ** 2,
                    done=None,
                    next_state=sums_sqr.next_state + new_sarsa.next_state ** 2,
                    next_action=sums_sqr.next_action + new_sarsa.next_action ** 2)

    @classmethod
    def running_variance(cls, sums, sums_sqr, num):
        """Computes running variance from running sum of data, running sum of squares of data and data count

            Args:
                sums: running sum of data
                sums_sqr: running sum of square of data
                num: running count of data

            Returns:
                running variance
        """

        return (1 / num) * (sums_sqr - ( (sums ** 2) / num) )


    @classmethod
    def normalize(cls, stats, running_num_sarsa, sarsa_list, normalize=(), eps=0.01):
        """Normalizes a list of sarsa and returns the split list into sarsa components

            Args:
                stats: the stats tuple returned from make_normalization_stats
                running_num_sarsa: running count of sarsa objects, could be len of sarsa_list
                sarsa_list: list of Sarsas to normalize
                normalize: tuple of str specifying which attributes of Sarsas to normalize
                eps: variance div by zero protection

            Returns:
                split list of states, actions, rewards, done, next_states, next_actions as np.array

            Raises:
                ValueError: for bad args
        """

        sums, sums_sqr = stats

        if  not isinstance(stats, tuple) or not isinstance(sums, cls) or not isinstance(sums_sqr, cls):
            raise ValueError("stats must be tuple returned by make_normalization_stats call and updated by update_normalize_stats")

        if not isinstance(sarsa_list, list):
            raise ValueError("sarsa_list must be of type list: representing list of Sarsas")


        states, actions, rewards, dones, next_states, next_actions = cls.split_list_to_np(sarsa_list)

        if "state" in normalize:
            state_mean = (1 / running_num_sarsa) * sums.state
            state_var = cls.running_variance(sums.state, sums_sqr.state, running_num_sarsa)

            states = (states - state_mean) / np.sqrt(np.maximum(state_var, eps))

        if "action" in normalize:
            action_mean = (1 / running_num_sarsa) * sums.action
            action_var = cls.running_variance(sums.action, sums_sqr.action, running_num_sarsa)

            actions = (actions - action_mean) / np.sqrt(np.maximum(action_var, eps))

        if "reward" in normalize:
            reward_mean = (1 / running_num_sarsa) * sums.reward
            reward_var = cls.running_variance(sums.reward, sums_sqr.reward, running_num_sarsa)

            rewards = (rewards - reward_mean) / np.sqrt(np.maximum(reward_var, eps))

        if "next_state" in normalize:
            next_state_mean = (1 / running_num_sarsa) * sums.next_state
            next_state_var = cls.running_variance(sums.next_state, sums_sqr.next_state, running_variance)

            next_states = (next_states - next_state_mean) / np.sqrt(np.maximum(next_state_var, eps))

        if "next_action" in normalize:
            next_action_mean = (1 / running_num_sarsa) * sums.next_action
            next_action_var = (1 / running_num_sarsa) * sums.next_action

            next_actions = (next_actions - next_action_mean) / np.sqrt(np.maximum(next_action_var, eps))


        return states, actions, rewards, dones, next_states, next_actions
