import unittest
import numpy as np
from utils.sarsa import Sarsa



class TestSarsa(unittest.TestCase):
    """Test cases for Sarsa object"""

    def setUp(self):
        self.stats = None
        self.sarsas = None

    def test_zero_initialize(self):
        s = Sarsa.zero_initialize(state_size=1,
                                    action_size=2,
                                    reward_size=3,
                                    #done_size=4,
                                    next_state_size=5)

        np.testing.assert_array_equal(s.state, np.array([0.0]))
        np.testing.assert_array_equal(s.action, np.array([0.0, 0.0]))
        np.testing.assert_array_equal(s.reward, np.array([0.0, 0.0, 0.0]))
        #np.testing.assert_array_equal(s.done, np.array([0.0, 0.0, 0.0, 0.0]))
        self.assertEqual(s.done, False)
        np.testing.assert_array_equal(s.next_state, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertEqual(s.next_action, 0)

    def test_split_list_to_np(self):
        s_list = [Sarsa.zero_initialize(state_size=1,
                                    action_size=2,
                                    reward_size=3,
                                    #done_size=4,
                                    next_state_size=5),
                    Sarsa.make(np.array([1.0]),
                            np.array([1.0] * 2),
                            np.array([1.0] * 3),
                            False,#np.array([1.0] * 4),
                            np.array([1.0] * 5))]
        states, actions, rewards, dones, next_states, next_actions = Sarsa.split_list_to_np(s_list)

        np.testing.assert_array_equal(states, np.array([[0.0], [1.0]]))
        np.testing.assert_array_equal(actions, np.array([[0.0, 0.0], [1.0, 1.0]]))
        np.testing.assert_array_equal(rewards, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        np.testing.assert_array_equal(dones, np.array([[False], [False]]))#np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]))
        np.testing.assert_array_equal(next_states, np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]))
        np.testing.assert_array_equal(next_actions, np.array([[0], [0]]))

    def test_update_normalization_stats(self):
        stats = Sarsa.make_normalization_stats(state_size=1,
                                    action_size=2,
                                    reward_size=3,
                                    #done_size=4,
                                    next_state_size=5)

        new_sarsa = Sarsa.make(np.array([2.0]),
                                    np.array([2.0] * 2),
                                    np.array([2.0] * 3),
                                    False,#np.array([2.0] * 4),
                                    np.array([2.0] * 5))
        stats = Sarsa.update_normalization_stats(stats, new_sarsa)


        sums, sums_sqr = stats

        np.testing.assert_array_equal(sums.state, np.array([2.0]))
        np.testing.assert_array_equal(sums.action, np.array([2.0, 2.0]))
        np.testing.assert_array_equal(sums.reward, np.array([2.0, 2.0, 2.0]))
        self.assertEqual(sums.done, None)
        np.testing.assert_array_equal(sums.next_state, np.array([2.0, 2.0, 2.0, 2.0, 2.0]))
        self.assertEqual(sums.next_action, 0)

        np.testing.assert_array_equal(sums_sqr.state, np.array([4.0]))
        np.testing.assert_array_equal(sums_sqr.action, np.array([4.0, 4.0]))
        np.testing.assert_array_equal(sums_sqr.reward, np.array([4.0, 4.0, 4.0]))
        self.assertEqual(sums.done, None)
        np.testing.assert_array_equal(sums_sqr.next_state, np.array([4.0, 4.0, 4.0, 4.0, 4.0]))
        self.assertEqual(sums_sqr.next_action, 0)

    def test_normalize(self):
        sarsas = []

        stats = Sarsa.make_normalization_stats(state_size=1,
                                    action_size=2,
                                    reward_size=3,
                                    #done_size=4,
                                    next_state_size=5)

        sarsa1 = Sarsa.make(np.array([2.0]),
                                    np.array([2.0] * 2),
                                    np.array([2.0] * 3),
                                    True,#np.array([1.0] * 4),
                                    np.array([2.0] * 5))

        stats = Sarsa.update_normalization_stats(stats, sarsa1)

        sarsas.append(sarsa1)

        sarsa2 = Sarsa.make(np.array([3.0]),
                                    np.array([3.0] * 2),
                                    np.array([3.0] * 3),
                                    False,#np.array([2.0] * 4),
                                    np.array([3.0] * 5))

        sarsas.append(sarsa2)

        stats = Sarsa.update_normalization_stats(stats, sarsa2)

        sarsa3 = Sarsa.make(np.array([4.0]),
                                    np.array([4.0] * 2),
                                    np.array([4.0] * 3),
                                    True,#np.array([3.0] * 4),
                                    np.array([4.0] * 5))

        sarsas.append(sarsa3)

        stats = Sarsa.update_normalization_stats(stats, sarsa3)

        states, actions, rewards, dones, next_states, next_actions = Sarsa.split_list_to_np(sarsas)

        states_n, actions_n, rewards_n, dones_n, next_states_n, next_actions_n = Sarsa.normalize(stats, 3, sarsas, normalize=("state", "reward"))


        states_mean = (sarsa1.state + sarsa2.state + sarsa3.state) / 3.0
        states_var = (((sarsa1.state - states_mean) ** 2) + ((sarsa2.state - states_mean) ** 2) + ((sarsa3.state - states_mean) ** 2)) / 3.0

        states_norm = (states - states_mean) / np.sqrt(states_var)

        rewards_mean = (sarsa1.reward + sarsa2.reward + sarsa3.reward) / 3.0
        rewards_var = (((sarsa1.reward - rewards_mean) ** 2) + ((sarsa2.reward - rewards_mean) ** 2) + ((sarsa3.reward - rewards_mean) ** 2)) / 3.0

        rewards_norm = (rewards - rewards_mean) / np.sqrt(rewards_var)



        np.testing.assert_array_equal(states_norm, states_n)
        np.testing.assert_array_equal(rewards_norm, rewards_n)

        np.testing.assert_array_equal(dones_n, np.array([[True], [False], [True]]))#np.array([ [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]))
        np.testing.assert_array_equal(next_states_n, np.array([ [2.0, 2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0, 4.0]]))
        np.testing.assert_array_equal(actions_n, np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]))
        np.testing.assert_array_equal(next_actions_n, np.array([[0], [0], [0]]))





unittest.main()
