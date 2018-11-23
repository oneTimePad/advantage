import unittest
import numpy as np
from elements.sarsa import Sarsa



class TestSarsa(unittest.TestCase):
    """Test cases for Sarsa object
        This also testing methods of 'Element', 'NumpyElementMixin', and
        'NormalizingElementMixin'.
    """

    def setUp(self):
        self.stats = None
        self.sarsas = None
    """
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
    """

    def test_make_element_zero(self):
        s = Sarsa.make_element_zero(state_col_dim=2,
                                    action_col_dim=3,
                                    reward_col_dim=4,
                                    next_state_col_dim=5)

        np.testing.assert_array_equal(s.state, np.array([0.0, 0.0]))
        np.testing.assert_array_equal(s.action, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(s.reward, np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(s.done, np.array([False]))
        np.testing.assert_array_equal(s.next_state, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(s.next_action, np.array([0.0]))

    def test_make_element(self):
        s = Sarsa.make_element(np.array([1.0], dtype=np.float32),
                                np.array([2.0], dtype=np.float32),
                                np.array([3.0], dtype=np.float32),
                                np.array([True], dtype=np.bool),
                                np.array([4.0], dtype=np.float32))


        np.testing.assert_array_equal(s.state, np.array([1.0]))
        np.testing.assert_array_equal(s.action, np.array([2.0]))
        np.testing.assert_array_equal(s.reward, np.array([3.0]))
        np.testing.assert_array_equal(s.done, np.array([True]))
        np.testing.assert_array_equal(s.next_state, np.array([4.0]))
        np.testing.assert_array_equal(s.next_action, np.array([0.0]))

    def test_reduce(self):
        s1 = Sarsa.make_element(np.array([1.0], dtype=np.float32),
                                np.array([2.0], dtype=np.float32),
                                np.array([3.0], dtype=np.float32),
                                np.array([True], dtype=np.bool),
                                np.array([4.0], dtype=np.float32))

        s2 = Sarsa.make_element(np.array([4.0], dtype=np.float32),
                                np.array([3.0], dtype=np.float32),
                                np.array([2.0], dtype=np.float32),
                                np.array([False], dtype=np.bool),
                                np.array([1.0], dtype=np.float32))
        bad_s = Sarsa.make_element(np.array([4.0], dtype=np.float32),
                                np.array([3.0], dtype=np.float32),
                                np.array([2.0], dtype=np.float32),
                                np.array([False, True], dtype=np.bool),
                                np.array([1.0], dtype=np.float32))

        try:
            reduced = Sarsa.reduce([s1, bad_s])
            self.assertEqual(1, 0)
        except ValueError:
            self.assertEqual(1, 1)
        except Exception:
            self.assertEqual(1, 0)

        reduced = Sarsa.reduce([s1, s2])
        np.testing.assert_array_equal(reduced.state, np.array([[1.0],[4.0]]))
        np.testing.assert_array_equal(reduced.action, np.array([[2.0],[3.0]]))
        np.testing.assert_array_equal(reduced.reward, np.array([[3.0],[2.0]]))
        np.testing.assert_array_equal(reduced.done, np.array([[True], [False]]))
        np.testing.assert_array_equal(reduced.next_state, np.array([[4.0], [1.0]]))
        np.testing.assert_array_equal(reduced.next_action, np.array([[0.0], [0.0]]))


    def test_update_normalization_stats(self):
        stats = Sarsa.make_stats(normalize_attrs=("state", "reward"),
                                state_col_dim=1,
                                action_col_dim=2,
                                reward_col_dim=3,
                                next_state_col_dim=5)

        self.assertEqual("action" in stats, False)
        self.assertEqual("done" in stats, False)
        self.assertEqual("next_action" in stats, False)

        np.testing.assert_array_equal(stats["state"][0], np.array([0.0]))
        np.testing.assert_array_equal(stats["state"][1], np.array([0.0]))

        np.testing.assert_array_equal(stats["reward"][0], np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(stats["reward"][1], np.array([0.0, 0.0, 0.0]))

        s1 = Sarsa.make_element(state=np.array([2.0], dtype=np.float32),
                                action=np.array([2.0] * 2, dtype=np.float32),
                                reward=np.array([2.0] * 3, dtype=np.float32),
                                done=np.array([False], dtype=np.bool),
                                next_state=np.array([2.0] * 5, dtype=np.float32))


        Sarsa.update_normalize_stats(stats, s1)

        s2 = Sarsa.make_element(state=np.array([3.0], dtype=np.float32),
                                action=np.array([3.0] * 2, dtype=np.float32),
                                reward=np.array([3.0] * 3, dtype=np.float32),
                                done=np.array([True], dtype=np.bool),
                                next_state=np.array([3.0] * 5, dtype=np.float32))

        Sarsa.update_normalize_stats(stats, s2)

        s3 = Sarsa.make_element(state=np.array([4.0], dtype=np.float32),
                                action=np.array([4.0] * 2, dtype=np.float32),
                                reward=np.array([4.0] * 3, dtype=np.float32),
                                done=np.array([False], dtype=np.bool),
                                next_state=np.array([4.0] * 5, dtype=np.float32))

        Sarsa.update_normalize_stats(stats, s3)

        sqr = (2.0 **2 + 3.0 ** 2 + 4.0 ** 2)
        np.testing.assert_array_equal(stats["state"][0], np.array([9.0]))
        np.testing.assert_array_equal(stats["state"][1], np.array([sqr]))

        np.testing.assert_array_equal(stats["reward"][0], np.array([9.0, 9.0, 9.0]))
        np.testing.assert_array_equal(stats["reward"][1], np.array([sqr, sqr, sqr]))

        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.stats = stats

    def test_normalize_element(self):
        self.test_update_normalization_stats()

        reduced_obj = Sarsa.reduce([self.s1, self.s2, self.s3])

        normalized = Sarsa.normalize_element(self.stats, 3, reduced_obj)


        mean_v = (2.0 + 3.0 + 4.0) / 3.0
        var_v = (((2.0 - mean_v) ** 2 + (3.0 - mean_v) ** 2 + (4.0 - mean_v) ** 2)) / 3.0

        state_mean = np.array([mean_v], dtype=np.float32)
        state_var = np.array([var_v], dtype=np.float32)

        reward_mean = np.array([mean_v, mean_v, mean_v], dtype=np.float32)
        reward_var = np.array([var_v, var_v, var_v], dtype=np.float32)

        s1_state = (self.s1.state - state_mean) / np.sqrt(state_var)
        s2_state = (self.s2.state - state_mean) / np.sqrt(state_var)
        s3_state = (self.s3.state - state_mean) / np.sqrt(state_var)

        s1_reward = (self.s1.reward - reward_mean) / np.sqrt(reward_var)
        s2_reward = (self.s2.reward - reward_mean) / np.sqrt(reward_var)
        s3_reward = (self.s3.reward - reward_mean) / np.sqrt(reward_var)

        norm_states = np.vstack([s1_state, s2_state, s3_state])
        norm_rewards = np.vstack([s1_reward, s2_reward, s3_reward])

        actions = np.vstack([self.s1.action, self.s2.action, self.s3.action])
        next_states = np.vstack([self.s1.next_state, self.s2.next_state, self.s3.next_state])
        next_actions = np.vstack([self.s1.next_action, self.s2.next_action, self.s3.next_action])
        dones = np.vstack([self.s1.done, self.s2.done, self.s3.done])


        np.testing.assert_array_equal(normalized.state, norm_states)
        np.testing.assert_array_equal(normalized.reward, norm_rewards)
        np.testing.assert_array_equal(normalized.action, actions)
        np.testing.assert_array_equal(normalized.next_state, next_states)
        np.testing.assert_array_equal(normalized.next_action, next_actions)
        np.testing.assert_array_equal(normalized.done, dones)


unittest.main()
