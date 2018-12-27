import tensorflow as tf
from advantage.agents.base.base_agents import OffPolicyValueAgent
from advantage.utils.value_agent import bellman_operator, decayed_epsilon

class DeepQAgent(OffPolicyValueAgent):
    """ Implements the DeepQNetworks Agent. The DQN Agent
    utilizes two networks to stabilize Q-Learning for Deep RL approximators.
    A ExperienceReplayBuffer is utilize to allow for the I.I.D necessary condition
    for Neural Networks.
    """
    # pylint: disable=too-many-arguments
    # reason-disabled: argument format acceptable
    def __init__(self,
                 environment,
                 graph,
                 agent_scope,
                 discount_factor,
                 policy_q_network,
                 target_q_network,
                 epsilon):

        self._target = target_q_network

        self._epsilon = epsilon

        self._copy = None

        super().__init__(policy=policy_q_network,
                         target=target_q_network,
                         environment=environment,
                         graph=graph,
                         agent_scope=agent_scope,
                         discount_factor=discount_factor)



    def set_up_train(self):

        self.epsilon_func = decayed_epsilon(self,
                                            self._epsilon)


        target_net = self._target.network

        with self._agent_scope():
            # the bellman operator targets for training target network
            target_plh = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="target_plh")
            self._target.add_target_placeholder(target_plh)

            # the actions taken by the policy leading to the bellman transition
            action_taken_plh = tf.placeholder(shape=[None, 1],
                                              dtype=tf.int32,
                                              name="action_taken_plh")
            self._target.add_target_placeholder(action_taken_plh)

            # extract the Q-value for the action taken
            action_q = tf.reduce_sum(tf.one_hot(indices=action_taken_plh,
                                                depth=self.num_of_actions) * target_net,
                                     axis=1)

            # the mean square error between target and network
            loss = tf.reduce_mean(tf.square(target_plh - action_q), name="DeepQAgent_target_loss")

        self._target.minimize(loss)

        self._policy.initialize(self.session)
        self._target.initialize(self.session)

        self._copy = self._policy.make_copy_op(self.session,
                                               self._target)

    def set_up(self):
        self._policy.initialize(self.session)

    def evaluate_policy(self, state):
        return self._policy.inference(self.session, {"policy_state_plh" : [state]})

    def improve_policy(self):
        """Policy is improved by copying target params to policy network
        """
        self._copy()

    def improve_target(self, sarsa):
        """ Trains the Target Q-Network on I.I.D samples batch from the Replay Buffer
                Args:
                    Sarsa: object containing aggregated results
        """

        states, actions_taken, targets = bellman_operator(self.session,
                                                          self._policy,
                                                          sarsa,
                                                          self._discount_factor,
                                                          "policy_state_plh")

        feed_dict_in = {"tgt_state_plh" : states}

        feed_dict_target = {"target_plh" : targets, "action_taken_plh": actions_taken}

        self._target.update(self.session, feed_dict_in, feed_dict_target)
