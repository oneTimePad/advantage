from advantage.models.base.base_models import LearningModel
from advantage.agents import DeepQAgent
from advantage.elements import Sarsa
from advantage.checkpoint import checkpointable

@checkpointable
class DeepQModel(LearningModel):
    """ Model for Deep-Q Networks """

    def __init__(self,
                 graph,
                 environment,
                 model_scope,
                 agent,
                 improve_policy_modulo,
                 steps_for_act_iter,
                 replay_buffer,
                 sarsa_attrs_to_normalize,
                 improve_target_modulo,
                 iterations_of_improvement,
                 batch_size,
                 train_sample_less):

        self._replay_buffer = replay_buffer

        self._improve_target_modulo = improve_target_modulo

        self._iterations_of_improvement = iterations_of_improvement

        self._batch_size = batch_size

        self._sarsa_attrs_to_normalize = sarsa_attrs_to_normalize

        self._train_sample_less = train_sample_less

        self._norm_stats = None

        if not isinstance(agent, DeepQAgent):
            raise ValueError("Agent must be of type DeepQAgent but is %s" % type(agent))

        super().__init__(graph,
                         environment,
                         model_scope,
                         agent,
                         improve_policy_modulo,
                         steps_for_act_iter)
    @property
    def replay_buffer(self):
        """ Allow read access to replay_buffer """
        return self._replay_buffer

    @property
    def steps(self):
        """ Returns the number of steps the DQNAgent takes
        """
        return self._agent.total_steps

    def set_up_train(self):
        dims = self._environment.dims

        self._norm_stats = Sarsa.make_stats(normalize_attrs=self._sarsa_attrs_to_normalize,
                                            **dims)

        self.add_session(self._agent)

        # pylint: disable=W0201
        # reason disabled: defined by decorator
        self.restore_session = self._agent.session

        self._agent.set_up_train()

    def set_up(self):
        self.add_session(self._agent)

        # pylint: disable=W0201
        # reason disabled: defined by decorator
        self.restore_session = self._agent.session

        self._agent.set_up()

    def act_iteration(self):
        """ Runs the agent and collects Sarsas to put in the replay buffer
        """

        traj_rewards = []
        env_dict = {"done" : True}
        for _, env_dict in self._agent.act_for_steps(self._steps_for_act_iter, training=True):
            sarsa = Sarsa.make_element_from_env(env_dict)

            Sarsa.update_normalize_stats(self._norm_stats, sarsa)

            self._replay_buffer.push(sarsa)

            if env_dict["done"]:
                traj_rewards.append(self._agent.traj_reward)

        if not env_dict["done"]:
            traj_rewards.append(self._agent.traj_reward)

        return {"steps" : self.steps, "traj_rewards" : traj_rewards}

    def train_iteration(self, info_dict):
        """ Determines whether to update policy or target based on step count """
        step = info_dict["steps"]

        if step % self._improve_policy_modulo == 0:
            self._agent.improve_policy()

        # TODO these both might run...not sure if this is correct
        if step % self._improve_target_modulo == 0:
            for _ in range(self._iterations_of_improvement):
                batch = self._replay_buffer.random_sample(self._batch_size,
                                                          sample_less=self._train_sample_less)

                reduced_sarsa = Sarsa.reduce(batch)
                normalized_sarsa = Sarsa.normalize_element(self._norm_stats,
                                                           self.steps,
                                                           reduced_sarsa)

                self._agent.improve_target(normalized_sarsa)
