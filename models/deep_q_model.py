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
                 delay_improvement,
                 train_target_modulo,
                 train_iterations,
                 replay_buffer,
                 sarsa_attrs_to_normalize,
                 batch_size,
                 sample_less):

        self._improve_policy_modulo = improve_policy_modulo

        self._replay_buffer = replay_buffer

        self._train_target_modulo = train_target_modulo

        self._train_iterations = train_iterations

        self._batch_size = batch_size

        self._sarsa_attrs_to_normalize = sarsa_attrs_to_normalize

        self._sample_less = sample_less

        self._delay_improvement = delay_improvement

        self._num_target_train_steps = 0

        if not isinstance(agent, DeepQAgent):
            raise ValueError("Agent must be of type DeepQAgent but is %s" % type(agent))

        super().__init__(graph,
                         environment,
                         model_scope,
                         agent)
    @property
    def replay_buffer(self):
        """ Allow read access to replay_buffer """
        return self._replay_buffer

    def set_up_train(self):

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

        for env_dict in self._agent.act_for_trajs(self._train_target_modulo, training=True):

            sarsa = Sarsa.make_element_from_env(env_dict)

            self._replay_buffer.push(sarsa)

        return {}

    def improve_iteration(self, info_dict):
        """ Determines whether to update policy or target based on step count """

        if self._agent.num_traj > self._delay_improvement:
            if self._agent.num_traj % self._train_target_modulo == 0:

                for _ in range(self._train_iterations):

                    batch = self._replay_buffer.random_sample_and_pop(self._batch_size,
                                                                      sample_less=self._sample_less)

                    normalized_sarsa = Sarsa.stack(batch, self._sarsa_attrs_to_normalize)
                    self._agent.improve_target(normalized_sarsa)
                    self._num_target_train_steps += 1

            if self._num_target_train_steps % self._improve_policy_modulo == 0:
                self._agent.improve_policy()
                return True
        return False
