from .base_models import LearningModel
from elements.sarsa import Sarsa

class DeepQModel(LearningModel):
    """ Model for Deep-Q Networks """

    def __init__(self, graph,
                    environment,
                    agent,
                    steps_to_run_for,
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

        super().__init__(graph, environment,
                            agent,
                            steps_to_run_for,
                            improve_policy_modulo,
                            steps_for_act_iter)

    def set_up(self):
        self._agent.set_up()

        dims = self._environment.dims

        self._norm_stats = Sarsa.make_stats(normalize_attrs=self._sarsa_attrs_to_normalize,
                                            **dims)

        self._sarsa_count = 0



    def act_iteration(self):
        """ Runs the agent and collects Sarsas to put in the replay buffer """
        for step, env_dict in self._agent.act_for_steps(self._steps_for_act_iter, training=True):
            sarsa = Sarsa.make_element_from_env(env_dict)

            self._sarsa_count += 1

            Sarsa.update_normalize_stats(self._norm_stats, sarsa)

            self._replay_buffer.push(sarsa)

        return {"steps" : self._sarsa_count}



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
                                                    self._sarsa_count,
                                                    reduced_sarsa)

                self._agent.improve_target(normalized_sarsa)

    @property
    def replay_buffer(self):
        return self._replay_buffer
