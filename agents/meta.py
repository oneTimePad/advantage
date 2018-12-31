
""" This module consists of `meta` agents. These
are agents coordinate other agents. meta agents
can spawn other agents in a distributed manner or
utilize results from regular agents and aggregate them
in some manner (meta-RL)
"""

class MetaLearningAgent:
    """ `Meta`-agent that represents the overall
    coordinator of agents. This agent spawns
    other agents. All `Meta`-agents subclass
    this class. This is not to be confused with
    `Meta-Learning`
    """
    pass

class MetaObjectiveAgent:
    """ This `Meta`-agent represents actual
    RL `Meta-Learning`. This `Meta`-agent
    aggregates results from the agents it's spawns
    to achieve some overall objective that is not
    the current goal of the agents it spawns
    """
    pass

class EvolutionaryMetaAgent(MetaObjectiveAgent):
    """ A type of `MetaObjective` agent that
    utilizes evolutionary strategies
    """
    pass
