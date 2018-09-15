# Advantage
A TensorFlow-based Reinforcement Learning Framework. This framework allows for easy deployment of various RL algorithms, both discrete (i.e. Atari games) and continuous (i.e. Robotics) models, with a little amount of coding. Advantage is compatable with OpenAI Gym. Users can develop simulators using OpenAI Gym, and then simply using configuration files, train their models in the simulator. Trained models can then be easily deployed using TensorFlow protobufs. Advantage's goal is to implement the common paradigms of Reinforcement Learning to take advantage of code reuse when implementing models; this allows for new models to be added to the framework with ease.

First version is still in development and will support the simple DeepQNetwork. To get involved check out the dev branch, Projects board and Issues list!


Planned additions:
  - Discrete:
    - C51, Implicit Quantile Agents
  - Continuous:
     - Evolved Policy Gradients
     - Proximal Policy Optimization
  - Multi-agent
