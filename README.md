# advantage
A TensorFlow-based Reinforcement Learning Framework. This framework allows for easy deployment of various RL algorithms, both discrete (i.e. Atari games) and continuous (i.e. Robotics) action-space models, with a little amount of coding. advantage is compatable with OpenAI Gym. Users can develop simulators using OpenAI Gym, and then simply using configuration files, train their models in the simulator. Trained models can then be easily deployed using TensorFlow protobufs. advantage's goal is to implement the common paradigms of Reinforcement Learning to take advantage of code reuse when implementing models; this allows for new models to be added to the framework with ease.

Currently supports:
 - Deep-Q Network

Planned additions:
  - Value-based:
    - C51, Implicit Quantile Agents
  - Policy-Gradients:
     - Evolved Policy Gradients
     - Proximal Policy Optimization
  - Multi-agent

### Training
``` python
import advantage as adv
agent = adv.make("{path_to}/samples/dqn.config")
agent.train()
````

### Inference
``` python
with agent.infer() as infer:
    env = infer.env
    for _ in infer.run_trajectory(run_through=False):
        env.render()
```

If there are any problems with the learning algorithm please open an issue
