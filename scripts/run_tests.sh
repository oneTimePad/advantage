#!/bin/bash

#agents
python3 tests/agents/test_deep_q_agent.py

#approximators
python3 tests/approximators/test_deep_convolutional.py
python3 tests/approximators/test_deep_dense.py

#builders
python3 tests/builders/test_agents_builder.py
python3 tests/builders/test_approximators_builder.py
