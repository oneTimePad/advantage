from advantage.approximators.base.base_approximators import _deep_approximator
from advantage.approximators.deep_convolutional import DeepConvolutional
from advantage.approximators.deep_dense import DeepDense

DeepConvolutional = _deep_approximator(DeepConvolutional)
DeepDense = _deep_approximator(DeepDense)
