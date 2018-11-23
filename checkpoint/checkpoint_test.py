import unittest
import os
import tempfile
import tensorflow as tf
import time
from advantage.checkpoint import checkpointable

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

@checkpointable
class TestAgent:
    def __init__(self, graph):
        self._model = None
        self._graph = graph
        self._session = tf.Session(graph=self._graph)

    @property
    def session(self):
        return self._session

    @property
    def model(self):
        return self._model

    def set_up(self):
        with self._graph.as_default():
            self._model = tf.Variable(1, name="test")

@checkpointable(chain_head=True)
class TestModel:
    def __init__(self, graph, value):
        self._graph = graph
        self._agent = TestAgent(graph)
        with self._graph.as_default():
            self.net = tf.Variable(value, name="test2")
        self.restore_session = self._agent.session

    @property
    def graph(self):
        return self._graph

    @property
    def agent(self):
        return self._agent

    def set_up(self):
        self._agent.set_up()


class TestCkptSystem(unittest.TestCase):
    """ Tests for the Checkpoint System """

    def setUp(self):
        graph = tf.Graph()
        self.model = TestModel(graph, 1)

        self.tmp_dir = tempfile.mkdtemp()

        self.model.checkpoint_dir_path = self.tmp_dir
        self.model.checkpoint_file_prefix = "model_ckpt"
        self.model.checkpoint_freq_sec = 10

        self.model.set_up()

    def test_ckpt_construction(self):
        self.assertEqual(True, self.model.agent.model.name in self.model.tf_var_set_scopes)
        self.assertEqual(True, self.model.net.name in self.model.tf_var_set_scopes)
        self.assertEqual(True, hasattr(self.model, "_tf_saver"))
        self.assertEqual(True, not self.model._tf_saver is None)

    def test_save(self):
        with self.model.graph.as_default():
            inc = tf.assign_add(self.model.tf_global_step, 1, name="increment")
            self.model.agent.session.run(tf.global_variables_initializer())
        self.model.start_checkpoint_system()
        past = time.time()
        while time.time() - past < 15:
            self.model.agent.session.run(inc)
            time.sleep(0.2)
        self.model.stop_checkpoint_system()
        match = list(filter(lambda x: x.endswith(".index"), os.listdir(self.tmp_dir)))

        # should be two checkpoints in 15seconds with 10 second frequency
        # one as time 10 seconds and another from stop_checkpoint_system
        self.assertEqual(2, len(match))

    def test_restore(self):
        with self.model.graph.as_default():
            inc = tf.assign_add(self.model.tf_global_step, 1, name="increment")
            self.model.agent.session.run(tf.global_variables_initializer())
        self.model.start_checkpoint_system()
        past = time.time()
        while time.time() - past < 15:
            self.model.agent.session.run(inc)
            time.sleep(0.2)
        self.model.stop_checkpoint_system()

        graph = tf.Graph()
        model = TestModel(graph, 2)

        model.checkpoint_dir_path = self.tmp_dir
        model.checkpoint_file_prefix = "model_ckpt"
        model.checkpoint_freq_sec = 10

        model.set_up()

        global_step = model.tf_global_step
        session = model.agent.session

        self.assertEqual(session.run(global_step), 75)
        self.assertEqual(session.run(model.agent.model), 1)




unittest.main()
