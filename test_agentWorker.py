from unittest import TestCase

import ray

from ray_workers import AgentWorker
from test_experiment import TestExperiment


class TestAgentWorker(TestCase):

    def setUp(self):
        ray.init()

    def test_setup(self):
        agent = AgentWorker._modified_class(0, TestExperiment.SIMPLE_CONFIG, '/tmp/tf-logdir')
        agent.setup()
