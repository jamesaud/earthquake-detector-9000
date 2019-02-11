from unittest import TestCase
from utils import Evaluator
from tests.test_loaders import TestLoader
import utils
from loaders.base_loader import SpectrogramBaseDataset
import os
class TestEvaluator(TestCase):

    def setUp(self):
        self.info = {0: {'amount_correct': 400,
                         'amount_total': 600},

                     1: {'amount_correct': 100,
                         'amount_total': 1000}}

        self.evaluator = Evaluator(self.info)

    def test_inits_correctly(self):
        self.assertEqual(self.info, self.evaluator.class_info)

    def test_update_accuracy(self):
        self.evaluator.update_accuracy(0, 200, 800)
        self.assertEqual(self.evaluator.class_info[0].amount_correct, 200)
        self.assertEqual(self.evaluator.class_info[0].amount_total, 800)

    def test_percent_correct(self):
        class_info = self.evaluator.class_info[1]
        self.assertEqual(class_info.amount_correct / class_info.amount_total, .1)

    def test_total_percent_correct(self):
        self.assertEqual(self.evaluator.total_percent_correct(), 500/1600)


class TestUtils(TestLoader):
    IMG_PATH = 'tests/spectrograms'
    WIDTH = 316
    HEIGHT = 217

    def setUp(self):
       self.dataset = SpectrogramBaseDataset(img_path=self.IMG_PATH, divide_test=.2, transform=None)
       self.path = os.path.join(os.getcwd(), self.IMG_PATH) 

    
    def test_weighted_random_sampler(self):
        random_sampler = utils.make_weighted_sampler(self.dataset, 2, weigh_classes=[2, 1])
        random_sampler_2 = utils.make_weighted_sampler(self.dataset, 2, weigh_classes=[1, 1])

        self.assertEqual(len(self.dataset), random_sampler.num_samples)
        
        weights = [0, 0]
        for (images, label), weight in zip(self.dataset, random_sampler.weights):
            weights[label] += weight

        self.assertTrue(weights[0] / 2 == weights[1])

        weights2 = [0, 0]
        for (images, label), weight in zip(self.dataset, random_sampler_2.weights):
            weights2[label] += weight

        self.assertEqual(weights[1], weights2[1])
        self.assertEqual(weights[0] / 2, weights2[0])