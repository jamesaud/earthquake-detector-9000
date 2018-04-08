from unittest import TestCase
from utils import Evaluator

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