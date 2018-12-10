from unittest import TestCase
import os
from pyksburden.ksburden import KSBurden
import logging

logging.basicConfig(level=logging.INFO)


class TestKSBurden(TestCase):

    def setUp(self):
        assert os.path.isdir('data')
        self.plink_file = 'data/chr22_rare_test_data'
        self.variant_file = 'data/chr22_rare_test_data.list'
        self.pheno_file = 'data/heartattack_male_60_simple.csv'

    def test_muiltithread(self):
        rr = KSBurden(self.plink_file, self.pheno_file, self.variant_file)
        pval = rr.run_models(n_iter=100)
        print(pval)
        pval.to_csv('test_results.csv')
        self.fail()

