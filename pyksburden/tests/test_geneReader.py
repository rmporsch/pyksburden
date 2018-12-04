from unittest import TestCase
from pyksburden.genereader import GeneReader
import os
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)


class TestGeneReader(TestCase):

    def setUp(self):
        assert os.path.isdir('data')
        self.plink_file = 'data/chr22_rare_test_data'
        self.variant_file = 'data/chr22_rare_test_data.list'
        self.pheno_file = 'data/heartattack_male_60_simple.csv'

    def test_gene_iterator(self):
        rr = GeneReader(self.plink_file, self.pheno_file, self.variant_file)
        gene_reader = rr.gene_iterator()
        genotypes, gene = next(gene_reader)
        self.assertTrue(np.sum(genotypes) > 0)
        self.assertTrue(genotypes.shape[0] > 10)
        self.assertTrue(genotypes.shape[1] > 10)
