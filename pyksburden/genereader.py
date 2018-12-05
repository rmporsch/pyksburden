import numpy as np
import os
import pandas as pd
import logging
from typing import Tuple
from pyplink import PyPlink


lg = logging.getLogger(__name__)


class GeneReader(object):

    def __init__(self, plink_path: str, pheno_path: str, variant_path: str):
        lg.debug("""
        Loading the following files:
        Plink: %s Pheno %s Variants: %s
         """, plink_path, pheno_path, variant_path)
        assert os.path.isfile(plink_path+'.bed')
        assert os.path.isfile(variant_path)
        self.plink_path = plink_path
        self.variant_path = variant_path
        self.bfile = PyPlink(self.plink_path)
        self.bim = self.bfile.get_bim()
        self.bim['rsid'] = self.bim.index.values
        self.fam = self.bfile.get_fam()
        self.n_chrom = self.bim.chrom.nunique()
        self.variants = self._get_var(self.variant_path)
        self.genes = self.variants.gene.unique()
        self.pheno = self._get_pheno(pheno_path)
        self.bfile.close()

    def _get_var(self, variant_path: str) -> pd.DataFrame:
        dat = pd.read_table(variant_path, header=None)
        lg.debug(dat.head())
        nrow, ncol = dat.shape
        assert ncol == 4
        assert nrow > 3
        dat.columns = ['chrom', 'pos', 'rsid', 'gene']
        n_chrom = dat.chrom.nunique()
        chromosomes = dat.chrom.unique()
        n_genes = dat.gene.nunique()
        lg.info('Got %s genes in variant file', n_genes)
        lg.info('Got %s variants in variant file', nrow)
        lg.debug('Chromosomes: %s', n_chrom)
        chrom_check = [k for k in chromosomes if k in self.bim.chrom.unique()]
        lg.info('Found %s out of %s chromosomes in bim file',
                len(chrom_check), self.n_chrom)
        lg.debug(self.bim.head())
        dat = pd.merge(dat, self.bim, on=['pos', 'chrom', 'rsid'],
                       how='inner')
        n_var = dat.shape[0]
        lg.info('After merging with the bim file there are %s variants left',
                n_var)
        if n_var < nrow:
            lg.warning('After merging I lost %s variants',
                       nrow - n_var)
        return dat

    def _get_pheno(self, pheno_file: str) -> pd.DataFrame:
        dat = pd.read_table(pheno_file, header=None)
        nrow, ncol = dat.shape
        assert ncol >= 3
        assert nrow > 1
        lg.debug(dat.head())
        if ncol == 3:
            dat.columns = ['fid', 'iid', 'Pheno']
            dat['fid'] = dat['fid'].astype(str)
            dat['iid'] = dat['fid'].astype(str)
        elif ncol == 6:
            dat.columns = ['fid', 'iid', 'father', 'mother', 'gender', 'Pheno']
            dat['fid'] = dat['fid'].astype(str)
            dat['iid'] = dat['fid'].astype(str)
        else:
            raise ValueError('Need at either a 3 or 6 column file')
        lg.debug(self.fam.head())
        dat = pd.merge(self.fam, dat, on=['fid', 'iid'])
        self.n = dat.shape[0]
        lg.info('Using %s out of %s samples', self.n, nrow)
        if self.n < nrow:
            lg.warning('%s samples not in fam file', (nrow - self.n))
            if self.n < 2:
                raise AssertionError('Sample size is smaller than 2.')
        self.case_controls = (dat.Pheno > 0).values
        lg.info('Found %s cases and %s controls',
                np.sum(self.case_controls), np.sum(~self.case_controls))
        return dat

    def _read_gene(self, gene: str) -> np.array:
        temp = self.variants[self.variants.gene == gene]
        chrom = temp.chrom.unique()
        assert len(chrom) == 1
        lg.debug(temp.head())
        marker = temp.rsid.values
        lg.debug(marker)
        p = len(marker)
        assert p > 3
        genotype_matrix = np.zeros((self.n, p))
        reader = PyPlink(self.plink_path)
        u = 0
        lg.info('Reading %s', gene)
        for i, g in reader.iter_geno_marker(marker):
            genotype_matrix[:, u] = g
            u += 1
            lg.debug('Processed variant %s', i)
        genotype_matrix[genotype_matrix == -1] = 0
        reader.close()
        return genotype_matrix

    def gene_iterator(self, genes=None) -> np.array:
        if genes is None:
            genes = self.genes
        for gene_name in genes:
            lg.debug('Getting gene %s', gene_name)
            yield self._read_gene(gene_name)
