import pandas as pd
import logging
import numpy as np
from pyksburden.genereader import GeneReader
from typing import Callable, Tuple
import multiprocessing as mp
from itertools import repeat
import copy
from scipy.stats import chi2

lg = logging.getLogger(__name__)


class Models(GeneReader):

    def __init__(self, plink_path: str, pheno_path: str, variant_path: str,
                 models: Tuple[str] = ('ks', 'cmc', 'burden'),
                 min_iter: int = 100):
        GeneReader.__init__(self, plink_path, pheno_path, variant_path)
        self._models = {'ks': self._ks,
                        'burden': self._burden,
                        'cmc': self._cmc}
        self._min_iter = min_iter
        for m in models:
            if m not in self._models.keys():
                raise ValueError("%s is not a valid model name" % m)
        self.models = models

    def _permutation(self, genotypes: np.array,
                     fun: Callable[[np.array, np.array], float],
                     n_iter: int) -> float:
        """
        Permutation for a given test.

        :param genotypes: numpy array with the genotypes
        :param fun: function with two arguments, case and control arrays
        :param n_iter: number of maximal iterations
        :return: p-value
        """
        case_control_index = copy.copy(self.case_controls)
        stat = fun(genotypes, case_control_index)

        null = 0
        more_threshold = True
        for i in range(n_iter):
            np.random.shuffle(case_control_index)
            null_stat = fun(genotypes, case_control_index)
            if null_stat >= stat:
                null += 1
            if (i >= self._min_iter) & more_threshold:
                pval = (null + 1) / (n_iter + 1)
                more_threshold = False
                if pval >= 0.05:
                    break

        pval = (null + 1) / (n_iter + 1)
        return pval

    @staticmethod
    def _split(arr: np.array, cond: np.array) -> Tuple:
        """
        Split numpy array into cases and control

        :param arr: numpy array with genotypes
        :param cond: numpy array
        :return: Tuple of cases and control arrays
        """
        return arr[cond], arr[~cond]

    def _ks(self, x: np.array, cond: np.array) -> float:
        """
        KS Test
        :param x: numpy array of the genoytpes
        :param cond: case control status
        :return: KS Statistics
        """
        cases, controls = self._split(x, cond)
        sum_cases = np.sum(cases)
        sum_controls = np.sum(controls)
        c_cases = np.cumsum(np.sum(cases, axis=0)) / sum_cases
        c_control = np.cumsum(np.sum(controls, axis=0)) / sum_controls
        stat = np.max(np.abs(c_control - c_cases))
        return stat

    def _burden(self, x: np.array, cond: np.array) -> float:
        """
        Simple Burden test.
        :param x: numpy array of the genoytpes
        :param cond: case control status
        :return: Burden test statistic
        """
        cases, controls = self._split(x, cond)
        sum_cases = np.sum(cases)
        sum_controls = np.sum(controls)
        return np.abs(sum_cases - sum_controls)

    def _cmc(self, x: np.array, cond: np.array) -> float:
        """
        Simple Burden test.
        :param x: numpy array of the genoytpes
        :param cond: case control status
        :return: CMC test statistic
        """
        cases, controls = self._split(x, cond)
        sum_cases = np.sum(cases, axis=0)
        sum_controls = np.sum(controls, axis=0)
        cmc_cases = np.sum(sum_cases > 1)
        cmc_controls = np.sum(sum_controls > 1)
        return np.abs(cmc_cases - cmc_controls)

    def _run_models(self, genotypes: np.array,
                    n_iter: int = 1000) -> np.array:
        """
        Run all models for a given genotype matrix
        :param genotypes: numpy array of genotypes
        :param n_iter: number of iterations
        :return: array with the pvalues of the tests and number of variants
        """
        results = np.zeros(len(self.models)+1)
        results[-1] = genotypes.shape[1]
        for h, gene_test in enumerate(self.models):
            results[h] = self._permutation(genotypes, self._models[gene_test],
                                           n_iter)
        return results

    def run_gene(self, gene_name: str, n_iter: int = 1000) -> np.array:
        """
        Run a single gene with pre-specified models
        :param gene_name: string name of the gene
        :param n_iter:  number of iteration for the permuation
        :return: array with the results
        """
        gene_iterator = self.gene_iterator([gene_name])
        genotypes = next(gene_iterator)
        output = self._run_models(genotypes, n_iter)
        return output


def multi_gene(plink_path: str, pheno_path: str, variant_path,
               gene: str, models: Tuple[str] = ('ks', 'burden', 'cmc'),
               n_iter: int = 1000):
    """
    Run multiple genes.

    :param plink_path: path to plink file
    :param pheno_path:  path to pheno file
    :param variant_path: path to variant file
    :param gene: name of gene
    :param models: Tuple with model names
    :param n_iter: max. number of iterations
    :return: array with the results
    """
    mm = Models(plink_path, pheno_path, variant_path, models)
    return mm.run_gene(gene, n_iter)


class KSBurden(object):

    def __init__(self, plink_path: str, pheno_path: str,
                 variant_path: str):
        self.plink_path = plink_path
        self.pheno_path = pheno_path
        self.variant_path = variant_path
        self._reader = GeneReader(plink_path, pheno_path, variant_path)
        self.genes = self._reader.genes

    def run_models(self, n_threads: int = 1,
                   models: Tuple[str] = ('ks', 'burden', 'cmc'),
                   genes=None, n_iter: int = 1000) -> pd.DataFrame:
        """
        Run Models.

        :param n_threads: number of threads
        :param models: tumple of models
        :param genes: list of genes, if none all genes are used
        :param n_iter: number of maximal iteration for MC
        :return: DataFrame with the results
        """
        if genes is None:
            genes = self.genes
        else:
            assert len(genes) > 0
        n_genes = len(self.genes)
        lg.info('Testing %s genes', n_genes)
        iterator = zip(repeat(self.plink_path), repeat(self.pheno_path),
                       repeat(self.variant_path), genes,
                       repeat(models), repeat(n_iter))

        pool = mp.Pool(n_threads)
        results = pool.starmap(multi_gene, iterator)
        pool.close()
        pool.join()
        output = pd.DataFrame({'Gene': genes})
        results = pd.DataFrame(results)
        col_names = list(models)
        col_names.append('num_var')
        results.columns = col_names
        output = pd.concat([output, results], axis=1)
        if ('ks' in models) & ('burden' in models):
            critical_tau = self._get_null((0.05 / n_genes) * 100)
            output['ksburden'] = self._ksburden(results.ks.values,
                                                results.burden.values,
                                                critical_tau)
            output['ksburden_fisher'] = self._fisher_combination(
                results.ks.values,
                results.burden.values)
        return output

    @staticmethod
    def _ksburden(ks: np.array, burden: np.array, tau) -> np.array:
        alt = np.array([ks, burden])
        alt = np.sort(alt, axis=0)

        def compute_w(x):
            return x[0], np.prod(x)

        w = np.apply_along_axis(compute_w, 0, alt)
        g = [w[0] <= tau[0],
             w[1] <= tau[1]]
        final = g[0] * g[1]
        return final

    @staticmethod
    def _get_null(percentile: float) -> Tuple[float, float]:
        """
        Compute the null distribution for two p-values.

        :param percentile: critical p-value threshold
        :return: critical values for first and second test.
        """
        lg.debug('Percentile for KS-Burden: %s', percentile)
        n_tests = 2
        large_l = 1000
        null = np.array([np.random.uniform(0, 1, size=large_l),
                         np.random.uniform(0, 1, size=large_l)])
        null = null.reshape(large_l, n_tests)
        null = np.sort(null, axis=0)

        def compute_w(x):
            return x[0], np.prod(x)

        w = np.apply_along_axis(compute_w, 0, null)
        c1 = np.percentile(w[:, 0], percentile)
        c2 = np.percentile(w[:, 1], percentile)
        return c1, c2

    @staticmethod
    def _fisher_combination(p1: float, p2: float):
        """
        Fisher's Method to combine p-values.

        :param p1: value of first p-value
        :param p2: value of second p-value
        :return: float object of the combined p-value
        """
        chi = -2*(np.log(p1) + np.log(p2))
        return 1-chi2.cdf(chi, df=4)

