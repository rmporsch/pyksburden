import pandas as pd
import logging
import numpy as np
from pyksburden.genereader import GeneReader
from typing import Callable, Tuple
import multiprocessing.dummy as mp
from itertools import repeat

lg = logging.getLogger(__name__)


class KSBurden(GeneReader):

    def __init__(self, plink_path: str, pheno_path: str,
                 variant_path: str,
                 models: Tuple[str] = ('ks', 'cmc', 'burden')):
        GeneReader.__init__(self, plink_path, pheno_path, variant_path)
        self._models = {'ks': self._ks,
                        'burden': self._burden,
                        'cmc': self._cmc}
        self.models = models

    def _permutation(self, genotypes: np.array,
                     fun: Callable[[np.array, np.array], float],
                     n_iter: int) -> float:
        case_control_index = self.case_controls
        stat = fun(genotypes, case_control_index)

        null = np.zeros(n_iter)
        for i in range(n_iter):
            np.random.shuffle(case_control_index)
            null[i] = fun(genotypes, case_control_index)

        pval = (sum(null >= stat) + 1) / (n_iter + 1)
        return pval

    @staticmethod
    def _split(arr: np.array, cond: np.array):
        return [arr[cond], arr[~cond]]

    def _ks(self, x: np.array, cond: np.array) -> float:
        cases, controls = self._split(x, cond)
        sum_cases = np.sum(cases)
        sum_controls = np.sum(controls)
        c_cases = np.cumsum(np.sum(cases, axis=0)) / sum_cases
        c_control = np.cumsum(np.sum(controls, axis=0)) / sum_controls
        stat = np.max(np.abs(c_control - c_cases))
        return stat

    def _burden(self, x: np.array, cond: np.array) -> float:
        cases, controls = self._split(x, cond)
        sum_cases = np.sum(cases)
        sum_controls = np.sum(controls)
        return np.abs(sum_cases - sum_controls)

    def _cmc(self, x: np.array, cond: np.array) -> float:
        cases, controls = self._split(x, cond)
        sum_cases = np.sum(cases, axis=0)
        sum_controls = np.sum(controls, axis=0)
        cmc_cases = np.sum(sum_cases > 1)
        cmc_controls = np.sum(sum_controls > 1)
        return np.abs(cmc_cases - cmc_controls)

    def run_multithreaded(self, n_threads: int = 1,
                          genes=None, n_iter: int = 1000) -> pd.DataFrame:
        if genes is None:
            genes = self.genes
        else:
            assert len(genes) > 0
        n_genes = len(self.genes)
        lg.info('Testing %s genes', n_genes)

        gene_iterator = self.gene_iterator(genes)
        pool = mp.Pool(n_threads)
        results = pool.starmap(self._run_models,
                               zip(gene_iterator, repeat(n_iter)))
        pool.close()
        pool.join()
        output = pd.DataFrame({'Gene': genes})
        results = pd.DataFrame(results)
        col_names = list(self.models)
        col_names.append('num_var')
        results.columns = col_names
        output = pd.concat([output, results], axis=1)
        if ('ks' in self.models) & ('burden' in self.models):
            critical_tau = self._get_null((0.05 / n_genes) * 100)
            output['ksburden'] = self._ksburden(results.ks.values,
                                                results.burden.values,
                                                critical_tau)
        return output

    def _run_models(self, genotypes: np.array,
                    n_iter: int = 1000) -> np.array:

        results = np.zeros(len(self.models)+1)
        results[-1] = genotypes.shape[1]
        for h, gene_test in enumerate(self.models):
            results[h] = self._permutation(genotypes, self._models[gene_test],
                                           n_iter)
        return results



    def run_gene_test(self, models: Tuple[str] = ('ks', 'cmc', 'burden'),
                      genes=None, n_iter: int = 1000) -> pd.DataFrame:
        if genes is None:
            genes = self.genes
        else:
            assert len(genes) > 0
        n_genes = len(self.genes)
        lg.info('Testing %s genes', n_genes)

        gene_iterator = self.gene_iterator(genes)
        results = np.zeros((len(genes), len(models)))
        num_var = np.zeros(len(genes))
        for i, g in enumerate(gene_iterator):
            genotypes, gene_name = g
            num_var[i] = int(genotypes.shape[1])
            lg.debug('Processing %s', gene_name)
            for h, gene_test in enumerate(models):
                results[i, h] = self._permutation(genotypes,
                                                  self._models[gene_test],
                                                  n_iter)
        output = pd.DataFrame({'Gene': genes,
                               'num_var': num_var})
        results = pd.DataFrame(results)
        results.columns = models
        output = pd.concat([output, results], axis=1)
        if ('ks' in models) & ('burden' in models):
            critical_tau = self._get_null((0.05 / n_genes) * 100)
            output['ksburden'] = self._ksburden(results.ks.values,
                                                results.burden.values,
                                                critical_tau)
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
