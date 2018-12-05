from pyksburden.ksburden import KSBurden
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO)
lg = logging.getLogger(__name__)


pars = argparse.ArgumentParser(description='KS-Burden Test')

pars.add_argument('--bfile', type=str, required=True,
                  help='Path to plink file stem')
pars.add_argument('--pheno', type=str, required=True,
                  help='Path to pheno file (no header)')
pars.add_argument('--variants', type=str, required=True,
                  help='Path to variant file (no header)')
pars.add_argument('--th', type=int, default=1,
                  help='Number of threads')
pars.add_argument('--tests', default='ks,burden,cmc', type=str,
                  help='comma separated list of tests to run')
pars.add_argument('--iter', default=1000, type=int,
                  help='max. number of iteration for MC')
pars.add_argument('--out', default='ksburden_results',
                  type=str, help='output path')

args = pars.parse_args()

if __name__ == '__main__':
    work_dir = os.getcwd()
    lg.info('Starting KS-Burden')
    lg.debug('Currently working dir: %s', work_dir)

    models = args.tests.split(',')
    lg.info('Running %s tests: %s', len(models), args.tests)

    gene_runner = KSBurden(args.bfile, args.pheno, args.variants,
                           models=tuple(models))

    lg.info('Starting')
    output = gene_runner.run_multithreaded(args.th, n_iter=args.iter)
    lg.info('Finished tests')
    output.to_csv(args.out+'.tsv', sep='\t')
