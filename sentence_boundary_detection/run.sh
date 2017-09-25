# module load python/2.7
python SBD.py SBD.train SBD.test 2>&1 | tee LOG.main
# python -W ignore SBD.py small.txt SBD.test 2>&1 | tee LOG.small