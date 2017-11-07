python Viterbi.py POS.train POS.test 2>&1 | tee LOG.main
python Viterbi.py POS.train.large POS.test 2>&1 | tee LOG.large
python Viterbi.py POS.train.large POS.test 2>&1 | tee LOG.freq