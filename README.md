# MetaRL-PRL
A demo code to train a Meta-RL network in a probabilistic reversal learning task (dynamic foraging task) from Hattori et al., Cell, 2019. The code trains a recurrent neural network using the method proposed in [Wang et al., arXiv, 2016]. The network updates its synaptic weights using advantage actor critic (A2C) algorithm across training sessions, and the trained network uses its recurrent activity dynamics for their trial-by-trial RL behaviors. The code also plots performance improvement across training episodes.

## Dependencies
I tested the code in Python 3.7.9. Following libraries need to be installed to run the code:
- numpy (>=1.19.5)
- tensorflow (>=2.5.0)
- sklearn (>=0.23.2)
- matplotlib (>=3.3.2)
- seaborn (>=0.11.0)
