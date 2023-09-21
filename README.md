# MetaRL-PRL
A demo code to train a Meta-RL network in a probabilistic reversal learning task (dynamic foraging task). The code trains a recurrent neural network using the method proposed in [Wang et al., arXiv, 2016]. The network updates its synaptic weights using advantage actor critic (A2C) algorithm across training sessions, and the trained network uses its recurrent activity dynamics for their trial-by-trial RL behaviors. The code also plots performance improvement across training episodes.

## Dependencies
I tested the code in Python 3.7.9. Following libraries need to be installed to run the code:
- numpy (>=1.19.5)
- tensorflow (>=2.5.0)
- sklearn (>=0.23.2)
- matplotlib (>=3.3.2)
- seaborn (>=0.11.0)


## References
#### Behaviors of the Meta-RL network were characterized in the following publication:
  - Hattori, R., Hedrick, N.G., Jain, A., Chen, S., You, H., Hattori, M., Choi, J.H., Lim, B.K., Yasuda, R. and Komiyama, T. Meta-reinforcement learning  via orbitofrontal cortex. _Nature Neuroscience, in press_.
#### The same behavior task was use for training mice in the following publications:
  - Hattori, R., Hedrick, N.G., Jain, A., Chen, S., You, H., Hattori, M., Choi, J.H., Lim, B.K., Yasuda, R. and Komiyama, T. Meta-reinforcement learning  via orbitofrontal cortex. _Nature Neuroscience, in press_.
  - Hattori, R. and Komiyama, T. (2022). Context-dependent persistency as a coding mechanism for robust and widely distributed value coding. _Neuron_, doi: https://doi.org/10.1016/j.neuron.2021.11.001.
  - Hattori, R., Danskin, B., Babic, Z., Mlynaryk, N. and Komiyama, T. (2019). Area-Specificity and Plasticity of History-Dependent Value Coding During Learning. _Cell_, doi: https://doi.org/10.1016/j.cell.2019.04.027.
