# Q-Learning Algorithm

The fundamental idea of q-learning is to **control**. We have `agent`, `action`, `state`, `reward`.

## Basic Q-Learning Algorithm, with Tabular Representation

- Tutorial: [A Painless Q-learning Tutorial](http://mnemstudio.org/path-finding-q-learning-tutorial.htm)
- Implementation: [q-tabular-rep.ipynb](q-tabular-rep.ipynb)
- Weakness:
    1. cannot extend to high dimension naturally. For example, for [Othello](https://en.wikipedia.org/wiki/Reversi) game, the dimension of state space and action space will be `3^64`, which is nearly impossible to train.
    2. the matrix does not have mathematical meaning, i.e. the matrix cannot be understood as some ideas in algebra, such as linear transformation. Then the optimization is slow and we find the implementation with `tensorflow` is even slower than raw `numpy` one.
    3. optimization may stick in local minimum and neglect global minimum.
- Solution:
    - To solve 1 & 2, we can use deep q-learning (DQL).
    - To solve 3, we use epsilon-greedy method.

## Deep Q-Learning Algorithm

- Paper: [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- Implementation: None
- Weakness:
    1. the action space must be discrete since we need evaluate value function for actions one by one. It is unnecessary that the state space is discrete.
- Solution:
    - one idea is just using policy gradient. But this is an almost different model.
    - Another idea is to replace epsilon-greedy with [VIME: Variational Information Maximizing
Exploration](https://arxiv.org/pdf/1605.09674.pdf). Then the model can be extended to continuous action space naturally.
