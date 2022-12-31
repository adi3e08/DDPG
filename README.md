# Deep Deterministic Policy Gradient (DDPG)
This repository contains a clean and minimal implementation of Deep Deterministic Policy Gradient (DDPG) algorithm in Pytorch.

DDPG is a model-free RL algorithm for continuous action spaces. It adopts an off-policy actor-critic approach and uses deterministic policies.

You can find more details about how DDPG works in my accompanying blog post [here](https://adi3e08.github.io/blog/ddpg/).

## Results
I trained DDPG on a few continuous control tasks from [Deepmind Control Suite](https://github.com/deepmind/dm_control/tree/master/dm_control/suite). Results are below.

* Cartpole Swingup - Swing up and balance an unactuated pole by applying forces to a cart at its base.
<p align="center">
<img src=".media/ddpg_cartpole_swingup.png" width="40%"/>
<img src=".media/ddpg_cartpole_swingup.gif" width="40%"/>
</p>

* Reacher Hard - Control a two-link robotic arm to reach a randomized target location.
<p align="center">
<img src=".media/ddpg_reacher_hard.png" width="40%"/>
<img src=".media/ddpg_reacher_hard.gif" width="40%"/>
</p>

## References
* Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015. [Link](https://arxiv.org/abs/1509.02971)
