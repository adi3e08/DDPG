# Deep Deterministic Policy Gradient (DDPG)
This repository contains a clean and minimal implementation of Deep Deterministic Policy Gradient (DDPG) algorithm in Pytorch.

DDPG is a model-free RL algorithm for continuous action spaces. It adopts an off-policy actor-critic approach and uses deterministic policies.

You can find more details about how DDPG works in my accompanying blog post [here](https://adi3e08.github.io/blog/ddpg/).

## Results
I trained DDPG on a few continuous control tasks from [Deepmind Control Suite](https://github.com/deepmind/dm_control/tree/master/dm_control/suite). Results are below.

* Cartpole Swingup : Swing up and balance an unactuated pole by applying forces to a cart at its base.
<p align="center">
<img src="https://adi3e08.github.io/files/blog/ddpg/imgs/ddpg_cartpole_swingup.png" width="40%"/>
<img src="https://adi3e08.github.io/files/blog/ddpg/imgs/ddpg_cartpole_swingup.gif" width="31%"/>
</p>

* Reacher Hard : Control a two-link robotic arm to reach a random target location.
<p align="center">
<img src="https://adi3e08.github.io/files/blog/ddpg/imgs/ddpg_reacher_hard.png" width="40%"/>
<img src="https://adi3e08.github.io/files/blog/ddpg/imgs/ddpg_reacher_hard.gif" width="31%"/>
</p>

* Cheetah Run : Control a planar biped to run.
<p align="center">
<img src="https://adi3e08.github.io/files/blog/ddpg/imgs/ddpg_cheetah_run.png" width="40%"/>
<img src="https://adi3e08.github.io/files/blog/ddpg/imgs/ddpg_cheetah_run.gif" width="31%"/>
</p>

* Walker Run : Control a planar biped to run.
<p align="center">
<img src="https://adi3e08.github.io/files/blog/ddpg/imgs/ddpg_walker_run.png" width="40%"/>
<img src="https://adi3e08.github.io/files/blog/ddpg/imgs/ddpg_walker_run.gif" width="31%"/>
</p>

## Requirements
- Python
- Numpy
- Pytorch
- Tensorboard
- Matplotlib
- Deepmind Control Suite

## Usage
To train DDPG on Walker Run task, run,

    python ddpg.py --domain walker --task run --mode train --episodes 3000 --seed 0 

The data from this experiment will be stored in the folder "./log/walker_run/seed_0". This folder will contain two sub folders, (i) models : here model checkpoints will be stored and (ii) tensorboard : here tensorboard plots will be stored.

To evaluate DDPG on Walker Run task, run,

    python ddpg.py --domain walker --task run --mode eval --episodes 3 --seed 100 --checkpoint ./log/walker_run/seed_0/models/3000.ckpt --render

## References
* Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015. [Link](https://arxiv.org/abs/1509.02971)
