import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from collections import deque
import random
from copy import deepcopy

class OUNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OUNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class GaussianNoise:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(mu, sigma)

    def __repr__(self):
        return 'GaussianNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, o, a, r, o_1, d):            
        self.buffer.append((o, a, r, o_1, d))
    
    def sample(self, batch_size):
        O, A, R, O_1, D = zip(*random.sample(self.buffer, batch_size))
        return torch.tensor(np.array(O), dtype=torch.float, device=self.device),\
               torch.tensor(np.array(A), dtype=torch.float, device=self.device),\
               torch.tensor(np.array(R), dtype=torch.float, device=self.device),\
               torch.tensor(np.array(O_1), dtype=torch.float, device=self.device),\
               torch.tensor(np.array(D), dtype=torch.float, device=self.device)

    def __len__(self):
        return len(self.buffer)

# Fully Connected Q network
class Q_FC(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Q_FC, self).__init__()
        self.fc1 = torch.nn.Linear(state_size+action_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 1)

    def forward(self, x, a):
        y1 = F.relu(self.fc1(torch.cat((x,a),1)))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2).view(-1)        
        return y

# Fully Connected Policy network
class mu_FC(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(mu_FC, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, action_size)

    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        y = torch.tanh(self.fc3(y2))        
        return y

def process_dmc_observation(time_step):
    """
    Function to parse observation dictionary returned by Deepmind Control Suite.
    """
    o_1 = np.array([])
    for k in time_step.observation:
        if time_step.observation[k].shape:
            o_1 = np.concatenate((o_1, time_step.observation[k].flatten()))
        else :
            o_1 = np.concatenate((o_1, np.array([time_step.observation[k]])))
    r = time_step.reward
    done = time_step.last()
    return o_1, r, done

def process_observation(x, simulator):
    if simulator == "dm_control":
        o_1, r, done = process_dmc_observation(x)
        if r is None:
            return o_1
        else:
            return o_1, r, done
    elif simulator == "gym":
        if type(x) is np.ndarray:
            return x
        elif type(x) is tuple:
            o_1, r, done, info = x
            return o_1, r, done

# DDPG
class DDPG:
    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.arglist = parse_args()
        self.env = make_env(seed)
        if self.arglist.use_gpu:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.actor = mu_FC(self.env.state_size,self.env.action_size).to(self.device)
        self.actor_target = deepcopy(self.actor)       
        for param in self.actor_target.parameters():
            param.requires_grad = False
        self.critic = Q_FC(self.env.state_size,self.env.action_size).to(self.device)
        self.critic_target = deepcopy(self.critic)       
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)
        if self.arglist.noise_type == "gaussian":
            self.noise = GaussianNoise(mu=np.zeros(self.env.action_size), sigma=0.2)
        elif self.arglist.noise_type == "OU":
            self.noise = OUNoise(mu=np.zeros(self.env.action_size), sigma=0.2)
        
        self.actor_loss_fn =  torch.nn.MSELoss()
        self.critic_loss_fn =  torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.arglist.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.arglist.lr)
        
        self.exp_dir = os.path.join("./log", self.arglist.exp_name)
        self.model_dir = os.path.join(self.exp_dir, "models")
        self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")
        if os.path.exists("./log"):
            pass            
        else:
            os.mkdir("./log")
        os.mkdir(self.exp_dir)
        os.mkdir(os.path.join(self.tensorboard_dir))
        os.mkdir(self.model_dir)

    def save_checkpoint(self, name):
        checkpoint = {'actor' : self.actor.state_dict()}
        torch.save(checkpoint, os.path.join(self.model_dir, name))

    def soft_update(self, target, source, tau):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)

    def train(self):
        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        for episode in range(self.arglist.episodes):
            o = process_observation(self.env.reset(), self.env.simulator)
            ep_r = 0
            while True:
                with torch.no_grad():
                    a = self.actor(torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0))
                a = a.cpu().numpy()[0] 
                a += self.noise()
                a = np.clip(a,-1.0,1.0)
                o_1, r, done = process_observation(self.env.step(a), self.env.simulator)

                if self.env.simulator == "dm_control":
                    terminal = 0 # deep mind control suite tasks are infinite horizon i.e don't have a terminal state
                elif self.env.simulator == "gym":
                    terminal = int(done) # open ai gym tasks have a terminal state 
                self.replay_buffer.push(o, a, r, o_1, terminal)

                ep_r += r
                o = o_1
                if self.replay_buffer.__len__() < self.arglist.replay_fill:
                    pass
                else :
                    O, A, R, O_1, D = self.replay_buffer.sample(self.arglist.batch_size)

                    q_value = self.critic(O, A)

                    with torch.no_grad():
                        next_q_value = self.critic_target(O_1,self.actor_target(O_1))                                    
                    expected_q_value = R + self.arglist.gamma * next_q_value * (1 - D)

                    critic_loss = self.critic_loss_fn(q_value, expected_q_value)
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    actor_loss = - torch.mean(self.critic(O,self.actor(O)))
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    self.soft_update(self.actor_target, self.actor, self.arglist.tau)
                    self.soft_update(self.critic_target, self.critic, self.arglist.tau)

                if done:
                    writer.add_scalar('ep_r', ep_r, episode)
                    if episode % self.arglist.eval_every == 0 or episode == self.arglist.episodes-1:
                        eval_ep_r_list = self.eval(self.arglist.eval_over)
                        writer.add_scalar('eval_ep_r', np.mean(eval_ep_r_list), episode)
                        self.save_checkpoint(str(episode)+".ckpt")
                    break   

    def eval(self, episodes):
        ep_r_list = []
        for episode in range(episodes):
            o = process_observation(self.env.reset(), self.env.simulator)
            ep_r = 0
            while True:
                with torch.no_grad():
                    a = self.actor(torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0))
                a = a.cpu().numpy()[0] 
                o_1, r, done = process_observation(self.env.step(a), self.env.simulator)
                ep_r += r
                o = o_1
                if done:
                    ep_r_list.append(ep_r)
                    break
        return ep_r_list    

def parse_args():
    parser = argparse.ArgumentParser("DDPG")
    parser.add_argument("--exp-name", type=str, default="expt_ddpg_reacher_hard", help="name of experiment")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="use gpu")
    parser.add_argument("--episodes", type=int, default=10000, help="number of episodes")
    # Core training parameters
    parser.add_argument("--noise-type", type=str, default="gaussian", help="gaussian / OU")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="actor learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--tau", type=float, default=0.005, help="soft target update parameter")
    parser.add_argument("--replay-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--replay-fill", type=int, default=int(1e5), help="elements in replay buffer before training starts")
    parser.add_argument("--eval-every", type=int, default=50, help="eval every _ episodes")
    parser.add_argument("--eval-over", type=int, default=50, help="eval over _ episodes")
    return parser.parse_args()

def make_env(env_seed):
    
    # import gym
    # env = gym.make('BipedalWalker-v3')
    # env.seed(env_seed)
    # env.state_size = 24
    # env.action_size = 4
    # env.simulator = "gym"

    from dm_control import suite
    env = suite.load(domain_name="reacher", task_name="hard", task_kwargs={'random': env_seed})
    env.state_size = 6
    env.action_size = 2
    env.simulator = "dm_control"

    # from dm_control import suite
    # env = suite.load(domain_name="cartpole", task_name="swingup", task_kwargs={'random': env_seed})
    # env.state_size = 5
    # env.action_size = 1
    # env.simulator = "dm_control"
    
    return env

if __name__ == '__main__':
    
    ddpg = DDPG()
    ddpg.train()

