import os
import argparse
from copy import deepcopy
from collections import deque
import math
import random
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dm_control import suite
import glob
import subprocess
import matplotlib.pyplot as plt

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, o, a, r, o_1):            
        self.buffer.append((o, a, r, o_1))
    
    def sample(self, batch_size):
        O, A, R, O_1 = zip(*random.sample(self.buffer, batch_size))
        return torch.tensor(np.array(O), dtype=torch.float64, device=self.device),\
               torch.tensor(np.array(A), dtype=torch.float64, device=self.device),\
               torch.tensor(np.array(R), dtype=torch.float64, device=self.device),\
               torch.tensor(np.array(O_1), dtype=torch.float64, device=self.device)

    def __len__(self):
        return len(self.buffer)

# Fully Connected Q network
class Q_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Q_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size+action_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 1)

    def forward(self, x, a):
        y1 = F.relu(self.fc1(torch.cat((x,a),1)))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2).view(-1)        
        return y

# Fully Connected Policy network
class mu_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(mu_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, action_size)

    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        y = torch.tanh(self.fc3(y2))        
        return y

# Parse observation dictionary returned by Deepmind Control Suite
def process_observation(time_step):
    o_1 = np.array([])
    for k in time_step.observation:
        if time_step.observation[k].shape:
            o_1 = np.concatenate((o_1, time_step.observation[k].flatten()))
        else :
            o_1 = np.concatenate((o_1, np.array([time_step.observation[k]])))
    r = time_step.reward
    done = time_step.last()
    
    return o_1, r, done

# DDPG algorithm
class DDPG:
    def __init__(self, arglist):
        self.arglist = arglist
        
        random.seed(self.arglist.seed)
        np.random.seed(self.arglist.seed)
        torch.manual_seed(self.arglist.seed)
        
        self.env = suite.load(domain_name=self.arglist.domain, task_name=self.arglist.task, task_kwargs={'random': self.arglist.seed})
        obs_spec = self.env.observation_spec()
        action_spec = self.env.action_spec()
        self.obs_size = np.sum([math.prod(obs_spec[k].shape) for k in obs_spec])
        self.action_size = math.prod(action_spec.shape)

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.actor = mu_FC(self.obs_size,self.action_size).to(self.device)
        
        if self.arglist.mode == "train":
            self.actor_target = deepcopy(self.actor)       
            self.actor_loss_fn =  torch.nn.MSELoss()

            self.critic = Q_FC(self.obs_size,self.action_size).to(self.device)
            self.critic_target = deepcopy(self.critic)                   
            self.critic_loss_fn =  torch.nn.MSELoss()
            
            path = "./log/"+self.arglist.domain+"_"+self.arglist.task
            self.exp_dir = os.path.join(path, "seed_"+str(self.arglist.seed))
            self.model_dir = os.path.join(self.exp_dir, "models")
            self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")

            if self.arglist.resume:
                checkpoint = torch.load(os.path.join(self.model_dir,"backup.ckpt"))
                self.start_episode = checkpoint['episode'] + 1

                self.actor.load_state_dict(checkpoint['actor'])
                self.actor_target.load_state_dict(checkpoint['actor_target'])
                self.critic.load_state_dict(checkpoint['critic'])
                self.critic_target.load_state_dict(checkpoint['critic_target'])

                self.replay_buffer = checkpoint['replay_buffer']

            else:
                self.start_episode = 0

                self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)

                if not os.path.exists(path):
                    os.makedirs(path)
                os.mkdir(self.exp_dir)
                os.mkdir(self.tensorboard_dir)
                os.mkdir(self.model_dir)

            for param in self.actor_target.parameters():
                param.requires_grad = False
            
            for param in self.critic_target.parameters():
                param.requires_grad = False

            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.arglist.lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.arglist.lr)

            if self.arglist.resume:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

                print("Done loading checkpoint ...")

            self.train()

        elif self.arglist.mode == "eval":
            checkpoint = torch.load(self.arglist.checkpoint,map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            ep_r_list = self.eval(self.arglist.episodes,self.arglist.render,self.arglist.save_video)

    def save_checkpoint(self, name):
        checkpoint = {'actor' : self.actor.state_dict()}
        torch.save(checkpoint, os.path.join(self.model_dir, name))

    def save_backup(self, episode):
        checkpoint = {'episode' : episode,\
                      'actor' : self.actor.state_dict(),\
                      'actor_optimizer': self.actor_optimizer.state_dict(),\
                      'critic' : self.critic.state_dict(),\
                      'critic_optimizer': self.critic_optimizer.state_dict(),\
                      'actor_target' : self.actor_target.state_dict(),\
                      'critic_target' : self.critic_target.state_dict(),\
                      'replay_buffer' : self.replay_buffer \
                      }
        torch.save(checkpoint, os.path.join(self.model_dir, "backup.ckpt"))

    def soft_update(self, target, source, tau):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)

    def train(self):
        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        for episode in range(self.arglist.episodes):
            o,_,_ = process_observation(self.env.reset())
            ep_r = 0
            while True:
                with torch.no_grad():
                    a = self.actor(torch.tensor(o, dtype=torch.float64, device=self.device).unsqueeze(0))
                a = a.cpu().numpy()[0] 
                a += np.random.normal(0.0, 0.2, self.action_size) # Gaussian Noise
                a = np.clip(a,-1.0,1.0)
                o_1, r, done = process_observation(self.env.step(a))

                self.replay_buffer.push(o, a, r, o_1)

                ep_r += r
                o = o_1
                if self.replay_buffer.__len__() >= self.arglist.replay_fill:
                    O, A, R, O_1 = self.replay_buffer.sample(self.arglist.batch_size)

                    q_value = self.critic(O, A)

                    with torch.no_grad():
                        next_q_value = self.critic_target(O_1,self.actor_target(O_1))                                    
                    expected_q_value = R + self.arglist.gamma * next_q_value

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
                        # Evaluate agent performance
                        eval_ep_r_list = self.eval(self.arglist.eval_over)
                        writer.add_scalar('eval_ep_r', np.mean(eval_ep_r_list), episode)
                        self.save_checkpoint(str(episode)+".ckpt")
                    if (episode % 250 == 0 or episode == self.arglist.episodes-1) and episode > self.start_episode:
                        self.save_backup(episode)
                    break     

    def eval(self, episodes, render=False, save_video=False):
        # Evaluate agent performance over several episodes

        if render and save_video: 
            t = 0
            folder = "./media/"+self.arglist.domain+"_"+self.arglist.task
            subprocess.call(["mkdir","-p",folder])

        ep_r_list = []
        for episode in range(episodes):
            if render:
                vid = None
            o,_,_ = process_observation(self.env.reset())
            ep_r = 0
            while True:
                with torch.no_grad():
                    a = self.actor(torch.tensor(o, dtype=torch.float64, device=self.device).unsqueeze(0))
                a = a.cpu().numpy()[0] 
                o_1, r, done = process_observation(self.env.step(a))
                if render:
                    img = self.env.physics.render(height=240,width=240,camera_id=0)
                    if vid is None:
                        vid = plt.imshow(img)
                    else:
                        vid.set_data(img)
                    plt.axis('off')
                    plt.pause(0.01)
                    plt.draw()
                    if save_video:
                        plt.savefig(folder + "/file%04d.png" % t, bbox_inches='tight')
                        t += 1
                ep_r += r
                o = o_1
                if done:
                    ep_r_list.append(ep_r)
                    if render:
                        print("Episode finished with total reward ",ep_r)
                        plt.pause(0.5)                    
                    break        
        if self.arglist.mode == "eval":
            print("Average return :",np.mean(ep_r_list))
            if save_video:
                os.chdir(folder)
                subprocess.call(['ffmpeg', '-i', 'file%04d.png','-r','10','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-pix_fmt', 'yuv420p','video.mp4'])
                for file_name in glob.glob("*.png"):
                    os.remove(file_name)
                subprocess.call(['ffmpeg','-i','video.mp4','video.gif'])
        
        return ep_r_list   

def parse_args():
    parser = argparse.ArgumentParser("DDPG")
    # Common settings
    parser.add_argument("--domain", type=str, default="", help="cartpole / reacher")
    parser.add_argument("--task", type=str, default="", help="swingup / hard")
    parser.add_argument("--mode", type=str, default="", help="train or eval")
    parser.add_argument("--episodes", type=int, default=0, help="number of episodes")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    # Core training parameters
    parser.add_argument("--resume", action="store_true", default=False, help="resume training")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="actor, critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--tau", type=float, default=0.005, help="soft target update parameter")
    parser.add_argument("--replay-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--replay-fill", type=int, default=int(1e4), help="elements in replay buffer before training starts")
    parser.add_argument("--eval-every", type=int, default=50, help="eval every _ episodes during training")
    parser.add_argument("--eval-over", type=int, default=50, help="each time eval over _ episodes")
    # Eval settings
    parser.add_argument("--checkpoint", type=str, default="", help="path to checkpoint")
    parser.add_argument("--render", action="store_true", default=False, help="render")
    parser.add_argument("--save-video", action="store_true", default=False, help="save video")

    return parser.parse_args()

if __name__ == '__main__':
    arglist = parse_args()
    ddpg = DDPG(arglist)
