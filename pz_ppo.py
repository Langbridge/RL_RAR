"""Basic code which shows what it's like to run PPO on the Pistonball env using the parallel API, this code is inspired by CleanRL.

This code is exceedingly basic, with no logging or weights saving.
The intention was for users to have a (relatively clean) ~200 line file to refer to when they want to design their own learning algorithm.

Author: Jet (https://github.com/jjshoots)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from pettingzoo_env import CustomEnvironment

class Agent(nn.Module):
    def __init__(self, num_actions, kernel_x=1, kernel_y=2):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=(kernel_x, kernel_y))
        self.linear = nn.Linear(num_actions-kernel_y+1, 32)

        self.actor = self._layer_init(nn.Linear(32, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(32, 1))

    def forward(self, x):
        x = torch.unsqueeze(x, dim=0)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = x[0,:,:]

        x = self.linear(x)
        return x

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action_mask, action=None):
        hidden = self.forward(x)
        logits = self.actor(hidden)
        logits = torch.where(action_mask, logits, -1e8)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        print(logits)
        print(probs.probs)
        print(probs.log_prob(action))
        # [agents]
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def batchify_obs(env, new_obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    obs = []
    masks = []
    for agent in env.possible_agents:
        if agent in env.agents:
            obs.append(new_obs[agent]['pollution'])
            masks.append(new_obs[agent]['action_mask'])
        else:
            obs.append(np.zeros_like(new_obs[env.agents[0]]['pollution']))
            masks.append(np.zeros_like(new_obs[env.agents[0]]['action_mask']))

    obs = np.stack(obs, axis=0)
    obs = torch.tensor(obs.astype(np.float32)).to(device)
    masks = np.stack(masks, axis=0)
    masks = torch.tensor(masks.astype(bool)).to(device)

    return obs, masks


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    out = []
    for agent in env.possible_agents:
        if agent in env.agents:
            out.append(x[agent])
        else:
            out.append(np.zeros_like(x[env.agents[0]]))

    out = np.stack(out, axis=0)
    out = torch.tensor(out).to(device)

    return out


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.agents)}

    return x


def obs_to_mask(obs):
    return torch.where((obs < 1e5), True, False)

if __name__ == "__main__":

    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 16 #32
    max_cycles = 256 #125
    total_episodes = 64

    """ ENV SETUP """
    env = CustomEnvironment(
        num_agents=3,
        map_size=3,
        num_iters=1_000
    )
    num_agents = len(env.possible_agents)
    num_actions = env.num_nodes
    observation_size = env.num_nodes
    print(f"Setup env with {num_agents} agents, {num_actions} actions and {observation_size} obs")

    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = -1
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents, observation_size)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):

        # collect an episode
        with torch.no_grad():

            # collect observations and convert to batch of torch tensors
            next_obs = env.reset(seed=None)
            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            for step in range(0, max_cycles):

                # rollover the observation
                obs, action_mask = batchify_obs(env, next_obs, device)

                # get action from the agent
                actions, logprobs, _, values = agent.get_action_and_value(obs, action_mask)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )
                print(obs, actions, logprobs, values, '\n')

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # # if we reach termination or truncation, end
                # if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                #     end_step = step
                #     break
                if any([truncs[a] for a in truncs]):
                    print("truncating env...")
                    print(truncs)
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        # [:end_step]
        b_obs = torch.flatten(rb_obs, start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs, start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions, start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns, start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values, start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages, start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]

                mask = obs_to_mask(b_obs[batch_index])
                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index], mask, b_actions.long()[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"Training episode {episode}")
        print(f"Episodic Return: {np.mean(total_episodic_return)}")
        print(f"Episode Length: {end_step}")
        print("")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(clip_fracs)}")
        print(f"Explained Variance: {explained_var}")
        print("\n-------------------------------------------\n")