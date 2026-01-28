import torch
import torch.nn.functional as F
import numpy as np
import random
import collections
from plugins.Multi_Agent.PhysicsGAT import MoE_GAT_Actor, GAT_Critic


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, ego, neighbors, mask, action, reward, next_ego, next_neighbors, next_mask, done):
        self.buffer.append((ego, neighbors, mask, action, reward, next_ego, next_neighbors, next_mask, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        ego, neigh, mask, a, r, n_ego, n_neigh, n_mask, d = zip(*transitions)
        return (
            np.array(ego), np.array(neigh), np.array(mask), np.array(a),
            np.array(r), np.array(n_ego), np.array(n_neigh), np.array(n_mask), np.array(d)
        )

    def size(self):
        return len(self.buffer)


class GAT_DDPG:
    def __init__(self, ego_dim, neighbor_dim, hidden_dim, action_dim, device):
        self.device = device
        self.gamma = 0.99
        self.tau = 0.01
        self.action_dim = action_dim

        self.actor = MoE_GAT_Actor(ego_dim, neighbor_dim, hidden_dim, action_dim).to(device)
        self.critic = GAT_Critic(ego_dim, neighbor_dim, hidden_dim, action_dim).to(device)

        self.target_actor = MoE_GAT_Actor(ego_dim, neighbor_dim, hidden_dim, action_dim).to(device)
        self.target_critic = GAT_Critic(ego_dim, neighbor_dim, hidden_dim, action_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def take_action(self, ego, neighbors, mask, noise_sigma=0.0):
        ego = torch.FloatTensor(ego).unsqueeze(0).to(self.device)
        neighbors = torch.FloatTensor(neighbors).unsqueeze(0).to(self.device)
        mask = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

        action, router_logits, _ = self.actor(ego, neighbors, mask, hard=True)
        action = action.cpu().detach().numpy()[0]

        expert_idx = torch.argmax(router_logits, dim=1).item()

        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma)

            if expert_idx == 0:
                pass

            elif expert_idx == 1:
                action[0] = np.clip(action[0] + noise, -1.0, 1.0)
                action[1] = 0.0
                action[2] = 0.0

            elif expert_idx == 2:
                action[0] = 0.0
                action[2] = 0.0

            elif expert_idx == 3:
                action[2] = np.clip(action[2] + noise, -1.0, 1.0)
                action[0] = 0.0
                action[1] = 0.0

        return action, expert_idx

    def update(self, transition_batch):
        ego = torch.FloatTensor(transition_batch['ego']).to(self.device)
        neigh = torch.FloatTensor(transition_batch['neigh']).to(self.device)
        mask = torch.FloatTensor(transition_batch['mask']).to(self.device)
        action = torch.FloatTensor(transition_batch['action']).to(self.device)
        reward = torch.FloatTensor(transition_batch['reward']).view(-1, 1).to(self.device)
        next_ego = torch.FloatTensor(transition_batch['next_ego']).to(self.device)
        next_neigh = torch.FloatTensor(transition_batch['next_neigh']).to(self.device)
        next_mask = torch.FloatTensor(transition_batch['next_mask']).to(self.device)
        done = torch.FloatTensor(transition_batch['done']).view(-1, 1).to(self.device)

        with torch.no_grad():
            next_action, _, _ = self.target_actor(next_ego, next_neigh, next_mask, hard=True)
            target_q = self.target_critic(next_ego, next_neigh, next_action, next_mask)
            target_v = reward + (1 - done) * self.gamma * target_q

        current_q = self.critic(ego, neigh, action, mask)
        critic_loss = F.mse_loss(current_q, target_v)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pred_action, router_logits, _ = self.actor(ego, neigh, mask, hard=False)

        actor_loss = -self.critic(ego, neigh, pred_action, mask).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)

    def predict_with_explanation(self, ego, neighbors, mask):
        ego = torch.FloatTensor(ego).unsqueeze(0).to(self.device)
        neighbors = torch.FloatTensor(neighbors).unsqueeze(0).to(self.device)
        mask = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

        action, router_logits, attn_weights = self.actor(ego, neighbors, mask, hard=True)

        action = action.cpu().detach().numpy()[0]
        expert_idx = torch.argmax(router_logits, dim=1).item()

        attention = attn_weights.squeeze().cpu().detach().numpy()

        router_probs = torch.softmax(router_logits, dim=1).cpu().detach().numpy()[0]

        return action, expert_idx, attention, router_probs

    def compute_feature_gradients(self, ego, neighbors, mask):
        ego_tensor = torch.FloatTensor(ego).unsqueeze(0).to(self.device)
        ego_tensor.requires_grad_(True)

        neigh_tensor = torch.FloatTensor(neighbors).unsqueeze(0).to(self.device)
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

        action, _, _ = self.actor(ego_tensor, neigh_tensor, mask_tensor, hard=True)

        target = action.pow(2).sum()

        self.actor.zero_grad()
        target.backward()

        grads = ego_tensor.grad.abs().cpu().detach().numpy()[0]

        return grads

    def _soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        torch.save(self.actor.state_dict(), path + "_actor.pth")
        torch.save(self.critic.state_dict(), path + "_critic.pth")

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(path + "_critic.pth", map_location=self.device))