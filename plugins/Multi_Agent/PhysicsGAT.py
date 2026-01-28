import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsAttentionLayer(nn.Module):
    def __init__(self, ego_dim, neighbor_dim, hidden_dim):
        super(PhysicsAttentionLayer, self).__init__()
        self.ego_encoder = nn.Linear(ego_dim, hidden_dim)
        self.neighbor_encoder = nn.Linear(neighbor_dim, hidden_dim)
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, ego_state, neighbor_states, mask=None):
        batch_size, num_neighbors, _ = neighbor_states.size()
        ego_embed = self.ego_encoder(ego_state).unsqueeze(1)
        neigh_embed = self.neighbor_encoder(neighbor_states)
        ego_expanded = ego_embed.repeat(1, num_neighbors, 1)
        combined = torch.cat([ego_expanded, neigh_embed], dim=-1)
        scores = self.attn_net(combined)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.sum(attn_weights * neigh_embed, dim=1)
        return context, attn_weights


class MoE_GAT_Actor(nn.Module):
    def __init__(self, ego_dim, neighbor_dim, hidden_dim, action_dim):
        super(MoE_GAT_Actor, self).__init__()

        self.gat = PhysicsAttentionLayer(ego_dim, neighbor_dim, hidden_dim)

        self.shared_net = nn.Sequential(
            nn.Linear(hidden_dim + ego_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.router = nn.Linear(hidden_dim // 2, 4)

        self.expert_speed = nn.Linear(hidden_dim // 2, 1)
        self.expert_hdg = nn.Linear(hidden_dim // 2, 1)

        self.expert_alt = nn.Linear(hidden_dim // 2, 3)

        self.alt_values = torch.tensor([-1.0, 0.0, 1.0])

    def forward(self, ego, neighbors, mask=None, hard=False):
        context, attn_weights = self.gat(ego, neighbors, mask)
        features = torch.cat([ego, context], dim=1)
        shared_feat = self.shared_net(features)

        router_logits = self.router(shared_feat)
        router_weights = F.gumbel_softmax(router_logits, tau=1.0, hard=hard, dim=-1)

        out_spd = torch.tanh(self.expert_speed(shared_feat))
        out_hdg = torch.tanh(self.expert_hdg(shared_feat))

        alt_logits = self.expert_alt(shared_feat)
        alt_probs = F.gumbel_softmax(alt_logits, tau=1.0, hard=hard, dim=-1)

        ref_vals = self.alt_values.to(ego.device)
        out_alt = torch.matmul(alt_probs, ref_vals).unsqueeze(1)

        batch_size = ego.size(0)
        zeros = torch.zeros(batch_size, 1).to(ego.device)

        act_noop = torch.cat([zeros, zeros, zeros], dim=1).unsqueeze(1)

        act_spd = torch.cat([out_spd, zeros, zeros], dim=1).unsqueeze(1)

        act_alt = torch.cat([zeros, out_alt, zeros], dim=1).unsqueeze(1)

        act_hdg = torch.cat([zeros, zeros, out_hdg], dim=1).unsqueeze(1)

        all_expert_actions = torch.cat([act_noop, act_spd, act_alt, act_hdg], dim=1)

        final_action = torch.matmul(router_weights.unsqueeze(1), all_expert_actions).squeeze(1)

        return final_action, router_logits, attn_weights

class GAT_Critic(nn.Module):
    def __init__(self, ego_dim, neighbor_dim, hidden_dim, action_dim):
        super(GAT_Critic, self).__init__()
        self.gat = PhysicsAttentionLayer(ego_dim, neighbor_dim, hidden_dim)
        self.decode = nn.Sequential(
            nn.Linear(hidden_dim + ego_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, ego, neighbors, action, mask=None):
        context, _ = self.gat(ego, neighbors, mask)
        combined = torch.cat([ego, context, action], dim=1)
        q_value = self.decode(combined)
        return q_value