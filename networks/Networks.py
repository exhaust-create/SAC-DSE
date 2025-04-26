import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

# Critic
class Q_Critic_NN(nn.Module):
    """
    Args
    -------
    input_dim:
        = num_components * embed_dim
    output_dim:
        = num_components
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None, device='cpu'):
        super(Q_Critic_NN, self).__init__()
        self.device = device
        self.input_dim = input_dim 
        fc_dim = 10*input_dim if hidden_dim is None else hidden_dim
        self.fc1 = nn.Linear(input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, output_dim)
        self.to(device)

    def forward(self, state):
        """
        Args
        -------
        state:
            Shape is like (num_batch, num_components, embed_dim)
        
        Returns
        -------
        x:
            Shape is like (num_batch, num_components)
        """
        x = state.float().view(-1, self.input_dim)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class V_Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=None, device='cpu'):
        super(V_Critic, self).__init__()
        self.device = device
        fc_dim = 10*state_dim if hidden_dim is None else hidden_dim
        self.fc1 = nn.Linear(state_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, 1)
        self.to(device)
    
    def forward(self, state):
        x = F.elu(self.fc1(state.float()))
        x = F.elu(self.fc2(x))
        value = self.fc3(x)
        return value

class Actor_NN(nn.Module):
    def __init__(self, state_dim, output_dim, device='cpu'):
        super(Actor_NN, self).__init__()
        self.device = device
        self.input_dim = state_dim
        fc_dim = 10*state_dim
        self.fc1 = nn.Linear(state_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, output_dim)
        self.to(device)

    def forward(self, state, option, action_mask):
        """
        Args
        -------
        option:
            Should be the idx of the specified component.
        """
        x = torch.concat([state, 2*option], dim=1).float()
        x = x.view(-1, self.input_dim)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        action_probs = self.fc3(x)
        action_probs = action_probs + ~torch.tensor(action_mask).to(self.device)*(-1e9)
        action_probs = F.softmax(action_probs, dim=1)
        action_probs = action_probs + torch.tensor(action_mask).to(self.device)*0.0001
        return action_probs
    
class Actor_NN_no_option(nn.Module):
    def __init__(self, state_dim, output_dim, fc_dim=None, device='cpu'):
        super(Actor_NN_no_option, self).__init__()
        self.device = device
        self.output_dim = torch.tensor(output_dim)
        fc_dim = 4*state_dim if fc_dim is None else fc_dim
        self.fc1 = nn.Linear(state_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, output_dim)
        self.to(device)

    def forward(self, state, action_mask):
        x = F.elu(self.fc1(state.float()))
        x = F.elu(self.fc2(x))
        action_probs = self.fc3(x)
        action_probs = action_probs + ~action_mask*(-1e9)
        action_probs = F.softmax(action_probs/torch.sqrt(self.output_dim), dim=1)
        action_probs = action_probs + action_mask*1e-9
        return action_probs