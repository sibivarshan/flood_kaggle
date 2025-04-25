import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flood_monitoring.config_flood import FloodConfig

class FloodGRUActor(nn.Module):
    """GRU-based actor for the UAV agent."""
    def __init__(self, state_dim=FloodConfig.STATE_DIM, action_dim=FloodConfig.ACTION_DIM, 
                 hidden_dim=FloodConfig.HIDDEN_DIM, config=None):
        super(FloodGRUActor, self).__init__()
        self.config = config or FloodConfig
        
        # GRU layers to process sequential data
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=self.config.GRU_LAYERS,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Add a flood estimation network (auxiliary task)
        self.flood_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.hidden_state = None
        
    def forward(self, state, hidden=None):
        # If just a single state is provided and not a batch
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        
        # Add sequence dimension if not present
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # [batch_size, seq_len=1, state_dim]
            
        # Use provided hidden state or initialize if None
        if hidden is None:
            batch_size = state.shape[0]
            hidden = torch.zeros(self.config.GRU_LAYERS, batch_size, 
                                self.config.HIDDEN_DIM).to(state.device)
                                
        # Pass through GRU
        gru_out, new_hidden = self.gru(state, hidden)
        
        # Apply attention if sequence length > 1
        if gru_out.shape[1] > 1:
            # Calculate attention weights
            attn_weights = self.attention(gru_out).squeeze(-1)  # [batch, seq_len]
            attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)  # [batch, 1, seq_len]
            
            # Apply attention
            context = torch.bmm(attn_weights, gru_out).squeeze(1)  # [batch, hidden_dim]
        else:
            context = gru_out.squeeze(1)  # [batch, hidden_dim]
        
        # Feed through policy network
        action_probs = F.softmax(self.policy(context), dim=-1)
        
        # Generate flood estimate from context (auxiliary task)
        flood_estimate = self.flood_estimator(context)
        
        return action_probs, new_hidden, context, flood_estimate

class FloodGRUCritic(nn.Module):
    """GRU-based critic with twin Q-networks for the UAV agent."""
    def __init__(self, state_dim=FloodConfig.STATE_DIM, action_dim=FloodConfig.ACTION_DIM, 
                 hidden_dim=FloodConfig.HIDDEN_DIM, config=None):
        super(FloodGRUCritic, self).__init__()
        self.config = config or FloodConfig
        
        # GRU layer to process sequential states
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=self.config.GRU_LAYERS,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Twin Q-networks
        # Make sure the dimensions match properly here - this was causing the error
        # The input should be hidden_dim (from GRU output) + action_dim (from the action input)
        combined_dim = hidden_dim + action_dim
        
        self.q1 = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.hidden_state = None
        
    def forward(self, state, action, hidden=None):
        # Handle input shape
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            
        # Add sequence dimension if not present
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # [batch_size, seq_len=1, state_dim]
            
        # Use provided hidden state or initialize if None
        if hidden is None:
            batch_size = state.shape[0]
            hidden = torch.zeros(self.config.GRU_LAYERS, batch_size, 
                                self.config.HIDDEN_DIM).to(state.device)
            
        # Process state sequence through GRU
        gru_out, new_hidden = self.gru(state, hidden)
        
        # Apply attention if sequence length > 1
        if gru_out.shape[1] > 1:
            # Calculate attention weights
            attn_weights = self.attention(gru_out).squeeze(-1)  # [batch, seq_len]
            attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)  # [batch, 1, seq_len]
            
            # Apply attention
            context = torch.bmm(attn_weights, gru_out).squeeze(1)  # [batch, hidden_dim]
        else:
            context = gru_out.squeeze(1)  # [batch, hidden_dim]
        
        # Ensure action has batch dimension
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        # Combine state representation with action
        combined = torch.cat([context, action], dim=1)
        
        # Get Q-values from twin networks
        q1_value = self.q1(combined)
        q2_value = self.q2(combined)
        
        return q1_value, q2_value, new_hidden