import torch
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple

from flood_monitoring.config_flood import FloodConfig
from flood_monitoring.models.gru_attention_flood import FloodGRUActor, FloodGRUCritic
from flood_monitoring.models.apf_flood import FloodAPF

# Define ReplayBuffer tuple
Experience = namedtuple('Experience', 
                        field_names=['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer for MATD3 agent"""
    
    def __init__(self, buffer_size=FloodConfig.BUFFER_SIZE, batch_size=FloodConfig.BATCH_SIZE):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = FloodConfig.DEVICE
    
    def add(self, state, action, reward, next_state, done):
        """Add new experience to memory"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def sample(self):
        """Sample a batch of experiences"""
        indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            exp = self.memory[idx]
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)

class MATD3Agent:
    """Multi-Agent Twin Delayed DDPG with GRU for UAV flood monitoring"""
    
    def __init__(self, agent_id, state_dim=FloodConfig.STATE_DIM, action_dim=FloodConfig.ACTION_DIM, 
                 hidden_dim=FloodConfig.HIDDEN_DIM, config=None):
        self.agent_id = agent_id
        self.config = config or FloodConfig
        self.device = self.config.DEVICE
        
        # Initialize actor networks
        self.actor = FloodGRUActor(state_dim).to(self.device)
        self.actor_target = FloodGRUActor(state_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        
        # Initialize critic networks
        self.critic = FloodGRUCritic(state_dim).to(self.device)
        self.critic_target = FloodGRUCritic(state_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Hidden states for recurrent networks
        self.actor_hidden = None
        self.critic_hidden = None
        self.target_actor_hidden = None
        self.target_critic_hidden = None
        
        # APF navigation module
        self.apf = FloodAPF(self.config)
        
        # Exploration noise
        self.noise_scale = 0.1
        self.noise_decay = 0.99
        
        # Training parameters
        self.gamma = self.config.GAMMA
        self.tau = self.config.TAU
        self.policy_noise = self.config.POLICY_NOISE
        self.noise_clip = self.config.NOISE_CLIP
        self.policy_freq = self.config.POLICY_FREQ
        self.learn_steps = 0
        
    def select_action(self, state, noise=0.0, deterministic=False, flood_grid=None, visited_mask=None,
                     agent_visited_mask=None, agent_positions=None):
        """
        Select action based on current policy with optional noise for exploration
        
        Args:
            state: Current state observation
            noise: Exploration noise scale
            deterministic: If True, return deterministic action
            flood_grid, visited_mask, agent_visited_mask, agent_positions: 
                Optional APF inputs if hybrid navigation is desired
                
        Returns:
            Action as (dx, dy) direction pair
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Get action from actor network (now a continuous 2D vector)
        with torch.no_grad():
            action, self.actor_hidden, _, _ = self.actor(state_tensor, self.actor_hidden)
            action = action.cpu().numpy()
            
            # Add exploration noise if needed
            if not deterministic and noise > 0:
                action += noise * np.random.normal(0, 1, size=action.shape)
                action = np.clip(action, -1.0, 1.0)
        
        # Convert action to discrete direction first
        action_idx = np.argmax(action)
        
        # Calculate APF direction if components are provided
        if all(v is not None for v in [flood_grid, visited_mask, agent_visited_mask, agent_positions]):
            position = agent_positions[self.agent_id]
            
            # Get assigned region for this agent (divide map into sectors)
            region_mask = self._get_agent_region_mask(flood_grid.shape, self.agent_id)
            
            # Calculate APF direction with region focus
            apf_direction = self.apf.compute_force(
                position, 
                flood_grid * region_mask,  # Focus on assigned region
                visited_mask,
                agent_visited_mask,
                agent_positions,
                self.agent_id
            )
            
            # Blend APF with policy action
            return self.apf.get_action_blend(apf_direction, action_idx)
        
        # If no APF, just convert the discrete action to a direction vector
        directions = [
            (0, -1),   # North
            (1, -1),   # Northeast
            (1, 0),    # East
            (1, 1),    # Southeast
            (0, 1),    # South
            (-1, 1),   # Southwest
            (-1, 0),   # West
            (-1, -1),  # Northwest
            (0, 0)     # Hover
        ]
        
        return np.array(directions[action_idx])
    
    def _get_agent_region_mask(self, grid_shape, agent_id):
        """Generate a mask that emphasizes agent's assigned region"""
        num_agents = self.config.NUM_AGENTS
        region_mask = np.ones(grid_shape)
        
        # Simple region division based on agent index
        if num_agents > 1:
            # Divide map into regions (can be improved with clustering)
            # Example: for 4 agents, divide into quadrants
            # For now, use a softer mask that emphasizes the region
            # but still allows crossing boundaries when needed
            
            # Create a gradient that emphasizes agent's region
            h, w = grid_shape
            y, x = np.mgrid[0:h, 0:w]
            
            # Calculate centers of regions
            centers = []
            sqrt_agents = int(np.ceil(np.sqrt(num_agents)))
            for i in range(num_agents):
                row = i // sqrt_agents
                col = i % sqrt_agents
                center_y = h * (row + 0.5) / sqrt_agents
                center_x = w * (col + 0.5) / sqrt_agents
                centers.append((center_y, center_x))
            
            # Calculate distance to this agent's center
            cy, cx = centers[agent_id]
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            
            # Create soft mask (higher values in agent's region)
            max_dist = np.sqrt(h**2 + w**2) / 2
            region_mask = 1.0 + (1.0 - np.minimum(dist / max_dist, 1.0))
        
        return region_mask
    
    def update(self, replay_buffer, batch_size=None):
        """Update policy and value parameters using batched experiences"""
        batch_size = batch_size or self.config.BATCH_SIZE
        
        if len(replay_buffer) < batch_size:
            return {
                'actor_loss': 0,
                'critic_loss': 0,
                'learning_rate': self.actor_optimizer.param_groups[0]['lr']
            }
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample()
        
        # Update critics
        with torch.no_grad():
            # Get next actions from target actor with noise
            next_action_probs, _, _, _ = self.actor_target(next_states)
            noise = torch.clamp(torch.randn_like(next_action_probs) * self.policy_noise, 
                               -self.noise_clip, self.noise_clip)
            
            # Add noise to action probabilities
            noisy_probs = next_action_probs + noise
            noisy_probs = F.softmax(noisy_probs, dim=1)
            
            # Compute target Q values
            target_q1, target_q2, _ = self.critic_target(next_states, noisy_probs)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Compute current Q values
        current_q1, current_q2, _ = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        actor_loss = torch.tensor(0.0).to(self.device)
        if self.learn_steps % self.policy_freq == 0:
            # Compute actor loss
            action_probs, _, flood_estimate, _ = self.actor(states)
            q1, _, _ = self.critic(states, action_probs)
            
            # Actor loss combines policy gradient and auxiliary flood estimation
            actor_loss = -q1.mean()
            
            # Update actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        
        self.learn_steps += 1
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'learning_rate': self.actor_optimizer.param_groups[0]['lr']
        }
        
    def reset_hidden_states(self):
        """Reset hidden states for recurrent networks"""
        self.actor_hidden = None
        self.critic_hidden = None
        self.target_actor_hidden = None
        self.target_critic_hidden = None
    
    def _soft_update(self, source, target):
        """Soft update target model parameters: θ′ ← τθ + (1 − τ)θ′"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, directory, name):
        """Save model parameters"""
        torch.save(self.actor.state_dict(), f"{directory}/{name}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{name}_critic.pth")
        torch.save(self.actor_target.state_dict(), f"{directory}/{name}_actor_target.pth")
        torch.save(self.critic_target.state_dict(), f"{directory}/{name}_critic_target.pth")
    
    def load(self, directory, name):
        """Load model parameters"""
        self.actor.load_state_dict(torch.load(f"{directory}/{name}_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{directory}/{name}_critic.pth", map_location=self.device))
        self.actor_target.load_state_dict(torch.load(f"{directory}/{name}_actor_target.pth", map_location=self.device))
        self.critic_target.load_state_dict(torch.load(f"{directory}/{name}_critic_target.pth", map_location=self.device))