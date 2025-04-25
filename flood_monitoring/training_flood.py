import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

from flood_monitoring.environment.flood_env import FloodMonitoringEnv
from flood_monitoring.agents.matd3_flood import MATD3Agent, ReplayBuffer
from flood_monitoring.config_flood import FloodConfig
from flood_monitoring.models.apf_flood import FloodAPF

def train_matd3_flood():
    """Train MATD3 agents for flood monitoring environment"""
    
    config = FloodConfig
    
    # Create environment
    env = FloodMonitoringEnv(render_mode=None)  # Set to "human" for visualization
    
    # Create agents
    agents = [
        MATD3Agent(agent_id=i, 
                  state_dim=config.STATE_DIM,
                  action_dim=config.ACTION_DIM,
                  hidden_dim=config.HIDDEN_DIM,
                  config=config) 
        for i in range(config.NUM_AGENTS)
    ]
    
    # Create replay buffers (one per agent)
    replay_buffers = [
        ReplayBuffer(buffer_size=config.BUFFER_SIZE, 
                    batch_size=config.BATCH_SIZE)
        for _ in range(config.NUM_AGENTS)
    ]
    
    # Training statistics
    episode_rewards = []
    coverage_stats = []
    avg_rewards = []
    episode_durations = []
    
    # Create directories for saving models and results
    os.makedirs("flood_monitoring/models", exist_ok=True)
    os.makedirs("flood_monitoring/results", exist_ok=True)
    os.makedirs("flood_monitoring/visualizations", exist_ok=True)
    
    print("Starting training...")
    
    # Training loop
    for episode in range(config.MAX_EPISODES):
        start_time = time.time()
        
        # Reset environment and agents
        observations, _ = env.reset()
        
        # Reset hidden states of agents
        for agent in agents:
            agent.reset_hidden_states()
        
        episode_reward = 0
        noise_scale = max(0.05, config.POLICY_NOISE * (0.9 ** (episode // 50)))
        
        # Episode loop
        done = False
        total_timesteps = 0
        
        while not done:
            # Select actions for all agents
            actions = []
            
            for i, agent in enumerate(agents):
                # Get APF components if needed
                if config.APF_BLEND_FACTOR > 0:
                    action = agent.select_action(
                        observations[i],
                        noise=noise_scale,
                        deterministic=False,
                        flood_grid=env.flood_grid.grid,
                        visited_mask=env.global_visited_mask,
                        agent_visited_mask=env.agent_visited_masks[i],
                        agent_positions=env.agent_positions
                    )
                else:
                    action = agent.select_action(
                        observations[i],
                        noise=noise_scale,
                        deterministic=False
                    )
                    
                actions.append(action)
            
            # Take a step in the environment
            next_observations, rewards, dones, truncated, infos = env.step(actions)
            
            # Store experiences in replay buffers
            for i in range(config.NUM_AGENTS):
                # Convert direction back to index for the replay buffer
                action_idx = None
                if np.all(actions[i] == 0):  # Hover
                    action_idx = 8
                else:
                    # Find the closest discrete direction
                    dx, dy = actions[i]
                    closest_dir = 0
                    best_dot = -float('inf')
                    directions = [
                        (0, -1),   # North
                        (1, -1),   # Northeast
                        (1, 0),    # East
                        (1, 1),    # Southeast
                        (0, 1),    # South
                        (-1, 1),   # Southwest
                        (-1, 0),   # West
                        (-1, -1),  # Northwest
                    ]
                    for dir_idx, (dir_x, dir_y) in enumerate(directions):
                        dot_product = dx * dir_x + dy * dir_y
                        if dot_product > best_dot:
                            best_dot = dot_product
                            closest_dir = dir_idx
                    
                    action_idx = closest_dir
                
                replay_buffers[i].add(
                    observations[i],
                    np.eye(config.ACTION_DIM)[action_idx],  # One-hot encoded action
                    rewards[i],
                    next_observations[i],
                    dones[i] or truncated[i]
                )
            
            # Update observations
            observations = next_observations
            
            # Calculate total reward
            episode_reward += sum(rewards)
            
            # Check if episode is done
            done = all(dones) or all(truncated)
            total_timesteps += 1
        
        # Train agents after episode completion
        for i, agent in enumerate(agents):
            if len(replay_buffers[i]) > config.BATCH_SIZE:
                for _ in range(total_timesteps):  # Update once per step
                    agent.update(replay_buffers[i])
        
        # Track statistics
        episode_rewards.append(episode_reward)
        coverage_stats.append(env.coverage)
        episode_durations.append(total_timesteps)
        
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)
        
        # Print episode statistics
        print(f"Episode {episode+1}/{config.MAX_EPISODES} | "
              f"Steps: {total_timesteps} | "
              f"Reward: {episode_reward:.2f} | "
              f"Coverage: {env.coverage:.2%} | "
              f"Noise: {noise_scale:.2f} | "
              f"Time: {time.time() - start_time:.2f}s")
        
        # Evaluate and save models periodically
        if (episode + 1) % config.EVAL_FREQ == 0:
            # Save models
            for i, agent in enumerate(agents):
                agent.save("flood_monitoring/models", f"agent_{i}_episode_{episode+1}")
            
            # Save visualization
            if config.VISUALIZATION_FREQ > 0 and (episode + 1) % config.VISUALIZATION_FREQ == 0:
                # Temporarily set render mode to get visualization
                env.close()
                eval_env = FloodMonitoringEnv(render_mode="human")
                
                # Run one episode for visualization
                eval_observations, _ = eval_env.reset()
                eval_done = False
                
                while not eval_done:
                    eval_actions = []
                    for i, agent in enumerate(agents):
                        eval_action = agent.select_action(
                            eval_observations[i],
                            noise=0.0,  # No noise during evaluation
                            deterministic=True,
                            flood_grid=eval_env.flood_grid.grid,
                            visited_mask=eval_env.global_visited_mask,
                            agent_visited_mask=eval_env.agent_visited_masks[i],
                            agent_positions=eval_env.agent_positions
                        )
                        eval_actions.append(eval_action)
                    
                    eval_observations, _, eval_dones, eval_truncated, _ = eval_env.step(eval_actions)
                    eval_done = all(eval_dones) or all(eval_truncated)
                
                # Save visualization data
                eval_env.save_data(episode + 1)
                eval_env.close()
                
                # Restore training environment
                env = FloodMonitoringEnv(render_mode=None)
    
    # Save training progress plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(coverage_stats)
    plt.title('Coverage Percentage')
    plt.xlabel('Episode')
    plt.ylabel('Coverage (%)')
    plt.ylim(0, 1)
    
    plt.subplot(1, 3, 3)
    plt.plot(episode_durations)
    plt.title('Episode Duration')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig('flood_monitoring/results/training_progress.png')
    
    # Save final models
    for i, agent in enumerate(agents):
        agent.save("flood_monitoring/models", f"agent_{i}_final")
    
    print("Training complete!")
    return agents

if __name__ == "__main__":
    train_matd3_flood()