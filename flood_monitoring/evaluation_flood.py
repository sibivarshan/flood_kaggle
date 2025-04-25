import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

from flood_monitoring.environment.flood_env import FloodMonitoringEnv
from flood_monitoring.agents.matd3_flood import MATD3Agent
from flood_monitoring.config_flood import FloodConfig

def evaluate_matd3_flood(model_dir="flood_monitoring/models", model_prefix="agent", 
                        num_episodes=10, render=True, save_metrics=True):
    """
    Evaluate trained MATD3 agents on the flood monitoring environment
    
    Args:
        model_dir: Directory containing saved model files
        model_prefix: Prefix for model filenames (e.g., "agent" for "agent_0_final.pth")
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
        save_metrics: Whether to save evaluation metrics
    
    Returns:
        Dictionary of evaluation metrics
    """
    config = FloodConfig
    
    # Create environment
    render_mode = "human" if render else None
    env = FloodMonitoringEnv(render_mode=render_mode)
    
    # Create agents
    agents = [
        MATD3Agent(agent_id=i, 
                  state_dim=config.STATE_DIM,
                  action_dim=config.ACTION_DIM,
                  hidden_dim=config.HIDDEN_DIM,
                  config=config) 
        for i in range(config.NUM_AGENTS)
    ]
    
    # Load trained models
    for i, agent in enumerate(agents):
        agent.load(model_dir, f"{model_prefix}_{i}_final")
    
    # Metrics to track
    metrics = defaultdict(list)
    all_trajectories = []
    all_flood_maps = []
    all_coverage_maps = []
    all_coverage_over_time = []
    all_distances_traveled = []
    all_coverage_efficiency = []
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        print(f"Running evaluation episode {episode+1}/{num_episodes}...")
        observations, _ = env.reset()
        
        # Reset hidden states of agents
        for agent in agents:
            agent.reset_hidden_states()
        
        # Episode tracking
        episode_reward = 0
        timesteps = 0
        done = False
        episode_coverage_over_time = []
        
        # Store initial state
        episode_coverage_over_time.append((0, env.coverage))
        
        while not done:
            # Select actions for all agents
            actions = []
            
            for i, agent in enumerate(agents):
                action = agent.select_action(
                    observations[i],
                    noise=0.0,  # No exploration noise during evaluation
                    deterministic=True,
                    flood_grid=env.flood_grid.grid,
                    visited_mask=env.global_visited_mask,
                    agent_visited_mask=env.agent_visited_masks[i],
                    agent_positions=env.agent_positions
                )
                actions.append(action)
            
            # Take a step in the environment
            observations, rewards, dones, truncated, infos = env.step(actions)
            
            # Update metrics
            episode_reward += sum(rewards)
            timesteps += 1
            
            # Record coverage progress
            episode_coverage_over_time.append((timesteps, env.coverage))
            
            # Check if episode is done
            done = all(dones) or all(truncated)
        
        # Save per-episode metrics
        metrics['episode_reward'].append(episode_reward)
        metrics['episode_length'].append(timesteps)
        metrics['final_coverage'].append(env.coverage)
        metrics['high_flood_coverage'].append(infos[0]['high_flood_coverage'])
        metrics['total_distance'].append(np.sum(env.distances_traveled))
        metrics['mean_agent_efficiency'].append(np.mean(env.coverage_efficiency))
        metrics['overall_efficiency'].append(env.coverage / max(0.1, np.sum(env.distances_traveled)))
        
        # Track per-agent distances
        for i in range(config.NUM_AGENTS):
            metrics[f'agent_{i}_distance'].append(env.distances_traveled[i])
            metrics[f'agent_{i}_efficiency'].append(env.coverage_efficiency[i])
        
        # Track trajectories and maps
        trajectories = []
        for i in range(config.NUM_AGENTS):
            agent_trajectory = [(image['position'][0], image['position'][1]) 
                               for image in env.captured_images[i]]
            trajectories.append(agent_trajectory)
        
        all_trajectories.append(trajectories)
        all_flood_maps.append(env.flood_grid.grid.copy())
        all_coverage_maps.append(env.global_visited_mask.copy())
        all_coverage_over_time.append(episode_coverage_over_time)
        all_distances_traveled.append(env.distances_traveled.copy())
        all_coverage_efficiency.append(env.coverage_efficiency.copy())
        
        # Save visualization data for this episode
        env.save_data(f"eval_{episode+1}")
        
        print(f"Episode {episode+1} | "
              f"Reward: {episode_reward:.2f} | "
              f"Steps: {timesteps} | "
              f"Coverage: {env.coverage:.2%} | "
              f"High Flood Coverage: {infos[0]['high_flood_coverage']:.2%} | "
              f"Total Distance: {np.sum(env.distances_traveled):.2f} | "
              f"Efficiency: {env.coverage / max(0.1, np.sum(env.distances_traveled)):.4f}")
    
    # Close environment
    env.close()
    
    # Calculate aggregate metrics
    metrics_summary = {
        'mean_reward': np.mean(metrics['episode_reward']),
        'std_reward': np.std(metrics['episode_reward']),
        'mean_coverage': np.mean(metrics['final_coverage']),
        'std_coverage': np.std(metrics['final_coverage']),
        'mean_high_flood_coverage': np.mean(metrics['high_flood_coverage']),
        'std_high_flood_coverage': np.std(metrics['high_flood_coverage']),
        'mean_episode_length': np.mean(metrics['episode_length']),
        'std_episode_length': np.std(metrics['episode_length']),
        'mean_total_distance': np.mean(metrics['total_distance']),
        'std_total_distance': np.std(metrics['total_distance']),
        'mean_efficiency': np.mean(metrics['overall_efficiency']),
        'std_efficiency': np.std(metrics['overall_efficiency']),
    }
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Mean Reward: {metrics_summary['mean_reward']:.2f} ± {metrics_summary['std_reward']:.2f}")
    print(f"Mean Coverage: {metrics_summary['mean_coverage']:.2%} ± {metrics_summary['std_coverage']:.2%}")
    print(f"Mean High Flood Coverage: {metrics_summary['mean_high_flood_coverage']:.2%} ± {metrics_summary['std_high_flood_coverage']:.2%}")
    print(f"Mean Episode Length: {metrics_summary['mean_episode_length']:.1f} ± {metrics_summary['std_episode_length']:.1f}")
    print(f"Mean Total Distance: {metrics_summary['mean_total_distance']:.2f} ± {metrics_summary['std_total_distance']:.2f}")
    print(f"Mean Coverage Efficiency: {metrics_summary['mean_efficiency']:.4f} ± {metrics_summary['std_efficiency']:.4f}")
    
    # Save metrics if requested
    if save_metrics:
        os.makedirs("flood_monitoring/evaluation", exist_ok=True)
        
        # Save overall metrics
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv("flood_monitoring/evaluation/metrics.csv", index=False)
        
        # Save summary metrics
        summary_df = pd.DataFrame([metrics_summary])
        summary_df.to_csv("flood_monitoring/evaluation/summary_metrics.csv", index=False)
        
        # Plot coverage over time for all episodes
        plt.figure(figsize=(10, 6))
        for i, coverage_data in enumerate(all_coverage_over_time):
            steps, coverages = zip(*coverage_data)
            plt.plot(steps, coverages, label=f"Episode {i+1}", alpha=0.7)
            
        plt.title("Coverage Progress Over Time")
        plt.xlabel("Time Steps")
        plt.ylabel("Coverage (%)")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("flood_monitoring/evaluation/coverage_over_time.png", dpi=200)
        
        # Generate comprehensive metrics visualization
        plt.figure(figsize=(15, 12))
        
        # Coverage vs Distance scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(metrics['total_distance'], metrics['final_coverage'], alpha=0.7)
        plt.title("Coverage vs. Distance Traveled")
        plt.xlabel("Total Distance")
        plt.ylabel("Coverage (%)")
        plt.grid(True, alpha=0.3)
        
        # Efficiency histogram
        plt.subplot(2, 2, 2)
        plt.hist(metrics['overall_efficiency'], bins=10, alpha=0.7)
        plt.title("Coverage Efficiency Distribution")
        plt.xlabel("Efficiency (Coverage/Distance)")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        
        # Per-agent distance comparison (box plot)
        agent_distances = []
        agent_labels = []
        for i in range(config.NUM_AGENTS):
            agent_distances.append(metrics[f'agent_{i}_distance'])
            agent_labels.append(f"Agent {i}")
            
        plt.subplot(2, 2, 3)
        plt.boxplot(agent_distances, labels=agent_labels)
        plt.title("Distance Traveled by Agent")
        plt.ylabel("Distance")
        plt.grid(True, alpha=0.3)
        
        # Per-agent efficiency comparison (box plot)
        agent_efficiencies = []
        for i in range(config.NUM_AGENTS):
            agent_efficiencies.append(metrics[f'agent_{i}_efficiency'])
            
        plt.subplot(2, 2, 4)
        plt.boxplot(agent_efficiencies, labels=agent_labels)
        plt.title("Coverage Efficiency by Agent")
        plt.ylabel("Efficiency")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("flood_monitoring/evaluation/efficiency_analysis.png", dpi=200)
        
        # Generate example trajectory visualization for best efficiency
        best_episode = np.argmax(metrics['overall_efficiency'])
        plt.figure(figsize=(12, 10))
        
        # Plot flood intensity map
        plt.subplot(2, 1, 1)
        plt.title(f"Most Efficient Episode (#{best_episode+1}) - Flood Intensity and Agent Trajectories")
        plt.imshow(all_flood_maps[best_episode].T, origin='lower', cmap='Blues')
        plt.colorbar(label='Flood Intensity')
        
        # Plot agent trajectories
        colors = ['red', 'yellow', 'magenta', 'cyan']
        for i, trajectory in enumerate(all_trajectories[best_episode]):
            x_vals, y_vals = zip(*trajectory) if trajectory else ([], [])
            plt.plot(x_vals, y_vals, '-', color=colors[i % len(colors)], 
                    linewidth=2, label=f"Agent {i} - Distance: {all_distances_traveled[best_episode][i]:.2f}")
            plt.scatter(x_vals[-1], y_vals[-1], color=colors[i % len(colors)], 
                       marker='x', s=100)
        
        plt.legend()
        
        # Plot coverage map
        plt.subplot(2, 1, 2)
        plt.title(f"Final Coverage Map - {metrics['final_coverage'][best_episode]:.2%} Coverage | "
                 f"Efficiency: {metrics['overall_efficiency'][best_episode]:.4f}")
        plt.imshow(all_coverage_maps[best_episode].T, origin='lower', cmap='Greens')
        plt.colorbar(label='Visited')
        
        plt.tight_layout()
        plt.savefig("flood_monitoring/evaluation/best_efficiency_episode.png", dpi=200)
    
    return metrics_summary

if __name__ == "__main__":
    evaluate_matd3_flood()