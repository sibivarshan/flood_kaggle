import argparse
import os

from flood_monitoring.training_flood import train_matd3_flood
from flood_monitoring.evaluation_flood import evaluate_matd3_flood
from flood_monitoring.config_flood import FloodConfig

def main():
    """Main entry point for flood monitoring application"""
    
    parser = argparse.ArgumentParser(description='UAV Flood Monitoring with MATD3-GRU-Attention and APF')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'both'],
                        help='Mode to run: train, eval, or both')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes to train or evaluate')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    parser.add_argument('--model_dir', type=str, default='flood_monitoring/models',
                        help='Directory to load/save models')
    
    args = parser.parse_args()
    
    # Create needed directories
    os.makedirs("flood_monitoring/models", exist_ok=True)
    os.makedirs("flood_monitoring/results", exist_ok=True)
    os.makedirs("flood_monitoring/visualizations", exist_ok=True)
    os.makedirs("flood_monitoring/evaluation", exist_ok=True)
    
    # Run in specified mode
    if args.mode in ['train', 'both']:
        print("\n=== TRAINING MODE ===\n")
        max_episodes = args.episodes or FloodConfig.MAX_EPISODES
        FloodConfig.MAX_EPISODES = max_episodes
        trained_agents = train_matd3_flood()
    
    if args.mode in ['eval', 'both']:
        print("\n=== EVALUATION MODE ===\n")
        eval_episodes = args.episodes or 10
        evaluate_matd3_flood(
            model_dir=args.model_dir,
            model_prefix="agent",
            num_episodes=eval_episodes,
            render=args.render,
            save_metrics=True
        )
    
    print("\nDone!")

if __name__ == "__main__":
    main()