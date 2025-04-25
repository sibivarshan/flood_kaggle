import torch
import numpy as np

class FloodConfig:
    # Grid world parameters
    GRID_SIZE = (20, 20)  # 20x20 grid
    CELL_SIZE = 10  # Each cell is 10x10 meters
    
    # Environment
    NUM_AGENTS = 3
    MAX_STEPS = 2000  # Battery simulation
    STATE_DIM = 57  # Changed from 73 to 57 to match actual observation shape
    ACTION_DIM = 9  # 8 directions + hover
    AGENT_RADIUS = 0.5  # Radius for collision detection
    
    # Observation parameters
    OBS_WINDOW_SIZE = 5  # 5x5 local observation window
    
    # Camera parameters
    CAMERA_RANGE = 2  # How far the drone camera can see in each direction
    
    # Flood parameters
    HIGH_FLOOD_THRESHOLD = 0.7  # Threshold for high flood cells
    LOW_FLOOD_THRESHOLD = 0.3   # Threshold for low flood cells
    
    # Reward parameters
    HIGH_FLOOD_REWARD = 1.0
    LOW_FLOOD_REWARD = 0.1
    OVERLAP_PENALTY = -1.0
    COLLISION_PENALTY = -1.0
    REVISIT_PENALTY = -0.5
    IDLE_PENALTY = -0.2
    COVERAGE_REWARD_SCALE = 5.0  # Scale factor for coverage improvement rewards
    NEW_IMAGE_REWARD = 0.2  # Reward for capturing images of new cells
    
    # APF parameters
    APF_ATTRACTION_WEIGHT = 1.0
    APF_REPULSION_WEIGHT = 1.5
    APF_VISITED_REPULSION = 0.8
    APF_BLEND_FACTOR = 0.6  # How much APF influences actions vs. policy (0.0-1.0)
    APF_FRONTIER_WEIGHT = 0.8 
    
    # MATD3 parameters
    GAMMA = 0.99
    TAU = 0.005
    POLICY_NOISE = 0.1
    NOISE_CLIP = 0.2
    POLICY_FREQ = 2
    
    # GRU parameters
    HIDDEN_DIM = 128
    GRU_LAYERS = 2
    
    # Training
    BATCH_SIZE = 256
    BUFFER_SIZE = int(1e6)
    MAX_EPISODES = 1000
    EVAL_FREQ = 20
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Visualization
    VISUALIZATION_FREQ = 10