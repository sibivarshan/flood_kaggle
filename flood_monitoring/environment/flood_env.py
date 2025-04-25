import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

from flood_monitoring.environment.flood_grid import FloodGrid
from flood_monitoring.config_flood import FloodConfig
from flood_monitoring.models.apf_flood import FloodAPF

class FloodMonitoringEnv(gym.Env):
    """Environment for UAV flood monitoring with multi-agent RL"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, flood_pattern="clustered"):
        super(FloodMonitoringEnv, self).__init__()
        
        self.config = FloodConfig
        self.num_agents = self.config.NUM_AGENTS
        self.grid_size = self.config.GRID_SIZE
        self.max_steps = self.config.MAX_STEPS
        self.obs_window_size = self.config.OBS_WINDOW_SIZE
        
        # Environment components
        self.flood_grid = FloodGrid(grid_size=self.grid_size, flood_pattern=flood_pattern)
        self.apf = FloodAPF()
        
        # Agent state tracking
        self.agent_positions = np.zeros((self.num_agents, 2), dtype=int)  # [x, y] grid coordinates
        self.global_visited_mask = np.zeros(self.grid_size, dtype=bool)   # Global visited mask for all agents
        self.agent_visited_masks = [np.zeros(self.grid_size, dtype=bool) for _ in range(self.num_agents)]  # Per-agent visited masks
        self.current_step = 0
        self.coverage = 0.0  # Percentage of flooded area covered
        self.high_flood_cells_total = 0  # Count of high flood cells in the environment
        
        # Track total distance traveled by each drone
        self.distances_traveled = np.zeros(self.num_agents)
        # Track coverage efficiency (coverage percentage / distance traveled)
        self.coverage_efficiency = np.zeros(self.num_agents)
        # Track coverage rate over time
        self.coverage_history = []
        # Track visited flood cells (low and high)
        self.visited_low_flood_cells = 0
        self.visited_high_flood_cells = 0
        self.total_low_flood_cells = 0
        self.total_high_flood_cells = 0
        
        # Image data collection (placeholder for synthetic images)
        self.captured_images = [[] for _ in range(self.num_agents)]
        
        # Observation history for GRU
        self.observation_history = [deque(maxlen=5) for _ in range(self.num_agents)]
        
        # Change from discrete to continuous action space
        # Actions: [delta_x, delta_y] in range [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation space: local flood map + position + other agents' positions + visited mask
        # Flattened 5x5 flood map (25) + position (2) + other agents (2*2) + visited mask (25) + distance to closest unvisited high flood (1)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.config.STATE_DIM,), 
            dtype=np.float32
        )
        
        # Rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Ensure output directories exist
        os.makedirs("flood_monitoring/results", exist_ok=True)
        os.makedirs("flood_monitoring/visualizations", exist_ok=True)
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Generate new flood grid
        self.flood_grid.generate_flood()
        
        # Reset tracking variables
        self.current_step = 0
        self.global_visited_mask = np.zeros(self.grid_size, dtype=bool)
        self.agent_visited_masks = [np.zeros(self.grid_size, dtype=bool) for _ in range(self.num_agents)]
        self.captured_images = [[] for _ in range(self.num_agents)]
        self.observation_history = [deque(maxlen=5) for _ in range(self.num_agents)]
        
        # Reset distance and coverage tracking
        self.distances_traveled = np.zeros(self.num_agents)
        self.coverage_efficiency = np.zeros(self.num_agents)
        self.coverage_history = []
        
        # Count flood cells of different intensities
        self.total_low_flood_cells = np.sum((self.flood_grid.grid >= self.config.LOW_FLOOD_THRESHOLD) & 
                                           (self.flood_grid.grid < self.config.HIGH_FLOOD_THRESHOLD))
        self.total_high_flood_cells = np.sum(self.flood_grid.grid >= self.config.HIGH_FLOOD_THRESHOLD)
        self.visited_low_flood_cells = 0
        self.visited_high_flood_cells = 0
        
        # Place agents at random starting positions (corners/edges are good starting points)
        start_positions = []
        
        # Edge positions for starting
        edge_positions = []
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if x < 2 or y < 2 or x >= self.grid_size[0] - 2 or y >= self.grid_size[1] - 2:
                    edge_positions.append((x, y))
        
        # Randomly select positions
        np.random.shuffle(edge_positions)
        
        for i in range(self.num_agents):
            if i < len(edge_positions):
                self.agent_positions[i] = edge_positions[i]
            else:
                # Fallback to random positions
                x = np.random.randint(0, self.grid_size[0])
                y = np.random.randint(0, self.grid_size[1])
                self.agent_positions[i] = (x, y)
            
            # Mark initial position as visited
            x, y = self.agent_positions[i]
            self.global_visited_mask[x, y] = True
            self.agent_visited_masks[i][x, y] = True
            
            # Capture image of initial cell
            self._capture_image(i)
        
        # Count high flood cells
        self.high_flood_cells_total = np.sum(self.flood_grid.grid >= self.config.HIGH_FLOOD_THRESHOLD)
        
        # Get initial observations
        observations = [self._get_observation(i) for i in range(self.num_agents)]
        
        # Update observation history
        for i in range(self.num_agents):
            self.observation_history[i].append(observations[i])
            # Fill history with initial observation if empty
            while len(self.observation_history[i]) < 5:
                self.observation_history[i].appendleft(observations[i])
        
        if self.render_mode == "human":
            self.render()
            
        return observations, {}
    
    def step(self, actions):
        """
        Take a step in the environment with actions for each agent
        
        Args:
            actions: List of arrays representing each agent's action [dx, dy]
        
        Returns:
            observations, rewards, dones, truncated, info
        """
        self.current_step += 1
        rewards = np.zeros(self.num_agents)
        dones = np.zeros(self.num_agents, dtype=bool)
        truncated = np.zeros(self.num_agents, dtype=bool)
        infos = [{} for _ in range(self.num_agents)]
        
        # Process each agent's action
        for i in range(self.num_agents):
            # Get current position
            x, y = self.agent_positions[i]
            
            # Process continuous action
            dx, dy = actions[i]
            
            # Scale the movement (can be tuned)
            dx = np.clip(dx, -1.0, 1.0)
            dy = np.clip(dy, -1.0, 1.0)
            
            # Calculate new position with continuous movement
            new_x = x + dx
            new_y = y + dy
            
            # Calculate distance traveled in this step
            distance_moved = np.sqrt(dx**2 + dy**2)
            self.distances_traveled[i] += distance_moved
            
            # Discretize to grid cell for environment interactions
            grid_x = int(np.clip(new_x, 0, self.grid_size[0] - 1))
            grid_y = int(np.clip(new_y, 0, self.grid_size[1] - 1))
            
            # Check for collisions with other agents
            collision = False
            for j in range(self.num_agents):
                if i == j:
                    continue
                    
                other_x, other_y = self.agent_positions[j]
                distance = ((new_x - other_x)**2 + (new_y - other_y)**2)**0.5
                
                if distance < self.config.AGENT_RADIUS * 2:  # Assuming agents have a radius
                    collision = True
                    rewards[i] += self.config.COLLISION_PENALTY
                    infos[i]['collision'] = True
                    break
            
            if not collision:
                # Store continuous position internally
                self.agent_positions[i] = (new_x, new_y)
                # Use grid position for cell visitation
                already_visited_by_self = self.agent_visited_masks[i][grid_x, grid_y]
                
                # Main cell reward calculation
                cell_flood_value = self.flood_grid.grid[grid_x, grid_y]
                
                # Revisiting own cells is wasteful
                if already_visited_by_self:
                    rewards[i] += self.config.REVISIT_PENALTY
                    infos[i]['revisited'] = True
                else:
                    # New primary cell visit
                    if cell_flood_value >= self.config.HIGH_FLOOD_THRESHOLD:
                        rewards[i] += self.config.HIGH_FLOOD_REWARD
                        infos[i]['high_flood'] = True
                    elif cell_flood_value >= self.config.LOW_FLOOD_THRESHOLD:
                        rewards[i] += self.config.LOW_FLOOD_REWARD
                        infos[i]['low_flood'] = True
                
                # Capture images in all directions and get newly observed cells count
                newly_observed_cells = self._capture_image(i)
                
                # Add reward for new observations from all directions
                image_reward = newly_observed_cells * self.config.NEW_IMAGE_REWARD
                rewards[i] += image_reward
                infos[i]['newly_observed'] = newly_observed_cells
                infos[i]['image_reward'] = image_reward
            
            # Idle penalty if the agent didn't move (dx=0, dy=0)
            if dx == 0 and dy == 0 and not collision:
                rewards[i] += self.config.IDLE_PENALTY
                infos[i]['idle'] = True
        
        # Update coverage percentage
        visited_flood_cells = np.sum(self.global_visited_mask & (self.flood_grid.grid > self.config.LOW_FLOOD_THRESHOLD))
        total_flood_cells = np.sum(self.flood_grid.grid > self.config.LOW_FLOOD_THRESHOLD)
        old_coverage = self.coverage
        self.coverage = visited_flood_cells / max(1, total_flood_cells)
        
        # Store coverage history for analysis
        self.coverage_history.append((self.current_step, self.coverage))
        
        # Calculate coverage efficiency for each agent
        for i in range(self.num_agents):
            if self.distances_traveled[i] > 0:
                # Coverage per unit distance traveled
                agent_visited_count = np.sum(self.agent_visited_masks[i] & (self.flood_grid.grid > self.config.LOW_FLOOD_THRESHOLD))
                self.coverage_efficiency[i] = agent_visited_count / self.distances_traveled[i]
        
        # Calculate coverage-focused reward
        for i in range(self.num_agents):
            rewards[i] += self._calculate_coverage_reward(i, old_coverage, self.coverage)
        
        # Check if all high flood cells have been visited
        high_flood_mask = self.flood_grid.grid >= self.config.HIGH_FLOOD_THRESHOLD
        high_flood_visited = np.sum(self.global_visited_mask & high_flood_mask)
        high_flood_total = np.sum(high_flood_mask)
        high_flood_coverage = high_flood_visited / max(1, high_flood_total)
        
        # Check for episode termination
        # 1. Maximum steps reached
        if self.current_step >= self.max_steps:
            truncated = np.ones(self.num_agents, dtype=bool)
            
        # 2. All high flood cells visited (mission complete)
        if high_flood_coverage >= 0.95:  # Allow for some small margin
            dones = np.ones(self.num_agents, dtype=bool)
            # Bonus reward for mission completion
            rewards += 5.0
        
        # Get new observations
        observations = [self._get_observation(i) for i in range(self.num_agents)]
        
        # Update observation history
        for i in range(self.num_agents):
            self.observation_history[i].append(observations[i])
        
        # Add coverage and distance info to info dict
        for i in range(self.num_agents):
            infos[i]['coverage'] = self.coverage
            infos[i]['high_flood_coverage'] = high_flood_coverage
            infos[i]['distance_traveled'] = self.distances_traveled[i]
            infos[i]['coverage_efficiency'] = self.coverage_efficiency[i]
            infos[i]['step'] = self.current_step
        
        if self.render_mode == "human":
            self.render()
        
        return observations, rewards, dones, truncated, infos
    
    def _calculate_coverage_reward(self, agent_idx, old_coverage, new_coverage):
        """Calculate reward based on improvement in coverage and efficiency"""
        coverage_change = new_coverage - old_coverage
        
        # If coverage improved
        if coverage_change > 0:
            # Base reward for coverage improvement
            base_reward = self.config.COVERAGE_REWARD_SCALE * (1.0 / (1.0 - new_coverage + 0.01) - 1.0) * coverage_change
            
            # Efficiency bonus - reward agents who cover more with less distance
            distance_moved = self.distances_traveled[agent_idx]
            if distance_moved > 0:
                # Higher reward for agents who covered more ground per unit distance
                agent_visited_count = np.sum(self.agent_visited_masks[agent_idx] & 
                                           (self.flood_grid.grid > self.config.LOW_FLOOD_THRESHOLD))
                efficiency = agent_visited_count / distance_moved
                
                # Scale efficiency bonus based on current coverage
                # Higher bonus at higher coverage levels (when finding new areas gets harder)
                efficiency_bonus = efficiency * (0.5 + new_coverage)
                
                return base_reward + efficiency_bonus * 0.2  # Scale factor for efficiency bonus
            
            return base_reward
        
        return 0.0

    def _get_observation(self, agent_idx):
        """Generate observation for a specific agent"""
        x, y = self.agent_positions[agent_idx]
        
        # 1. Local flood map (5x5 window)
        local_flood = self.flood_grid.get_local_area(x, y, self.obs_window_size)
        
        # 2. Local visited mask
        local_visited = np.zeros((self.obs_window_size, self.obs_window_size))
        half_size = self.obs_window_size // 2
        
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    local_visited[dx + half_size, dy + half_size] = self.global_visited_mask[nx, ny]
        
        # 3. Agent position (normalized)
        position = np.array([x / self.grid_size[0], y / self.grid_size[1]])
        
        # 4. Other agents' positions (relative)
        other_positions = []
        for i in range(self.num_agents):
            if i != agent_idx:
                ox, oy = self.agent_positions[i]
                # Relative positions normalized to [-1, 1]
                rel_x = (ox - x) / self.grid_size[0]
                rel_y = (oy - y) / self.grid_size[1]
                other_positions.extend([rel_x, rel_y])
        
        # 5. Distance to closest unvisited high flood cell
        distance_to_high_flood = self._get_distance_to_closest_high_flood(x, y)
        
        # Concatenate all components
        observation = np.concatenate([
            local_flood.flatten(),                   # 25 values
            position,                                # 2 values
            np.array(other_positions),              # 2*(num_agents-1) values
            local_visited.flatten(),                # 25 values
            np.array([distance_to_high_flood])      # 1 value
        ])
        
        return observation
    
    def _get_distance_to_closest_high_flood(self, x, y):
        """Calculate normalized distance to closest unvisited high flood cell"""
        min_dist = float('inf')
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                # Skip if not high flood or already visited
                if self.flood_grid.grid[i, j] < self.config.HIGH_FLOOD_THRESHOLD or self.global_visited_mask[i, j]:
                    continue
                
                # Calculate Manhattan distance
                dist = abs(x - i) + abs(y - j)
                min_dist = min(min_dist, dist)
        
        if min_dist == float('inf'):
            return 1.0  # No unvisited high flood cells
            
        # Normalize by maximum possible distance
        max_dist = self.grid_size[0] + self.grid_size[1]
        return min_dist / max_dist
    
    def _capture_image(self, agent_idx):
        """Simulate capturing an image at the agent's position with pictures in all directions"""
        x, y = self.agent_positions[agent_idx]
        
        # Get all cells in the camera view range (like a drone taking panoramic pictures)
        camera_range = self.config.CAMERA_RANGE
        
        # Store the primary cell information
        cell_flood_value = self.flood_grid.grid[x, y]
        
        # Add the primary position image data
        primary_image_data = {
            'position': (x, y),
            'flood_value': cell_flood_value,
            'timestamp': self.current_step,
            'is_primary': True
        }
        self.captured_images[agent_idx].append(primary_image_data)
        
        # For each direction, capture an image and add to coverage
        direction_images = []
        newly_observed_cells = 0
        
        for dx in range(-camera_range, camera_range + 1):
            for dy in range(-camera_range, camera_range + 1):
                nx, ny = x + dx, y + dy
                # Skip if out of bounds or it's the center cell (already processed)
                if (dx == 0 and dy == 0) or not (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]):
                    continue
                
                # Calculate the distance from the drone
                distance = np.sqrt(dx**2 + dy**2)
                
                # Skip if beyond camera range
                if distance > camera_range:
                    continue
                    
                # Check if this is a new coverage
                cell_already_visited = self.global_visited_mask[nx, ny]
                
                # Mark as observed/visited
                self.global_visited_mask[nx, ny] = True
                self.agent_visited_masks[agent_idx][nx, ny] = True
                
                # Update visited flood cell counts if newly observed
                if not cell_already_visited:
                    newly_observed_cells += 1
                    observed_flood_value = self.flood_grid.grid[nx, ny]
                    if observed_flood_value >= self.config.HIGH_FLOOD_THRESHOLD:
                        self.visited_high_flood_cells += 1
                    elif observed_flood_value >= self.config.LOW_FLOOD_THRESHOLD:
                        self.visited_low_flood_cells += 1
                
                # Store data about this observed cell
                direction_image_data = {
                    'position': (nx, ny),
                    'flood_value': self.flood_grid.grid[nx, ny],
                    'timestamp': self.current_step,
                    'is_primary': False,
                    'distance_from_drone': distance
                }
                direction_images.append(direction_image_data)
        
        # Add all direction images to the captured images list
        self.captured_images[agent_idx].extend(direction_images)
        
        # Return number of newly observed cells for reward calculation
        return newly_observed_cells
    
    def _action_to_direction(self, action):
        """Convert action to direction"""
        # 8 directions + hover
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
        
        if isinstance(action, np.ndarray):
            # If continuous action, convert to discrete by finding nearest direction
            # Assuming action is normalized to [-1, 1]^2
            ax, ay = action[0], action[1]
            
            # Find the closest direction
            closest_action = 0
            min_distance = float('inf')
            
            for i, (dx, dy) in enumerate(directions[:-1]):  # Exclude hover
                distance = (ax - dx)**2 + (ay - dy)**2
                if distance < min_distance:
                    min_distance = distance
                    closest_action = i
            
            # Check if hover is the intended action
            hover_threshold = 0.3
            if abs(ax) < hover_threshold and abs(ay) < hover_threshold:
                closest_action = 8  # Hover
                
            return directions[closest_action]
        else:
            # Discrete action
            return directions[min(action, len(directions) - 1)]
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        try:
            import pygame
            
            # Initialize pygame if needed
            if self.window is None:
                pygame.init()
                pygame.display.set_caption("UAV Flood Monitoring")
                self.window = pygame.display.set_mode((800, 800))
                
            if self.clock is None:
                self.clock = pygame.time.Clock()
                
            # Clear the screen
            self.window.fill((0, 0, 0))
            
            # Calculate cell size
            cell_width = 800 // self.grid_size[0]
            cell_height = 800 // self.grid_size[1]
            
            # Draw the flood grid
            for x in range(self.grid_size[0]):
                for y in range(self.grid_size[1]):
                    flood_value = self.flood_grid.grid[x, y]
                    
                    # Color based on flood value
                    if flood_value >= self.config.HIGH_FLOOD_THRESHOLD:
                        color = (0, 0, int(255 * min(1, flood_value * 1.5)))  # Deep blue for high flood
                    elif flood_value >= self.config.LOW_FLOOD_THRESHOLD:
                        color = (0, 0, int(200 * flood_value))  # Light blue for low flood
                    else:
                        color = (100, 220, 100)  # Green for dry land
                    
                    # Mark visited cells
                    if self.global_visited_mask[x, y]:
                        # Add a visited overlay
                        color = (color[0] // 2 + 120, color[1] // 2 + 120, color[2] // 2)
                    
                    pygame.draw.rect(
                        self.window,
                        color,
                        pygame.Rect(
                            x * cell_width,
                            y * cell_height,
                            cell_width,
                            cell_height
                        )
                    )
                    
                    # Draw grid lines
                    pygame.draw.rect(
                        self.window,
                        (50, 50, 50),
                        pygame.Rect(
                            x * cell_width,
                            y * cell_height,
                            cell_width,
                            cell_height
                        ),
                        1
                    )
            
            # Draw agents and their camera view areas
            for i in range(self.num_agents):
                x, y = self.agent_positions[i]
                center_x = x * cell_width + cell_width // 2
                center_y = y * cell_height + cell_height // 2
                
                # Agent color based on index
                colors = [(255, 0, 0), (255, 255, 0), (255, 0, 255)]
                color = colors[i % len(colors)]
                
                # Draw camera viewing range
                camera_range = self.config.CAMERA_RANGE
                camera_radius_px = camera_range * (cell_width + cell_height) // 2
                camera_surface = pygame.Surface((camera_radius_px * 2, camera_radius_px * 2), pygame.SRCALPHA)
                camera_color = (*color, 40)  # semi-transparent color
                pygame.draw.circle(camera_surface, camera_color, (camera_radius_px, camera_radius_px), camera_radius_px)
                self.window.blit(camera_surface, (center_x - camera_radius_px, center_y - camera_radius_px))
                
                # Draw agent
                pygame.draw.circle(
                    self.window,
                    color,
                    (center_x, center_y),
                    min(cell_width, cell_height) // 2 - 2
                )
                
                # Draw agent index
                font = pygame.font.SysFont(None, 24)
                text = font.render(str(i), True, (0, 0, 0))
                text_rect = text.get_rect(center=(center_x, center_y))
                self.window.blit(text, text_rect)
            
            # Display coverage info
            font = pygame.font.SysFont(None, 30)
            coverage_text = font.render(f"Coverage: {self.coverage:.2%}", True, (255, 255, 255))
            self.window.blit(coverage_text, (10, 10))
            
            step_text = font.render(f"Step: {self.current_step}/{self.max_steps}", True, (255, 255, 255))
            self.window.blit(step_text, (10, 40))
            
            # Display camera coverage info
            camera_text = font.render(f"Camera Range: {self.config.CAMERA_RANGE}", True, (255, 255, 255))
            self.window.blit(camera_text, (10, 70))
            
            # Update the display
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
            if self.render_mode == "rgb_array":
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
                )
        
        except ImportError:
            # If pygame is not available, fall back to matplotlib
            self._render_with_matplotlib()
    
    def _render_with_matplotlib(self):
        """Render using matplotlib if pygame is not available"""
        plt.figure(figsize=(10, 10))
        
        # Plot flood grid
        plt.imshow(self.flood_grid.grid.T, origin='lower', cmap='Blues')
        
        # Mark visited cells
        visited_mask = np.zeros(self.grid_size)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if self.global_visited_mask[x, y]:
                    visited_mask[x, y] = 0.5
        
        plt.imshow(visited_mask.T, origin='lower', cmap='Greens', alpha=0.3)
        
        # Plot agents
        for i in range(self.num_agents):
            x, y = self.agent_positions[i]
            colors = ['red', 'yellow', 'magenta', 'cyan']
            plt.scatter(x, y, color=colors[i % len(colors)], s=100, zorder=10)
            plt.text(x, y, str(i), ha='center', va='center', fontsize=12)
        
        plt.title(f"UAV Flood Monitoring - Step {self.current_step}/{self.max_steps}\nCoverage: {self.coverage:.2%}")
        plt.grid(True)
        plt.draw()
        plt.pause(0.1)
        plt.clf()
    
    def close(self):
        """Close environment and pygame"""
        if hasattr(self, "window") and self.window is not None:
            import pygame
            pygame.quit()
            self.window = None
            self.clock = None
    
    def save_data(self, episode_num):
        """Save monitoring data and visualizations"""
        # Save coverage map
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot flood grid
        im1 = axes[0, 0].imshow(self.flood_grid.grid.T, origin='lower', cmap='Blues')
        axes[0, 0].set_title("Flood Intensity Map")
        axes[0, 0].set_xlabel("X")
        axes[0, 0].set_ylabel("Y")
        plt.colorbar(im1, ax=axes[0, 0], label="Flood Intensity")
        
        # Plot visited cells/coverage
        im2 = axes[0, 1].imshow(self.global_visited_mask.T, origin='lower', cmap='Greens')
        axes[0, 1].set_title(f"Coverage Map (Final: {self.coverage:.2%})")
        axes[0, 1].set_xlabel("X")
        axes[0, 1].set_ylabel("Y")
        plt.colorbar(im2, ax=axes[0, 1], label="Visited")
        
        # Plot agent trajectories on both top visualizations
        for i in range(self.num_agents):
            positions = [(image['position'][0], image['position'][1]) for image in self.captured_images[i]]
            x_vals, y_vals = zip(*positions) if positions else ([], [])
            colors = ['red', 'yellow', 'magenta', 'cyan']
            
            for ax in [axes[0, 0], axes[0, 1]]:
                ax.plot(x_vals, y_vals, '-', color=colors[i % len(colors)], linewidth=1.5, label=f"Agent {i}")
                ax.scatter(x_vals[-1], y_vals[-1], color=colors[i % len(colors)], marker='x', s=100)
        
        axes[0, 1].legend()
        
        # Plot coverage over time
        if self.coverage_history:
            steps, coverages = zip(*self.coverage_history)
            axes[1, 0].plot(steps, coverages, 'b-', linewidth=2)
            axes[1, 0].set_title("Coverage Progress Over Time")
            axes[1, 0].set_xlabel("Steps")
            axes[1, 0].set_ylabel("Coverage (%)")
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].grid(True, alpha=0.3)
            
        # Create efficiency metrics table
        total_distance = np.sum(self.distances_traveled)
        total_visited = np.sum(self.global_visited_mask)
        overall_efficiency = self.coverage / max(0.1, total_distance)
        
        table_data = []
        table_data.append(["Agent", "Distance", "Coverage", "Efficiency"])
        
        for i in range(self.num_agents):
            agent_visited = np.sum(self.agent_visited_masks[i])
            table_data.append([
                f"Agent {i}", 
                f"{self.distances_traveled[i]:.2f}", 
                f"{agent_visited}/{self.grid_size[0]*self.grid_size[1]}", 
                f"{self.coverage_efficiency[i]:.4f}"
            ])
        
        table_data.append(["Total", f"{total_distance:.2f}", f"{self.coverage:.2%}", f"{overall_efficiency:.4f}"])
        table_data.append(["High Flood", "---", f"{self.visited_high_flood_cells}/{self.total_high_flood_cells}", 
                          f"{self.visited_high_flood_cells/max(1, self.total_high_flood_cells):.2%}"])
        
        # Remove the last plot area and replace with metrics text
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)
        axes[1, 1].set_title("Coverage and Efficiency Metrics")
        
        plt.tight_layout()
        plt.savefig(f"flood_monitoring/visualizations/episode_{episode_num}_coverage.png", dpi=200)
        plt.close()
        
        # Save trajectory plot separately with distance information
        plt.figure(figsize=(12, 10))
        plt.imshow(self.flood_grid.grid.T, origin='lower', cmap='Blues', alpha=0.6)
        
        for i in range(self.num_agents):
            positions = [(image['position'][0], image['position'][1]) for image in self.captured_images[i]]
            x_vals, y_vals = zip(*positions) if positions else ([], [])
            colors = ['red', 'yellow', 'magenta', 'cyan']
            
            plt.plot(x_vals, y_vals, '-', color=colors[i % len(colors)], linewidth=2, 
                    label=f"Agent {i} - Distance: {self.distances_traveled[i]:.2f}")
            plt.scatter(x_vals[0], y_vals[0], color=colors[i % len(colors)], marker='o', s=100)  # Start
            plt.scatter(x_vals[-1], y_vals[-1], color=colors[i % len(colors)], marker='x', s=100)  # End
            
        plt.colorbar(label="Flood Intensity")
        plt.title(f"Agent Trajectories - Total Distance: {total_distance:.2f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"flood_monitoring/visualizations/episode_{episode_num}_trajectories.png", dpi=200)
        plt.close()
        
        # Save collected data
        coverage_data = {
            'flood_grid': self.flood_grid.grid,
            'visited_mask': self.global_visited_mask,
            'coverage': self.coverage,
            'agent_trajectories': self.captured_images,
            'steps': self.current_step,
            'distances_traveled': self.distances_traveled,
            'coverage_efficiency': self.coverage_efficiency,
            'coverage_history': self.coverage_history,
            'visited_high_flood_cells': self.visited_high_flood_cells,
            'total_high_flood_cells': self.total_high_flood_cells,
            'visited_low_flood_cells': self.visited_low_flood_cells,
            'total_low_flood_cells': self.total_low_flood_cells
        }
        
        np.save(f"flood_monitoring/results/episode_{episode_num}_data.npy", coverage_data)
        
        return coverage_data