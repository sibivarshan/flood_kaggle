import numpy as np
from flood_monitoring.config_flood import FloodConfig

class FloodAPF:
    """
    Artificial Potential Field implementation for flood environment navigation.
    Combines attraction to high flood cells with repulsion from visited cells and other agents.
    """
    
    def __init__(self, config=None):
        self.config = config or FloodConfig
    
    def compute_force(self, position, flood_grid, global_visited_mask, agent_visited_mask, 
                      agent_positions, agent_idx):
        """
        Enhanced APF with coverage guarantee
        """
        x, y = position
        grid_size = flood_grid.shape
        
        # Initialize forces
        attraction_x, attraction_y = 0, 0
        repulsion_x, repulsion_y = 0, 0
        
        # 1. Attraction to high flood cells with diminishing returns for coverage
        unvisited_flood_mask = (~global_visited_mask) & (flood_grid > self.config.LOW_FLOOD_THRESHOLD)
        
        # Calculate distance field to nearest unvisited flooded cell
        if np.any(unvisited_flood_mask):
            # Find all unvisited flooded cells
            unvisited_coords = np.where(unvisited_flood_mask)
            
            # Find closest unvisited flooded cell
            min_dist = float('inf')
            closest_pos = None
            
            for i in range(len(unvisited_coords[0])):
                nx, ny = unvisited_coords[0][i], unvisited_coords[1][i]
                dist = ((nx - x)**2 + (ny - y)**2)**0.5
                
                # Weight by flood value
                weighted_dist = dist / max(0.3, flood_grid[nx, ny])
                
                if weighted_dist < min_dist:
                    min_dist = weighted_dist
                    closest_pos = (nx, ny)
            
            # Attraction toward closest unvisited flooded cell
            if closest_pos:
                # Direction vector to closest unvisited
                dx = closest_pos[0] - x
                dy = closest_pos[1] - y
                
                # Normalize direction vector
                dist = max(0.1, (dx**2 + dy**2)**0.5)
                attraction_x = dx / dist * self.config.APF_ATTRACTION_WEIGHT
                attraction_y = dy / dist * self.config.APF_ATTRACTION_WEIGHT
        
        # 2. Enhanced repulsion from visited cells
        for dx in range(-3, 4):  # Smaller radius for repulsion
            for dy in range(-3, 4):
                nx, ny = x + dx, y + dy
                
                # Skip out-of-bounds
                if not (0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]):
                    continue
                
                if agent_visited_mask[nx, ny]:
                    # Calculate repulsive force
                    dist = max(0.1, (dx*dx + dy*dy)**0.5)  # Avoid division by zero
                    repulsion_factor = self.config.APF_VISITED_REPULSION / dist**2
                    repulsion_x -= dx * repulsion_factor
                    repulsion_y -= dy * repulsion_factor
        
        # 3. Global coverage strategy: use a frontier-based approach
        # Add attraction to the frontier (boundary between visited and unvisited)
        frontier_attraction_x, frontier_attraction_y = self._compute_frontier_force(
            position, global_visited_mask, flood_grid
        )
        
        # Combine all forces
        total_x = attraction_x + repulsion_x + frontier_attraction_x * 0.5
        total_y = attraction_y + repulsion_y + frontier_attraction_y * 0.5
        
        # Normalize
        magnitude = (total_x**2 + total_y**2)**0.5
        if magnitude > 0:
            total_x /= magnitude
            total_y /= magnitude
        
        return np.array([total_x, total_y])
    
    def get_action_blend(self, apf_direction, policy_action):
        """
        Blend APF direction with policy action
        
        Args:
            apf_direction: Direction from APF (dx, dy) normalized
            policy_action: Action from policy network (8 directions + hover)
            
        Returns:
            Blended action as a (dx, dy) direction pair
        """
        # Get policy direction
        policy_direction = self._action_to_direction(policy_action)
        
        # Blend directions
        blend_factor = self.config.APF_BLEND_FACTOR
        blended_x = (1 - blend_factor) * policy_direction[0] + blend_factor * apf_direction[0]
        blended_y = (1 - blend_factor) * policy_direction[1] + blend_factor * apf_direction[1]
        
        # Convert to direction tuple
        return np.array([blended_x, blended_y])
    
    def _action_to_direction(self, action):
        """Convert discrete action to direction vector"""
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
        
        if isinstance(action, np.ndarray) and action.size > 1:
            return action  # If action is already a vector
        else:
            idx = min(int(action), len(directions) - 1)
            return directions[idx]
    
    def _direction_to_action(self, dx, dy):
        """Convert direction vector to closest discrete action"""
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
        
        # Check for hover first
        if abs(dx) < 0.3 and abs(dy) < 0.3:
            return 8  # Hover
        
        # Find closest direction
        best_match = 0
        best_dot = -float('inf')
        
        # Normalize input direction
        mag = max(0.001, (dx*dx + dy*dy)**0.5)
        dx_norm, dy_norm = dx/mag, dy/mag
        
        for i, (dir_x, dir_y) in enumerate(directions[:8]):  # Check only the 8 movement directions
            dot_product = dx_norm * dir_x + dy_norm * dir_y
            if dot_product > best_dot:
                best_dot = dot_product
                best_match = i
                
        return best_match
    
    def _compute_frontier_force(self, position, visited_mask, flood_grid):
        """Calculate attraction to frontier of unexplored area with efficiency consideration"""
        # Initialize frontier force
        frontier_x, frontier_y = 0, 0
        
        x, y = position
        grid_size = visited_mask.shape
        search_radius = 10  # Radius to search for frontier cells
        
        # Frontier cells are unvisited cells adjacent to visited cells
        frontier_cells = []
        
        # Search in a window around the current position
        x_min = max(0, int(x) - search_radius)
        y_min = max(0, int(y) - search_radius)
        x_max = min(grid_size[0], int(x) + search_radius + 1)
        y_max = min(grid_size[1], int(y) + search_radius + 1)
        
        # Find frontier cells (unvisited cells adjacent to visited cells)
        for nx in range(x_min, x_max):
            for ny in range(y_min, y_max):
                # Skip visited cells
                if visited_mask[nx, ny]:
                    continue
                
                # Check if this is a frontier cell (adjacent to a visited cell)
                is_frontier = False
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        adj_x, adj_y = nx + dx, ny + dy
                        if 0 <= adj_x < grid_size[0] and 0 <= adj_y < grid_size[1]:
                            if visited_mask[adj_x, adj_y]:
                                is_frontier = True
                                break
                    if is_frontier:
                        break
                
                # If it's a frontier cell, add to our list with flood value as weight
                if is_frontier:
                    # Enhanced weighting: balance flood value and distance
                    flood_value = max(0.1, flood_grid[nx, ny])
                    dist = max(0.1, ((nx - x)**2 + (ny - y)**2)**0.5)
                    
                    # Calculate efficiency score - prefer frontiers with:
                    # 1. Higher flood values (more important to cover)
                    # 2. Lower travel cost (closer or on the way to other frontiers)
                    efficiency_score = flood_value / (dist ** 0.8)  # Less than square to reduce distance penalty
                    
                    frontier_cells.append((nx, ny, flood_value, dist, efficiency_score))
        
        # If we found frontier cells, calculate attraction force
        if frontier_cells:
            # Find the frontier cell with highest weighted attraction
            best_score = -float('inf')
            best_cell = None
            
            for nx, ny, flood_value, dist, score in frontier_cells:
                if score > best_score:
                    best_score = score
                    best_cell = (nx, ny)
            
            # Calculate attraction to the best frontier cell
            if best_cell:
                nx, ny = best_cell
                dx = nx - x
                dy = ny - y
                
                # Normalize
                dist = max(0.1, (dx**2 + dy**2)**0.5)
                frontier_x = dx / dist * self.config.APF_FRONTIER_WEIGHT
                frontier_y = dy / dist * self.config.APF_FRONTIER_WEIGHT
                
                # Add a small bias to keep agents moving forward in their current direction
                # to avoid oscillation between equidistant frontiers
                momentum_x = frontier_x * 0.1
                momentum_y = frontier_y * 0.1
                
                frontier_x += momentum_x
                frontier_y += momentum_y
        
        return frontier_x, frontier_y