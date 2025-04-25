import numpy as np
from scipy.ndimage import gaussian_filter

class FloodGrid:
    """Represents a 2D grid world with flood intensity values."""
    
    def __init__(self, grid_size=(20, 20), flood_pattern="random"):
        """
        Initialize a flood grid with specified size and pattern
        
        Args:
            grid_size: Tuple of (width, height)
            flood_pattern: Type of flood pattern to generate
                           "random": Fully randomized
                           "clustered": Clustered areas of flooding
                           "river": Simulates a river overflow
        """
        self.grid_size = grid_size
        self.flood_pattern = flood_pattern
        self.grid = np.zeros(grid_size)
        self.generate_flood()
        
    def generate_flood(self):
        """Generate flood intensity values based on the specified pattern"""
        if self.flood_pattern == "random":
            self.grid = np.random.uniform(0, 1, self.grid_size)
            
        elif self.flood_pattern == "clustered":
            # Create random flood centers
            num_clusters = np.random.randint(2, 5)
            self.grid = np.zeros(self.grid_size)
            
            for _ in range(num_clusters):
                center_x = np.random.randint(0, self.grid_size[0])
                center_y = np.random.randint(0, self.grid_size[1])
                intensity = np.random.uniform(0.7, 1.0)
                radius = np.random.randint(3, 7)
                
                # Create a cluster around center
                for x in range(max(0, center_x - radius), min(self.grid_size[0], center_x + radius)):
                    for y in range(max(0, center_y - radius), min(self.grid_size[1], center_y + radius)):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if dist < radius:
                            decay = 1 - (dist / radius)
                            self.grid[x, y] = max(self.grid[x, y], intensity * decay)
            
            # Add some background noise
            noise = np.random.uniform(0, 0.3, self.grid_size)
            self.grid = np.maximum(self.grid, noise)
            
        elif self.flood_pattern == "river":
            # Simulate a river overflowing
            self.grid = np.zeros(self.grid_size)
            
            # Create a river path
            river_start_x = np.random.randint(0, self.grid_size[0])
            river_width = np.random.randint(2, 4)
            
            # Generate winding river path
            x_pos = river_start_x
            for y in range(self.grid_size[1]):
                # Random walk for x position
                x_pos += np.random.randint(-1, 2)
                x_pos = max(river_width, min(self.grid_size[0] - river_width - 1, x_pos))
                
                # Draw river at current position
                for x in range(x_pos - river_width, x_pos + river_width + 1):
                    if 0 <= x < self.grid_size[0]:
                        # Higher intensity in the center of the river
                        dist_from_center = abs(x - x_pos)
                        intensity = 0.8 - 0.3 * (dist_from_center / river_width)
                        self.grid[x, y] = intensity
            
            # Create flood areas around river
            flood_grid = self.grid.copy()
            for _ in range(3):
                flood_center_y = np.random.randint(0, self.grid_size[1])
                flood_radius = np.random.randint(3, 6)
                
                for x in range(self.grid_size[0]):
                    for y in range(max(0, flood_center_y - flood_radius), 
                                   min(self.grid_size[1], flood_center_y + flood_radius)):
                        if self.grid[x, y] > 0.3:  # Near the river
                            dist = abs(y - flood_center_y)
                            if dist < flood_radius:
                                decay = 1 - (dist / flood_radius)
                                spread = max(1, int(decay * 5))
                                
                                # Spread outward from river
                                for dx in range(-spread, spread+1):
                                    nx = x + dx
                                    if 0 <= nx < self.grid_size[0]:
                                        intensity = 0.9 * decay
                                        flood_grid[nx, y] = max(flood_grid[nx, y], intensity)
            
            self.grid = flood_grid
        
        # Apply smoothing for more natural appearance
        self.grid = gaussian_filter(self.grid, sigma=1.0)
        
    def get_flood_value(self, x, y):
        """Get flood intensity at a specific cell"""
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            return self.grid[x, y]
        return 0.0
    
    def get_local_area(self, x, y, window_size=5):
        """
        Get a local window of flood values around position (x,y)
        
        Returns: 
            2D array of shape (window_size, window_size)
        """
        half_size = window_size // 2
        local_area = np.zeros((window_size, window_size))
        
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    local_area[dx + half_size, dy + half_size] = self.grid[nx, ny]
        
        return local_area
    
    def get_most_flooded_direction(self, x, y, window_size=5, visited_mask=None):
        """
        Find the direction of highest flooding from position (x,y)
        
        Args:
            x, y: Current position
            window_size: Size of observation window
            visited_mask: Binary mask of visited cells
            
        Returns:
            (dx, dy): Direction vector to most flooded unvisited cell
        """
        half_size = window_size // 2
        max_value = -float('inf')
        best_dir = (0, 0)  # Default: stay in place
        
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                nx, ny = x + dx, y + dy
                
                # Skip out-of-bounds cells
                if not (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]):
                    continue
                
                # Skip already visited cells if mask provided
                if visited_mask is not None and visited_mask[nx, ny]:
                    continue
                
                flood_value = self.grid[nx, ny]
                
                # Prioritize unvisited flooded cells
                if flood_value > max_value:
                    max_value = flood_value
                    best_dir = (dx, dy)
        
        return best_dir