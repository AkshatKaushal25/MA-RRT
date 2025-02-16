import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import random

class PathPlanningEnv:
    def __init__(self, width=10, height=10, num_obstacles=200):  # Increased default obstacles
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        
        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.setup_environment()
        
    def setup_environment(self):
        """Initialize the environment with many small obstacles, start, and goal positions"""
        self.obstacles = []
        self.start = None
        self.goal = None
        
        # Set up the plot
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # Create a grid-based distribution for more even obstacle placement
        grid_size = min(self.width, self.height) / np.sqrt(self.num_obstacles)
        for i in range(self.num_obstacles):
            # Try to place each obstacle multiple times
            for _ in range(10):
                # Add some randomness to grid positions
                grid_x = random.randint(0, int(self.width/grid_size)-1)
                grid_y = random.randint(0, int(self.height/grid_size)-1)
                
                # Add small random offset
                x = (grid_x + random.uniform(0.2, 0.8)) * grid_size
                y = (grid_y + random.uniform(0.2, 0.8)) * grid_size
                
                # Smaller radius with less variation
                radius = random.uniform(0.1, 0.2)  # Reduced radius range
                
                # Check if obstacle overlaps with existing obstacles
                valid = True
                for obs in self.obstacles:
                    if np.sqrt((x - obs[0])**2 + (y - obs[1])**2) < (radius + obs[2] + 0.05):
                        valid = False
                        break
                
                if valid:
                    self.obstacles.append((x, y, radius))
                    break
        
        print(f"Created {len(self.obstacles)} obstacles out of {self.num_obstacles} requested")

        # Generate start and goal positions with better clearance
        def is_position_valid(x, y, min_obstacle_dist=0.3):
            if x < 0.5 or x > self.width-0.5 or y < 0.5 or y > self.height-0.5:
                return False
            for obs in self.obstacles:
                if np.sqrt((x - obs[0])**2 + (y - obs[1])**2) < (obs[2] + min_obstacle_dist):
                    return False
            return True

        # Keep trying until we find valid start and goal positions
        max_attempts = 1000
        attempts = 0
        while attempts < max_attempts:
            start_x = random.uniform(0.5, self.width-0.5)
            start_y = random.uniform(0.5, self.height-0.5)
            goal_x = random.uniform(0.5, self.width-0.5)
            goal_y = random.uniform(0.5, self.height-0.5)
            
            if (is_position_valid(start_x, start_y) and 
                is_position_valid(goal_x, goal_y) and
                np.sqrt((start_x - goal_x)**2 + (start_y - goal_y)**2) > self.width/3):  # Ensure minimum separation
                self.start = (start_x, start_y)
                self.goal = (goal_x, goal_y)
                break
            attempts += 1
        
        if attempts == max_attempts:
            raise Exception("Could not find valid start and goal positions")
    
    def draw_environment(self):
        """Draw the current state of the environment"""
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.grid(True)
        
        # Draw obstacles
        for obs in self.obstacles:
            circle = Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.3)  # More transparent
            self.ax.add_patch(circle)
        
        # Draw start and goal
        if self.start:
            self.ax.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        if self.goal:
            self.ax.plot(self.goal[0], self.goal[1], 'bo', markersize=10, label='Goal')
        
        self.ax.legend()
        plt.draw()
    
    def draw_path(self, path):
        """Draw a path on the environment"""
        if path:
            path = np.array(path)
            self.ax.plot(path[:, 0], path[:, 1], 'g--', linewidth=2, label='Path')
            self.ax.legend()
            plt.draw()
    
    def is_collision(self, point):
        """Check if a point collides with any obstacle"""
        for obs in self.obstacles:
            if np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2) < (obs[2] + 0.05):  # Added small safety margin
                return True
        return False
    
    def get_state(self):
        """Return the current state of the environment"""
        return {
            'start': self.start,
            'goal': self.goal,
            'obstacles': self.obstacles,
            'dimensions': (self.width, self.height)
        }

class Node:
    def __init__(self, pos, parent=None):
        self.pos = np.array(pos)
        self.parent = parent
        self.cost = 0 if parent is None else parent.cost + np.linalg.norm(pos - parent.pos)
        self.children = []
        self.e_in = None
        self.h_cost = 0  

class Edge:
    def __init__(self, start_node, end_node):
        self.st = start_node
        self.en = end_node
        self.len = np.linalg.norm(start_node.pos - end_node.pos)
        self.cost = self.len
        self.vec = self.en.pos - self.st.pos

class RRTStarPlanner:
    def __init__(self, env, step_size=0.5, neighbor_radius=1.0, max_iterations=1000, 
                 goal_bias=0.2, heuristic_bias=0.3):
        self.env = env
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias  # Probability of sampling goal directly
        self.heuristic_bias = heuristic_bias  # Influence of A* heuristic
        self.nodes = []
        self.edges = []
        self.goal_nodes = []
        self.best_path_cost = float('inf')
        self.best_goal_node = None
        
    def _calculate_heuristic(self, pos):
        """Calculate heuristic cost (Euclidean distance to goal)"""
        return np.linalg.norm(pos - self.env.goal)
    
    def _sample_with_heuristic(self):
        """Sample point with bias towards goal and unexplored areas with low heuristic cost"""
        rand_val = random.random()
        
        if rand_val < self.goal_bias:
            # Direct goal sampling
            return np.array(self.env.goal)
        elif rand_val < (self.goal_bias + self.heuristic_bias):
            # Heuristic-biased sampling
            # Sample multiple points and choose the one with best heuristic
            num_samples = 10
            samples = []
            for _ in range(num_samples):
                sample = np.array([
                    random.uniform(0, self.env.width),
                    random.uniform(0, self.env.height)
                ])
                if not self.env.is_collision(sample):
                    samples.append(sample)
            
            if not samples:
                return None
                
            # Choose point with best combination of heuristic cost and distance to nearest node
            best_score = float('inf')
            best_sample = samples[0]
            
            for sample in samples:
                h_cost = self._calculate_heuristic(sample)
                # Find distance to nearest node to encourage exploration
                nearest_dist = min(np.linalg.norm(node.pos - sample) for node in self.nodes)
                # Combine heuristic and exploration factors
                score = h_cost - 0.5 * nearest_dist  # Balance between goal-seeking and exploration
                if score < best_score:
                    best_score = score
                    best_sample = sample
            return best_sample
        else:
            # Regular random sampling
            while True:
                sample = np.array([
                    random.uniform(0, self.env.width),
                    random.uniform(0, self.env.height)
                ])
                if not self.env.is_collision(sample):
                    return sample
        return None

    def plan(self):
        # Initialize root node at start position
        root = Node(self.env.start)
        root.h_cost = self._calculate_heuristic(root.pos)
        self.nodes.append(root)
        
        for iteration in range(self.max_iterations):
            # Sample point using heuristic-based strategy
            x_rand = self._sample_with_heuristic()
            if x_rand is None:
                continue
            
            # Find nearest node using combined cost (actual + heuristic)
            nearest_node = min(self.nodes, 
                             key=lambda n: np.linalg.norm(n.pos - x_rand) + 
                                         self.heuristic_bias * n.h_cost)
            
            # Steer towards random point
            direction = x_rand - nearest_node.pos
            distance = np.linalg.norm(direction)
            if distance > self.step_size:
                direction = direction / distance * self.step_size
            new_pos = nearest_node.pos + direction
            
            if self.env.is_collision(new_pos):
                continue
                
            # Find nearby nodes
            nearby_nodes = [n for n in self.nodes 
                          if np.linalg.norm(n.pos - new_pos) < self.neighbor_radius]
            
            # Find best parent considering both actual cost and heuristic
            min_total_cost = float('inf')
            best_parent = None
            
            for node in nearby_nodes:
                cost_to_node = node.cost + np.linalg.norm(node.pos - new_pos)
                heuristic = self._calculate_heuristic(new_pos)
                total_cost = cost_to_node + self.heuristic_bias * heuristic
                
                if total_cost < min_total_cost and not self._edge_collision(node.pos, new_pos):
                    min_total_cost = total_cost
                    best_parent = node
            
            if best_parent is None:
                continue
                
            # Create new node
            new_node = Node(new_pos, best_parent)
            new_node.h_cost = self._calculate_heuristic(new_pos)
            new_edge = Edge(best_parent, new_node)
            new_node.e_in = new_edge
            
            self.nodes.append(new_node)
            self.edges.append(new_edge)
            best_parent.children.append(new_node)
            
            # Rewire nearby nodes considering heuristic cost
            for node in nearby_nodes:
                potential_cost = new_node.cost + np.linalg.norm(new_node.pos - node.pos)
                if potential_cost < node.cost and not self._edge_collision(new_node.pos, node.pos):
                    old_parent = node.parent
                    if old_parent:
                        old_parent.children.remove(node)
                        self.edges.remove(node.e_in)
                    
                    node.parent = new_node
                    new_edge = Edge(new_node, node)
                    node.e_in = new_edge
                    node.cost = potential_cost
                    new_node.children.append(node)
                    self.edges.append(new_edge)
                    
                    # Update costs for all children
                    self._update_children_costs(node)
            
            # Check if we can connect to goal
            dist_to_goal = np.linalg.norm(new_pos - self.env.goal)
            if dist_to_goal < self.step_size and not self._edge_collision(new_pos, self.env.goal):
                goal_node = Node(self.env.goal, new_node)
                goal_node.h_cost = 0  # Zero heuristic cost at goal
                goal_edge = Edge(new_node, goal_node)
                goal_node.e_in = goal_edge
                self.nodes.append(goal_node)
                self.edges.append(goal_edge)
                new_node.children.append(goal_node)
                self.goal_nodes.append(goal_node)
                
                if goal_node.cost < self.best_path_cost:
                    self.best_path_cost = goal_node.cost
                    self.best_goal_node = goal_node
                    print(f"Found better path at iteration {iteration} with cost {self.best_path_cost:.2f}")
            
            # Visualization code remains the same
            if iteration % 20 == 0:
                self.env.draw_environment()
                self._draw_tree()
                if self.best_goal_node:
                    best_path = self._extract_path(self.best_goal_node)
                    self.env.draw_path(best_path)
                plt.pause(0.01)
        
        return self._extract_path(self.best_goal_node) if self.best_goal_node else None

    def _update_children_costs(self, node):
        """Recursively update costs for all children after rewiring"""
        for child in node.children:
            child.cost = node.cost + np.linalg.norm(child.pos - node.pos)
            self._update_children_costs(child)

    def _edge_collision(self, start, end):
        """Check if edge collides with obstacles using linear interpolation"""
        points = np.linspace(start, end, 10)
        return any(self.env.is_collision(point) for point in points)
    
    def _extract_path(self, goal_node):
        """Extract path from goal node to start node"""
        path = []
        current = goal_node
        while current is not None:
            path.append(current.pos)
            current = current.parent
        return path[::-1]
    
    def _draw_tree(self):
        """Visualize the RRT* tree"""
        for edge in self.edges:
            plt.plot([edge.st.pos[0], edge.en.pos[0]], 
                    [edge.st.pos[1], edge.en.pos[1]], 
                    'y-', alpha=0.3, linewidth=1)
        
        if self.best_goal_node:
            current = self.best_goal_node
            while current.parent is not None:
                plt.plot([current.pos[0], current.parent.pos[0]], 
                        [current.pos[1], current.parent.pos[1]], 
                        'g-', alpha=0.7, linewidth=2)
                current = current.parent

# Example usage
if __name__ == "__main__":
 
    
    # Create environment
    env = PathPlanningEnv(width=10, height=10, num_obstacles=100)
    
    # Create planner
    planner = RRTStarPlanner(env, step_size=0.5, neighbor_radius=1.0, max_iterations=1000)
    
    # Find path
    path = planner.plan()
    
    if path is not None:
        # Draw final path
        env.draw_environment()
        planner._draw_tree()
        env.draw_path(path)
        plt.show()
    else:
        print("No path found!")