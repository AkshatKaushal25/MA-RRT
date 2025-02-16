import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import numpy as np
import math
import random

# ================== Environment Class ==================
class PathPlanningEnv:
    def __init__(self, width=100, height=100, num_obstacles=10):
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        
        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_environment()
        
    def setup_environment(self):
        """Initialize the environment with obstacles, start, and goal positions"""
        self.obstacles = []
        self.start = None
        self.goal = None
        
        # Set up the plot
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # Add obstacles
        for _ in range(self.num_obstacles):
            x = np.random.uniform(10,90)
            y = np.random.uniform(10, 90)
            size = np.random.uniform(2, 8)
            self.obstacles.append({'pos': (x, y), 'size': (size, size)})
        
        # Add start and goal positions
        self.start = (10, 10)
        self.goal = (90, 90)
        
    def draw_environment(self):
        """Draw the current state of the environment"""
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.grid(True)
        
        # Draw obstacles
        for obs in self.obstacles:
            rect = patches.Rectangle(
                obs['pos'], obs['size'][0], obs['size'][1],
                linewidth=1, edgecolor='r', facecolor='r'
            )
            self.ax.add_patch(rect)
        
        # Draw start and goal
        self.ax.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        self.ax.plot(self.goal[0], self.goal[1], 'bo', markersize=10, label='Goal')
        
        self.ax.legend()
        plt.draw()
    
    def draw_path(self, path):
        """Draw a path on the environment"""
        if path is not None:
            self.ax.plot(path[:, 0], path[:, 1], 'g--', linewidth=2, label='Path')
            self.ax.legend()
            plt.draw()
    
    def is_collision(self, point):
        """Check if a point collides with any obstacle"""
        for obs in self.obstacles:
            ox, oy = obs['pos']
            ow, oh = obs['size']
            if (ox <= point[0] <= ox+ow) and (oy <= point[1] <= oy+oh):
                return True
        return False
    
    def random_pos_collision_free(self):
        """Generate a random position that is collision-free"""
        while True:
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            if not self.is_collision((x, y)):
                return np.array([x, y])

# ================== RRT* Classes ==================
class Node:
    def __init__(self, pos, parent=None):
        self.pos = pos
        self.parent = parent
        self.children = []
        self.cost = 0 if parent is None else parent.cost + np.linalg.norm(pos - parent.pos)
        self.h_cost = 0  # Heuristic cost
        
class Edge:
    def __init__(self, st, en):
        self.st = st
        self.en = en
        self.len = np.linalg.norm(en.pos - st.pos)

class RRTStarPlanner:
    def __init__(self, start, goal, env, 
                 step_size=5.0, 
                 neighbor_radius=15.0, 
                 max_iter=1000,
                 goal_bias=0.2,
                 heuristic_bias=0.3):
        
        self.env = env
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.max_iter = max_iter
        self.goal_bias = goal_bias
        self.heuristic_bias = heuristic_bias
        
        self.nodes = []
        self.edges = []
        self.goal_nodes = []
        self.best_path_cost = float('inf')
        self.best_goal_node = None
        
        # Initialize tree
        root = Node(self.start)
        root.h_cost = self._calculate_heuristic(root.pos)
        self.nodes.append(root)
        
    def _calculate_heuristic(self, pos):
        return np.linalg.norm(pos - self.goal)
    
    def _sample_with_heuristic(self):
        rand_val = random.random()
    
        # Direct goal sampling
        if rand_val < self.goal_bias:
            return self.goal
        
        # Corridor-based sampling with deviation
        elif rand_val < (self.goal_bias + self.heuristic_bias):
            # Create samples along start-goal line with perpendicular noise
            line_vector = self.goal - self.start
            line_length = np.linalg.norm(line_vector)
            unit_vector = line_vector / line_length
            
            # Generate points along the line with lateral deviation
            samples = []
            for _ in range(10):
                # Random position along start-goal line (60-140% extension)
                t = random.uniform(-0.4, 1.4)
                along_line = self.start + t * line_vector
                
                # Add perpendicular noise (deviation up to 20% of line length)
                max_deviation = 0.2 * line_length
                deviation = random.uniform(-max_deviation, max_deviation)
                perp_vector = np.array([-unit_vector[1], unit_vector[0]]) * deviation
                
                sample = along_line + perp_vector
                
                # Clamp to environment bounds
                sample = np.clip(sample, 0, self.env.width)
                if self.env.is_collision(sample):
                    continue
                samples.append(sample)
            
            if not samples:
                return self.env.random_pos_collision_free()
                
            # Select best sample considering both progress and tree density
            return min(samples,
                    key=lambda s: (self._calculate_heuristic(s) + 
                                0.3 * min(np.linalg.norm(n.pos - s) for n in self.nodes)))
        
        # Pure random exploration
        else:
            return self.env.random_pos_collision_free()
            
    def _edge_collision(self, start, end):
        steps = int(np.linalg.norm(end - start)/self.step_size*10) + 1
        for t in np.linspace(0, 1, steps):
            test_point = start*(1-t) + end*t
            if self.env.is_collision(test_point):
                return True
        return False
        
    def plan(self):
        for iteration in range(self.max_iter):
            x_rand = self._sample_with_heuristic()
            if x_rand is None:
                continue
                
            # Find nearest node with heuristic consideration
            nearest_node = min(self.nodes, 
                key=lambda n: np.linalg.norm(n.pos - x_rand) + self.heuristic_bias*n.h_cost)
                
            # Steer towards random point
            direction = x_rand - nearest_node.pos
            distance = np.linalg.norm(direction)
            if distance > self.step_size:
                direction = direction/distance * self.step_size
            new_pos = nearest_node.pos + direction
            
            if self.env.is_collision(new_pos):
                continue
                
            # Find nearby nodes
            nearby_nodes = [n for n in self.nodes 
                           if np.linalg.norm(n.pos - new_pos) < self.neighbor_radius]
                
            # Find best parent with cost+heuristic optimization
            min_total_cost = float('inf')
            best_parent = None
            for node in nearby_nodes:
                cost = node.cost + np.linalg.norm(node.pos - new_pos)
                total_cost = cost + self.heuristic_bias*self._calculate_heuristic(new_pos)
                if total_cost < min_total_cost and not self._edge_collision(node.pos, new_pos):
                    min_total_cost = total_cost
                    best_parent = node
                    
            if best_parent is None:
                continue
                
            # Create new node
            new_node = Node(new_pos, best_parent)
            new_node.h_cost = self._calculate_heuristic(new_pos)
            new_edge = Edge(best_parent, new_node)
            self.nodes.append(new_node)
            self.edges.append(new_edge)
            best_parent.children.append(new_node)
            
            # Rewire nearby nodes
            for node in nearby_nodes:
                new_cost = new_node.cost + np.linalg.norm(new_node.pos - node.pos)
                if new_cost < node.cost and not self._edge_collision(new_node.pos, node.pos):
                    # Update parent
                    old_parent = node.parent
                    if old_parent:
                        if node in old_parent.children:
                            old_parent.children.remove(node)  # Remove from old parent's children
                    
                    node.parent = new_node
                    node.cost = new_cost
                    new_node.children.append(node)  # Add to new parent's children
                    new_edge = Edge(new_node, node)
                    if hasattr(node, 'e_in'):
                        self.edges.remove(node.e_in)  # Remove old edge if it exists
                    node.e_in = new_edge
                    self.edges.append(new_edge)
                    
                    # Update children costs
                    self._update_children_costs(node)
                    
            # Check goal connection
            if np.linalg.norm(new_pos - self.goal) < self.step_size:
                if not self._edge_collision(new_pos, self.goal):
                    goal_node = Node(self.goal, new_node)
                    goal_node.h_cost = 0
                    goal_edge = Edge(new_node, goal_node)
                    self.goal_nodes.append(goal_node)
                    self.nodes.append(goal_node)
                    self.edges.append(goal_edge)
                    
                    if goal_node.cost < self.best_path_cost:
                        self.best_path_cost = goal_node.cost
                        self.best_goal_node = goal_node
                        
            # Update visualization every 50 iterations
            if iteration % 50 == 0:
                self.env.draw_environment()
                self._draw_tree()
                if self.best_goal_node:
                    best_path = self._extract_path()
                    self.env.draw_path(best_path)
                plt.pause(0.1)
                        
        return self._extract_path() if self.best_goal_node else None
        
    def _update_children_costs(self, node):
        for child in node.children:
            child.cost = node.cost + np.linalg.norm(child.pos - node.pos)
            self._update_children_costs(child)
            
    def _extract_path(self):
        path = []
        current = self.best_goal_node
        while current is not None:
            path.append(current.pos)
            current = current.parent
        return np.array(path[::-1])
    
    def _draw_tree(self):
        """Visualize the RRT* tree"""
        for edge in self.edges:
            self.env.ax.plot([edge.st.pos[0], edge.en.pos[0]], 
                            [edge.st.pos[1], edge.en.pos[1]], 
                            'y-', alpha=0.3, linewidth=1)
        
        if self.best_goal_node:
            current = self.best_goal_node
            while current.parent is not None:
                self.env.ax.plot([current.pos[0], current.parent.pos[0]], 
                                [current.pos[1], current.parent.pos[1]], 
                                'g-', alpha=0.7, linewidth=2)
                current = current.parent

# ================== Main Execution ==================
if __name__ == "__main__":
    # Create environment
    env = PathPlanningEnv(width=100, height=100, num_obstacles=40)
    
    # Plan path
    planner = RRTStarPlanner(env.start, env.goal, env, 
                             step_size=5.0,
                             neighbor_radius=15.0,
                             max_iter=1000,
                             goal_bias=0.2,
                             heuristic_bias=0.3)
    path = planner.plan()
    
    # Final visualization
    env.draw_environment()
    if path is not None:
        env.draw_path(path)
    plt.show()