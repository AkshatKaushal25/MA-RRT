import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class PathPlanningEnv:
    def __init__(self, width=100, height=100, num_obstacles=30, num_agents=2, starts=None, goals=None):
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.num_agents = num_agents
        self.starts = starts if starts is not None else []
        self.goals = goals if goals is not None else []
        
        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_environment()
        
    def setup_environment(self):
        """Initialize the environment with obstacles, start, and goal positions"""
        self.obstacles = []
        
        # Set up the plot
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # Add obstacles
        for _ in range(self.num_obstacles):
            x = np.random.uniform(10, 90)
            y = np.random.uniform(10, 90)
            size = np.random.uniform(2, 8)
            self.obstacles.append({'pos': (x, y), 'size': (size, size)})
        
        # Validate or generate start positions
        if not self.starts:
            self.starts = [self.random_pos_collision_free() for _ in range(self.num_agents)]
        else:
            if len(self.starts) != self.num_agents:
                raise ValueError(f"Expected {self.num_agents} start positions, got {len(self.starts)}.")
        
        # # Validate or generate goal positions
        # if not self.goals:
        #     self.goals = [self.random_pos_collision_free() for _ in range(self.num_agents)]
        # else:
        #     if len(self.goals) != self.num_agents:
        #         raise ValueError(f"Expected {self.num_agents} goal positions, got {len(self.goals)}.")
        
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
        
        # Draw start positions (green) and goal positions (blue)
        for i, (start, goal) in enumerate(zip(self.starts, self.goals)):
            start_label = 'Start' if i == 0 else None
            goal_label = 'Goal' if i == 0 else None
            self.ax.plot(10, 10, 'go', markersize=10, label=start_label)
            for goal in self.goals:
                self.ax.plot(goal[0], goal[1], 'bo', markersize=10, label=goal_label)
        
        self.ax.legend()
        plt.draw()
    
    def draw_paths(self, paths):
        """Draw paths for multiple agents on the environment"""
        if paths is not None:
            for i, path in enumerate(paths):
                if path is not None and len(path) > 0:
                    # Ensure the path is a 2D array with shape (N, 2)
                    path = np.array(path)  # Convert to numpy array if not already
                    if path.ndim == 1:
                        # If the path is 1D, reshape it into a 2D array with shape (N//2, 2)
                        path = path.reshape(-1, 2)
                    
                    # Plot the path
                    color = plt.cm.tab10(i % 10)  # Cycle through colors for up to 10 agents
                    self.ax.plot(path[:, 0], path[:, 1], '--', color=color, linewidth=2, label=f'Agent {i} Path')
            self.ax.legend()
            plt.draw()
    
    def is_collision(self, point):
        """Check if a point collides with any obstacle"""
        for obs in self.obstacles:
            ox, oy = obs['pos']
            ow, oh = obs['size']
            if (ox <= point[0] <= ox + ow) and (oy <= point[1] <= oy + oh):
                return True
        return False
    
    def random_pos_collision_free(self):
        """Generate a random position that is collision-free"""
        while True:
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            if not self.is_collision((x, y)):
                return np.array([x, y])
            
import random
import numpy as np
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
    def __init__(self, start, goals, env, 
                 step_size=5.0, 
                 neighbor_radius=15.0, 
                 max_iter=1000,
                 goal_bias=0.2,
                 heuristic_bias=0.3):
        
        self.env = env
        self.start = np.array(start)
        self.goals = [np.array(goal) for goal in goals]  # List of goals
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.max_iter = max_iter
        self.goal_bias = goal_bias
        self.heuristic_bias = heuristic_bias
        
        self.nodes = []
        self.edges = []
        self.goal_nodes = []  # List of nodes that reach any goal
        self.best_path_costs = [float('inf')] * len(self.goals)  # Track best cost for each goal
        self.best_goal_nodes = [None] * len(self.goals)  # Track best node for each goal
        
        # Initialize tree
        root = Node(self.start)
        root.h_cost = self._calculate_heuristic(root.pos)
        self.nodes.append(root)
        
    def _calculate_heuristic(self, pos):
        """Calculate the minimum heuristic cost to any goal."""
        return min(np.linalg.norm(pos - goal) for goal in self.goals)
    
    def _sample_with_heuristic(self):
        rand_val = random.random()
    
        # Direct goal sampling (choose one of the goals randomly)
        if rand_val < self.goal_bias:
            return random.choice(self.goals)
        
        # Corridor-based sampling with deviation
        elif rand_val < (self.goal_bias + self.heuristic_bias):
            # Choose a random goal for corridor sampling
            chosen_goal = random.choice(self.goals)
            line_vector = chosen_goal - self.start
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
                    
            # Check goal connection for all goals
            for i, goal in enumerate(self.goals):
                if np.linalg.norm(new_pos - goal) < self.step_size:
                    if not self._edge_collision(new_pos, goal):
                        goal_node = Node(goal, new_node)
                        goal_node.h_cost = 0
                    
                        goal_edge = Edge(new_node, goal_node)
                        self.goal_nodes.append(goal_node)
                        self.nodes.append(goal_node)
                        self.edges.append(goal_edge)
                        
                        if goal_node.cost < self.best_path_costs[i]:
                            self.best_path_costs[i] = goal_node.cost
                            self.best_goal_nodes[i] = goal_node
                        
            # Update visualization every 50 iterations
            if iteration % 50 == 0:
                

                self.env.draw_environment()

                self._draw_tree()
                self._draw_all_best_paths()  # Draw best paths for all goals
                plt.pause(0.1)
                        
        return [self._extract_path(node) for node in self.best_goal_nodes if node is not None]
        
    def _update_children_costs(self, node):
        for child in node.children:
            child.cost = node.cost + np.linalg.norm(child.pos - node.pos)
            self._update_children_costs(child)
            
    def _extract_path(self, goal_node):
        path = []
        current = goal_node
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
    
    def _draw_all_best_paths(self):
        """Draw the best path for each goal"""
        colors = plt.cm.tab10.colors  # Use a colormap for distinct colors
        for i, goal_node in enumerate(self.best_goal_nodes):
            if goal_node is not None:
                path = self._extract_path(goal_node)
                self.env.ax.plot(path[:, 0], path[:, 1], '--', 
                                 color=colors[i % len(colors)], 
                                 linewidth=2, 
                                 label=f'Goal {i+1} Path')
        self.env.ax.legend()
# Initialize environment with multiple goals

goals = [(90, 90)] 
env = PathPlanningEnv(num_agents=1,goals=goals) # Multiple goals
planner = RRTStarPlanner(start=(10, 10), goals=goals, env=env,max_iter=700)
# Plan and visualize
paths = planner.plan()
if paths:


    env.draw_environment()
    for i, path in enumerate(paths):
       
        env.draw_paths(path)
    plt.legend()
    plt.show()
