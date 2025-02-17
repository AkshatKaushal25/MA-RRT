import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from scipy.optimize import linear_sum_assignment
from itertools import permutations

class MultiAgentPathPlanner:
    def __init__(self, width=100, height=100, num_obstacles=30, min_distance=10):
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.min_distance = min_distance  # Minimum distance between goals/obstacles
        self.obstacles = []
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        
    def generate_random_point(self):
        """Generate a random point with buffer from edges"""
        buffer = 5
        return np.array([
            random.uniform(buffer, self.width - buffer),
            random.uniform(buffer, self.height - buffer)
        ])
        
    def is_point_valid(self, point, existing_points, min_distance):
        """Check if point is far enough from existing points and obstacles"""
        # Check distance from existing points
        for p in existing_points:
            if np.linalg.norm(point - np.array(p)) < min_distance:
                return False
                
        # Check distance from obstacles
        for obs in self.obstacles:
            ox, oy = obs['pos']
            if np.linalg.norm(point - np.array([ox, oy])) < min_distance:
                return False
                
        return True
        
    def generate_obstacles(self):
        """Generate random obstacles with varying sizes"""
        self.obstacles = []
        existing_positions = []
        
        for _ in range(self.num_obstacles):
            for attempt in range(50):  # Limited attempts to find valid position
                pos = self.generate_random_point()
                if self.is_point_valid(pos, existing_positions, self.min_distance):
                    width = random.uniform(2, 6)
                    height = random.uniform(2, 6)
                    self.obstacles.append({
                        'pos': (pos[0], pos[1]),
                        'size': (width, height)
                    })
                    existing_positions.append(pos)
                    break
                    
    def generate_random_goals(self, num_goals, starts):
        """Generate random goals with minimum distance constraints"""
        goals = []
        existing_points = [np.array(start) for start in starts]
        
        for _ in range(num_goals):
            for attempt in range(50):  # Limited attempts to find valid position
                point = self.generate_random_point()
                if self.is_point_valid(point, existing_points, self.min_distance):
                    goals.append(point)
                    existing_points.append(point)
                    break
                    
        return goals

    def is_collision_free(self, point):
        """Check if a point is collision-free"""
        for obs in self.obstacles:
            ox, oy = obs['pos']
            ow, oh = obs['size']
            if (ox-ow/2 <= point[0] <= ox+ow/2 and 
                oy-oh/2 <= point[1] <= oy+oh/2):
                return False
        return True
    
    def is_path_collision_free(self, start, end, step_size=0.5):
        """Check if a path between two points is collision-free"""
        direction = end - start
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return True
            
        steps = int(distance / step_size)
        for i in range(steps + 1):
            point = start + (direction * i / steps)
            if not self.is_collision_free(point):
                return False
        return True
        
    def visualize_assignment(self, starts, goals, goal_assignments):
        """Visualize the goal assignments for each agent"""
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        
        # Draw obstacles
        for obs in self.obstacles:
            x, y = obs['pos']
            w, h = obs['size']
            rect = patches.Rectangle(
                (x-w/2, y-h/2), w, h,
                linewidth=1, edgecolor='r', facecolor='r', alpha=0.3
            )
            self.ax.add_patch(rect)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(starts)))
        
        # Draw starts and goals with assignments
        for i, (start, assigned_goals) in enumerate(zip(starts, goal_assignments)):
            color = colors[i]
            # Draw start position
            self.ax.plot(start[0], start[1], 'o', color=color, markersize=10, 
                        label=f'Agent {i+1} Start')
            
            # Draw assigned goals and connections
            for j, goal in enumerate(assigned_goals):
                self.ax.plot(goal[0], goal[1], '^', color=color, markersize=8)
                if j == 0:
                    self.ax.plot([start[0], goal[0]], [start[1], goal[1]], '--', 
                                color=color, alpha=0.3)
                if j > 0:
                    prev_goal = assigned_goals[j-1]
                    self.ax.plot([prev_goal[0], goal[0]], [prev_goal[1], goal[1]], 
                                '--', color=color, alpha=0.3)
        
        # Draw remaining goals
        self.ax.legend()
        plt.pause(0.01)

class RRTStarNode:
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent
        self.children = []
        self.cost = 0 if parent is None else (
            parent.cost + np.linalg.norm(self.position - parent.position)
        )

class MultiAgentRRTStar:
    def __init__(self, planner, start, goal, 
                 max_iter=1000, 
                 step_size=5.0,
                 search_radius=15.0,
                 goal_sample_rate=0.2):
        self.planner = planner
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.max_iter = max_iter
        self.step_size = step_size
        self.search_radius = search_radius
        self.goal_sample_rate = goal_sample_rate
        self.nodes = []
        self.goal_node = None
        
    def plan(self):
        """Execute RRT* path planning"""
        # Initialize with start node
        self.nodes = [RRTStarNode(self.start)]
        
        for i in range(self.max_iter):
            # Sample random point
            if random.random() < self.goal_sample_rate:
                sample = self.goal
            else:
                sample = self._random_valid_state()
                
            # Find nearest node
            nearest_node = self._get_nearest_node(sample)
            
            # Steer towards sample
            new_position = self._steer(nearest_node.position, sample)
            if new_position is None:
                continue
                
            # Check path validity
            if not self.planner.is_path_collision_free(nearest_node.position, new_position):
                continue
                
            # Find nearby nodes for rewiring
            nearby_nodes = self._get_nodes_within_radius(new_position)
            
            # Find best parent
            min_cost = float('inf')
            best_parent = None
            
            for node in nearby_nodes:
                potential_cost = (node.cost + 
                                np.linalg.norm(new_position - node.position))
                if (potential_cost < min_cost and 
                    self.planner.is_path_collision_free(node.position, new_position)):
                    min_cost = potential_cost
                    best_parent = node
                    
            if best_parent is None:
                continue
                
            # Create new node
            new_node = RRTStarNode(new_position, best_parent)
            best_parent.children.append(new_node)
            self.nodes.append(new_node)
            
            # Rewire nearby nodes
            self._rewire_nodes(new_node, nearby_nodes)
            
            # Check if goal is reached
            if (np.linalg.norm(new_position - self.goal) < self.step_size and 
                self.planner.is_path_collision_free(new_position, self.goal)):
                goal_node = RRTStarNode(self.goal, new_node)
                new_node.children.append(goal_node)
                self.nodes.append(goal_node)
                self.goal_node = goal_node
                break
                
            # Visualize progress periodically
            if i % 100 == 0:
                self._visualize_tree()
                
        return self._extract_path()
    
    def _random_valid_state(self):
        """Generate random valid state"""
        while True:
            state = np.array([
                random.uniform(0, self.planner.width),
                random.uniform(0, self.planner.height)
            ])
            if self.planner.is_collision_free(state):
                return state
                
    def _get_nearest_node(self, point):
        """Find nearest node to given point"""
        return min(self.nodes, 
                  key=lambda n: np.linalg.norm(n.position - point))
                  
    def _steer(self, from_pos, to_pos):
        """Steer from current position towards target"""
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        if distance < self.step_size:
            return to_pos
        return from_pos + (direction / distance) * self.step_size
        
    def _get_nodes_within_radius(self, position):
        """Find all nodes within search radius"""
        return [node for node in self.nodes 
                if np.linalg.norm(node.position - position) < self.search_radius]
                
    def _rewire_nodes(self, new_node, nearby_nodes):
        """Rewire nearby nodes through new node if it provides better path"""
        for node in nearby_nodes:
            if node.parent == new_node:
                continue
                
            potential_cost = (new_node.cost + 
                            np.linalg.norm(node.position - new_node.position))
                            
            if (potential_cost < node.cost and 
                self.planner.is_path_collision_free(new_node.position, node.position)):
                # Remove from old parent
                if node.parent:
                    node.parent.children.remove(node)
                    
                # Update parent
                node.parent = new_node
                node.cost = potential_cost
                new_node.children.append(node)
                
                # Update children costs
                self._update_children_costs(node)
                
    def _update_children_costs(self, node):
        """Update costs of all child nodes"""
        for child in node.children:
            child.cost = (node.cost + 
                         np.linalg.norm(child.position - node.position))
            self._update_children_costs(child)
            
    def _extract_path(self):
        """Extract path from goal to start"""
        if self.goal_node is None:
            return None
            
        path = []
        current = self.goal_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return np.array(path[::-1])
        
    def _visualize_tree(self):
        """Visualize the RRT* tree"""
        self.planner.ax.clear()
        self.planner.ax.set_xlim(0, self.planner.width)
        self.planner.ax.set_ylim(0, self.planner.height)
        
        # Draw obstacles
        for obs in self.planner.obstacles:
            x, y = obs['pos']
            w, h = obs['size']
            rect = patches.Rectangle(
                (x-w/2, y-h/2), w, h,
                linewidth=1, edgecolor='r', facecolor='r', alpha=0.5
            )
            self.planner.ax.add_patch(rect)
            
        # Draw tree edges
        for node in self.nodes:
            if node.parent is not None:
                self.planner.ax.plot(
                    [node.position[0], node.parent.position[0]],
                    [node.position[1], node.parent.position[1]],
                    'g-', alpha=0.3, linewidth=1
                )
                
        # Draw start and goal
        self.planner.ax.plot(self.start[0], self.start[1], 'go', markersize=10)
        self.planner.ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10)
        
        plt.pause(0.01)

def optimize_goal_sequence(start, goals):
    """Optimize the sequence of goals using TSP"""
    if not goals:
        return []
    
    best_sequence = []
    min_cost = float('inf')
    
    for perm in permutations(goals):
        cost = np.linalg.norm(start - perm[0])
        for i in range(len(perm)-1):
            cost += np.linalg.norm(perm[i] - perm[i+1])
        if cost < min_cost:
            min_cost = cost
            best_sequence = list(perm)
            
    return best_sequence

def assign_goals_to_agents(starts, goals, planner):
    """Assign multiple goals to agents and optimize sequences"""
    num_agents = len(starts)
    goals_per_agent = len(goals) // num_agents
    remaining_goals = len(goals) % num_agents
    
    # Convert to numpy arrays
    starts = [np.array(start) for start in starts]
    goals = [np.array(goal) for goal in goals]
    
    # Initialize assignments
    assignments = [[] for _ in range(num_agents)]
    remaining = goals.copy()
    
    # Distribute goals evenly among agents
    for i in range(num_agents):
        agent_goals = []
        num_goals = goals_per_agent + (1 if i < remaining_goals else 0)
        
        # Find best goals for current agent
        for _ in range(num_goals):
            if not remaining:
                break
                
            # Create cost matrix for remaining goals
            costs = [np.linalg.norm(starts[i] - goal) for goal in remaining]
            best_idx = np.argmin(costs)
            agent_goals.append(remaining.pop(best_idx))
        
        # Optimize sequence for this agent's goals
        if agent_goals:
            optimized_sequence = optimize_goal_sequence(starts[i], agent_goals)
            assignments[i] = optimized_sequence
    
    return assignments

def plan_multi_agent_paths(starts, goals, planner):
    """Plan paths for multiple agents with multiple goals each"""
    # Assign goals to agents
    goal_assignments = assign_goals_to_agents(starts, goals, planner)
    
    # Visualize assignments
    planner.visualize_assignment(starts, goals, goal_assignments)
    
    # Plan paths for each agent
    all_paths = []
    for agent_idx, (start, assigned_goals) in enumerate(zip(starts, goal_assignments)):
        agent_path = []
        current_pos = np.array(start)
        
        # Plan path through each assigned goal
        for goal in assigned_goals:
            rrt_planner = MultiAgentRRTStar(
                planner=planner,
                start=current_pos,
                goal=goal,
                max_iter=1000,
                step_size=3.0,
                search_radius=10.0,
                goal_sample_rate=0.2
            )
            path = rrt_planner.plan()
            if path is not None:
                agent_path.extend(path)
                current_pos = goal
                
        all_paths.append(np.array(agent_path))
    
    return all_paths, goal_assignments

# Example usage
def main():
    # Initialize environment
    planner = MultiAgentPathPlanner(width=100, height=100, num_obstacles=40, min_distance=10)
    planner.generate_obstacles()
    
    # Define starts (could also be randomized if desired)
    starts = [
        (10, 10),  # Agent 1
        (90, 10),  # Agent 2
        (50, 10)   # Agent 3
    ]
    
    # Generate 10 random goals
    goals = planner.generate_random_goals(10, starts)
    
    # Plan paths for all agents
    paths, goal_assignments = plan_multi_agent_paths(starts, goals, planner)
    
    # Visualize final results
    planner.ax.clear()
    planner.ax.set_xlim(0, planner.width)
    planner.ax.set_ylim(0, planner.height)
    
    # Draw obstacles
    for obs in planner.obstacles:
        x, y = obs['pos']
        w, h = obs['size']
        rect = patches.Rectangle(
            (x-w/2, y-h/2), w, h,
            linewidth=1, edgecolor='r', facecolor='r', alpha=0.3
        )
        planner.ax.add_patch(rect)
    
    # Draw paths and assignments with different colors for each agent
    colors = plt.cm.tab10(np.linspace(0, 1, len(starts)))
    
    for i, (path, assigned_goals) in enumerate(zip(paths, goal_assignments)):
        if path is not None and len(path) > 0:
            color = colors[i]
            
            # Draw start position
            planner.ax.plot(starts[i][0], starts[i][1], 'o', color=color, 
                          markersize=10, label=f'Agent {i+1} Start')
            
            # Draw path
            planner.ax.plot(path[:, 0], path[:, 1], '-', color=color, 
                          linewidth=2, alpha=0.7)
            
            # Draw assigned goals
            for j, goal in enumerate(assigned_goals):
                planner.ax.plot(goal[0], goal[1], '^', color=color, 
                              markersize=8, label=f'Agent {i+1} Goal {j+1}')
    
    planner.ax.grid(True)
    planner.ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
