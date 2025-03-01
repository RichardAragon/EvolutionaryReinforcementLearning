import numpy as np
import random
import matplotlib.pyplot as plt

# Efficient Hyperdimensional Computing (HDC) representation
class HDMemory:
    def __init__(self, dim=1000):
        self.dim = dim
        self.memory = {}
        # Pre-generate random vectors for common values to avoid repeated computations
        self.common_vectors = {}
    
    def encode(self, key, data):
        # Check if we've already encoded this value
        data_hash = hash(str(data)) % 2**32
        if data_hash in self.common_vectors:
            return self.common_vectors[data_hash]
            
        # Simplified encoding with smaller vectors
        if isinstance(data, (int, float)):
            np.random.seed(int(data_hash))
            hv = np.random.choice([-1, 1], size=self.dim)
            np.random.seed(None)
            # Cache common values
            if len(self.common_vectors) < 1000:  # Limit cache size
                self.common_vectors[data_hash] = hv
        else:
            hv = np.random.choice([-1, 1], size=self.dim)
        
        self.memory[key] = hv
        return hv
    
    def retrieve(self, key):
        return self.memory.get(key, np.zeros(self.dim))
    
    def similarity(self, hv1, hv2):
        # Fast cosine similarity
        return np.dot(hv1, hv2) / (self.dim)  # Normalized for binary vectors

# Standard Q-Learning Baseline (unchanged)
class QLearningAgent:
    def __init__(self, state_size=10, action_size=4, learning_rate=0.1, 
                 discount_factor=0.9, exploration_prob=0.1):
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
    
    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_prob:
            return random.randint(0, self.q_table.shape[1] - 1)
        return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (
            reward + self.discount_factor * self.q_table[next_state, best_next_action] - 
            self.q_table[state, action]
        )
    
    def train(self, episodes=100):
        performance_history = []
        for episode in range(episodes):
            state = random.randint(0, self.q_table.shape[0] - 1)
            action = self.choose_action(state)
            reward = random.uniform(0, 1)
            next_state = random.randint(0, self.q_table.shape[0] - 1)
            self.update_q_table(state, action, reward, next_state)
            performance_history.append(reward)
        return performance_history

# Optimized Swarm Agent
class SwarmAgent:
    def __init__(self, id, queen, state_size=10, action_size=4, hd_dim=1000):
        self.id = id
        self.queen = queen
        self.state_size = state_size
        self.action_size = action_size
        
        # Enhanced agent genetic traits
        self.traits = {
            'learning_rate': random.uniform(0.05, 0.5),
            'discount_factor': random.uniform(0.7, 0.99),
            'exploration_prob': random.uniform(0.05, 0.3),
            'eligibility_trace': random.uniform(0, 0.3),
            'reward_scaling': random.uniform(0.8, 1.2)
        }
        
        # HD memory for pattern recognition
        self.hd_memory = HDMemory(dim=hd_dim)
        self.state_vectors = {}  # Cache for encoded states
        
        # Sparse representation of Q-values for memory efficiency
        self.q_values = {}
        self.performance = 0.5
        self.memory = []
        self.experience_count = 0
    
    def encode_state(self, state):
        # Get or create HD vector for this state
        if state not in self.state_vectors:
            self.state_vectors[state] = self.hd_memory.encode(f"state_{state}", state)
        return self.state_vectors[state]
    
    def choose_action(self, state):
        state_key = f"s{state}"
        
        # Initialize state entry if needed
        if state_key not in self.q_values:
            self.q_values[state_key] = np.random.uniform(0, 0.1, size=self.action_size)
        
        # Dynamic exploration strategy based on experience
        current_exploration = max(0.05, self.traits['exploration_prob'] * 
                                (1.0 / (1.0 + 0.01 * self.experience_count)))
        
        # Explore or exploit
        if random.uniform(0, 1) < current_exploration:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_values[state_key])
    
    def update_knowledge(self, state, action, reward, next_state, is_shared_exp=False):
        state_key = f"s{state}"
        next_state_key = f"s{next_state}"
        
        # Apply individual reward scaling
        scaled_reward = reward * self.traits['reward_scaling']
        
        # Initialize if needed
        if state_key not in self.q_values:
            self.q_values[state_key] = np.random.uniform(0, 0.1, size=self.action_size)
        if next_state_key not in self.q_values:
            self.q_values[next_state_key] = np.random.uniform(0, 0.1, size=self.action_size)
        
        # Encode states with HD vectors for pattern recognition
        state_vector = self.encode_state(state)
        next_state_vector = self.encode_state(next_state)
        
        # Similarity can influence learning
        state_similarity = self.hd_memory.similarity(state_vector, next_state_vector)
        similarity_factor = 1.0 + 0.2 * (state_similarity + 1.0) / 2.0  # Range [1.0, 1.2]
        
        # Adaptive learning rate based on whether experience is shared
        effective_lr = self.traits['learning_rate']
        if is_shared_exp:
            effective_lr *= 0.5  # Reduced learning from shared experiences
        
        # Q-learning update with eligibility traces influence
        best_next_value = np.max(self.q_values[next_state_key])
        td_error = scaled_reward + self.traits['discount_factor'] * best_next_value - self.q_values[state_key][action]
        
        # Apply eligibility trace effect to neighboring actions (simplified)
        for a in range(self.action_size):
            # Calculate action similarity - closer actions get more update
            action_similarity = 1.0 - abs(a - action) / self.action_size
            eligibility = self.traits['eligibility_trace'] * action_similarity
            
            if a == action:
                # Direct update for chosen action
                self.q_values[state_key][a] += effective_lr * td_error * similarity_factor
            elif eligibility > 0.05:  # Skip tiny updates for efficiency
                # Neighboring actions get smaller updates
                self.q_values[state_key][a] += effective_lr * td_error * eligibility
        
        # Update memory for performance tracking
        if not is_shared_exp:  # Only track performance on own experiences
            self.memory.append(scaled_reward)
            if len(self.memory) > 10:  # Fixed small memory for efficiency
                self.memory.pop(0)
            
            # Update overall performance with recency bias
            self.performance = 0.3 * self.performance + 0.7 * np.mean(self.memory)
            self.experience_count += 1
            
            # Adaptive exploration decay based on performance
            if self.experience_count % 10 == 0:
                self.traits['exploration_prob'] = max(
                    0.05,  # Minimum exploration
                    self.traits['exploration_prob'] * (1.0 - 0.01 * self.performance)
                )
    
    def mutate_traits(self, mutation_rate=0.1):
        # Enhanced mutation with adaptive rates
        for trait in self.traits:
            if random.random() < mutation_rate:
                if trait == 'learning_rate':
                    self.traits[trait] = max(0.01, min(0.5, self.traits[trait] + random.uniform(-0.1, 0.1)))
                elif trait == 'discount_factor':
                    self.traits[trait] = max(0.5, min(0.99, self.traits[trait] + random.uniform(-0.1, 0.1)))
                elif trait == 'exploration_prob':
                    self.traits[trait] = max(0.01, min(0.5, self.traits[trait] + random.uniform(-0.1, 0.1)))
                elif trait == 'eligibility_trace':
                    self.traits[trait] = max(0, min(0.5, self.traits[trait] + random.uniform(-0.1, 0.1)))
                elif trait == 'reward_scaling':
                    self.traits[trait] = max(0.5, min(1.5, self.traits[trait] + random.uniform(-0.2, 0.2)))

# Optimized Swarm Queen
class SwarmQueen:
    def __init__(self, num_agents=10, state_size=10, action_size=4):
        self.agents = [SwarmAgent(i, self, state_size, action_size) for i in range(num_agents)]
        self.state_size = state_size
        self.action_size = action_size
        self.elite_size = max(1, num_agents // 5)  # Top 20% are elite
        self.generation = 0
        
        # Track best agent traits for convergence analysis
        self.best_traits_history = []
    
    def select_parents(self):
        # Tournament selection with diversity bonus
        sorted_agents = sorted(self.agents, key=lambda a: a.performance, reverse=True)
        
        # Calculate diversity metrics
        diversity_scores = []
        for agent in sorted_agents:
            # Measure trait distance from average
            avg_traits = {
                t: np.mean([a.traits[t] for a in sorted_agents[:self.elite_size]])
                for t in agent.traits
            }
            
            dist = sum(
                abs(agent.traits[t] - avg_traits[t]) / avg_traits[t]
                for t in agent.traits
            )
            
            diversity_scores.append(dist)
        
        # Normalize diversity scores
        if max(diversity_scores) > 0:
            diversity_scores = [d / max(diversity_scores) for d in diversity_scores]
        
        # Create weighted selection probabilities (performance + diversity)
        performance_ranks = [1.0 / (i + 1) for i in range(len(sorted_agents))]
        selection_weights = [
            0.7 * p + 0.3 * d 
            for p, d in zip(performance_ranks, diversity_scores)
        ]
        
        # Normalize weights to probabilities
        total_weight = sum(selection_weights)
        selection_probs = [w / total_weight for w in selection_weights]
        
        # Select parents based on these probabilities
        parent_indices = np.random.choice(
            len(sorted_agents),
            size=min(10, len(sorted_agents)),
            p=selection_probs,
            replace=False
        )
        
        return [sorted_agents[i] for i in parent_indices]
    
    def crossover(self, parent1, parent2):
        # Create child agent
        child = SwarmAgent(len(self.agents), self, self.state_size, self.action_size)
        
        # Inherit traits with intelligent crossover
        for trait in parent1.traits:
            # Interpolation rather than binary selection
            mix_ratio = random.random()
            child.traits[trait] = (
                mix_ratio * parent1.traits[trait] + 
                (1 - mix_ratio) * parent2.traits[trait]
            )
            
            # Add small random variation
            child.traits[trait] += random.uniform(-0.05, 0.05)
            
            # Ensure bounds
            if trait == 'learning_rate':
                child.traits[trait] = max(0.01, min(0.5, child.traits[trait]))
            elif trait == 'discount_factor':
                child.traits[trait] = max(0.5, min(0.99, child.traits[trait]))
            elif trait == 'exploration_prob':
                child.traits[trait] = max(0.01, min(0.5, child.traits[trait]))
            elif trait == 'eligibility_trace':
                child.traits[trait] = max(0, min(0.5, child.traits[trait]))
            elif trait == 'reward_scaling':
                child.traits[trait] = max(0.5, min(1.5, child.traits[trait]))
        
        # Knowledge transfer from parents - inherit some Q-values
        # This is computationally expensive but very effective for learning
        if random.random() < 0.5:  # 50% chance to transfer knowledge
            for state_key in set(parent1.q_values.keys()).intersection(parent2.q_values.keys()):
                if len(child.q_values) < 20:  # Limit to prevent memory bloat
                    # Weighted average favoring the better performing parent
                    if parent1.performance > parent2.performance:
                        weight1, weight2 = 0.7, 0.3
                    else:
                        weight1, weight2 = 0.3, 0.7
                        
                    child.q_values[state_key] = (
                        weight1 * parent1.q_values[state_key] + 
                        weight2 * parent2.q_values[state_key]
                    )
        
        return child
    
    def evolve_population(self):
        # Track evolution progress
        self.generation += 1
        
        # Select parents
        parents = self.select_parents()
        
        # Preserve elites
        sorted_agents = sorted(self.agents, key=lambda a: a.performance, reverse=True)
        new_agents = []
        
        # Direct elitism - keep top performers unchanged
        elite_count = min(self.elite_size, len(sorted_agents))
        new_agents.extend(sorted_agents[:elite_count])
        
        # Track best agent traits
        if elite_count > 0:
            self.best_traits_history.append(sorted_agents[0].traits.copy())
        
        # Generate offspring for the rest of the population
        while len(new_agents) < len(self.agents):
            # Select two parents
            if len(parents) >= 2:
                parent1, parent2 = random.sample(parents, k=2)
                
                # Create offspring
                child = self.crossover(parent1, parent2)
                
                # Mutation rate increases as generations progress to avoid premature convergence
                adaptive_mutation_rate = 0.1 + 0.05 * min(1.0, self.generation / 20)
                child.mutate_traits(mutation_rate=adaptive_mutation_rate)
                
                new_agents.append(child)
            else:
                # If not enough parents, create a random agent
                new_agent = SwarmAgent(len(new_agents), self, self.state_size, self.action_size)
                new_agents.append(new_agent)
        
        # Update IDs and transfer population
        for i, agent in enumerate(new_agents):
            agent.id = i
        
        self.agents = new_agents
    
    def run_evaluation_cycle(self):
        # Create shared experience buffer
        shared_experiences = []
        
        # Evaluate all agents
        for agent in self.agents:
            agent_experiences = []
            
            # Each agent tries multiple tasks
            for _ in range(5):  # 5 evaluations per cycle for statistical stability
                state = random.randint(0, self.state_size - 1)
                action = agent.choose_action(state)
                
                # Enhanced reward function that creates better learning signal
                base_reward = random.uniform(0, 1)
                
                # Action-state matching component
                action_match = 1.0 - 0.3 * abs((action / self.action_size) - (state / self.state_size))
                
                # Progressive difficulty based on agent performance
                difficulty_factor = 1.0 + 0.3 * (agent.performance)
                
                # Final reward calculation
                reward = (base_reward * action_match) / difficulty_factor
                
                # State transition model with some stochasticity
                next_state = (state + action + random.randint(-1, 1)) % self.state_size
                
                # Store experience
                agent_experiences.append((state, action, reward, next_state))
                
                # Agent learns from its own experience
                agent.update_knowledge(state, action, reward, next_state)
            
            # Add best experience to shared pool
            if agent_experiences:
                best_exp = max(agent_experiences, key=lambda x: x[2])  # Select by reward
                shared_experiences.append(best_exp)
        
        # Experience sharing among elite agents
        if shared_experiences:
            elite_agents = sorted(self.agents, key=lambda a: a.performance, reverse=True)[:self.elite_size]
            
            # Each elite agent learns from 2 random shared experiences
            for agent in elite_agents:
                for _ in range(min(2, len(shared_experiences))):
                    exp = random.choice(shared_experiences)
                    state, action, reward, next_state = exp
                    # Learn from shared experience with reduced weight
                    agent.update_knowledge(state, action, reward, next_state, is_shared_exp=True)
        
        # Return average and best performance
        performances = [agent.performance for agent in self.agents]
        return np.mean(performances), max(performances) if performances else 0
    
    def run_evolution_cycle(self, cycles_per_generation=3):
        # Run multiple evaluation cycles
        cycle_avg_performances = []
        cycle_best_performances = []
        
        for _ in range(cycles_per_generation):
            avg_perf, best_perf = self.run_evaluation_cycle()
            cycle_avg_performances.append(avg_perf)
            cycle_best_performances.append(best_perf)
        
        # Evolve population
        self.evolve_population()
        
        return np.mean(cycle_avg_performances), np.mean(cycle_best_performances)

# Optimized Evolutionary Reinforcement Learning Framework
class ERLFramework:
    def __init__(self, state_size=10, action_size=4, num_agents=10):
        self.swarm_queen = SwarmQueen(
            num_agents=num_agents,
            state_size=state_size, 
            action_size=action_size
        )
        self.avg_performance_history = []
        self.best_performance_history = []
        self.population_size_history = []
    
    def train(self, iterations=50, cycles_per_generation=3):
        for i in range(iterations):
            # Adaptive cycles - more evaluation as training progresses
            adaptive_cycles = cycles_per_generation
            if i > iterations // 2:
                adaptive_cycles += 1  # One more cycle in latter half of training
            
            # Run evolution cycle
            avg_performance, best_performance = self.swarm_queen.run_evolution_cycle(adaptive_cycles)
            
            # Track histories
            self.avg_performance_history.append(avg_performance)
            self.best_performance_history.append(best_performance)
            self.population_size_history.append(len(self.swarm_queen.agents))
            
            # Dynamic population adjustments (every 10 iterations)
            if i > 0 and i % 10 == 0 and i < iterations - 10:
                # Check if performance has plateaued
                recent_best = self.best_performance_history[-5:]
                if max(recent_best) - min(recent_best) < 0.05:  # Small variance indicates plateau
                    # Add new agents with traits similar to best agent but more diverse
                    best_agent = max(self.swarm_queen.agents, key=lambda a: a.performance)
                    
                    # Add 2 new agents with traits influenced by best agent
                    for _ in range(2):
                        new_agent = SwarmAgent(
                            len(self.swarm_queen.agents), 
                            self.swarm_queen,
                            self.swarm_queen.state_size, 
                            self.swarm_queen.action_size
                        )
                        
                        # Copy and mutate best agent's traits with higher mutation
                        for trait in best_agent.traits:
                            mutation = random.uniform(-0.2, 0.2)
                            new_agent.traits[trait] = max(0.01, min(0.99, best_agent.traits[trait] + mutation))
                        
                        self.swarm_queen.agents.append(new_agent)
        
        return self.avg_performance_history, self.best_performance_history

# Compare frameworks
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    iterations = 50
    state_size = 10
    action_size = 4
    
    print("Training ERL framework...")
    erl_system = ERLFramework(state_size=state_size, action_size=action_size, num_agents=10)
    erl_avg_performance, erl_best_performance = erl_system.train(iterations, cycles_per_generation=3)
    
    print("Training Q-Learning baseline...")
    q_agent = QLearningAgent(state_size=state_size, action_size=action_size)
    q_performance = q_agent.train(iterations)
    
    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.plot(erl_best_performance, label='ERL Best Agent', color='green', linewidth=2)
    plt.plot(erl_avg_performance, label='ERL Average', color='blue')
    plt.plot(q_performance, label='Q-Learning', color='red', linestyle='dashed')
    plt.xlabel('Iteration')
    plt.ylabel('Performance')
    plt.title('ERL vs. Q-Learning Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Additional analysis plot for ERL
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(erl_best_performance, label='Best Agent', color='green', linewidth=2)
    plt.plot(erl_avg_performance, label='Swarm Average', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Performance')
    plt.title('ERL Performance Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot population size over time
    plt.subplot(2, 1, 2)
    plt.plot(erl_system.population_size_history, label='Population Size', color='purple')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Agents')
    plt.title('ERL Population Size')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
