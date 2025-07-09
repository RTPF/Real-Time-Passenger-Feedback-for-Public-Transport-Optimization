# -*- coding: utf-8 -*-
"""
Created on Sat Jun 7 2025
@author: Raul - Bus Frequency Optimization based on User Feedback

Context: 500 post-trip surveys to optimize bus frequencies
- route_id: Route used by the passenger
- sentiment: Comment evaluation (Negative, Neutral, Positive)
- avg_satisfaction: Satisfaction from 1 (very dissatisfied) to 5 (very satisfied)
- avg_wait_time_rating: Wait time from 1 (very long) to 5 (very short)
- proportion_positive: Proportion of positive sentiments per route

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io


raw_data = pd.read_csv("metrics_by_route.csv")

# Calculate aggregated metrics per route
metrics_agg = raw_data.groupby('route_id').agg(
    avg_satisfaction=('avg_satisfaction', 'mean'),
    avg_wait_time_rating=('avg_wait_time_rating', 'mean')
).reset_index()

# Calculate the proportion of positive sentiments per route
total_counts = raw_data.groupby('route_id').size()
positive_counts = raw_data[raw_data['sentiment'] == 'Positivo'].groupby('route_id').size()
proportion_positive_sr = (positive_counts / total_counts).fillna(0)
proportion_positive_sr.name = 'proportion_positive'

# Merge everything into a single DataFrame
metrics = pd.merge(metrics_agg, proportion_positive_sr, on='route_id')

print("\n--- Aggregated and Calculated Metrics per Route ---")
print("Based on 500 post-trip user surveys of public transportation")
print(metrics)
print("-" * 50)

# --- STEP 1: READ AND NORMALIZE DATA ---
routes = metrics["route_id"].values
initial_satisfaction = metrics.set_index("route_id")["avg_satisfaction"].to_dict()
initial_wait_time = metrics.set_index("route_id")["avg_wait_time_rating"].to_dict()
proportion_positive = metrics.set_index("route_id")["proportion_positive"].to_dict()

# Normalize metrics (scales 1-5 to 0-1)
initial_satisfaction = {k: (v - 1) / 4 for k, v in initial_satisfaction.items()}
initial_wait_time = {k: (v - 1) / 4 for k, v in initial_wait_time.items()}

# --- STEP 2: DEFINE THE SIMPLIFIED ENVIRONMENT ---
class SimpleTransportEnv:
    def __init__(self, routes):
        self.routes = routes
        self.satisfaction = {route: initial_satisfaction[route] for route in routes}
        self.wait_time = {route: initial_wait_time[route] for route in routes}
        self.prop_positive = {route: proportion_positive[route] for route in routes}
        self.frequency = {route: 15.0 for route in routes}  # Initial frequency: 15 min

    def reset(self):
        self.satisfaction = {route: initial_satisfaction[route] for route in self.routes}
        self.wait_time = {route: initial_wait_time[route] for route in self.routes}
        self.frequency = {route: 15.0 for route in self.routes}
        return {route: np.array([self.satisfaction[route], self.wait_time[route], self.prop_positive[route]])
                for route in self.routes}

    def step(self, action_dict):
        obs, rewards, infos = {}, {}, {}
        for route in self.routes:
            action = action_dict[route]
            # Simulate the effect of the action
            if action == 0:  # Reduce frequency
                self.frequency[route] += 2
                self.wait_time[route] += 0.1
                self.satisfaction[route] -= 0.05
            elif action == 2:  # Increase frequency
                self.frequency[route] -= 2
                self.wait_time[route] -= 0.1
                self.satisfaction[route] += 0.05

            # Ensure limits
            self.frequency[route] = np.clip(self.frequency[route], 5, 30)
            self.satisfaction[route] = np.clip(self.satisfaction[route], 0, 1)
            self.wait_time[route] = np.clip(self.wait_time[route], 0, 1)

            # Observation and RAW reward
            obs[route] = np.array([self.satisfaction[route], self.wait_time[route], self.prop_positive[route]])
            raw_reward = (self.satisfaction[route] * (1 + self.prop_positive[route])) - self.wait_time[route]
            
            # NORMALIZE REWARD for better interpretation
            # Theoretical range: [-1, 2] -> Normalize to [0, 1]
            rewards[route] = (raw_reward + 1) / 3  # Normalization to 0-1 scale
            
            infos[route] = {
                "action": ["reduce", "maintain", "increase"][action],
                "raw_reward": raw_reward
            }
        return obs, rewards, infos

# --- STEP 3: ITERATIVE OPTIMIZATION ALGORITHM ---
def optimize_frequencies(env, max_iterations=10000):
    """
    Q-learning optimization algorithm that iteratively improves decisions
    based on real feedback from 500 public transport users.
    """
    
    # Initialize simple Q-table (route -> action -> value)
    q_table = {}
    for route in env.routes:
        q_table[route] = {0: 0.0, 1: 0.0, 2: 0.0}  # 3 possible actions
    
    # Learning parameters
    learning_rate = 0.1
    epsilon = 1.0  # High initial exploration
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    # Reward history for analysis
    reward_history = []
    raw_reward_history = []
    
    print(f"\n--- STARTING OPTIMIZATION ({max_iterations} iterations) ---")
    print("Optimizing frequencies based on user feedback...")
    
    for iteration in range(max_iterations):
        # Reset environment
        state = env.reset()
        
        # Select actions using epsilon-greedy
        action_dict = {}
        for route in env.routes:
            if np.random.random() < epsilon:
                # Explore: random action
                action_dict[route] = np.random.choice([0, 1, 2])
            else:
                # Exploit: best known action
                best_action = max(q_table[route], key=q_table[route].get)
                action_dict[route] = best_action
        
        # Execute actions and get rewards
        obs, rewards, infos = env.step(action_dict)
        
        # Update Q-table with the obtained rewards
        for route in env.routes:
            action = action_dict[route]
            reward = rewards[route]
            # Simple Q-learning update
            q_table[route][action] = q_table[route][action] + learning_rate * (reward - q_table[route][action])
        
        # Calculate total rewards (normalized and raw)
        total_reward = sum(rewards.values())
        total_raw_reward = sum(infos[route]["raw_reward"] for route in env.routes)
        
        reward_history.append(total_reward)
        raw_reward_history.append(total_raw_reward)
        
        # Epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Print progress every 100 iterations
        if (iteration + 1) % 100 == 0:
            avg_reward_last_100 = np.mean(reward_history[-100:])
            avg_raw_reward_last_100 = np.mean(raw_reward_history[-100:])
            print(f"Iteration {iteration + 1:4d}: "
                  f"Normalized Reward = {total_reward:.4f} (Avg-100: {avg_reward_last_100:.4f}), "
                  f"Raw Reward = {total_raw_reward:.4f} (Avg-100: {avg_raw_reward_last_100:.4f}), "
                  f"Epsilon = {epsilon:.3f}")
    
    # Finally, execute with the best learned actions
    print("\n--- APPLYING BEST LEARNED ACTIONS ---")
    final_state = env.reset()
    final_actions = {}
    for route in env.routes:
        final_actions[route] = max(q_table[route], key=q_table[route].get)
    
    final_obs, final_rewards, final_infos = env.step(final_actions)
    
    # Prepare output data
    output_data = []
    for route in env.routes:
        output_data.append({
            "route_id": route,
            "initial_satisfaction": initial_satisfaction[route],
            "initial_wait_time": initial_wait_time[route],
            "proportion_positive": proportion_positive[route],
            "action_taken": final_infos[route]["action"],
            "adjusted_frequency": env.frequency[route],
            "simulated_satisfaction": env.satisfaction[route],
            "simulated_wait_time": env.wait_time[route],
            "final_reward_normalized": final_rewards[route],
            "final_reward_raw": final_infos[route]["raw_reward"],
            "q_values": dict(q_table[route])
        })
    
    return output_data, reward_history, raw_reward_history

# --- STEP 4: EXECUTE AND SAVE RESULTS ---
env = SimpleTransportEnv(routes)
output_data, reward_history, raw_reward_history = optimize_frequencies(env, max_iterations=10000)
output_df = pd.DataFrame(output_data).sort_values(by="final_reward_normalized", ascending=False).reset_index(drop=True)

print("\n--- OPTIMIZATION RESULTS ---")
print("Best actions learned based on 500 user feedback:")
print(output_df[['route_id', 'action_taken', 'final_reward_normalized', 'final_reward_raw', 'adjusted_frequency']])

# Save results
output_df.to_csv("adjusted_frequencies_optimized.csv", index=False)
print(f"\nFile 'adjusted_frequencies_optimized.csv' saved successfully.")

# --- STEP 5: COMPREHENSIVE RESULTS VISUALIZATION (SEPARATE PLOTS) ---
print("\n--- Generating Comprehensive Results Graphs (Separate Plots) ---")

# Configure style
plt.style.use('seaborn-v0_8-whitegrid')
output_df_sorted = output_df.sort_values('route_id').reset_index()

# =============================================================================
# GRAPH 1: Satisfaction Comparison (Before and After)
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 7))
index = np.arange(len(output_df_sorted['route_id']))
bar_width = 0.35

bar1 = ax.bar(index - bar_width/2, output_df_sorted['initial_satisfaction'], bar_width,
              label='Initial Satisfaction', color='skyblue', alpha=0.8)
bar2 = ax.bar(index + bar_width/2, output_df_sorted['simulated_satisfaction'], bar_width,
              label='Optimized Satisfaction', color='darkblue', alpha=0.8)

ax.set_xlabel('Route ID', fontweight='bold')
ax.set_ylabel('Normalized Satisfaction (0-1)', fontweight='bold')
ax.set_title('Satisfaction Comparison by Route (Before vs After)\nBased on 500 User Feedback',
             fontweight='bold', fontsize=16)
ax.set_xticks(index)
ax.set_xticklabels(output_df_sorted['route_id'])
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("comparacion_satisfaccion.svg", format="svg", bbox_inches='tight')
print("Graph 'comparacion_satisfaccion.svg' saved.")
plt.show()
plt.close()

# =============================================================================
# GRAPH 2: Frequency Comparison (Before and After)
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 7))
initial_frequency = [15] * len(output_df_sorted)

bar1 = ax.bar(index - bar_width/2, initial_frequency, bar_width,
              label='Initial Frequency (min)', color='lightcoral', alpha=0.8)
bar2 = ax.bar(index + bar_width/2, output_df_sorted['adjusted_frequency'], bar_width,
              label='Optimized Frequency (min)', color='firebrick', alpha=0.8)

ax.set_xlabel('Route ID', fontweight='bold')
ax.set_ylabel('Frequency (minutes)', fontweight='bold')
ax.set_title('Change in Bus Frequency by Route\nOptimization based on Post-Trip Surveys',
             fontweight='bold', fontsize=16)
ax.set_xticks(index)
ax.set_xticklabels(output_df_sorted['route_id'])
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("cambio_frecuencia.svg", format="svg", bbox_inches='tight')
print("Graph 'cambio_frecuencia.svg' saved.")
plt.show()
plt.close()

# =============================================================================
# GRAPH 3: Final Reward and Action Taken
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 7))
palette = {'increase': 'forestgreen', 'maintain': 'gold', 'reduce': 'orangered'}

sns.barplot(x='route_id', y='final_reward_normalized', hue='action_taken',
            data=output_df_sorted, palette=palette, dodge=False, ax=ax)

ax.set_xlabel('Route ID', fontweight='bold')
ax.set_ylabel('Normalized Final Reward (0-1)', fontweight='bold')
#ax.set_title('Final Reward by Route and Action Taken\nBased on User Satisfaction Metrics',
            # fontweight='bold', fontsize=16)
ax.legend(title='Action Taken')
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("recompensa_final_por_ruta.svg", format="svg", bbox_inches='tight')
print("Graph 'recompensa_final_por_ruta.svg' saved.")
plt.show()
plt.close()

# =============================================================================
# GRAPH 4: Learning Progress (Normalized and Raw Reward)
# =============================================================================
# --- Plot 4a: Normalized Reward per Iteration ---
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(reward_history, alpha=0.6, color='blue', linewidth=0.8)
window_size = 50
if len(reward_history) >= window_size:
    moving_avg = pd.Series(reward_history).rolling(window=window_size).mean()
    ax.plot(moving_avg, color='red', linewidth=2, label=f'Moving Average ({window_size})')
    ax.legend()
ax.set_title('Progress: Normalized Reward per Iteration', fontweight='bold')
ax.set_xlabel('Iteration')
ax.set_ylabel('Normalized Reward (0-1)')
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("progreso_recompensa_normalizada.svg", format="svg", bbox_inches='tight')
print("Graph 'progreso_recompensa_normalizada.svg' saved.")
plt.show()
plt.close()

# --- Plot 4b: Raw Reward per Iteration ---
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(raw_reward_history, alpha=0.6, color='green', linewidth=0.8)
if len(raw_reward_history) >= window_size:
    moving_avg_raw = pd.Series(raw_reward_history).rolling(window=window_size).mean()
    ax.plot(moving_avg_raw, color='orange', linewidth=2, label=f'Moving Average ({window_size})')
    ax.legend()
ax.set_title('Progress: Raw Reward per Iteration', fontweight='bold')
ax.set_xlabel('Iteration')
ax.set_ylabel('Raw Reward')
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("progreso_recompensa_cruda.svg", format="svg", bbox_inches='tight')
print("Graph 'progreso_recompensa_cruda.svg' saved.")
plt.show()
plt.close()

# --- Plot 4c: Distribution of Actions Taken ---
fig, ax = plt.subplots(figsize=(8, 8))
action_counts = output_df['action_taken'].value_counts()
colors = {'increase': 'forestgreen', 'maintain': 'gold', 'reduce': 'orangered'}
action_colors = [colors[action] for action in action_counts.index]
wedges, texts, autotexts = ax.pie(action_counts.values, labels=action_counts.index,
                                  autopct='%1.1f%%', colors=action_colors, startangle=90)
ax.set_title('Distribution of Optimized Actions\n(Based on User Feedback)', fontweight='bold')
fig.tight_layout()
plt.savefig("distribucion_acciones.svg", format="svg", bbox_inches='tight')
print("Graph 'distribucion_acciones.svg' saved.")
plt.show()
plt.close()

# --- Plot 4d: Wait Time Comparison ---
fig, ax = plt.subplots(figsize=(14, 7))
bar1 = ax.bar(index - bar_width/2, output_df_sorted['initial_wait_time'], bar_width,
              label='Initial Wait Time', color='lightsteelblue', alpha=0.8)
bar2 = ax.bar(index + bar_width/2, output_df_sorted['simulated_wait_time'], bar_width,
              label='Optimized Wait Time', color='navy', alpha=0.8)
ax.set_xlabel('Route ID')
ax.set_ylabel('Normalized Wait Time Rating (0-1)')
ax.set_title('Improvement in Perceived Wait Time', fontweight='bold')
ax.set_xticks(index)
ax.set_xticklabels(output_df_sorted['route_id'])
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("comparacion_tiempo_espera.svg", format="svg", bbox_inches='tight')
print("Graph 'comparacion_tiempo_espera.svg' saved.")
plt.show()
plt.close()

# =============================================================================
# GRAPH 5: Correlation Analysis
# =============================================================================
# --- Plot 5a: Positive Proportion vs Final Reward ---
fig, ax = plt.subplots(figsize=(8, 6))
colors = {'increase': 'forestgreen', 'maintain': 'gold', 'reduce': 'orangered'}
ax.scatter(output_df['proportion_positive'], output_df['final_reward_normalized'],
           c=[colors[action] for action in output_df['action_taken']],
           s=100, alpha=0.7, edgecolors='black')
ax.set_xlabel('Proportion of Positive Sentiments', fontweight='bold')
ax.set_ylabel('Normalized Final Reward', fontweight='bold')
#ax.set_title('Relationship: Positive Sentiments vs Final Reward', fontweight='bold')
ax.grid(True, alpha=0.3)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[action], label=action.capitalize())
                   for action in colors.keys()]
ax.legend(handles=legend_elements, title='Action Taken')
fig.tight_layout()
plt.savefig("correlacion_sentimientos_recompensa.svg", format="svg", bbox_inches='tight')
print("Graph 'correlacion_sentimientos_recompensa.svg' saved.")
plt.show()
plt.close()

# --- Plot 5b: Initial vs Final Satisfaction ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(output_df['initial_satisfaction'], output_df['simulated_satisfaction'],
           c=[colors[action] for action in output_df['action_taken']],
           s=100, alpha=0.7, edgecolors='black')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No Change Line')
ax.set_xlabel('Normalized Initial Satisfaction', fontweight='bold')
ax.set_ylabel('Normalized Final Satisfaction', fontweight='bold')
#ax.set_title('Comparison: Initial vs Final Satisfaction', fontweight='bold')
legend_elements = [Patch(facecolor=colors[action], label=action.capitalize())
                   for action in colors.keys()] + [
    Patch(facecolor='none', edgecolor='black', linestyle='--', label='No Change')]
ax.legend(handles=legend_elements, title='Reference')
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("correlacion_satisfaccion_inicial_final.svg", format="svg", bbox_inches='tight')
print("Graph 'correlacion_satisfaccion_inicial_final.svg' saved.")
plt.show()
plt.close()