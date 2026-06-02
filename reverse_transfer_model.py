# =============================================
# Model Name: A Minimal Model of L3-to-L1 Reverse Transfer in Trilingual Word Recognition
# Based on: Lexical Competition and Language Node Inhibition Mechanisms in the BIA+ Model
# =============================================
#
# 【Scientific Question】
# When a native Chinese speaker learning Japanese (L3) encounters a Japanese-Chinese
# homograph (e.g., "大学"), judging "whether it is a Chinese word" becomes more difficult.
# This model simulates: After L3 learning, increased inhibition from L3 lexical nodes
# onto L1 nodes leads to a decrease in L1 activation.
#
# 【Model Logic (Pseudocode)】
# 1. Create three lexical nodes: L1_word (Chinese), L2_word (English), L3_word (Japanese)
# 2. Set the initial activation (baseline) for each node:
#    - L1 is the highest (native language advantage)
#    - L2 is medium
#    - L3 is very low before learning, and increases after learning
# 3. Simulate the time course (e.g., 50 time steps):
#    At each time step:
#    a. All nodes undergo natural decay (activation decreases over time)
#    b. Nodes inhibit each other (competitive dynamics)
#    c. A "language node," based on the task instruction ("judge whether it is a Chinese word"),
#       exerts additional top-down inhibition on non-target languages (L2, L3)
# 4. Run two scenarios:
#    Scenario A: Before L3 learning (L3 baseline is very low, 0.1)
#    Scenario B: After L3 learning (L3 baseline increases to 0.4)
# 5. Output: A comparison of L1 activation curves over time for the two scenarios

import numpy as np
import matplotlib.pyplot as plt
# =============================================
# Model Name: A Minimal Model of L3-to-L1 Reverse Transfer in Trilingual Word Recognition
# Based on: Lexical Competition and Language Node Inhibition Mechanisms in the BIA+ Model
# =============================================
#
# 【Scientific Question】
# When a native Chinese speaker learning Japanese (L3) encounters a Japanese-Chinese
# homograph (e.g., "大学"), judging "whether it is a Chinese word" becomes more difficult.
# This model simulates: After L3 learning, increased inhibition from L3 lexical nodes
# onto L1 nodes leads to a decrease in L1 activation.
#
# 【Model Logic (Pseudocode)】
# 1. Create three lexical nodes: L1_word (Chinese), L2_word (English), L3_word (Japanese)
# 2. Set the initial activation (baseline) for each node:
#    - L1 is the highest (native language advantage)
#    - L2 is medium
#    - L3 is very low before learning, and increases after learning
# 3. Simulate the time course (e.g., 50 time steps):
#    At each time step:
#    a. All nodes undergo natural decay (activation decreases over time)
#    b. Nodes inhibit each other (competitive dynamics)
#    c. A "language node," based on the task instruction ("judge whether it is a Chinese word"),
#       exerts additional top-down inhibition on non-target languages (L2, L3)
# 4. Run two scenarios:
#    Scenario A: Before L3 learning (L3 baseline is very low, 0.1)
#    Scenario B: After L3 learning (L3 baseline increases to 0.4)
# 5. Output: A comparison of L1 activation curves over time for the two scenarios


def run_simulation(L1_base, L2_base, L3_base, L3_inhibition_weight=1.0):
    """
    Run a single lexical competition simulation.

    Parameters:
        L1_base: initial activation value for the L1 (Chinese) node
        L2_base: initial activation value for the L2 (English) node
        L3_base: initial activation value for the L3 (Japanese) node
        L3_inhibition_weight: multiplier for L3's inhibitory effect on L1
                              (>1 means L3 inhibits L1 more strongly after learning)

    Returns:
        L1_activation_over_time: array of L1 activation values over time steps
        L2_activation_over_time: array of L2 activation values over time steps
        L3_activation_over_time: array of L3 activation values over time steps
    """

    # Simulation parameters
    time_steps = 50  # Number of time steps to simulate
    decay_rate = 0.1  # Natural decay rate per time step
    inhibition_strength = 0.08  # Reduced to allow longer competition window

    # Initialize activation arrays for the three lexical nodes
    L1_act = np.zeros(time_steps)
    L2_act = np.zeros(time_steps)
    L3_act = np.zeros(time_steps)

    # Set initial activation (time step 0)
    L1_act[0] = L1_base
    L2_act[0] = L2_base
    L3_act[0] = L3_base

    # Simulate the time course
    for t in range(1, time_steps):
        # 1. Natural decay (activation decreases over time)
        decay_L1 = decay_rate * L1_act[t-1]
        decay_L2 = decay_rate * L2_act[t-1]
        decay_L3 = decay_rate * L3_act[t-1]

        # 2. Competitive inhibition from other lexical nodes
        # L1 receives inhibition from L2 and L3
        inhibition_on_L1 = inhibition_strength * (L2_act[t-1] + L3_act[t-1] * L3_inhibition_weight)
        # L2 receives inhibition from L1 and L3
        inhibition_on_L2 = inhibition_strength * (L1_act[t-1] + L3_act[t-1])
        # L3 receives inhibition from L1 and L2
        inhibition_on_L3 = inhibition_strength * (L1_act[t-1] + L2_act[t-1])

        # Update activation values
        L1_act[t] = L1_act[t-1] - decay_L1 - inhibition_on_L1
        L2_act[t] = L2_act[t-1] - decay_L2 - inhibition_on_L2
        L3_act[t] = L3_act[t-1] - decay_L3 - inhibition_on_L3

        # Ensure activation does not drop below 0
        L1_act[t] = max(0, L1_act[t])
        L2_act[t] = max(0, L2_act[t])
        L3_act[t] = max(0, L3_act[t])

    return L1_act, L2_act, L3_act


# =============================================
# Scenario A: Before L3 learning (L3 baseline is very low)
# =============================================
L1_act_A, L2_act_A, L3_act_A = run_simulation(
    L1_base=0.8,   # Chinese native language advantage
    L2_base=0.4,   # English moderate proficiency
    L3_base=0.1,   # Japanese just started, very low activation
    L3_inhibition_weight=1.0  # Before learning, L3 has normal inhibitory weight
)

# =============================================
# Scenario B: After L3 learning (L3 baseline increases, inhibition strengthens)
# =============================================
L1_act_B, L2_act_B, L3_act_B = run_simulation(
    L1_base=0.8,
    L2_base=0.4,   # Assuming L2 also slightly improves
    L3_base=0.4,   # Kept identical to Scenario A to isolate L3's effect
    L3_inhibition_weight=1.8  # After learning, L3's inhibitory ability on L1 is enhanced
)

# =============================================
# Plot comparison
# =============================================
plt.figure(figsize=(12, 5))

# Left panel: Before L3 learning
plt.subplot(1, 2, 1)
plt.plot(L1_act_A, label='L1 (Chinese)', linewidth=2, color='black')
plt.plot(L2_act_A, label='L2 (English)', linewidth=2, color='blue')
plt.plot(L3_act_A, label='L3 (Japanese)', linewidth=2, color='red')
plt.title('Before L3 Learning')
plt.xlabel('Time Steps')
plt.ylabel('Activation')
plt.legend()
plt.ylim(0, 0.9)

# Right panel: After L3 learning
plt.subplot(1, 2, 2)
plt.plot(L1_act_B, label='L1 (Chinese)', linewidth=2, color='black')
plt.plot(L2_act_B, label='L2 (English)', linewidth=2, color='blue')
plt.plot(L3_act_B, label='L3 (Japanese)', linewidth=2, color='red')
plt.title('After L3 Learning')
plt.xlabel('Time Steps')
plt.ylabel('Activation')
plt.legend()
plt.ylim(0, 0.9)

plt.tight_layout()
plt.savefig('reverse_transfer_model.png', dpi=150)
plt.show()
input("Press Enter to close...")


