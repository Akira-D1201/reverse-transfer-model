import numpy as np
import matplotlib.pyplot as plt

# =============================================
# Model Name: L3-to-L1 Reverse Transfer (v2 - Enhanced L1 Drop)
# Based on: BIA+ Lexical Competition with Sustained L2 Presence
# =============================================
#
# 【Scientific Question】
# Can we demonstrate a clearer drop in L1 activation after L3 learning,
# while maintaining L2 visibility throughout the simulation?
#
# 【Key Adjustments】
# - inhibition_strength lowered to 0.04 (prevents L2 from vanishing instantly)
# - L3_inhibition_weight kept at 1.8 (strong targeted inhibition on L1)
# This ensures L1 receives sustained inhibition from both L2 and L3,
# resulting in a steeper and more visible decline in Scenario B.

def run_simulation(L1_base, L2_base, L3_base, L3_inhibition_weight=1.0):
    """
    Run a single lexical competition simulation.
    """
    time_steps = 50
    decay_rate = 0.1
    inhibition_strength = 0.04   # Reduced to keep L2 alive longer

    L1_act = np.zeros(time_steps)
    L2_act = np.zeros(time_steps)
    L3_act = np.zeros(time_steps)

    L1_act[0] = L1_base
    L2_act[0] = L2_base
    L3_act[0] = L3_base

    for t in range(1, time_steps):
        decay_L1 = decay_rate * L1_act[t-1]
        decay_L2 = decay_rate * L2_act[t-1]
        decay_L3 = decay_rate * L3_act[t-1]

        # L1 receives inhibition from L2 (normal) and L3 (weighted)
        inhibition_on_L1 = inhibition_strength * (L2_act[t-1] + L3_act[t-1] * L3_inhibition_weight)
        # L2 receives inhibition from L1 and L3 (normal)
        inhibition_on_L2 = inhibition_strength * (L1_act[t-1] + L3_act[t-1])
        # L3 receives inhibition from L1 and L2 (normal)
        inhibition_on_L3 = inhibition_strength * (L1_act[t-1] + L2_act[t-1])

        L1_act[t] = L1_act[t-1] - decay_L1 - inhibition_on_L1
        L2_act[t] = L2_act[t-1] - decay_L2 - inhibition_on_L2
        L3_act[t] = L3_act[t-1] - decay_L3 - inhibition_on_L3

        L1_act[t] = max(0, L1_act[t])
        L2_act[t] = max(0, L2_act[t])
        L3_act[t] = max(0, L3_act[t])

    return L1_act, L2_act, L3_act

# =============================================
# Scenario A: Before L3 learning
# =============================================
L1_act_A, L2_act_A, L3_act_A = run_simulation(
    L1_base=0.8,
    L2_base=0.4,
    L3_base=0.1,       # L3 barely present
    L3_inhibition_weight=1.0
)

# =============================================
# Scenario B: After L3 learning (L3 stronger, targeted inhibition)
# =============================================
L1_act_B, L2_act_B, L3_act_B = run_simulation(
    L1_base=0.8,
    L2_base=0.4,       # kept identical to Scenario A
    L3_base=0.4,       # L3 baseline increased
    L3_inhibition_weight=1.8   # enhanced inhibition specifically on L1
)

# =============================================
# Plot comparison
# =============================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(L1_act_A, label='L1 (Chinese)', linewidth=2, color='black')
plt.plot(L2_act_A, label='L2 (English)', linewidth=2, color='blue')
plt.plot(L3_act_A, label='L3 (Japanese)', linewidth=2, color='red')
plt.title('Before L3 Learning')
plt.xlabel('Time Steps')
plt.ylabel('Activation')
plt.legend()
plt.ylim(0, 0.9)

plt.subplot(1, 2, 2)
plt.plot(L1_act_B, label='L1 (Chinese)', linewidth=2, color='black')
plt.plot(L2_act_B, label='L2 (English)', linewidth=2, color='blue')
plt.plot(L3_act_B, label='L3 (Japanese)', linewidth=2, color='red')
plt.title('After L3 Learning (Enhanced L1 Drop)')
plt.xlabel('Time Steps')
plt.ylabel('Activation')
plt.legend()
plt.ylim(0, 0.9)

plt.tight_layout()
plt.savefig('reverse_transfer_v2.png', dpi=150)
plt.show()
input("Press Enter to close...")