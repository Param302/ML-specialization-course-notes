## Ch 2: State-action value function (Q Function)

## Introduction
When we start developing reinforcement learning algorithms, a key quantity will be the **state-action value function**, typically denoted as **Q**. Let’s break down what this function is and why it is crucial.

---

## What is the State-Action Value Function?

The **state-action value function** (or Q function) is a function that takes:
- A **state** `s`
- An **action** `a`

It outputs a number `Q(s, a)`, which represents the **expected return** if:
1. You start in state `s`.
2. Take the action `a` just once.
3. Behave optimally thereafter (take actions that yield the highest possible return).

---

### Circular Definition and Resolution
At first, this definition may seem circular:
- How can we compute `Q(s, a)` if we don’t know the optimal behavior yet?
- Why compute `Q(s, a)` if we already know the optimal policy?

This circularity will be resolved later when we explore specific reinforcement learning algorithms.

---

## Example: Mars Rover Problem

### Problem Setup
Consider a policy where:
- Go **left** from states `2, 3, 4`.
- Go **right** from state `5`.

This policy is optimal when the discount factor  $\gamma = 0.5$.

### Calculating `Q(s, a)`
Let’s calculate `Q(s, a)` for a few states and actions.

#### Example 1: `Q(2, right)`
- Start in state `2`, take the action **right**:
  - You reach state `3`.
  - Follow the optimal policy: $3 \to 2 \to 1 \to 100$.
- Return:
  - `0` (state 2) + `0.5 x 0` (state 3) + `0.5^2 x 0` (state 2) + `0.5^3 x 100 = 12.5`.

#### Example 2: `Q(2, left)`
- Start in state `2`, take the action **left**:
  - You reach the terminal state and receive `100`.
- return:
  - `0 + 0.5 x 100 = 50`.

#### Example 3: `Q(4, left)`
- Start in state `4`, take the action **left**:
  - Follow the optimal policy: $4 \to 3 \to 2 \to 1 \to 100$.
- Rewards:
  - `0 + 0.5 \cdot 0 + 0.5^2 \cdot 0 + 0.5^3 \cdot 100 = 12.5`.

### Summary Table
| State | Action | `Q(s, a)` |
|-------|--------|--------------|
| 2     | Left   | 50           |
| 2     | Right  | 12.5         |
| 4     | Left   | 12.5         |
| 4     | Right  | 10           |  
  
if you were to carry out this exercise for all of the other states and all of the other actions, you end up with:  

<img src="./images/q%20function.png" alt="Mars rover">  

---

## Policy from `Q(s, a)`

Once we compute `Q(s, a)` for all states and actions:
1. In each state `s`, choose the action `a` that maximizes `Q(s, a)`.
2. This defines the optimal policy $\pi(s)$.

For example:
- In state `4`, compare:
  - `Q(4, left) = 12.5`
  - `Q(4, right) = 10`
- Optimal action: **left**.

---

## Key Insights
1. **Optimal Return**: The best possible return from a state `s` is:
   $$
   \max_a Q(s, a)
   $$
2. **Optimal Action**: The optimal action $\pi(s)$ is:
   $$
   \arg\max_a Q(s, a)
   $$

---

## Terminology
- The **state-action value function** is often denoted as:
  - `Q(s, a)` or $Q^*(s, a)$.
  - $Q^*(s, a)$: Refers to the optimal Q function.
- These terms are used interchangeably in the reinforcement learning literature.

---

## The Bellman Equation

### Notation
To describe the Bellman Equation, the following notations are used:  
- ` S `: The current state.  
- ` R(S) `: The reward of the current state.  
- ` A `: The current action taken in state ` S `.  
- ` S' `: The next state after taking action ` A ` from ` S `.  
  - Example:  
    - Starting in ` State 4 ` and taking action `left`  leads to ` S' = State 3 `.  
- ` A' `: The action taken in ` S' ` (the next state).  

---

### The Bellman Equation
The Bellman Equation is as follows:  

$$
Q(S, A) = R(S) + \gamma \cdot \max_{A'} Q(S', A')
$$

Where:  
- ` R(S) `: Reward of the current state ` S `.  
- `γ`: Discount factor (e.g., ` γ = 0.5 `).  
- ` \max_{A'} Q(S', A') `: The maximum value of ` Q ` over all possible actions ` A' ` in the next state ` S' `.

---

### Example Calculations

#### Example 1: ` Q(2, right) `  
1. Current state: ` S = State 2 `.  
2. Current action: ` A = right `.  
3. Next state: ` S' = State 3 `.  
4. Rewards:  
   - $R(State 2) = 0$.  
   - $Q(State 3, A') = \max(25, 6.25) = 25$.  

Using the Bellman Equation:  

$$
Q(2, \text{right}) = R(2) + \gamma \cdot \max_{A'} Q(3, A')
$$

Substitute values:  
$$
Q(2, \text{right}) = 0 + 0.5 \cdot 25 = 12.5
$$

#### Example 2: ` Q(4, left) `  
1. Current state: ` S = State 4 `.  
2. Current action: ` A = left `.  
3. Next state: ` S' = State 3 `.  
4. Rewards:  
   - $R(\text{State 4}) = 0$.  
   - $Q(\text{State 3}, A') = \max(25, 6.25) = 25$.  

Using the Bellman Equation:  

$$
Q(4, \text{left}) = R(4) + \gamma \cdot \max_{A'} Q(3, A')
$$

Substitute values:  
$$
Q(4, \text{left}) = 0 + 0.5 \cdot 25 = 12.5
$$

#### Terminal States
In a **terminal state**, the Bellman Equation simplifies to:  
$$
Q(S, A) = R(S)
$$
This is because there's no ` S' `, so the second term disappears.

---

### Key Takeaways
1. **Definition Recap**:  
   $$
   Q(S, A) = R(S) + \gamma \cdot \max_{A'} Q(S', A')
   $$
   The total return consists of two parts:  
   - Immediate reward: ` R(S) `.  
   - Discounted future return: $\gamma \cdot \max_{A'} Q(S', A')$.  

2. **High-Level Intuition**:  
   - The **total return** in a reinforcement learning problem can be decomposed into:  
     - **Immediate reward**: $R(S)$. 
     - **Future return**: $\gamma \cdot \max_{A'} Q(S', A')$.  
   - The Bellman Equation captures this decomposition.

3. **Practical Note**:  
   Even if the Bellman Equation feels complex, you can still apply it systematically to compute values.

---

# Stochastic Environments in Reinforcement Learning

In many applications, the actions you take may not produce reliable or deterministic outcomes. For instance, when commanding a Mars rover to move left, environmental factors like slippery terrain or obstacles might cause it to slip or move in an unintended direction. Many robots face this challenge due to external influences like wind, uneven surfaces, or mechanical limitations.

This situation can be modeled using a **stochastic environment**, which is a generalization of the reinforcement learning (RL) framework. Let’s explore this using a simplified Mars rover example:

---

## Stochastic Behavior of Actions

Suppose your rover is in a grid with six states. If you command it to go **left**:
- There is a **90% probability (0.9)** that it will move to the intended state.
- There is a **10% probability (0.1)** that it will slip and move in the opposite direction.

For example:
- **In state 3**, commanding "left" has:
  - A **90% chance** of moving to state 2.
  - A **10% chance** of moving to state 4 instead.

Similarly:
- **Commanding "right"** in state 3:
  - Has a **90% chance** of moving to state 4.
  - Has a **10% chance** of moving to state 2.

This randomness makes the environment **stochastic**.

---

## Policies and Outcomes

Let’s consider a policy that specifies actions:
- Go **left** in states 2, 3, and 4.
- Go **right** in state 5.

If the rover starts in **state 4**, the sequence of states it visits will depend on the outcomes of its actions:
1. **First attempt**:
   - Command "left": Success! The rover moves to state 3.
   - Command "left" again: Success! It moves to state 2.
   - Command "left" again: Success! It moves to state 1 and collects the reward.

   Sequence: `4 → 3 → 2 → 1`, with rewards `0, 0, 0, 100`.

2. **Second attempt**:
   - Command "left": Success! The rover moves to state 3.
   - Command "left" again: Failure! It slips and moves back to state 4.
   - Command "left" again: Success! It moves to state 3, and so on.

   Sequence: `4 → 3 → 4 → 3 → 2 → 1`, with rewards `0, 0, 0, 0, 100`.

3. **Third attempt**:
   - Command "left": Failure! It slips and moves to state 5.
   - Command "right": Success! It moves to state 6.

   Sequence: `4 → 5 → 6`, with rewards `0, 0, 40`.

---

## Expected Return in Stochastic Environments

In a stochastic reinforcement learning problem:
- The **sequence of rewards** is random because the outcome of each action is uncertain.
- Instead of maximizing a single return, we focus on **maximizing the expected return**:
  - The average of the **sum of discounted rewards** over many trials.

Mathematically, the expected return is:
$$
\mathbb{E}[R_1 + \gamma R_2 + \gamma^2 R_3 + \dots]
$$
where:
- $\mathbb{E}$ denotes the expected value (average over all possible outcomes).
- $R_t$ is the reward at time $t$.
- $\gamma$ is the discount factor.

---

## The Bellman Equation in Stochastic Environments

In deterministic environments, the Bellman equation is:
$$
V(s) = R(s, a) + \gamma V(s')
$$
where `s'` is the next state after taking action `a` in state `s`.

In stochastic environments:
- The next state `s'` is **random**, so we take the **expected value** over all possible next states:
$$
V(s) = R(s, a) + \gamma \mathbb{E}_{s'}[V(s')]
$$
This accounts for the uncertainty in the transition from `s` to `s'`.

---

## Practical Example: Mars Rover Misstep Probability

Let’s define a **misstep probability**:
- `p = 0.1`: The rover slips 10% of the time.

If you follow the optimal policy:
- The **optimal return** will decrease as `p` increases because the rover’s control becomes less reliable.
- For example:
  - At `p = 0.1`, the optimal return is slightly reduced.
  - At `p = 0.4`, the optimal return drops significantly because the rover follows commands correctly only 60% of the time.

### Experiment:
You can simulate this by adjusting the misstep probability in a reinforcement learning lab or notebook. Observe how:
- The **expected return** changes.
- The **Q-values** (state-action values) decrease as control reliability diminishes.

---