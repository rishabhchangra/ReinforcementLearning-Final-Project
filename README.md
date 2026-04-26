# Reinforcement Learning Final Project: Atari Pong Benchmark

## Overview
This project benchmarks three Deep Q-Learning algorithms—Naive DQN, DQN, and Double DQN (DDQN)—on the Atari Pong environment. It demonstrates the evolution of reinforcement learning techniques, from a simple online Q-network to advanced experience replay and Double DQN strategies.

---

## Project Structure
```
ReinforcementLearning-Final-Project/
│
├── DQN/         # Standard DQN with Experience Replay
│   ├── main_dqn.py
│   ├── dqn_agent.py
│   ├── deep_q_network.py
│   ├── replay_memory.py
│   ├── utils.py
│   ├── models/
│   └── plots/
│
├── DDQN/        # Double DQN (Improved)
│   ├── main_ddqn.py
│   ├── ddqn_agent.py
│   ├── deep_q_network.py
│   ├── replay_memory.py
│   ├── utils.py
│   ├── models/
│   └── plots/
│
├── Naive/       # Baseline without Replay
│   ├── Naive_Approach.py
│   ├── Deep_Q_Network.py
│   ├── util.py
│   ├── models/
│   └── plots/
│
└── README.md
```

---

## Algorithms
### 1. Naive DQN (Naive/)
- **Single Q-network** (no target network)
- **No experience replay**; learns from each transition immediately
- Fast but unstable and poor performance

### 2. DQN (DQN/)
- **Separate evaluation and target networks**
- **Experience replay buffer** (10,000 transitions)
- Epsilon-greedy exploration
- Target network updated every 1,000 steps

### 3. Double DQN (DDQN/)
- **Reduces Q-value overestimation**
- Uses evaluation network for action selection, target network for evaluation
- Target network updated every 10 steps
- Most stable and best performance

---

## Neural Network Architecture
All agents use the same CNN:
- Input: (4, 84, 84) [4 stacked grayscale frames]
- Conv2d(4→32, 8x8, stride 4) + ReLU
- Conv2d(32→64, 4x4, stride 2) + ReLU
- Conv2d(64→64, 3x3, stride 1) + ReLU
- Flatten → FC(512) + ReLU → FC(n_actions)
- Optimizer: RMSprop | Loss: MSE

---

## Preprocessing Pipeline
1. Repeat action for 4 frames, take max pixel value
2. Grayscale conversion
3. Downsample to 84×84
4. Normalize to [0, 1]
5. Stack last 4 frames

---

## Training & Testing
### Training
```bash
cd <ALGORITHM_FOLDER>  # DQN, DDQN, or Naive
python main_*.py
```
- Models saved to `models/` after each episode
- Learning curves in `plots/`

### Testing
- Set `load_checkpoint = True` in the main file
- Uncomment `env.render()` to visualize
- Run the main file again

---

## Hyperparameters
| Parameter         | DQN/DDQN | Naive |
|-------------------|----------|-------|
| Episodes          | 250      | 250   |
| Batch Size        | 32       | 32    |
| Replay Memory     | 10,000   | —     |
| Epsilon (start)   | 1.0      | 1.0   |
| Epsilon (min)     | 0.1      | 0.1   |
| Epsilon Decay     | 1e-5     | 1e-6  |
| Discount (γ)      | 0.99     | 0.99  |
| Learning Rate     | 0.0001   | 0.00025|

---

## Dependencies
- torch
- gym
- numpy
- matplotlib
- opencv-python

Install with:
```bash
pip install torch gym numpy matplotlib opencv-python
```

---

## Algorithm Comparison
| Aspect              | DQN   | DDQN  | Naive |
|---------------------|-------|-------|-------|
| Experience Replay   | ✓     | ✓     | ✗     |
| Target Network      | ✓     | ✓     | ✗     |
| Overestimation Fix  | ✗     | ✓     | N/A   |
| Stability           | High  | Very High | Low |
| Memory Usage        | ~150MB| ~150MB| ~5MB  |
| Training Speed      | Medium| Medium| Fast  |
| Performance         | Good  | Best  | Poor  |

---

## Notes
- Always run scripts from the correct algorithm folder
- For testing, set `load_checkpoint = True` and uncomment `env.render()`
- Hyperparameters are at the top of each main file
- Models directory must exist before training
- GPU is used automatically if available

---

## Contact
For questions or contributions, please open an issue or pull request.

