# Atari Pong DDQN with PyTorch

This project implements a Double Deep Q-Network (DDQN) to play the Atari Pong game using OpenAI Gym and PyTorch. It includes environment setup, preprocessing, a convolutional neural network for policy learning, and replay buffer for experience replay.

## Installation

To run this project, you need to install the required libraries. Follow these steps to set up the environment:

1. Install OpenAI Gym with Atari support:
    ```bash
    pip install gym[atari]
    pip install gym[atari,accept-rom-license]
    pip install ale-py
    ```

2. Download Atari ROMs:
    ```bash
    mkdir /content/roms
    AutoROM --accept-license --install-dir /content/roms
    ```

3. Additional required libraries:
    ```bash
    pip install torch torchvision numpy matplotlib
    ```

## Environment Setup

The Pong environment is created using OpenAI Gym, and the observations are preprocessed using custom wrappers. Frames are processed by converting them to grayscale and resizing them to 80x80 pixels, while consecutive frames are merged to provide better context to the agent.

```python
def make_env(env):
    env = MergeFrame(env, merge=4)
    env = ProcessFrame(env)
    env = FrameStack(env, num_stack=4, new_step_api=True)
    return env
```

## Preprocessing

    FrameStack: Keeps track of 4 stacked frames to maintain temporal information.
    ProcessFrame: Converts RGB frames to grayscale, crops, and resizes them.
    MergeFrame: Merges consecutive frames to highlight motion.

## Model Architecture

The convolutional neural network (CNN) used for Pong is defined in PongCNN. It has three convolutional layers followed by two fully connected layers. Dropout and batch normalization are used to stabilize learning.

```python

class PongCNN(nn.Module):
    def __init__(self, action_space, num_stacked_frames=4, dropout_probability=0.1):
        # Define convolutional layers and fully connected layers
        pass
```

## Training Procedure

The agent is trained using DDQN with experience replay. The replay buffer stores transitions (state, action, reward, next_state, done) and uses mini-batches to update the network.
Hyperparameters

    num_episodes: 800
    batch_size: 64
    gamma: 0.99
    epsilon_start: 1.0
    epsilon_end: 0.01
    epsilon_decay: 150
    target_update: Every 4 episodes

## Loss Computation

The temporal difference (TD) loss is computed between the predicted Q-values and the target Q-values from the target network.

python

def compute_td_loss(policy_net, target_net, experiences, gamma, device):
    # Compute TD loss for DDQN
    pass

## Testing and Visualization

After training, you can visualize the rewards obtained by the agent during testing using a plotting function:

python

def test_and_plot_rewards(env, policy_net, n_episodes):
    # Test the trained policy and plot the rewards per episode
    pass

## Result

The agent showed promising results climbing up in rewards steadily over time with it being capable of competing with the automated player towards the end of the training. With more training time time model would probably perform even better.

## References

    OpenAI Gym: https://gym.openai.com/
    PyTorch: https://pytorch.org/
