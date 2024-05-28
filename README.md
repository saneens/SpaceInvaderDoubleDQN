# SpaceInvaderDoubleDQN

# Double DQN for Space Invaders

This repository contains an implementation of the Double Deep Q-Network (Double DQN) algorithm to train an agent to play the Atari game Space Invaders using TensorFlow and Keras. This was done as part of the Qualcomm 2020 AI Hackathon.

## Overview

The Double DQN algorithm is an extension of the Deep Q-Network (DQN) algorithm, which addresses the overestimation issue present in regular DQN. This implementation follows the approach outlined in the paper ["Deep Reinforcement Learning with Double Q-learning"](https://arxiv.org/abs/1509.06461) by Hado van Hasselt, Arthur Guez, and David Silver.

## Demo

Here's a GIF showcasing the trained agent playing the Space Invaders game:

[![Space Invaders Gameplay]()](spaceinvader.mp4)
[![Watch the video](spaceinvaderdemo.jpg)](path/to/your/video.mp4)


## Features

- Preprocess Atari frames by grayscaling, downsampling, and stacking 4 consecutive frames
- Deep convolutional neural network with 3 conv layers and 2 fully connected layers as the Q-network
- Double DQN implementation to address overestimation
- Experience replay with a replay memory of 1 million transitions
- Epsilon-greedy exploration strategy with decaying epsilon
- Tunable hyperparameters like batch size, learning rate, optimizer, target network update frequency, and loss function
- Integration with Weights & Biases for experiment tracking and visualization

## Requirements

- TensorFlow
- Keras
- OpenAI Gym
- OpenAI Gym Atari Environment
- Numpy
- Matplotlib (for visualization)
- Weights & Biases (optional, for experiment tracking)

## Potential Improvements

- Prioritized experience replay to focus on important transitions
- Increase replay memory capacity beyond 1 million transitions
- Add additional wrappers or modifications to the environment
