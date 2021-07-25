# DRL_navigation
Homework project for RL navigation

## Introduction
For this project, an agent has been trained to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.
Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.
Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.
The agent has been trained until it gets an average score > +15 over 100 consecutive episodes.

[![Trained Agent](https://www.youtube.com/watch?v=KBI7N-bWsrs/0.jpg)](https://www.youtube.com/watch?v=KBI7N-bWsrs)

## Getting Started
1. Download the environment from one of the links below. You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. Place the file in the `unity_env/` folder, and unzip (or decompress) the file.
3. Install the pre-requisites by following the instructions in `Navigation,ipynb`.

### Instructions
Follow the instructions in `Navigation.ipynb` to get started with training the agent!
