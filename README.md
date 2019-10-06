[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

# Deep Reinforcement Learning : Collaboration and Competition using MADDPG
In this repository i used MADDPG algorithm https://arxiv.org/pdf/1706.02275.pdf to solve Collaboration and Competition , Udacity's 3rd project for Deep RL Nanodegree from the paper "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"

## Project's goal

![Trained Agent][image1]

In this project, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  **Thus, the goal of each agent is to keep the ball in play.**

The task is episodic, and in order to solve the environment, **the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents)**. Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores is at least +0.5.**

### Solving the environment

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 consecutive episodes) of those **scores** is at least +0.5.

### Setting up the environment

1. The environment can be downloaded from one of the links below for all operating systems:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    - _For AWS_: To train the agent on AWS (without [enabled virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  The agent can **not** be watched without a virtual screen, but can be trained.  (_To watch the agent, one can follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the downloaded file in the same directory as this GitHub repository and unzip the file.

3. Use the `requirements.txt` file to set up a python environment with all necessary packages installed.

### Instructions

See the main file `Tennis.ipynb` to get an introduction to the environment and follow the steps to solving the environment. The main classes are defined in the file `MADDPG_agent.py`.

### Approach and solution

The reinforcement learning approach we use in this project is called Multi Agent Deep Deterministic Policy Gradients (MADDPG). see this [paper](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). In this model every agent itself is modeled as a Deep Deterministic Policy Gradient (DDPG) agent (see this [paper](https://arxiv.org/pdf/1509.02971.pdf)) where, however, some information is shared between the agents.

In particular, each of the agents in this model has its own actor and critic model. The actors each receive as input the individual state (observations) of the agent and output a (two-dimensional) action. The crit`ic model of each actor, however, receives the states and actions of all actors concatenated.

Throughout training the agents all use a common experience replay buffer (a set of stored previous 1-step experiences) and draw independent samples.

Details of the implementation including the neural nets to model actor and critic models can be found in the modules `MADDPG_agent.py` and `models.py` as well as the report (report.pdf`). With the current set of models and hyperparameters the environment can be solved in around 3200 steps.