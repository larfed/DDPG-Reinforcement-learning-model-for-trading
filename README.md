# Stock Trading with Deep Deterministic Policy Gradient (DDPG)

This repository contains a deep reinforcement learning model for stock trading using the Deep Deterministic Policy Gradient (DDPG) algorithm. The model is implemented using TensorFlow and TF-Agents. The goal is to create an intelligent agent capable of trading stocks to maximize returns.

## Table of Contents
- [Introduction](#introduction)
- [Model Description](#model-description)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)


## Introduction

Stock trading is a complex and dynamic problem involving multiple factors and continuous decision-making. Traditional approaches rely heavily on human expertise and heuristics. Reinforcement learning, particularly the DDPG algorithm, offers a powerful method for training an agent to make optimal trading decisions by learning from interactions with the market.

## Model Description

The DDPG algorithm combines the advantages of DQN (Deep Q-Learning) and policy gradient methods to handle continuous action spaces. It uses two neural networks:
- **Actor Network**: Determines the best action (e.g., buy, sell, hold) given the current state.
- **Critic Network**: Evaluates the action taken by the actor by estimating the Q-value (expected return).

### Key Components

- **Actor Network**: Maps states to a specific action.
- **Critic Network**: Estimates the Q-value for state-action pairs.
- **Replay Buffer**: Stores past experiences to break the correlation between consecutive samples.
- **Ornstein-Uhlenbeck Noise**: Added to actions during training for exploration.

## Project Structure

- **Environment Setup**: Installation of necessary libraries and packages.
- **Data Loading and Preprocessing**: Load and preprocess stock market data.
- **Model Definition**: Define the actor and critic networks, and set up the DDPG agent.
- **Training**: Train the model using the preprocessed data.
- **Evaluation**: Evaluate the trained model to assess its performance.
- **Model Saving**: Save the trained model for future use.

## Dependencies

To run this project, you need the following dependencies:

- Python 3.x
- TensorFlow 2.15
- TF-Agents
- Pandas
# Usage
## Install the necessary dependencies using pip:

```sh
pip install tensorflow==2.15 tf-agents pandas

````
## Environment Setup: Installs the required libraries.
Data Loading and Preprocessing: Loads and preprocesses the stock market data.
Model Definition: Defines the DDPG model architecture.
Training: Trains the model on the stock data.
Evaluation: Evaluates the trained model's performance.
Model Saving: Saves the trained model to disk.
 

## Environment Setup
### Install TensorFlow and TF-Agents to set up the environment for training and evaluating the model.
````
!pip install --upgrade -q tensorflow==2.15
!pip install -q tf-agents
````
## Data Loading and Preprocessing
### Load and preprocess the stock market data from a CSV file. This involves cleaning the data and extracting relevant features.
````
import pandas as pd
FILE = "cleaned.csv"
df = pd.read_csv(FILE)
INSTRUMENTS = CONFIG_TARGET_INSTRUMENTS

COLS = ['high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']
SCOLS = ["vh", "vl", "vc", "open_s", "volume_s", "quoteVolume_s", "weightedAverage_s"]
OBS_COLS = ['vh', 'vl', 'vc', 'open_s', 'volume_s', 'quoteVolume_s', 'weightedAverage_s',
            'vh_roll_7', 'vh_roll_14', 'vh_roll_30', 'vl_roll_7', 'vl_roll_14', 'vl_roll_30',
            'vc_roll_7', 'vc_roll_14', 'vc_roll_30', 'open_s_roll_7', 'open_s_roll_14', 
            'open_s_roll_30', 'volume_s_roll_7', 'volume_s_roll_14', 'volume_s_roll_30', 
            'quoteVolume_s_roll_7', 'quoteVolume_s_roll_14', 'quoteVolume_s_roll_30', 
            'weightedAverage_s_roll_7', 'weightedAverage_s_roll_14', 'weightedAverage_s_roll_30']

````
## Model Definition
### Define the DDPG model, including the actor and critic networks, and initialize the agent.

````
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

env = suite_gym.load('StockTrading-v0')
actor_net = actor_distribution_network.ActorDistributionNetwork(
    env.observation_spec(),
    env.action_spec(),
    fc_layer_params=(400, 300))
critic_net = value_network.ValueNetwork(
    (env.observation_spec(), env.action_spec()),
    fc_layer_params=(400, 300))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)

agent = ddpg_agent.DdpgAgent(
    env.time_step_spec(),
    env.action_spec(),
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=optimizer,
    critic_optimizer=optimizer,
    ou_stddev=0.2,
    ou_damping=0.15,
    target_update_tau=0.05,
    target_update_period=5,
    dqda_clipping=None,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=0.99,
    reward_scale_factor=1.0,
    train_step_counter=train_step_counter)

agent.initialize()
````
## Training
### Train the DDPG model using the preprocessed stock market data.
````

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=env.batch_size,
    max_length=replay_buffer_capacity)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
iterator = iter(dataset)

def train_one_iteration():
    for _ in range(num_iterations):
        collect_driver.run()
        experience, _ = next(iterator)
        train_loss = agent.train(experience).loss
        print(f"Iteration {iteration} - Loss: {train_loss}")

train_one_iteration()
````
## Evaluation
### Evaluate the trained model to determine its performance on the stock data.
````
def evaluate_policy():
    num_episodes = 10
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = env.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = agent.policy.action(time_step)
            time_step = env.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    print(f"Average Return: {avg_return}")

evaluate_policy()
````
## Model Saving
## Save the trained model for future use.

````
policy_dir = 'policy'
tf_policy_saver = policy_saver.PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)
````
