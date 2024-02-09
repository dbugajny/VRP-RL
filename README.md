# VRP-RL: Vehicle Routing Problem with Reinforcement Learning

## General description

The Vehicle Routing Problem (VRP) is a classic optimization challenge in logistics and operations research. It involves determining the most efficient routes for a set of vehicles to deliver goods or services to a set of customers while satisfying various constraints e.g. vehicle capacity limits or time windows for deliveries while minimizing overall travel distance or time.

VRP-RL means that reinforcement learning approach was used to tackle the VRP problem. Reinforcement learning is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. In the context of the VRP, RL is used to train agents (representing for example vehicles) to make decisions on which routes to take in order to optimize some objective function, such as minimizing total travel distance.

In our study we decided to use two neural networks:
- Actor - main network deciding what should be the next step
- Critic - auxillary network evaluating the Actor solution 

## Repository overview

Repository is divided in terms of file extensions. Main simulation (as the name suggests) is inside the Juptyher notebook *main_simulation.ipynb*. If You are interested in just playing around with simulation, change parameters, checkout the comparison to solver solution etc. this is the only file you need to check out. Files with *.py* extension contain the implementation and the test for some of it's components. If you want to change something in the implementation this is where you should look:
- actor.py - implementation of actors (there are two version, real implementation called DenseActor and DummyActor which just makes random decisions)
- critic.py - implementation of a critic (similar network as actor, but with single output neuron and a task to estimate the quality of the action)
- environment.py - class defining the environment (parameters such as number of samples, number of locations etc.)
- utils.py - auxillary functions helpful during the simulation
- or_tools - functions required to use or_tools solver with our problem
- test_environment.py, test_utils.py - files with test to avoid common errors during the simulation

## Useful links and resources:

During the creation of this repository, we visited a number of interesting sites, some of them are loosely gathered below.

### VRP related:
- Related repo about VRP-RL: https://github.com/OptMLGroup/VRP-RL:
- Library about Capacitated Vehicle Routing Problem:
http://vrp.galgos.inf.puc-rio.br/index.php/en/
- RL-VRP solving VRP problem using pointer Network (PyTorch implementation):
https://github.com/ajayn1997/RL-VRP-PtrNtwrk

### Theory:
- Attention Mechanism: https://blog.floydhub.com/attention-mechanism/
- Playlist about neural-networks with focus on NLP (useful to understand e.g. seq2seq models):
https://www.youtube.com/watch?v=zxagGtF9MeU&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1

### Others:
- Solving the Traveling Salesman Problem with Reinforcement Learning:
https://ekimetrics.github.io/blog/2021/11/03/tsp/