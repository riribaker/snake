# snake
Snake is a famous video game originated in the 1976 arcade game Blockade. The player uses up, down, left and right to control the snake which grows in length (when it eats the food pellet), with the snake body and walls around the environment being the primary obstacle. 
In this project, I trained an AI agent using temporal difference learning, more specifically the Q-learning algorithm, to play a simple version of the game snake.

# how it works
We can define a MDP (Markov Decision Process) to model the Q-learning agent's movement within the environment.

States: the agent’s internal representation of the environment <br />
Actions: the possible actions the agent can take in the environment <br />
Rewards: the numerical representation of the outcome of each action in the environment. <br />

In Q-learning, instead of explicitly learning a representation for transition probabilities between states, we let the agent observe its environment, choose an action, and obtain some reward.

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?x%3D%5Cfrac%7B-b%5Cpm%5Csqrt%7Bb%5E2-4ac%7D%7D%7B2a%7D)

