# snake
Snake is a famous video game originated in the 1976 arcade game Blockade. The player uses up, down, left and right to control the snake which grows in length (when it eats the food pellet), with the snake body and walls around the environment being the primary obstacle. 
In this project, I trained an AI agent using temporal difference learning, more specifically the Q-learning algorithm, to play a simple version of the game snake.

# how it works
We can define a MDP (Markov Decision Process) to model the Q-learning agent's movement within the environment.

States: the agent’s internal representation of the environment <br />
Actions: the possible actions the agent can take in the environment <br />
Rewards: the numerical representation of the outcome of each action in the environment. <br />

In Q-learning, instead of explicitly learning a representation for transition probabilities between states, we let the agent observe its environment, choose an action, and obtain some reward.

![Alt text](https://camo.githubusercontent.com/6c183cd0fcf753fa21f76d17168691c560813c2f5828f499a7f3b342389b7e08/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f78253344253543667261632537422d62253543706d2535437371727425374262253545322d3461632537442537442537423261253744)
