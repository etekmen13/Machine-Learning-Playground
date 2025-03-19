from neuralnet.models import MLP
from neuralnet.layers import Layer
from neuralnet.activations import ReLU, Tanh
from neuralnet.losses import MSE
from reinforcement.replay import ReplayBuffer
import numpy as np
import os
from copy import deepcopy

class DQN:
    def __init__(self, state_dim, action_dim, hidden_units, loss, optimizer):
        layers = [Layer(state_dim, hidden_units[0], activation=ReLU())]
        
        for i in range(len(hidden_units) - 1):
            layers.append(Layer(hidden_units[i], hidden_units[i + 1], activation=ReLU()))
        
        layers.append(Layer(hidden_units[-1], action_dim)) 

        self.q = MLP(*layers, loss=loss)
        self.target_q = deepcopy(self.q)
        self.loss = loss
        self.replay_buffer = ReplayBuffer(10000)
        self.optimizer = optimizer
        self.optimizer(self.q)
        self.optimizer.zero_grad()
    def get_params(self):
        return self.q.params

    def update_single(self, state, action, reward, next_state, done):
        q_current = self.q.forward(state)
        q_next = self.target_q.forward(next_state)
        q_target = q_current.copy()
        gamma = 0.99
        q_target[action] = reward + (1 - done) * gamma * np.max(q_next)
        loss = self.loss(q_current, q_target)
        grad = self.loss.gradient(q_current, q_target)
        self.q.backward(grad)
        return loss
    
    def get_action(self, state, eps):
        if np.random.rand() < eps:
            return np.random.randint(self.q.layers[-1].weights.get().shape[1])
        else:
            return self.get_action_greedy(state)
    def get_action_greedy(self, state):
        q_values = self.q.forward(state)
        action = np.argmax(q_values)
        return action.astype(np.int32)
    
    def update_batch(self, batch_size):
        self.optimizer.zero_grad()
        batch = self.replay_buffer.sample(batch_size)
        results = []
        for batch_item in batch:
            results.append(self.update_single(*batch_item))
        self.optimizer.step()

        for k in range(len(self.q.params)):
            self.target_q.params[k].data = 0.01 * self.q.params[k].data + (1 - 0.01) * self.target_q.params[k].data

            
        return np.mean(results)
    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))
    def clear_replay_buffer(self):
        self.replay_buffer.clear()
    def __len__(self):
        return len(self.replay_buffer)
    def full(self):
        return len(self.replay_buffer) == self.replay_buffer.capacity
    

# J(pi) = E[Q(s,a) - αlog(pi(a|s))]
# log(pi(a|s)) is the entropy term, 
# α is the temperature
# Q(s,a) is the action-value function
# pi(a|s) is the policy
class SAC:
    def __init__(self, actor_dims, critic_dims, loss, optimizer):

        # Initialize Actor Network
        actor_layers = [
            Layer(actor_dims[i], actor_dims[i + 1], activation=ReLU()) 
            for i in range(len(actor_dims) - 2)
        ]

        actor_layers.append(Layer(actor_dims[-2], actor_dims[-1], activation=Tanh()))  # Mean
        actor_layers.append(Layer(actor_dims[-2], actor_dims[-1], activation=Tanh()))  # Log Std

        self.actor = MLP(*actor_layers, loss=loss, output_heads=2)  # Two outputs

        # Initialize Critic Network
        critic_layers = [
            Layer(critic_dims[i], critic_dims[i + 1], activation=ReLU()) 
            for i in range(len(critic_dims) - 1)
        ]

        self.critic1 = MLP(*critic_layers, loss=loss, output_heads=1)  # One Q-function
        self.critic2 = deepcopy(self.critic1)  # Second Q-function

        # Initialize Target Networks
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)


        self.optimizer = optimizer
        # Actor Optimizer
        self.actor_optimizer = deepcopy(optimizer)
        self.actor_optimizer(self.actor)
        self.actor_optimizer.zero_grad()

        # Critic Optimizers
        self.critic_optimizer = deepcopy(optimizer)
        self.critic_optimizer(self.critic1)
        self.critic_optimizer.zero_grad()
        
        self.critic_optimizer2 = deepcopy(optimizer)
        self.critic_optimizer2(self.critic2)
        self.critic_optimizer2.zero_grad()


        self.loss = loss
        self.replay_buffer = ReplayBuffer(10000)
    
    def get_action(self, state):
        mean, log_std = self.actor.forward(state)
        std = np.exp(log_std)
        action = mean + std * np.random.randn(*mean.shape)
        return np.tanh(action)

    def critic_loss(self, state, action, reward, next_state, done):
        # Compute Q-values
        q1 = self.critic1.forward(np.concatenate([state,action]))
        q2 = self.critic2.forward(np.concatenate([state,action]))

        next_actions, next_log_std = self.actor.forward(next_state)
        # Compute target Q-values
        with Pool(processes=os.cpu_count()) as pool:
            target_q1 = pool.apply_async(self.target_critic1.forward, (np.concatenate([next_state, action])))
            target_q2 = pool.apply_async(self.target_critic2.forward, (np.concatenate([next_state, action])))
            target_q1 = target_q1.get()
            target_q2 = target_q2.get()

        # Compute the minimum of the two target Q-values
        min_target_q = np.minimum(target_q1, target_q2)

        # Compute the Bellman backup
        gamma = 0.99
        q_target = reward + (1 - done) * gamma * min_target_q

        # Compute the critic loss
        critic_loss1 = self.loss(q1, q_target)
        critic_loss2 = self.loss(q2, q_target)

        return critic_loss1, critic_loss2
