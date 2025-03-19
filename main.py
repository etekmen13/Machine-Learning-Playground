import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
from reinforcement.models import DQN
from neuralnet.losses import MSE
from neuralnet.optimizers import Adam
from metrics.tracker import MetricTracker

WEIGHTS_FILE = "dqn_weights.npz"
SEED = 42

def set_global_seed(seed):
    np.random.seed(seed)

def save_weights(agent, filename):
    params = agent.get_params()
    data = {}
    for i, param in enumerate(params):
        data[f'arr_{i}'] = param.data
    np.savez(filename, **data)
    print(f"Weights saved to {filename}")

def load_weights(agent, filename):
    loaded = np.load(filename)
    params = agent.get_params()
    for i, p in enumerate(params):
        p.data = loaded[f'arr_{i}']
    print(f"Weights loaded from {filename}")
    return agent

def create_environment(env_name, seed):
    env = gym.make(env_name)
    state, _ = env.reset(seed=seed)
    return env

def create_agent(env, agent_type="dqn"):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_units = [64, 64]
    if agent_type == "dqn":
        return DQN(state_dim, action_dim, hidden_units, MSE(), Adam())
    else:
        # add additional agent types here
        raise ValueError("Unsupported agent type.")

def train(agent, env, episodes=1000, warmup_steps=32):
    metric_tracker = MetricTracker()
    metric_tracker.update(episode=0, loss=0)
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    for episode in range(episodes):
        state, _ = env.reset(seed=SEED)
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        while not terminated and not truncated:
            action = agent.get_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward -= np.abs(next_state[0])
            agent.add_experience(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            
            if len(agent) > warmup_steps and steps % 4 == 0:
                loss = agent.update_batch(32)
                metric_tracker.update(episode=episode, loss=loss)
            steps += 1

        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Steps: {steps}")
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        
    env.close()
    metric_tracker.plot()
    return agent

def evaluate(agent, env_name="CartPole-v1", max_steps=1000, record_video=False):
    env = create_environment(env_name, SEED)
    if record_video:
        env = RecordVideo(env, video_folder='./videos', name_prefix='cartpole-demo')
    state, _ = env.reset(seed=SEED)
    terminated = False
    truncated = False
    total_reward = 0
    steps = 0
    while steps < max_steps and not terminated:
        action = agent.get_action(state, eps=0.0)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        env.render()
    env.close()
    print(f"Evaluation complete. Total reward: {total_reward}")
    return total_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Run training or evaluation")
    parser.add_argument("--env", choices=["CartPole-v1","InvertedDoublePendulum-v5"],default="CartPole-v1", help="Gym environment name")
    parser.add_argument("--agent", choices=["dqn"], default="dqn", help="Agent type")
    parser.add_argument("--load", action="store_true", help="Load saved weights instead of training")
    args = parser.parse_args()
    
    set_global_seed(SEED)
    env = create_environment(args.env, SEED)
    agent = create_agent(env, args.agent)
    
    if args.load:
        load_weights(agent, WEIGHTS_FILE)
    else:
        agent = train(agent, env)
        save_weights(agent, WEIGHTS_FILE)
    
    evaluate(agent, env_name=args.env, max_steps=1000, record_video=True)

if __name__ == "__main__":
    main()