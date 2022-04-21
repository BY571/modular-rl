from typing import Tuple
import gym
import numpy as np
import torch
import scipy.stats as stats
from reward_function import walker_reward
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_reward_function(env_name):
    if "walker" in env_name:
        return walker_reward
    else:
        raise NotImplementedError


class MPC():
    def __init__(self, env_name, max_num_limbs, n_planner, horizon)-> None:

        self.action_size = max_num_limbs
        self.action_low = -1
        self.action_high = 1

        self.n_planner = n_planner
        self.horizon = horizon
        self.reward_function = get_reward_function(env_name)

    
    def get_action(self, state: torch.Tensor, model: torch.nn.Module, noise: bool=False)-> torch.Tensor:
        raise NotImplementedError


class RandomShooting(MPC):
    def __init__(self, env_name, max_num_limbs, n_planner, horizon) -> None:
        super(RandomShooting, self).__init__(env_name=env_name, max_num_limbs=max_num_limbs, n_planner=n_planner, horizon=horizon)


    def _get_continuous_actions(self, )-> torch.Tensor:
        actions = np.random.uniform(low=self.action_low,
                                    high=self.action_high,
                                    size=(self.n_planner, self.horizon, self.action_size))
        return torch.from_numpy(actions).to(device).float()
    
    def get_action(self, state: np.array, model: torch.nn.Module)-> torch.Tensor:
        state = torch.from_numpy(state[None, :]).float()
        initial_states = state.repeat((self.n_planner, 1)).to(device)
        rollout_actions = self._get_continuous_actions()
        returns = self.compute_returns(initial_states, rollout_actions, model)
        best_action_idx = returns.argmax()
        optimal_action = rollout_actions[:, 0, :][best_action_idx]

        return optimal_action.cpu().numpy()


    def compute_returns(self, states: torch.Tensor, actions: torch.Tensor, model: torch.nn.Module)-> Tuple[torch.Tensor]:
        returns = torch.zeros((self.n_planner, 1)).to(device)
        for t in range(self.horizon):
            with torch.no_grad():
                delta_states = model.predict(states, actions[:, t, :])
                states += delta_states
            returns += self.reward_function(states, actions[:, t, :])

        return returns

class CEM(MPC):
    def __init__(self, env_name, max_num_limbs, n_planner, horizon, iter_update_steps=3, k_best=10, update_alpha=0.0)-> None:
        super(CEM, self).__init__(env_name=env_name, max_num_limbs=max_num_limbs, n_planner=n_planner, horizon=horizon)

        self.iter_update_steps = iter_update_steps
        self.k_best = k_best
        self.update_alpha = update_alpha
        self.epsilon = 0.001
        self.ub = 1
        self.lb = -1
        
    def get_action(self, state, model):
        state = torch.from_numpy(state[None, :]).float()
        initial_state = state.repeat((self.n_planner, 1)).to(device)
            
        mu = np.zeros(self.horizon*self.action_size)
        var = 5 * np.ones(self.horizon*self.action_size)
        X = stats.truncnorm(self.lb, self.ub, loc=np.zeros_like(mu), scale=np.ones_like(mu))
        i = 0
        while ((i < self.iter_update_steps) and (np.max(var) > self.epsilon)):
            states = initial_state
            returns = np.zeros((self.n_planner, 1))
            #variables
            lb_dist = mu - self.lb
            ub_dist = self.ub - mu
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
            
            actions = X.rvs(size=[self.n_planner, self.horizon*self.action_size]) * np.sqrt(constrained_var) + mu
            actions = np.clip(actions, self.action_low, self.action_high)
            actions_t = torch.from_numpy(actions.reshape(self.n_planner, self.horizon, self.action_size)).float().to(device)
            for t in range(self.horizon):
                with torch.no_grad():
                    states = model.predict(states, actions_t[:, t, :])
                returns += self.reward_function(states, actions_t[:, t, :]).cpu().numpy()
            
            k_best_rewards, k_best_actions = self.select_k_best(returns, actions)
            mu, var = self.update_gaussians(mu, var, k_best_actions)
            i += 1

        best_action_sequence = mu.reshape(self.horizon, -1)
        best_action = np.copy(best_action_sequence[0]) 
        assert best_action.shape == (self.action_size,)
        return best_action
            
    
    def select_k_best(self, rewards, action_hist):
        assert rewards.shape == (self.n_planner, 1)
        idxs = np.argsort(rewards, axis=0)

        elite_actions = action_hist[idxs][-self.k_best:, :].squeeze(1) # sorted (elite, horizon x action_space)
        k_best_rewards = rewards[idxs][-self.k_best:, :].squeeze(-1)

        assert k_best_rewards.shape == (self.k_best, 1)
        assert elite_actions.shape == (self.k_best, self.horizon*self.action_size)
        return k_best_rewards, elite_actions


    def update_gaussians(self, old_mu, old_var, best_actions):
        assert best_actions.shape == (self.k_best, self.horizon*self.action_size)

        new_mu = best_actions.mean(0)
        new_var = best_actions.var(0)

        mu = (self.update_alpha * old_mu + (1.0 - self.update_alpha) * new_mu)
        var = (self.update_alpha * old_var + (1.0 - self.update_alpha) * new_var)
        assert mu.shape == (self.horizon*self.action_size, )
        assert var.shape == (self.horizon*self.action_size, )
        return mu, var


class PDDM(MPC):
    def __init__(self, env_name, max_num_limbs, n_planner, horizon, gamma=1.0, beta=0.5)-> None:
        super(PDDM, self).__init__(env_name, max_num_limbs, n_planner, horizon)

        self.gamma = gamma
        self.beta = beta
        self.mu = np.zeros((self.horizon, self.action_size))
        
    def get_action(self, state, model):
        state = torch.from_numpy(state[None, :]).float()
        initial_states = state.repeat((self.n_planner, 1)).to(device)
        actions, returns = self.get_pred_trajectories(initial_states, model)
        optimal_action = self.update_mu(actions, returns)
        return optimal_action
        
    def update_mu(self, action_hist, returns):
        assert action_hist.shape == (self.n_planner, self.horizon, self.action_size)
        assert returns.shape == (self.n_planner, 1)

        c = np.exp(self.gamma * (returns) -np.max(returns))
        d = np.sum(c) + 1e-10
        assert c.shape == (self.n_planner, 1)
        assert d.shape == (), "Has shape {}".format(d.shape)
        c_expanded = c[:, :, None]
        assert c_expanded.shape == (self.n_planner, 1, 1)
        weighted_actions = c_expanded * action_hist
        self.mu = weighted_actions.sum(0) / d
        assert self.mu.shape == (self.horizon, self.action_size)       
        
        return self.mu[0]
    
    def sample_actions(self, past_action):
        u = np.random.normal(loc=0, scale=1.0, size=(self.n_planner, self.horizon, self.action_size))
        actions = u.copy()
        for t in range(self.horizon):
            if t == 0:
                actions[:, t, :] = self.beta * (self.mu[t, :] + u[:, t, :]) + (1 - self.beta) * past_action
            else:
                actions[:, t, :] = self.beta * (self.mu[t, :] + u[:, t, :]) + (1 - self.beta) * actions[:, t-1, :]
        assert actions.shape == (self.n_planner, self.horizon, self.action_size), "Has shape {} but should have shape {}".format(actions.shape, (self.n_planner, self.horizon, self.action_size))
        actions = np.clip(actions, self.action_low, self.action_high)
        return actions
    
    def get_pred_trajectories(self, states, model): 
        returns = np.zeros((self.n_planner, 1))
        np.random.seed()
        past_action = self.mu[0].copy()
        actions = self.sample_actions(past_action)
        torch_actions = torch.from_numpy(actions).float().to(device)
        for t in range(self.horizon):
            with torch.no_grad():
                actions_t = torch_actions[:, t, :]
                assert actions_t.shape == (self.n_planner, self.action_size)
                states = model.predict(states, actions_t)
            returns += self.reward_function(states, actions_t).cpu().numpy()
        return actions, returns