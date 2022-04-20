import torch

def walker_reward(observation, action):
    """
    Reward function is based on: https://arxiv.org/pdf/1907.02057.pdf

    where the reward is: x'_t -0.1||at||  
    with obseration: obs = np.concatenate([xpos, np.clip(self.data.get_body_xvelp(b), -10, 10), \
                                           self.data.get_body_xvelr(b), expmap, limb_type_vec])
    Suppose the x vel is position 3 in self.data_get_xvelp first entry. xpos has 3 values
    """
    xvel = observation[:, 3]
    action_penalty = -0.1 * torch.square(action).sum(dim=1)
    reward = xvel + action_penalty
    return reward.unsqueeze(-1)
