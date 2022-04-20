from git import Object
import numpy as np
import torch
import torch.nn.functional as F
from ModularModel import ModelGraphPolicy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseDynamicsModel(Object):
    def __init__(self, ):
        pass

    def train(self,):
        raise NotImplementedError

    def select_action(self,):
        raise NotImplementedError

    def save(self,):
        raise NotImplementedError

    def load(self,):
        raise NotImplementedError


class DynamicsModel(BaseDynamicsModel):
    def __init__(self, state_dim, action_dim, msg_dim, batch_size, max_children, disable_fold, td, bu, lr):


        self.batch_size = batch_size
        self.dynamics_model = ModelGraphPolicy(state_dim=state_dim,
                                               action_dim=action_dim,
                                               msg_dim=msg_dim,
                                               batch_size=batch_size,
                                               max_children=max_children,
                                               disable_fold=disable_fold,
                                               td=td,
                                               bu=bu)

        self.optimizer = torch.optim.Adam(self.dynamics_model.parameters(), lr=lr)
    
    def change_morphology(self, graph):
        self.dynamics_model.change_morphology(graph)

    def save(self, fname):
        torch.save(self.dynamics_model.state_dict(), '%s_dynamics_model.pth' % fname)

    def load(self, fname):
        self.dynamics_model.load_state_dict(torch.load('%s_dynamics_model.pth' % fname))

    def predict(self, observations, actions):
        return self.dynamics_model(observations, actions)

    def select_action(self, obs, mpc, max_num_limbs):
        # TODO: currently random actions for debugging
        action = np.random.uniform(low=-1, high=1, size=max_num_limbs)

        return action

    def train_single(self, buffer, iterations):
        # TODO: Currently only sampling random transitions of the replay buffer to train the model might change that to dataloader 
        # or other sampling methods 
        # ALSO PROBABLY IMPORTANT TO ADD SCALING!
        # Probabilistic predictions? 
        losses = []
        for it in range(iterations):

            # sample replay buffer
            x, y, u, r, d = buffer.sample(self.batch_size)
            state = torch.FloatTensor(x).to(device)
            next_state = torch.FloatTensor(y).to(device)
            action = torch.FloatTensor(u).to(device)
            reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(1 - d).to(device)

            target = next_state - state

            # TODO: add reward predictions? OR extract reward function for each environment for planning!
            pred_next_state = self.dynamics_model(state, action)
            # TODO: TRY DELTA STATE PREDICTION or direct state prediction
            loss = F.mse_loss(pred_next_state, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        
        return np.mean(losses)


    def train(self, replay_buffer_list, iterations_list, graphs=None, envs_train_names=None):
        # per_morph_iter = sum(iterations_list) // len(envs_train_names) # Updating steps per morphology model
        number_updates = 50 # using fixed updates for now
        # track losses
        env_losses = []
        for env_name in envs_train_names:
            replay_buffer = replay_buffer_list[env_name]
            self.change_morphology(graphs[env_name])
            loss = self.train_single(buffer=replay_buffer, iterations=number_updates)
            env_losses.append(loss)
        return dict(zip([i + "_model_train_loss" for i in envs_train_names], env_losses))

    