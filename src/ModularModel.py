from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MLPBase
import torchfold
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelVanilla(nn.Module):
    """a vanilla model module that outputs a node's observation predictions given only its observation and action(no message between nodes)"""
    def __init__(self, state_dim, action_dim):
        super(ModelVanilla, self).__init__()
        self.model = MLPBase(state_dim + action_dim, state_dim)

    def forward(self, x, u):
        xu = torch.cat([x, u], -1)
        x1 = self.model(xu)
        return x1


class ModelUp(nn.Module):
    """a bottom-up module used in bothway message passing that only passes message to its parent"""
    def __init__(self, state_dim, action_dim, msg_dim, max_children):
        super(ModelUp, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64 + msg_dim * max_children, 64)
        self.fc3 = nn.Linear(64, msg_dim)

    def forward(self, x, u, *m):
        m = torch.cat(m, dim=-1)
        xu = torch.cat([x, u], dim=-1)
        xu = self.fc1(xu)
        xu = F.normalize(xu, dim=-1)
        xum = torch.cat([xu, m], dim=-1)
        xum = torch.tanh(xum)
        xum = self.fc2(xum)
        xum = torch.tanh(xum)
        xum = self.fc3(xum)
        xum = F.normalize(xum, dim=-1)
        msg_up = xum

        return msg_up


class ModelUpAction(nn.Module):
    """a bottom-up module used in bottom-up-only message passing that passes message to its parent and outputs q-values"""
    def __init__(self, state_dim, action_dim, msg_dim, max_children):
        super(ModelUpAction, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64 + msg_dim * max_children, 64)
        self.fc3 = nn.Linear(64, msg_dim)
        self.model = MLPBase(state_dim + action_dim + msg_dim * max_children, state_dim)

    def forward(self, x, u, *m):
        m = torch.cat(m, dim=-1)
        xum = torch.cat([x, u, m], dim=-1)

        x1 = self.model(xum)

        xu = torch.cat([x, u], dim=-1)
        xu = self.fc1(xu)
        xu = F.normalize(xu, dim=-1)
        xum = torch.cat([xu, m], dim=-1)
        xum = torch.tanh(xum)
        xum = self.fc2(xum)
        xum = torch.tanh(xum)
        xum = self.fc3(xum)
        xum = F.normalize(xum, dim=-1)
        msg_up = xum

        return msg_up, x1


class ModelDownAction(nn.Module):
    """a top-down module used in bothway message passing that passes messages to children and outputs q-values"""
    # input dim is state dim if only using top down message passing
    # if using bottom up and then top down, it is the node's outgoing message dim
    def __init__(self, state_dim, action_dim, msg_dim, max_children):
        super(ModelDownAction, self).__init__()
        self.model = MLPBase(state_dim + action_dim + msg_dim, state_dim)
        self.msg_base = MLPBase(state_dim + msg_dim, msg_dim * max_children)

    def forward(self, x, u, m):
        xum = torch.cat([x, u, m], dim=-1)
        x1 = self.model(xum)
        xm = torch.cat([x, m], dim=-1)
        xm = torch.tanh(xm)
        msg_down = self.msg_base(xm)
        msg_down = F.normalize(msg_down, dim=-1)

        return x1, msg_down


class ModelGraphPolicy(nn.Module):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""
    def __init__(self, state_dim, action_dim, msg_dim, batch_size, max_children, disable_fold, td, bu):
        super(ModelGraphPolicy, self).__init__()
        self.num_limbs = 1
        self.x1 = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.input_action = [None] * self.num_limbs
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim

        assert self.action_dim == 1
        self.td = td
        self.bu = bu
        if self.bu:
            # bottom-up then top-down
            if self.td:
                self.sNet = nn.ModuleList([ModelUp(state_dim, action_dim, msg_dim, max_children)] * self.num_limbs).to(device)
            # bottom-up only
            else:
                self.sNet = nn.ModuleList([ModelUpAction(state_dim, action_dim, msg_dim, max_children)] * self.num_limbs).to(device)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "sNet" + str(i).zfill(3), self.sNet[i])
        # we pass msg_dim as first argument because in both-way message-passing, each node takes in its passed-up message as 'state'
        if self.td:
            # bottom-up then top-down
            if self.bu:
                self.dynamics = nn.ModuleList([ModelDownAction(msg_dim, action_dim, msg_dim, max_children)] * self.num_limbs).to(device)
            # top-down only
            else:
                self.dynamics = nn.ModuleList([ModelDownAction(state_dim, action_dim, msg_dim, max_children)] * self.num_limbs).to(device)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "dynamics" + str(i).zfill(3), self.dynamics[i])

        # no message passing
        if not self.bu and not self.td:
            self.dynamics = nn.ModuleList([ModelVanilla(state_dim, action_dim)] * self.num_limbs).to(device)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "dynamics" + str(i).zfill(3), self.dynamics[i])

        if not self.disable_fold:
            for i in range(self.max_children):
                setattr(self, 'get_{}'.format(i), self.addFunction(i))

    def forward(self, state, action):
        self.clear_buffer()
        if not self.disable_fold:
            self.fold = torchfold.Fold()
            self.fold.cuda()
            self.zeroFold_td = self.fold.add("zero_func_td")
            self.zeroFold_bu = self.fold.add("zero_func_bu")
            self.ds = []
        assert state.shape[1] == self.state_dim * self.num_limbs, 'state.shape[1] expects {} but got {} with num_limbs being {} and state_dim being {}'.format(self.state_dim * self.num_limbs, state.shape[1], self.num_limbs, self.state_dim)
        for i in range(self.num_limbs):
            self.input_state[i] = state[:, i * self.state_dim:(i + 1) * self.state_dim]
            self.input_action[i] = action[:, i]
            self.input_action[i] = torch.unsqueeze(self.input_action[i], -1)
            if not self.disable_fold:
                self.input_state[i] = torch.unsqueeze(self.input_state[i], 0)
                self.input_action[i] = torch.unsqueeze(self.input_action[i], 0)

        if self.bu:
            # bottom up transmission by recursion
            for i in range(self.num_limbs):
                self.bottom_up_transmission(i)

        if self.td:
            # top down transmission by recursion
            for i in range(self.num_limbs):
                self.top_down_transmission(i)

        if not self.bu and not self.td:
            for i in range(self.num_limbs):
                if not self.disable_fold:
                    self.delta_state[i] = self.fold.add('dynamics' + str(0).zfill(3), self.input_state[i], self.input_action[i])
                else:
                    self.delta_state[i] = self.dynamics[i](self.input_state[i], self.input_action[i])

        if not self.disable_fold:
            self.ds += self.delta_state
            self.delta_state = self.fold.apply(self, [self.ds])[0]
            self.delta_state = torch.transpose(self.delta_state, 0, 1)
            self.fold = None
            self.delta_state = self.delta_state.flatten(1)
        else:
            self.delta_state = torch.stack(self.delta_state, dim=-1) 

        return torch.squeeze(self.delta_state)


    def bottom_up_transmission(self, node):

        if node < 0:
            if not self.disable_fold:
                return self.zeroFold_bu
            else:
                return torch.zeros((self.batch_size, self.msg_dim), requires_grad=True).to(device)

        if self.msg_up[node] is not None:
            return self.msg_up[node]

        state = self.input_state[node]
        action = self.input_action[node]

        children = [i for i, x in enumerate(self.parents) if x == node]
        assert (self.max_children - len(children)) >= 0
        children += [-1] * (self.max_children - len(children))
        msg_in = [None] * self.max_children
        for i in range(self.max_children):
            msg_in[i] = self.bottom_up_transmission(children[i])

        if not self.disable_fold:
            if self.td:
                self.msg_up[node] = self.fold.add('sNet' + str(0).zfill(3), state, action, *msg_in)
            else:
                self.msg_up[node], self.delta_state[node] = self.fold.add('sNet' + str(0).zfill(3), state, action, *msg_in).split(2)
        else:
            if self.td:
                self.msg_up[node] = self.sNet[node](state, action, *msg_in)
            else:
                self.msg_up[node], self.delta_state[node] = self.sNet[node](state, action, *msg_in)

        return self.msg_up[node]

    def top_down_transmission(self, node):

        if node < 0:
            if not self.disable_fold:
                return self.zeroFold_td
            else:
                return torch.zeros((self.batch_size, self.msg_dim * self.max_children), requires_grad=True).to(device)

        elif self.msg_down[node] is not None:
            return self.msg_down[node]

        # in both-way message-passing, each node takes in its passed-up message as 'state'
        if self.bu:
            state = self.msg_up[node]
        else:
            state = self.input_state[node]

        action = self.input_action[node]
        parent_msg = self.top_down_transmission(self.parents[node])

        # find self children index (first child of parent, second child of parent, etc)
        # by finding the number of previous occurences of parent index in the list
        self_children_idx = self.parents[:node].count(self.parents[node])

        # if the structure is flipped, flip message order at the root
        if self.parents[0] == -2 and node == 1:
            self_children_idx = (self.max_children - 1) - self_children_idx

        if not self.disable_fold:
            msg_in = self.fold.add('get_{}'.format(self_children_idx), parent_msg)
        else:
            msg_in = self.msg_slice(parent_msg, self_children_idx)

        if not self.disable_fold:
            self.delta_state[node], self.msg_down[node] = self.fold.add('dynamics' + str(0).zfill(3), state, action, msg_in).split(2)
        else:
            self.delta_state[node], self.msg_down[node] = self.dynamics[node](state, action, msg_in)

        return self.msg_down[node]

    def zero_func_td(self):
        return torch.zeros((1, self.batch_size, self.msg_dim * self.max_children), requires_grad=True).to(device)

    def zero_func_bu(self):
        return torch.zeros((1, self.batch_size, self.msg_dim), requires_grad=True).to(device)

    # an ugly way to define functions in a for loop (for torchfold only)
    def addFunction(self, n):
        def f(x):
            return torch.split(x, x.shape[-1] // self.max_children, dim=-1)[n]
        return f

    def msg_slice(self, x, idx):
        return torch.split(x, x.shape[-1] // self.max_children, dim=-1)[idx]

    def clear_buffer(self):
        self.delta_state = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.input_action = [None] * self.num_limbs
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.zeroFold_td = None
        self.zeroFold_bu = None
        self.fold = None

    def change_morphology(self, parents):
        if not self.disable_fold:
            if self.bu:
                for i in range(1, self.num_limbs):
                    delattr(self, "sNet" + str(i).zfill(3))
            if not (self.bu and not self.td):
                for i in range(1, self.num_limbs):
                    delattr(self, "dynamics" + str(i).zfill(3))
        self.parents = parents
        self.num_limbs = len(parents)
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        if self.bu:
            self.sNet = nn.ModuleList([self.sNet[0]] * self.num_limbs)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "sNet" + str(i).zfill(3), self.sNet[i])
        if not (self.bu and not self.td):
            self.dynamics = nn.ModuleList([self.dynamics[0]] * self.num_limbs)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "dynamics" + str(i).zfill(3), self.dynamics[i])
