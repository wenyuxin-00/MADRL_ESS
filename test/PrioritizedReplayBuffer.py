import numpy as np
import torch

class SumTree:
    """
    SumTree for Prioritized Experience Replay.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.N = self.args.N
        self.capacity = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.tree = SumTree(self.capacity)
        
        # PER Hyperparameters
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the TD-error to priority
        self.beta = 0.4  # importance-sampling, from 0.4 to 1.0
        self.beta_increment_per_sampling = (1.0 - 0.4) / (args.max_train_steps / args.batch_size)
        self.abs_err_upper = 1.  # clipped abs error

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        transition = (obs_n, a_n, r_n, obs_next_n, done_n)
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self):
        batch_indices = np.empty((self.batch_size,), dtype=np.int32)
        ISWeights = np.empty((self.batch_size, 1), dtype=np.float32)
        
        pri_seg = self.tree.total_priority / self.batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        if min_prob == 0:
            min_prob = 0.00001

        for i in range(self.batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_priority
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            batch_indices[i] = idx
        
        # Unpack transitions
        obs_n, a_n, r_n, obs_next_n, done_n = zip(*[self.tree.data[self.tree.get_leaf(np.random.uniform(pri_seg * i, pri_seg * (i + 1)))[0] - self.capacity + 1] for i in range(self.batch_size)])

        # Convert to tensors
        obs_n = [torch.tensor(np.array(obs), dtype=torch.float32) for obs in zip(*obs_n)]
        a_n = [torch.tensor(np.array(a), dtype=torch.float32) for a in zip(*a_n)]
        r_n = [torch.tensor(np.array(r), dtype=torch.float32).view(-1, 1) for r in zip(*r_n)]
        obs_next_n = [torch.tensor(np.array(obs), dtype=torch.float32) for obs in zip(*obs_next_n)]
        done_n = [torch.tensor(np.array(done), dtype=torch.float32).view(-1, 1) for done in zip(*done_n)]
        
        ISWeights = torch.tensor(ISWeights, dtype=torch.float32)

        return batch_indices, (obs_n, a_n, r_n, obs_next_n, done_n), ISWeights

    def batch_update(self, tree_indices, abs_errors):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_indices, ps):
            self.tree.update(ti, p)

    @property
    def current_size(self):
        return self.tree.n_entries