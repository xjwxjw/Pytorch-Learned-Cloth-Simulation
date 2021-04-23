import torch
import os
import numpy as np
import scipy.io as io
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def collate_fn(batch):
    data0 = [item[0] for item in batch]
    data1 = [item[1] for item in batch]
    return [data0, data1]

class SphereDataset(Dataset):
    def __init__(self, data_dir, length):
        self.state_dir = os.path.join(data_dir, '0002')
        self.data = []
        for i in range(500):
            self.data.append(os.path.join(self.state_dir, '%03d_cloth.txt' % i))
        self.cloth_topo = np.load(os.path.join(data_dir, 'cloth_connection.npy'))
        self.sphere_topo = np.load(os.path.join(data_dir, 'sphere_connection.npy'))

        self.cloth_mean = np.load(os.path.join(data_dir, 'cloth_mean.npy'))
        self.sphere_mean = np.load(os.path.join(data_dir, 'ball_mean.npy'))
        self.cloth_std = np.load(os.path.join(data_dir, 'cloth_std.npy'))
        self.sphere_std = np.load(os.path.join(data_dir, 'ball_std.npy'))
        self.collision_distance = 0.05
        self.cloth_want_idx = [0,1,2,9,10,11,16,17]#0-2:position, 9-11:velocity, 16-17:u-v coord
    
    def __len__(self):
        return len(self.data) - 1

    def GetState(self, index):
        cloth_file = self.data[index]
        ball_file = self.data[index].replace('cloth', 'ball')
        cloth_data = []
        ball_data = []
        for line in open(cloth_file, 'r'):
            line = line.split('\n')[0]
            cloth_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
        
        for line in open(ball_file, 'r'):
            line = line.split('\n')[0]
            ball_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
        
        cloth_data = np.array(cloth_data)
        ball_data = np.array(ball_data)

        return cloth_data, ball_data
        # cloth_data = (np.array(cloth_data).astype(np.float32) - np.expand_dims(self.cloth_mean, 0)) / np.expand_dims(self.cloth_std, 0)
        # ball_data = (np.array(ball_data).astype(np.float32) - np.expand_dims(self.sphere_mean, 0)) / np.expand_dims(self.sphere_std, 0)

    def __getitem__(self, index):
        cloth_state, ball_state = self.GetState(index)
        cloth_nxt_state, ball_nxt_state = self.GetState(index + 1)
        #### compute the edge feature ####
        # cloth_dis = cloth_state[:, self.cloth_want_idx].copy()
        # cloth_nxt_dis = cloth_nxt_state[:, self.cloth_want_idx].copy()
        # dis_cloth = np.triu(np.linalg.norm(cloth_dis[:, None, :3] - cloth_dis[None, :, :3], axis=-1))
        # dis_ball = np.linalg.norm(cloth_dis[:, None, :3] - ball_state[None, :, :3], axis=-1)
        
        return torch.Tensor(cloth_state[:np.random.choice(10, 1)[0] + 10]), torch.Tensor(ball_state[:np.random.choice(10, 1)[0] + 10])

if __name__ == '__main__':
    spdataset = SphereDataset('../Data', 500)
    sploader = DataLoader(spdataset, batch_size = 2, shuffle = True, num_workers = 32, collate_fn = collate_fn)
    for cloth_data, ball_data in sploader:
        print(cloth_data[0].size(), ball_data[0].size())
        # cloth_data = cloth_data.cuda()
        # ball_data = ball_data.cuda()
        # print(cloth_data.size(), ball_data.size())
    # import plotly.graph_objects as go
    # import numpy as np

    # cloth_data = []
    # for line in open('../Data/0002/005_cloth.txt', 'r'):
    #     line = line.split('\n')[0]
    #     cloth_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
    # cloth_data = np.array(cloth_data)
    # x, y, z = cloth_data[:,:3].T
    # l = 0
    # i, j, k = [], [], []
    # cloth_connection = np.array([int(line.split('\n')[0]) for line in open('../Data/cloth_connection.txt', 'r')])
    # cloth_connection = np.reshape(cloth_connection, (-1, 3))
    # fig = go.Figure(data=[go.Mesh3d(x = x, y = y, z = z, \
    #                                 i = cloth_connection[:, 0], \
    #                                 j = cloth_connection[:, 1], \
    #                                 k = cloth_connection[:, 2], \
    #                                 color='lightpink', opacity=0.50)])
    # fig.show()
