import torch
import os
import numpy as np
import scipy.io as io
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def collate_fn(batch):
    data0 = [item[0] for item in batch]
    data1 = [item[1] for item in batch]
    data2 = [item[2] for item in batch]
    data3 = [item[3] for item in batch]
    data4 = [item[4] for item in batch]
    return [data0, data1, data2, data3, data4]

class SphereDataset(Dataset):
    def __init__(self, data_dir, length):
        self.data = []
        # self.seq = []
        self.length = length
        for data_id in range(100):
            state_dir = os.path.join(data_dir, ('data_0424/%04d' % data_id))
            for file_id in range(1, 499):
                cur_file = os.path.join(state_dir, '%03d_cloth.txt' % file_id)
                nxt_file = os.path.join(state_dir, '%03d_cloth.txt' % (file_id + 1))
                self.data.append([cur_file, nxt_file])
        self.cloth_topo = np.load(os.path.join(data_dir, 'cloth_connection.npy'), allow_pickle = True)
        self.sphere_topo = np.load(os.path.join(data_dir, 'sphere_connection.npy'), allow_pickle = True)
        self.adj_map = np.load(os.path.join(data_dir, 'adj_map.npy'), allow_pickle = True)
        self.uvedge_node_i = np.load(os.path.join(data_dir, 'uvedge_node_i.npy'), allow_pickle = True)
        self.uvedge_node_j = np.load(os.path.join(data_dir, 'uvedge_node_j.npy'), allow_pickle = True)

        self.cloth_mean = np.load(os.path.join(data_dir, 'cloth_mean.npy'), allow_pickle = True)
        self.sphere_mean = np.load(os.path.join(data_dir, 'ball_mean.npy'), allow_pickle = True)
        self.cloth_std = np.load(os.path.join(data_dir, 'cloth_std.npy'), allow_pickle = True)
        self.sphere_std = np.load(os.path.join(data_dir, 'ball_std.npy'), allow_pickle = True)
        self.collision_distance = 0.015
        self.cloth_want_idx = [0,1,2, 9,10,11]#0-2:position, 9-11:velocity, 16-17:u-v coord
    
    def __len__(self):
        return len(self.data) 

    def GetState(self, index, time_id):
        #### get current state, including vertex and edge information, as input ####
        cloth_file = self.data[index][0]
        ball_file = self.data[index][0].replace('cloth', 'ball')
        uv_file = self.data[index][0].replace('cloth', 'uv')
        world_file = self.data[index][0].replace('cloth', 'world')

        cloth_data = []
        ball_data = []
        uv_data = []
        world_data = []

        for line in open(cloth_file, 'r'):
            line = line.split('\n')[0]
            cloth_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
        
        for line in open(ball_file, 'r'):
            line = line.split('\n')[0]
            ball_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
        
        for line in open(uv_file, 'r'):
            line = line.split('\n')[0]
            uv_data.append(np.array([float(data) for data in line.split(' ')]))

        for line in open(world_file, 'r'):
            line = line.split('\n')[0]
            world_data.append(np.array([float(data) for data in line.split(' ')]))
            
        cloth_data = np.array(cloth_data)
        ball_data = np.array(ball_data)
        uv_data = np.array(uv_data)
        world_data = np.array(world_data)
        
        #### get next state, mainly the position, as output ####
        cloth_nxt_file = self.data[index][1]
        cloth_nxt_data = []
        for line in open(cloth_nxt_file, 'r'):
            line = line.split('\n')[0]
            cloth_nxt_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
        cloth_nxt_data = np.array(cloth_nxt_data)
        
        return cloth_data, ball_data, uv_data, world_data, cloth_nxt_data

    def __getitem__(self, index):
        time_id = np.random.choice(self.length - 2, 1)[0] + 1
        cloth_state, ball_state, uv_state, world_state, cloth_nxt_state = self.GetState(index, time_id)
        return torch.from_numpy(cloth_state[:, self.cloth_want_idx].astype(np.float32)),\
               torch.from_numpy(ball_state.astype(np.float32)), \
               torch.from_numpy(uv_state.astype(np.float32)),\
               torch.from_numpy(world_state.astype(np.float32)),\
               torch.from_numpy(cloth_nxt_state[:, :3].astype(np.float32))

if __name__ == '__main__':
    spdataset = SphereDataset('../Data', 500)
    sploader = DataLoader(spdataset, batch_size = 2, shuffle = True, num_workers = 8, collate_fn = collate_fn)
    for step, (cloth_state, ball_state, uv_state, world_state, cloth_nxt_state) in enumerate(sploader):
        print(step, cloth_state[0].size(), world_state[0].size())
    # cloth_connection = np.array([int(line.split('\n')[0]) for line in open('../Data/cloth_connection.txt', 'r')])
    # cloth_connection = np.reshape(cloth_connection, (-1, 3))
    # ball_connection = np.array([int(line.split('\n')[0]) for line in open('../Data/sphere_connection.txt', 'r')])
    # ball_connection = np.reshape(ball_connection, (-1, 3))

    # from mpl_toolkits.mplot3d import Axes3D as axes3d
    # import matplotlib.pyplot as plt
    
    # for i in range(500):
    #     cloth_data = []
    #     for line in open('../Data/0002/%03d_cloth.txt' % i, 'r'):
    #         line = line.split('\n')[0]
    #         cloth_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
    #     cloth_data = np.array(cloth_data)
    #     cloth_x, cloth_y, cloth_z = cloth_data[:,:3].T

    #     ball_data = []
    #     for line in open('../Data/0002/%03d_ball.txt' % i, 'r'):
    #         line = line.split('\n')[0]
    #         ball_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
    #     ball_data = np.array(ball_data)
    #     ball_x, ball_y, ball_z = ball_data[:,:3].T

    #     fig = plt.figure()
    #     ax = fig.gca(projection = '3d')
    #     ax.plot_trisurf(cloth_x, cloth_z, cloth_y, triangles = cloth_connection)
    #     ax.plot_trisurf(ball_x, ball_z, ball_y, triangles = ball_connection)
    #     ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    #     plt.savefig('../Results/%03d.png' % i)
    #     # plt.show()
    #     plt.close('all')
    
