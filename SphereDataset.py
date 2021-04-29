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
    # data5 = [item[5] for item in batch]
    return [data0, data1, data2, data3, data4]

class SphereDataset(Dataset):
    def __init__(self, data_dir, length):
        self.data = []
        # self.seq = []
        self.length = length
        for data_id in range(1):
            state_dir = os.path.join(data_dir, ('data_0424/%04d' % data_id))
            for file_id in range(2, 499):
                pre_file = os.path.join(state_dir, '%03d_cloth.txt' % (file_id - 1))
                cur_file = os.path.join(state_dir, '%03d_cloth.txt' % file_id)
                nxt_file = os.path.join(state_dir, '%03d_cloth.txt' % (file_id + 1))
                self.data.append([pre_file, cur_file, nxt_file])
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

        state_stat = np.load('state_stat.npz')
        self.cloth_mean = state_stat['arr_0'].item()['cloth_mean']
        self.cloth_std = state_stat['arr_0'].item()['cloth_std']
        self.ball_mean = state_stat['arr_0'].item()['ball_mean']
        self.ball_std = state_stat['arr_0'].item()['ball_std']
        self.uv_mean = state_stat['arr_0'].item()['uv_mean']
        self.uv_std = state_stat['arr_0'].item()['uv_std']
        self.world_mean = state_stat['arr_0'].item()['world_mean']
        self.world_std = state_stat['arr_0'].item()['world_std']
        self.cloth_nxt_mean = state_stat['arr_0'].item()['cloth_nxt_mean']
        self.cloth_nxt_std = state_stat['arr_0'].item()['cloth_nxt_std']
        
    def __len__(self):
        return len(self.data) 

    def GetState(self, index, time_id):
        #### get previous state to calculate the velocity ####
        cloth_pre_file = self.data[index][0]
        cloth_pre_data = []
        with open(cloth_pre_file, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                cloth_pre_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
            cloth_pre_data = np.array(cloth_pre_data)
            f.close()

        ball_pre_file = self.data[index][0].replace('cloth', 'ball')
        ball_pre_data = []
        with open(ball_pre_file, 'r'):
            for line in open(ball_pre_file, 'r'):
                line = line.split('\n')[0]
                ball_pre_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
            ball_pre_data = np.array(ball_pre_data)
            f.close()


        #### get current state, including vertex and edge information, as input ####
        cloth_file = self.data[index][1]
        ball_file = self.data[index][1].replace('cloth', 'ball')
        uv_file = self.data[index][1].replace('cloth', 'uv')
        world_file = self.data[index][1].replace('cloth', 'world')

        cloth_data = []
        ball_data = []
        uv_data = []
        world_data = []
        # worldmap = np.zeros((1946, 252))
        with open(cloth_file, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                cloth_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
            f.close()
        
        with open(ball_file, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                ball_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
            f.close()
        
        with open(uv_file, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                uv_data.append(np.array([float(data) for data in line.split(' ')]))
            f.close()

        with open(world_file, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                feat = np.array([float(data) for data in line.split(' ')])
                # worldmap[int(feat[0]), int(feat[1])] = 1.0
                world_data.append(feat)
            f.close()
            
        cloth_data = np.array(cloth_data)
        ball_data = np.array(ball_data)
        uv_data = np.array(uv_data)
        world_data = np.array(world_data)
        
        #### get next state, mainly the position, as output ####
        cloth_nxt_file = self.data[index][2]
        cloth_nxt_data = []
        with open(cloth_nxt_file, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                cloth_nxt_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
            f.close()
        cloth_nxt_data = np.array(cloth_nxt_data)
        
        #### get next kinematics node information, mainly the position, as actuator ####
        ball_nxt_file = self.data[index][2].replace('cloth', 'ball')
        ball_nxt_data = []
        with open(ball_nxt_file, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                ball_nxt_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
            f.close()
        ball_nxt_data = np.array(ball_nxt_data)

        #### get the velocity information ####
        cloth_vel = cloth_data[:, :3] - cloth_pre_data[:, :3]
        cloth_label = np.zeros((cloth_vel.shape[0], 2))
        cloth_label[:, 0] = 1.0
        cloth_state = np.concatenate([cloth_label, cloth_vel], -1)

        ball_vel = ball_nxt_data[:, :3] - ball_data[:, :3]
        ball_label = np.zeros((ball_vel.shape[0], 2))
        ball_label[:, 1] = 1.0
        ball_state = np.concatenate([ball_label, ball_vel], -1)

        #### get the final state information ####
        cloth_acc = cloth_nxt_data[:, :3] + cloth_pre_data[:, :3] - 2 * cloth_data[:, :3]
        cloth_state = (cloth_state - self.cloth_mean) / self.cloth_std
        ball_state = (ball_state - self.ball_mean) / self.ball_std
        uv_data = (uv_data - self.uv_mean) / self.uv_std
        if len(world_data) > 0:
            world_data = (world_data - self.world_mean) / self.world_std
        cloth_acc = (cloth_acc - self.cloth_nxt_mean) / self.cloth_nxt_std
        return cloth_state, ball_state, uv_data, world_data, cloth_acc#, worldmap

    def __getitem__(self, index):
        time_id = np.random.choice(self.length - 3, 1)[0] + 2
        cloth_state, ball_state, uv_state, world_state, cloth_nxt_state = self.GetState(index, time_id)
        return torch.from_numpy(cloth_state.astype(np.float32)),\
               torch.from_numpy(ball_state.astype(np.float32)), \
               torch.from_numpy(uv_state.astype(np.float32)),\
               torch.from_numpy(world_state.astype(np.float32)),\
               torch.from_numpy(cloth_nxt_state.astype(np.float32))

def GenDataStatics():
    spdataset = SphereDataset('../Data', 500)
    sploader = DataLoader(spdataset, batch_size = 32, shuffle = False, num_workers = 48, collate_fn = collate_fn)
    cloth_state_list = []
    ball_state_list = []
    uv_state_list = []
    world_state_list = []
    cloth_nxt_state_list = []
    torch.multiprocessing.set_sharing_strategy('file_system')
    for step, (cloth_state, ball_state, uv_state, world_state, cloth_nxt_state) in enumerate(sploader):
        print(step)
        for bs in range(len(cloth_state)):
            cloth_state_list.append(cloth_state[bs])
            ball_state_list.append(ball_state[bs])
            uv_state_list.append(uv_state[bs])
            if world_state[bs].shape[0] > 0:
                world_state_list.append(world_state[bs])
            cloth_nxt_state_list.append(cloth_nxt_state[bs])

    #### the conversion to np.float64 is very important to prevent bound error ####
    cloth_state = np.concatenate(cloth_state_list, 0)
    cloth_state = cloth_state.astype(np.float64)
    cloth_state_mean = np.mean(cloth_state, 0)
    cloth_state_std = np.std(cloth_state, 0)

    ball_state = np.concatenate(ball_state_list, 0)
    ball_state = ball_state.astype(np.float64)
    ball_state_mean = np.mean(ball_state, 0)
    ball_state_std = np.std(ball_state, 0)
    
    uv_state = np.concatenate(uv_state_list, 0)
    uv_state = uv_state.astype(np.float64)
    uv_state_mean = np.mean(uv_state, 0)
    uv_state_std = np.std(uv_state, 0)

    world_state = np.concatenate(world_state_list, 0)
    world_state = world_state.astype(np.float64)
    world_state_mean = np.mean(world_state, 0)
    world_state_std = np.std(world_state, 0)

    cloth_nxt_state = np.concatenate(cloth_nxt_state_list, 0)
    cloth_nxt_state = cloth_nxt_state.astype(np.float64)
    cloth_nxt_state_mean = np.mean(cloth_nxt_state, 0)
    cloth_nxt_state_std = np.std(cloth_nxt_state, 0)

    cloth_state_std[:2] = 1.0
    ball_state_std[:2] = 1.0
    ball_state_std[3:] = 1.0
    world_state_mean[:2] = 0.0
    world_state_std[:2] = 1.0

    print('cloth_state:', cloth_state_mean, cloth_state_std)
    print('ball_state:', ball_state_mean, ball_state_std)
    print('uv_state:', uv_state_mean, uv_state_std)
    print('world_state:', world_state_mean, world_state_std)
    print('cloth_nxt_state:', cloth_nxt_state_mean, cloth_nxt_state_std)
    np.savez('state_stat.npz', {'cloth_mean':cloth_state_mean.astype(np.float32), 'cloth_std':cloth_state_std.astype(np.float32),\
                                'ball_mean':ball_state_mean.astype(np.float32), 'ball_std':ball_state_std.astype(np.float32),\
                                'uv_mean':uv_state_mean.astype(np.float32), 'uv_std':uv_state_std.astype(np.float32),\
                                'world_mean':world_state_mean.astype(np.float32), 'world_std':world_state_std.astype(np.float32),\
                                'cloth_nxt_mean':cloth_nxt_state_mean.astype(np.float32), 'cloth_nxt_std':cloth_nxt_state_std.astype(np.float32)})

if __name__ == '__main__':    
    GenDataStatics()
    
    
