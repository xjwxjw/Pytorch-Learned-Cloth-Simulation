import os
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from SphereDataset import SphereDataset, collate_fn
from Model import Encoder, Decoder, Processor

def main():
    learning_rate = 1e-4
    batch_size = 2
    num_workers = 2
    shuffle = True
    num_epoch = 5000
    beta0 = 0.9
    beta1 = 0.999

    input_node_feature = 6
    input_uvedge_feature = 7
    input_worldedge_feature = 6
    hidden_feature = 128
    output_feature = 3

    spdataset = SphereDataset('../Data', 500)
    adj_map = torch.from_numpy(spdataset.adj_map.astype(np.float32)).cuda().unsqueeze(0)
    adj_map = torch.cat([adj_map for i in range(batch_size)], 0)
    uvedge_node_i = spdataset.uvedge_node_i.astype(np.float32)
    uvedge_node_j = spdataset.uvedge_node_j.astype(np.float32)

    node_encoder = Encoder(input_node_feature, hidden_feature, hidden_feature).cuda()
    uvedge_encoder = Encoder(input_uvedge_feature, hidden_feature, hidden_feature).cuda()
    worldedge_encoder = Encoder(input_worldedge_feature - 2, hidden_feature, hidden_feature).cuda()

    node_processor = Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3).cuda()
    uvedge_processor = Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3).cuda()
    worldedge_processor = Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3).cuda()

    sploader = DataLoader(spdataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, collate_fn = collate_fn)
    for cloth_state, ball_state, uv_state, world_state, cloth_nxt_state in sploader:
        cloth_state = torch.stack([item for item in cloth_state], 0).cuda()
        ball_state = torch.stack([item for item in ball_state], 0).cuda()
        uv_state = torch.stack([item for item in uv_state], 0).cuda()
        cloth_nxt_state = torch.stack([item for item in cloth_nxt_state], 0).cuda()
        
        #### encoder part ####
        node_feature = node_encoder(cloth_state)
        uvedge_feature = uvedge_encoder(uv_state)
        worldedge_state_list = []
        worldedge_feature_node_i_list = []
        worldedge_node_i_index_list = []
        worldedge_node_j_index_list = []
        worldedge_feature_node_j_list = []
        for bs in range(len(world_state)):
            if world_state[bs].size(0) > 0:
                worldedge_state_list.append(world_state[bs])
                node_i_index = world_state[bs][:, 0].detach().cpu().numpy()
                node_j_index = world_state[bs][:, 1].detach().cpu().numpy()
                worldedge_node_i_index_list.append(node_i_index)
                worldedge_node_j_index_list.append(node_j_index)
                worldedge_feature_node_i_list.append(node_feature[bs, node_i_index])
                worldedge_feature_node_j_list.append(node_feature[bs, node_j_index])
            else:
                worldedge_node_i_index_list.append([])
                worldedge_node_j_index_list.append([])

        worldedge_feature = None
        if len(worldedge_state_list) > 0:
            worldedge_state = torch.cat(worldedge_state_list).unsqueeze(0)
            worldedge_feature_node_i = torch.cat(worldedge_feature_node_i_list).unsqueeze(0)
            worldedge_feature_node_j = torch.cat(worldedge_feature_node_j_list).unsqueeze(0)
            worldedge_feature = worldedge_encoder(worldedge_state[:, :, 2:].cuda())
            worldedge_feature = torch.cat([worldedge_feature, worldedge_feature_node_i, worldedge_feature_node_j], -1)

        ### uv edge feature update ####    
        uvedge_feature = torch.cat([uvedge_feature, node_feature[:, uvedge_node_i], node_feature[:, uvedge_node_j]], -1)
        uvedge_feature = uvedge_processor(uvedge_feature)
        
        ### world edge feature update ####
        if worldedge_feature is not None:
            worldedge_feature = worldedge_processor(worldedge_feature)

        ### node feature update ####
        agr_uv_feature = torch.matmul(adj_map, uvedge_feature)
        agr_world_feature = torch.zeros((batch_size, cloth_state.size(1), hidden_feature)).cuda()
        for bs in range(len(world_state)):
            cnt = 0
            if world_state[bs].size(0) > 0:
                agr_world_feature[bs, worldedge_node_i_index_list[bs]] = worldedge_feature[0, cnt:cnt+len(worldedge_node_i_index_list[bs])]
                cnt += len(worldedge_node_i_index_list[bs])
        node_feature = torch.cat([node_feature, agr_uv_feature, agr_world_feature], -1)
        node_feature = node_processor(node_feature)

if __name__ == '__main__':
    main()
