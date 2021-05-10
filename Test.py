import os
import numpy as np
import time

import torch
import torch.nn as nn
from Model import Encoder, Decoder, Processor
from torch.utils.data import DataLoader
from SphereDataset import SphereDataset, collate_fn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def main():
    log_dir = '/home/xjwxjw/Documents/ClothSim/Models/2021-05-10-09:45:07'
    process_steps = 15
    train = False
    noise = False
    draw_ball = True

    input_cloth_feature = 6
    input_uvedge_feature = 7
    input_worldedge_feature = 6
    hidden_feature = 128
    output_feature = 3
    
    cloth_topo = np.load(os.path.join('../Data', 'cloth_connection.npy'), allow_pickle = True).item() 

    cloth_mesh_tri_array = []
    for line in open('../Data/cloth_connection.txt', 'r'):
        cloth_mesh_tri_array.append(int(line.split('\n')[0]))
    cloth_mesh_tri_array = np.array(cloth_mesh_tri_array).reshape(-1, 3)

    ball_mesh_tri_array = []
    for line in open('../Data/sphere_connection.txt', 'r'):
        ball_mesh_tri_array.append(int(line.split('\n')[0]))
    ball_mesh_tri_array = np.array(ball_mesh_tri_array).reshape(-1, 3)

    spdataset = SphereDataset('../Data', 500, train, noise)
    sploader = DataLoader(spdataset, batch_size = 1, shuffle = False, num_workers = 1, collate_fn = collate_fn)

    adj_map = torch.from_numpy(spdataset.adj_map.astype(np.float32)).cuda().unsqueeze(0)
    uvedge_node_i = spdataset.uvedge_node_i.astype(np.float32)
    uvedge_node_j = spdataset.uvedge_node_j.astype(np.float32)

    node_encoder = Encoder(input_cloth_feature, hidden_feature, hidden_feature, 'ln').cuda()
    node_encoder.load_state_dict(torch.load(os.path.join(log_dir, 'node_encoder.pkl')))
    node_encoder.eval()
    
    uvedge_encoder = Encoder(input_uvedge_feature, hidden_feature, hidden_feature, 'ln').cuda()
    uvedge_encoder.load_state_dict(torch.load(os.path.join(log_dir, 'uvedge_encoder.pkl')))
    uvedge_encoder.eval()
    
    worldedge_encoder = Encoder(input_worldedge_feature - 2, hidden_feature, hidden_feature, 'ln').cuda()
    worldedge_encoder.load_state_dict(torch.load(os.path.join(log_dir, 'worldedge_encoder.pkl')))
    worldedge_encoder.eval()

    decoder = Decoder(hidden_feature, output_feature, hidden_feature).cuda()
    decoder.load_state_dict(torch.load(os.path.join(log_dir, 'decoder.pkl')))
    decoder.eval()

    node_processor_list = []
    uvedge_processor_list = []
    worldedge_processor_list = []
    for l in range(process_steps):
        node_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3, 'ln').cuda())
        node_processor_list[-1].load_state_dict(torch.load(os.path.join(log_dir, 'node_processor_%02d.pkl' % l)))
        node_processor_list[-1].eval()

        uvedge_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3, 'ln').cuda())
        uvedge_processor_list[-1].load_state_dict(torch.load(os.path.join(log_dir, 'uvedge_processor_%02d.pkl' % l)))
        uvedge_processor_list[-1].eval()

        worldedge_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3, 'ln').cuda())
        worldedge_processor_list[-1].load_state_dict(torch.load(os.path.join(log_dir, 'worldedge_processor_%02d.pkl' % l)))
        worldedge_processor_list[-1].eval()
    print("All pretrained models successfully loaded")

    cloth_pre_pos = None
    cloth_cur_pos = None
    cloth_nxt_pos = None
    with torch.no_grad():
        for step, (cloth_state, ball_state, uv_state, worldcloth_state, worldball_state, cloth_pos, worldcloth_adjmap, worldball_adjmap) in enumerate(sploader):  
            ball_state = torch.stack([item for item in ball_state], 0).cuda()
            ball_nxt_pos = ball_state[:,:,6:]
            ball_state = ball_state[:,:,:6]
            if step == 0:
                cloth_pos = torch.stack([item for item in cloth_pos], 0).cuda()
                cloth_pre_pos = cloth_pos[:, :, 0:3]
                cloth_cur_pos = cloth_pos[:, :, 3:6]
                cloth_nxt_pos = cloth_pos[:, :, 6:9]
                tmp_std = torch.from_numpy(spdataset.cloth_nxt_std).unsqueeze(0).unsqueeze(0).cuda()
                tmp_mean = torch.from_numpy(spdataset.cloth_nxt_mean).unsqueeze(0).unsqueeze(0).cuda()

                cloth_state = torch.stack([item for item in cloth_state], 0).cuda()
                uv_state = torch.stack([item for item in uv_state], 0).cuda()
                worldedgecloth_state_list = []
                for bs in range(len(worldcloth_state)):
                    if worldcloth_state[bs].size(0) > 0:
                        worldedgecloth_state_list.append(worldcloth_state[bs])
                worldedgeball_state_list = []
                for bs in range(len(worldball_state)):
                    if worldball_state[bs].size(0) > 0:
                        worldedgeball_state_list.append(worldball_state[bs])
                
                fig = plt.figure()
                ax = fig.gca(projection='3d')

                cloth_cur_pos_np = cloth_cur_pos.detach().cpu().numpy()
                x = cloth_cur_pos_np[0,:,0]
                y = cloth_cur_pos_np[0,:,2]
                z = cloth_cur_pos_np[0,:,1]
                ax.plot_trisurf(x, y, z, triangles = cloth_mesh_tri_array, linewidth = 0.2, antialiased = True, color = 'b')
                if draw_ball:
                    ball_nxt_pos_np = ball_nxt_pos.detach().cpu().numpy()
                    x = ball_nxt_pos_np[0,:,0]
                    y = ball_nxt_pos_np[0,:,2]
                    z = ball_nxt_pos_np[0,:,1]
                    ax.plot_trisurf(x, y, z, triangles = ball_mesh_tri_array, linewidth = 0.2, antialiased = True, color = 'r')

                ax.set_xlim([-1.0, 1.0])
                ax.set_ylim([-1.0, 1.0])
                ax.set_zlim([-1.0, 1.0])
                plt.savefig('../Results/%03d.png' % (step))
                plt.close('all')
            else:
                #### compute the cloth node feature ####
                cloth_pre_pos = cloth_cur_pos.clone()
                cloth_cur_pos = cloth_nxt_pos.clone()
                cloth_cur_vel = cloth_cur_pos - cloth_pre_pos
                tmp_mean = torch.from_numpy(spdataset.cloth_mean[3:]).unsqueeze(0).unsqueeze(0).cuda()
                tmp_std = torch.from_numpy(spdataset.cloth_std[3:]).unsqueeze(0).unsqueeze(0).cuda()
                cloth_cur_vel = (cloth_cur_vel - tmp_mean) / tmp_std
                cloth_state = torch.cat([cloth_state[0][:, :3].unsqueeze(0).cuda(), cloth_cur_vel], -1)
                #### compute the uv edge feature ####
                cloth_uvworld_ = cloth_cur_pos[0, uvedge_node_i] - cloth_cur_pos[0, uvedge_node_j]
                cloth_uvworld_norm = torch.norm(cloth_uvworld_, p = 2, dim = -1, keepdim = True)
                cloth_uvworld_feature = torch.cat([cloth_uvworld_, cloth_uvworld_norm], -1)
                tmp_mean = torch.from_numpy(spdataset.uv_mean[3:]).unsqueeze(0).unsqueeze(0).cuda()
                tmp_std = torch.from_numpy(spdataset.uv_std[3:]).unsqueeze(0).unsqueeze(0).cuda()
                cloth_uvworld_feature = (cloth_uvworld_feature - tmp_mean) / tmp_std
                uv_state = torch.stack([item for item in uv_state], 0).cuda()
                uv_state[0, :, 3:] = cloth_uvworld_feature
                
                #### compute the world edge feature ####
                cloth_data = cloth_cur_pos[0].detach().cpu().numpy()
                ball_data = ball_nxt_pos[0].detach().cpu().numpy()

                #### collision between cloth and ball ####
                ball_world_dis = np.sum((cloth_data[None, :, :3] - ball_data[:, None, :3]) ** 2, -1) ** 0.5
                idxs_ball = np.argwhere(ball_world_dis < 0.04)
                worldball_state = []
                tmp_state_list = []
                tmp_adj_idx = []
                for idx in idxs_ball:
                    i_vertx = cloth_data[idx[1]]
                    j_vertx = ball_data[idx[0]]
                    xij = i_vertx[:3] - j_vertx[:3]
                    xij_norm = np.linalg.norm(xij, ord = 2)
                    worldball_feat = np.array([idx[1], idx[0], xij[0], xij[1], xij[2], xij_norm])
                    worldball_feat = (worldball_feat - spdataset.worldball_mean) / spdataset.worldball_std
                    tmp_state_list.append(torch.from_numpy(worldball_feat.astype(np.float32)).cuda())
                    tmp_adj_idx.append(idx[1])
                if len(tmp_state_list) == 0:
                    worldball_state.append(torch.from_numpy(np.array(tmp_state_list)))
                else:
                    worldball_state.append(torch.stack(tmp_state_list))
                
                worldball_adjmap = np.zeros((cloth_data.shape[0], len(tmp_state_list)))
                for i in range(worldball_adjmap.shape[1]):
                    worldball_adjmap[tmp_adj_idx[i], i] = 1.0
                if len(tmp_state_list) == 0:
                    worldball_adjmap = []
                else:
                    worldball_adjmap = [torch.from_numpy(worldball_adjmap.astype(np.float32))]
                
                worldedgeball_state_list = []
                for bs in range(len(worldball_state)):
                    if worldball_state[bs].size(0) > 0:
                        worldedgeball_state_list.append(worldball_state[bs])

                #### collision between cloth and cloth ####
                cloth_world_dis = np.sum((cloth_data[None, :, :3] - cloth_data[:, None, :3]) ** 2, -1) ** 0.5
                idxs_cloth = np.argwhere(cloth_world_dis < 0.02)
                worldcloth_state = []
                tmp_state_list = []
                tmp_adj_idx = []
                for idx in idxs_cloth:
                    i_vertx = cloth_data[idx[0]]
                    j_vertx = cloth_data[idx[1]]
                    if (idx[0] != idx[1]) and (idx[0] not in cloth_topo[idx[1]]):
                        xij = i_vertx[:3] - j_vertx[:3]
                        xij_norm = np.linalg.norm(xij, ord = 2)
                        worldcloth_feat = np.array([idx[0], idx[1], xij[0], xij[1], xij[2], xij_norm])
                        worldcloth_feat = (worldcloth_feat - spdataset.worldcloth_mean) / spdataset.worldcloth_std
                        tmp_state_list.append(torch.from_numpy(worldcloth_feat.astype(np.float32)).cuda())
                        tmp_adj_idx.append(idx[0])
                if len(tmp_state_list) == 0:
                    worldcloth_state.append(torch.from_numpy(np.array(tmp_state_list)))
                else:
                    worldcloth_state.append(torch.stack(tmp_state_list))
                
                worldcloth_adjmap = np.zeros((cloth_data.shape[0], len(tmp_state_list)))
                for i in range(worldcloth_adjmap.shape[1]):
                    worldcloth_adjmap[tmp_adj_idx[i], i] = 1.0
                if len(tmp_state_list) == 0:
                    worldcloth_adjmap = []
                else:
                    worldcloth_adjmap = [torch.from_numpy(worldcloth_adjmap.astype(np.float32))]

                worldedgecloth_state_list = []
                for bs in range(len(worldcloth_state)):
                    if worldcloth_state[bs].size(0) > 0:
                        worldedgecloth_state_list.append(worldcloth_state[bs])

            #### encoder part ####
            # print(step, 'input', cloth_state[0, 100, 3:].detach().cpu().numpy())
            cloth_feature = node_encoder(cloth_state)
            ball_feature = node_encoder(ball_state)
            uvedge_feature = uvedge_encoder(uv_state)

            worldedgecloth_feature = None
            if len(worldedgecloth_state_list) > 0:
                worldedgecloth_state = torch.cat(worldedgecloth_state_list).unsqueeze(0)
                worldedgecloth_feature = worldedge_encoder(worldedgecloth_state[:, :, 2:].cuda())
            
            worldedgeball_feature = None
            if len(worldedgeball_state_list) > 0:
                worldedgeball_state = torch.cat(worldedgeball_state_list).unsqueeze(0)
                worldedgeball_feature = worldedge_encoder(worldedgeball_state[:, :, 2:].cuda())

            for l in range(process_steps):
                #### uv edge feature update ####    
                uvedge_feature_cat = torch.cat([uvedge_feature, cloth_feature[:, uvedge_node_i], cloth_feature[:, uvedge_node_j]], -1)
                uvedge_nxt_feature = uvedge_processor_list[l](uvedge_feature_cat)
                
                ### cloth-ball world edge feature update ####
                if worldedgeball_feature is not None:
                    worldedge_feature_node_i_list = []
                    worldedge_feature_node_j_list = []
                    for bs in range(len(worldball_state)):
                        if worldball_state[bs].size(0) > 0:
                            node_i_index = worldball_state[bs][:, 0].detach().cpu().numpy()
                            node_j_index = worldball_state[bs][:, 1].detach().cpu().numpy() 
                            worldedge_feature_node_i_list.append(cloth_feature[bs, node_i_index])
                            worldedge_feature_node_j_list.append(ball_feature[bs, node_j_index])
                    worldedge_feature_node_i = torch.cat(worldedge_feature_node_i_list, 0).unsqueeze(0)
                    worldedge_feature_node_j = torch.cat(worldedge_feature_node_j_list, 0).unsqueeze(0)
                    worldedge_feature_cat = torch.cat([worldedgeball_feature, worldedge_feature_node_i, worldedge_feature_node_j], -1) 
                    worldedgeball_nxt_feature = worldedge_processor_list[l](worldedge_feature_cat)
                    #### NOTE: here we assume batch size is 1 ####
                    agr_worldball_feature = torch.matmul(worldball_adjmap[0].unsqueeze(0).cuda(), worldedgeball_nxt_feature)
                else:
                    agr_worldball_feature = torch.zeros((len(cloth_state), cloth_state[0].size(0), hidden_feature)).cuda()
                
                ### cloth-cloth world edge feature update ####
                if worldedgecloth_feature is not None:
                    worldedge_feature_node_i_list = []
                    worldedge_feature_node_j_list = []
                    for bs in range(len(worldcloth_state)):
                        if worldcloth_state[bs].size(0) > 0:
                            node_i_index = worldcloth_state[bs][:, 0].detach().cpu().numpy()
                            node_j_index = worldcloth_state[bs][:, 1].detach().cpu().numpy() 
                            worldedge_feature_node_i_list.append(cloth_feature[bs, node_i_index])
                            worldedge_feature_node_j_list.append(cloth_feature[bs, node_j_index])
                    worldedge_feature_node_i = torch.cat(worldedge_feature_node_i_list, 0).unsqueeze(0)
                    worldedge_feature_node_j = torch.cat(worldedge_feature_node_j_list, 0).unsqueeze(0)
                    worldedge_feature_cat = torch.cat([worldedgecloth_feature, worldedge_feature_node_i, worldedge_feature_node_j], -1) 
                    worldedgecloth_nxt_feature = worldedge_processor_list[l](worldedge_feature_cat)
                    #### NOTE: here we assume batch size is 1 ####
                    agr_worldcloth_feature = torch.matmul(worldcloth_adjmap[0].unsqueeze(0).cuda(), worldedgecloth_nxt_feature)
                else:
                    agr_worldcloth_feature = torch.zeros((len(cloth_state), cloth_state[0].size(0), hidden_feature)).cuda()

                ### node feature update ####
                agr_uv_feature = torch.matmul(adj_map[:cloth_feature.size(0)], uvedge_nxt_feature)
                cloth_feature_cat = torch.cat([cloth_feature, agr_uv_feature, agr_worldball_feature + agr_worldcloth_feature], -1)
                cloth_nxt_feature = node_processor_list[l](cloth_feature_cat)

                #### residual connection ####
                uvedge_feature = uvedge_feature + uvedge_nxt_feature
                if worldedgeball_feature is not None:
                    worldedgeball_feature = worldedgeball_feature + worldedgeball_nxt_feature
                if worldedgecloth_feature is not None:
                    worldedgecloth_feature = worldedgecloth_feature + worldedgecloth_nxt_feature
                cloth_feature = cloth_feature + cloth_nxt_feature

            output = decoder(cloth_feature)
            # print(step, output[0,100])
            #### use predicted acc to calculate the position ####
            tmp_std = torch.from_numpy(spdataset.cloth_nxt_std).unsqueeze(0).unsqueeze(0).cuda()
            tmp_mean = torch.from_numpy(spdataset.cloth_nxt_mean).unsqueeze(0).unsqueeze(0).cuda()
            output = (output * tmp_std) + tmp_mean
            output[0, 1] = 0.0
            output[0, 645] = 0.0
            cloth_nxt_pos = 2 * cloth_cur_pos + output - cloth_pre_pos
            # print(step, 'after', cloth_pre_pos[0,100], cloth_cur_pos[0,100], cloth_nxt_pos[0,100])
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            cloth_nxt_pos_np = cloth_nxt_pos.detach().cpu().numpy()
            x = cloth_nxt_pos_np[0,:,0]
            y = cloth_nxt_pos_np[0,:,2]
            z = cloth_nxt_pos_np[0,:,1]
            ax.plot_trisurf(x, y, z, triangles = cloth_mesh_tri_array, linewidth = 0.2, antialiased = True, color = 'b')
            if draw_ball:
                ball_nxt_pos_np = ball_nxt_pos.detach().cpu().numpy()
                x = ball_nxt_pos_np[0,:,0]
                y = ball_nxt_pos_np[0,:,2]
                z = ball_nxt_pos_np[0,:,1]
                ax.plot_trisurf(x, y, z, triangles = ball_mesh_tri_array, linewidth = 0.2, antialiased = True, color = 'r')

            ax.set_xlim([-1.0, 1.0])
            ax.set_ylim([-1.0, 1.0])
            ax.set_zlim([-1.0, 1.0])
            plt.savefig('../Results/%03d.png' % (step + 1))
            plt.close('all')

if __name__ == "__main__":
    main()
