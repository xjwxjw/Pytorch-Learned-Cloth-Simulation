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
    log_dir = '/home/xjwxjw/Documents/ClothSim/Models/2021-05-05-15:15:14'
    process_steps = 15
    train = False
    noise = False

    input_cloth_feature = 5
    input_uvedge_feature = 7
    input_worldedge_feature = 6
    hidden_feature = 128
    output_feature = 3

    mesh_tri_array = []
    cnt = 0
    for line in open('../Data/cloth_connection.txt', 'r'):
        mesh_tri_array.append(int(line.split('\n')[0]))
    mesh_tri_array = np.array(mesh_tri_array).reshape(-1, 3)

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
    acc_pred_list = []
    acc_gt_list = []
    with torch.no_grad():
        for step, (cloth_state, ball_state, uv_state, world_state, cloth_pos) in enumerate(sploader):   
            cloth_pos = torch.stack([item for item in cloth_pos], 0).cuda()
            cloth_pre_pos = cloth_pos[:, :, 0:3]
            cloth_cur_pos = cloth_pos[:, :, 3:6]
            cloth_nxt_pos_gt = cloth_pos[:, :, 6:9]
            print(step)
            for t in range(500):
                ball_state = torch.stack([item for item in ball_state], 0).cuda()
                if t == 0:
                    cloth_state = torch.stack([item for item in cloth_state], 0).cuda()
                    uv_state = torch.stack([item for item in uv_state], 0).cuda()
                    
                    worldedge_state_list = []
                    worldedge_node_i_index_list = []
                    worldedge_node_j_index_list = []
                    for bs in range(len(world_state)):
                        if world_state[bs].size(0) > 0:
                            worldedge_state_list.append(world_state[bs])
                            node_i_index = world_state[bs][:, 0].detach().cpu().numpy()
                            node_j_index = world_state[bs][:, 1].detach().cpu().numpy()
                            worldedge_node_i_index_list.append(node_i_index)
                            worldedge_node_j_index_list.append(node_j_index)
                        else:
                            worldedge_node_i_index_list.append([])
                            worldedge_node_j_index_list.append([])
                else:
                    #### compute the cloth node feature ####
                    cloth_pre_pos = cloth_cur_pos.clone()
                    cloth_cur_pos = cloth_nxt_pos.clone()
                    cloth_cur_vel = cloth_cur_pos - cloth_pre_pos
                    # print(cloth_cur_vel[0,645])
                    tmp_mean = torch.from_numpy(spdataset.cloth_mean[2:]).unsqueeze(0).unsqueeze(0).cuda()
                    tmp_std = torch.from_numpy(spdataset.cloth_std[2:]).unsqueeze(0).unsqueeze(0).cuda()
                    cloth_cur_vel = (cloth_cur_vel - tmp_mean) / tmp_std
                    cloth_state = torch.cat([cloth_state[:, :, :2], cloth_cur_vel], -1)

                    #### compute the uv edge feature ####
                    cloth_uvworld_ = cloth_cur_pos[0, uvedge_node_i] - cloth_cur_pos[0, uvedge_node_j]
                    cloth_uvworld_norm = torch.norm(cloth_uvworld_, p = 2, dim = -1, keepdim = True)
                    cloth_uvworld_feature = torch.cat([cloth_uvworld_, cloth_uvworld_norm], -1)
                    tmp_mean = torch.from_numpy(spdataset.uv_mean[3:]).unsqueeze(0).unsqueeze(0).cuda()
                    tmp_std = torch.from_numpy(spdataset.uv_std[3:]).unsqueeze(0).unsqueeze(0).cuda()
                    cloth_uvworld_feature = (cloth_uvworld_feature - tmp_mean) / tmp_std
                    uv_state[0, :, 3:] = cloth_uvworld_feature
                    
                    #### compute the world edge feature ####
                    cloth_data = cloth_cur_pos[0].detach().cpu().numpy()
                    ball_data = ball_state[0].detach().cpu().numpy()
                    ball_world_dis = np.sum((cloth_data[None, :, :3] - ball_data[:, None, :3])**2, -1)**0.5
                    # idxs_ball = np.argwhere(ball_world_dis < 0.02)
                    idxs_ball = []
                    world_state = []
                    tmp_state_list = []
                    #### currently, do not consider the collision between cloth and cloth ####
                    for idx in idxs_ball:
                        i_vertx = cloth_data[idx[1]]
                        j_vertx = ball_data[idx[0]]
                        xij = i_vertx[:3] - j_vertx[:3]
                        xij_norm = np.linalg.norm(xij, ord = 2)
                        tmp_state_list.append(torch.from_numpy(np.array([idx[1], idx[0], xij[0], xij[1], xij[2], xij_norm])).cuda())
                    if len(tmp_state_list) == 0:
                        world_state.append(torch.from_numpy(np.array(tmp_state_list)))
                    else:
                        world_state.append(torch.stack(tmp_state_list))

                    worldedge_state_list = []
                    worldedge_node_i_index_list = []
                    worldedge_node_j_index_list = []
                    for bs in range(len(world_state)):
                        if world_state[bs].size(0) > 0:
                            worldedge_state_list.append(world_state[bs])
                            node_i_index = world_state[bs][:, 0].detach().cpu().numpy()
                            node_j_index = world_state[bs][:, 1].detach().cpu().numpy()
                            worldedge_node_i_index_list.append(node_i_index)
                            worldedge_node_j_index_list.append(node_j_index)
                        else:
                            worldedge_node_i_index_list.append([])
                            worldedge_node_j_index_list.append([])
                
                #### encoder part ####
                cloth_feature = node_encoder(cloth_state)
                ball_feature = node_encoder(ball_state)
                uvedge_feature = uvedge_encoder(uv_state)

                worldedge_feature = None
                if len(worldedge_state_list) > 0:
                    worldedge_state = torch.cat(worldedge_state_list).unsqueeze(0)
                    worldedge_feature = worldedge_encoder(worldedge_state[:, :, 2:].cuda())

                for l in range(process_steps):
                    ### uv edge feature update ####    
                    uvedge_feature_cat = torch.cat([uvedge_feature, cloth_feature[:, uvedge_node_i], cloth_feature[:, uvedge_node_j]], -1)
                    uvedge_nxt_feature = uvedge_processor_list[l](uvedge_feature_cat)
                    
                    ### world edge feature update ####
                    if worldedge_feature is not None:
                        worldedge_feature_node_i_list = []
                        worldedge_feature_node_j_list = []
                        for bs in range(len(world_state)):
                            if world_state[bs].size(0) > 0:
                                node_i_index = world_state[bs][:, 0].detach().cpu().numpy()
                                node_j_index = world_state[bs][:, 1].detach().cpu().numpy() 
                                worldedge_feature_node_i_list.append(cloth_feature[bs, node_i_index])
                                worldedge_feature_node_j_list.append(ball_feature[bs, node_j_index])
                        worldedge_feature_node_i = torch.cat(worldedge_feature_node_i_list, 0).unsqueeze(0)
                        worldedge_feature_node_j = torch.cat(worldedge_feature_node_j_list, 0).unsqueeze(0)
                        worldedge_feature_cat = torch.cat([worldedge_feature, worldedge_feature_node_i, worldedge_feature_node_j], -1) 
                        worldedge_nxt_feature = worldedge_processor_list[l](worldedge_feature_cat)

                    ### node feature update ####
                    agr_uv_feature = torch.matmul(adj_map[:cloth_feature.size(0)], uvedge_nxt_feature)
                    agr_world_feature = torch.zeros((len(cloth_state), cloth_state[0].size(0), hidden_feature)).cuda()
                    cnt = 0
                    for bs in range(len(world_state)):
                        if world_state[bs].size(0) > 0:
                            for node_i_idx in worldedge_node_i_index_list[bs]:
                                agr_world_feature[bs, int(node_i_idx)] += worldedge_nxt_feature[0, cnt]
                                cnt += 1
                    cloth_feature_cat = torch.cat([cloth_feature, agr_uv_feature, agr_world_feature], -1)
                    cloth_nxt_feature = node_processor_list[l](cloth_feature_cat)

                    uvedge_feature = uvedge_feature + uvedge_nxt_feature
                    if worldedge_feature is not None:
                        worldedge_feature = worldedge_feature + worldedge_nxt_feature
                    cloth_feature = cloth_feature + cloth_nxt_feature
                
                output = decoder(cloth_feature)

                #### use predicted acc to aclculate the position ####
                tmp_std = torch.from_numpy(spdataset.cloth_nxt_std).unsqueeze(0).unsqueeze(0).cuda()
                tmp_mean = torch.from_numpy(spdataset.cloth_nxt_mean).unsqueeze(0).unsqueeze(0).cuda()
                output = (output * tmp_std) + tmp_mean
                output[0, 1] = 0.0
                output[0, 645] = 0.0
                cloth_nxt_pos = 2 * cloth_cur_pos + output - cloth_pre_pos
                # cloth_nxt_pos -= (cloth_nxt_pos[0, 1] + cloth_nxt_pos[0, 645]) / 2.0

                cloth_nxt_pos_np = cloth_nxt_pos.detach().cpu().numpy()
                output_np = output.detach().cpu().numpy()
                acc_gt_list.append(cloth_nxt_pos_gt + cloth_pre_pos - 2 * cloth_cur_pos)
                acc_pred_list.append(output_np)
                x = cloth_nxt_pos_np[0,:,0]
                y = cloth_nxt_pos_np[0,:,2]
                z = cloth_nxt_pos_np[0,:,1]
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.plot_trisurf(x, y, z, triangles=mesh_tri_array, linewidth=0.2, antialiased=True)
                ax.set_xlim([-1.0, 1.0])
                ax.set_ylim([-1.0, 1.0])
                ax.set_zlim([-1.0, 1.0])
                plt.savefig('../Results/%03d.png' % t)
                plt.close('all')
        # acc_gt_list = np.concatenate(acc_gt_list, 1)
        # acc_pred_list = np.concatenate(acc_pred_list, 1)
        # plt.hist(acc_gt_list[0, :, 0] - acc_pred_list[0, :, 0], 200, color = 'r')
        # # plt.hist(acc_pred_list[0, :, 0], 200, color = 'b')
        # plt.show()
        # plt.hist(acc_gt_list[0, :, 1] - acc_pred_list[0, :, 1], 200, color = 'r')
        # # plt.hist(acc_pred_list[0, :, 1], 200, color = 'b')
        # plt.show()
        # plt.hist(acc_gt_list[0, :, 2] - acc_pred_list[0, :, 2], 200, color = 'r')
        # # plt.hist(acc_pred_list[0, :, 2], 200, color = 'b')
        # plt.show()
        # print(np.std(err_list, (0, 1)))

if __name__ == "__main__":
    main()
