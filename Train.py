import os
import numpy as np
import time 
import shutil
from tensorboardX import SummaryWriter

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
    num_epochs = 5001
    beta0 = 0.9
    beta1 = 0.999

    process_steps = 15

    input_node_feature = 6
    input_uvedge_feature = 7
    input_worldedge_feature = 6
    hidden_feature = 128
    output_feature = 3

    now = int(time.time())     
    timeArray = time.localtime(now)
    otherStyleTime = time.strftime("%Y-%m-%d-%H:%M:%S", timeArray)
    log_dir = '../Logs/%s' % otherStyleTime
    model_dir = '../Models/%s' % otherStyleTime

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    def copydirs(from_file, to_file):
            if not os.path.exists(to_file):  
                os.makedirs(to_file)
            files = os.listdir(from_file)  
            for f in files:
                if os.path.isdir(from_file + '/' + f):  
                    copydirs(from_file + '/' + f, to_file + '/' + f)  
                else:
                    if '.git' not in from_file:
                        shutil.copy(from_file + '/' + f, to_file + '/' + f) 
    copydirs('./', log_dir + '/Src')

    writer = SummaryWriter(log_dir)

    spdataset = SphereDataset('../Data', 500)
    adj_map = torch.from_numpy(spdataset.adj_map.astype(np.float32)).cuda().unsqueeze(0)
    adj_map = torch.cat([adj_map for i in range(batch_size)], 0)
    uvedge_node_i = spdataset.uvedge_node_i.astype(np.float32)
    uvedge_node_j = spdataset.uvedge_node_j.astype(np.float32)

    node_encoder = Encoder(input_node_feature, hidden_feature, hidden_feature).cuda()
    uvedge_encoder = Encoder(input_uvedge_feature, hidden_feature, hidden_feature).cuda()
    worldedge_encoder = Encoder(input_worldedge_feature - 2, hidden_feature, hidden_feature).cuda()

    decoder = Decoder(hidden_feature, output_feature, hidden_feature).cuda()

    node_processor_list = []
    uvedge_processor_list = []
    worldedge_processor_list = []
    for l in range(process_steps):
        node_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3).cuda())
        uvedge_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3).cuda())
        worldedge_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3).cuda())

    sploader = DataLoader(spdataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, collate_fn = collate_fn)
    
    node_encoder.train()
    uvedge_encoder.train()
    worldedge_encoder.train()
    for l in range(process_steps):
        node_processor_list[l].train()
        worldedge_processor_list[l].train()
        uvedge_processor_list[l].train()
    decoder.train()
    
    parm_list = []
    parm_list += node_encoder.parameters()
    parm_list += uvedge_encoder.parameters()
    parm_list += worldedge_encoder.parameters()
    for l in range(process_steps):
        parm_list += node_processor_list[l].parameters()
        parm_list += worldedge_processor_list[l].parameters()
        parm_list += uvedge_processor_list[l].parameters()
    parm_list += decoder.parameters()

    optimizer = optim.Adam(parm_list, lr=learning_rate, betas=(beta0, beta1))
    total_step = len(sploader)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
    for num_epoch in range(num_epochs):
        for step, (cloth_state, ball_state, uv_state, world_state, cloth_nxt_state) in enumerate(sploader):
            cloth_state = torch.stack([item for item in cloth_state], 0).cuda()
            ball_state = torch.stack([item for item in ball_state], 0).cuda()
            uv_state = torch.stack([item for item in uv_state], 0).cuda()
            cloth_nxt_state = torch.stack([item for item in cloth_nxt_state], 0).cuda()
            
            #### encoder part ####
            node_feature = node_encoder(cloth_state)
            uvedge_feature = uvedge_encoder(uv_state)
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

            worldedge_feature = None
            if len(worldedge_state_list) > 0:
                worldedge_state = torch.cat(worldedge_state_list).unsqueeze(0)
                worldedge_feature = worldedge_encoder(worldedge_state[:, :, 2:].cuda())

            for l in range(process_steps):
                ### uv edge feature update ####    
                uvedge_feature = torch.cat([uvedge_feature, node_feature[:, uvedge_node_i], node_feature[:, uvedge_node_j]], -1)
                uvedge_feature = uvedge_processor_list[l](uvedge_feature)
                
                ### world edge feature update ####
                if worldedge_feature is not None:
                    worldedge_feature_node_i_list = []
                    worldedge_feature_node_j_list = []
                    for bs in range(len(world_state)):
                        if world_state[bs].size(0) > 0:
                            node_i_index = world_state[bs][:, 0].detach().cpu().numpy()
                            node_j_index = world_state[bs][:, 1].detach().cpu().numpy() 
                            worldedge_feature_node_i_list.append(node_feature[bs, node_i_index])
                            worldedge_feature_node_j_list.append(node_feature[bs, node_j_index])
                    worldedge_feature_node_i = torch.cat(worldedge_feature_node_i_list, 0).unsqueeze(0)
                    worldedge_feature_node_j = torch.cat(worldedge_feature_node_j_list, 0).unsqueeze(0)
                    worldedge_feature = torch.cat([worldedge_feature, worldedge_feature_node_i, worldedge_feature_node_j], -1) 
                    worldedge_feature = worldedge_processor_list[l](worldedge_feature)

                ### node feature update ####
                agr_uv_feature = torch.matmul(adj_map[:node_feature.size(0)], uvedge_feature)
                agr_world_feature = torch.ones((node_feature.size(0), cloth_state.size(1), hidden_feature)).cuda()
                for bs in range(len(world_state)):
                    cnt = 0
                    if world_state[bs].size(0) > 0:
                        agr_world_feature[bs, worldedge_node_i_index_list[bs]] = worldedge_feature[0, cnt:cnt+len(worldedge_node_i_index_list[bs])]
                        cnt += len(worldedge_node_i_index_list[bs])
                node_feature = torch.cat([node_feature, agr_uv_feature, agr_world_feature], -1)
                node_feature = node_processor_list[l](node_feature)
            
            output = decoder(node_feature)
            loss = torch.mean(torch.abs(output - cloth_nxt_state))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("epoch: %04d, step: %04d, loss: %06f" % (num_epoch, step, loss))
            writer.add_scalar('train_loss', loss.detach().cpu().numpy(), global_step = num_epoch * total_step + step)

        if num_epoch % 10 == 0:
            torch.save(node_encoder.state_dict(), model_dir + '/node_encoder.pkl')
            torch.save(uvedge_encoder.state_dict(), model_dir + '/uvedge_encoder.pkl')
            torch.save(worldedge_encoder.state_dict(), model_dir + '/worldedge_encoder.pkl')
            for l in range(process_steps):
                torch.save(node_processor_list[l], model_dir + '/node_processor_%02d.pkl' % l)
                torch.save(uvedge_processor_list[l], model_dir + '/uvedge_processor_%02d.pkl' % l)
                torch.save(worldedge_processor_list[l], model_dir + '/worldedge_processor_%02d.pkl' % l)
            torch.save(decoder.state_dict(), model_dir + '/decoder.pkl')
        scheduler.step()

if __name__ == '__main__':
    main()
