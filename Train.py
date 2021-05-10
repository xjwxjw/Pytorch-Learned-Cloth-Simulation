import os
import numpy as np
import time 
import shutil
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from SphereDataset import SphereDataset, collate_fn
from Model import Encoder, Decoder, Processor, Processor_Res

def main():
    pretrained_model = None#'/home/xjwxjw/Documents/ClothSim/Models/2021-05-05-15:15:14'
    learning_rate = 1e-4
    batch_size = 1
    num_workers = 8
    shuffle = True
    train = True
    noise = True
    num_epochs = 5001
    beta0 = 0.9
    beta1 = 0.999
    use_scheduler = True

    process_steps = 15

    input_cloth_feature = 6
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

    spdataset = SphereDataset('../Data', 500, train, noise)
    adj_map = torch.from_numpy(spdataset.adj_map.astype(np.float32)).cuda().unsqueeze(0)
    adj_map = torch.cat([adj_map for i in range(batch_size)], 0)
    uvedge_node_i = spdataset.uvedge_node_i.astype(np.float32)
    uvedge_node_j = spdataset.uvedge_node_j.astype(np.float32)

    def truncated_normal_(tensor, mean = 0, std = 0.2):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size+(4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def init_weights(m):
        if type(m) == nn.Linear:
            truncated_normal_(m.weight)
            m.bias.data.fill_(0.0)

    if pretrained_model is not None:
        node_encoder = Encoder(input_cloth_feature, hidden_feature, hidden_feature, 'ln').cuda()
        node_encoder.load_state_dict(torch.load(os.path.join(pretrained_model, 'node_encoder.pkl')))
        
        uvedge_encoder = Encoder(input_uvedge_feature, hidden_feature, hidden_feature, 'ln').cuda()
        uvedge_encoder.load_state_dict(torch.load(os.path.join(pretrained_model, 'uvedge_encoder.pkl')))
        
        worldedge_encoder = Encoder(input_worldedge_feature - 2, hidden_feature, hidden_feature, 'ln').cuda()
        worldedge_encoder.load_state_dict(torch.load(os.path.join(pretrained_model, 'worldedge_encoder.pkl')))

        decoder = Decoder(hidden_feature, output_feature, hidden_feature).cuda()
        decoder.load_state_dict(torch.load(os.path.join(pretrained_model, 'decoder.pkl')))

        node_processor_list = []
        uvedge_processor_list = []
        worldedge_processor_list = []
        for l in range(process_steps):
            node_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3, 'ln').cuda())
            node_processor_list[-1].load_state_dict(torch.load(os.path.join(pretrained_model, 'node_processor_%02d.pkl' % l)))

            uvedge_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3, 'ln').cuda())
            uvedge_processor_list[-1].load_state_dict(torch.load(os.path.join(pretrained_model, 'uvedge_processor_%02d.pkl' % l)))

            worldedge_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3, 'ln').cuda())
            worldedge_processor_list[-1].load_state_dict(torch.load(os.path.join(pretrained_model, 'worldedge_processor_%02d.pkl' % l)))
        print("All pretrained models successfully loaded")
    else:
        node_encoder = Encoder(input_cloth_feature, hidden_feature, hidden_feature, 'ln').cuda()
        node_encoder.apply(init_weights)
        
        uvedge_encoder = Encoder(input_uvedge_feature, hidden_feature, hidden_feature, 'ln').cuda()
        uvedge_encoder.apply(init_weights)
        
        worldedge_encoder = Encoder(input_worldedge_feature - 2, hidden_feature, hidden_feature, 'ln').cuda()
        worldedge_encoder.apply(init_weights)

        decoder = Decoder(hidden_feature, output_feature, hidden_feature).cuda()
        decoder.apply(init_weights)

        node_processor_list = []
        uvedge_processor_list = []
        worldedge_processor_list = []
        for l in range(process_steps):
            node_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3, 'ln').cuda())
            node_processor_list[-1].apply(init_weights)

            uvedge_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3, 'ln').cuda())
            uvedge_processor_list[-1].apply(init_weights)

            worldedge_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3, 'ln').cuda())
            worldedge_processor_list[-1].apply(init_weights)

    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    sploader = DataLoader(spdataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, collate_fn = collate_fn, worker_init_fn=worker_init_fn)
    
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
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
    
    # world_feature = None
    for num_epoch in range(num_epochs):
        np.random.seed()
        for step, (cloth_state, ball_state, uv_state, worldcloth_state, worldball_state, cloth_nxt_state, worldcloth_adjmap, worldball_adjmap) in enumerate(sploader):                
            
            cloth_state = torch.stack([item for item in cloth_state], 0).cuda()
            ball_state = torch.stack([item for item in ball_state], 0).cuda()
            uv_state = torch.stack([item for item in uv_state], 0).cuda()
            cloth_nxt_state = torch.stack([item for item in cloth_nxt_state], 0).cuda()

            #### encoder part ####
            cloth_feature = node_encoder(cloth_state)
            ball_feature = node_encoder(ball_state)
            
            uvedge_feature = uvedge_encoder(uv_state)
            
            worldedge_state_list = []
            worldedge_node_i_index_list = []
            worldedge_node_j_index_list = []
            for bs in range(len(worldcloth_state)):
                if worldcloth_state[bs].size(0) > 0:
                    worldedge_state_list.append(worldcloth_state[bs])
                    node_i_index = worldcloth_state[bs][:, 0].detach().cpu().numpy()
                    node_j_index = worldcloth_state[bs][:, 1].detach().cpu().numpy()
                    worldedge_node_i_index_list.append(node_i_index)
                    worldedge_node_j_index_list.append(node_j_index)
                else:
                    worldedge_node_i_index_list.append([])
                    worldedge_node_j_index_list.append([])
            worldedgecloth_feature = None
            if len(worldedge_state_list) > 0:
                worldedge_state = torch.cat(worldedge_state_list).unsqueeze(0)
                worldedgecloth_feature = worldedge_encoder(worldedge_state[:, :, 2:].cuda())
            
            worldedge_state_list = []
            worldedge_node_i_index_list = []
            worldedge_node_j_index_list = []
            for bs in range(len(worldball_state)):
                if worldball_state[bs].size(0) > 0:
                    worldedge_state_list.append(worldball_state[bs])
                    node_i_index = worldball_state[bs][:, 0].detach().cpu().numpy()
                    node_j_index = worldball_state[bs][:, 1].detach().cpu().numpy()
                    worldedge_node_i_index_list.append(node_i_index)
                    worldedge_node_j_index_list.append(node_j_index)
                else:
                    worldedge_node_i_index_list.append([])
                    worldedge_node_j_index_list.append([])
            worldedgeball_feature = None
            if len(worldedge_state_list) > 0:
                worldedge_state = torch.cat(worldedge_state_list).unsqueeze(0)
                worldedgeball_feature = worldedge_encoder(worldedge_state[:, :, 2:].cuda())

            for l in range(process_steps):
                ### uv edge feature update ####    
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

            #### zero-out kinematic node ####
            # kinematic_node = [1, 645]
            # output[:, kinematic_node, :] = 0.0
            # cloth_nxt_state[:, kinematic_node, :] = 0.0

            loss = torch.sum((output - cloth_nxt_state) ** 2) / (output.size(0) * output.size(1))
            print(num_epoch, step, output[0, 100, :].detach().cpu().numpy(), cloth_nxt_state[0, 100, :].detach().cpu().numpy(), loss.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('train_loss', loss.detach().cpu().numpy(), global_step = num_epoch * total_step + step)

        torch.save(node_encoder.state_dict(), model_dir + '/node_encoder.pkl')
        torch.save(uvedge_encoder.state_dict(), model_dir + '/uvedge_encoder.pkl')
        torch.save(worldedge_encoder.state_dict(), model_dir + '/worldedge_encoder.pkl')
        for l in range(process_steps):
            torch.save(node_processor_list[l].state_dict(), model_dir + '/node_processor_%02d.pkl' % l)
            torch.save(uvedge_processor_list[l].state_dict(), model_dir + '/uvedge_processor_%02d.pkl' % l)
            torch.save(worldedge_processor_list[l].state_dict(), model_dir + '/worldedge_processor_%02d.pkl' % l)
        torch.save(decoder.state_dict(), model_dir + '/decoder.pkl')
        if use_scheduler:
            scheduler.step()

if __name__ == '__main__':
    main()
