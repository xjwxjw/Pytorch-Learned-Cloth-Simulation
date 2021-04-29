import os
import numpy as np
import time

import torch
import torch.nn as nn
from Model import Encoder, Decoder, Processor
from torch.utils.data import DataLoader
from SphereDataset import SphereDataset, collate_fn

def main():
    log_dir = '/home/xjwxjw/Documents/ClothSim/Models/2021-04-29-11:31:39'
    process_steps = 15

    input_cloth_feature = 5
    input_uvedge_feature = 7
    input_worldedge_feature = 6
    hidden_feature = 128
    output_feature = 3

    spdataset = SphereDataset('../Data', 500)
    adj_map = torch.from_numpy(spdataset.adj_map.astype(np.float32)).cuda().unsqueeze(0)
    uvedge_node_i = spdataset.uvedge_node_i.astype(np.float32)
    uvedge_node_j = spdataset.uvedge_node_j.astype(np.float32)

    node_encoder = Encoder(input_cloth_feature, hidden_feature, hidden_feature, 'bn').cuda()
    node_encoder.load_state_dict(torch.load(os.path.join(log_dir, 'node_encoder.pkl')))
    node_encoder.eval()
    
    uvedge_encoder = Encoder(input_uvedge_feature, hidden_feature, hidden_feature, 'bn').cuda()
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
        node_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3, 'bn').cuda())
        node_processor_list[-1].load_state_dict(torch.load(os.path.join(log_dir, 'node_processor_%02d.pkl' % l)))
        node_processor_list[-1].eval()

        uvedge_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3, 'bn').cuda())
        uvedge_processor_list[-1].load_state_dict(torch.load(os.path.join(log_dir, 'uvedge_processor_%02d.pkl' % l)))
        uvedge_processor_list[-1].eval()

        worldedge_processor_list.append(Processor(hidden_feature * 3, hidden_feature, hidden_feature * 3, 'ln').cuda())
        worldedge_processor_list[-1].load_state_dict(torch.load(os.path.join(log_dir, 'worldedge_processor_%02d.pkl' % l)))
        worldedge_processor_list[-1].eval()
    print("All pretrained models successfully loaded")

if __name__ == "__main__":
    main()