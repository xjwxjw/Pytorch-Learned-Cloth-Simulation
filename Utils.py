import torch
import numpy as np
import os

def GenDataStat():
    target_list = ['cloth', 'ball']
    for target in target_list:
        target_data = []
        for i in range(500):
            target_path = os.path.join('../Data/0002/%03d_%s.txt' % (i, target))
            for line in open(target_path, 'r'):
                line = line.split('\n')[0]
                target_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
        target_data = np.array(target_data)
        target_mean = np.mean(target_data, 0)
        target_std = np.std(target_data, 0)
        np.save('../Data/%s_mean.npy' % target, target_mean)
        np.save('../Data/%s_std.npy' % target, target_std + 1e-10)
        print('%s_mean:' % target, target_mean)
        print('%s_std:' % target, target_std + 1e-10)

def GenEdgeFeature():
    cloth_connection = np.load('../Data/cloth_connection.npy').item()    
    ball_connection = np.load('../Data/sphere_connection.npy').item()    
    for i in range(500):
        print(i)
        cloth_data = []
        cloth_path = ('../Data/0002/%03d_cloth.txt' % i)
        for line in open(cloth_path, 'r'):
            line = line.split('\n')[0]
            cloth_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
        cloth_data = np.array(cloth_data)

        ball_data = []
        ball_path = ('../Data/0002/%03d_ball.txt' % i)
        for line in open(ball_path, 'r'):
            line = line.split('\n')[0]
            ball_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
        ball_data = np.array(ball_data)

        foutuv = open('../Data/0002/%03d_uv.txt' % i, 'w')
        for key in cloth_connection.keys():
            for val in cloth_connection[key]:
                i_vertx = cloth_data[key]
                j_vertx = cloth_data[val]
                uij = i_vertx[15:18] - j_vertx[15:18]
                uij_norm = np.linalg.norm(uij, ord = 2)
                xij = i_vertx[:3] - j_vertx[:3]
                xij_norm = np.linalg.norm(xij, ord = 2)
                foutuv.write("%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % \
                            (uij[0], uij[1], uij_norm, xij[0], xij[1], xij[2], xij_norm))
        foutuv.close()

        foutworld = open('../Data/0002/%03d_world.txt' % i, 'w')
        cloth_world_dis = np.sum((cloth_data[None, :, :3] - cloth_data[:, None, :3])**2, -1)**0.5
        ball_world_dis = np.sum((cloth_data[None, :, :3] - ball_data[:, None, :3])**2, -1)**0.5
        idxs_cloth = np.argwhere(cloth_world_dis < 0.015)
        idxs_ball = np.argwhere(ball_world_dis < 0.015)
        for idx in idxs_cloth:
            i_vertx = cloth_data[idx[0]]
            j_vertx = cloth_data[idx[1]]
            xij = i_vertx[:3] - j_vertx[:3]
            xij_norm = np.linalg.norm(xij, ord = 2)
            if xij_norm < 0.015 and (idx[0] != idx[1]):
                #print("cloth, %d, %d" % (idx[0], idx[1]))
                foutworld.write("%d %.6f %.6f %.6f %.6f\n" % (idx[0], xij[0], xij[1], xij[2], xij_norm))
        
        for idx in idxs_ball:
            i_vertx = ball_data[idx[0]]
            j_vertx = cloth_data[idx[1]]
            xij = i_vertx[:3] - j_vertx[:3]
            xij_norm = np.linalg.norm(xij, ord = 2)
            if xij_norm < 0.015:
                #print("ball, %d, %d" % (idx[0], idx[1]))
                foutworld.write("%d %.6f %.6f %.6f %.6f\n" % (idx[1], xij[0], xij[1], xij[2], xij_norm))
        foutworld.close()

def GenWorldEdgeFeature():
    pass

if __name__ == "__main__":
    GenEdgeFeature()

