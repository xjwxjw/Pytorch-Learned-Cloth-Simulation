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

def draw_gt_data():
    for i in range(100):
        for j in range(500):
            for line in open('../Data/data/%04d/%03d_world.txt' % (i, j), 'r'):
                if (line.split('\n')[0].split(' ')[0] == '0'):
                 print(line.split('\n')[0].split(' '))
    cloth_connection = np.array([int(line.split('\n')[0]) for line in open('../Data/cloth_connection.txt', 'r')])
    cloth_connection = np.reshape(cloth_connection, (-1, 3))
    ball_connection = np.array([int(line.split('\n')[0]) for line in open('../Data/sphere_connection.txt', 'r')])
    ball_connection = np.reshape(ball_connection, (-1, 3))

    from mpl_toolkits.mplot3d import Axes3D as axes3d
    import matplotlib.pyplot as plt
    
    for i in range(500):
        cloth_data = []
        for line in open('../Data/data/0002/%03d_cloth.txt' % i, 'r'):
            line = line.split('\n')[0]
            cloth_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
        cloth_data = np.array(cloth_data)
        cloth_x, cloth_y, cloth_z = cloth_data[:,:3].T

        ball_data = []
        for line in open('../Data/data/0002/%03d_ball.txt' % i, 'r'):
            line = line.split('\n')[0]
            ball_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
        ball_data = np.array(ball_data)
        ball_x, ball_y, ball_z = ball_data[:,:3].T

        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.plot_trisurf(cloth_x, cloth_z, cloth_y, triangles = cloth_connection)
        ax.plot_trisurf(ball_x, ball_z, ball_y, triangles = ball_connection)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        plt.savefig('../Results/%03d.png' % i)
        # plt.show()
        plt.close('all')

def GenEdgeFeature(seq_id):
    cloth_connection = np.load('../Data/cloth_connection.npy', allow_pickle=True).item()    
    # ball_connection = np.load('../Data/sphere_connection.npy', allow_pickle=True).item()    
    for i in range(500):
        print(i)
        cloth_data = []
        cloth_path = ('../Data/data/%04d/%03d_cloth.txt' % (seq_id, i))
        for line in open(cloth_path, 'r'):
            line = line.split('\n')[0]
            cloth_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
        cloth_data = np.array(cloth_data)

        cloth_idx = i
        if i < 499:
            cloth_idx = i+1
        ball_data = []
        ball_path = ('../Data/data/%04d/%03d_ball.txt' % (seq_id, cloth_idx))
        for line in open(ball_path, 'r'):
            line = line.split('\n')[0]
            ball_data.append(np.array([float(data) for data in line.split(' ')[:-1]]))
        ball_data = np.array(ball_data)

        foutuv = open('../Data/data/%04d/%03d_uv.txt' % (seq_id, i), 'w')
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

        foutworld = open('../Data/data/%04d/%03d_world.txt' % (seq_id, i), 'w')
        cloth_world_dis = np.sum((cloth_data[None, :, :3] - cloth_data[:, None, :3])**2, -1)**0.5
        ball_world_dis = np.sum((cloth_data[None, :, :3] - ball_data[:, None, :3])**2, -1)**0.5
        idxs_cloth = np.argwhere(cloth_world_dis < 0.02)
        idxs_ball = np.argwhere(ball_world_dis < 0.04)

        for idx in idxs_cloth:
            i_vertx = cloth_data[idx[0]]
            j_vertx = cloth_data[idx[1]]
            xij = i_vertx[:3] - j_vertx[:3]
            xij_norm = np.linalg.norm(xij, ord = 2)
            if (idx[0] != idx[1]) and (idx[0] not in cloth_connection[idx[1]]):
                foutworld.write("0 %d %d %.6f %.6f %.6f %.6f\n" % (idx[0], idx[1], xij[0], xij[1], xij[2], xij_norm))
        
        for idx in idxs_ball:
            i_vertx = cloth_data[idx[1]]
            j_vertx = ball_data[idx[0]]
            xij = i_vertx[:3] - j_vertx[:3]
            xij_norm = np.linalg.norm(xij, ord = 2)
            foutworld.write("1 %d %d %.6f %.6f %.6f %.6f\n" % (idx[1], idx[0], xij[0], xij[1], xij[2], xij_norm))
        foutworld.close()

import threading
import multiprocessing

if __name__ == "__main__":
    for i in range(100):
        p1 = multiprocessing.Process(target = GenEdgeFeature, args = (i,))
        p1.start()

