import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def hist_vis():
    has_ball = True
    cloth_mesh_tri_array = []
    ball_mesh_tri_array = []
    for line in open('../Data/cloth_connection.txt', 'r'):
        cloth_mesh_tri_array.append(int(line.split('\n')[0]))
    cloth_mesh_tri_array = np.array(cloth_mesh_tri_array).reshape(-1, 3)
    if has_ball:
        for line in open('../Data/sphere_connection.txt', 'r'):
            ball_mesh_tri_array.append(int(line.split('\n')[0]))
        ball_mesh_tri_array = np.array(ball_mesh_tri_array).reshape(-1, 3)
    
    pos = []
    for line in open('/media/xjwxjw/626898FF2DF873F3/Documents/ClothSimData/data_0424/0043/%03d_ball.txt' % 3, 'r'):
        line = line.split('\n')[0].split(' ')[:3]
        pos.append(np.array([float(x) for x in line]))
    pos = np.stack(pos, 0)
    plt.hist(np.linalg.norm(pos[ball_mesh_tri_array[:,0]] - pos[ball_mesh_tri_array[:,1]], ord = 2, axis = -1), 200)
    plt.show()

    pos = []
    for line in open('/media/xjwxjw/626898FF2DF873F3/Documents/ClothSimData/data_0424/0043/%03d_cloth.txt' % 3, 'r'):
        line = line.split('\n')[0].split(' ')[:3]
        pos.append(np.array([float(x) for x in line]))
    pos = np.stack(pos, 0)
    plt.hist(np.linalg.norm(pos[cloth_mesh_tri_array[:,0]] - pos[cloth_mesh_tri_array[:,1]], ord = 2, axis = -1), 200)
    plt.show()

def acc_vis():
    acc_array = []
    for t in range(3, 496):
        for line in open('../Results_Acc/%04d.txt' % t, 'r'):
            acc = line.split('\n')[0].split(' ')
            acc = np.array([ float(a) for a in acc])
            acc_array.append(acc)
        acc_array = np.array(acc_array)
        plt.hist(acc_array[:,1], 200)
        plt.show()

def vel_vis():
    vel_list = []
    for t in range(3, 496):
        vel_array = []
        for line in open('../Data/0000/%03d_cloth.txt' % t, 'r'):
            vel = line.split('\n')[0].split(' ')[:3]
            vel = np.array([ float(v) for v in vel])
            vel_array.append(vel)
        vel_array = np.array(vel_array)

        vel_nxt_array = []
        for line in open('../Data/0000/%03d_cloth.txt' % (t+1), 'r'):
            vel = line.split('\n')[0].split(' ')[:3]
            vel = np.array([ float(v) for v in vel])
            vel_nxt_array.append(vel)
        vel_nxt_array = np.array(vel_nxt_array)
        vel_list.append(vel_nxt_array - vel_array)
    vel_list = np.concatenate(vel_list, 0)
    # plt.hist(vel_list[:,2], 200)
    # plt.show()
    std = np.std(vel_list, 0)
    print(std)

def cloth_vis():
    has_ball = True
    cloth_mesh_tri_array = []
    ball_mesh_tri_array = []
    for line in open('../Data/cloth_connection.txt', 'r'):
        cloth_mesh_tri_array.append(int(line.split('\n')[0]))
    cloth_mesh_tri_array = np.array(cloth_mesh_tri_array).reshape(-1, 3)
    if has_ball:
        for line in open('../Data/sphere_connection.txt', 'r'):
            ball_mesh_tri_array.append(int(line.split('\n')[0]))
        ball_mesh_tri_array = np.array(ball_mesh_tri_array).reshape(-1, 3)

    for file_idx in range(3, 499):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        pos = []
        for line in open('/media/xjwxjw/626898FF2DF873F3/Documents/ClothSimData/data_0424/0043/%03d_cloth.txt' % file_idx, 'r'):
            line = line.split('\n')[0].split(' ')[:3]
            pos.append(np.array([float(x) for x in line]))
        pos = np.stack(pos, 0)
        x = pos[:,0]
        y = pos[:,2]
        z = pos[:,1]
        ax.plot_trisurf(x, y, z, triangles=cloth_mesh_tri_array, linewidth=0.2, antialiased=True, color = 'b')

        pos = []
        for line in open('/media/xjwxjw/626898FF2DF873F3/Documents/ClothSimData/data_0424/0043/%03d_ball.txt' % file_idx, 'r'):
            line = line.split('\n')[0].split(' ')[:3]
            pos.append(np.array([float(x) for x in line]))
        pos = np.stack(pos, 0)
        x = pos[:,0]
        y = pos[:,2]
        z = pos[:,1]
        ax.plot_trisurf(x, y, z, triangles=ball_mesh_tri_array, linewidth=0.2, antialiased=True, color = 'r')

        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])
        ax.set_zlim([-1.0, 1.0])
        plt.savefig('../Results/%03d.png' % (file_idx - 0))
        plt.close('all')

def gen_video():
    import cv2
    import os

    def get_file_names(search_path):
        for (dirpath, _, filenames) in os.walk(search_path):
            for filename in filenames:
                yield filename  # os.path.join(dirpath, filename)

    def save_to_video(output_path, output_video_file, frame_rate):
        list_files = sorted([int(i.split('_')[-1].split('.')[0]) for i in get_file_names(output_path)])
        # 拿一张图片确认宽高
        img0 = cv2.imread(os.path.join(output_path, '%03d.png' % list_files[0]))
        img1 = cv2.imread(os.path.join(output_path, '%03d.png' % list_files[0]).replace('Results', 'Pred'))
        img = np.concatenate([img0, img1], 1)

        # print(img0)
        height, width, layers = img.shape
        # 视频保存初始化 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videowriter = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (width, height))
        # 核心，保存的东西
        font = cv2.FONT_HERSHEY_SIMPLEX
        for f in list_files:
            try:
                f = '%03d.png' % f
                # print("saving..." + f)
                
                img0 = cv2.imread(os.path.join(output_path, f))
                img1 = cv2.imread(os.path.join(output_path, f).replace('Results', 'Pred'))
                
                img = np.concatenate([img0, img1], 1)
                img = cv2.putText(img, f, (0, 100), font, 1.2, (255, 0, 0), 2)
                img = cv2.putText(img, 'houdini', (250, 100), font, 1.2, (0, 0, 255), 2)
                img = cv2.putText(img, 'model pred', (850, 100), font, 1.2, (0, 0, 255), 2)
                videowriter.write(img)
            except:
                print(os.path.join(output_path, f).replace('Results', 'Pred'))
        videowriter.release()
        cv2.destroyAllWindows()
        print('Success save %s!' % output_video_file)
        pass

    # 图片变视频
    output_dir = '../Results'
    output_path = os.path.join(output_dir, '')  # 输入图片存放位置
    output_video_file = './gt.mp4'  # 输入视频保存位置以及视频名称
    save_to_video(output_path, output_video_file, 20)

if __name__ == "__main__":
    hist_vis()
