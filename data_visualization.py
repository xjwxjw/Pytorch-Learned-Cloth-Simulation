import numpy as np
import matplotlib.pyplot as plt

def acc_vis():
    
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

if __name__ == "__main__":
    vel_vis()
