
from elements import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import matplotlib.cm as cm
import matplotlib.patches as patches
from multiprocessing import Process
import multiprocessing
import copy
import Queue


def scen_geometry(input_file):
    with open(input_file) as f:
        lines = f.readlines()
    for line in lines:
        if line[0] != '#' and line[0] != '\n':
            words = line.split()
            if words[0] == 'global':
                nadir = words[6].split(',')
                s = Scen(int(float(words[3])), float(words[4]), float(words[1]), float(words[2]), float(words[5]), (float(nadir[0]), float(nadir[1])))
            elif words[0] == 'plain':
                s.plains.append(Plain(float(words[1]), float(words[2]), float(words[3]), float(words[4]),\
                                      float(words[5]), float(words[6]), float(words[7]), float(words[8]), float(words[9])))
            elif words[0] == 'sub':
                s.plains[len(s.plains)-1].sub_plains.append([ float(words[2]), float(words[3]), float(words[4])])
    return s


def compute_disp(s,  i_start, i_end, queue):
    for plain in s.plains:
        rstrike = np.deg2rad(plain.STK)
        sub_stk_dim = plain.strike_length / plain.strike_sub
        sub_dip_dim = plain.dip_length / plain.dip_sub
        sub_plains = np.array(plain.sub_plains)
        sub_plains = sub_plains.reshape(plain.strike_sub, plain.dip_sub, 3)
        for k in range(plain.strike_sub):
            for l in range(plain.dip_sub):
                for i in range(i_start, i_end, 1):
                    for j in range(s.size):
                        ec = j - int(plain.XO / s.pixel)
                        nc = i - int(plain.YO / s.pixel)
                        x = np.cos(rstrike) * nc + np.sin(rstrike) * ec
                        y = np.sin(rstrike) * nc - np.cos(rstrike) * ec
                        s.img[s.img.shape[0] - 1 - i, j] += \
                            dc3dwrapper(s.alpha, [x * s.pixel, y * s.pixel, 0], np.abs(plain.ZO), plain.DIP,
                                        [k * sub_stk_dim, (k + 1) * sub_stk_dim],
                                      [l * sub_dip_dim, (l + 1) * sub_dip_dim], sub_plains[k, l, :])[1]
    queue.put((s, rstrike))
    return True


def sum_disp(s, queue, main_queue_in, main_queue_out, fualt_cord):
    done = 0
    while True:
        try:
            part_s = queue.get(timeout=1)
            if fualt_cord:
                s.img[:, :, 0] += part_s[0].img[:, :, 0]
                s.img[:, :, 1] += part_s[0].img[:, :, 1]
            else:
                s.img[:, :, 0] += np.sin(part_s[1]) * part_s[0].img[:, :, 0] - \
                                  np.cos(part_s[1]) * part_s[0].img[:, :, 1]
                s.img[:, :, 1] += np.cos(part_s[1]) * part_s[0].img[:, :, 0] + \
                                  np.sin(part_s[1]) * part_s[0].img[:, :, 1]
            s.img[:, :, 2] += part_s[0].img[:, :, 2]
            done += 1
        except Queue.Empty:
            try:
                to_finish = main_queue_in.get(timeout=0.01)
                if to_finish:
                    main_queue_out.put(s)
                    return True
            except Queue.Empty:
                pass


def build_scen(input, fualt_cord=False):
    s = scen_geometry(input)
    cpu_count = multiprocessing.cpu_count()
    q = multiprocessing.Queue()
    main_q_out = multiprocessing.Queue()
    main_q_in = multiprocessing.Queue()
    sum_disp_p = Process(target=sum_disp, args=(copy.deepcopy(s), q, main_q_out, main_q_in, fualt_cord))
    sum_disp_p.daemon = True
    sum_disp_p.start()
    process = []
    chunk = int(s.img.shape[0]/ cpu_count)
    for i in range(cpu_count):
        if i == cpu_count-1:
            finish = s.img.shape[0]
        else:
            finish = (i+1) * chunk
        p = Process(target=compute_disp, args=(copy.deepcopy(s), i*chunk, finish, q))
        process.append(p)
        p.daemon = True
        p.start()
    for p in process:
        p.join()
    main_q_out.put(True)
    s = main_q_in.get()
    sum_disp_p.join()
    return s


def build_inversion(input_file):
    plains = []
    im_size = 0
    pixel = 0
    size = 0
    disp = 0
    lame_lambda = 0
    nu = 0
    station = []
    sub_plain_num = 0
    azimuth = 0
    nadir = 0
    cord = 0
    smooth = False
    beta = 0

    with open(input_file) as f:
        lines = f.readlines()
    for line in lines:
        if line[0] != '#' and line[0] != '\n':
            words = line.split()
            if words[0] == 'global':
                lame_lambda = float(words[1])
                mu = float(words[2])
                im_size = float(words[3])
                pixel = float(words[4])
                if words[5] == 'uniform':
                    station = words[5]
                else:
                    st = pd.read_csv(words[5], sep=' ', header=None)
                    st = st.as_matrix()
                    station = [Station(x, y) for x, y in zip(st[:, 0], st[:, 1])]
                cord = words[6]
                azimuth = float(words[7])
                nadir = (float(words[8].split(',')[0]), float(words[8].split(',')[1]))
                smooth = words[9] == 'True'
                if smooth:
                    beta = float(words[10])

            elif words[0] == 'disp':
                if words[1] == 'npy':
                    disp = np.load(words[3])
                elif words[1] == 'tif':
                    pass
            elif words[0] == 'plain':
                plains.append(Plain(float(words[1]), float(words[2]), float(words[3]), float(words[4]),
                                      float(words[5]), float(words[6]), float(words[7]), float(words[8]),
                                      float(words[9])))
                sub_plain_num += int(float(words[8])) * int(float(words[9]))
    return Inversion(station, plains, im_size, pixel, lame_lambda, mu, disp, sub_plain_num, azimuth, nadir, cord,
                     smooth, beta)


def plot_xyz_disp(disp):
    fig = plt.figure(figsize=(16, 10))
    ax11 = fig.add_subplot(1, 3, 1)
    ax12 = fig.add_subplot(1, 3, 2)
    ax13 = fig.add_subplot(1, 3, 3)

    im11 = ax11.imshow(disp[:, :, 0], cmap='bwr')
    ax11.set_title('x')
    fig.colorbar(im11, ax=ax11)

    im12 = ax12.imshow(disp[:, :, 1], cmap='bwr')
    ax12.set_title('y')
    fig.colorbar(im12, ax=ax12)

    im13 = ax13.imshow(disp[:, :, 2], cmap='bwr')
    ax13.set_title('z')
    fig.colorbar(im13, ax=ax13)

    plt.show()

def plot_fualt(plains, movment=None):
    if movment is not None:
        length = movment.shape[0]
        all_sub_plains = np.concatenate((movment, np.zeros(length * 2)))
        all_sub_plains = all_sub_plains.reshape((3, length)).T
    else:
        all_sub_plains = [p.sub_plains for p in plains]
        all_sub_plains = np.concatenate(all_sub_plains)
    my_cmap = cm.get_cmap('jet')
    if len(all_sub_plains) != 0:
        norm = matplotlib.colors.Normalize(np.min(all_sub_plains[:, 0]), np.max(all_sub_plains[:, 0]))
    rectangels = []
    i = 0
    for p in plains:
        for k in range(p.dip_sub):
            for l in range(p.strike_sub):
                if len(all_sub_plains) == 0:
                    color_sub_plain = 'b'
                else:
                    color_sub_plain = my_cmap(norm(all_sub_plains[i][0]))
                rectangels.append(patches.Rectangle((l*p.strike_length/p.strike_sub, k*p.dip_length/p.dip_sub +
                                                    p.ZO / np.sin(np.deg2rad(p.DIP))),
                                                    p.strike_length/p.strike_sub, p.dip_length/p.dip_sub,
                                                    fc=color_sub_plain, ec='k'))
                i += 1
    figure(figsize=(16,12))
    ax = gca()
    for rect in rectangels:
        ax.add_patch(rect)
    if len(all_sub_plains) != 0:
        cmmapable = cm.ScalarMappable(norm, my_cmap)
        cmmapable.set_array(np.linspace(np.min(all_sub_plains[:, 0]), np.max(all_sub_plains[:, 0])))
        colorbar(cmmapable)

    fault_length = np.array([p.strike_length for p in plains])
    fault_depth = np.array([[p.ZO for p in plains], [p.DIP for p in plains], [p.dip_length for p in plains]])
    fault_depth = np.vstack((fault_depth[0, :] / np.sin(np.deg2rad(fault_depth[1, :])), fault_depth[2, :]))
    ax.set_xlim((0, fault_length.max()))
    ax.set_ylim(fault_depth[0, :].min(), fault_depth[0, :].max() + fault_depth[1, np.argmax(fault_depth[0, :])])
    ax.set_aspect('equal', adjustable='box')
    show()


def compute_station_disp(inv, s):
    disp = np.zeros((inv.sub_plain_num, 3))
    plain_index = 0
    rstrike = 0
    for plain in inv.plains:
        sub_stk_dim = float(plain.strike_length) / plain.strike_sub
        sub_dip_dim = float(plain.dip_length) / plain.dip_sub
        rstrike = np.deg2rad(plain.STK)
        east = int(s.east / inv.pixel) * inv.pixel
        north = int(s.north / inv.pixel) * inv.pixel
        #possible that for real data i dont need the round up to pixel
        ec = east - int(plain.XO / inv.pixel) * inv.pixel
        nc = north - int(plain.YO / inv.pixel) * inv.pixel
        x = np.cos(rstrike) * nc + np.sin(rstrike) * ec
        y = np.sin(rstrike) * nc - np.cos(rstrike) * ec
        for k in range(plain.strike_sub):
            for l in range(plain.dip_sub):
                # need to take out of the loop the calculation of the station cordination in the fualt cord
                disp[plain_index] += \
                    dc3dwrapper(inv.alpha, [x, y, 0], np.abs(plain.ZO),
                                plain.DIP,
                                [k * sub_stk_dim, (k + 1) * sub_stk_dim],
                                [l * sub_dip_dim, (l + 1) * sub_dip_dim], [1, 0, 0])[1]
                plain_index += 1
    temp = np.zeros(disp.shape)
    temp[:, 0] += np.sin(rstrike) * disp[:, 0] - np.cos(rstrike) * disp[:, 1]
    temp[:, 1] += np.cos(rstrike) * disp[:, 0] + np.sin(rstrike) * disp[:, 1]
    temp[:, 2] = disp[:, 2]
    nadir = inv.nadir[int(s.east / inv.pixel)]
    disp = (np.cos(inv.azimuth) * temp[:, 0] - np.sin(inv.azimuth) * temp[:, 1]) *\
             np.sin(nadir) + temp[:, 2] * np.cos(nadir)
    return disp


def resample(inv):
    cn_vec = []
    i_values = np.linalg.svd(inv.A, compute_uv=False)
    prev_cn = i_values.max() / i_values.min()
    cur_cn = prev_cn
    cn_vec.append(cur_cn)
    #while prev_cn < 1.05*cur_cn:
    n = 0 # temporery condition for debaging
    while n < 5:
        cn = []
        for i in range(len(inv.station)):
            if inv.station[i].x_size/2 < inv.pixel or inv.station[i].y_size/2 < inv.pixel:
                cn.append(inf)
                continue
            S = make_new_stations(inv.station[i])
            B = np.delete(inv.A, i, 0)
            for s, j in zip(S, [i, i+1, i+2, i+3]):
                B = np.insert(B.T, j, compute_station_disp(inv, s), axis=1).T
            ig_val = np.linalg.svd(B, compute_uv=False)
            a = ig_val.max()/ig_val.min()
            cn.append(a)
        five_percent = len(cn) / 100 * 5
        if five_percent < 1:
            five_percent = 1
        indices = np.array(cn).argsort()[:five_percent]
        indices = np.sort(indices)
        # indices = np.random.choice(indices, 1)
        new_stations = list(inv.station)
        shift = 0
        for i in indices:
            shifted = i + shift
            S = make_new_stations(new_stations[shifted])
            new_stations[shifted] = S[0]
            new_stations.insert(shifted+1, S[1])
            new_stations.insert(shifted + 2, S[2])
            new_stations.insert(shifted + 3, S[3])
            inv.A = np.delete(inv.A, shifted, 0)
            for s, j in zip(S, [shifted, shifted + 1, shifted + 2, shifted + 3]):
                inv.A = np.insert(inv.A.T, j, compute_station_disp(inv, s), axis=1).T
            shift += 3
            # inv.station = new_stations
            # inv.plot_stations(False)

            # inv.plot_stations(False)
        inv.station = new_stations
        prev_cn = cur_cn
        ig_val = np.linalg.svd(inv.A, compute_uv=False)
        cur_cn = ig_val.max()/ig_val.min()
        cn_vec.append(cur_cn)
        print n
        n += 1
    print cur_cn
    plt.plot(np.arange(len(cn_vec)), np.array(cn_vec))
    plt.show()


def make_new_stations(s):
    s1 = copy.deepcopy(s)
    s1.x_size /= 2
    s1.y_size /= 2
    s2 = copy.deepcopy(s1)
    s2.east += s1.x_size
    s3 = copy.deepcopy(s1)
    s3.north += s2.y_size
    s4 = copy.deepcopy(s1)
    s4.north += s3.y_size
    s4.east += s3.x_size
    return s1, s2, s3, s4








