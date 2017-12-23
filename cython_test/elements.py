import numpy as np
from matplotlib.pylab import *
from okada_wrapper import dc3dwrapper
from scipy import optimize
from scipy import ndimage
import matplotlib.patches as patches
import Queue
from multiprocessing import Process
import multiprocessing
import copy


class Station:
    def __init__(self, east, north, disp=0, x_size=1, y_size=1):
        self.east = east
        self.north = north
        self.disp = disp
        self.x_size = x_size
        self.y_size = y_size


class Plain:
    def __init__(self, XO=0, YO=0, ZO=0, STK=0, DIP=0, strike_length=1, dip_length=1, strike_sub=1, dip_sub=1):
        self.XO = XO
        self.YO = YO
        self.ZO = ZO
        self.STK = STK
        self.DIP = DIP
        self.strike_length = strike_length
        self.dip_length = dip_length
        self.strike_sub = int(strike_sub)
        self.dip_sub = int(dip_sub)
        self.sub_plains = []
        self.los = None

class sub_plain:
    def __init__(self, x, y, x_size, y_size):
        self.x = x
        self.y = y
        self.x_size = x_size
        self.y_size = y_size

class Scen:
    def __init__(self, im_size=100, pixel=1, lame_lambda=0.25, mu=7e10, azimuth=0, nadir=0):
        self.im_size = im_size
        self.pixel = pixel
        self.size = int(im_size / pixel)
        self.plains = []
        self.img = np.zeros((self.size, self.size, 3))
        self.alpha = (lame_lambda + mu) / (lame_lambda + 2 * mu)
        self.azimuth = np.deg2rad(np.mod(azimuth, 360))
        self.nadir = np.linspace(np.deg2rad(nadir[0]), np.deg2rad(nadir[1]), self.size)
        self.los = None

    def compute_disp(self):
        for plain in self.plains:
            sub_stk_dim = plain.strike_length / plain.strike_sub
            sub_dip_dim = plain.dip_length / plain.dip_sub
            rstrike = np.deg2rad(plain.STK)
            plain_disp = np.zeros((self.size, self.size, 3))
            sub_plains = np.array(plain.sub_plains)
            sub_plains = sub_plains.reshape(plain.strike_sub, plain.dip_sub, 3)
            for k in range(plain.strike_sub):
                for l in range(plain.dip_sub):
                    for i in range(self.size):
                        for j in range(self.size):
                            ec = j - int(plain.XO / self.pixel)
                            nc = i - int(plain.YO / self.pixel)
                            x = np.cos(rstrike) * nc + np.sin(rstrike) * ec
                            y = np.sin(rstrike) * nc - np.cos(rstrike) * ec
                            plain_disp[self.img.shape[0] - 1 - i, j] += \
                                dc3dwrapper(self.alpha, [x * self.pixel, y * self.pixel, 0], np.abs(plain.ZO), plain.DIP
                                            , [k * sub_stk_dim, (k + 1) * sub_stk_dim],
                                            [l * sub_dip_dim, (l + 1) * sub_dip_dim], sub_plains[k, l, :])[1]
            self.img[:, :, 0] += np.sin(rstrike) * plain_disp[:, :, 0] - np.cos(rstrike) * plain_disp[:, :, 1]
            self.img[:, :, 1] += np.cos(rstrike) * plain_disp[:, :, 0] + np.sin(rstrike) * plain_disp[:, :, 1]
            self.img[:, :, 2] += plain_disp[:, :, 2]

    def compute_disp_slip_cord(self):

        for plain in self.plains:
            sub_stk_dim = plain.strike_length / plain.strike_sub
            sub_dip_dim = plain.dip_length / plain.dip_sub
            rstrike = np.deg2rad(plain.STK)
            plain_disp = np.zeros((self.size, self.size, 3))
            sub_plains = np.array(plain.sub_plains)
            sub_plains = sub_plains.reshape(plain.strike_sub, plain.dip_sub, 3)
            for k in range(plain.strike_sub):
                for l in range(plain.dip_sub):
                    for i in range(self.size):
                        for j in range(self.size):
                            ec = j - int(plain.YO / self.pixel)
                            nc = i - int(plain.XO / self.pixel)
                            x = np.cos(rstrike) * nc + np.sin(rstrike) * ec
                            y = np.sin(rstrike) * nc - np.cos(rstrike) * ec
                            plain_disp[self.size - 1 - i, j] +=\
                                dc3dwrapper(self.alpha, [x * self.pixel, y * self.pixel, 0], np.abs(plain.ZO), plain.DIP
                                            , [k * sub_stk_dim, (k + 1) * sub_stk_dim],
                                            [l * sub_dip_dim, (l + 1) * sub_dip_dim], sub_plains[k, l, :])[1]
            self.img[:, :, 0] += plain_disp[:, :, 0]
            self.img[:, :, 1] += plain_disp[:, :, 1]
            self.img[:, :, 2] += plain_disp[:, :, 2]

    def compute_los(self):
        self.los = np.sin(self.nadir) * (np.cos(self.azimuth) * self.img[:, :, 0] -
                                         np.sin(self.azimuth) * self.img[:, :, 1]) + np.cos(self.nadir) \
                   * self.img[:, :, 2]

    def add_uncorelated_noise(self, mu=0, sigma=0.00001):
        self.los += np.random.normal(mu, sigma, size=self.los.shape)

    def add_corelated_noise(self, sigma=10, max_noise=0.001, num_of_points=10):
        im = np.zeros(self.los.shape)
        points = self.los.shape[0] * np.random.random((2, num_of_points))
        points_val = np.random.uniform(-max_noise, max_noise, num_of_points)
        im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = points_val
        im = ndimage.gaussian_filter(im, sigma=sigma)
        self.los += im


class Inversion:

    def __init__(self, station='uniform', plains=[],  im_size=100, pixel=1, lame_lambda=8e9, mu=8e9, disp=0,
                 sub_plain_num=1, azimuth=0, nadir=0, cord='cartisian', smooth=False, beta=0):
        self.plains = plains
        self.im_size = im_size
        self.pixel = pixel
        self.size = int(im_size / pixel)
        self.disp = disp
        self.alpha = (lame_lambda + mu) / (lame_lambda + 2 * mu)
        self.station = []
        self.solution = nan
        if station == 'uniform':
            station_num = 20#np.ceil(np.sqrt(sub_plain_num))
            # add y_size when change image to have different y and x dimensions
            # if station_num < 3:
            #     station_num = 3
            steps = np.linspace(0, im_size-1, station_num+1)
            steps = np.floor(steps)
            size_to_next = steps[1]
            for i in range(steps.shape[0]-1):
                for j in range(steps.shape[0]-1):
                    self.station.append(Station(steps[i], steps[j], x_size=steps[i+1]- steps[i],
                                                y_size=steps[j+1] - steps[j]))
            # for x_step in steps:
            #     for y_step in steps:
            #         self.station.append(Station(x_step, y_step, x_size=size_to_next, y_size=size_to_next))
        else:
            self.station = station
        self.calculate_station_disp()
        self.A = nan
        self.sub_plain_num = sub_plain_num
        self.azimuth = np.deg2rad(np.mod(azimuth, 360))
        self.nadir = np.linspace(np.deg2rad(nadir[0]), np.deg2rad(nadir[1]), self.size)
        self.cord = cord
        self.smooth = smooth
        self.beta = beta

    def calculate_station_disp(self):
        for s in self.station:
            s.disp = self.disp[self.disp.shape[0] - 1 - int(s.north/self.pixel), int(s.east/self.pixel)]

    def calc_A(self):
        self.A = np.zeros((self.sub_plain_num, len(self.station), 3))
        plain_index = 0
        rstrike = 0
        for plain in self.plains:
            sub_stk_dim = float(plain.strike_length) / plain.strike_sub
            sub_dip_dim = float(plain.dip_length) / plain.dip_sub
            rstrike = np.deg2rad(plain.STK)
            for k in range(plain.strike_sub):
                for l in range(plain.dip_sub):
                    for s, i in zip(self.station, range(len(self.station))):
                        #need to take out of the loop the calculation of the station cordination in the fualt cord
                        east = int(s.east/self.pixel)*self.pixel
                        north = int(s.north/self.pixel)*self.pixel
                        # possible that for real data i dont need the round up to pixel
                        ec = east - int(plain.XO/self.pixel)*self.pixel
                        nc = north - int(plain.YO/self.pixel)*self.pixel
                        x = np.cos(rstrike) * nc + np.sin(rstrike) * ec
                        y = np.sin(rstrike) * nc - np.cos(rstrike) * ec
                        self.A[plain_index, i] +=  \
                            dc3dwrapper(self.alpha, [x, y, 0], np.abs(plain.ZO),
                                        plain.DIP,
                                        [k * sub_stk_dim, (k + 1) * sub_stk_dim],
                                        [l * sub_dip_dim, (l + 1) * sub_dip_dim], [1, 0, 0])[1]
                    plain_index += 1
        if self.cord != 'fault':
            temp = np.zeros(self.A.shape)
            temp[:, :, 0] += np.sin(rstrike) * self.A[:, :, 0] - np.cos(rstrike) * self.A[:, :, 1]
            temp[:, :, 1] += np.cos(rstrike) * self.A[:, :, 0] + np.sin(rstrike) * self.A[:, :, 1]
            temp[:, :, 2] = self.A[:, :, 2]
            self.A = temp
            if self.cord != 'cartesian':
                # nadir = np.array([s.east for s in self.station])/self.im_size*(self.nadir[1]-self.nadir[0])+self.nadir[0]
                nadir = np.array([self.nadir[int(s.east/self.pixel)] for s in self.station])
                self.A = (np.cos(self.azimuth) * self.A[:, :, 0] - np.sin(self.azimuth) * self.A[:, :, 1]) *\
                         np.sin(nadir) + self.A[:, :, 2] * np.cos(nadir)

                # self.A = (np.sin(nadir) * (np.cos(self.azimuth) * self.A[:, :, 0] -
                #                                  np.sin(self.azimuth) * self.A[:, :, 1]).T).T + (np.cos(
                #     nadir)* self.A[:, :, 2].T).T
        self.A = self.A.T

    def solve(self):
        if self.cord != 'sat':
            b = np.concatenate([s.disp for s in self.station])
            A = np.hstack(np.array([self.A.T[j, i, :] for j in range(self.A.T.shape[0])]) for i in range(self.A.T.shape[1])).T
        else:
            b = np.array([s.disp for s in self.station])
            A = self.A
            if self.smooth:
                S = self.beta * self.build_smothing()
                A = np.concatenate((A, S))
                b = np.concatenate((b, np.zeros(self.sub_plain_num)))

        self.solution = optimize.nnls(A, b)

    def plot_stations(self, dots=True ):
        plt.imshow(self.disp, cmap='bwr')
        if dots:
            s_x = [s.east / self.pixel for s in self.station]
            s_y = [self.disp.shape[0] - 1 - s.north / self.pixel for s in self.station]
            plt.scatter(s_x, s_y, s=1, color='k')
        else:
            for s in self.station:
                plt.axes().add_patch(patches.Rectangle((s.east/self.pixel, (self.disp.shape[0] - 1 + -s.north / self.pixel)- s.y_size/self.pixel), s.x_size / self.pixel, s.y_size / self.pixel, fill=False, ec='k'))
        plt.xlim([0, self.size])
        plt.ylim([self.size, 0])
        plt.show()

    def build_smothing(self):
        def adjacent(sp1, sp2):
            return (((sp1.y == sp2.y - sp2.y_size) or (sp1.y - sp1.y_size == sp2.y)) and sp2.x + sp2.x_size > sp1.x and\
                   sp2.x < sp1.x + sp1.x_size) or ((sp1.x + sp1.x_size == sp2.x) and sp1.y == sp2.y)\
                   or ((sp1.x - sp1.x_size == sp2.x) and sp1.y == sp2.y)

        S = np.zeros((self.sub_plain_num, self.sub_plain_num))
        sub_plains = []
        y = 0
        for p in self.plains:
            y += p.dip_length
            x = 0
            x_size  = p.strike_length / p.strike_sub
            for i in range(p.strike_sub):
                sub_plains.append(sub_plain(x, y, x_size, p.dip_length))
                x += x_size
        for i in range(len(sub_plains)):
            for j in range(len(sub_plains)):
                if adjacent(sub_plains[i], sub_plains[j]):
                    S[i, i] += 1
                    S[i, j] -= 1
        return S


