from okada_wrapper import *
import numpy as np

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