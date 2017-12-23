from okada_wrapper import *
import numpy as np
cimport numpy as np

cpdef compute_station_disp(inv, s):
    cdef int k
    cdef int l
    cdef double sub_stk_dim
    cdef double sub_dip_dim
    cdef np.ndarray[np.double_t, ndim = 2] disp = np.zeros((inv.sub_plain_num, 3))
    cdef np.ndarray[np.double_t, ndim = 2] temp = np.zeros((inv.sub_plain_num, 3))
    cdef np.ndarray[np.double_t, ndim = 1] los_disp
    cdef int plain_index = 0
    cdef double rstrike = 0
    cdef double x
    cdef double y
    cdef i
    for i in range(len(inv.plains)):
        sub_stk_dim = float(inv.plains[i].strike_length) / inv.plains[i].strike_sub
        sub_dip_dim = float(inv.plains[i].dip_length) / inv.plains[i].dip_sub
        rstrike = np.deg2rad(inv.plains[i].STK)
        east = int(s.east / inv.pixel) * inv.pixel
        north = int(s.north / inv.pixel) * inv.pixel
        #possible that for real data i dont need the round up to pixel
        ec = east - int(inv.plains[i].XO / inv.pixel) * inv.pixel
        nc = north - int(inv.plains[i].YO / inv.pixel) * inv.pixel
        x = np.cos(rstrike) * nc + np.sin(rstrike) * ec
        y = np.sin(rstrike) * nc - np.cos(rstrike) * ec
        for k in range(inv.plains[i].strike_sub):
            for l in range(inv.plains[i].dip_sub):
                # need to take out of the loop the calculation of the station cordination in the fualt cord
                disp[plain_index] += \
                    dc3dwrapper(inv.alpha, [x, y, 0], np.abs(inv.plains[i].ZO),
                                inv.plains[i].DIP,
                                [k * sub_stk_dim, (k + 1) * sub_stk_dim],
                                [l * sub_dip_dim, (l + 1) * sub_dip_dim], [1, 0, 0])[1]
                plain_index += 1
    temp[:, 0] += np.sin(rstrike) * disp[:, 0] - np.cos(rstrike) * disp[:, 1]
    temp[:, 1] += np.cos(rstrike) * disp[:, 0] + np.sin(rstrike) * disp[:, 1]
    temp[:, 2] = disp[:, 2]
    nadir = inv.nadir[int(s.east / inv.pixel)]
    los_disp = (np.cos(inv.azimuth) * temp[:, 0] - np.sin(inv.azimuth) * temp[:, 1]) *\
             np.sin(nadir) + temp[:, 2] * np.cos(nadir)
    return los_disp