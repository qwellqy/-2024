import numpy as np
import matplotlib.pyplot as plt
import iapws 
from iapws import IAPWS97 as gas

class Steam_Rashod:
    def __init__(self, p_o, t_o, p_nn, t_nn, t_nv, p_k, effect_oi, effect_mex, effect_eg, n):

        self.n = n
        delt_p_o = 0.05 * p_o
        delt_p_nn = 0.1 * p_nn
        delt_p_1 = 0.03 * p_nn
        p_1 = p_nn + delt_p_nn
        p_nv = 1.85 * p_o
        p_to_iapws = 10 ** (-6)
        self.effect_oi = effect_oi
        self.effect_mex = effect_mex
        self.effect_eg = effect_eg 
        self.point_o_teor = gas(P = p_o * p_to_iapws, T = t_o + 273.15)
        self.point_nn_teor = gas(P = (p_nn - delt_p_nn) * p_to_iapws, T = t_nn + 273.15)
        self.point_nn = gas(P = p_nn * p_to_iapws, T = t_nn + 273.15)
        self.point_k_teor = gas(P = p_k * p_to_iapws, s = self.point_nn.s)
        self.point_1_teor = gas(P = p_1 * p_to_iapws, s = self.point_o_teor.s)
        self.point_k_o = gas(P = p_k * p_to_iapws, x = 0)
        self.point_nv = gas(P = p_nv * p_to_iapws, T = t_nv + 273.15)
        self.point_o_o = gas(P=p_o*p_to_iapws, T=t_o + 273.15)
        h_1 = self.point_o_teor.h - (self.point_o_teor.h - self.point_1_teor.h) * self.effect_oi
        h_k = self.point_nn.h - (self.point_nn.h - self.point_k_teor.h) * self.effect_oi
        self.point_o = gas(P = (p_o - delt_p_o) * p_to_iapws, h = self.point_o_teor.h)
        self.point_k = gas(P = p_k * p_to_iapws, h = h_k)
        self.point_1 = gas(P = (p_1 + delt_p_1) * p_to_iapws, h = h_1)
        
    def calc_xi(self):

        a = 1 - ((self.point_k_teor.T) * (self.point_nn.s - self.point_k_o.s) / ((self.point_o.h - self.point_1_teor.h) + (self.point_nn.h - self.point_k_o.h)))
        b = 1 - ((self.point_k_teor.T) * (self.point_nn.s - self.point_nv.s) / ((self.point_o.h - self.point_1_teor.h) + (self.point_nn.h - self.point_nv.h)))
        xi_besk = 1 - a / b
        otn = ((self.point_nv.T) - self.point_k_teor.T) / (self.point_o_o.T - self.point_k_teor.T)
        if (otn < 0.635294):
            xi = (-0.867 * (otn) ** 2 + 1.5222 * otn + 0.147) * xi_besk
        if (otn > 0.635294) and (otn < 0.641177):
            xi = (5.4399 * otn - 2.6879) * xi_besk
        if (otn > 0.641177) and (otn < 0.752941):
            xi = (0.0415 * (otn) ** 2 + 0.0925 * otn + 0.7236) * xi_besk
        if (otn > 0.752941) and (otn < 0.761176):
            xi = (3.7886 * otn - 2.0358) * xi_besk
        if (otn > 0.761176) and (otn < 0.858824):
            xi = (-3.8618 * (otn) ** 2 + 6.338 * otn - 1.7389) * xi_besk
        if (otn > 0.858824) and (otn < 0.870588):
            xi = (2.72 * otn - 1.48) * xi_besk
        if (otn > 0.870588) and (otn < 0.976471):
            xi = (-0.2644 * otn + 1.1182) * xi_besk
        return xi
        
    def calc_rashod(self):
        
        N_to_G = 10 ** (-3)
        self.effect_ip = ((self.point_o.h - self.point_1_teor.h) * self.effect_oi + (self.point_nn.h - self.point_k_teor.h) * self.effect_oi) / ((self.point_o.h - self.point_1_teor.h) * self.effect_oi + (self.point_nn.h - self.point_k_o.h)) / (1 - self.calc_xi())
        h_1 = self.point_o.h - (self.point_o.h - self.point_1_teor.h) * self.effect_oi 
        h_i = self.effect_ip * ((self.point_o.h - self.point_nv.h) + (self.point_nn.h - h_1)) 
        self.g_o = self.n * N_to_G / (h_i * self.effect_mex * self.effect_eg)
        self.g_k = self.n * N_to_G / ((self.point_k.h - self.point_k_o.h) * self.effect_mex * self.effect_eg) * ((1 / self.effect_ip) - 1)
        return self.g_o, self.g_k