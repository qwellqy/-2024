import numpy as np
import matplotlib.pyplot as plt
from iapws import IAPWS97
from watersteam import WaterSteam  

class Steam_Rashod:
    def __init__(self, p_o, t_o, p_nn, t_nn, t_nv, p_k, effect_oi, effect_mex, effect_eg, n):
        self.n = n
        self.effect_oi = effect_oi
        self.effect_mex = effect_mex 
        self.effect_eg = effect_eg
        self.p_o = p_o  # Сохраняем p_o как атрибут объекта
        self.p_nn = p_nn  # Сохраняем p_nn как атрибут объекта
        self.p_k = p_k  # Сохраняем p_k как атрибут объекта
        self.t_o = t_o  # Сохраняем t_o как атрибут объекта
        self.t_nn = t_nn  # Сохраняем t_nn как атрибут объекта
        self.t_nv = t_nv  # Сохраняем t_nv как атрибут объекта
        
        self._init_pressures(p_o, p_nn)
        self._init_steam_points(p_o, t_o, p_nn, t_nn, t_nv, p_k)
        self._calc_h_values()
        
    def _init_pressures(self, p_o, p_nn):
        self.delt_p_o = 0.05 * p_o
        self.delt_p_nn = 0.1 * p_nn
        self.delt_p_1 = 0.03 * p_nn
        self.p_1 = p_nn + self.delt_p_nn
        self.p_nv = 1.85 * p_o
        self.p_to_mpa = 1e-6

    def _init_steam_points(self, p_o, t_o, p_nn, t_nn, t_nv, p_k):
        self.point_o_teor = WaterSteam(p=p_o * self.p_to_mpa, t=t_o + 273.15)
        self.point_nn_teor = WaterSteam(p=(p_nn - self.delt_p_nn) * self.p_to_mpa, t=t_nn + 273.15)
        self.point_nn = WaterSteam(p=p_nn * self.p_to_mpa, t=t_nn + 273.15)
        
        s_nn = self.point_nn.find_s()
        self.point_k_teor = WaterSteam(p=p_k * self.p_to_mpa, s=s_nn)
        
        s_o_teor = self.point_o_teor.find_s()
        self.point_1_teor = WaterSteam(p=self.p_1 * self.p_to_mpa, s=s_o_teor)
        
        self.point_k_o = WaterSteam(p=p_k * self.p_to_mpa, x=0)
        self.point_nv = WaterSteam(p=self.p_nv * self.p_to_mpa, t=t_nv + 273.15)
        self.point_o_o = WaterSteam(p=p_o * self.p_to_mpa, t=t_o + 273.15)

    def _calc_h_values(self):
        h_1 = self.point_o_teor.find_h() - (self.point_o_teor.find_h() - self.point_1_teor.find_h()) * self.effect_oi
        h_k = self.point_nn.find_h() - (self.point_nn.find_h() - self.point_k_teor.find_h()) * self.effect_oi
        
        self.point_o = WaterSteam(p=(self.p_o - self.delt_p_o) * self.p_to_mpa, h=self.point_o_teor.find_h())
        self.point_k = WaterSteam(p=self.p_k * self.p_to_mpa, h=h_k)
        self.point_1 = WaterSteam(p=(self.p_1 + self.delt_p_1) * self.p_to_mpa, h=h_1)

    def calc_xi(self):
        t_k_teor = self.point_k_teor.find_t()
        s_nn = self.point_nn.find_s()
        s_k_o = self.point_k_o.find_s()
        s_nv = self.point_nv.find_s()
        h_o = self.point_o.find_h()
        h_1_teor = self.point_1_teor.find_h()
        h_nn = self.point_nn.find_h()
        h_k_o = self.point_k_o.find_h()
        h_nv = self.point_nv.find_h()
        t_nv = self.point_nv.find_t()
        t_o_o = self.point_o_o.find_t()
        
        a = 1 - (t_k_teor * (s_nn - s_k_o)) / ((h_o - h_1_teor) + (h_nn - h_k_o))
        b = 1 - (t_k_teor * (s_nn - s_nv)) / ((h_o - h_1_teor) + (h_nn - h_nv))
        xi_besk = 1 - a / b
        
        otn = (t_nv - t_k_teor) / (t_o_o - t_k_teor)
        
        xi = self._calc_xi_from_otn(otn, xi_besk)
        return xi

    def _calc_xi_from_otn(self, otn, xi_besk):
        if otn < 0.635294:
            return (-0.867 * otn**2 + 1.5222 * otn + 0.147) * xi_besk
        elif 0.635294 <= otn < 0.641177:
            return (5.4399 * otn - 2.6879) * xi_besk
        elif 0.641177 <= otn < 0.752941:
            return (0.0415 * otn**2 + 0.0925 * otn + 0.7236) * xi_besk
        elif 0.752941 <= otn < 0.761176:
            return (3.7886 * otn - 2.0358) * xi_besk
        elif 0.761176 <= otn < 0.858824:
            return (-3.8618 * otn**2 + 6.338 * otn - 1.7389) * xi_besk
        elif 0.858824 <= otn < 0.870588:
            return (2.72 * otn - 1.48) * xi_besk
        elif 0.870588 <= otn < 0.976471:
            return (-0.2644 * otn + 1.1182) * xi_besk
        return xi_besk

    def calc_rashod(self):
        N_to_G = 1e-3
        
        h_o = self.point_o.find_h()
        h_1_teor = self.point_1_teor.find_h()
        h_nn = self.point_nn.find_h()
        h_k_teor = self.point_k_teor.find_h()
        h_k_o = self.point_k_o.find_h()
        h_nv = self.point_nv.find_h()
        h_k = self.point_k.find_h()
        
        self.effect_ip = self._calc_effect_ip(h_o, h_1_teor, h_nn, h_k_teor, h_k_o)
        
        h_1 = h_o - (h_o - h_1_teor) * self.effect_oi
        h_i = self.effect_ip * ((h_o - h_nv) + (h_nn - h_1))
        
        self.g_o = self._calc_g_o(h_i, N_to_G)
        self.g_k = self._calc_g_k(h_k, h_k_o, N_to_G)
        
        return self.g_o, self.g_k

    def _calc_effect_ip(self, h_o, h_1_teor, h_nn, h_k_teor, h_k_o):
        numerator = (h_o - h_1_teor) * self.effect_oi + (h_nn - h_k_teor) * self.effect_oi
        denominator = (h_o - h_1_teor) * self.effect_oi + (h_nn - h_k_o)
        return numerator / denominator / (1 - self.calc_xi())

    def _calc_g_o(self, h_i, N_to_G):
        return self.n * N_to_G / (h_i * self.effect_mex * self.effect_eg)

    def _calc_g_k(self, h_k, h_k_o, N_to_G):
        return (self.n * N_to_G / ((h_k - h_k_o) * self.effect_mex * self.effect_eg)) * ((1 / self.effect_ip) - 1)