import numpy as np
import matplotlib.pyplot as plt
from iapws import IAPWS97
from watersteam import WaterSteam  

class Steam_Rashod:
    def __init__(self, p_o, t_o, p_nn, t_nn, t_nv, p_k, effect_oi, effect_mex, effect_eg, n):
        self.n = n
        delt_p_o = 0.05 * p_o
        delt_p_nn = 0.1 * p_nn
        delt_p_1 = 0.03 * p_nn
        p_1 = p_nn + delt_p_nn
        p_nv = 1.85 * p_o
        p_to_mpa = 1e-6  
        
        self.effect_oi = effect_oi
        self.effect_mex = effect_mex
        self.effect_eg = effect_eg 
        
       
        self.point_o_teor = WaterSteam(p=p_o * p_to_mpa, t=t_o + 273.15)
        self.point_nn_teor = WaterSteam(p=(p_nn - delt_p_nn) * p_to_mpa, t=t_nn + 273.15)
        self.point_nn = WaterSteam(p=p_nn * p_to_mpa, t=t_nn + 273.15)
        
        # Для point_k_teor используем энтропию из point_nn
        s_nn = self.point_nn.find_s()
        self.point_k_teor = WaterSteam(p=p_k * p_to_mpa, s=s_nn)
        
        # Для point_1_teor используем энтропию из point_o_teor
        s_o_teor = self.point_o_teor.find_s()
        self.point_1_teor = WaterSteam(p=p_1 * p_to_mpa, s=s_o_teor)
        
        self.point_k_o = WaterSteam(p=p_k * p_to_mpa, x=0)
        self.point_nv = WaterSteam(p=p_nv * p_to_mpa, t=t_nv + 273.15)
        self.point_o_o = WaterSteam(p=p_o * p_to_mpa, t=t_o + 273.15)
        
        # Расчет h_1 и h_k с учетом КПД
        h_1 = self.point_o_teor.find_h() - (self.point_o_teor.find_h() - self.point_1_teor.find_h()) * self.effect_oi
        h_k = self.point_nn.find_h() - (self.point_nn.find_h() - self.point_k_teor.find_h()) * self.effect_oi
        
        # Инициализация остальных точек
        self.point_o = WaterSteam(p=(p_o - delt_p_o) * p_to_mpa, h=self.point_o_teor.find_h())
        self.point_k = WaterSteam(p=p_k * p_to_mpa, h=h_k)
        self.point_1 = WaterSteam(p=(p_1 + delt_p_1) * p_to_mpa, h=h_1)
        
    def calc_xi(self):
        # Получаем все необходимые свойства один раз
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
        
        # Расчет a и b 
        a = 1 - (t_k_teor * (s_nn - s_k_o)) / ((h_o - h_1_teor) + (h_nn - h_k_o))
        b = 1 - (t_k_teor * (s_nn - s_nv)) / ((h_o - h_1_teor) + (h_nn - h_nv))
        xi_besk = 1 - a / b
        
        # Расчет отношения температур
        otn = (t_nv - t_k_teor) / (t_o_o - t_k_teor)
        
        # Определение xi в зависимости от otn
        if otn < 0.635294:
            xi = (-0.867 * otn**2 + 1.5222 * otn + 0.147) * xi_besk
        elif 0.635294 <= otn < 0.641177:
            xi = (5.4399 * otn - 2.6879) * xi_besk
        elif 0.641177 <= otn < 0.752941:
            xi = (0.0415 * otn**2 + 0.0925 * otn + 0.7236) * xi_besk
        elif 0.752941 <= otn < 0.761176:
            xi = (3.7886 * otn - 2.0358) * xi_besk
        elif 0.761176 <= otn < 0.858824:
            xi = (-3.8618 * otn**2 + 6.338 * otn - 1.7389) * xi_besk
        elif 0.858824 <= otn < 0.870588:
            xi = (2.72 * otn - 1.48) * xi_besk
        elif 0.870588 <= otn < 0.976471:
            xi = (-0.2644 * otn + 1.1182) * xi_besk
        else:
            xi = xi_besk  
            
        return xi
        
    def calc_rashod(self):
        N_to_G = 1e-3  
        
        # Получаем все необходимые свойства
        h_o = self.point_o.find_h()
        h_1_teor = self.point_1_teor.find_h()
        h_nn = self.point_nn.find_h()
        h_k_teor = self.point_k_teor.find_h()
        h_k_o = self.point_k_o.find_h()
        h_nv = self.point_nv.find_h()
        h_k = self.point_k.find_h()
        
        # Расчет эффективности
        numerator = (h_o - h_1_teor) * self.effect_oi + (h_nn - h_k_teor) * self.effect_oi
        denominator = (h_o - h_1_teor) * self.effect_oi + (h_nn - h_k_o)
        self.effect_ip = numerator / denominator / (1 - self.calc_xi())
        
        # Расчет расходов
        h_1 = h_o - (h_o - h_1_teor) * self.effect_oi
        h_i = self.effect_ip * ((h_o - h_nv) + (h_nn - h_1))
        
        self.g_o = self.n * N_to_G / (h_i * self.effect_mex * self.effect_eg)
        self.g_k = (self.n * N_to_G / ((h_k - h_k_o) * self.effect_mex * self.effect_eg)) * ((1 / self.effect_ip) - 1)
        
        return self.g_o, self.g_k