import numpy as np
from watersteam import WaterSteam
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class SBS:
    
    def __init__(self, d, n, n_stages, G_0, P_o, h_o, P_z, fi, alpha_1, ro, mu, efficiency, veernost_1):
        self.speed_stage_diam = d
        self.rotation_speed = n
        self.n_stages = n_stages
        self.mass_flow = G_0
        self.p0 = P_o
        self.h0 = h_o
        self.pz = P_z
        self.delta_diam = 0.2
        self.speed_coefficient = fi
        self.alpha_1 = alpha_1 
        self.root_reaction_degree = ro
        self.discharge_coefficient = mu
        self.overlapping = 0.003
        self.efficiency = efficiency 
        self.veernost_1 = veernost_1
        
    def get_reaction_degree(self, root_dor, veernost):
        return root_dor + (1.8 / (veernost + 1.8))
        
    def get_u_cf(self, dor):
        cos = np.cos(np.deg2rad(self.alpha_1))
        return self.speed_coefficient * cos / (2 * (1 - dor) ** 0.5)

    def get_heat_drop(self, diameter, u_cf):
        first = (diameter / u_cf) ** 2
        second = (self.rotation_speed / 50) ** 2
        return 12.3 * first * second
        
    def equation_to_solve(self, x):
        return x ** 2 + x * self.root_diameter - self.avg_diam_1 * self.blade_length_2 * self.point_z.find_v() / self.point_2.find_v()
        
    def linear_distribution(self, left, right, x):
        return (right - left) * x + left
        
    def calc(self):
        self.avg_diam_1 = self.speed_stage_diam - self.delta_diam
        self.point_0 = WaterSteam(p=self.p0, h=self.h0)
        self.avg_reaction_degree_1 = self.get_reaction_degree(self.root_reaction_degree, self.veernost_1)
        self.u_cf_1 = self.get_u_cf(self.avg_reaction_degree_1)
        self.heat_drop_1 = self.get_heat_drop(self.avg_diam_1, self.u_cf_1)
        self.h1 = self.point_0.find_h() - self.heat_drop_1
        self.point_2 = WaterSteam(h=self.h1, s=self.point_0.find_s())
        self.upper = self.mass_flow * self.point_2.find_v() * self.u_cf_1
        self.lower = self.discharge_coefficient * np.sin(np.deg2rad(self.alpha_1)) * self.rotation_speed * (np.pi * self.avg_diam_1) ** 2 * (1 - self.avg_reaction_degree_1) ** 0.5
        self.blade_length_1 = self.upper / self.lower
        self.blade_length_2 = self.blade_length_1 + self.overlapping
        self.root_diameter = self.avg_diam_1 - self.blade_length_2
        self.point_zt = WaterSteam(p=self.pz, s=self.point_0.find_s())
        self.full_heat_drop = self.h0 - self.point_zt.find_h()
        self.actual_heat_drop = self.full_heat_drop * self.efficiency
        self.hz = self.h0 - self.actual_heat_drop
        self.point_z = WaterSteam(p=self.pz, h=self.hz)
        self.blade_length_z = fsolve(self.equation_to_solve, 0.01)[0]
        self.avg_diam_2 = self.root_diameter + self.blade_length_z
        self.x = np.cumsum(np.ones(self.n_stages) * 1 / (self.n_stages - 1)) - 1 / (self.n_stages - 1)
        self.diameters = self.linear_distribution(self.avg_diam_1, self.avg_diam_2, self.x)
        self.blade_lengths = self.linear_distribution(self.blade_length_2, self.blade_length_z, self.x)
        self.veernosts = self.diameters / self.blade_lengths
        self.reaction_degrees = self.get_reaction_degree(root_dor=self.root_reaction_degree, veernost=self.veernosts)
        self.u_cf = self.get_u_cf(dor=self.reaction_degrees)
        self.heat_drops = self.get_heat_drop(self.diameters, self.u_cf)
        self.output_speed_coeff_loss = np.full_like(self.heat_drops, 0.95)
        self.output_speed_coeff_loss[0] = 1
        self.actual_heat_drops = self.output_speed_coeff_loss * self.heat_drops
        self.mean_heat_drop = np.mean(self.actual_heat_drops)
        self.reheat_factor = 4.8 * 10 ** (-4) * (1 - self.efficiency) * self.full_heat_drop * (self.n_stages - 1) / self.n_stages
        self.bias = self.full_heat_drop * (1 + self.reheat_factor) - np.sum(self.actual_heat_drops)
        self.bias = self.bias / self.n_stages
        self.new_actual_heat_drop = self.actual_heat_drops + self.bias
        
        return self.full_heat_drop * (1 + self.reheat_factor) / self.mean_heat_drop, self.avg_diam_1 / self.blade_length_1
        
    def plot_distribution(self, values, ax_name, title):
        fig, ax = plt.subplots(1, 1, figsize=(15,5))
        ax.plot(range(1, 13), values, marker='o')
        ax.set_xlabel("Номер ступени")
        ax.set_ylabel(ax_name)
        plt.title(f"Распределение {title} по проточной части")
        ax.grid(True)
        plt.show()
        
    def plot_charts(self):
        self.plot_distribution(self.diameters, "d, m", "средних диаметров")
        self.plot_distribution(self.blade_lengths, "l, m", "высот лопаток")
        self.plot_distribution(self.veernosts, "Веерность", "обратной верности")
        self.plot_distribution(self.reaction_degrees, "Степень реактивности", "степени реактивности")
        self.plot_distribution(self.u_cf, "U/Cф", "U/c_f")
        self.plot_distribution(self.new_actual_heat_drop, "Теплоперепады по ступеням", "теплоперепадов с учетом невязки")

    def table(self):
        self.tabl_one = pd.DataFrame({
            "Веерность": [np.round(self.veernost_1,2), np.round(self.avg_diam_1 / self.blade_length_1,2)],
            "": ["Величина обратной веерности, задаваемая студентом", "Величина расчитанной обратной веерности"],
        }).set_index("")
        
        self.tabl_two = pd.DataFrame({
            "Число ступеней": [np.round(self.n_stages,0), np.round(self.full_heat_drop * (1 + self.reheat_factor) / self.mean_heat_drop,0)],
            "": ["Величина числа ступеней, задаваемая студентом", "Величина расчитанного числа ступеней"],
        }).set_index("")
        return self.tabl_one, self.tabl_two