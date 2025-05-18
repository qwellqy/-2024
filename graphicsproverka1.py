from typing import List, Tuple, Optional
from watersteam import WaterSteam
import matplotlib.pyplot as plt
import numpy as np

def legend_without_duplicate_labels(ax: plt.Axes) -> None:
    """Убирает дубликаты из легенды графика"""
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def plot_process(ax: plt.Axes, points: List[WaterSteam], **kwargs) -> None:
    """Отрисовка процесса расширения по точкам"""
    ax.plot([point.find_s() for point in points], [point.find_h() for point in points], **kwargs)

def get_isobar(point: WaterSteam) -> Tuple[List[float], List[float]]:
    """Собрать координаты изобары в hs осях"""
    s = point.find_s()
    s_values = np.arange(s * 0.9, s * 1.1, 0.2 * s / 1000)
    h_values = [WaterSteam(p=point.find_p(), s=_s).find_h() for _s in s_values]
    return s_values, h_values

def _get_isoterm_steam(point: WaterSteam) -> Tuple[List[float], List[float]]:
    """Собрать координаты изотермы для пара в hs осях"""
    t = point.find_t()
    p = point.find_p()
    s = point.find_s()
    s_max = s * 1.2
    s_min = s * 0.8
    p_values = np.arange(p * 0.8, p * 1.2, 0.4 * p / 1000)
    h_values = np.array([WaterSteam(p=_p, t=t).find_h() for _p in p_values])
    s_values = np.array([WaterSteam(p=_p, t=t).find_s() for _p in p_values])
    mask = (s_values >= s_min) & (s_values <= s_max)
    return s_values[mask], h_values[mask]

def _get_isoterm_two_phases(point: WaterSteam) -> Tuple[List[float], List[float]]:
    """Собрать координаты изотермы для влажного пара в hs осях"""
    x = point.find_x()
    p = point.find_p()
    x_values = np.arange(x * 0.9, min(x * 1.1, 1), (1 - x) / 1000)
    h_values = np.array([WaterSteam(p=p, x=_x).find_h() for _x in x_values])
    s_values = np.array([WaterSteam(p=p, x=_x).find_s() for _x in x_values])
    return s_values, h_values

def get_isoterm(point: WaterSteam) -> Tuple[List[float], List[float]]:
    """Собрать координаты изотермы в hs осях"""
    if point.find_phase() == 'Two phases':
        return _get_isoterm_two_phases(point)
    return _get_isoterm_steam(point)

def plot_isolines(ax: plt.Axes, point: WaterSteam) -> None:
    """Отрисовка изобары и изотермы"""
    s_isobar, h_isobar = get_isobar(point)
    s_isoterm, h_isoterm = get_isoterm(point)
    ax.plot(s_isobar, h_isobar, color='green', label='Изобара')
    ax.plot(s_isoterm, h_isoterm, color='blue', label='Изотерма')

def plot_points(ax: plt.Axes, points: List[WaterSteam]) -> None:
    """Отрисовать точки на hs-диаграмме"""
    for point in points:
        ax.scatter(point.find_s(), point.find_h(), s=50, color="red")
        plot_isolines(ax, point)

def get_humidity_constant_line(
    point: WaterSteam,
    max_p: float,
    min_p: float,
    x: Optional[float]=None
) -> Tuple[List[float], List[float]]:
    """Собрать координаты линии с постоянной степенью сухости в hs осях"""
    _x = x if x else point.find_x()
    p_values = np.arange(min_p, max_p, (max_p - min_p) / 1000)
    h_values = np.array([WaterSteam(p=_p, x=_x).find_h() for _p in p_values])
    s_values = np.array([WaterSteam(p=_p, x=_x).find_s() for _p in p_values])
    return s_values, h_values

def plot_humidity_lines(ax: plt.Axes, points: List[WaterSteam]) -> None:
    """Отрисовать изолинии для степеней сухости на hs-диаграмме"""
    pressures = [point.find_p() for point in points]
    min_pressure = min(pressures) if min(pressures) > 700/1e6 else 700/1e6
    max_pressure = max(pressures) if max(pressures) < 22 else 22
    for point in points:
        if point.find_phase() == 'Two phases':
            s_values, h_values = get_humidity_constant_line(point, max_pressure, min_pressure, x=1)
            ax.plot(s_values, h_values, color="gray")
            s_values, h_values = get_humidity_constant_line(point, max_pressure, min_pressure)
            ax.plot(s_values, h_values, color="gray", label='Линия сухости')
            ax.text(s_values[10], h_values[10], f'x={round(point.find_x(), 2)}')

def plot_hs_diagram(ax: plt.Axes, points: List[WaterSteam]) -> None:
    """
    Построить изобары и изотермы для переданных точек. 
    Если степень сухости у точки не равна 1, то построется
    дополнительно линия соответствующей степени сухости
    """
    plot_points(ax, points)
    plot_humidity_lines(ax, points)
    ax.set_xlabel(r"S, $\frac{кДж}{кг * K}$", fontsize=14)
    ax.set_ylabel(r"h, $\frac{кДж}{кг}$", fontsize=14)
    ax.set_title("HS-диаграмма процесса расширения", fontsize=18)
    ax.legend()
    ax.grid()
    legend_without_duplicate_labels(ax)