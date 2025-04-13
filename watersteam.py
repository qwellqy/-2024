from iapws import IAPWS97

class WaterSteam:
 
    
    def __init__(self, p=None, t=None, h=None, s=None, x=None, v=None, phase=None, *args, **kwargs):
        """
        Параметры:
            p: Давление [MPa]
            t: Температура [K]
            h: Энтальпия [kJ/kg]
            s: Энтропия [kJ/kg·K]
            x: Степень сухости (0-1)
            v: Удельный объем [m³/kg]
            phase: Фазовое состояние
        """
        self.p = p
        self.t = t 
        self.h = h
        self.s = s
        self.x = x
        self.v = v
        self.phase = phase

    def _create_state(self, **kwargs):
        """Создает объект IAPWS97 с текущими параметрами."""
        params = {
            'P': self.p,
            'T': self.t,
            'h': self.h,
            's': self.s,
            'x': self.x,
            'v': self.v,
            'phase': self.phase
        }
        # Объединяем с переданными параметрами, исключая None
        return IAPWS97(**{**{k: v for k, v in params.items() if v is not None}, **kwargs})

    # Базовые свойства
    def find_p(self, *args, **kwargs):
        """Возвращает давление [MPa]."""
        return self._create_state().P
        
    def find_t(self, *args, **kwargs):
        """Возвращает температуру [K]."""
        return self._create_state().T
    
    def find_h(self, *args, **kwargs):
        """Возвращает энтальпию [kJ/kg]."""
        return self._create_state().h
        
    def find_v(self, *args, **kwargs):
        """Возвращает удельный объем [m³/kg]."""
        return self._create_state().v
      
    def find_s(self, *args, **kwargs):
        """Возвращает энтропию [kJ/kg·K]."""
        return self._create_state().s
        
    def find_x(self, *args, **kwargs):
        """Возвращает степень сухости [0-1]."""
        return self._create_state().x
        
    def find_phase(self, *args, **kwargs):
        """Возвращает фазовое состояние."""
        return self._create_state().phase

    # Свойства насыщения
    def find_t_nas_po_p(self, *args, **kwargs):
        """Температура насыщения по давлению [K]."""
        if self.p is None:
            raise ValueError("Требуется задать давление (p)")
        return IAPWS97(P=self.p, x=1).T
        
    def find_h_nas_po_p(self, *args, **kwargs):
        """Энтальпия насыщения по давлению [kJ/kg]."""
        if self.p is None:
            raise ValueError("Требуется задать давление (p)")
        return IAPWS97(P=self.p, x=1).h
        
    def find_s_nas_po_p(self, *args, **kwargs):
        """Энтропия насыщения по давлению [kJ/kg·K]."""
        if self.p is None:
            raise ValueError("Требуется задать давление (p)")
        return IAPWS97(P=self.p, x=1).s
    
    def find_h_nas_po_t(self, *args, **kwargs):
        """Энтальпия насыщения по температуре [kJ/kg]."""
        if self.t is None:
            raise ValueError("Требуется задать температуру (t)")
        return IAPWS97(T=self.t, x=1).h
        
    def find_p_nas_po_t(self, *args, **kwargs):
        """Давление насыщения по температуре [MPa]."""
        if self.t is None:
            raise ValueError("Требуется задать температуру (t)")
        return IAPWS97(T=self.t, x=0).P
        
    def find_s_nas_po_t(self, *args, **kwargs):
        """Энтропия насыщения по температуре [kJ/kg·K]."""
        if self.t is None:
            raise ValueError("Требуется задать температуру (t)")
        return IAPWS97(T=self.t, x=1).s

    # Термодинамические процессы
    def calculation_isentropic_process(self, p_1=None, t_1=None, h_1=None, s=None, p_2=None, t_2=None, h_2=None, *args, **kwargs):
        """Расчет изоэнтропного процесса."""
        initial = IAPWS97(P=p_1, T=t_1, h=h_1, s=s)
        final = IAPWS97(P=p_2, T=t_2, h=h_2, s=initial.s)
        return final.P, final.v, final.T
        
    def calculation_isothermal_process(self, p_1=None, t=None, h_1=None, s_1=None, p_2=None, s_2=None, h_2=None, *args, **kwargs):
        """Расчет изотермического процесса."""
        initial = IAPWS97(P=p_1, T=t, h=h_1, s=s_1)
        final = IAPWS97(P=p_2, T=initial.T, h=h_2, s=s_2)
        return final.P, final.v, final.T
        
    def calculation_isobaric_process(self, p=None, t_1=None, h_1=None, s_1=None, t_2=None, s_2=None, h_2=None):
        """Расчет изобарного процесса."""
        initial = IAPWS97(P=p, T=t_1, h=h_1, s=s_1)
        final = IAPWS97(P=initial.P, T=t_2, h=h_2, s=s_2)
        return final.P, final.v, final.T
        
    def calculation_isochoric_process(self, p_1=None, t_1=None, h_1=None, s_1=None, p_2=None, t_2=None, s_2=None, h_2=None, v=None):
        """Расчет изохорного процесса."""
        initial = IAPWS97(P=p_1, T=t_1, h=h_1, s=s_1, V=v)
        final = IAPWS97(P=p_2, T=t_2, h=h_2, s=s_2, v=initial.v)
        return final.P, final.v, final.T
    
    
