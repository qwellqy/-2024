class IdealGas:
    def __init__(
        self,
        c_p=None,
        r=None,
        t=None,
        p=None,
        h=None,
        c_v=None,
        gamma=None,
        v=None,
        *args,
        **kwargs
    ):
        """Инициализация параметров идеального газа.
        
        Аргументы:
            c_p: Удельная теплоемкость при постоянном давлении [Дж/(кг·К)]
            r: Газовая постоянная [Дж/(кг·К)]
            t: Температура [К]
            p: Давление [Па]
            h: Энтальпия [Дж/кг]
            c_v: Удельная теплоемкость при постоянном объеме [Дж/(кг·К)]
            gamma: Показатель адиабаты (Cp/Cv)
            v: Удельный объем [м³/кг]
        """
        self.t = t  # Температура
        self.p = p  # Давление
        self.c_p = c_p  # Теплоемкость при постоянном давлении
        self.r = r  # Газовая постоянная
        self.h = h  # Энтальпия
        self.c_v = c_v  # Теплоемкость при постоянном объеме
        self.gamma = gamma  # Показатель адиабаты
        self.v = v  # Удельный объем

    def _validate_params(self, required_params, error_message):
        """Внутренний метод для проверки наличия необходимых параметров."""
        if all(getattr(self, param) is None for param in required_params):
            raise ValueError(error_message)

    def find_p(self):
        """Рассчитать давление по уравнению состояния идеального газа: P = R*T/v"""
        if self.p is None:
            self._validate_params(
                ["t", "v", "r"],
                "Недостаточно параметров для расчета давления (требуются t, v, r)"
            )
            return self.r * self.t / self.v
        return self.p

    def find_v(self):
        """Рассчитать удельный объем по уравнению состояния: v = R*T/P"""
        if self.v is None:
            self._validate_params(
                ["t", "p", "r"],
                "Недостаточно параметров для расчета объема (требуются t, p, r)"
            )
            return self.r * self.t / self.p
        return self.v

    def find_h(self):
        """Рассчитать энтальпию: h = Cp*T"""
        if self.h is None:
            self._validate_params(
                ["c_p", "t"],
                "Недостаточно параметров для расчета энтальпии (требуются c_p, t)"
            )
            return self.c_p * self.t
        return self.h

    def find_t(self):
        """Рассчитать температуру по уравнению состояния: T = P*v/R"""
        if self.t is None:
            self._validate_params(
                ["p", "v", "r"],
                "Недостаточно параметров для расчета температуры (требуются p, v, r)"
            )
            return self.p * self.v / self.r
        return self.t

    def find_c_v(self):
        """Рассчитать теплоемкость при постоянном объеме: Cv = Cp - R"""
        if self.c_v is None:
            self._validate_params(
                ["c_p", "r"],
                "Недостаточно параметров для расчета Cv (требуются c_p, r)"
            )
            return self.c_p - self.r
        return self.c_v

    def find_gamma(self):
        """Рассчитать показатель адиабаты: γ = Cp/(Cp - R)"""
        if self.gamma is None:
            self._validate_params(
                ["c_p", "r"],
                "Недостаточно параметров для расчета γ (требуются c_p, r)"
            )
            return self.c_p / (self.c_p - self.r)
        return self.gamma

    def calculation_isothermal_process(self, v_1=None, p_1=None, v_2=None, p_2=None):
        """Расчет параметров изотермического процесса (T=const)."""
        if sum(param is not None for param in [v_1, p_1, v_2, p_2]) < 3:
            raise ValueError("Для изотермического процесса требуется минимум 3 параметра")

        t = p_1 * v_1 / self.r
        if v_2 is not None:
            p_2 = self.r * t / v_2
        else:
            v_2 = self.r * t / p_2
        return p_2, v_2, t

    def calculation_isobaric_process(self, v_1=None, t_1=None, v_2=None, t_2=None):
        """Расчет параметров изобарного процесса (P=const)."""
        if sum(param is not None for param in [v_1, t_1, v_2, t_2]) < 3:
            raise ValueError("Для изобарного процесса требуется минимум 3 параметра")

        p = self.r * t_1 / v_1
        if v_2 is not None:
            t_2 = p * v_2 / self.r
        else:
            v_2 = self.r * t_2 / p
        return p, v_2, t_2

    def calculation_isochoric_process(self, t_1=None, p_1=None, t_2=None, p_2=None):
        """Расчет параметров изохорного процесса (V=const)."""
        if sum(param is not None for param in [t_1, p_1, t_2, p_2]) < 3:
            raise ValueError("Для изохорного процесса требуется минимум 3 параметра")

        v = self.r * t_1 / p_1
        if p_2 is not None:
            t_2 = p_2 * v / self.r
        else:
            p_2 = self.r * t_2 / v
        return p_2, v, t_2

    def calculation_isentropic_process_pv(self, p_1=None, p_2=None, v_1=None, v_2=None):
        """Расчет параметров изоэнтропного процесса через зависимость P-V."""
        if sum(param is not None for param in [p_1, p_2, v_1, v_2]) < 3:
            raise ValueError("Для изоэнтропного PV-процесса требуется минимум 3 параметра")

        gamma = self.find_gamma()
        if p_2 is not None:
            v_2 = ((p_1 * v_1 ** gamma) / p_2) ** (1/gamma)
        else:
            p_2 = (p_1 * v_1 ** gamma) / (v_2 ** gamma)
        return p_2, v_2, gamma

    def calculation_isentropic_process_tv(self, t_1=None, t_2=None, v_1=None, v_2=None):
        """Расчет параметров изоэнтропного процесса через зависимость T-V."""
        if sum(param is not None for param in [t_1, t_2, v_1, v_2]) < 3:
            raise ValueError("Для изоэнтропного TV-процесса требуется минимум 3 параметра")

        gamma = self.find_gamma()
        if t_2 is not None:
            v_2 = ((t_1 * v_1 ** (gamma - 1)) / t_2) ** (1/(gamma - 1))
        else:
            t_2 = (t_1 * v_1 ** (gamma - 1)) / (v_2 ** (gamma - 1))
        return v_2, t_2, gamma

    def determine_state(self, p=None, v=None, t=None):
        """Определить состояние газа"""
        return "Gas"

    def find_quality(self, p=None, v=None, t=None):
        """Найти степень сухости"""
        return 1
