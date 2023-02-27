import numpy as np
import random

class Cyclist():
    # fixed global params
    g = 9.81
    Cd = 0.7
    A = 0.5
    Cr = 0.001
    ro = 1.225
    n_mech = 0.97
    mmd = 0.53
    max_his = np.inf # maximum number of historic power exertion to consider

    def __init__(self, sex, mass, hr_0, rise_time, hr_max, kf, c, n=0):
        self.sex = sex
        self.mass = mass
        self.hr_0 = hr_0
        self.rise_time = rise_time
        self.hr_max = hr_max
        self.kf = kf
        self.c = c

        self.n = n

        self.reset()

    def reset(self):
        self.hr = self.hr_0
        self.power_history = []
        self.total_rdd = 0

    def eval_segment(self, exposure, power, t=np.inf):
        percieved_power = power + self.kf*sum(self.power_history)

        if len(self.power_history) > self.max_his:
            self.power_history.pop(0)
        for i in range(int(t)):
            self.power_history.append(power)

        self.hr = self._steadystate_hr(percieved_power, t)

        rdd = self._vent_rate() * self._deposition_frac() * t/60 * exposure/1000
        self.total_rdd += rdd

        return rdd
    
    def get_segment_power(self, d_height, l, v=20):
        P_g = self.g * self.mass * d_height/l * v
        P_a = 0.5 * self.Cd * self.ro * self.A * v**3
        P_f = self.Cr * self.mass * self.g * v

        return sum([P_g, P_a, P_f])
        # return max(P_g + P_a + P_f, 0)
    
    def _steadystate_hr(self, power, t=np.inf):
        hr_ss = self.hr_0 + self.c*power
        if hr_ss > self.hr_max:
            return self.hr_max
        
        hr = hr_ss + (self.hr_0 - hr_ss) * pow(np.e, -t/self.rise_time)
        if hr < self.hr_0:
            return self.hr_0
        return hr

    def _vent_rate(self):
        if self.sex=='M':
            return np.exp(0.021*self.hr + 1.03)
        else:
            return np.exp(0.023*self.hr + 0.57)

    def _inhaled_frac(self):
        return 1 - 0.5*(1 - 1/(1 + 0.00076 * self.mmd**2.8))

    def _deposition_frac(self):
        return self._inhaled_frac() * (0.0587 + 0.911/(1 + np.exp(4.77 + 1.485 * np.log(self.mmd))) + 0.943/(1 + np.exp(0.508 - 2.58 * np.log(self.mmd))))

    def __repr__(self):
        return f"mass: {self.mass:.2f}\thr_0: {self.hr_0:.2f}\trise_time: {self.rise_time:.2f}\thr_max: {self.hr_max:.2f}\tkf: {self.kf:.2f}\tc: {self.c:.2f}"

def random_cyclists(n, mass_mean, mass_std, hr_0_mean, hr_0_std, rise_time_mean, rise_time_std,
                    hr_max_mean, hr_max_std, kf_mean, kf_std, c_mean, c_std):
    cyclists = []
    for i in range(n):
        cyclist = Cyclist(sex=random.choice(['M','F']), mass=random.normalvariate(mass_mean, mass_std),
                          hr_0=random.normalvariate(hr_0_mean, hr_0_std), rise_time=random.normalvariate(rise_time_mean, rise_time_std),
                          hr_max=random.normalvariate(hr_max_mean, hr_max_std), kf=random.normalvariate(kf_mean, kf_std),
                          c=random.normalvariate(c_mean, c_std), n=i)
        cyclists.append(cyclist)

    return cyclists

config = {
    'sex': 'M',
    'mass': 90,
    'hr_0': 60,
    'rise_time': 30,
    'hr_max': 200,
    'kf': 6e-5,
    'c': 0.15,
}

if __name__ == "__main__":
    # cyclist = Cyclist(**config)
    cyclist = random_cyclists(1, mass_mean=90, mass_std=5, hr_0_mean=70, hr_0_std=10, rise_time_mean=30, rise_time_std=5,
                              hr_max_mean=190, hr_max_std=20, kf_mean=3e-5, kf_std=1e-5, c_mean=0.3, c_std=0.05)[0]
    print(cyclist)
    ambient_pm = 10
    velocity = 20

    tot_d, tot_h, tot_t = 0, 0, 0

    for i in range(20):
        dh = random.normalvariate(0, 1)
        dl = random.randint(0, 1000)

        p_req = cyclist.get_segment_power(dh, dl, velocity/3.6)
        cyclist.eval_segment(ambient_pm, p_req, dl*3.6/velocity)
        print(cyclist.hr, "\t", cyclist.total_rdd)

        tot_d += dl
        tot_h += dh
        tot_t += dl*3.6/velocity

    print(f"total distance {tot_d/1000:.2f} km, net elevation change {tot_h:.2f} m, duration {tot_t/60:.2f} minutes")