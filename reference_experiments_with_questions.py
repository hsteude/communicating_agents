import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import optimize
from loguru import logger


K_E = 8.99e9  # Coulomb constant
G = 9.81  # Gravety constant


class RefExperimentMass():

    def __init__(self, m=[2e-12, 1e-12], m_ref=2e-12, v_ref=1, N=100,
                 alpha=[0, 0], dt=.1, gravity=True):
        self.m = np.array(m)
        self.dt = dt
        self.m_ref = m_ref
        self.v_ref = v_ref
        self.alpha = np.array(alpha)
        self.N = N
        self.g = G if gravity else 0
        self.set_initial_state()

    def set_initial_state(self):
        self.t = 0
        self.m = np.array(self.m)
        self.x = np.zeros(2)
        self.z = np.zeros(2)
        self.v_x = np.zeros(2)
        self.v_z = np.zeros(2)
        self.x_series = np.empty((self.N, len(self.m)))
        self.z_series = np.empty_like(self.x_series)
        self.t_series = np.empty((self.N, 1))
        self.v_x_series = np.empty_like(self.x_series)
        self.v_z_series = np.empty_like(self.x_series)

    def _get_initial_velocity(self):
        v_abs = 2 * self.m_ref / (self.m + self.m_ref) * self.v_ref
        v_x = v_abs * np.cos(self.alpha)
        v_z = v_abs * np.sin(self.alpha)
        return v_x, v_z

    def _update_velocity(self):
        # velocity after elastic collision
        if self.t == 0:
            self.v_x, self.v_z = self._get_initial_velocity()
        else:
            self.v_x = self.v_x
            self.v_z = self.v_z - self.g * self.dt

    def _update_position(self):
        # update position
        self.x += self.v_x * self.dt
        self.z += self.v_z * self.dt + .5 * self.g * self.dt**2
        # update time
        self.t += self.dt

    def run(self):
        for i in range(self.N):
            self.x_series[i, :] = self.x
            self.z_series[i, :] = self.z
            self.v_x_series[i, :] = self.v_x
            self.v_z_series[i, :] = self.v_z
            self.t_series[i, :] = self.t
            self._update_velocity()
            self._update_position()
            self.t += self.dt

    def visualize(self, golf_hole_loc=20):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        fig = make_subplots(rows=2, cols=1)
        trace_pp1 = go.Scatter(x=self.x_series[:, 0], y=self.z_series[:, 0],
                               name='Particle 1')
        trace_pp2 = go.Scatter(x=self.x_series[:, 1], y=self.z_series[:, 1],
                               name='Particle 2')
        trace_golf_hole = go.Scatter(
            x=[golf_hole_loc - .1 * golf_hole_loc,
                golf_hole_loc + .1 * golf_hole_loc],
            y=[0, 0],
            name='Golf hole')
        fig.add_trace(trace_pp1, row=1, col=1)
        fig.add_trace(trace_pp2, row=1, col=1)
        fig.add_trace(trace_golf_hole, row=1, col=1)
        title = f'RE1: m_ref = {self.m_ref} kg, m = {self.m} kg'\
            f' alha = [{round(self.alpha[0]/np.pi, 2)} '\
            f'{round(self.alpha[1]/np.pi, 2)}] pi, v_ref = {self.v_ref}'
        fig.update_layout(title_text=title)
        fig.update_xaxes(title_text="Position x [m]", row=1, col=1)
        fig.update_yaxes(title_text="Position z [m]", row=1, col=1)
        fig.show()


class RefExperimentCharge():

    def __init__(self, m=[2e-12, 1e-12], q=[3e-9, 4e-9],
                 q_ref=-1e-9, d=.1, N=10):
        self.q = np.array(q)
        self.m = np.array(m)
        self.x = np.zeros_like(self.m)
        self.v = np.zeros_like(self.m)
        self.a = np.zeros_like(self.m)
        self.last_a = np.zeros_like(m)
        self.q_ref = q_ref
        self.x_ref = d
        self.dt = 1e-6
        self.t = 0
        self.N = N
        self.x_series = np.empty((N, 2))
        self.v_series = np.empty_like(self.x_series)
        self.a_series = np.empty_like(self.v_series)
        self.t_series = np.empty((N, 1))

    def _update_acceleration(self):
        # get coulomb forces
        F_c = K_E * self.q * self.q_ref / ((self.x - self.x_ref)**2) \
            * (self.x - self.x_ref) / np.abs(self.x - self.x_ref)

        if any(F_c) < 1:
            logger.warning('Parcicle passed by reference particel,'
                           ' this is not considered a valid examle')

        # update acceleration for each particle
        self.a = F_c / self.m

    def _update_velocity(self):
        self.v = self.v + .5 * (self.last_a + self.a) * self.dt

    def _update_position(self):
        self.x += self.v * self.dt + .5 * self.a * self.dt**2

    def run(self):
        for i in range(self.N):
            self._update_acceleration()
            self._update_velocity()
            self._update_position()
            self.last_a = self.a
            self.t += self.dt
            self.x_series[i, :] = self.x
            self.v_series[i, :] = self.v
            self.a_series[i, :] = self.a
            self.t_series[i, :] = self.t

    def plot(self, i):
        import plotly.express as px
        fig = px.line(
            x=self.t_series, y=self.x_series[:, i],
            labels={'x': 't [s]', 'y': 'x_position [m]'},
            title=f'Ref. exp charge: m_ref = {self.m_ref}, m = {self.m[i]},'
            f' q_ref = {self.q_ref} q = {self.q[i]}, d0 = {self.x_ref}')
        fig.show()


def objective_golf_mass(alpha, exp, golf_hole_loc=0.05):
    PENALTY_VALUE = 100
    exp.set_initial_state()
    exp.alpha = alpha
    exp.run()

    def get_value_for_particle(i):
        zero_cross = np.where(np.diff(np.sign(exp.z_series[1:, i])))[0]
        if len(zero_cross) != 1:
            logger.warning(f'Particle {i} did not hit the ground')
            value = PENALTY_VALUE
        else:
            x_at_zero_cross = (
                exp.x_series[zero_cross + 1][0][i]
                + exp.x_series[zero_cross + 2][0][i]) / 2
            value = np.abs(x_at_zero_cross - golf_hole_loc)
        return value

    return np.sum([get_value_for_particle(i) for i in [0, 1]])


if __name__ == '__main__':
    DT = .01
    N = 100
    GOLF_HOLE_LOC = 0.1
    rem = RefExperimentMass(N=100, dt=.01)
    rem.run()
    rem.visualize(golf_hole_loc=GOLF_HOLE_LOC)
    initial_guess = [0.4999 * np.pi, 0.4999 * np.pi]
    alpha_star = optimize.minimize(
        objective_golf_mass, initial_guess, method='Nelder-Mead', args=(rem, GOLF_HOLE_LOC)).x
    rem_opt = RefExperimentMass(alpha=alpha_star, N=100, dt=.01)
    rem_opt.run()
    rem_opt.visualize(golf_hole_loc=GOLF_HOLE_LOC)
