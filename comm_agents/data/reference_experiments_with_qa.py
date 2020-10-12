import numpy as np
import math
from scipy import optimize
from loguru import logger


K_E = 8.99e9  # Coulomb constant
G = 9.81  # Gravety constant


class RefExperimentMass():

    def __init__(self, m=[2e-12, 1e-12], m_ref=2e-12, v_ref=1, N=100,
                 alpha=[0, 0], dt=.1, gravity=True, **kwargs):
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
        fig = make_subplots(rows=1, cols=1)
        trace_pp1 = go.Scatter(x=self.x_series[:, 0], y=self.z_series[:, 0],
                               name='Particle 1', mode='lines+markers')
        trace_pp2 = go.Scatter(x=self.x_series[:, 1], y=self.z_series[:, 1],
                               name='Particle 2', mode='lines+markers')
        trace_golf_hole = go.Scatter(
            x=[golf_hole_loc - .1 * golf_hole_loc,
                golf_hole_loc + .1 * golf_hole_loc],
            y=[0, 0],
            name='Golf hole')
        fig.add_trace(trace_pp1, row=1, col=1)
        fig.add_trace(trace_pp2, row=1, col=1)
        fig.add_trace(trace_golf_hole, row=1, col=1)
        title = f'RE1: m_ref = {self.m_ref} kg, m = {self.m} kg'\
            f' alpha = [{round(self.alpha[0]/np.pi, 2)} '\
            f'{round(self.alpha[1]/np.pi, 2)}] pi, v_ref = {self.v_ref}'
        fig.update_layout(title_text=title)
        fig.update_xaxes(title_text="Position x [m]", row=1, col=1)
        fig.update_yaxes(title_text="Position z [m]", row=1, col=1)
        fig.show()


class RefExperimentCharge():

    def __init__(self, m=[2e-12, 1e-12], q=[3e-9, -4e-9], m_ref=2e-12, v_ref=1,
                 q_ref=-1e-12, d=.1, N=100, alpha=[0, 0], dt=.1, is_golf_game=True, **kwargs):
        self.m = np.array(m)
        self.q = np.array(q)
        self.dt = dt
        self.m_ref = m_ref
        self.v_ref = v_ref
        self.q_ref = np.array([q_ref]*2)
        self.x_ref = d
        self.y_ref = 0
        self.alpha = np.array(alpha)
        self.N = N
        self.set_initial_state()
        if is_golf_game:
            self.q_ref = np.flip(q)

    def set_initial_state(self):
        self.t = 0
        self.m = np.array(self.m)
        self.x = np.zeros(2)
        self.y = np.zeros(2)
        self.v_x = np.zeros(2)
        self.v_y = np.zeros(2)
        self.a_x = np.zeros_like(self.m)
        self.a_y = np.zeros_like(self.m)
        self.last_a_x = np.zeros_like(self.m)
        self.last_a_y = np.zeros_like(self.m)
        self.x_series = np.empty((self.N, len(self.m)))
        self.y_series = np.empty_like(self.x_series)
        self.t_series = np.empty((self.N, 1))
        self.v_x_series = np.empty_like(self.x_series)
        self.v_y_series = np.empty_like(self.x_series)
        self.a_x_series = np.empty_like(self.x_series)
        self.a_y_series = np.empty_like(self.x_series)

    def _get_initial_velocity(self):
        v_abs = 2 * self.m_ref / (self.m + self.m_ref) * self.v_ref
        v_x = v_abs * np.cos(self.alpha)
        v_y = v_abs * np.sin(self.alpha)
        return v_x, v_y

    def _update_acceleration(self):
        # get coulomb forces
        dist = np.sqrt((self.x - self.x_ref)**2 + (self.y - self.y_ref)**2)

        F_c = K_E * self.q * self.q_ref / (dist**2)

        direc = ((self.x - self.x_ref), (self.y - self.y_ref)) / dist
        F_c_x = direc[0, :] * F_c
        F_c_y = direc[1, :] * F_c

        # update acceleration for each particle
        self.a_x = F_c_x / self.m
        self.a_y = F_c_y / self.m

    def _update_velocity(self):
        # velocity after elastic collision
        if self.t == 0:
            self.v_x, self.v_y = self._get_initial_velocity()
        else:
            self.v_x = self.v_x + .5 * (self.last_a_x + self.a_x) * self.dt
            self.v_y = self.v_y + .5 * (self.last_a_y + self.a_y) * self.dt

    def _update_position(self):
        # update position
        self.x += self.v_x * self.dt + .5 * self.a_x * self.dt**2
        self.y += self.v_y * self.dt + .5 * self.a_y * self.dt**2
        # update time
        self.t += self.dt

        # for i in range(len(self.m)):
            # if self.y[i] < 0:  # this should not happen, propably the ref. par.
                # self.x[i] = self.x_ref
                # self.y[i] = self.y_ref
                # logger.warning(f'Particle {i} hit the reference particle,'
                               # ' this is not a valid example')

    def run(self):
        for i in range(self.N):
            self.x_series[i, :] = self.x
            self.y_series[i, :] = self.y
            self.v_x_series[i, :] = self.v_x
            self.v_y_series[i, :] = self.v_y
            self.a_x_series[i, :] = self.a_x
            self.a_y_series[i, :] = self.a_y
            self.t_series[i, :] = self.t
            self._update_acceleration()
            self._update_velocity()
            self._update_position()
            self.last_a_x = self.a_x
            self.last_a_y = self.a_y
            self.t += self.dt

    def visualize(self, golf_hole_loc):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        fig = make_subplots(rows=1, cols=1)
        trace_pp1 = go.Scatter(x=self.x_series[:, 0], y=self.y_series[:, 0],
                               name='Particle 1', mode='lines+markers',
                               opacity=0.5)
        trace_pp2 = go.Scatter(x=self.x_series[:, 1], y=self.y_series[:, 1],
                               name='Particle 2', mode='lines+markers',
                               opacity=0.5)
        trace_p_ref = go.Scatter(
            x=[self.x_ref - .001 * self.x_ref,
                self.x_ref + .001 * self.x_ref],
            y=[0, 0],
            name='Reference particle')
        trace_golf_hole = go.Scatter(x=[0, 0],
                                     y=[golf_hole_loc - .01 * golf_hole_loc,
                                        golf_hole_loc + .01 * golf_hole_loc],
                                     name='Golf hole')
        fig.add_trace(trace_pp1, row=1, col=1)
        fig.add_trace(trace_pp2, row=1, col=1)
        fig.add_trace(trace_p_ref, row=1, col=1)
        fig.add_trace(trace_golf_hole, row=1, col=1)
        title = f'RE2: m_ref = {self.m_ref} kg, m = {self.m} kg'\
            f' alpha = [{round(self.alpha[0]/np.pi, 2)} '\
            f'{round(self.alpha[1]/np.pi, 2)}] pi, v_ref = {self.v_ref}'\
            f', q = {self.q}'
        fig.update_layout(title_text=title)
        fig.update_xaxes(title_text="Position x [m]", row=1, col=1)
        fig.update_yaxes(title_text="Position y [m]", row=1, col=1)
        fig.show()

if __name__ == '__main__':
    DT = .01
    N = 100
    GOLF_HOLE_LOC = 0.1
    rem = RefExperimentMass(N=100, dt=.01)
    rem.run()
    # rem.visualize(golf_hole_loc=GOLF_HOLE_LOC)
    # initial_guess = [0.4999 * np.pi, 0.4999 * np.pi]
    # alpha_star = optimize.minimize(
    # objective_golf_mass, initial_guess, method='Nelder-Mead', args=(rem, GOLF_HOLE_LOC)).x
    # rem_opt = RefExperimentMass(alpha=alpha_star, N=100, dt=.01)
    # rem_opt.run()
    # rem_opt.visualize(golf_hole_loc=GOLF_HOLE_LOC)

    req = RefExperimentCharge(N=100, dt=.0001, d=1,
                              alpha=[.5 * np.pi, .75 * np.pi],
                              v_ref=100,
                              m_ref=5e-10)
    req.run()
    req.visualize(1)
