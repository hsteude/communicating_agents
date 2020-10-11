import numpy as np
import matplotlib.pyplot as plt
from loguru import logger


K_E = 8.99e9  # Coulomb constant


class RefExperimentMass():

    def __init__(self, m=[1, 2], m_ref=2, v_ref=10, N=10):
        self.m = np.array(m)
        self.x = np.zeros(2)
        self.v = np.zeros(2)
        self.m_ref = m_ref
        self.v_ref = v_ref
        self.dt = 1
        self.t = 0
        self.N = N
        self.x_series = np.empty((N, len(m)))
        self.t_series = np.empty((N, 1))
        self.v_series = np.empty_like(self.x_series)

    def _update_velocity(self):
        # velocity after elastic collision
        self.v = np.array(
            [2 * self.m_ref / (m + self.m_ref) * self.v_ref
             for m in self.m])

    def _update_position(self):
        # update position
        self.x += self.v * self.dt
        # update time
        self.t += self.dt

    def run(self):
        for i in range(self.N):
            self._update_velocity()
            self._update_position()
            self.t += self.dt
            self.x_series[i, :] = self.x
            self.v_series[i, :] = self.v
            self.t_series[i, :] = (self.t)

    def plot(self, i):
        import plotly.express as px
        fig = px.line(
            x=self.t_series[:, 0], y=self.x_series[:, i],
            labels={'x': 't [s]', 'y': 'x_position [m]'},
            title=f'Ref. exp. masses: m_ref = {self.m_ref}, m = {self.m[i]}')
        fig.show()


class RefExperimentCharge():

    def __init__(self, m=[2e-12, 1e-12], q=[3e-9, 4e-9], m_ref=1,
                 q_ref=-1e-9, d=.1, N=10):
        self.q = np.array(q)
        self.m = np.array(m)
        self.x = np.zeros_like(self.m)
        self.v = np.zeros_like(self.m)
        self.a = np.zeros_like(self.m)
        self.last_a = np.zeros_like(m)
        self.m_ref = m_ref
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
            x=self.t_series[:, 0], y=self.x_series[:, i],
            labels={'x': 't [s]', 'y': 'x_position [m]'},
            title=f'Ref. exp charge: m_ref = {self.m_ref}, m = {self.m[i]},'
            f' q_ref = {self.q_ref} q = {self.q[i]}, d0 = {self.x_ref}')
        fig.show()


if __name__ == '__main__':
    rem = RefExperimentMass()
    rem.run()

    req = RefExperimentCharge(N=100)
    req.run()
