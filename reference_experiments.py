import numpy as np


class RefExperimentMass():

    def __init__(self, m=[1, 2], m_ref=2, v_ref=10, N=10):
        self.m = m
        self.x = np.zeros(2)
        self.v = np.zeros(2)
        self.m_ref = m_ref
        self.v_ref = v_ref
        self.time_delta = 1
        self.t = 0
        self.N = N
        self.x_series = np.empty((N, len(m)))
        self.t_series = np.empty((N, 1))

    def update_velocity(self):
        # velocity after elastic collision
        self.v = np.array(
                [2 * self.m_ref / (m + self.m_ref) * self.v_ref
                    for m in self.m])

    def step(self):
        # update position
        self.x += self.v * self.time_delta
        # update time
        self.t += self.time_delta

    def run(self):
        self.update_velocity()
        for i in range(self.N):
            self.step()
            self.x_series[i, :] = self.x
            self.t_series[i, :] = (self.t)


if __name__ == '__main__':
    breakpoint()
    rem = RefExperimentMass()
    rem.run()














