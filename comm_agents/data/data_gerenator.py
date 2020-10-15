from comm_agents.data.datahandler import DataSet
from comm_agents.data.reference_experiments import RefExperimentCharge, RefExperimentMass
from comm_agents.data.optimal_answers import get_alpha_star, get_phi_star
import numpy as np
from tqdm import tqdm

SAMPLE_SIZE_OPT = 100
DT_OPT = .001

# TODO: PROBLEM: v_ref wird auch fuer das charge experiment geupdatet, obwohl v_ref in de fall gleich null ist!!!

class DataGenerator():
    def __init__(self, param_dict, sample_size, m_range, q0_range, q1_range,
                 v_ref_range):
        self.param_dict = param_dict
        self.sample_size = sample_size
        self.m_range = m_range
        self.q0_range = q0_range
        self.q1_range = q1_range
        self.v_ref_range = v_ref_range
        np.random.seed(124)

    def _get_random_experimental_setting(self):
        m = np.random.uniform(*self.m_range, 2)
        q0 = np.random.uniform(*self.q0_range, 1)
        q1 = np.random.uniform(*self.q1_range, 1)
        v_ref = np.random.uniform(*self.v_ref_range, 2)
        return m, np.array([q0, q1]).ravel(), v_ref

    def _run_reference_experiments(self, m, q, v_ref):
        self.param_dict.update(m=m, q=q, v_ref_m=v_ref[0], v_ref_c=v_ref[1])
        rem = RefExperimentMass(**self.param_dict)
        req = RefExperimentCharge(**self.param_dict)
        rem.run()
        req.run()
        return rem.x_series, req.x_series

    def _get_questions(self, v_ref):
        qa0 = (self.param_dict.m_ref_m, v_ref[0])
        qa1 = (self.param_dict.m_ref_c, v_ref[1])
        return qa0, qa1

    def _get_optimal_answers(self):
        rem_opt = RefExperimentMass(**self.param_dict)
        req_opt = RefExperimentCharge(**self.param_dict)
        rem_opt.N = req_opt.N = SAMPLE_SIZE_OPT
        rem_opt.dt = req_opt.N = DT_OPT
        alpha_star, loss0 = get_alpha_star(rem_opt)
        phi_star, loss1 = get_alpha_star(rem_opt)
        rem_opt.set_initial_state()
        rem_opt.angle = alpha_star
        rem_opt.run()
        req_opt.set_initial_state()
        req_opt.angle = phi_star
        req_opt.run()
        return (alpha_star, loss0, rem_opt.check_for_hole_in_one(),phi_star,
                loss1, req_opt.check_for_hole_in_one())

    def _update_data_set():
        pass

    def generate(self):
        for _ in tqdm(range(self.sample_size)):
            m, q, v_ref = self._get_random_experimental_setting()
            o0, o1 = self._run_reference_experiments(m, q, v_ref)
            qa, qb = self._get_questions(v_ref)
            a0, loss0, hio0, a1, loss1, hio1 = self._get_optimal_answers(m, q)
            self._update_data_set(m, q, o0, o1, qa, qb, a0, a1)

if __name__ == '__main__':
    GOLF_HOLE_LOC_M = .1
    GOLF_HOLE_LOC_C = .1
    TOLERANCE = .1
    PARAM_DICT = dict(
        m=[1e-20, 1e-20],
        q=[1e-16, -1e-15],
        m_ref_m=2e-20,
        q_ref=[-1e-17, -1e-17],
        v_ref_m=2,
        m_ref_c=2e-20,
        v_ref_c=0,
        N=10,
        alpha=[0, 0],
        phi=[0, 0],
        dt=.01,
        d=.1,
        is_golf_game=False,
        gravity=True)
    M_RANGES = [1e-20, 5e-20]
    Q0_RANGE = [1e-16, 2e-16]
    Q1_RANGE = [-1e-15, -2e-15]
    V_REF_RANGE = [1, 2]
    SAMPLE_SIZE = 10

    breakpoint()
    dg = DataGenerator(PARAM_DICT, SAMPLE_SIZE, M_RANGES, Q0_RANGE, Q1_RANGE,
            V_REF_RANGE)
    dg.generate()
