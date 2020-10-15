from comm_agents.data.reference_experiments import (RefExperimentCharge,
                                                    RefExperimentMass)
from comm_agents.data.optimal_answers import (get_alpha_star,
                                              get_phi_star)
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

SAMPLE_SIZE_OPT = 100
DT_OPT = .001
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


class DataGenerator():
    def __init__(self, param_dict, sample_size, m_range, q0_range, q1_range,
                 v_ref_range):
        self.param_dict = param_dict
        self.sample_size = sample_size
        self.m_range = m_range
        self.q0_range = q0_range
        self.q1_range = q1_range
        self.v_ref_range = v_ref_range
        self.data_set = dict(
            observations_0=[],
            observations_1=[],
            questions_a=[],
            questions_b=[],
            opt_answers_a=[],
            opt_answers_b=[],
            hidden_states=[]
        )
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
        qa0 = (self.param_dict['m_ref_m'], v_ref[0])
        qa1 = (self.param_dict['m_ref_c'], v_ref[1])
        return qa0, qa1

    def _get_optimal_answers(self, m, q, v_ref):
        rem_opt = RefExperimentMass(**self.param_dict)
        req_opt = RefExperimentCharge(**self.param_dict)
        req_opt.is_golf_game = True
        req_opt.v_ref = v_ref[0]
        rem_opt.N = req_opt.N = SAMPLE_SIZE_OPT
        rem_opt.dt = req_opt.dt = DT_OPT
        alpha_star, loss0 = get_alpha_star(rem_opt)
        phi_star, loss1 = get_phi_star(req_opt)
        rem_opt.set_initial_state()
        rem_opt.angle = alpha_star
        rem_opt.run()
        req_opt.set_initial_state()
        req_opt.angle = phi_star
        req_opt.run()
        return (alpha_star, loss0, rem_opt.check_for_hole_in_one(
            golf_hole_loc=GOLF_HOLE_LOC_M, tolerance=TOLERANCE),
            phi_star, loss1, req_opt.check_for_hole_in_one(
            golf_hole_loc=GOLF_HOLE_LOC_C, tolerance=TOLERANCE))

    def _update_data_set(self, m, q, o0, o1, qa, qb, a0, a1):
        self.data_set['observations_0'].append(o0)
        self.data_set['observations_1'].append(o1)
        self.data_set['questions_a'].append(qa)
        self.data_set['questions_b'].append(qb)
        self.data_set['opt_answers_a'].append(a0)
        self.data_set['opt_answers_b'].append(a1)
        self.data_set['hidden_states'].append(np.array([m, q]))

    def trans_data_set_to_tabular(self):
        df_hs = pd.DataFrame(
            [x.ravel() for x in self.data_set['hidden_states']],
            columns=['m0', 'm1', 'q0', 'q1'])
        df_o0_0 = pd.DataFrame(
            [o[:, 0] for o in dg.data_set['observations_0']],
            columns=[f'o0_0_{i}' for i in range(self.param_dict['N'])])
        df_o0_1 = pd.DataFrame(
            [o[:, 1] for o in dg.data_set['observations_0']],
            columns=[f'o0_1_{i}' for i in range(self.param_dict['N'])])
        df_o1_0 = pd.DataFrame(
            [o[:, 0] for o in dg.data_set['observations_1']],
            columns=[f'o1_0_{i}' for i in range(self.param_dict['N'])])
        df_o1_1 = pd.DataFrame(
            [o[:, 1] for o in dg.data_set['observations_1']],
            columns=[f'o1_1_{i}' for i in range(self.param_dict['N'])])
        df_qa = pd.DataFrame(
            dg.data_set['questions_a'], columns=['m_ref_a', 'v_ref_a'])
        df_qb = pd.DataFrame(
            dg.data_set['questions_a'], columns=['m_ref_b', 'v_ref_b'])
        df_oa_a = pd.DataFrame(
            dg.data_set['opt_answers_a'],
            columns=['alpha_star0', 'alpha_star1'])
        df_oa_b = pd.DataFrame(
            dg.data_set['opt_answers_b'], columns=['phi_star0', 'phi_star1'])
        return pd.concat([df_hs, df_o0_0, df_o0_1, df_o1_0, df_o1_1, df_oa_a,
                          df_oa_b, df_qa, df_qb], axis=1)

    def generate(self):
        for _ in tqdm(range(self.sample_size)):
            m, q, v_ref = self._get_random_experimental_setting()
            o0, o1 = self._run_reference_experiments(m, q, v_ref)
            qa, qb = self._get_questions(v_ref)
            a0, loss0, hio0, a1, loss1, hio1 = self._get_optimal_answers(
                m, q, v_ref)
            if all([lo < TOLERANCE for lo in [loss0, loss1]]) \
                    and all([hio0[0], hio0[1], hio1[0], hio1[1]]):
                self._update_data_set(m, q, o0, o1, qa, qb, a0, a1)
        df = self.trans_data_set_to_tabular()
        return df


if __name__ == '__main__':
    SAMPLE_SIZE = 1000
    PATH = 'data/reference_experiment_dat.csv'
    dg = DataGenerator(PARAM_DICT, SAMPLE_SIZE, M_RANGES, Q0_RANGE, Q1_RANGE,
                       V_REF_RANGE)

    df = dg.generate()
    df.to_csv(PATH, index=False)
