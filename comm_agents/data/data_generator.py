from comm_agents.data.reference_experiments import (RefExperimentCharge,
                                                    RefExperimentMass)
from comm_agents.data.optimal_answers import (get_alpha_star,
                                              get_phi_star)
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import joblib
from loguru import logger
from tqdm import tqdm

SAMPLE_SIZE_OPT = 1000
DT_OPT = .001
GOLF_HOLE_LOC_M = .1
GOLF_HOLE_LOC_C = .1
TOLERANCE = .01
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
        np.random.seed(10000)

    def _get_random_experimental_setting(self):
        m = np.random.uniform(*self.m_range, 2)
        q0 = np.random.uniform(*self.q0_range, 1)
        q1 = np.random.uniform(*self.q1_range, 1)
        v_ref_a, v_ref_b = np.random.uniform(*self.v_ref_range, 2)
        return m, np.array([q0, q1]).ravel(), v_ref_a, v_ref_b

    def _run_reference_experiments(self, m, q, v_ref_a, v_ref_b):
        self.param_dict.update(m=m, q=q, v_ref_m=v_ref_a, v_ref_c=v_ref_b)
        rem = RefExperimentMass(**self.param_dict)
        req = RefExperimentCharge(**self.param_dict)
        rem.run()
        req.run()
        return rem.x_series, req.x_series

    def _get_questions(self, v_ref_a, v_ref_b):
        qa0 = (self.param_dict['m_ref_m'], v_ref_a)
        qa1 = (self.param_dict['m_ref_c'], v_ref_b)
        return qa0, qa1

    def _get_optimal_answers(self, m, q, v_ref_a, v_ref_b):
        rem_opt = RefExperimentMass(**self.param_dict)
        req_opt = RefExperimentCharge(**self.param_dict)
        req_opt.is_golf_game = True
        req_opt.v_ref = v_ref_b
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

    def trans_data_set_to_tabular(self, data):
        # m, q, o_a, o_b, q_a, q_b, a_a, a_b
        data = [d for d in data if d]
        df_m = pd.DataFrame(
            [x.ravel() for x in [d[0] for d in data]],
            columns=['m0', 'm1'])
        df_q = pd.DataFrame(
            [x.ravel() for x in [d[1] for d in data]],
            columns=['q0', 'q1'])
        df_o_a_0 = pd.DataFrame(
            [o[:, 0] for o in [d[2] for d in data]],
            columns=[f'o_a_0_{i}' for i in range(self.param_dict['N'])])
        df_o_a_1 = pd.DataFrame(
            [o[:, 1] for o in [d[2] for d in data]],
            columns=[f'o_a_1_{i}' for i in range(self.param_dict['N'])])
        df_o_b_0 = pd.DataFrame(
            [o[:, 0] for o in [d[3] for d in data]],
            columns=[f'o_b_0_{i}' for i in range(self.param_dict['N'])])
        df_o_b_1 = pd.DataFrame(
            [o[:, 1] for o in [d[3] for d in data]],
            columns=[f'o_b_1_{i}' for i in range(self.param_dict['N'])])
        df_q_a = pd.DataFrame(
            [d[4] for d in data], columns=['m_ref_a', 'v_ref_a'])
        df_q_b = pd.DataFrame(
            [d[5] for d in data], columns=['m_ref_b', 'v_ref_b'])
        df_oa_a = pd.DataFrame(
            [d[6] for d in data],
            columns=['alpha_star0', 'alpha_star1'])
        df_oa_b = pd.DataFrame(
            [d[7] for d in data], columns=['phi_star0', 'phi_star1'])
        return pd.concat([df_m, df_q, df_o_a_0, df_o_a_1, df_o_b_0, df_o_b_1,
                          df_q_a, df_q_b, df_oa_a, df_oa_b], axis=1)

    def generate(self, parallel=True, njobs=6):
        def _in_parallel(self):
            try:
                m, q, v_ref_a, v_ref_b = self._get_random_experimental_setting()
                o_a, o_b = self._run_reference_experiments(
                    m, q, v_ref_a, v_ref_b)
                q_a, q_b = self._get_questions(v_ref_a, v_ref_b)
                a_a, loss_a, hio_a, a_b, loss_b, hio_b = self._get_optimal_answers(
                    m, q, v_ref_a, v_ref_b)
                if all([lo < TOLERANCE for lo in [loss_a, loss_b]]) \
                        and all([hio_a[0], hio_a[1], hio_b[0], hio_b[1]]):
                    return m, q, o_a, o_b, q_a, q_b, a_a, a_b
            except TypeError:
                logger.debug(f'Optimization failed for combination'
                             f' m: {m}, q: {q}, v_ref: {v_ref_a}, {v_ref_b}')
        if parallel:
            data = []
            logger.debug(f'Started sampling reference experiments in parallel in'
                         f' {njobs} processes')
            for _ in tqdm(range(int(self.sample_size/BATCH_SIZE))):
                d = Parallel(n_jobs=njobs)(delayed(_in_parallel)(self)
                                           for _ in range(BATCH_SIZE))
                data.extend(d)
        else:
            logger.debug('Started sampling reference experiments in parallel in'
                         ' 1 processes')
            data = [_in_parallel(self) for _ in tqdm(range(self.sample_size))]
        breakpoint()
        data = [d for d in data if d]
        logger.debug(f'Successfully computed {len(data)} of {self.sample_size}'
                     f' experimental settings')
        df = self.trans_data_set_to_tabular(data)
        return df


if __name__ == '__main__':
    SAMPLE_SIZE = 10000 
    BATCH_SIZE = 36
    PATH = 'data/reference_experiment_dat_1000_new_new_new4.csv'
    dg = DataGenerator(PARAM_DICT, SAMPLE_SIZE, M_RANGES, Q0_RANGE, Q1_RANGE,
                       V_REF_RANGE)

    df = dg.generate(parallel=True, njobs=6)
    df.to_csv(PATH, index=False)
    logger.debug(f'Successfully wrote data frame to: {PATH}')
