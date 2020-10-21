from comm_agents.data.reference_experiments import (RefExperimentCharge,
                                                    RefExperimentMass)
from comm_agents.data.optimal_answers import (get_alpha_star,
                                              get_phi_star)
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm
import json


class DataGenerator():
    def __init__(self, param_dict, sample_size, m_range, q0_t_q1_range,
                 v_ref_range, seed, batch_size):
        self.param_dict = param_dict
        self.sample_size = sample_size
        self.m_range = m_range
        self.v_ref_range = v_ref_range
        self.q0_t_q1_range = q0_t_q1_range
        self.batch_size = batch_size
        self.data_set = dict(
            observations_0=[],
            observations_1=[],
            questions_a=[],
            questions_b=[],
            opt_answers_a=[],
            opt_answers_b=[],
            hidden_states=[]
        )
        np.random.seed(seed)

    def _get_random_experimental_setting(self):
        m = np.random.uniform(*self.m_range, 2)
        q0_t_q1 = np.random.uniform(*self.q0_t_q1_range, 1)
        c = np.random.uniform(.5, 1.5, 1)
        q0 = np.sqrt(q0_t_q1 / c)
        q1 = -q0 * c
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
            for _ in tqdm(range(int(self.sample_size/self.batch_size))):
                d = Parallel(n_jobs=njobs)(delayed(_in_parallel)(self)
                                           for _ in range(self.batch_size))
                data.extend(d)
        else:
            logger.debug('Started sampling reference experiments in parallel in'
                         ' 1 processes')
            data = [_in_parallel(self) for _ in tqdm(range(self.sample_size))]
        data = [d for d in data if d]
        logger.debug(f'Successfully computed {len(data)} of {self.sample_size}'
                     f' experimental settings')
        df = self.trans_data_set_to_tabular(data)
        return df


if __name__ == '__main__':

    with open('config.json') as config_file:
        conf_dct = json.load(config_file)
    default_params = conf_dct['refExperimentDefaultParams']
    dg_params = conf_dct['dataGeneration']
    PATH = 'data/test.csv'
    dg = DataGenerator(
        param_dict=conf_dct['refExperimentDefaultParams'],
        sample_size=dg_params['SAMPLE_SIZE'],
        m_range=dg_params['M_RANGES'],
        q0_t_q1_range=['Q0_T_Q1_RANGE'],
        v_ref_range=['V_REF_RANGE'],
        seed=dg_params['SEED'],
        batch_size=dg_params['BATCH_SIZE'])

    df = dg.generate(parallel=False, njobs=-1)
    df.to_csv(PATH, index=False)
    logger.debug(f'Successfully wrote data frame to: {PATH}')
