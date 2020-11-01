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
    """This class generates the training data set using the reference experiments

    This includes many optimization steps, so we recommend to use a beefy
    machine (many cores) and use multiprocessing (supported by this class,
    see generate method)

    Parameters
    ----------
    param_dict : dict
        Dictionary holding the default parameters for the reference experiments
    sample_size : int
        Number of samples to generate
    sample_size_opt : int
        Sample size for each optimization experiment (to find opt answer)
    golf_hole_loc_m : float
        position of the golf hole of the mass experiment in x direction
    golf_hole_loc_c : float
        position of the golf hole of the charge experiment in y direction
    tolerance : float
        Fraction of golf_hole_loc by which the particle can miss the golf
        (still being a success)
    dt_opt : float
        Time delta for optimization experiments.
        The smaller, the more accurate the optimal answer
    m_range : list of two floats
        Min and max value for the uniform random generation of
        experimental settings (mass)
    v_ref_range : list of two floats
        Min and max value for the uniform random generation of
        experimental settings (velocity of reference particles for questions)
    q0_t_q1_range : list of two floats
        Min and max value for the uniform random generation of
        experimental settings (product of the changes of both particles)
    batch_size : int
        batch_size of the samples are run in parallel, so that a progress
        bar can be displayed
    seed : int
        For reproducibility
    """
    def __init__(self, param_dict, sample_size, m_range, q0_t_q1_range,
                 v_ref_range, seed, batch_size, sample_size_opt,
                 golf_hole_loc_m, golf_hole_loc_c, tolerance, dt_opt):
        self.param_dict = param_dict
        self.sample_size = sample_size
        self.sample_size_opt = sample_size_opt
        self.golf_hole_loc_m = golf_hole_loc_m
        self.golf_hole_loc_c = golf_hole_loc_c
        self.tolerance = tolerance
        self.dt_opt = dt_opt
        self.m_range = m_range
        self.v_ref_range = v_ref_range
        self.q0_t_q1_range = q0_t_q1_range
        self.batch_size = batch_size
        np.random.seed(seed)

    def _get_random_experimental_setting(self):
        """Samples masses, charges and velocities for the experiments

        Returns
        -------
        m : np.array
        q : mp.array
        v_ref_a : float
        v_ref_b : float
        """
        m = np.random.uniform(*self.m_range, 2)
        q0_t_q1 = np.random.uniform(*self.q0_t_q1_range, 1)
        c = np.random.uniform(.5, 1.5, 1)
        q0 = np.sqrt(q0_t_q1 / c)
        q1 = -q0 * c
        v_ref_a, v_ref_b = np.random.uniform(*self.v_ref_range, 2)
        return m, np.array([q0, q1]).ravel(), v_ref_a, v_ref_b

    def _run_reference_experiments(self, m, q):
        """Runs both reference experiment to generate observations

        Note that we do NOT use the sampled reference velocities for the
        observations but constant values for all samples.
        The encoder is not supposed to encode the velocity of the particle
        but only its mass

        Returns
        -------
        ref_a_obs : np.array (shpae 10x2)
        ref_b_obs : np.array (shpae 10x2)
        """
        self.param_dict.update(m=m, q=q)
        rem = RefExperimentMass(**self.param_dict)
        req = RefExperimentCharge(**self.param_dict)
        rem.run()
        req.run()
        return rem.x_series, req.x_series

    def _get_questions(self, v_ref_a, v_ref_b):
        """Returns questions (including the mass and the velocity of the ref particle)

        Returns
        -------
        q_a : np.array
        q_b : np.array
        """
        q_a = (self.param_dict['m_ref_m'], v_ref_a)
        q_b = (self.param_dict['m_ref_c'], v_ref_b)
        return q_a, q_b

    def _get_optimal_answers(self, v_ref_a, v_ref_b):
        """Computes the optimal answers for both experiments and both particles

        Note that this method does not require m and q, since those were
        updated in the param dict in method _run_reference_experiments.

        Parameters
        ----------
        v_ref_a : float
            Velocity of reference particle for the mass experiment
        v_ref_b : float
            Velocity of reference particle for the charge experiment

        Returns
        -------
        a_a : np.array (two floats)
            Optimal angle reference experiment a for both particles
        loss_a : np.array (two floats)
            Optimization loss reference experiment a for both particles
        hio_a : np.array (two bools)
            True if particle one performed a hole in one
        a_b : np.array (two floats)
            Optimal angle reference experiment a for both particles
        loss_b : np.array (two floats)
            Optimization loss reference experiment a for both particles
        hio_b : np.array (two bools)
            True if particle one performed a hole in one
        """
        rem_opt = RefExperimentMass(**self.param_dict)
        req_opt = RefExperimentCharge(**self.param_dict)
        req_opt.is_golf_game = True
        rem_opt.v_ref = v_ref_a
        req_opt.v_ref = v_ref_b
        rem_opt.N = req_opt.N = self.sample_size_opt
        rem_opt.dt = req_opt.dt = self.dt_opt
        alpha_star, loss0 = get_alpha_star(rem_opt)
        phi_star, loss1 = get_phi_star(req_opt)
        rem_opt.set_initial_state()
        rem_opt.angle = alpha_star
        rem_opt.run()
        req_opt.set_initial_state()
        req_opt.angle = phi_star
        req_opt.run()
        return (alpha_star, loss0, rem_opt.check_for_hole_in_one(
            golf_hole_loc=self.golf_hole_loc_m,
                tolerance=self.tolerance),
                phi_star, loss1, req_opt.check_for_hole_in_one(
            golf_hole_loc=self.golf_hole_loc_c, tolerance=self.tolerance))

    def trans_data_set_to_tabular(self, data):
        """Wraps all the attributes in a pandas df

        Parameters
        ----------
        data : list or tuple
            Holding m, q, o_a, o_b, q_a, q_b, a_a, a_b
            Each of those is described in on of the docstrings above

        Returns
        -------
        pd.DataFrame
            Data frame with each row being on sample. The columns hold the
            hidden states, the observations, the questions and the optimal
            answers in correspondingly named columns
        """
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

    def generate(self, parallel=True, njobs=-1):
        """Main of this class, that generates the dataset as pd data frame

        Parameters
        ----------
        parallel : bool
            If true joblib is used to perform multiprocessing
        njobs : int
            Number of jobs to run in parallel
        """
        def _in_parallel(self):
            try:
                m, q, v_ref_a, v_ref_b = \
                    self._get_random_experimental_setting()
                o_a, o_b = self._run_reference_experiments(m, q)
                q_a, q_b = self._get_questions(v_ref_a, v_ref_b)
                a_a, loss_a, hio_a, a_b, loss_b, hio_b = \
                    self._get_optimal_answers(v_ref_a, v_ref_b)
                if all([lo < self.tolerance for lo in [loss_a, loss_b]]) \
                        and all([hio_a[0], hio_a[1], hio_b[0], hio_b[1]]):
                    return m, q, o_a, o_b, q_a, q_b, a_a, a_b
            except TypeError:
                logger.debug(f'Optimization failed for combination'
                             f' m: {m}, q: {q}, v_ref: {v_ref_a}, {v_ref_b}')
        if parallel:
            data = []
            logger.debug(f'Started sampling reference experiments in parallel'
                         f' in {njobs} processes')
            for _ in tqdm(range(int(self.sample_size/self.batch_size))):
                d = Parallel(n_jobs=njobs)(delayed(_in_parallel)(self)
                                           for _ in range(self.batch_size))
                data.extend(d)
        else:
            logger.debug(
                'Started sampling reference experiments in'
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
        q0_t_q1_range=dg_params['Q0_T_Q1_RANGE'],
        v_ref_range=dg_params['V_REF_RANGE'],
        seed=dg_params['SEED'],
        batch_size=dg_params['BATCH_SIZE'],
        sample_size_opt=dg_params['SAMPLE_SIZE_OPT'],
        dt_opt=dg_params['DT_OPT'],
        golf_hole_loc_m=dg_params['GOLF_HOLE_LOC_M'],
        golf_hole_loc_c=dg_params['GOLF_HOLE_LOC_C'],
        tolerance=dg_params['TOLERANCE']
    )

    df = dg.generate(parallel=True, njobs=-1)
    df.to_csv(PATH, index=False)
    logger.debug(f'Successfully wrote data frame to: {PATH}')
