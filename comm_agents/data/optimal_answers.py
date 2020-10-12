import numpy as np
from loguru import logger
from scipy import optimize
from comm_agents.data.reference_experiments_with_qa import \
    RefExperimentMass


GOLF_HOLE_LOC_M = .1
GOLF_HOLE_LOC_C = .1
PENALTY_VALUE = 100
INITIAL_GUESS_M = [0.4999 * np.pi, 0.4999 * np.pi]
INITIAL_GUESS_C = [0.5001 * np.pi, 0.5001 * np.pi]
ALPHA_THRES = .01


def get_single_mass_loss_value(exp, i, golf_hole_loc):
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


def get_single_charge_loss_value(exp, i, golf_hole_loc):
    zero_cross = np.where(np.diff(np.sign(exp.x_series[1:, i])))[0]
    if len(zero_cross) != 1:
        logger.warning(f'Particle {i} did not cross line with x=0')
        value = PENALTY_VALUE
    else:
        y_at_zero_cross = (
            exp.y_series[zero_cross + 1][0][i]
            + exp.y_series[zero_cross + 2][0][i]) / 2
        value = np.abs(y_at_zero_cross - golf_hole_loc)
    return value


def golf_loss(alpha, exp, objective, golf_hole_loc):
    exp.set_initial_state()
    exp.alpha = alpha
    exp.run()
    return np.sum([objective(i=i, exp=exp, golf_hole_loc=golf_hole_loc)
                   for i in [0, 1]])


def get_alpha_star(exp):
    golf_hole_loc = GOLF_HOLE_LOC_M if isinstance(exp, RefExperimentMass)\
        else GOLF_HOLE_LOC_C
    initial_guess = INITIAL_GUESS_M if isinstance(exp, RefExperimentMass)\
        else INITIAL_GUESS_C
    loss_func = get_single_mass_loss_value if isinstance(
        exp, RefExperimentMass)\
        else get_single_charge_loss_value
    alpha_star = optimize.minimize(
        golf_loss,
        initial_guess,
        method='Nelder-Mead',
        args=(exp, loss_func, golf_hole_loc))
    opt_val = golf_loss(alpha_star.x, exp, loss_func, golf_hole_loc)
    if opt_val < ALPHA_THRES:
        print(opt_val)
        return alpha_star.x
    else:
        logger.debug('No optimal answer founnd, returning None')


if __name__ == '__main__':
    pass
