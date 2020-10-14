import numpy as np
from loguru import logger
from scipy import optimize


GOLF_HOLE_LOC_M = .1
GOLF_HOLE_LOC_C = .1
PENALTY_VALUE = 100
INITIAL_GUESS_M = [0.01 * np.pi, 0.01 * np.pi]
INITIAL_GUESS_C = [0.501 * np.pi, 0.501 * np.pi]
THRES = .01
ALPHA_BOUNDS = [(0.01, .5 * np.pi)]*2
PHI_BOUNDS = [(0.501 * np.pi, .999 * np.pi)]*2


def get_single_mass_loss_value(exp, i, golf_hole_loc):
    distances = [np.linalg.norm(np.array([x, z]) - np.array([golf_hole_loc, 0]))
                 for x, z in zip(exp.x_series[:, i], exp.z_series[:, i])]
    value = min(distances)
    return value


def get_single_charge_loss_value(exp, i, golf_hole_loc):
    distances = [np.linalg.norm(np.array([x, y]) - np.array([0, golf_hole_loc]))
                 for x, y in zip(exp.x_series[:, i], exp.y_series[:, i])]
    value = min(distances)
    return value


def golf_loss(angle, exp, objective, golf_hole_loc):
    exp.set_initial_state()
    exp.angle = angle
    exp.run()
    return np.mean([objective(i=i, exp=exp, golf_hole_loc=golf_hole_loc)
                   for i in [0, 1]])


def get_alpha_star(exp):
    alpha_star = optimize.minimize(
        golf_loss,
        INITIAL_GUESS_M,
        args=(exp, get_single_mass_loss_value, GOLF_HOLE_LOC_M),
        bounds=ALPHA_BOUNDS)
    if alpha_star.success:
        opt_val = golf_loss(
            alpha_star.x, exp, get_single_mass_loss_value, GOLF_HOLE_LOC_M)
        return alpha_star.x, opt_val
    else:
        logger.warning('Alpha optimization did not succeed')


def get_phi_star(exp):
    phi_star = optimize.minimize(
        golf_loss,
        INITIAL_GUESS_C,
        args=(exp, get_single_charge_loss_value, GOLF_HOLE_LOC_C),
        bounds=PHI_BOUNDS)
    if phi_star.success:
        opt_val = golf_loss(
            phi_star.x, exp, get_single_charge_loss_value, GOLF_HOLE_LOC_C)
        return phi_star.x, opt_val
    else:
        logger.warning('Alpha optimization did not succeed')
