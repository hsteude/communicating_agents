import numpy as np
from loguru import logger
from scipy import optimize
import json


# load config dict and define some constants
with open('config.json') as config_file:
    conf_dct = json.load(config_file)
golf_hole_loc_m = conf_dct['dataGeneration']['GOLF_HOLE_LOC_M']
golf_hole_loc_c = conf_dct['dataGeneration']['GOLF_HOLE_LOC_C']
INITIAL_GUESS_M = [0.249 * np.pi, 0.249 * np.pi]
INITIAL_GUESS_C = [0.501 * np.pi, 0.501 * np.pi]
ALPHA_BOUNDS = [(0.01, .2499 * np.pi)]*2
PHI_BOUNDS = [(0.501 * np.pi, .7499 * np.pi)]*2


def get_single_mass_loss_value(exp, i, golf_hole_loc):
    """
    Computes the loss of a single golf mass game (two particle)

    Parameters
    ----------
    exp : RefExperimentMass
        An instance of the mass reference experiment
    i : int
        Index of the particle
    golf_hole_loc : float
        Position of the golf hole in x direction

    Returns
    -------
    float
        Loss
    """

    distances = \
        [np.linalg.norm(np.array([x, z]) - np.array([golf_hole_loc, 0]))
         for x, z in zip(exp.x_series[:, i], exp.z_series[:, i])]
    value = min(distances)
    return value


def get_single_charge_loss_value(exp, i, golf_hole_loc):
    """
    Computes the loss of a single golf charge game (two particle)

    Parameters
    ----------
    exp : RefExperimentCharge
        An instance of the charge reference experiment
    i : int
        Index of the particle
    golf_hole_loc : float
        Position of the golf hole in x direction

    Returns
    -------
    float
        Loss
    """
    distances = \
        [np.linalg.norm(np.array([x, y]) - np.array([0, golf_hole_loc]))
         for x, y in zip(exp.x_series[:, i], exp.y_series[:, i])]
    value = min(distances)
    return value


def golf_loss(angle, exp, objective, golf_hole_loc):
    """
    Computes the loss of an arbitrary ref experiment given its objective

    Parameters
    ----------
    angle :  list of floats
        Angle of the reference particles velocity
    exp : RefExperimentCharge or RefExperimentMass
        Instance of the reference experiment
    objective : function
        Objective function to compute loss
    golf_hole_loc : float
        Position of the golf hole in x direction
    """
    exp.set_initial_state()
    exp.angle = angle
    exp.run()
    return np.mean([objective(i=i, exp=exp, golf_hole_loc=golf_hole_loc)
                    for i in [0, 1]])


def get_alpha_star(exp):
    """
    Optimizes the answer / angle for the mass experiment,
    using scipy optimize

    Parameters
    ----------
    exp : RefExperimentMass

    Returns:
    --------
    alpha_star : list of two floats
    opt_val : loss of optimization
    """
    alpha_star = optimize.minimize(
        golf_loss,
        INITIAL_GUESS_M,
        args=(exp, get_single_mass_loss_value, golf_hole_loc_m),
        bounds=ALPHA_BOUNDS)
    if alpha_star.success:
        opt_val = golf_loss(
            alpha_star.x, exp, get_single_mass_loss_value, golf_hole_loc_m)
        return alpha_star.x, opt_val
    else:
        logger.warning('Alpha optimization did not succeed')


def get_phi_star(exp):
    """
    Optimizes the answer / angle for the charge experiment,
    using scipy optimize

    Parameters
    ----------
    exp : RefExperimentCharge

    Returns:
    --------
    alpha_star : list of two floats
    opt_val : loss of optimization
    """
    phi_star = optimize.minimize(
        golf_loss,
        INITIAL_GUESS_C,
        args=(exp, get_single_charge_loss_value, golf_hole_loc_c),
        bounds=PHI_BOUNDS)
    if phi_star.success:
        opt_val = golf_loss(
            phi_star.x, exp, get_single_charge_loss_value, golf_hole_loc_c)
        return phi_star.x, opt_val
    else:
        logger.warning('Alpha optimization did not succeed')
