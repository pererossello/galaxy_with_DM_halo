import os

import numpy as np
import h5py
from numba import jit, prange, float32, int32, void

import main.plot_utils as pu


def read_data(path):

    data = np.genfromtxt(path, delimiter=" ")

    MASS = data[:, 0]
    POS = data[:, 1:4]
    VEL = data[:, 4:]

    return POS, VEL, MASS


def generate_ics(savepath, POS, VEL, MASS):

    savefold = "/".join(savepath.split("/")[:-1])
    if not os.path.exists(savefold):
        os.makedirs(savefold)

    N = POS.shape[0]
    DIM = POS.shape[1]

    with h5py.File(savepath, "w") as f:
        header_grp = f.create_group("Header")
        header_grp.attrs["Dimensions"] = np.int32(DIM)
        header_grp.attrs["N"] = np.int32(N)

        part_type_grp = f.create_group("Bodies")
        part_type_grp.create_dataset("Positions", data=np.array(POS, dtype=np.float32))
        part_type_grp.create_dataset("Velocities", data=np.array(VEL, dtype=np.float32))
        part_type_grp.create_dataset("Masses", data=np.array(MASS, dtype=np.float32))






def galaxy_ics(
    savepath,
    N,
    R_d,
    R_s=20.0,
    rho_0=5932371.0,
    G=4.302e-6,
    pos_std=1,
    vel_std=0.1,
    total_mass=5e10,
    e=1e-3
):
    
    """
    Generates simple initial conditions for a galaxy with an exponential disk and NFW halo. Saves the data to an HDF5 file.

    :savepath: str - path to save the hdf5 file
    :N: int - number of particles
    :R_d: float - disk scale length
    :R_s: float - NFW scale radius
    :rho_0: float - NFW density parameter
    :G: float - gravitational constant
    :pos_std: float - standard deviation of the noise added to the positions
    :vel_std: float - standard deviation of the noise added to the velocities
    :total_mass: float - total mass of the galaxy
    :e: float - softening parameter for gravitational field calculation
    """


    # GENERATE POSITIONS

    MASS = total_mass / N * np.ones(N)

    r_arr = np.random.exponential(scale=R_d, size=N)
    phi_arr = np.random.uniform(low=0, high=2 * np.pi, size=N)

    x_arr = r_arr * np.cos(phi_arr)
    y_arr = r_arr * np.sin(phi_arr)

    x_noise = np.random.normal(loc=0, scale=pos_std, size=N)
    y_noise = np.random.normal(loc=0, scale=pos_std, size=N)
    z_noise = np.random.normal(loc=0, scale=pos_std, size=N)

    x_arr = x_arr + x_noise
    y_arr = y_arr + y_noise
    z_arr = z_noise

    POS = np.column_stack((x_arr, y_arr, z_arr))


    # GENERATE VELOCITIES

    # We are computing the circular velocity taking into account the term due to the NFW potential and the extra term due to the N-body interactions
    e = np.float32(e)
    G = np.float32(G)
    N = np.int32(N)
    POS = np.array(POS, dtype=np.float32)
    MASS = np.array(MASS, dtype=np.float32)
    g_mag = g_magnitude(N, POS, MASS, G, e)
    extra_term = g_mag

    V_arr = V_circ_NFW(r_arr, rho_0, R_s, G, extra_term=extra_term)
    
    # Printing the min and max periods of the circular orbits for reference
    r_max_idx = np.argmax(r_arr)
    r_min_idx = np.argmin(r_arr)

    max_period = 2 * np.pi * np.max(r_arr) / V_arr[r_max_idx]
    min_period = 2 * np.pi * np.min(r_arr) / V_arr[r_min_idx]

    print('Min period:', min_period)
    print('Max period:', max_period)

    # Adding noise
    vel_x_noise = np.random.normal(loc=0, scale=vel_std, size=N)
    vel_y_noise = np.random.normal(loc=0, scale=vel_std, size=N)
    vel_z_noise = np.random.normal(loc=0, scale=vel_std, size=N)

    v_x = V_arr * np.sin(phi_arr) + vel_x_noise
    v_y = -V_arr * np.cos(phi_arr) + vel_y_noise
    v_z = vel_z_noise

    VEL = np.column_stack((v_x, v_y, v_z))

    savefold = "/".join(savepath.split("/")[:-1])
    if not os.path.exists(savefold):
        os.makedirs(savefold)

    DIM = 3

    with h5py.File(savepath, "w") as f:
        header_grp = f.create_group("Header")
        header_grp.attrs["Dimensions"] = np.int32(DIM)
        header_grp.attrs["N"] = np.int32(N)

        part_type_grp = f.create_group("Bodies")
        part_type_grp.create_dataset("Positions", data=np.array(POS, dtype=np.float32))
        part_type_grp.create_dataset("Velocities", data=np.array(VEL, dtype=np.float32))
        part_type_grp.create_dataset("Masses", data=np.array(MASS, dtype=np.float32))



"""
Utility functions for galaxy_ics
"""

def dPHI_dr_NFW(r, rho_0, R_s, G):

    """
    Radial derivative of the NFW potential, used to calculate circular velocities.
    """

    num_1 = 4 * np.pi * G * rho_0 * R_s**3
    num_2 = -r + (r + R_s) * np.log(1 + r / R_s)
    den_1 = r**2 * (r + R_s)

    result = num_1 * num_2 / den_1

    return result


def V_circ_NFW(r, rho_0, R_s, G, extra_term=0):
    """
    Circular velocity profile for an NFW halo, with an optional extra term to account for extra interaction terms
    """

    result = np.sqrt(r * dPHI_dr_NFW(r, rho_0, R_s, G) + r * extra_term)
    return result



@jit(
    float32[:](
        int32,
        float32[:, :],
        float32[:],
        float32,
        float32,
    ),
    nopython=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def g_magnitude(N, POS, MASS, G, e):
    """
    Calculates the magnitude of the gravitational field due to N-body interactions
    """
    fields = np.zeros((N, 3), dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    for i in prange(N):
        acc = acc * 0
        for j in prange(N):
            if i == j:
                continue
            dx = POS[i][0] - POS[j][0]
            dy = POS[i][1] - POS[j][1]
            dz = POS[i][2] - POS[j][2]
            r2 = dx * dx + dy * dy + dz * dz
            F_over_mm = -G * np.power(r2 + e * e, -1.5)
            Fx = F_over_mm * dx
            Fy = F_over_mm * dy
            Fz = F_over_mm * dz

            acc[j][0] = Fx * MASS[j]
            acc[j][1] = Fy * MASS[j]
            acc[j][2] = Fz * MASS[j]

        fields[i, 0] = np.sum(acc[:, 0])
        fields[i, 1] = np.sum(acc[:, 1])
        fields[i, 2] = np.sum(acc[:, 2])

    g_mag = np.sqrt(fields[:, 0] ** 2 + fields[:, 1] ** 2 + fields[:, 2] ** 2)

    return g_mag