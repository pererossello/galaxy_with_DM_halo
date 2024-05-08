import os
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange, float32, int32, void
import h5py


@jit(float32(float32), nopython=True, fastmath=True, cache=True)
def m_NFW(r):
    """
    Calculates the mass enclosed within a radius r for a Navarro-Frenk-White (NFW) profile. Optimized with Numba for performance.
    """

    R_s = np.float32(20.0)  # Use np.float32 to ensure type consistency
    rho_0 = np.float32(5932371.0)  # Use np.float32 for consistency
    result = 4 * np.pi * rho_0 * R_s**3 * (np.log(1 + r / R_s) - r / (r + R_s))
    return result  # The return is already of type float32, no need to cast


@jit(
    void(float32[:, :], int32, float32[:, :], float32),
    nopython=True,
    fastmath=True,
    cache=True,
)
def net_fields(fields, N, POS, G):
    """
    Calculates gravitational forces between N bodies, avoiding self-interaction and using pairwise calculations. Optimized with Numba for performance.
    """
    for i in range(N):
        r_i = np.sqrt(POS[i][0] ** 2 + POS[i][1] ** 2 + POS[i][2] ** 2)
        M_i = m_NFW(r_i)
        NFW_field_factor = -G * M_i / r_i**3

        fields[i, 0] = NFW_field_factor * POS[i][0]
        fields[i, 1] = NFW_field_factor * POS[i][1]
        fields[i, 2] = NFW_field_factor * POS[i][2]


@jit(
    void(float32[:, :], float32[:, :], float32[:, :], float32, int32, float32),
    nopython=True,
    fastmath=True,
    parallel=False,
)
def rk4_step(FIELDS, POS, VEL, DT, N, G):

    net_fields(FIELDS, N, POS, G)
    # First set of calculations (k1)
    k1_vel = FIELDS * DT
    k1_pos = VEL * DT

    # Second set of calculations (k2)
    net_fields(FIELDS, N, POS + np.float32(0.5) * k1_pos, G)
    k2_vel = FIELDS * DT
    k2_pos = (VEL + np.float32(0.5) * k1_vel) * DT

    # Third set of calculations (k3)
    net_fields(FIELDS, N, POS + np.float32(0.5) * k2_pos, G)
    k3_vel = FIELDS * DT
    k3_pos = (VEL + np.float32(0.5) * k2_vel) * DT

    # Fourth set of calculations (k4)
    net_fields(FIELDS, N, POS + k3_pos, G)
    k4_vel = FIELDS * DT
    k4_pos = (VEL + k3_vel) * DT

    # Combine the results
    POS += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6
    VEL += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6


@jit(
    void(float32[:, :], float32[:, :], float32[:, :], float32, int32, float32),
    nopython=True,
    fastmath=True,
    parallel=False,
)
def euler_step(FIELDS, POS, VEL, DT, N, G):

    net_fields(FIELDS, N, POS, G)

    # Update position
    POS += VEL * DT
    # Update velocity
    VEL += FIELDS * DT


integrator_dic = {
    "RK4": rk4_step,
    "euler": euler_step,
}


class Simulation:
    def __init__(self, path_ics, snap_path):
        """
        Initializes the simulation with paths for initial conditions and snapshot output, and loads initial state data from an HDF5 file.
        """

        self.simulation_done = False
        self.path_ics = path_ics
        self.path_output = snap_path

        self._create_output_file()

        with h5py.File(self.path_ics, "r") as file:
            self.dimensions = file["Header"].attrs["Dimensions"]
            self.N = file["Header"].attrs["N"]

            self.POS = np.array(file["Bodies"]["Positions"])
            self.VEL = np.array(file["Bodies"]["Velocities"])
            self.MASS = np.array(file["Bodies"]["Masses"])

    def set_time(self, duration, time_step, snapshots=100):
        """
        Sets the simulation duration, time step, and the number of snapshots based on the total duration.
        """

        self.duration = np.array(duration, dtype=np.float32)
        self.time_step = np.float32(time_step)

        max_snapshots = int(duration / time_step)

        if snapshots > max_snapshots:
            snapshots = max_snapshots

        self.snap_times = np.linspace(0.0, duration, snapshots)

    def set_integrator(self, integrator):

        if integrator not in integrator_dic:
            raise ValueError(f"Integrator {integrator} not supported")

        self.update_positions = integrator_dic[integrator]

    def run_simulation(self, G=4.302e-6):
        """
        Runs the N-body simulation, managing time steps, updating positions, handling first step differentiation, and saving snapshots at predefined times.
        """

        if self.simulation_done:
            raise ValueError("Create new instance to run a new simulation.")

        self.G = np.float32(G)

        fields = np.zeros((self.N, 3), dtype=np.float32)
        self.time = 0.0
        self.snapshot_idx = 0
        start_clock = time.time()
        while self.time < self.duration:

            self.update_positions(
                fields,
                self.POS,
                self.VEL,
                self.time_step,
                self.N,
                self.G,
            )

            self.time += self.time_step
            if self.time >= self.snap_times[self.snapshot_idx]:
                self.snapshot_idx += 1
                self._save_snapshot()
                perc = self.time / (self.duration + self.time_step) * 100
                print(f"\r{perc:.2f}%", end="")

        end_clock = time.time()
        print(f"\nElapsed time: {end_clock - start_clock:.2f} s")

        self.simulation_done = True

        self._finish_up_output_file()

    def _create_output_file(self):
        """
        Prepares the output file structure based on the initial conditions file, ensuring directories exist and old data is cleared.
        """

        if not os.path.exists(self.path_output):
            save_dir = os.path.dirname(self.path_output)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            os.remove(self.path_output)

        with h5py.File(self.path_ics, "r") as file_ics:
            with h5py.File(self.path_output, "w") as file_output:

                for group_name in file_ics.keys():

                    if group_name != "Bodies":
                        group = file_output.create_group(group_name)
                    else:
                        step = 0
                        new_name = f"{step:04d}"
                        group = file_output.create_group(new_name)

                    for attr_name, attr_value in file_ics[group_name].attrs.items():
                        group.attrs[attr_name] = attr_value

                    for dataset_name in file_ics[group_name].keys():
                        source_dataset = file_ics[group_name][dataset_name]
                        group.create_dataset(dataset_name, data=source_dataset[...])

                step = 0
                file_output[f"{step:04d}"].attrs["Time"] = np.float32(0.0)

    def _save_snapshot(self):
        """
        Saves the current state of the simulation to an output file as a snapshot at the current time step.
        """
        with h5py.File(self.path_output, "a") as f:

            step_group = f.create_group(f"{self.snapshot_idx:04d}")
            step_group.create_dataset("Positions", data=self.POS, dtype=np.float32)
            step_group.create_dataset("Velocities", data=self.VEL, dtype=np.float32)
            step_group.attrs["Time"] = self.time

        return

    def _finish_up_output_file(self):
        """
        Finalizes the output file by updating the header with the total number of snapshots.
        """
        with h5py.File(self.path_output, "a") as file:
            file["Header"].attrs["NSnapshots"] = len(list(file.keys())) - 1

    def get_output(self):
        with h5py.File(self.path_output, "r") as file:
            N = file["Header"].attrs["N"]
            M = file["Header"].attrs["NSnapshots"]

            POS = np.zeros((M, N, 3))
            VEL = np.zeros((M, N, 3))
            MASS = file[f"{0:04d}"]["Masses"][()]
            TIME = np.zeros(M)
            for ii, i in enumerate(range(M)):
                POS[ii] = file[f"{i:04d}"]["Positions"][()]
                VEL[ii] = file[f"{i:04d}"]["Velocities"][()]
                TIME[ii] = file[f"{i:04d}"].attrs["Time"]

        return POS, VEL, MASS, TIME
