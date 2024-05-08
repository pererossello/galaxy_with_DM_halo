# Simple N-body Galaxy in NFW DM Halo 

Python program for N-body simulation of a galaxy inside a Navarro-Frenk-White potential. 

## Usage

Main functionality inside modules in `main/`. Notebooks are exercises relying on those modules for performing simulations. This work is part of an assignment for the subject Numerical Simulation of the master degree in astrophysics at Universidad de La Laguna (ULL).

## Structure

 - `main/`
    - `make_ics.py`: Module for creating initial conditions. Either read from .txt file or generated from scratch with the `galaxy_ics()` function. 
    - `nfw.py`: Integration orbits of bodies only considering the NFW potential.  
    - `nbody_nfw.py`: Integration orbits of bodies considering both the NFW potential and N-body interaction.
- `data/`: Some example data of N-bodies forming a disk galaxy. 
- `figures/`: Figures for the assignment.
Notebooks contain simulations for the specified exercises. 



## Dependencies
- [numpy](https://github.com/numpy/numpy): For math operations.
- [numba](https://numba.pydata.org/): For optimization.
- [matplotlib](https://github.com/matplotlib/matplotlib): For plotting.
- [h5py](https://github.com/h5py/h5py): For saving simulation output data.
- [PIL](https://github.com/python-pillow/Pillow): For saving frames. 
- [ffmpeg](https://ffmpeg.org/): For rendering animations.


## License
This project is licensed under the [MIT License](LICENSE.md).