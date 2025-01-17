# %% Fluid-Solid boundary 
# =======================
#
# This example demonstrates how to run a fluid-solid simulation and define
# sources and receivers from Python instead of using `Par_file` and
# `specfem_config.yml`, `sources.yml`, and `STATIONS` files.
#
# This example is contributed by Sirawich Pipatprathanporn and is part of the
# publication [Pipatprathanporn et al.
# (2024)](https://doi.org/10.1093/gji/ggae238)
#
# Creating the mesh
# -----------------
#
# We need a mesh to run the simulation. This step still requires a ``Par_file``
# and a call to ``xmeshfem2D``. For the explanation of the ``Par_file``, please
# refer to the [SPECFEM++
# documentation](https://specfem2d-kokkos.readthedocs.io/en/latest/meshfem2d/index.html).

from subprocess import call
from os import makedirs

from specfempp import Config, execute

# Create the output directory for the database files.
makedirs("OUTPUT_FILES/results", exist_ok=True)
call("xmeshfem2D -p Par_file", shell=True)

# %%
# The output should be quite long, but end with something like:
#
# .. code-block:: none
#
#     This will be a serial simulation
#
#
# Now, that we have a mesh, we can set up the simulation. This is entirely done
# from python. We start by importing functions to load a parameter file
# get and set parameters, and execute the simulation.
# So that we don't start completely from scratch, let's load a parameter file
# that is already set up for a fluid-solid simulation.

config = Config("specfem_config.yml")

# %%
# The parameters can be inspected using the `get_par` function.

print(config.get_par("header.description"))
print("DT:", config.get_par("simulation-setup.solver.time-marching.time-scheme.dt"))
print("NT:", config.get_par("simulation-setup.solver.time-marching.time-scheme.nstep"))

# %%
# Let's set the number of time steps to 1000 and the time step size to 0.002.
dt = 0.002
nstep = 5000
config.set_par('simulation-setup.solver.time-marching.time-scheme.dt', dt)
config.set_par('simulation-setup.solver.time-marching.time-scheme.nstep', nstep)

# %%
# Defining a source
# -----------------
# We can define the sources using a file, which is the default behaviour for
# ``specfem2d``. The file is called `sources.yml` and is loaded during execution
# via

print(config.get_par("sources"))

# %%
# We can update the sources programmatically by setting the the ``source-file``
# parameter to a dictionary value. For this example, we want to define a line
# source that immitates a plane wave impinging on the domain from the left at
# an angle of 30 degrees. The source is located at the top of the domain and

# Define the number of sources in x in z direction
number_of_sources_x = 197
number_of_sources_z = 37

# Define the source dictionary
source_dict = dict()
source_dict["number-of-sources"] = number_of_sources_x + number_of_sources_z
source_list = list()

# Append sources in x direction
for i in range(number_of_sources_x):
    moment_tensor = {"moment-tensor": dict()}
    moment_tensor["moment-tensor"]["x"] = 200.0 + i * 100.0
    moment_tensor["moment-tensor"]["z"] = 720.0
    moment_tensor["moment-tensor"]["Mxx"] = 1.0
    moment_tensor["moment-tensor"]["Mzz"] = 1.0
    moment_tensor["moment-tensor"]["Mxz"] = 0.0
    moment_tensor["moment-tensor"]["angle"] = 0.0
    moment_tensor["moment-tensor"]["Ricker"] = dict()
    moment_tensor["moment-tensor"]["Ricker"]["factor"] = 9.836e-10
    moment_tensor["moment-tensor"]["Ricker"]["tshift"] = 5.309e-03 * i
    moment_tensor["moment-tensor"]["Ricker"]["f0"] = 1.0
    source_list.append(moment_tensor)

# Append sources in z direction
for i in range(number_of_sources_z):
    moment_tensor = {"moment-tensor": dict()}
    moment_tensor["moment-tensor"]["x"] = 200.0
    moment_tensor["moment-tensor"]["z"] = 820.0 + i * 100.0
    moment_tensor["moment-tensor"]["Mxx"] = 1.0
    moment_tensor["moment-tensor"]["Mzz"] = 1.0
    moment_tensor["moment-tensor"]["Mxz"] = 0.0
    moment_tensor["moment-tensor"]["angle"] = 0.0
    moment_tensor["moment-tensor"]["Ricker"] = dict()
    moment_tensor["moment-tensor"]["Ricker"]["factor"] = 1.805e-10
    moment_tensor["moment-tensor"]["Ricker"]["tshift"] = 2.893e-02 * (i + 1)
    moment_tensor["moment-tensor"]["Ricker"]["f0"] = 1.0
    source_list.append(moment_tensor)

# Finally, we set the source-file parameter to the source dictionary
source_dict["sources"] = source_list
config.set_par("sources", source_dict)

# %%
# Defining a receiver
# -------------------
#
# The next step is to define the receivers. This is done in the same way as the
# sources. The default file is called `STATIONS` and is loaded during execution
# via

print(config.get_par("receivers.stations"))

# %%
# We can update the receivers programmatically by setting the the
# ``stations-file`` parameter to a dictionary value. For this example, we want
# to define two receivers AA.S0001 and AA.S0002 at x = 10000.0 and z = 8082.0
# and x = 10000.0 and z = 5472.0, respectively.

receiver_list = list()
receiver1 = dict(network="AA", station="S0003", x=10000.0, z=8082.0)
receiver2 = dict(network="AA", station="S0004", x=10000.0, z=5472.0)
receiver_list.extend([receiver1, receiver2])

config.set_par("receivers.stations", receiver_list)


# %%
# Running the simulation
# ----------------------

execute(config)

# %%
# Plotting the results
# --------------------
#
# We can now read the traces and plot them. The traces are stored in the
# `OUTPUT_FILES/results` directory. We can use the `obspy` library to read the
# traces and plot them.

import glob
import os
import numpy as np
import obspy
import matplotlib.pyplot as plt

def get_traces(directory):
    traces = []
    files = glob.glob(directory + "/*.sem*")
    ## iterate over all seismograms
    for filename in files:
        station_name = os.path.splitext(filename)[0]
        station_name = station_name.split("/")[-1]
        trace = np.loadtxt(filename, delimiter=" ")
        starttime = trace[0, 0]
        dt = trace[1, 0] - trace[0, 0]
        traces.append(
            obspy.Trace(
                trace[:, 1],
                {"network": station_name, "starttime": starttime, "delta": dt},
            )
        )

    stream = obspy.Stream(traces)

    return stream


stream = get_traces("OUTPUT_FILES/results")
fig = plt.figure(figsize=(10, 8))
stream.plot(fig=fig)
plt.show(block=False)

# %%
# We can also load the wavefield snapshots and plot them. The snapshots are
# stored in the `OUTPUT_FILES/results` directory. We can use the `matplotlib`
# library to plot them.

import matplotlib.pyplot as plt

def plot_snapshots(directory):
    files = glob.glob(directory + "/*.png")
    files.sort()
    N = len(files)
    Nx = np.ceil(np.sqrt(N)).astype(int)
    Ny = np.ceil(N / Nx).astype(int)
    fig, ax = plt.subplots(Nx, Ny, figsize=(10, 6))
    ax = ax.flatten()
    for i in range(Nx*Ny):
        if i >= N:
            ax[i].axis("off")
        else:
          img = plt.imread(files[i])
          ax[i].imshow(img[700:1900,100:-100,:])
          ax[i].axis("off")
          ax[i].text(0.05, 0.925, f"T={np.round(i*dt,4)}s", fontsize=8, color="black",
                     transform=ax[i].transAxes, ha="left", va="top")
    plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.show(block=False)
    
plot_snapshots("OUTPUT_FILES/results")