# %% Fluid-Solid boundary
# ====================
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


# %%
# So that we don't start completely from scratch, let's load a parameter file
config = Config("specfem_config.yml")

# %%
# The parameters can be inspected using the `get_par` function.

print(config.get_par("header.description"))
print("DT:", config.get_par("simulation-setup.solver.time-marching.time-scheme.dt"))
print("NT:", config.get_par("simulation-setup.solver.time-marching.time-scheme.nstep"))

# %%
# Let's set the number of time steps to 1000 and the time step size to 0.002.
config.set_par("simulation-setup.solver.time-marching.time-scheme.dt", 0.002)
config.set_par("simulation-setup.solver.time-marching.time-scheme.nstep", 5000)

# %%
# Defining a source
# -----------------
# We can define the sources using a file, which is the default behaviour for
# ``specfem2d``. The file is called `sources.yml` and is loaded during execution
# via

print(config.get_par("databases.source-file"))

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
config.del_par("databases.source-file")
config.set_par("databases.source-dict", source_dict)

# %%
# Defining a receiver
# -------------------
#
# The next step is to define the receivers. This is done in the same way as the
# sources. The default file is called `STATIONS` and is loaded during execution
# via

print(config.get_par("receivers.stations-file"))

# %%
# We can update the receivers programmatically by setting the the
# ``stations-file`` parameter to a dictionary value. For this example, we want
# to define two receivers AA.S0001 and AA.S0002 at x = 10000.0 and z = 8082.0
# and x = 10000.0 and z = 5472.0, respectively.

receiver_list = list()
receiver1 = dict(network="AA", station="S0001", x=10000.0, z=8082.0)
receiver2 = dict(network="AA", station="S0002", x=10000.0, z=5472.0)
receiver_list.extend([receiver1, receiver2])

config.set_par("receivers.stations-file", receiver_list)


# %%
# Running the simulation
# ----------------------

execute(config)
