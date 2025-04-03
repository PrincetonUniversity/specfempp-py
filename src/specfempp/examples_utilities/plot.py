


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
import typing as tp
from matplotlib.axes import Axes
import matplotlib.patheffects as path_effects
import matplotlib



def xyz_line(x, y, z, *args, **kwargs):
    """Plot multicolored lines by passing norm and cmap to a LineCollection/

    Mosly taken from matplotlib tutorial on multicolored lines (`MPL`_), just
    streamlined here.

    .. _MPL: https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

    Parameters
    ----------
    x : ArrayLike
        x values
    y : ArrayLike
        y values
    z : ArrayLike
        z values

    Raises
    ------
    ValueError
        Errors when x,y,z don' have the same shape

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2025.03.22 20.23 

    """

    # Check if vectors have the same length:
    if not (x.shape == y.shape == z.shape):
        raise ValueError("x, y, and z mustt have he same size")

    # Create Line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create LineCollection
    lines = LineCollection(segments, *args, array=z, 
                           path_effects=[path_effects.Stroke(capstyle="round")], **kwargs)

    # Get current axes
    ax = plt.gca()
    ax.add_collection(lines)

    # autoscale since the add_collection does not autoscale
    ax.autoscale()

    # Create a scalarmappable for colorbar purpose
    sm = ScalarMappable(cmap=lines.cmap, norm=lines.norm)

    return lines, sm



def axes_from_axes(
        ax: Axes, z: int,
        extent: tp.Iterable = [0.2, 0.2, 0.6, 1.0],
        **kwargs) -> Axes:
    """Uses the location of an existing axes to create another axes in relative
    coordinates. IMPORTANT: Unlike ``inset_axes``, this function propagates
    ``*args`` and ``**kwargs`` to the ``pyplot.axes()`` function, which allows
    for the use of the projection ``keyword``.
    Parameters
    ----------
    ax : Axes
        Existing axes
    z : int
        zorder of the new axes   
    extent : list, optional
        new position in axes relative coordinates,
        by default [0.2, 0.2, 0.6, 1.0]
    Returns
    -------
    Axes
        New axes
    Notes
    -----
    DO NOT CHANGE THE INITIAL POSITION, this position works DO NOT CHANGE!
    :Author:
        Lucas Sawade (lsawade@princeton.edu)
    :Last Modified:
        2021.07.13 18.30
    """

    newax = ax.inset_axes(extent, transform=ax.transAxes, zorder=z, **kwargs)
    
    # return new axes
    return newax


class FixPointNormalize(matplotlib.colors.Normalize):
    """ 
    Inspired by https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint 
    somewhere in the middle of the colormap.

    This may be useful for a `terrain` map, to set the "sea level" 
    to a color in the blue/turquise range. 
    """
    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val = 0.21875, clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent the sealevel.
        self.col_val = col_val
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)
        self.x = np.array([self.vmin, self.sealevel, self.vmax])
        self.y = np.array([0, self.col_val, 1])
    def __call__(self, value, clip=None):
        return np.ma.masked_array(np.interp(value, self.x, self.y))


def plot_snapshots(directory, dt):
    import glob
    files = glob.glob(directory + "/*.png")
    files.sort()
    N = len(files)
    Nx = np.ceil(np.sqrt(N)).astype(int)
    Ny = np.ceil(N / Nx).astype(int)
    fig, ax = plt.subplots(Nx, Ny, figsize=(10, 5))
    ax = ax.flatten()
    for i in range(Nx*Ny):
        if i >= N:
            ax[i].axis("off")
        else:
          timestep = int(files[i].split("/")[-1].split(".")[0][9:])
          img = plt.imread(files[i])
          
          ax[i].imshow(img[700:1900,100:-100,:])
          ax[i].axis("off")
          ax[i].text(0.05, 0.925, f"T={np.round(timestep*dt,4)}s", fontsize=8, color="black",
                     transform=ax[i].transAxes, ha="left", va="top")
    plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.show(block=False)
 