import os
import shutil
import time
import subprocess
import PIL

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import h5py


class Figure:

    def __init__(
        self,
        subplot_1=1,
        subplot_2=1,
        fig_size=720,
        ratio=1,
        dpi=300,
        width_ratios=None,
        height_ratios=None,
        hspace=None,
        wspace=None,
    ):
        """
        Initializes a matplotlib figure with configurable subplots, figure size, DPI, aspect ratios, and spacing. Sets up figure and axes aesthetics like colors and grid settings.
        """
        self.fig_size = fig_size
        self.ratio = ratio
        self.dpi = dpi
        self.subplot_1 = subplot_1
        self.subplot_2 = subplot_2
        self.width_ratios = width_ratios
        self.height_ratios = height_ratios
        self.hspace = hspace
        self.wspace = wspace

        fig_width, fig_height = fig_size * ratio / dpi, fig_size / dpi
        fs = np.sqrt(fig_width * fig_height)
        self.fs = fs

        self.fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        self.ts = 2
        self.sw = 0.2
        self.pad = 0.21
        self.minor_ticks = True
        self.grid = False
        self.ax_color = "k"
        self.facecolor = "w"
        self.text_color = "k"

    def get_axes(self, flat=False):
        """
        Generates and returns the axes of the figure based on the subplot configuration. Optionally returns all axes as a flat list.
        """
        plt.rcParams.update({"text.color": self.text_color})
        self.fig.patch.set_facecolor(self.facecolor)

        subplots = (self.subplot_1, self.subplot_2)
        self.subplots = subplots
        self.gs = mpl.gridspec.GridSpec(
            nrows=subplots[0],
            ncols=subplots[1],
            figure=self.fig,
            width_ratios=self.width_ratios or [1] * subplots[1],
            height_ratios=self.height_ratios or [1] * subplots[0],
            hspace=self.hspace,
            wspace=self.wspace,
        )

        self.axes = []
        for i in range(self.subplots[0]):
            row_axes = []
            for j in range(self.subplots[1]):
                ax = self.fig.add_subplot(self.gs[i, j])
                row_axes.append(ax)
                self.customize_axes(ax)

                if self.hspace == 0 and i != self.subplots[0] - 1:
                    ax.set_xticklabels([])

            self.axes.append(row_axes)

        if self.subplot_1 == 1 and self.subplot_2 == 1:
            return self.axes[0][0]

        self.axes_flat = [ax for row in self.axes for ax in row]

        if flat:
            return self.axes_flat
        else:
            return self.axes

    def customize_axes(
        self,
        ax,
    ):
        """
        Applies custom settings to axes, including tick parameters, colors, grid visibility, and minor tick settings.
        """
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.ts * self.fs,
            size=self.fs * self.sw * 5,
            width=self.fs * self.sw * 0.9,
            pad=self.pad * self.fs,
            top=True,
            right=True,
            direction="inout",
            color=self.ax_color,
            labelcolor=self.ax_color,
        )

        if self.minor_ticks == True:
            ax.minorticks_on()

            ax.tick_params(
                axis="both",
                which="minor",
                direction="inout",
                top=True,
                right=True,
                size=self.fs * self.sw * 2.5,
                width=self.fs * self.sw * 0.8,
                color=self.ax_color,
            )

        ax.set_facecolor(self.facecolor)

        for spine in ax.spines.values():
            spine.set_linewidth(self.fs * self.sw)
            spine.set_color(self.ax_color)

        if self.grid:

            ax.grid(
                which="major",
                linewidth=self.fs * self.sw * 0.5,
                color=self.ax_color,
                alpha=0.25,
            )

    def save(self, path, bbox_inches="tight", pad_inches=None):
        """
        Saves the figure to the specified path with options for DPI and padding adjustments.
        """
        self.fig.savefig(
            path, dpi=self.dpi, bbox_inches=bbox_inches, pad_inches=pad_inches
        )

        self.path = path


def make_animation(
    path_output,
    ratio=1,
    fig_size=1400,
    lim=1,
    marker_color="k",
    marker_size=None,
    facecolor="w",
    ax_color="k",
    ax_spines=True,
    fps=30,
    delete_frames=False,
    reverse=False,
    N_max=None,
):

    if isinstance(lim, (float, int)):
        lim_x = [-lim, lim]
        lim_y = [-lim, lim]

    # get folder of the data_path
    savefold = path_output.split("/")[:-1]
    savefold = "/".join(savefold) + "/frames/"

    if not os.path.exists(savefold):
        os.makedirs(savefold)
    else:
        shutil.rmtree(savefold)
        os.makedirs(savefold)

    Fig = Figure(ratio=ratio, fig_size=fig_size)
    Fig.facecolor = facecolor
    Fig.ax_color = ax_color
    ax = Fig.get_axes()
    fs = Fig.fs

    ax.axis("equal")

    ax.set_xlim(lim_x)
    ax.set_ylim(lim_y)

    with h5py.File(path_output, "r") as file:

        N = file["Header"].attrs["N"]
        M = file["Header"].attrs["NSnapshots"]

        if N_max is None:
            N_max = M
        else:
            N_max = min(N_max, M)

        if marker_size is None:
            marker_size = 100 / N

        time_start = time.time()
        for ii, i in enumerate(range(N_max)):
            t = file[f"{i:04d}"].attrs["Time"]
            POS = file[f"{i:04d}"]["Positions"]

            scatter = ax.scatter(
                POS[:, 0],
                POS[:, 1],
                c=marker_color,
                s=fs * marker_size,
                lw=0.0 * fs,
                alpha=1,
            )

            text = ax.text(
                0.98,
                0.02,
                f"Time: {1e3 * 0.98*t:.2f} Myr",
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=ax.transAxes,
                fontsize=fs * 1.5,
                color=ax_color,
            )

            fig_name = f"render_{ii:04d}.jpg"
            save_path = savefold + fig_name

            if ax_spines:
                Fig.save(save_path)
            else:

                Fig.fig.subplots_adjust(
                    top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
                )

                ax.set_xticks([])
                ax.set_yticks([])

                for spine in ax.spines.values():
                    spine.set_visible(False)

                Fig.fig.patch.set_facecolor("grey")

                Fig.save(save_path, bbox_inches="tight", pad_inches=0)

            plt.close()

            scatter.remove()
            text.remove()

    print(f"Save images time: {time.time() - time_start:.2f} s")

    time_start = time.time()

    png_to_mp4(savefold, fps=fps, reverse=reverse)

    print(f"Video creation time: {time.time() - time_start:.2f} s")

    if delete_frames:
        shutil.rmtree(savefold)

    return


def png_to_mp4(
    fold,
    title="video",
    fps=36,
    digit_format="04d",
    res=None,
    resize_factor=1,
    custom_bitrate=None,
    extension=".jpg",
    reverse=False,  # Adding reverse parameter with default value False
):

    # Get a list of all image files in the directory with the specified extension
    files = [f for f in os.listdir(fold) if f.endswith(extension)]
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    if not files:
        raise ValueError("No image files found in the specified folder.")

    im = PIL.Image.open(os.path.join(fold, files[0]))
    resx, resy = im.size

    if res is not None:
        resx, resy = res
    else:
        resx = int(resize_factor * resx)
        resy = int(resize_factor * resy)
        resx += resx % 2  # Ensuring even dimensions
        resy += resy % 2

    basename = os.path.splitext(files[0])[0].split("_")[0]

    ffmpeg_path = "ffmpeg"
    abs_path = os.path.abspath(fold)
    parent_folder = os.path.dirname(abs_path) + os.sep
    output_file = os.path.join(parent_folder, f"{title}.mp4")

    crf = 5  # Lower CRF for higher quality, higher for lower quality
    bitrate = custom_bitrate if custom_bitrate else "5000k"
    preset = "slow"
    tune = "film"

    # Construct the ffmpeg command
    command = f'{ffmpeg_path} -y -r {fps} -i {os.path.join(fold, f"{basename}_%{digit_format}{extension}")} -c:v libx264 -profile:v high -crf {crf} -preset {preset} -tune {tune} -b:v {bitrate} -pix_fmt yuv420p -vf "scale={resx}:{resy}'
    if reverse:
        command += ",reverse"  # Appends the reverse filter if reverse is True
    command += f'" {output_file}'

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error during video conversion:", e)


def plot_euler_rk4_orbits(POS_E, VEL_E, POS_R, VEL_R, TIME, savepath):

    c0 = 0.25
    cmap1 = plt.get_cmap("Blues")
    colors = cmap1(np.linspace(c0, 1, 256))
    cmap1 = LinearSegmentedColormap.from_list("blues", colors)

    cmap2 = plt.get_cmap("Reds")
    colors = cmap2(np.linspace(c0, 1, 256))
    cmap2 = LinearSegmentedColormap.from_list("reds", colors)

    c1 = cmap1(0.8)
    c2 = cmap2(0.8)

    ts = 2
    Fig = Figure(3, 1, ratio=0.55)
    fs = Fig.fs
    Fig.ts = ts
    h_rt = 0.3
    Fig.height_ratios = [h_rt, h_rt, 1]

    axs = Fig.get_axes(flat=True)

    r_1 = np.sqrt(POS_E[:, 0, 0] ** 2 + POS_E[:, 0, 1] ** 2 + POS_E[:, 0, 2] ** 2)
    r_2 = np.sqrt(POS_R[:, 0, 0] ** 2 + POS_R[:, 0, 1] ** 2 + POS_R[:, 0, 2] ** 2)
    axs[0].plot(TIME, r_1, color=c1, lw=fs * 0.5, label="Euler Step")
    axs[0].plot(TIME, r_2, color=c2, lw=fs * 0.5, label="RK4 Step")

    axs[0].tick_params(axis="x", labelbottom=False, labeltop=True)
    axs[0].set_xlim(0, TIME[-1])
    axs[0].set_xlabel(
        r"Time [$18.6\times10^{8}$ yr]", fontsize=ts * fs, labelpad=1 * fs
    )
    axs[0].xaxis.set_label_position("top")
    axs[0].set_ylabel("r [kpc]", fontsize=ts * fs, labelpad=0.5 * fs)

    axs[0].legend(fontsize=0.85 * ts * fs, loc="upper left", frameon=False)

    v_1 = np.sqrt(VEL_E[:, 0, 0] ** 2 + VEL_E[:, 0, 1] ** 2 + VEL_E[:, 0, 2] ** 2)
    v_2 = np.sqrt(VEL_R[:, 0, 0] ** 2 + VEL_R[:, 0, 1] ** 2 + VEL_R[:, 0, 2] ** 2)
    axs[1].plot(TIME, v_1, color=c1, lw=fs * 0.5)
    axs[1].plot(TIME, v_2, color=c2, lw=fs * 0.5)
    axs[1].set_xticklabels([])
    axs[1].set_xlim(0, TIME[-1])

    axs[1].set_ylabel("v [km/s]", fontsize=ts * fs, labelpad=0.5 * fs)

    lw = 0.25
    x_E, y_E = POS_E[:, 0, 0], POS_E[:, 0, 1]
    points = np.array([x_E, y_E]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm1 = plt.Normalize(0, len(segments))
    lc1 = LineCollection(
        segments, cmap=cmap1, norm=norm1, alpha=1, linewidth=fs * lw, zorder=0
    )
    lc1.set_array(np.arange(len(segments)))
    line1 = axs[2].add_collection(lc1)

    x_R, y_R = POS_R[:, 0, 0], POS_R[:, 0, 1]
    points = np.array([x_R, y_R]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm2 = plt.Normalize(0, len(segments))
    lc2 = LineCollection(
        segments, cmap=cmap2, norm=norm2, alpha=1, linewidth=fs * lw, zorder=0
    )
    lc2.set_array(np.arange(len(segments)))
    line2 = axs[2].add_collection(lc2)

    lim = 15
    axs[2].set_xlim(-lim, lim)
    axs[2].set_ylim(-lim, lim)
    axs[2].set_aspect("equal")

    axs[2].set_xlabel("x [kpc]", fontsize=ts * fs, labelpad=0.5 * fs)
    axs[2].set_ylabel("y [kpc]", fontsize=ts * fs, labelpad=-2 * fs)

    position = axs[1].get_position()
    left, bottom, width, height = (
        position.x0,
        position.y0,
        position.width,
        position.height,
    )
    new_bottom = bottom + 0.045
    axs[1].set_position([left, new_bottom, width, height])

    position = axs[2].get_position()
    left, bottom, width, height = (
        position.x0,
        position.y0,
        position.width,
        position.height,
    )
    new_bottom = bottom + 0.06
    axs[2].set_position([left, new_bottom, width, height])

    Fig.save(savepath)


def plot_galaxy_orbits(
    POS10,
    MASS10,
    POS100,
    MASS100,
    POS1000,
    MASS1000,
    savepath,
    projection="XY",
    lim=30,
    res=100,
    vmax=25,
):

    cmap = "turbo"
    c0 = 0.01
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(c0, 1, 256))
    cmap = LinearSegmentedColormap.from_list("ok", colors)

    ts = 2
    Fig = Figure(1, 3, ratio=2, wspace=0)
    fs = Fig.fs
    Fig.grid = True
    Fig.ts = ts
    Fig.ax_color = "k"
    axs = Fig.get_axes(flat=True)

    for ax in axs:
        ax.set_aspect("equal")

    im_10, min_10, max_10 = plot_heatmap(
        axs[0], POS10, MASS10, cmap, res, lim, vmax=vmax, projection=projection
    )
    im_100, min_100, max_100 = plot_heatmap(
        axs[1], POS100, MASS100, cmap, res, lim, vmax=vmax, projection=projection
    )
    im_1000, min_1000, max_1000 = plot_heatmap(
        axs[2], POS1000, MASS1000, cmap, res, lim, vmax=vmax, projection=projection
    )

    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])

    if projection == "XY":
        x_lab = "x"
        y_lab = "y"
    elif projection == "XZ":
        x_lab = "x"
        y_lab = "z"
    elif projection == "YZ":
        x_lab = "y"
        y_lab = "z"

    for i in range(3):
        axs[i].set_xlabel(f"{x_lab} [kpc]", fontsize=fs * ts, labelpad=0 * fs)

    axs[0].set_ylabel(f"{y_lab} [kpc]", fontsize=fs * ts, labelpad=-1 * fs)

    max_ = max(max_10, max_100, max_1000) / 1e10
    min_ = min(min_10, min_100, min_1000)

    # add a colorbar to the right
    cax = Fig.fig.add_axes([0.92, 0.25, 0.02, 0.5])
    norm = plt.Normalize(0, vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    Fig.fig.colorbar(sm, cax=cax)

    # change the fontsize of the colorbar
    cax.tick_params(labelsize=fs * ts)
    # add label
    cax.set_ylabel(
        r"[$10^{10} \, \text{M}_{\!\odot} \, \text{kpc}^{-2} \, 18.6\times10^{8} \, \text{yr}$]",
        fontsize=fs * ts,
        labelpad=1 * fs,
    )

    Fig.save(savepath)


def plot_heatmap(
    ax, POS, MASS, cmap, res, lim, vmin=0, vmax=25, projection="XY", log=False
):

    if projection == "XY":
        id1, id2 = 0, 1
        id_int = 2
    elif projection == "XZ":
        id1, id2 = 0, 2
        id_int = 1
    elif projection == "YZ":
        id1, id2 = 1, 2
        id_int = 0

    ranges = [(-lim, lim), (-lim, lim), (-lim, lim)]

    M, N, _ = POS.shape
    POS_ = POS.reshape(M * N, 3)
    hist, edges = np.histogramdd(POS_, bins=[res] * 3, range=ranges)
    projection = np.sum(hist, axis=id_int) * MASS[0] / 1e10

    if log == True:
        projection = np.log10(projection + 1)

    bins = np.linspace(-lim, lim, res)
    xs = (edges[id1][1:] + edges[id1][:-1]) / 2
    ys = (edges[id2][1:] + edges[id2][:-1]) / 2
    X, Y = np.meshgrid(xs, ys)

    im = ax.pcolormesh(X, Y, projection.T, cmap=cmap, vmin=vmin, vmax=vmax)

    max_ = np.max(projection) * 1e10
    min_ = np.min(projection) * 1e10

    return im, min_, max_


def plot_ts(POS, VEL, TIME, threshold=100, color="k", lw=0.5, alpha=0.15, vlim=500):

    ts = 2
    Fig = Figure(1, 1, ratio=2, hspace=0, wspace=0.05)
    Fig.grid = True
    Fig.ts = ts
    Fig.ax_color = "k"
    fs = Fig.fs
    ax = Fig.get_axes()

    R = np.linalg.norm(POS, axis=2)
    V = np.linalg.norm(VEL, axis=2)
    N = POS.shape[1]

    # Find indices where the radius exceeds the limit
    R_max = np.max(R, axis=0)
    idx = np.where(R_max > threshold)[0]

    V_dif = np.diff(V, axis=0)

    tss, tss_v = [], []
    highlighted_count = 0

    for i in range(N):
        if i in idx:
            if i == idx[0]:
                ax.plot(
                    TIME[:], R[:, i], color="r", alpha=alpha, label="Leave the system"
                )
            else:
                ax.plot(TIME[:], R[:, i], color="r", alpha=alpha)
            highlighted_count += 1
        else:
            tss.append(R[:, i])
            tss_v.append(V[:, i])

    print("Particles Leaving the System:", highlighted_count)

    # Compute statistics
    tss = np.array(tss)
    tss_max = np.quantile(tss, 0.9, axis=0)
    tss_min = np.quantile(tss, 0.1, axis=0)
    tss_mean = np.mean(tss, axis=0)

    # Plot statistics
    ax.plot(TIME, tss_mean, color="k", alpha=1, label="Mean trajectory")
    ax.fill_between(
        TIME, tss_min, tss_max, color="k", alpha=0.25, lw=0, label="90% of particles"
    )

    ax.set_ylim(0, 1.5 * threshold)
    ax.set_xlim(0.01, TIME[-1] - 0.1)
    ax.set_xlabel(r"Time [$18.6\times10^{8}$ yr]")
    ax.set_ylabel(r"$r$ [kpc]")

    ax.legend(loc="upper right", fontsize=fs * ts)


def plot_galaxy(path_output, savepath, vmaxs=[25, 25], log=True):

    with h5py.File(path_output, "r") as file:

        N = file["Header"].attrs["N"]
        M = file["Header"].attrs["NSnapshots"]

        POS = np.zeros((M, N, 3))
        MASS = file[f"{0:04d}"]["Masses"][()]

        for i in range(M):
            POS[i] = file[f"{i:04d}"]["Positions"][()]

    ts = 2
    Fig = Figure(1, 2, ratio=2, fig_size=1080, wspace=0.65)
    fs = Fig.fs
    axs = Fig.get_axes(flat=True)

    plot_heatmap(
        axs[0],
        POS,
        MASS,
        cmap="turbo",
        res=100,
        lim=30,
        vmax=vmaxs[0],
        projection="XY",
        log=log,
    )
    plot_heatmap(
        axs[1],
        POS,
        MASS,
        cmap="turbo",
        res=100,
        lim=30,
        vmax=vmaxs[1],
        projection="YZ",
        log=log,
    )

    for ax in axs:
        ax.set_aspect("equal")

    axs[0].set_ylabel("y [kpc]", labelpad=0 * fs)
    axs[0].set_xlabel("x [kpc]")

    axs[1].set_ylabel("z [kpc]", labelpad=0 * fs)
    axs[1].set_xlabel("y [kpc]")

    cmap = "turbo"
    c0 = 0.01
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(c0, 1, 256))
    cmap = LinearSegmentedColormap.from_list("ok", colors)

    cax = Fig.fig.add_axes([0.435, 0.25, 0.02, 0.5])
    norm = plt.Normalize(0, vmaxs[0])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    Fig.fig.colorbar(sm, cax=cax)
    # change the fontsize of the colorbar
    cax.tick_params(labelsize=fs * ts)
    # add label
    cax.set_ylabel(
        r"$\log$ [$10^{10} \, \text{M}_{\!\odot} \, \text{kpc}^{-2} \, 18.6\times10^{8} \, \text{yr}$]",
        fontsize=fs * ts,
        labelpad=1 * fs,
    )

    cax = Fig.fig.add_axes([0.92, 0.25, 0.02, 0.5])
    norm = plt.Normalize(0, vmaxs[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    Fig.fig.colorbar(sm, cax=cax)
    # change the fontsize of the colorbar
    cax.tick_params(labelsize=fs * ts)
    # add label
    cax.set_ylabel(
        r"$\log$ [$10^{10} \, \text{M}_{\!\odot} \, \text{kpc}^{-2} \, 18.6\times10^{8} \, \text{yr}$]",
        fontsize=fs * ts,
        labelpad=1 * fs,
    )

    Fig.save(savepath)
