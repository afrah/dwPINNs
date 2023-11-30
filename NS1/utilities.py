
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd

import tensorflow as tf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scipy.io
import random
import glob
import json

import h5py  # for loading matlab data
from datetime import datetime
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.animation import PillowWriter


import logging

import torch
os.environ["KMP_WARNINGS"] = "FALSE" 


# import torch.optim as optim
# from torch.utils.data import DataLoader

# from Common import NeuralNet
# from scipy.stats import qmc

# torch.manual_seed(1234)
# from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F


from matplotlib.animation import FuncAnimation
import logging

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # CPU:-1; GPU0: 1; GPU1: 0;
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
random.seed(1234)
np.random.seed(1234)

###############################
os.environ["KMP_WARNINGS"] = "FALSE"
import matplotlib.tri as tri

# import seaborn as sns

import matplotlib.animation as mpa




##########################################################################################

class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0  # for torchscript, CoreML and ONN


# plot_dataset( model.dirname , WALL , INLET , OUTLET , coll , INITIAL , XY_c , dist)


def plot_dataset(file, wall, inlet, outlet, collo, initial, dist):
    plt.figure(figsize=(16, 8), dpi=150)
    plt.scatter(wall[:, 1], wall[:, 0], label="WALL Points")
    plt.scatter(inlet[:, 1], inlet[:, 0], label="INLET Points")
    plt.scatter(outlet[:, 1], outlet[:, 0], label="OUTLET Points")
    #     plt.scatter(coll[:, 1], coll[:, 0], label="Interior Points")
    plt.scatter(initial[:, 1], initial[:, 0], label="Initial Points")
    plt.scatter(collo[:, 1], collo[:, 0], label="Sensor Points", marker="*")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend()
    text = str(dist) + "_distribution.png"
    plt.savefig(os.path.join(file, text))
    plt.close()

##plot_dataset(model.dirname , up , bottom , left , right , domain , INITIAL , XY_c , dist)


#############################################################################################
def lp_error(pred, exact, text, model, p, printTxt=False):
    num = np.sum(np.abs(pred - exact) ** p)
    denum = np.sum(np.abs(exact) ** p)
    if denum == 0.0:
        denum = 1.0
        text = text + " (Absolute (denominator is zero))"
    result = ((num / denum) ** (1 / p)) * 100
    if printTxt:
        model.print("%s : %5.3e %%" % (text, result))
    return result


#############################################################################################


def predict_result(model, XY_c, N_data, tstep, text="all", stm=False):
    model.print("\nPredicting nn solution")
    # [[tfa , xfa , yfa , ufa , vfa  , pfa] = test_data
    tfa = XY_c[:, 0].flatten()[:, None]  # test_data[0]
    xfa = XY_c[:, 1].flatten()[:, None]  # test_data[1]
    yfa = XY_c[:, 2].flatten()[:, None]  # test_data[2]
    ufa = XY_c[:, 3].flatten()[:, None]  # test_data[3]
    vfa = XY_c[:, 4].flatten()[:, None]  # test_data[4]
    pfa = XY_c[:, 5].flatten()[:, None]  # test_data[5]
    ############################################
    #     inputs = XY_c[: , 0:3]

    #     output = model.approximate_solution(torch.from_numpy(inputs))
    #     output = output.detach().numpy()
    #     u_preda, v_preda , p_preda = output[: ,0:1] , output[: ,1:2] , output[: ,2:3]
    u_preda, v_preda, p_preda = model.predict(tfa, xfa, yfa)

    #########################################

    model.print("\n Relative L2 ERROR:")
    lp_error(u_preda, ufa, (text + " U velocity "), model, 2, True)
    lp_error(v_preda, vfa, (text + " V velocity "), model, 2, True)
    lp_error(p_preda, pfa, (text + " P Pressure "), model, 2, True)

    model.print("\n Relative l1 error")

    lp_error(u_preda, ufa, (text + " U velocity "), model, 1, True)
    lp_error(v_preda, vfa, (text + " V velocity "), model, 1, True)
    lp_error(p_preda, pfa, (text + " P Pressure "), model, 1, True)

    if stm:
        file = text + "_result.mat"
        save_to_matlab(
            tfa,
            xfa,
            yfa,
            ufa,
            vfa,
            pfa,
            u_preda,
            v_preda,
            p_preda,
            fileName=os.path.join(model.dirname, file),
        )

    file = text + ".png"
    #     plot_error_over_time(tstep , N_data , ufa , vfa , pfa , u_preda , v_preda  , p_preda ,
    #                          os.path.join(model.dirname,file) , model)

    # if ( text == 'all'):
    return [tfa, xfa, yfa, ufa, vfa, pfa, u_preda, v_preda, p_preda]
    # return


###################################################


def draw_error_over_time(peotDic,path):
    fig = plt.figure(figsize=(18, 5))

    for i, key in enumerate(peotDic.keys()):
        ax = plt.subplot(1, len(peotDic), i + 1)
        ax.plot(peotDic[key][0], label="$\mathcal{u}_{error}$")
        ax.plot(peotDic[key][1], label="$\mathcal{v}_{error}$")
        ax.plot(peotDic[key][2], label="$\mathcal{p}_{error}$")
        # ax.set_yscale('log')
        ax.set_xlabel("time")
        ax.set_ylabel("$L_2}$")
        ax.set_title(key)
    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.42, -0.03),
        borderaxespad=0,
        bbox_transform=fig.transFigure,
        ncol=3,
    )
    plt.tight_layout()
    text = "error_over_time"
    plt.savefig(os.path.join(path, text) , bbox_inches ="tight")
    plt.close("all")



################################################################################################
################################################################################################
def error_over_time(tstep, N_data, ufw, vfw, pfw, u_predw, v_predw, p_predw):
    ufw = ufw.reshape(tstep, N_data)
    vfw = vfw.reshape(tstep, N_data)
    pfw = pfw.reshape(tstep, N_data)

    u_predw = u_predw.reshape(tstep, N_data)
    v_predw = v_predw.reshape(tstep, N_data)
    p_predw = p_predw.reshape(tstep, N_data)

    ufwError = []
    vfwError = []
    pfwError = []

    for index in range(tstep):
        ufwError.append(np.linalg.norm((u_predw[index, :] - ufw[index, :]), 2))
        vfwError.append(np.linalg.norm((v_predw[index, :] - vfw[index, :]), 2))
        pfwError.append(np.linalg.norm((p_predw[index, :] - pfw[index, :]), 2))

    return [ufwError, vfwError, pfwError]


################################################################################################

#UVelocity_profile(model , 0.2 , xValues , xf , u_pred)
## drawing velocity profile
def UVelocity_profile(dirname, initSpeed , xValues, xf, u_preda):
    Uavg = initSpeed
    # average velocity [m/s] (inlet velocity)
    D = 0.015
    # diameter [m]

    R = D / 2.0
    # radius [m]
    drR = R / 100

    n = 101
    timeStep = 0

    colors = ["b", "c", "m", "y", "k", "w"]
    # xValues = [0.2275]
    fig_2 = plt.figure(2)
    ax = fig_2.add_subplot(1, 1, 1)
    rR = np.linspace(0, R, n)

    uxR = 2 * Uavg * (1 - rR**2 / R**2)
    # fully developed laminar vel. profile

    ax.plot(uxR, -rR, "g-.", label="$\mathcal{u}_{profile}$")
    ax.plot(uxR, rR, "g-.")

    # Drawing 6 predicted velocity profiles
    for index in range(len(xValues)):
        vIdx = np.where(xf[:,] == xValues[index])[0]
        u0p = u_preda[vIdx]
        rR = np.linspace(0, R, u0p.shape[0])
        ax.plot(
            u0p,
            -rR,
            colors[index],
            label="$\mathcal{{Ux=}}_{{{}}}$".format(xValues[index]),
        )
        ax.plot(u0p, rR, colors[index])

    # ax.set_yscale('log')
    ax.set_xlim([uxR.min(), uxR.max()])
    ax.set_ylim([-rR.max(), rR.max()])
    ax.set_xlabel("Ux [m/s]")
    ax.set_ylabel("radius [m]")
    plt.legend()
    plt.tight_layout()
    text = "vProfile.png"
    plt.savefig(os.path.join(dirname, text))
    plt.close("all")


###############################################################


###############################################################


def postProcess2(dirname, result, s=2, alpha=0.5, marker="o"):
    [tf, xf, yf, ufa, vfa, pfa, u_preda, v_preda, p_preda] = result

    y = [u_preda, v_preda, p_preda, ufa, vfa, pfa]
    title = [
        "$u$ (m/s)Pred",
        "$v$ (m/s)Pred",
        "Pressure (Pa)Pred",
        "$u$ (m/s)",
        "$v$ (m/s)",
        "Pressure (Pa)",
    ]
    nrows = 2
    ncols = 3

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 4))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    for i in range(nrows):
        for j in range(ncols):
            cf = ax[j, i].scatter(
                xf,
                yf,
                c=y[3 * i + j],
                alpha=alpha - 0.1,
                edgecolors="none",
                cmap="rainbow",
                marker=marker,
                s=int(s),
            )
            ax[j, i].axis("square")
            for key, spine in ax[0, 0].spines.items():
                if key in ["right", "top", "left", "bottom"]:
                    spine.set_visible(False)
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ax[j, i].set_xlim([xf.min(), xf.max()])
            ax[j, i].set_ylim([yf.min(), yf.max()])
            # cf.cmap.set_under('whitesmoke')
            # cf.cmap.set_over('black')
            ax[j, i].set_title(r"%s" % (title[3 * i + j]))
            fig.colorbar(cf, ax=ax[j, i], fraction=0.046, pad=0.04)

    plt.savefig(dirname, dpi=300)
    plt.close("all")


##################################################


def scatter_plot(
    dirname, x, y, data, titles, nrows_ncols, timeStep_list, fontsize=9, labelsize=9
):
    for index in timeStep_list:
        fig = plt.figure()
        grid = ImageGrid(
            fig,
            111,
            direction="row",
            nrows_ncols=nrows_ncols,
            label_mode="1",
            axes_pad=0.5,
            share_all=False,
            cbar_mode="each",
            cbar_location="right",
            cbar_size="5%",
            cbar_pad=0.0,
        )

        # CREATE ARGUMENTS DICT FOR CONTOURPLOTS
        minmax_list = []
        kwargs_list = []
        for d in data:
            minmax_list.append([np.min(d[index, :]), np.max(d[index, :])])
            kwargs_list.append(
                dict(cmap="coolwarm", vmin=minmax_list[-1][0], vmax=minmax_list[-1][1])
            )

        # CREATE PLOTS
        for ax, z, kwargs, minmax, title in zip(
            grid, data, kwargs_list, minmax_list, titles
        ):
            pcf = [
                ax.scatter(
                    x,
                    y,
                    c=z[index, :],
                    alpha=0.4,
                    edgecolors="none",
                    marker="o",
                    s=int(2),
                    **kwargs
                )
            ]
            cb = ax.cax.colorbar(
                pcf[0],
                extend="neither",
                ticks=np.linspace(np.min(z[index, :]), np.max(z[index, :]), 5),
                spacing="proportional",
                shrink=0.7,
                format="%.2e",
            )
            ax.cax.tick_params(labelsize=labelsize)
            ax.set_title(title, fontsize=fontsize, pad=10)
            ax.set_xlim([x.min(), x.max()])
            ax.set_ylim([y.min(), y.max()])
            ax.set_ylabel("y/R", labelpad=15, fontsize=fontsize, rotation="horizontal")
            ax.set_xlabel("x", fontsize=fontsize)
            ax.tick_params(labelsize=fontsize)
            ax.set_aspect("equal")

        fig.set_size_inches(10, 10, True)
        fig.subplots_adjust(
            left=0, bottom=0, right=0.1, top=1, wspace=None, hspace=None
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(dirname, "scatter_" + str(index) + ".png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(
            "all",
        )


###################################################
def update_contourf(i, xs, ys, minmax, data, axis, pcfsets, kwargs):
    list_of_collections = []
    npts = 1000  # len(xs)
    xi = np.linspace(minmax[0], minmax[1], npts)
    yi = np.linspace(minmax[2], minmax[3], npts)

    for x, y, z, ax, pcfset, kw in zip(xs, ys, data, axis, pcfsets, kwargs):
        # for tp in pcfset[0].collections:
        #    tp.remove()
        # pcfset[0] = ax.tricontourf(x, y, z[i,:], **kw)
        # list_of_collections += pcfset[0].collections

        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z[i, :])
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        pcfset[0] = ax.pcolormesh(
            xi, yi, zi, shading="gouraud", **kw
        )  # shading='gouraud' ,
        list_of_collections.append(pcfset[0])

    return list_of_collections


def grid_contour_plots(
    data, nrows_ncols, titles, x, y, timeStp, dirname, ticks=8 , fontsize=15.5, labelsize=15.5 , axes_pad=1.5):
    '''
      CREATE FIGURE AND AXIS
    '''
    npts = 1000
    fig = plt.figure()
    xi = np.linspace(x.min(), x.max(), npts)
    yi = np.linspace(y.min(), y.max(), npts)

    grid = ImageGrid(
        fig,
        111,
        direction="row",
        nrows_ncols=nrows_ncols,
        label_mode="1",
        axes_pad=axes_pad,
        share_all=False,
        cbar_mode="each",
        cbar_location="right",
        cbar_size="5%",
        cbar_pad=0.0,
    )

    # CREATE ARGUMENTS DICT FOR CONTOURPLOTS
    minmax_list = []
    kwargs_list = []
    for d in data:
        minmax_list.append([np.min(d[timeStp, :]), np.max(d[timeStp, :])])
        kwargs_list.append(
            dict(cmap="coolwarm", vmin=minmax_list[-1][0], vmax=minmax_list[-1][1])
        )

    # CREATE PLOTS
    pcfsets = []
    for ax, z, kwargs, minmax, title in zip(
        grid, data, kwargs_list, minmax_list, titles
    ):
        # pcf = [ax.tricontourf(x, y, z[0,:], **kwargs)]
        # pcfsets.append(pcf)

        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z[timeStp, :])
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        pcf = [
            ax.pcolormesh(xi, yi, zi, shading="gouraud", **kwargs)
        ]  # shading='gouraud' ,
        pcfsets.append(pcf)
        cb = ax.cax.colorbar(
            pcf[0], ticks=np.linspace(minmax[0], minmax[1], ticks), format="%.3e"
        )
        ax.cax.tick_params(labelsize=labelsize)
        ax.set_title(title, fontsize=fontsize, pad=7)
        ax.set_ylabel("y", labelpad=labelsize, fontsize=fontsize, rotation="horizontal")
        ax.set_xlabel("x", fontsize=fontsize)
        ax.tick_params(labelsize=labelsize)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_aspect("equal")

    fig.set_size_inches(20, 20, True)
    fig.subplots_adjust(left=0.7, bottom=0, right=2.2, top=1, wspace=None, hspace=None)
    plt.savefig(dirname, dpi=300, bbox_inches="tight")
    plt.close(
        "all",
    )
    return fig, grid, pcfsets, kwargs_list


#########################################################
def draw_contourf(t, x, y, data, x_ref ,y_ref , dirname , ticks , fontsize ,labelsize , axes_pad):
    '''
    draw_contourf(t, x, y, data, x_ref ,y_ref , dirname)
    '''
    y = y * y_ref
    x = x * x_ref
    minmax = [x.min(), x.max(), y.min(), y.max()]

    titles = [
        "u_pinn",
        "v_pinn",
        "p_pinn",
        "u_cfd",
        "v_cfd",
        "p_cfd",
        "error_u",
        "error_v",
        "error_p",
    ]
    nrows_ncols = (3, 3)
    values = [0,  99,  50]
    for index in values:
        file = os.path.join(dirname, "tricontourf_" + str(index) + ".png")
        fig, grid, pcfsets, kwargs = grid_contour_plots(
            data, nrows_ncols, titles, x, y, index, file , ticks , fontsize=fontsize , labelsize= labelsize , axes_pad=axes_pad)


#     ani = FuncAnimation(fig, update_contourf, frames=len(t), fargs=([x]*np.prod(nrows_ncols),
#                                                                     [y]*np.prod(nrows_ncols), minmax ,
#                                                                     data,
#                                                                     [ax for ax in grid],
#                                                                     pcfsets, kwargs),
#                         interval=50, blit=False, repeat=False , save_count=sys.maxsize)

#     FFwriter = mpa.FFMpegWriter(fps=30, codec="libx264")
#     writergif = PillowWriter(fps=30)
#     ani.save(os.path.join(model.dirname,'result.mp4') , writer=FFwriter)

#########################################################################


#############################################
def save_to_matlab(
    t, x, y, ucfd, vcfd, pcfd, upinn, vpinn, ppin, fileName="result.mat"
):
    data_all = np.hstack([t, x, y, ucfd, upinn, vcfd, vpinn, pcfd, ppin])
    data_all = data_all[np.argsort(data_all[:, 0])]
    pdata_all = pd.DataFrame(
        data=data_all,
        columns=["t", "x", "y", "ucfd", "upinn", "vcfd", "vpinn", "pcfd", "ppin"],
    )
    scipy.io.savemat(fileName, {"struct": pdata_all.to_dict("list")})


def save_data_to_matlab(fileName, collm, INLETm, OUTLETm, WALLm, INITIALm):
    collm = pd.DataFrame(data=collm, columns=["t", "x", "y", "u", "v", "p"])

    INLETm = pd.DataFrame(data=INLETm, columns=["t", "x", "y", "u", "v", "p"])

    OUTLETm = pd.DataFrame(data=OUTLETm, columns=["t", "x", "y", "u", "v", "p"])

    WALLm = pd.DataFrame(data=WALLm, columns=["t", "x", "y", "u", "v", "p"])

    INITIALm = pd.DataFrame(data=INITIALm, columns=["t", "x", "y", "u", "v", "p"])

    scipy.io.savemat(
        fileName,
        {
            "struct": [
                collm.to_dict("list"),
                INLETm.to_dict("list"),
                OUTLETm.to_dict("list"),
                WALLm.to_dict("list"),
                INITIALm.to_dict("list"),
            ]
        },
    )

#############################################################################################

# def generate_Halton_sequence(low , high , n):

#     sampler = qmc.Halton(d=1, scramble=True)
#     sample = sampler.random(n=n)

#     bounds = [low,high]

#     input_tb = soboleng.draw(n)
#     result = np.floor((bounds[0] + (bounds[1] - bounds[0]) * input_tb))
#     result = [ int(i) for i in result]
#     return result


def generate_sobol_sequence(low, high, n):
    soboleng = torch.quasirandom.SobolEngine(dimension=1)
    bounds = [low, high]

    input_tb = soboleng.draw(n)
    result = np.floor((bounds[0] + (bounds[1] - bounds[0]) * input_tb))
    result = [int(i) for i in result]
    return result

#############################################################################################

#############################################################################################



################################################################


def plot_time_profile(dirname, x, y, time, u, uPINN, ylabel):
    N = 70
    timeStp = [20, 50, 99]
    idx_x = np.random.choice(x.shape[0], N, replace=False)
    x0 = x[idx_x]

    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=0.85, bottom=0.25, left=0.05, right=0.95, wspace=0.75, hspace=0.5)

    text = ["Data", "Exact", "Precition"]
    color = ["rx", "r--", "b-"]
    for j in range(len(timeStp)):
        ax = plt.subplot(gs1[0, j])
        ax.plot(x, u[timeStp[j], :], "g-.", linewidth=1, label="Exact")

        u0 = u[timeStp[j] : timeStp[j] + 1, idx_x].T
        ax.plot(x0, u0, "rx", linewidth=1, label="Data")
        ax.plot(x, uPINN[timeStp[j], :], "b.", linewidth=1, label="Precition")

        ax.set_ylabel(ylabel)
        ax.set_title("$t = %.2f$ " % (time[timeStp[j]]), fontsize=10)
        ax.set_xlim([x.min() - 0.1, x.max() + 0.1])
        ax.set_xlabel("$x$")
        if j == 1:
            plt.legend(
                loc="upper center", bbox_to_anchor=(0.3, -0.1), ncol=3, frameon=False
            )

    plt.tight_layout()
    text = "time_profile" + str(ylabel) + ".png"
    plt.savefig(os.path.join(dirname, text), dpi=300, bbox_inches="tight")
    plt.close(
        "all",
    )


#############################################################################################
def lp_error2(pred, exact, text, model, p, printTxt=False):
    num = np.mean(np.abs(pred - exact) ** p)
    result = ((num) ** (1 / p)) * 100
    if printTxt:
        model.print("%s : %5.3e %%" % (text, result))
    return result


#############################################################################################
def l1l2Erorr(text, u_preda, v_preda, p_preda, ufa, vfa, pfa, model):
    #########################################
    model.print("\n  L2 ERROR:")
    lp_error2(u_preda, ufa, (text + " U velocity "), model, 2, True)
    lp_error2(v_preda, vfa, (text + " V velocity "), model, 2, True)
    lp_error2(p_preda, pfa, (text + " P Pressure "), model, 2, True)

    model.print("\n  l1 error")

    lp_error2(u_preda, ufa, (text + " U velocity "), model, 1, True)
    lp_error2(v_preda, vfa, (text + " V velocity "), model, 1, True)
    lp_error2(p_preda, pfa, (text + " P Pressure "), model, 1, True)

#############################################################################################
def grid_contour_plots_regular(data, nrows_ncols, titles, x, y, timeStp , dirname , ticks = 7 , fontsize=14.5, labelsize=14.5 , axes_pad = 1.5 , local=True):
    
    # CREATE FIGURE AND AXIS
    fig = plt.figure()
    
    grid = ImageGrid(fig, 111, direction="row", nrows_ncols=nrows_ncols, 
                     label_mode="1", axes_pad=axes_pad, share_all=False, 
                     cbar_mode="each", cbar_location="right", 
                     cbar_size="5%", cbar_pad=0.0)

    # CREATE ARGUMENTS DICT FOR CONTOURPLOTS
    minmax_list = []
    kwargs_list = []
    for d in data:
        if(local):
            minmax_list.append([np.min(d), np.max(d)])
        else:
            minmax_list.append([np.min(d), np.max(d)])

        kwargs_list.append(dict(levels=np.linspace(minmax_list[-1][0],minmax_list[-1][1], 60),
            cmap="coolwarm", vmin=minmax_list[-1][0], vmax=minmax_list[-1][1]))

    # CREATE PLOTS
    pcfsets = []
    for ax, z, kwargs, minmax, title in zip(grid, data, kwargs_list, minmax_list, titles):

        #pcf = [ax.tricontourf(x, y, z[0,:], **kwargs)]
        #pcfsets.append(pcf)
        # if (timeStp == 0):
            #  print( z[timeStp,:,:])
        pcf = [ax.contourf(x, y, z[timeStp,:,:], **kwargs)]
        pcfsets.append(pcf)
        cb = ax.cax.colorbar(pcf[0], ticks=np.linspace(minmax[0],minmax[1],ticks),  format='%.3e')
        ax.cax.tick_params(labelsize=labelsize)
        ax.set_title(title, fontsize=fontsize, pad=7)
        ax.set_ylabel("y", labelpad=labelsize, fontsize=fontsize, rotation="horizontal")
        ax.set_xlabel("x", fontsize=fontsize)
        ax.tick_params(labelsize=labelsize)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_aspect("equal")
        
    fig.set_size_inches(20,20,True)
    fig.subplots_adjust(left=0.7, bottom=0, right=2.2, top=1, wspace=None, hspace=None)
    plt.tight_layout()
    plt.savefig(dirname, dpi=300 , bbox_inches='tight')
    plt.close("all" , )
    return fig, grid, pcfsets, kwargs_list



