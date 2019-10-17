import itertools
import sys
import numpy as np
import os

import logging
import re
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns


# plt.rc('text', usetex=True)
plt.rc('font', family='serif')


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_data(path, params):
    # Extract results for each valid (algorithm, id) pair
    ids = os.listdir(path)
    data = pd.DataFrame()
    for algo in params.keys():
        results = pd.DataFrame()
        for id in ids:
            # Results directory
            file_id = "{}/{}/{}/results".format(path, id, algo)
            if not os.path.exists(file_id):
                continue

            logger.info("Loading from {}".format(file_id))
            # Results files
            id_results = pd.DataFrame()
            for file_param in os.listdir(file_id):
                m = re.search("=(.*).result", file_param)
                if m:  # valid filename
                    param = m.group(1)
                    res = pd.read_csv(file_id + "/" + file_param, sep=' ', names=['R', 'C', 'Rd', 'Cd'])
                    res["parameter"] = param
                    id_results = id_results.append(res, sort=False)
                else:
                    continue
            # logger.info("id_results : \n{}".format(id_results))
            if id_results.empty:
                logging.warning("Could not find any result at {}".format(file_id))
                continue
            id_results["id"] = "id=" + id
            results = results.append(id_results, sort=False)
        results["algorithm"] = algo
        data = data.append(results, sort=False)
    return data


def plot_all(data, path, params):
    means = data.groupby(['algorithm', 'parameter', 'id']).mean().reset_index()
    stds = data.groupby(['algorithm', 'parameter', 'id']).std().reset_index()
    counts = data.groupby(['algorithm', 'parameter', 'id']).count().reset_index()

    means_of_means = means.groupby(['algorithm', 'parameter']).mean().reset_index()
    std_of_means = means.groupby(['algorithm', 'parameter']).std().reset_index()
    mean_of_stds = stds.groupby(['algorithm', 'parameter']).mean().reset_index()

    for show_annotation in [True, False]:
        for type_varation in ["std", "ci"]:
            for type in ["intra_seed", "extra_seed"]:
                show_std = type_varation == "std"
                show_ci = type_varation == "ci"
                means = means_of_means
                if type == "intra_seed":
                    stds = mean_of_stds
                elif type == "extra_seed":
                    stds = std_of_means

                plot_patch(type, means, stds, counts, show_annotation=show_annotation,
                           show_std=show_std, show_confidence=show_ci, x="Cd", y="Rd",
                           curves="algorithm", points="parameter",
                           params=params,
                           filename=os.path.join(path, "{}_{}_annot={}.png".format(type, type_varation,
                                                                                   show_annotation)))
                plot_patch(type, means, stds, counts, show_annotation=show_annotation,
                           show_std=show_std, show_confidence=show_ci, x="Cd", y="Rd",
                           curves="algorithm", points="parameter",
                           params=params,
                           filename=os.path.join(path, "{}_{}_annot={}.svg".format(type, type_varation,
                                                                                   show_annotation)))

            # plot_patch(means_of_means, mean_of_stds, counts, show_annotation=show_annotation,x="Cd", y="Rd", curves="algorithm", points="parameter",
            #            params=params, filename=os.path.join(path, "results_disc_extra_annot={}.png".format(show_annotation)))
            # plot_patch(means_of_means, std_of_means, counts,show_annotation=show_annotation, x="Cd", y="Rd", curves="algorithm", points="parameter",
            #            params=params, filename=os.path.join(path, "results_disc_intra_annot={}.png".format(show_annotation)))
            # plot_patch(means_of_means, mean_of_stds, counts, show_annotation=show_annotation,x="C", y="R", curves="algorithm", points="parameter",
            #            params=params, filename=os.path.join(path, "results_extra_annot={}.png".format(show_annotation)))
            # plot_patch(means_of_means, std_of_means, counts, show_annotation=show_annotation,x="C", y="R", curves="algorithm", points="parameter",
            #            params=params, filename=os.path.join(path, "results_intra_annot={}.png".format(show_annotation)))

    try:
        plot_lines(means, x='Cd', y='Rd', hue="id", style="algorithm", points="parameter",
                   filename=os.path.join(path, "results_disc_ids.png"))
    except ValueError:
        logger.warning("Too many ids to use different styles")
        plot_lines(means, x='Cd', y='Rd', hue="algorithm", points="parameter",
                   filename=os.path.join(path, "results_disc_ids.png"))


def plot_patch(type, mean, std, counts, x, y, curves, points, params, show_std=False, show_annotation=True,
               show_confidence=True, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    result = pd.concat([mean, std.add_suffix("_std"), counts.add_suffix("_count")], axis=1, sort=False)
    print(result)
    # nb_run =

    # handlers = []
    labels = []
    for group_label, data in result.groupby(curves):
        if type == "extra_seed":
            nb_run = len(pd.unique(data["id_count"]))
        elif type == "intra_seed":
            nb_run = pd.unique(data["R_count"])
            if len(nb_run) == 1:
                nb_run = nb_run[0]
            else:
                raise Exception("malformed data")
        else:
            raise Exception("wrong type : {}".format(type))
        print(type, nb_run)
        # exit()
        ax1 = data.plot.scatter(x=x, y=y, ax=ax, c=[params[group_label][0]], marker=params[group_label][1], s=30,
                                zorder=2, label=group_label)

        ax_xlabel = ax.xaxis.get_label()
        ax_ylabel = ax.yaxis.get_label()
        plt.xlabel(rename_label(ax_xlabel._text))
        plt.ylabel(rename_label(ax_ylabel._text))
        # ax1.legend(group_label)
        # lines,_ = ax1.get_legend_handles_labels()
        # handlers.append(lines)
        # print(lines)
        labels.append(group_label)
        for param, point in data.groupby(points):
            # print(point)
            # sqr_n = np.sqrt(point[x + "_count"].values)
            sqr_n = np.sqrt(nb_run)
            # print("!!!!",point[x + "_count"].values)
            if show_confidence:
                confidence_x = 1.96 * (point[x + "_std"].values / sqr_n)
                confidence_y = 1.96 * (point[y + "_std"].values / sqr_n)
                rect = patches.Rectangle((point[x].values - confidence_x, point[y].values - confidence_y),
                                         2 * confidence_x, 2 * confidence_y,
                                         linewidth=1.0,
                                         fill=True,
                                         edgecolor=(*params[group_label][0], 1),
                                         facecolor=(*params[group_label][0], 0.2), zorder=0)
                ax.add_patch(rect)
            if show_std:
                std_x = point[x + "_std"].values
                std_y = point[y + "_std"].values
                rect = patches.Rectangle((point[x].values - std_x, point[y].values - std_y),
                                         2 * std_x, 2 * std_y,
                                         linewidth=1.0,
                                         fill=True,
                                         edgecolor=(*params[group_label][0], 1),
                                         facecolor=(*params[group_label][0], 0.2), zorder=0)
                ax.add_patch(rect)
            if show_annotation:
                plt.annotate("{:.2f}".format(float(param)), (point[x].values, point[y].values))

    # plt.legend(handles=[patches.Patch(label=param[2], color=param[0],hatch=param[1]) for _, param in params.items()])
    # plt.legend(handlers,
    #            labels,
    #            scatterpoints=1,
    #            loc='lower left',
    #            ncol=3,
    #            fontsize=8)
    # plt.legends()

    if filename:
        plt.savefig(filename)
    plt.show()
    plt.close()


def plot_lines(data, x=None, y=None, filename=None, points=None, **kwargs):
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x=x, y=y, ax=ax, estimator=None, **kwargs)
    if points:
        for param, point in data.groupby(points):
            for _, row in point.iterrows():
                plt.annotate("{:.2f}".format(float(param)), (row[x], row[y]))
    if filename:
        plt.savefig(filename)
    plt.show()
    plt.close()


def rename_fields(data, algos):
    # data = data.replace("ftq_duplicate", "FTQ")
    # data = data.replace("ftq_egreedy", "FTQ")
    data = data.replace("bftq_egreedy", "BFTQ(risk-sensitive)")
    data = data.replace("bftq_duplicate", "BFTQ(risk-neutral)")
    algos["BFTQ(risk-neutral)"] = algos["bftq_duplicate"]
    algos["BFTQ(risk-sensitive)"] = algos["bftq_egreedy"]
    return data, algos


def rename_label(label):
    replacement = {"Rd": r"$\mathcal{R}^\pi$", "Cd": r"$\mathcal{C}^\pi$"}
    return replacement.get(label, label)




def main(workspace):
    palette = itertools.cycle(sns.color_palette())
    marker = itertools.cycle(('^', 'v', '*', '+', '*'))
    algos = {
        "bftq_egreedy": [next(palette), next(marker), r"bftq risk-sensitive($\beta$)"],
        "bftq_duplicate": [next(palette), next(marker), r"bftq risk-neutral($\beta$)"],
        # "ftq_duplicate": [next(palette), next(marker), r"ftq duplicate($\lambda$)"],
        # "ftq_egreedy": [next(palette), next(marker), r"ftq egreedy($\lambda$)"],
    }
    data = parse_data(workspace, algos)
    data, algos = rename_fields(data, algos)
    print(data)
    plot_all(data, workspace, algos)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("Usage: plot_data.py <path>")
    main(sys.argv[1])
