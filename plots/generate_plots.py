import argparse
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import inspect
import pickle
from tqdm import tqdm
import os

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['text.usetex'] = True

COLORS = [plt.cm.Set3(i) for i in range(20)]
OUT_DIR = "./plots/plots/"
SEMANTIC_PRESERVING_TRANSFORMATIONS = ["no_transformation", "tf_1", "tf_2", "tf_3", "tf_4", "tf_5", "tf_6", "tf_7", "tf_8", "tf_9", "tf_10", "tf_11"]
DPI = 300
FIG_HEIGHT = 3.7
FIG_WIDTH = 6
X_TICK_LABELS_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
MEDIAN_LINE_COLOR = "black"
DATASET_TO_METRIC = {
    "CodeXGLUE" : "accuracy",
    "VulDeePecker" : "f1"
}

mpl.rcParams['font.size'] = LEGEND_FONT_SIZE

np.random.seed(42)

def load_results():
    
    results = dict()
    
    results["CodeXGLUE"] = dict()
    results["VulDeePecker"] = dict()
    
    results["CodeXGLUE"]["VulBERTa"]  = dict()
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        results_file_name = './results/CodeXGLUE-VulBERTa-{}.pkl'.format(transformation)
        if os.path.isfile(results_file_name):
                with open(results_file_name, 'rb') as handle:
                        results["CodeXGLUE"]["VulBERTa"][transformation] = pickle.load(handle)
                        
    results_file_name = './results/CodeXGLUE-VulBERTa-AdversarialTraining.pkl'
    if os.path.isfile(results_file_name):
            with open(results_file_name, 'rb') as handle:
                    results["CodeXGLUE"]["VulBERTa"]["ADV"] = pickle.load(handle)
                        
    results["VulDeePecker"]["VulBERTa"] = dict()
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        results_file_name = './results/VulDeePecker-VulBERTa-{}.pkl'.format(transformation)
        if os.path.isfile(results_file_name):
                with open(results_file_name, 'rb') as handle:
                        results["VulDeePecker"]["VulBERTa"][transformation] = pickle.load(handle)
                        
    results["CodeXGLUE"]["CoTexT"] = dict()
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        results_file_name = './results/CodeXGLUE-CoTexT-{}.pkl'.format(transformation)
        if os.path.isfile(results_file_name):
                with open(results_file_name, 'rb') as handle:
                        results["CodeXGLUE"]["CoTexT"][transformation] = pickle.load(handle)
    
    results_file_name = './results/CodeXGLUE-CoTexT-AdversarialTraining.pkl'
    if os.path.isfile(results_file_name):
            with open(results_file_name, 'rb') as handle:
                    results["CodeXGLUE"]["CoTexT"]["ADV"] = pickle.load(handle)
    
    results["VulDeePecker"]["CoTexT"]= dict()
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        results_file_name = './results/VulDeePecker-CoTexT-{}.pkl'.format(transformation)
        if os.path.isfile(results_file_name):
                with open(results_file_name, 'rb') as handle:
                        results["VulDeePecker"]["CoTexT"][transformation] = pickle.load(handle)
                        
    results["CodeXGLUE"]["PLBart"] = dict()
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        results_file_name = './results/CodeXGLUE-PLBart-{}.pkl'.format(transformation)
        if os.path.isfile(results_file_name):
                with open(results_file_name, 'rb') as handle:
                        results["CodeXGLUE"]["PLBart"][transformation] = pickle.load(handle)
                        
    results["VulDeePecker"]["PLBart"] = dict()
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        results_file_name = './results/VulDeePecker-PLBart-{}.pkl'.format(transformation)
        if os.path.isfile(results_file_name):
                with open(results_file_name, 'rb') as handle:
                        results["VulDeePecker"]["PLBart"][transformation] = pickle.load(handle)
                        
    results_file_name = './results/CodeXGLUE-PLBart-AdversarialTraining.pkl'
    if os.path.isfile(results_file_name):
            with open(results_file_name, 'rb') as handle:
                    results["CodeXGLUE"]["PLBart"]["ADV"] = pickle.load(handle)
                        
    # random guessing 
    results_file_name = './results/CodeXGLUE-VulBERTa-RG.pkl'
    with open(results_file_name, 'rb') as handle:
        results["CodeXGLUE"]["VulBERTa"]["RG"] = pickle.load(handle)
                        
                
    return results

def parse_args():
    parser=argparse.ArgumentParser(description="Script to generate all plots for paper.")
    args=parser.parse_args()
    return args

def reduce(array_to_reduce):
    return np.max(np.array(array_to_reduce))

def get_score_list(data, metric = "accuracy"):
        
    accuracies = []
    for epoch in range(max(data.keys()) + 1):
        accuracies.append(data[epoch]["test/{}".format(metric)])
        
    return accuracies
                
    
def fig_rq1_boxplot():
    
    
    results = load_results()
    
    train_no_trafo_test_trafo_vulberta = []
    train_trafo_test_trafo_vulberta = []
    train_no_trafo_test_trafo_cotext = []
    train_trafo_test_trafo_cotext = []
    train_no_trafo_test_trafo_plbart = []
    train_trafo_test_trafo_plbart = []
    
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        if transformation != "no_transformation":
            train_no_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"][transformation])))
            train_no_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"]["no_transformation"][transformation])))
            train_no_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"]["no_transformation"][transformation])))
            
            train_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"][transformation][transformation])))
            train_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"][transformation][transformation])))
            if transformation in results["CodeXGLUE"]["PLBart"].keys():
                train_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"][transformation][transformation])))


    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))

    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["CoTexT"]["no_transformation"]["no_transformation"]))
    ax.axhline(clean_accuracy, label="CoTexT - Train: Standard, Test: Standard", color=COLORS[3])

    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"]["no_transformation"]))
    ax.axhline(clean_accuracy, label="VulBERTa - Train: Standard, Test: Standard", color=COLORS[4])
    
    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["PLBart"]["no_transformation"]["no_transformation"]))
    ax.axhline(clean_accuracy, label="PLBart - Train: Standard, Test: Standard", color="olivedrab")

    ax.set_yticks(np.arange(0,1,0.01))

    #ax.set_xlim([0, 11])
    ax.set_ylim([0.555, 0.72])

    ax.set_ylabel("test set accuracy")

    handles = []
    labels = []

    bp = ax.boxplot(x=[train_no_trafo_test_trafo_cotext, train_trafo_test_trafo_cotext, train_no_trafo_test_trafo_vulberta, train_trafo_test_trafo_vulberta, train_no_trafo_test_trafo_plbart, train_trafo_test_trafo_plbart],  # sequence of arrays
    positions=[0.9, 1.1, 1.9, 2.1, 2.9, 3.1],   # where to put these arrays
    widths=(0.13, 0.13, 0.13, 0.13, 0.13, 0.13),
    patch_artist=True, sym="")
    
    for median in bp['medians']:
        median.set_color(MEDIAN_LINE_COLOR)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["CoTexT","VulBERTa", "PLBart"],rotation=0, fontsize=X_TICK_LABELS_FONT_SIZE)

    patch_colors = [COLORS[0], COLORS[2], COLORS[0], COLORS[2], COLORS[0], COLORS[2]]
    for patch, color in zip(bp['boxes'], patch_colors):
            patch.set_facecolor(color)

            
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color=COLORS[3]))
    labels.append("CoTexT: $s[f[Tr], Te]$")
    handles.append(Line2D([0], [0], color=COLORS[4]))
    labels.append("VulBERTa: $s[f[Tr], Te]$")
    handles.append(Line2D([0], [0], color="olivedrab"))
    labels.append("PLBart: $s[f[Tr], Te]$")
    handles.append(Line2D([0], [0], color=COLORS[0], lw=0.0, marker="s", markersize=8))
    labels.append("$s[f[Tr], Te_k]$ $\\forall$ $t_k \\in T$")
    handles.append(Line2D([0], [0], color=COLORS[2], lw=0.0, marker="s", markersize=8))
    labels.append("$s[f[Tr_k], Te_k]$ $\\forall$ $t_k \\in T$")

    ax.legend(ncol=2, handles=handles, labels=labels,loc='upper left', frameon=False)

    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")
    
    
def fig_rq1_lineplot():
    
    results = load_results()

    x = np.arange(1,11,1)

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.set_xticks(x)
    ax.set_yticks(np.arange(0,1,0.02))

    #ax.set_xlim([0, 11])
    ax.set_ylim([0.465, 0.69])

    ax.set_xlabel("training epoch")
    ax.set_ylabel("test set accuracy")

    #ax.yaxis.grid(color='gray')

    handles = []
    labels = []
    
    pair = ("tf_00", "tf_00")
    label = "$s[f[Tr], Te]$"
    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"]["no_transformation"]), marker='o', color = COLORS[3])
    handles.append(p[0])
    labels.append(label)
    
    pair = ("tf_00", "tf_41")
    label =  "$s[f[Tr], Te_{10}]$"
    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"]["tf_10"]), marker='o', color = COLORS[0])
    handles.append(p[0])
    labels.append(label)
    
    pair = ("tf_41", "tf_41")
    label =  "$s[f[Tr_{10}], Te_{10}]$"
    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["tf_10"]["tf_10"]), marker='o', color = COLORS[2])
    handles.append(p[0])
    labels.append(label)
    
    pair = ("rg", "tf_00")
    label =  "Random Guessing"
    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["RG"]["no_transformation"]), marker='o', color = "darkgray")
    handles.append(p[0])
    labels.append(label)
            
            
    ax.legend(ncol=2, handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(0,1), frameon=False)

    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")
    
def fig_rq2_boxplot():
    
    results = load_results()
    
    train_no_trafo_test_trafo_vulberta = []
    train_trafo_test_trafo_vulberta = []
    train_other_trafo_test_trafo_vulberta = []
    train_no_trafo_test_trafo_cotext = []
    train_trafo_test_trafo_cotext = []
    train_other_trafo_test_trafo_cotext = []
    train_no_trafo_test_trafo_plbart = []
    train_trafo_test_trafo_plbart = []
    train_other_trafo_test_trafo_plbart = []
    
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        if transformation != "no_transformation":
            train_no_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"][transformation])))
            train_no_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"]["no_transformation"][transformation])))
            train_no_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"]["no_transformation"][transformation])))
            
            train_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"][transformation][transformation])))
            train_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"][transformation][transformation])))
            
            if transformation in results["CodeXGLUE"]["PLBart"].keys():
                train_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"][transformation][transformation])))
                
            for other_trafo in SEMANTIC_PRESERVING_TRANSFORMATIONS:
                if transformation != other_trafo and other_trafo != "tf_11":
                    train_other_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"][other_trafo][transformation])))
                    train_other_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"][other_trafo][transformation])))
                    
                    if other_trafo in results["CodeXGLUE"]["PLBart"].keys():
                        train_other_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"][other_trafo][transformation])))
                        

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))


    # clean_accuracy = reduce(df_cotext["train: tf_00 test: tf_00 - test/accuracy"].to_list())
    # ax.axhline(clean_accuracy, label="CoTexT - Train: Standard, Test: Standard", color="pink")

    # clean_accuracy = reduce(df_vulberta["train: tf_00 test: tf_00 - test/accuracy"].to_list())
    # ax.axhline(clean_accuracy, label="VulBERTa - Train: Standard, Test: Standard", color="lightblue")

    ax.set_yticks(np.arange(0,1,0.01))

    #ax.set_xlim([0, 11])
    ax.set_ylim([0.555, 0.68])

    ax.set_ylabel("test set accuracy")

    handles = []
    labels = []

    bp = ax.boxplot(x=[train_no_trafo_test_trafo_cotext, train_trafo_test_trafo_cotext, train_other_trafo_test_trafo_cotext, train_no_trafo_test_trafo_vulberta, train_trafo_test_trafo_vulberta, train_other_trafo_test_trafo_vulberta, train_no_trafo_test_trafo_plbart, train_trafo_test_trafo_plbart, train_other_trafo_test_trafo_plbart],  # sequence of arrays
    positions=[0.8, 1.0, 1.2, 1.8, 2.0, 2.2, 2.8, 3.0, 3.2],   # where to put these arrays
    widths=(0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13),
    patch_artist=True, sym="")
    
    for median in bp['medians']:
        median.set_color(MEDIAN_LINE_COLOR)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["CoTexT","VulBERTa", "PLBart"],rotation=0, fontsize=X_TICK_LABELS_FONT_SIZE)

    patch_colors = [COLORS[0], COLORS[2], COLORS[1], COLORS[0], COLORS[2], COLORS[1], COLORS[0], COLORS[2], COLORS[1]]
    for patch, color in zip(bp['boxes'], patch_colors):
            patch.set_facecolor(color)

            
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color=COLORS[0], lw=0.0, marker="s", markersize=8))
    labels.append("$s[f[Tr], Te_k]$ $\\forall$ $t_k \\in T$")
    handles.append(Line2D([0], [0], color=COLORS[2], lw=0.0, marker="s", markersize=8))
    labels.append("$s[f[Tr_k], Te_k]$ $\\forall$ $t_k \\in T$")
    handles.append(Line2D([0], [0], color=COLORS[1], lw=0.0, marker="s", markersize=8))
    labels.append("$s[f[Tr_k], Te_j]$ $\\forall$ $t_k \\in T$  $\\forall$ $t_{j \\neq k} \\in T$")
    # handles.append(Line2D([0], [0], color="pink"))
    # labels.append("CoTexT - Train: Standard, Test: Standard")
    # handles.append(Line2D([0], [0], color="lightblue"))
    # labels.append("VulBERTa - Train: Standard, Test: Standard")

    ax.legend(handles=handles, labels=labels,loc='lower left', frameon=False)
    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")
    
def fig_naturalness():
    
    naturalness = dict()
    
    for trafo in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        results_file_name = './results/naturalness_codexglue_test_{}.pkl'.format(trafo)
        if os.path.isfile(results_file_name):
                with open(results_file_name, 'rb') as handle:
                        naturalness[trafo] = pickle.load(handle)
                        

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))


    # clean_accuracy = reduce(df_cotext["train: tf_00 test: tf_00 - test/accuracy"].to_list())
    # ax.axhline(clean_accuracy, label="CoTexT - Train: Standard, Test: Standard", color="pink")

    # clean_accuracy = reduce(df_vulberta["train: tf_00 test: tf_00 - test/accuracy"].to_list())
    # ax.axhline(clean_accuracy, label="VulBERTa - Train: Standard, Test: Standard", color="lightblue")

    ax.set_yticks(np.arange(0,10,0.5))

    #ax.set_xlim([0, 11])
    ax.set_ylim([2, 7])

    ax.set_ylabel("cross entropy")

    handles = []
    labels = []

    boxplot_data = []
    
    for trafo in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        boxplot_data.append(naturalness[trafo])
    bp = ax.boxplot(x=boxplot_data,  # sequence of arrays
    positions=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],   # where to put these arrays
    widths=(0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26),
    patch_artist=True, sym="")
    
    for median in bp['medians']:
        median.set_color(MEDIAN_LINE_COLOR)
        
    x_tick_labels = ["None", "$t_{1}$", "$t_{2}$", "$t_{3}$", "$t_{4}$", "$t_{5}$", "$t_{6}$", "$t_{7}$", "$t_{8}$", "$t_{9}$", "$t_{10}$", "$t_{11}$"]

    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=X_TICK_LABELS_FONT_SIZE)

    patch_colors = [COLORS[0], COLORS[2], COLORS[2], COLORS[2], COLORS[2], COLORS[2], COLORS[2], COLORS[2], COLORS[2], COLORS[2], COLORS[2], COLORS[2]]
    for patch, color in zip(bp['boxes'], patch_colors):
            patch.set_facecolor(color)

            
    from matplotlib.lines import Line2D
    # handles.append(Line2D([0], [0], color="pink"))
    # labels.append("CoTexT - Train: Standard, Test: Standard")
    # handles.append(Line2D([0], [0], color="lightblue"))
    # labels.append("VulBERTa - Train: Standard, Test: Standard")
    
    clean_accuracy = np.mean(np.array(naturalness["no_transformation"]))
    ax.axhline(clean_accuracy, label="Mean cross entropy clean testing set", color=COLORS[0])

    ax.legend(handles=handles, labels=labels,loc='upper left', frameon=False)
    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")

def fig_rq2_boxplot_vuldeepecker():
    
    results = load_results()
    
    train_no_trafo_test_trafo_vulberta = []
    train_trafo_test_trafo_vulberta = []
    train_other_trafo_test_trafo_vulberta = []
    train_no_trafo_test_trafo_cotext = []
    train_trafo_test_trafo_cotext = []
    train_other_trafo_test_trafo_cotext = []
    train_no_trafo_test_trafo_plbart = []
    train_trafo_test_trafo_plbart = []
    train_other_trafo_test_trafo_plbart = []
    
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        if transformation != "no_transformation" and transformation != "tf_9":
            train_no_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["VulDeePecker"]["VulBERTa"]["no_transformation"][transformation], metric=DATASET_TO_METRIC["VulDeePecker"])))
            train_no_trafo_test_trafo_cotext.append(reduce(get_score_list(results["VulDeePecker"]["CoTexT"]["no_transformation"][transformation], metric=DATASET_TO_METRIC["VulDeePecker"])))
            train_no_trafo_test_trafo_plbart.append(reduce(get_score_list(results["VulDeePecker"]["PLBart"]["no_transformation"][transformation], metric=DATASET_TO_METRIC["VulDeePecker"])))
            
            train_trafo_test_trafo_cotext.append(reduce(get_score_list(results["VulDeePecker"]["CoTexT"][transformation][transformation], metric=DATASET_TO_METRIC["VulDeePecker"])))
            train_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["VulDeePecker"]["VulBERTa"][transformation][transformation], metric=DATASET_TO_METRIC["VulDeePecker"])))
            
            if transformation in results["VulDeePecker"]["PLBart"].keys():
                train_trafo_test_trafo_plbart.append(reduce(get_score_list(results["VulDeePecker"]["PLBart"][transformation][transformation], metric=DATASET_TO_METRIC["VulDeePecker"])))
                
            for other_trafo in SEMANTIC_PRESERVING_TRANSFORMATIONS:
                if transformation != other_trafo and transformation != "tf_11" and other_trafo != "no_transformation" and other_trafo != "tf_9":
                    train_other_trafo_test_trafo_cotext.append(reduce(get_score_list(results["VulDeePecker"]["CoTexT"][other_trafo][transformation], metric=DATASET_TO_METRIC["VulDeePecker"])))
                    train_other_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["VulDeePecker"]["VulBERTa"][other_trafo][transformation], metric=DATASET_TO_METRIC["VulDeePecker"])))
                    
                    if other_trafo in results["VulDeePecker"]["PLBart"].keys():
                        train_other_trafo_test_trafo_plbart.append(reduce(get_score_list(results["VulDeePecker"]["PLBart"][other_trafo][transformation], metric=DATASET_TO_METRIC["VulDeePecker"])))
                        

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))


    # clean_accuracy = reduce(df_cotext["train: tf_00 test: tf_00 - test/accuracy"].to_list())
    # ax.axhline(clean_accuracy, label="CoTexT - Train: Standard, Test: Standard", color="pink")

    # clean_accuracy = reduce(df_vulberta["train: tf_00 test: tf_00 - test/accuracy"].to_list())
    # ax.axhline(clean_accuracy, label="VulBERTa - Train: Standard, Test: Standard", color="lightblue")

    ax.set_yticks(np.arange(0,1,0.01))

    #ax.set_xlim([0, 11])
    ax.set_ylim([0.735, 0.885])

    ax.set_ylabel("test set F1-score")

    handles = []
    labels = []

    bp = ax.boxplot(x=[train_no_trafo_test_trafo_cotext, train_trafo_test_trafo_cotext, train_other_trafo_test_trafo_cotext, train_no_trafo_test_trafo_vulberta, train_trafo_test_trafo_vulberta, train_other_trafo_test_trafo_vulberta, train_no_trafo_test_trafo_plbart, train_trafo_test_trafo_plbart, train_other_trafo_test_trafo_plbart],  # sequence of arrays
    positions=[0.8, 1.0, 1.2, 1.8, 2.0, 2.2, 2.8, 3.0, 3.2],   # where to put these arrays
    widths=(0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13),
    patch_artist=True, sym="")
    
    for median in bp['medians']:
        median.set_color(MEDIAN_LINE_COLOR)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["CoTexT","VulBERTa", "PLBart"],rotation=0, fontsize=X_TICK_LABELS_FONT_SIZE)

    patch_colors = [COLORS[0], COLORS[2], COLORS[1], COLORS[0], COLORS[2], COLORS[1], COLORS[0], COLORS[2], COLORS[1]]
    for patch, color in zip(bp['boxes'], patch_colors):
            patch.set_facecolor(color)

            
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color=COLORS[0], lw=0.0, marker="s", markersize=8))
    labels.append("$s[f[Tr], Te_k]$ $\\forall$ $t_k \\in T$")
    handles.append(Line2D([0], [0], color=COLORS[2], lw=0.0, marker="s", markersize=8))
    labels.append("$s[f[Tr_k], Te_k]$ $\\forall$ $t_k \\in T$")
    handles.append(Line2D([0], [0], color=COLORS[1], lw=0.0, marker="s", markersize=8))
    labels.append("$s[f[Tr_k], Te_j]$ $\\forall$ $t_k \\in T$  $\\forall$ $t_{j \\neq k} \\in T$")
    # handles.append(Line2D([0], [0], color="pink"))
    # labels.append("CoTexT - Train: Standard, Test: Standard")
    # handles.append(Line2D([0], [0], color="lightblue"))
    # labels.append("VulBERTa - Train: Standard, Test: Standard")

    ax.legend(handles=handles, labels=labels,loc='lower left', frameon=False)
    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")
    
def fig_rq2_boxplot_damp():
    
    results = load_results()
    
    train_no_trafo_test_trafo_vulberta = []
    train_trafo_test_trafo_vulberta = []
    train_ADV_test_trafo_vulberta = []
    train_no_trafo_test_trafo_cotext = []
    train_trafo_test_trafo_cotext = []
    train_ADV_test_trafo_cotext = []
    train_no_trafo_test_trafo_plbart = []
    train_trafo_test_trafo_plbart = []
    train_ADV_test_trafo_plbart = []
    
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        if transformation != "no_transformation":
            train_no_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"][transformation])))
            train_no_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"]["no_transformation"][transformation])))
            train_no_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"]["no_transformation"][transformation])))
            
            train_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"][transformation][transformation])))
            train_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"][transformation][transformation])))
            
            if transformation in results["CodeXGLUE"]["PLBart"].keys():
                train_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"][transformation][transformation])))
                
            train_ADV_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"]["ADV"][transformation])))
            train_ADV_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"]["ADV"][transformation])))
            train_ADV_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"]["ADV"][transformation])))
                

    

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))


    # clean_accuracy = reduce(df_cotext["train: tf_00 test: tf_00 - test/accuracy"].to_list())
    # ax.axhline(clean_accuracy, label="CoTexT - Train: Standard, Test: Standard", color="pink")

    # clean_accuracy = reduce(df_vulberta["train: tf_00 test: tf_00 - test/accuracy"].to_list())
    # ax.axhline(clean_accuracy, label="VulBERTa - Train: Standard, Test: Standard", color="lightblue")

    ax.set_yticks(np.arange(0,1,0.01))

    #ax.set_xlim([0, 11])
    ax.set_ylim([0.535, 0.68])

    ax.set_ylabel("test set accuracy")

    handles = []
    labels = []

    bp = ax.boxplot(x=[train_no_trafo_test_trafo_cotext, train_trafo_test_trafo_cotext, train_ADV_test_trafo_cotext, train_no_trafo_test_trafo_vulberta, train_trafo_test_trafo_vulberta, train_ADV_test_trafo_vulberta, train_no_trafo_test_trafo_plbart, train_trafo_test_trafo_plbart, train_ADV_test_trafo_plbart],  # sequence of arrays
    positions=[0.8, 1.0, 1.2, 1.8, 2.0, 2.2, 2.8, 3.0, 3.2],   # where to put these arrays
    widths=(0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13),
    patch_artist=True, sym="")
    
    for median in bp['medians']:
        median.set_color(MEDIAN_LINE_COLOR)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["CoTexT","VulBERTa", "PLBart"],rotation=0, fontsize=X_TICK_LABELS_FONT_SIZE)

    patch_colors = [COLORS[0], COLORS[2], COLORS[7], COLORS[0], COLORS[2], COLORS[7], COLORS[0], COLORS[2], COLORS[7]]
    for patch, color in zip(bp['boxes'], patch_colors):
            patch.set_facecolor(color)

            
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color=COLORS[0], lw=0.0, marker="s", markersize=8))
    labels.append("$s[f[Tr], Te_k]$ $\\forall$ $t_k \\in T$")
    handles.append(Line2D([0], [0], color=COLORS[2], lw=0.0, marker="s", markersize=8))
    labels.append("$s[f[Tr_k], Te_k]$ $\\forall$ $t_k \\in T$")
    handles.append(Line2D([0], [0], color=COLORS[7], lw=0.0, marker="s", markersize=8))
    labels.append("$s[f[Tr_{ADV}], Te_k]$ $\\forall$ $t_{k} \\in T$")
    # handles.append(Line2D([0], [0], color="pink"))
    # labels.append("CoTexT - Train: Standard, Test: Standard")
    # handles.append(Line2D([0], [0], color="lightblue"))
    # labels.append("VulBERTa - Train: Standard, Test: Standard")

    ax.legend(ncol=2, handles=handles, labels=labels,loc='lower left', frameon=False)
    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")
    
def fig_rq2_lineplot():
    
    results = load_results()

    x = np.arange(1,11,1)

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.set_xticks(x)
    ax.set_yticks(np.arange(0,1,0.02))

    #ax.set_xlim([0, 11])
    ax.set_ylim([0.465, 0.72])

    ax.set_xlabel("training epoch")
    ax.set_ylabel("test set accuracy")

    #ax.yaxis.grid(color='gray')

    handles = []
    labels = []
    
    pair = ("tf_00", "tf_00")
    label = "$s[f[Tr], Te]$"
    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"]["no_transformation"]), marker='o', color = COLORS[3])
    handles.append(p[0])
    labels.append(label)
    
    pair = ("tf_00", "tf_41")
    label =  "$s[f[Tr], Te_{10}]$"
    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"]["tf_10"]), marker='o', color = COLORS[0])
    handles.append(p[0])
    labels.append(label)
    
    pair = ("tf_41", "tf_41")
    label =  "$s[f[Tr_{10}], Te_{10}]$"
    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["tf_10"]["tf_10"]), marker='o', color = COLORS[2])
    handles.append(p[0])
    labels.append(label)
    
    pair = ("rg", "tf_00")
    label =  "Random Guessing"
    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["RG"]["no_transformation"]), marker='o', color = "darkgray")
    handles.append(p[0])
    labels.append(label)
    
    for i, trafo in enumerate(SEMANTIC_PRESERVING_TRANSFORMATIONS):
            if trafo != "no_transformation" and trafo != "tf_11" and trafo != "tf_10":
                p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"][trafo]["tf_10"]), color="gray", linewidth=0.5)
                handles.append(p[0])
                            
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color="gray", lw=0.5))
    labels.append("$s[f[Tr_k], Te_{10}]$ $\\forall$ $t_{k \\neq 10} \\in T$")
            
            
    ax.legend(ncol=2, handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(0,1), frameon=False)

    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")
    
def table_rq2():
    
    results = load_results()

    datasets = ["CodeXGLUE", "VulDeePecker"]
    techniques = ["VulBERTa", "CoTexT", "PLBart"]
    
    s_Tr_Te = dict()
    s_Tr_Te_k = dict()
    s_Tr_k_Te_k = dict()
    s_Tr_j_Te_k = dict()
    
    for dataset in datasets:
        
        s_Tr_Te[dataset] = dict()
        s_Tr_Te_k[dataset] = dict()
        s_Tr_k_Te_k[dataset] = dict()
        s_Tr_j_Te_k[dataset] = dict()
        
        for technique in techniques:
    
            s_Tr_Te[dataset][technique] = reduce(get_score_list(results[dataset][technique]["no_transformation"]["no_transformation"], metric=DATASET_TO_METRIC[dataset]))
            
            s_Tr_Te_k[dataset][technique] = []
            
            s_Tr_k_Te_k[dataset][technique] = []
            
            s_Tr_j_Te_k[dataset][technique] = []
            
    
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        if transformation != "no_transformation":
            
            for dataset in datasets:
                for technique in techniques:
                    if transformation in results[dataset][technique]["no_transformation"].keys():
                        s_Tr_Te_k[dataset][technique].append(reduce(get_score_list(results[dataset][technique]["no_transformation"][transformation], metric=DATASET_TO_METRIC[dataset])) - s_Tr_Te[dataset][technique])
                    if transformation in results[dataset][technique].keys():
                        s_Tr_k_Te_k[dataset][technique].append(reduce(get_score_list(results[dataset][technique][transformation][transformation], metric=DATASET_TO_METRIC[dataset])) - s_Tr_Te[dataset][technique])

            for other_trafo in SEMANTIC_PRESERVING_TRANSFORMATIONS:
                if other_trafo != transformation and other_trafo != "tf_11":
                    for dataset in datasets:
                        for technique in techniques:
                            if other_trafo in results[dataset][technique].keys() and transformation in results[dataset][technique][other_trafo].keys():
                                s_Tr_j_Te_k[dataset][technique].append(reduce(get_score_list(results[dataset][technique][other_trafo][transformation], metric=DATASET_TO_METRIC[dataset])) - s_Tr_Te[dataset][technique])

                    
            
    
    figname = inspect.stack()[0][3]
    with open("{}{}.txt".format(OUT_DIR, figname), "w") as file:
        lines = []
        average_restorations = dict()
        average_further_drops = dict()
        for dataset in datasets:
            
            scores = []
            average_restorations[dataset] = []
            average_further_drops[dataset] = []
            
            for technique in techniques:
                
                scores.append([np.array(s_Tr_Te[dataset][technique]).mean(), 
                            np.array(s_Tr_Te_k[dataset][technique]).mean(), 
                            np.array(s_Tr_k_Te_k[dataset][technique]).mean(), 
                            np.array(s_Tr_j_Te_k[dataset][technique]).mean()])
                
                lines.append("{} & {} & {:.3f} & {:.3f} $\\downarrow$ & {:.3f} $\\uparrow$ & {:.3f} $\\downarrow$ \\\\ \n".format(dataset, 
                                                                                                technique, 
                                                                                                round(np.array(s_Tr_Te[dataset][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_Te_k[dataset][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_k_Te_k[dataset][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_j_Te_k[dataset][technique]).mean(), 3)
                             ))
                average_restorations[dataset].append(1 - (round(np.array(s_Tr_k_Te_k[dataset][technique]).mean(), 3) / round(np.array(s_Tr_Te_k[dataset][technique]).mean(), 3)))
                average_further_drops[dataset].append((round(np.array(s_Tr_j_Te_k[dataset][technique]).mean(), 3) / round(np.array(s_Tr_Te_k[dataset][technique]).mean(), 3)) - 1)

            lines[-1] = lines[-1][:-2]
            lines[-1] = lines[-1] + " \\hline\n"
            scores = np.array(scores)   
            lines.append(" &  &  & \\textbf{{{:.3f}}} $\\downarrow$ & \\textbf{{{:.3f}}} $\\uparrow$ & \\textbf{{{:.3f}}} $\\downarrow$ \\\\ \n".format(
                                                                                            round(scores[:, 1].mean(), 3),
                                                                                            round(scores[:, 2].mean(), 3),
                                                                                            round(scores[:, 3].mean(), 3)))
            if dataset == "CodeXGLUE":
                lines[-1] = lines[-1][:-2]
                lines[-1] = lines[-1] + " \\hline\n"
                
        lines.append("\n\n\n")
        for dataset in datasets:
            lines.append("Average restoration for {}: {:.1f}\n".format(dataset, np.mean(np.array(average_restorations[dataset])) * 100))
            lines.append("Average fruther drops for {}: {:.1f}\n".format(dataset, np.mean(np.array(average_further_drops[dataset])) * 100))
            
        
        file.writelines(lines)
        
def table_rq3():
    
    results = load_results()

    dataset = "CodeXGLUE"
    techniques = ["VulBERTa", "CoTexT", "PLBart"]
    
    s_Tr_Te = dict()
    s_Tr_Te[dataset] = dict()
    s_Tr_VPTe = dict()
    s_Tr_VPTe[dataset] = dict()
    s_Tr_k_VPTe = dict()
    s_Tr_k_VPTe[dataset] = dict()
    e_Tr_k_VPTe = dict()
    e_Tr_k_VPTe[dataset] = dict()
    
    for technique in techniques:
    
        s_Tr_Te[dataset][technique] = reduce(get_score_list(results[dataset][technique]["no_transformation"]["no_transformation"]))
        s_Tr_VPTe[dataset][technique] = reduce(get_score_list(results[dataset][technique]["no_transformation"]["fixed_nonfixed"]))
        s_Tr_k_VPTe[dataset][technique] = []     
        e_Tr_k_VPTe[dataset][technique] = []
    
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        if transformation != "no_transformation":
            for technique in techniques:
                if transformation in results[dataset][technique].keys():
                    s_Tr_k_VPTe[dataset][technique].append(reduce(get_score_list(results[dataset][technique][transformation]["fixed_nonfixed"])))
                    e_Tr_k_VPTe[dataset][technique].append(reduce(get_score_list(results[dataset][technique][transformation]["fixed_nonfixed"])) - s_Tr_VPTe[dataset][technique])

                    
            
    
    figname = inspect.stack()[0][3]
    with open("{}{}.txt".format(OUT_DIR, figname), "w") as file:
        lines = []
        scores = []
        for technique in techniques:
            
            scores.append([s_Tr_Te[dataset][technique], 
                            s_Tr_VPTe[dataset][technique],
                            np.array(s_Tr_k_VPTe[dataset][technique]).mean(),
                            np.array(e_Tr_k_VPTe[dataset][technique]).mean(),
                            np.array(e_Tr_k_VPTe[dataset][technique]).mean() / np.array(s_Tr_k_VPTe[dataset][technique]).mean() * 100
                        ])
            
            lines.append("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \n".format(
                                                                                                technique, 
                                                                                                s_Tr_Te[dataset][technique], 
                                                                                                s_Tr_VPTe[dataset][technique],
                                                                                                np.array(s_Tr_k_VPTe[dataset][technique]).mean(),
                                                                                                np.array(e_Tr_k_VPTe[dataset][technique]).mean()
                                                                                                ))
        lines[-1] = lines[-1][:-2]
        lines[-1] = lines[-1] + " \\hline\n"
        scores = np.array(scores)   
        lines.append(" & \\textbf{{{:.3f}}} & \\textbf{{{:.3f}}} & \\textbf{{{:.3f}}} & \\textbf{{{:.3f}}}\\\\ \n".format(round(scores[:, 0].mean(), 3),
                                                                     round(scores[:, 1].mean(), 3),
                                                                     round(scores[:, 2].mean(), 3),
                                                                     round(scores[:, 3].mean(), 3)
                                                                     ))
        
        file.writelines(lines)
        
def additional_stats():
    
    results = load_results()

    dataset = "CodeXGLUE"
    techniques = ["VulBERTa", "CoTexT", "PLBart"]
    
    s_ADV_Tr_k = dict()
    s_ADV_Tr_k[dataset] = dict()
    
    for technique in techniques:
    
        for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
            if transformation != "no_transformation":
                s_ADV_Tr_k[dataset][technique] = reduce(get_score_list(results[dataset][technique]["ADV"][transformation])) - reduce(get_score_list(results[dataset][technique]["no_transformation"][transformation]))

                    
            
    
    figname = inspect.stack()[0][3]
    with open("{}{}.txt".format(OUT_DIR, figname), "w") as file:
        lines = []
        for technique in techniques:
            
            lines.append("Average improvement for {}: {:.3f}\n".format(technique, s_ADV_Tr_k[dataset][technique].mean()))
        
        file.writelines(lines)
            

        
        
def main(params):
    
    fig_functions = [
        fig_rq1_boxplot,
        fig_rq1_lineplot,
        fig_rq2_boxplot,
        fig_rq2_lineplot,
        fig_rq2_boxplot_vuldeepecker,
        fig_rq2_boxplot_damp,
        table_rq2,
        table_rq3,
        additional_stats,
        fig_naturalness
    ]
    
    progress_bar = tqdm(range(len(fig_functions)))
    
    for func in fig_functions:
        func()
        progress_bar.update(1)

if __name__ == '__main__':
    params=parse_args()
    main(params)