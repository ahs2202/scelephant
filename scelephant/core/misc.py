# # %% MISC %%
# # import internal modules

from . import BA
from . import biobookshelf as bk

from typing import Union, List, Literal, Dict, Callable, Set, Iterable, Tuple
import os
import pandas as pd
import numpy as np
import multiprocessing as mp
import math
import logging
from copy import deepcopy
import pickle
import time
import glob
import gzip  # to handle gzip file
import shutil  # for copying file
import base64  # for converting binary to text data (web application)
import json  # to read and write JSON file
import matplotlib.pyplot as plt
import scipy.sparse
import io
import concurrent.futures  # for multiprocessing

pd.options.mode.chained_assignment = None  # default='warn' # to disable worining

# SCElephant is currently implemented using Zarr
import zarr
import numcodecs
from bitarray import bitarray  ## binary arrays

import shelve  # for persistent database (key-value based database)

from tqdm import tqdm as progress_bar  # for progress bar


# plotly visualization functions
def pl_umap(
    adata,
    color: str,
    name_col_umap: str = "X_umap",
    x_range: Union[tuple, None] = None,
    y_range: Union[tuple, None] = None,
):
    """# 2022-11-29 23:08:58
    Plot an interactive scatter plot (using UMAP embeddings) of a given anndata using plotly express

    adata # input AnnData object
    color : str # name of the gene name or column containing barcode annotations in adata.obs dataframe.
    name_col_umap : str = 'X_umap' # name of array in adata.obsm containing UMAP embeddings
    x_range : Union[ tuple, None ] = None, y_range : Union[ tuple, None ] = None # ranges to visualize. set to None to plot all ranges

    """
    # for plotly python
    import plotly.express as px

    df = deepcopy(adata.obs)

    flag_is_gene_expression_data_being_plotted = False

    if color in df.columns.values:
        name_col_color = color  # set 'name_col_color'
    if color in adata.var.index.values:
        df["gene_expr"] = adata[:, [color]].X.toarray().ravel()
        name_col_color = "gene_expr"  # set 'name_col_color'
        flag_is_gene_expression_data_being_plotted = True  # set a flag

    # retrieve umap coordinates
    x = adata.obsm[name_col_umap][:, 0]
    y = adata.obsm[name_col_umap][:, 1]
    df["UMAP-1"] = x
    df["UMAP-2"] = y

    # retrieving data of the barcodes in the given ranges
    mask = np.ones(len(df), dtype=bool)
    if x_range is not None:
        x_min, x_max = x_range
        if x_min is not None:
            mask &= x >= x_min
        if x_max is not None:
            mask &= x <= x_max
    if y_range is not None:
        y_min, y_max = y_range
        if y_min is not None:
            mask &= y >= y_min
        if y_max is not None:
            mask &= y <= y_max
    df = df[mask]

    # draw scatter graph
    fig = px.scatter(
        df,
        x="UMAP-1",
        y="UMAP-2",
        color=name_col_color,
        hover_data=["name_dataset", "name_sample"],
        color_continuous_scale=px.colors.sequential.Reds,
        title=(
            f"gene expression of {color}"
            if flag_is_gene_expression_data_being_plotted
            else name_col_color
        ),
    )
    return fig


def scanpy_tutorial_recipe(adata):
    """# 2023-02-28 08:00:53
    a recipe for the scanpy pbmc3k tutorial
    "https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html"
    """
    import scanpy as sc

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pl.highest_expr_genes(
        adata,
        n_top=20,
    )
    # annotate the group of mitochondrial genes as 'mt'
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
    )
    sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt")
    sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(adata)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pl.pca_variance_ratio(adata, log=True)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    return adata


def SCANPY_UMAP_Plot_for_each_anno(adata, name_col: str):
    """# 2023-03-06 16:42:09
    adata, # AnnData to plot
    name_col : str # name of the column of 'obs' that contains annotations
    """
    for name_anno in adata.obs[name_col].unique():
        adata.obs[name_anno] = (
            (adata.obs[name_col].values == name_anno).astype(str).astype(object)
        )
        sc.pl.umap(adata, color=name_anno)


def Shankey_Compare_Annotations(
    l_anno_1,
    l_anno_2,
    int_min_num_entries_for_an_overlap: int = 3,
    flag_show_label: bool = True,
    font_size: int = 10,
    title: str = "",
    color="blue",
):
    """# 2023-03-05 16:32:49
    draw a Shankey diagram using Plotly for the given lists of annotations

    l_anno_1 # first list of annotations
    l_anno_2 # second list of annotations
    int_min_num_entries_for_an_overlap : int = 3 # the minmum number of entries for a link (overlaps between two annotations) to be valid.
    title : Union[ None, str ] = None # the name of the figure. if None is given, no title will be shown
    font_size : int = 10, # the font size of the title
    color = 'blue' # argument for plotly.graph_objects.Sankey
    """

    def _map(arr_anno, start_pos: int = 0):
        """
        return a dictionary for mapping annotation to its integer representation and a list of unique annotation labels
        """
        l_anno_unique = bk.LIST_COUNT(
            arr_anno, duplicate_filter=None
        ).index.values  # retrieve a list of unique annotations
        return (
            dict((e, i + start_pos) for i, e in enumerate(l_anno_unique)),
            l_anno_unique,
        )

    dict_map_1, arr_anno_unique_1 = _map(l_anno_1, start_pos=0)
    dict_map_2, arr_anno_unique_2 = _map(l_anno_2, start_pos=len(dict_map_1))
    label = (
        list(arr_anno_unique_1) + list(arr_anno_unique_2) if flag_show_label else None
    )  # compose a list of unique labels # does not show labels if 'flag_show_label' is False

    # retrieve values for drawing the diagram
    source, target, value = (
        bk.LIST_COUNT(
            np.array(
                [
                    list(dict_map_1[e] for e in l_anno_1),
                    list(dict_map_2[e] for e in l_anno_2),
                ],
                dtype=int,
            ).T,
            duplicate_filter=int_min_num_entries_for_an_overlap,
        )
        .reset_index(drop=False)
        .values.T
    )
    # compose a dataframe

    # draw a plot
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=label,
                    color=color,
                ),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )
    if title is not None:
        fig.update_layout(title_text=title, font_size=font_size)
    return fig
