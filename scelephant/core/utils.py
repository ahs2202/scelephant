# # %% UTILs %%
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

# set logging format
logging.basicConfig(
    format="%(asctime)s [%(name)s] <%(levelname)s> (%(funcName)s) - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("SC-Elephant")

""" previosuly written for biobookshelf """


def CB_Parse_list_of_id_cell(l_id_cell, dropna=True):
    """# 2022-03-25 16:35:23
    parse a given list of id_cells into a dataframe using 'SC.CB_detect_cell_barcode_from_id_cell' function
    'dropna' : drop id_cells that does not contains cell barcodes
    """
    df = pd.DataFrame(
        list([e] + list(CB_detect_cell_barcode_from_id_cell(e)) for e in l_id_cell),
        columns=["id_cell", "CB", "id_sample_from_id_cell"],
    ).set_index("id_cell")
    return df


def CB_Build_dict_id_sample_to_set_cb(l_id_cell):
    """# 2022-03-28 22:24:30
    Build a set of cell barcode for each id_sample from the given list of id_cells
    """
    df = CB_Parse_list_of_id_cell(l_id_cell)
    dict_id_sample_to_set_cb = dict()
    for cb, id_sample in df.values:
        if id_sample not in dict_id_sample_to_set_cb:
            dict_id_sample_to_set_cb[id_sample] = set()
        dict_id_sample_to_set_cb[id_sample].add(cb)
    return dict_id_sample_to_set_cb


def CB_Match_Batch(
    l_id_cell_1, l_id_cell_2, flag_calculate_proportion_using_l_id_cell_2=True
):
    """# 2022-03-28 23:10:43
    Find matching batches between two given lists of id_cells by finding the batches sharing the largest number of cell barcodes

    'l_id_cell_1' : first list of id_cells (e.g. unannotated barcodes)
    'l_id_cell_2' : second list of id_cells (e.g. annotated barcodes)
    'flag_calculate_proportion_using_l_id_cell_2' : if True, finding matching batches using the shared proportion calculated using the cell barcodes from 'l_id_cell_2'. if False, proportion of the matching barcodes will be calculated using the cell barcodes from 'l_id_cell_1'

    return:
    df_id_cell_matched, df_sample_matched
    """
    # retrieve set of cell barcodes
    df_id_cell_1 = CB_Parse_list_of_id_cell(l_id_cell_1)
    df_id_cell_2 = CB_Parse_list_of_id_cell(l_id_cell_2)
    dict_id_sample_to_set_cb_1 = CB_Build_dict_id_sample_to_set_cb(l_id_cell_1)
    dict_id_sample_to_set_cb_2 = CB_Build_dict_id_sample_to_set_cb(l_id_cell_2)

    # Find matching id_samples of the two given list of id_cells
    # calculate the proportion of matching cell barcodes between each pair of samples from the two given list of id_cells
    l_l = []
    for id_sample_1 in dict_id_sample_to_set_cb_1:
        for id_sample_2 in dict_id_sample_to_set_cb_2:
            set_cb_1 = dict_id_sample_to_set_cb_1[id_sample_1]
            set_cb_2 = dict_id_sample_to_set_cb_2[id_sample_2]
            float_prop_matching_cb = len(set_cb_1.intersection(set_cb_2)) / (
                len(set_cb_2)
                if flag_calculate_proportion_using_l_id_cell_2
                else len(set_cb_1)
            )
            l_l.append([id_sample_1, id_sample_2, float_prop_matching_cb])
    df = pd.DataFrame(
        l_l, columns=["id_sample_1", "id_sample_2", "float_prop_matching_cb"]
    )  # search result
    df_sample_matched = (
        df.sort_values("float_prop_matching_cb", ascending=False)
        .drop_duplicates("id_sample_2", keep="first")
        .drop_duplicates("id_sample_1", keep="first")
    )  # retrieve the best matches between samples so that a unique mapping exists for every sample

    # Find matching id_cells of given two list of id_cells
    df_id_cell_1.reset_index(inplace=True, drop=False)
    df_id_cell_2.reset_index(inplace=True, drop=False)
    df_id_cell_1.rename(
        columns={"id_sample_from_id_cell": "id_sample_from_id_cell_1"}, inplace=True
    )
    df_id_cell_2.rename(
        columns={"id_sample_from_id_cell": "id_sample_from_id_cell_2"}, inplace=True
    )
    df_id_cell_1["id_sample_from_id_cell_2"] = (
        df_id_cell_1.id_sample_from_id_cell_1.apply(
            bk.Map(df_sample_matched.set_index("id_sample_1").id_sample_2.to_dict()).a2b
        )
    )
    df_id_cell_1.dropna(
        subset=["id_sample_from_id_cell_2"], inplace=True
    )  # ignore cells without matching id_sample from the '2' batch
    df_id_cell_1.set_index(["CB", "id_sample_from_id_cell_2"], inplace=True)
    df_id_cell_matched = df_id_cell_1.join(
        df_id_cell_2[~pd.isnull(df_id_cell_2.id_sample_from_id_cell_2)].set_index(
            ["CB", "id_sample_from_id_cell_2"]
        ),
        lsuffix="_1",
        rsuffix="_2",
    )  # match id_cells from two given list of id_cells
    df_id_cell_matched.reset_index(drop=False, inplace=True)
    df_id_cell_matched = df_id_cell_matched[
        [
            "id_cell_1",
            "id_cell_2",
            "CB",
            "id_sample_from_id_cell_1",
            "id_sample_from_id_cell_2",
        ]
    ]  # reorder columns

    return df_id_cell_matched, df_sample_matched


def SC_Add_metadata(adata, df_metadata, suffix: str = "", inplace: bool = True):
    """# 2023-09-08 14:20:58
    Add metadata to the given AnnData

    adata
    df_metadata
    suffix : str = '', the suffix to the columns that will be added to AnnData.obs
    inplace : bool = True #
    """
    df_matched, _ = CB_Match_Batch(adata.obs.index.values, df_metadata.index.values)

    df_metadata = df_metadata.copy()  # copy the metadata dataframe
    df_matched.dropna(subset=["id_cell_2", "id_cell_1"], inplace=True)
    df_metadata["id_cell_1"] = df_matched.set_index("id_cell_2")[
        "id_cell_1"
    ]  # map id_cell
    df_metadata.dropna(subset=["id_cell_1"], inplace=True)  # drop invalid cells
    df_metadata.set_index("id_cell_1", inplace=True)
    if not inplace:
        adata = adata.copy()  # copy the anndata
    adata.obs = adata.obs.join(df_metadata, rsuffix=suffix)  # add the metadata

    return adata  # return the anndata


def SCANPY_Detect_cell_barcode_from_cell_id(adata):
    """# 2022-03-24 20:35:22
    Detect cell barcodes from id_cell (index of adata.obs), and add new two columns to the adata.obs [ 'CB', 'id_sample_from_id_cell' ]
    """
    adata.obs = adata.obs.join(
        pd.DataFrame(
            list(
                [e] + list(CB_detect_cell_barcode_from_id_cell(e))
                for e in adata.obs.index.values
            ),
            columns=["id_cell", "CB", "id_sample_from_id_cell"],
        ).set_index("id_cell")
    )


def SCANPY_Retrieve_Markers_as_DataFrame(adata):
    """# 2022-02-15 14:40:02
    receive scanpy anndata and return a dataframe contianing marker genes

    --- return ---
    df_marker : a dataframe contianing marker genes
    """
    l_df = []
    for index_clus, name_clus in enumerate(
        adata.uns["rank_genes_groups"]["names"].dtype.names
    ):
        df = pd.DataFrame(
            dict(
                (name_col, adata.uns["rank_genes_groups"][name_col][name_clus])
                for name_col in [
                    "logfoldchanges",
                    "names",
                    "pvals",
                    "pvals_adj",
                    "scores",
                ]
            )
        )
        df["name_clus"] = name_clus
        df["index_clus"] = index_clus
        l_df.append(df)
    df_marker = pd.concat(l_df)
    return df_marker


def summarize_expression_for_each_clus(
    adata,
    name_col_cluster: str,
):
    """
    summarize expression of each cluster for the given adata
    # 2024-02-26 21:01:16
    """
    l_df = []

    def _parse_array(arr):
        if len(arr.shape) == 1:
            return arr
        else:
            return arr[0]

    for name_cluster in set(adata.obs[name_col_cluster].values):
        adata_subset = adata[adata.obs[name_col_cluster] == name_cluster]  # retrieve
        arr_num_cell_with_expr = _parse_array(
            np.array((adata_subset.X > 0).sum(axis=0))
        )
        arr_avg_expr = (
            _parse_array(np.array(adata_subset.X.sum(axis=0))) / arr_num_cell_with_expr
        )  # calculate average expression in cells expressing the gene
        arr_avg_expr[np.isnan(arr_avg_expr)] = (
            0  # when number of cells expressing is zero, set expression values as 0
        )
        arr_prop_expr = arr_num_cell_with_expr / len(adata_subset.obs)
        _df = pd.DataFrame({"avg_expr": arr_avg_expr, "prop_expr": arr_prop_expr})
        _df[name_col_cluster] = name_cluster
        _df["gene_name"] = adata_subset.var.index.values
        l_df.append(_df)
    df_gene_expr = pd.concat(l_df)
    df_gene_expr["score"] = (
        df_gene_expr.avg_expr * df_gene_expr.prop_expr
    )  # calculate the score
    return df_gene_expr


def search_uniquely_expressed_marker_genes(
    adata,
    name_col_cluster: str,
    float_max_score: float = 1000,
):
    """
    find unique markers for each clusters.
    *score is calculated as the product of the proportion expressed and the average expression values

    name_col_cluster : str, # name of the column in the 'adata.obs' containing cluster labels
    float_max_score : 1000, # 'infinite' score ratios will be replaced by this value

    # 2024-02-20 13:24:17
    """
    """
    survey proportion expressed and avg expression
    """
    df_gene_expr = summarize_expression_for_each_clus(adata, name_col_cluster)

    """
    identify marker genes uniquely expressed in a single cluster
    """
    l_l = []
    for gene_name, _df in df_gene_expr.groupby("gene_name"):
        # retrieve values
        (
            arr_avg_expr,
            arr_prop_expr,
            arr_str_int_cell_type_subclustered_temp,
            _,
            arr_score,
        ) = _df.values.T
        # sort by score
        arr_score_argsort = arr_score.argsort()
        (
            arr_avg_expr,
            arr_prop_expr,
            arr_str_int_cell_type_subclustered_temp,
            arr_score,
        ) = (
            arr_avg_expr[arr_score_argsort],
            arr_prop_expr[arr_score_argsort],
            arr_str_int_cell_type_subclustered_temp[arr_score_argsort],
            arr_score[arr_score_argsort],
        )

        avg_expr_highest, prop_expr_highest = arr_avg_expr.max(), arr_prop_expr.max()
        avg_expr_2nd, avg_expr_1st = arr_avg_expr[-2:]
        prop_expr_2nd, prop_expr_1st = arr_prop_expr[-2:]
        (
            str_int_cell_type_subclustered_temp_2nd,
            str_int_cell_type_subclustered_temp_1st,
        ) = arr_str_int_cell_type_subclustered_temp[-2:]
        score_2nd, score_1st = arr_score[-2:]
        l_l.append(
            [
                gene_name,
                str_int_cell_type_subclustered_temp_1st,
                avg_expr_1st,
                prop_expr_1st,
                score_1st,
                avg_expr_2nd,
                prop_expr_2nd,
                score_2nd,
                avg_expr_highest,
                prop_expr_highest,
            ]
        )  # add a record
    df_unique_marker_gene = pd.DataFrame(
        l_l,
        columns=[
            "gene_name",
            "str_int_cell_type_subclustered_temp_1st",
            "avg_expr_1st",
            "prop_expr_1st",
            "score_1st",
            "avg_expr_2nd",
            "prop_expr_2nd",
            "score_2nd",
            "avg_expr_highest",
            "prop_expr_highest",
        ],
    )
    arr_score_ratio = (
        df_unique_marker_gene.score_1st.values / df_unique_marker_gene.score_2nd.values
    )
    arr_score_ratio[np.isnan(arr_score_ratio)] = float_max_score
    df_unique_marker_gene["score_ratio"] = arr_score_ratio
    return df_unique_marker_gene


def identify_batch_specific_genes(
    adata,
    name_col_batch: str,
    name_col_cluster: str = None,
    l_name_cluster=None,
    min_prop_expr=0.15,
    min_score_ratio=3,
):
    """
    Identify genes that are consistently differently expressed in each sample for multiple clusters
    Note)
    The primary aim of this function is for identifying batch specific genes in an entire dataset or in a set of clusters to improve UMAP embedding/clustering without/with the help of other batch-correction algorithms.
    Empirically, batch effect can be significantly reduced (with some loss of information) simply by excluding a set of genes contributing to the batch effects, which is often sufficient for clustering analysis.
    # 2024-02-26 22:34:55
    """

    def _filter_marker_gene(df_unique_marker_gene):
        return bk.PD_Threshold(
            df_unique_marker_gene,
            prop_expr_1sta=min_prop_expr,
            score_ratioa=min_score_ratio,
        )

    if l_name_cluster is None:
        set_name_gene_to_exclude = set(
            _filter_marker_gene(
                search_uniquely_expressed_marker_genes(adata, name_col_batch)
            ).gene_name.values
        )
    else:
        l_l_name_gene = list(
            _filter_marker_gene(
                search_uniquely_expressed_marker_genes(
                    adata[adata.obs[name_col_cluster] == name_cluster].copy(),
                    name_col_batch,
                )
            ).gene_name.values
            for name_cluster in l_name_cluster
        )  # retrieve batch specific genes for each cluster
        # identify batch specific genes shared between all clusters
        set_name_gene_to_exclude = set(
            l_l_name_gene[0]
        )  # initialize 'set_name_gene_to_exclude'
        for l_name_gene in l_l_name_gene[1:]:
            set_name_gene_to_exclude = set_name_gene_to_exclude.intersection(
                l_name_gene
            )
    return set_name_gene_to_exclude


def CB_detect_cell_barcode_from_id_cell(
    id_cell, int_min_number_atgc_in_cell_barcode=16
):
    """# 2023-04-02 14:10:46
    retrieve cell_barcode from id_cell
    'int_min_number_atgc_in_cell_barcode' : number of ATGC characters in the cell barcode
    """
    int_count_atgc = 0
    int_start_appearance_of_atgc = None
    set_atgc = set("ATGC")

    def __retrieve_cell_barcode_and_id_channel_from_id_cell__(
        id_cell, int_start_appearance_of_atgc, int_count_atgc
    ):
        """__retrieve_cell_barcode_and_id_channel_from_id_cell__"""
        int_cb_start = int_start_appearance_of_atgc
        int_cb_end = int_start_appearance_of_atgc + int_count_atgc
        return [
            id_cell[int_cb_start:int_cb_end],
            id_cell[:int_cb_start] + "|" + id_cell[int_cb_end:],
        ]  # return cell_barcode, id_channel

    for index_c, c in enumerate(
        id_cell.upper()
    ):  # case-insensitive detection of cell-barcodes
        if c in set_atgc:
            if int_start_appearance_of_atgc is None:
                int_start_appearance_of_atgc = index_c
            int_count_atgc += 1
        else:
            """identify cell barcode and return the cell barcode"""
            if int_start_appearance_of_atgc is not None:
                if int_count_atgc >= int_min_number_atgc_in_cell_barcode:
                    return __retrieve_cell_barcode_and_id_channel_from_id_cell__(
                        id_cell, int_start_appearance_of_atgc, int_count_atgc
                    )
            # initialize the next search
            int_count_atgc = 0
            int_start_appearance_of_atgc = None
    """ identify cell barcode and return the cell barcode """
    if int_start_appearance_of_atgc is not None:
        if int_count_atgc >= int_min_number_atgc_in_cell_barcode:
            return __retrieve_cell_barcode_and_id_channel_from_id_cell__(
                id_cell, int_start_appearance_of_atgc, int_count_atgc
            )
    """ return None when cell_barcode was not found """
    return [None, None]


def Read_10X(path_folder_mtx_10x, verbose=False):
    """# 2021-11-24 13:00:13
    read 10x count matrix
    'path_folder_mtx_10x' : a folder containing files for 10x count matrix
    return df_mtx, df_feature
    """
    # handle inputs
    if path_folder_mtx_10x[-1] != "/":
        path_folder_mtx_10x += "/"

    # define input file directories
    path_file_bc = f"{path_folder_mtx_10x}barcodes.tsv.gz"
    path_file_feature = f"{path_folder_mtx_10x}features.tsv.gz"
    path_file_mtx = f"{path_folder_mtx_10x}matrix.mtx.gz"

    # check whether all required files are present
    if sum(
        list(
            not filesystem_operations("exists", path_folder)
            for path_folder in [path_file_bc, path_file_feature, path_file_mtx]
        )
    ):
        if verbose:
            logger.info(f"required file(s) is not present at {path_folder_mtx_10x}")

    # read mtx file as a tabular format
    df_mtx = pd.read_csv(path_file_mtx, sep=" ", comment="%")
    df_mtx.columns = ["id_row", "id_column", "read_count"]

    # read barcode and feature information
    df_bc = pd.read_csv(path_file_bc, sep="\t", header=None)
    df_bc.columns = ["barcode"]
    df_feature = pd.read_csv(path_file_feature, sep="\t", header=None)
    df_feature.columns = ["id_feature", "feature", "feature_type"]

    # mapping using 1 based coordinates (0->1 based coordinate )
    df_mtx["barcode"] = df_mtx.id_column.apply(
        bk.Map(bk.DICTIONARY_Build_from_arr(df_bc.barcode.values, index_start=1)).a2b
    )  # mapping using 1 based coordinates (0->1 based coordinate )
    df_mtx["id_feature"] = df_mtx.id_row.apply(
        bk.Map(
            bk.DICTIONARY_Build_from_arr(df_feature.id_feature.values, index_start=1)
        ).a2b
    )
    df_mtx.drop(
        columns=["id_row", "id_column"], inplace=True
    )  # drop unnecessary columns

    return df_mtx, df_feature


def Write_10X(df_mtx, df_feature, path_folder_output_mtx_10x):
    """# 2021-11-24 12:57:30
    'df_feature' should contains the following column names : [ 'id_feature', 'feature', 'feature_type' ]
    'df_mtx' should contains the following column names : [ 'id_feature', 'barcode', 'read_count' ]
    'path_folder_output_mtx_10x' : an output folder directory where the mtx_10x files will be written

    """
    import scipy.io

    df_mtx = deepcopy(df_mtx)  # create a copy of df_mtx before modification

    # create an output folder
    filesystem_operations("mkdir", path_folder_output_mtx_10x, exist_ok=True)

    """ save barcode file """
    # retrieve list of barcodes
    arr_barcode = bk.LIST_COUNT(df_mtx.barcode, duplicate_filter=None).index.values
    pd.DataFrame(arr_barcode).to_csv(
        f"{path_folder_output_mtx_10x}barcodes.tsv.gz",
        sep="\t",
        index=False,
        header=False,
    )

    """ save feature file """
    # compose a feature dataframe
    df_feature[["id_feature", "feature", "feature_type"]].to_csv(
        f"{path_folder_output_mtx_10x}features.tsv.gz",
        sep="\t",
        index=False,
        header=False,
    )  # save as a file
    # retrieve list of features
    arr_id_feature = df_feature.id_feature.values

    """ save matrix file """
    # convert feature and barcode to integer indices
    df_mtx.id_feature = df_mtx.id_feature.apply(
        bk.Map(
            bk.DICTIONARY_Build_from_arr(arr_id_feature, order_index_entry=False)
        ).a2b
    )  # 0-based coordinates
    df_mtx.barcode = df_mtx.barcode.apply(
        bk.Map(bk.DICTIONARY_Build_from_arr(arr_barcode, order_index_entry=False)).a2b
    )  # 0-based coordinates
    # save count matrix as a gzipped matrix market format
    row, col, data = df_mtx[["id_feature", "barcode", "read_count"]].values.T
    sm = scipy.sparse.coo_matrix(
        (data, (row, col)), shape=(len(arr_id_feature), len(arr_barcode))
    )
    scipy.io.mmwrite(f"{path_folder_output_mtx_10x}matrix", sm)
    # remove previous output file to overwrite the file
    path_file_mtx_output = f"{path_folder_output_mtx_10x}matrix.mtx.gz"
    if filesystem_operations("exists", path_file_mtx_output):
        filesystem_operations("rm", path_file_mtx_output)
    bk.OS_Run(["gzip", f"{path_folder_output_mtx_10x}matrix.mtx"])  # gzip the mtx file


def AnnData_Convert_to_10X_MTX(
    adata,
    path_folder_mtx_output,
    dict_var_rename: dict = {"feature_types": "feature_type", "gene_ids": "id_feature"},
    dtype_value=np.int64,
):
    """# 2022-12-14 02:14:31
    write AnnData count matrix as a 10X matrix object

    'dict_var_rename' : a dictionary for renaming columns of adata.var columns
    """
    import scipy.io

    # compose df_var
    df_feature = adata.var
    df_feature.rename(columns=dict_var_rename, inplace=True)

    # create an output folder
    filesystem_operations("mkdir", path_folder_mtx_output, exist_ok=True)

    """ save barcode file """
    # retrieve list of barcodes
    arr_barcode = adata.obs.index.values
    pd.DataFrame(arr_barcode).to_csv(
        f"{path_folder_mtx_output}barcodes.tsv.gz", sep="\t", index=False, header=False
    )

    """ save feature file """
    # compose a feature dataframe
    df_feature[["id_feature", "feature", "feature_type"]].to_csv(
        f"{path_folder_mtx_output}features.tsv.gz", sep="\t", index=False, header=False
    )  # save as a file
    # retrieve list of features
    arr_id_feature = df_feature.id_feature.values

    """ save matrix file """
    # save count matrix as a gzipped matrix market format
    arr_int_barcode, arr_int_id_feature, arr_read_count = scipy.sparse.find(adata.X)
    # convert dtype of the values
    if dtype_value is not None:
        arr_read_count = arr_read_count.astype(dtype_value)
    # compose a sparse matrix
    sm = scipy.sparse.coo_matrix(
        (arr_read_count, (arr_int_id_feature, arr_int_barcode)),
        shape=(len(arr_id_feature), len(arr_barcode)),
    )
    scipy.io.mmwrite(f"{path_folder_mtx_output}matrix", sm)
    # remove previous output file to overwrite the file
    path_file_mtx_output = f"{path_folder_mtx_output}matrix.mtx.gz"
    if filesystem_operations("exists", path_file_mtx_output):
        filesystem_operations("rm", path_file_mtx_output)
    bk.OS_Run(["gzip", f"{path_folder_mtx_output}matrix.mtx"])  # gzip the mtx file


def __function_for_adjusting_thresholds_for_filtering_empty_droplets__(
    path_folder_mtx_10x_output, min_counts, min_features, min_cells
):
    """# 2022-02-23 14:26:07
    This function is intended for the use in 'MTX_10X_Filter' function for filtering cells from the 10X dataset (before chromium X, 10,000 cells per channel)

    Assuming a typical number of droplets in a experiment is 100,000, adjust 'min_counts' to reduce the number of filtered cells below 'int_max_num_cells'
    """
    s_count = (
        pd.read_csv(
            f"{path_folder_mtx_10x_output}dict_id_column_to_count.before_filtering.tsv.gz",
            sep="\t",
            header=None,
            index_col=0,
        )[1]
        .sort_values(ascending=False)
        .iloc[:100000]
    )

    int_max_num_cells = 20000  # maximum number of allowed cells
    min_counts_maximum = 2000

    def function_for_increasing_min_counts(min_counts):
        return min_counts * 2

    while True:
        """increase threshold if the number of filtered cells is larger than 'int_max_num_cells'"""
        if (
            len(s_count[s_count > min_counts]) > int_max_num_cells
            and min_counts < min_counts_maximum
        ):
            min_counts = function_for_increasing_min_counts(min_counts)
        else:
            break
    return min_counts, min_features, min_cells


def _get_matrix_market_data_type_from_line(line):
    """# 2023-09-10 16:06:24
    analyze the line to get matrix market data type in string format ('real' or 'integer')
    """
    try:
        return line.split("%%MatrixMarket")[1].strip().split()[2]
    except:
        return  # if an error occurs, return None


def _MTX_Detect_data_type(path_file_mtx):
    """# 2023-09-10 16:01:34
    detect data type of the input matrix file
    """
    with (
        gzip.open(path_file_mtx, "rt")
        if ".gz" == path_file_mtx[-3:]
        else open(path_file_mtx, "r")
    ) as file:
        str_data_type = _get_matrix_market_data_type_from_line(
            file.readline()
        )  # matrix market header line is present in the first line
    return str_data_type  # return the datatype


def MTX_10X_Split(
    path_folder_mtx_10x_output,
    int_max_num_entries_for_chunk=10000000,
    flag_split_mtx=True,
    flag_split_mtx_again=False,
):
    """# 2022-04-28 01:16:35
    split input mtx file into multiple files and write a flag file indicating the splitting has been completed.
    return the list of split mtx files

    'flag_split_mtx' : if 'flag_split_mtx' is True, split input mtx file into multiple files. if False, does not split the input matrix, and just return the list containing a single path pointing to the input matrix. This flag exists for the compatibility with single-thread operations
    'flag_split_mtx_again' : split the input matrix again even if it has beem already split. It will remove previously split files.
    """
    # 'flag_split_mtx' : if False, does not split the input matrix, and just return the list containing a single path pointing to the input matrix
    if not flag_split_mtx:
        return [f"{path_folder_mtx_10x_output}matrix.mtx.gz"]

    """ if 'flag_split_mtx_again' flag is True, remove previously split files """
    path_file_flag = f"{path_folder_mtx_10x_output}matrix.mtx.gz.split.flag"
    if flag_split_mtx_again:
        filesystem_operations("rm", path_file_flag)  # remove the flag
        # remove previously split files
        for path_file in filesystem_operations(
            "glob", f"{path_folder_mtx_10x_output}matrix.mtx.gz.*.gz"
        ):
            filesystem_operations("rm", path_file)

    """ split input matrix file """
    if not filesystem_operations(
        "exists", path_file_flag
    ):  # check whether the flag exists
        index_mtx_10x = 0
        newfile = gzip.open(
            f"{path_folder_mtx_10x_output}matrix.mtx.gz.{index_mtx_10x}.gz", "wb"
        )
        l_path_file_mtx_10x = [
            f"{path_folder_mtx_10x_output}matrix.mtx.gz.{index_mtx_10x}.gz"
        ]
        int_num_entries_written_for_the_current_chunk = 0
        with gzip.open(f"{path_folder_mtx_10x_output}matrix.mtx.gz", "rb") as file:
            while True:
                line = file.readline()  # binary string
                if len(line) == 0:
                    newfile.close()  # close the output file
                    break
                """ write the line to the current chunk and update the number of entries written for the current chunk """
                newfile.write(line)
                int_num_entries_written_for_the_current_chunk += 1
                """ initialize the next chunk if a sufficient number of entries were written """
                if (
                    int_num_entries_written_for_the_current_chunk
                    >= int_max_num_entries_for_chunk
                ):
                    newfile.close()  # close the output file
                    # initialize the next chunk
                    index_mtx_10x += 1
                    int_num_entries_written_for_the_current_chunk = 0
                    newfile = gzip.open(
                        f"{path_folder_mtx_10x_output}matrix.mtx.gz.{index_mtx_10x}.gz",
                        "wb",
                    )
                    l_path_file_mtx_10x.append(
                        f"{path_folder_mtx_10x_output}matrix.mtx.gz.{index_mtx_10x}.gz"
                    )
        with open(path_file_flag, "w") as file:
            file.write("completed")
    else:
        """retrieve the list of split mtx files"""
        df = bk.GLOB_Retrive_Strings_in_Wildcards(
            f"{path_folder_mtx_10x_output}matrix.mtx.gz.*.gz"
        )
        df.wildcard_0 = df.wildcard_0.astype(int)
        df.sort_values("wildcard_0", ascending=True, inplace=True)
        l_path_file_mtx_10x = df.path.values
    return l_path_file_mtx_10x


dict_id_feature_to_index_feature = dict()


def __MTX_10X_Combine__renumber_feature_mtx_10x__(
    path_file_input, path_folder_mtx_10x_output
):
    """# deprecated
    internal function for MTX_10X_Combine
    # 2022-02-22 00:38:33
    """
    #     dict_id_feature_to_index_feature = bk.PICKLE_Read( f'{path_folder_mtx_10x_output}dict_id_feature_to_index_feature.pickle' ) # retrieve id_feature to index_feature mapping
    for (
        path_folder_mtx_10x,
        int_total_n_barcodes_of_previously_written_matrices,
        index_mtx_10x,
    ) in pd.read_csv(path_file_input, sep="\t").values:
        # directly write matrix.mtx.gz file without header
        with gzip.open(
            f"{path_folder_mtx_10x_output}matrix.mtx.gz.{index_mtx_10x}.gz", "wb"
        ) as newfile:
            arr_id_feature = pd.read_csv(
                f"{path_folder_mtx_10x}features.tsv.gz", sep="\t", header=None
            ).values[
                :, 0
            ]  # retrieve a list of id_feature for the current dataset
            with gzip.open(
                f"{path_folder_mtx_10x}matrix.mtx.gz", "rb"
            ) as file:  # retrieve a list of features
                line = file.readline().decode()  # read the first line
                # if the first line of the file contains a comment line, read all comment lines and a description line following the comments.
                if len(line) > 0 and line[0] == "%":
                    # read comment and the description line
                    while True:
                        if line[0] != "%":
                            break
                        line = file.readline().decode()  # read the next line
                    line = (
                        file.readline().decode()
                    )  # discard the description line and read the next line
                # process entries
                while True:
                    if len(line) == 0:
                        break
                    index_row, index_col, int_value = tuple(
                        map(int, line.strip().split())
                    )  # parse each entry of the current matrix
                    newfile.write(
                        (
                            " ".join(
                                tuple(
                                    map(
                                        str,
                                        [
                                            dict_id_feature_to_index_feature[
                                                arr_id_feature[index_row - 1]
                                            ],
                                            index_col
                                            + int_total_n_barcodes_of_previously_written_matrices,
                                            int_value,
                                        ],
                                    )
                                )
                            )
                            + "\n"
                        ).encode()
                    )  # translate indices of the current matrix to that of the combined matrix
                    line = file.readline().decode()  # read the next line


def Read_SPLiT_Seq(path_folder_mtx):
    """# 2022-04-22 07:10:50
    Read SPLiT-Seq pipeline output
    return:
    df_feature, df_mtx
    """
    path_file_bc = f"{path_folder_mtx}cell_metadata.csv"
    path_file_feature = f"{path_folder_mtx}genes.csv"
    path_file_mtx = f"{path_folder_mtx}DGE.mtx"

    # read mtx file as a tabular format
    df_mtx = pd.read_csv(path_file_mtx, sep=" ", comment="%")
    df_mtx.columns = ["id_row", "id_column", "read_count"]

    # read barcode and feature information
    df_bc = pd.read_csv(path_file_bc)[["cell_barcode"]]
    df_bc.columns = ["barcode"]
    df_feature = pd.read_csv(path_file_feature, index_col=0)
    df_feature.columns = ["id_feature", "feature", "genome"]

    # mapping using 1 based coordinates (0->1 based coordinate )
    df_mtx["barcode"] = df_mtx.id_row.apply(
        bk.Map(bk.DICTIONARY_Build_from_arr(df_bc.barcode.values, index_start=1)).a2b
    )  # mapping using 1 based coordinates (0->1 based coordinate )
    df_mtx["id_feature"] = df_mtx.id_column.apply(
        bk.Map(
            bk.DICTIONARY_Build_from_arr(df_feature.id_feature.values, index_start=1)
        ).a2b
    )
    df_mtx.drop(
        columns=["id_row", "id_column"], inplace=True
    )  # drop unnecessary columns
    return df_feature, df_mtx


def MTX_10X_Barcode_add_prefix_or_suffix(
    path_file_barcodes_input,
    path_file_barcodes_output=None,
    barcode_prefix="",
    barcode_suffix="",
):
    """# 2022-05-13 17:54:13
    Add prefix or suffix to the 'barcode' of a given 'barcodes.tsv.gz' file
    'path_file_barcodes_output' : default: None. by default, the input 'path_file_barcodes_input' file will be overwritten with the modified barcodes
    """
    flag_replace_input_file = (
        path_file_barcodes_output is None
    )  # retrieve a flag indicating the replacement of original input file with modified input file
    if flag_replace_input_file:
        path_file_barcodes_output = (
            f"{path_file_barcodes_input}.writing.tsv.gz"  # set a temporary output file
        )
    newfile = gzip.open(path_file_barcodes_output, "wb")  # open an output file
    with gzip.open(path_file_barcodes_input, "rb") as file:
        while True:
            line = file.readline()
            if len(line) == 0:
                break
            barcode = line.decode().strip()  # parse a barcode
            barcode_new = (
                barcode_prefix + barcode + barcode_suffix
            )  # compose new barcode
            newfile.write((barcode_new + "\n").encode())  # write a new barcode
    newfile.close()  # close the output file
    # if the output file path was not given, replace the original file with modified file
    if flag_replace_input_file:
        filesystem_operations("rm", path_file_barcodes_input)
        filesystem_operations("mv", path_file_barcodes_output, path_file_barcodes_input)


def MTX_10X_Feature_add_prefix_or_suffix(
    path_file_features_input,
    path_file_features_output=None,
    id_feature_prefix="",
    id_feature_suffix="",
    name_feature_prefix="",
    name_feature_suffix="",
):
    """# 2022-05-13 17:54:17
    Add prefix or suffix to the id_feature and name_feature of a given 'features.tsv.gz' file
    'path_file_features_output' : default: None. by default, the input 'path_file_features_input' file will be overwritten with the modified features
    """
    flag_replace_input_file = (
        path_file_features_output is None
    )  # retrieve a flag indicating the replacement of original input file with modified input file
    if flag_replace_input_file:
        path_file_features_output = (
            f"{path_file_features_input}.writing.tsv.gz"  # set a temporary output file
        )
    newfile = gzip.open(path_file_features_output, "wb")  # open an output file
    with gzip.open(path_file_features_input, "rb") as file:
        while True:
            line = file.readline()
            if len(line) == 0:
                break
            id_feature, name_feature, type_feature = (
                line.decode().strip().split("\t")
            )  # parse a feature
            id_feature_new = (
                id_feature_prefix + id_feature + id_feature_suffix
            )  # compose new id_feature
            name_feature_new = (
                name_feature_prefix + name_feature + name_feature_suffix
            )  # compose new name_feature
            newfile.write(
                (
                    "\t".join([id_feature_new, name_feature_new, type_feature]) + "\n"
                ).encode()
            )  # write a new feature
    newfile.close()  # close the output file
    # if the output file path was not given, replace the original file with modified file
    if flag_replace_input_file:
        filesystem_operations("rm", path_file_features_input)
        filesystem_operations("mv", path_file_features_output, path_file_features_input)


def __MTX_10X_Combine__renumber_barcode_or_feature_index_mtx_10x__(
    path_file_input: str,
    path_folder_mtx_10x_output: str,
    flag_renumber_feature_index: bool,
    str_data_type: str,
):
    """
    internal function for MTX_10X_Combine
    # 2023-09-10 17:15:05

    flag_renumber_feature_index : bool : if True, assumes barcodes are not shared between matrices and renumber features only. If False, assumes features are not shared between matrices and renumber barcodes only.
    str_data_type : str # matrix market datatype in string format
    """
    global dict_id_entry_to_index_entry
    flag_matrix_contain_float_values = (
        str_data_type == "real"
    )  # retrieve a flag indicating the matrix is containing float values
    for (
        path_folder_mtx_10x,
        int_total_n_entries_of_previously_written_matrices,
        index_mtx_10x,
    ) in pd.read_csv(path_file_input, sep="\t").values:
        # directly write matrix.mtx.gz file without header
        with gzip.open(
            f"{path_folder_mtx_10x_output}matrix.mtx.gz.{index_mtx_10x}.gz", "wb"
        ) as newfile:
            arr_id_entry = pd.read_csv(
                f"{path_folder_mtx_10x}{'features' if flag_renumber_feature_index else 'barcodes'}.tsv.gz",
                sep="\t",
                header=None,
            ).values[
                :, 0
            ]  # retrieve a list of id_feature for the current dataset
            with gzip.open(
                f"{path_folder_mtx_10x}matrix.mtx.gz", "rb"
            ) as file:  # retrieve a list of features
                line = file.readline().decode()  # read the first line
                # if the first line of the file contains a comment line, read all comment lines and a description line following the comments.
                if len(line) > 0 and line[0] == "%":
                    # read comment and the description line
                    while True:
                        if line[0] != "%":
                            break
                        line = file.readline().decode()  # read the next line
                    line = (
                        file.readline().decode()
                    )  # discard the description line and read the next line
                # process entries
                while True:
                    if len(line) == 0:
                        break
                    """ parse a record """
                    l_str = (
                        line.strip().split()
                    )  # parse a record of a matrix-market format file
                    if flag_matrix_contain_float_values:  # parse float data type
                        id_row, id_column, value = (
                            int(l_str[0]),
                            int(l_str[1]),
                            float(l_str[2]),
                        )
                    else:  # parse integer data type
                        id_row, id_column, value = tuple(int(float(e)) for e in l_str)

                    newfile.write(
                        (
                            " ".join(
                                tuple(
                                    map(
                                        str,
                                        (
                                            [
                                                dict_id_entry_to_index_entry[
                                                    arr_id_entry[id_row - 1]
                                                ],
                                                id_column
                                                + int_total_n_entries_of_previously_written_matrices,
                                            ]
                                            if flag_renumber_feature_index
                                            else [
                                                id_row
                                                + int_total_n_entries_of_previously_written_matrices,
                                                dict_id_entry_to_index_entry[
                                                    arr_id_entry[id_column - 1]
                                                ],
                                            ]
                                        )
                                        + [value],
                                    )
                                )
                            )
                            + "\n"
                        ).encode()
                    )  # translate indices of the current matrix to that of the combined matrix
                    line = file.readline().decode()  # read the next line


def MTX_10X_Combine(
    path_folder_mtx_10x_output,
    *l_path_folder_mtx_10x_input,
    int_num_threads=15,
    flag_split_mtx=True,
    flag_split_mtx_again=False,
    int_max_num_entries_for_chunk=10000000,
    flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs=None,
    flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs=None,
    verbose=False,
):
    """
    # 2023-09-10 17:14:57
    Combine 10X count matrix files from the given list of folders and write combined output files to the given output folder 'path_folder_mtx_10x_output'
    If there are no shared cells between matrix files, a low-memory mode will be used. The output files will be simply combined since no count summing operation is needed. Only feature matrix will be loaded and updated in the memory.
    'id_feature' should be unique across all features. if id_feature is not unique, features with duplicated id_features will lead to combining of the features into a single feature (with combined counts/values).

    'int_num_threads' : number of threads to use when combining datasets. multiple threads will be utilized only when datasets does not share cells and thus can be safely concatanated.
    'flag_split_mtx' : split the resulting mtx file so that the contents in the output mtx file can be processed in parallel without ungzipping the mtx.gz file and spliting the file.
    'flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs' : a flag for entering low-memory mode when there is no shared cells between given input matrices. By default (when None is given), matrices will be examined and the flag will be set automatically by the program. To reduce running time and memory, this flag can be manually set by users. Explicitly setting this flag will dramatically reduce the memory consumption.
    'flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs' : a flag for entering low-memory mode when there is no shared features between given input matrices. By default (when None is given), matrices will be examined and the flag will be set automatically by the program. To reduce running time and memory, this flag can be manually set by users. Explicitly setting this flag will dramatically reduce the memory consumption.
    """

    # create an output folder
    filesystem_operations("mkdir", path_folder_mtx_10x_output, exist_ok=True)

    str_data_type = (
        "real"
        if sum(
            _MTX_Detect_data_type(f"{path_folder}matrix.mtx.gz") == "real"
            for path_folder in l_path_folder_mtx_10x_input
        )
        else "integer"
    )  # retrieve datatype of the operation. if at least one matrix contains real data type, treat all matrices as matrix containing real data

    if (
        not flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs
        and flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs is None
    ):
        """retrieve cell barcodes of all 10X matrices and check whether cell barcodes are not shared between matrices"""
        int_total_n_barcodes_of_previously_written_matrices = (
            0  # follow the number of barcodes that are previously written
        )
        l_int_total_n_barcodes_of_previously_written_matrices = (
            []
        )  # calculate the number of barcodes of the previous dataset in the combined mtx.
        set_barcode = set()  # update a set of unique barcodes
        for path_folder_mtx_10x in l_path_folder_mtx_10x_input:
            arr_barcode = (
                pd.read_csv(
                    f"{path_folder_mtx_10x}barcodes.tsv.gz", sep="\t", header=None
                )
                .squeeze("columns")
                .values
            )  # retrieve a list of features
            set_barcode.update(arr_barcode)  # update a set of barcodes
            l_int_total_n_barcodes_of_previously_written_matrices.append(
                int_total_n_barcodes_of_previously_written_matrices
            )
            int_total_n_barcodes_of_previously_written_matrices += len(
                arr_barcode
            )  # update the number of barcodes
        """ check whether there are shared cell barcodes between matrices and set a flag for entering a low-memory mode """
        flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs = (
            len(set_barcode) == int_total_n_barcodes_of_previously_written_matrices
        )  # update flag
    elif flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs:
        """retrieve cell barcodes of all 10X matrices and check whether cell barcodes are not shared between matrices"""
        int_total_n_barcodes_of_previously_written_matrices = (
            0  # follow the number of barcodes that are previously written
        )
        l_int_total_n_barcodes_of_previously_written_matrices = (
            []
        )  # calculate the number of barcodes of the previous dataset in the combined mtx.
        for path_folder_mtx_10x in l_path_folder_mtx_10x_input:
            l_int_total_n_barcodes_of_previously_written_matrices.append(
                int_total_n_barcodes_of_previously_written_matrices
            )
            int_total_n_barcodes_of_previously_written_matrices += len(
                pd.read_csv(
                    f"{path_folder_mtx_10x}barcodes.tsv.gz", sep="\t", header=None
                )
            )  # retrieve a list of barcodes and # update the number of barcodes

    if (
        not flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs
        and flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs is None
    ):
        """retrieve features of all 10X matrices and check whether features are not shared between matrices"""
        int_total_n_features_of_previously_written_matrices = (
            0  # follow the number of features that are previously written
        )
        l_int_total_n_features_of_previously_written_matrices = (
            []
        )  # calculate the number of features of the previous dataset in the combined mtx.
        set_feature = set()  # update a set of unique features
        for path_folder_mtx_10x in l_path_folder_mtx_10x_input:
            arr_feature = (
                pd.read_csv(
                    f"{path_folder_mtx_10x}features.tsv.gz",
                    sep="\t",
                    header=None,
                    usecols=[0],
                )
                .squeeze("columns")
                .values
            )  # retrieve a list of features
            set_feature.update(arr_feature)  # update a set of features
            l_int_total_n_features_of_previously_written_matrices.append(
                int_total_n_features_of_previously_written_matrices
            )
            int_total_n_features_of_previously_written_matrices += len(
                arr_feature
            )  # update the number of features
        """ check whether there are shared features between matrices and set a flag for entering a low-memory mode """
        flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs = (
            len(set_feature) == int_total_n_features_of_previously_written_matrices
        )  # update flag
    elif flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs:
        """retrieve features of all 10X matrices and check whether features are not shared between matrices"""
        int_total_n_features_of_previously_written_matrices = (
            0  # follow the number of features that are previously written
        )
        l_int_total_n_features_of_previously_written_matrices = (
            []
        )  # calculate the number of features of the previous dataset in the combined mtx.
        for path_folder_mtx_10x in l_path_folder_mtx_10x_input:
            l_int_total_n_features_of_previously_written_matrices.append(
                int_total_n_features_of_previously_written_matrices
            )
            int_total_n_features_of_previously_written_matrices += len(
                pd.read_csv(
                    f"{path_folder_mtx_10x}features.tsv.gz",
                    sep="\t",
                    header=None,
                    usecols=[0],
                )
            )  # retrieve a list of features and update the number of features

    flag_low_memory_mode = (
        flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs
        or flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs
    )  # retrieve flag for low-memory mode
    if flag_low_memory_mode:
        """low-memory mode"""
        flag_renumber_feature_index = flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs  # retrieve a flag for renumbering features
        if verbose:
            logger.info(
                f"entering low-memory mode, re-numbering {'features' if flag_renumber_feature_index else 'barcodes'} index because {'barcodes' if flag_renumber_feature_index else 'features'} are not shared across the matrices."
            )

        """ write a combined barcodes/features.tsv.gz - that are not shared between matrices """
        bk.OS_Run(
            ["cat"]
            + list(
                f"{path_folder_mtx_10x}{'barcodes' if flag_renumber_feature_index else 'features'}.tsv.gz"
                for path_folder_mtx_10x in l_path_folder_mtx_10x_input
            ),
            path_file_stdout=f"{path_folder_mtx_10x_output}{'barcodes' if flag_renumber_feature_index else 'features'}.tsv.gz",
            stdout_binary=True,
            return_output=False,
        )  # combine the files in order

        """ collect a set of unique entries and a list of entries for each 10X matrix """
        set_t_entry = set()  # update a set unique id_entry (either id_cell or id_entry)
        for path_folder_mtx_10x in l_path_folder_mtx_10x_input:
            set_t_entry.update(
                list(
                    map(
                        tuple,
                        pd.read_csv(
                            f"{path_folder_mtx_10x}{'features' if flag_renumber_feature_index else 'barcodes'}.tsv.gz",
                            sep="\t",
                            header=None,
                        ).values,
                    )
                )
            )  # update a set of feature tuples

        """ write a combined features/barcodes.tsv.gz - that are shared between matrices """
        l_t_entry = list(set_t_entry)  # convert set to list
        with gzip.open(
            f"{path_folder_mtx_10x_output}{'features' if flag_renumber_feature_index else 'barcodes'}.tsv.gz",
            "wb",
        ) as newfile:
            for t_entry in l_t_entry:
                newfile.write(("\t".join(t_entry) + "\n").encode())

        """ build a mapping of id_entry to index_entry, which will be consistent across datasets - for features/barcodes that are shared between matrices """
        global dict_id_entry_to_index_entry  # use global variable for multiprocessing
        dict_id_entry_to_index_entry = dict(
            (t_entry[0], index_entry + 1)
            for index_entry, t_entry in enumerate(l_t_entry)
        )  # 0>1 based index
        bk.PICKLE_Write(
            f"{path_folder_mtx_10x_output}dict_id_entry_to_index_entry.pickle",
            dict_id_entry_to_index_entry,
        )  # save id_feature to index_feature mapping as a pickle file

        """ collect the number of records for each 10X matrix """
        int_total_n_records = 0
        for path_folder_mtx_10x in l_path_folder_mtx_10x_input:
            with gzip.open(
                f"{path_folder_mtx_10x}matrix.mtx.gz", "rt"
            ) as file:  # retrieve a list of features
                line = file.readline()
                while line[0] == "%":
                    line = file.readline()
                int_total_n_records += int(
                    line.strip().split()[2]
                )  # update the total number of entries

        """ write a part of a combined matrix.mtx.gz for each dataset using multiple processes """
        # compose inputs for multiprocessing
        df_input = pd.DataFrame(
            {
                "path_folder_input_mtx_10x": l_path_folder_mtx_10x_input,
                "int_total_n_barcodes_of_previously_written_matrices": (
                    l_int_total_n_barcodes_of_previously_written_matrices
                    if flag_renumber_feature_index
                    else l_int_total_n_features_of_previously_written_matrices
                ),
                "index_mtx_10x": np.arange(
                    len(l_int_total_n_barcodes_of_previously_written_matrices)
                    if flag_renumber_feature_index
                    else len(l_int_total_n_features_of_previously_written_matrices)
                ),
            }
        )
        bk.Multiprocessing(
            df_input,
            __MTX_10X_Combine__renumber_barcode_or_feature_index_mtx_10x__,
            int_num_threads,
            global_arguments=[
                path_folder_mtx_10x_output,
                flag_renumber_feature_index,
                str_data_type,
            ],
        )
        #         filesystem_operations( 'rm', f'{path_folder_mtx_10x_output}dict_id_entry_to_index_entry.pickle' ) # remove pickle file

        """ combine parts and add the MTX file header to compose a combined mtx file """
        df_file = bk.GLOB_Retrive_Strings_in_Wildcards(
            f"{path_folder_mtx_10x_output}matrix.mtx.gz.*.gz"
        )
        df_file.wildcard_0 = df_file.wildcard_0.astype(int)
        df_file.sort_values("wildcard_0", inplace=True)

        # write header
        path_file_header = f"{path_folder_mtx_10x_output}matrix.mtx.header.txt.gz"
        with gzip.open(path_file_header, "wt") as newfile:
            newfile.write(
                f"%%MatrixMarket matrix coordinate {str_data_type} general\n%\n{len( l_t_entry ) if flag_renumber_feature_index else int_total_n_features_of_previously_written_matrices} {int_total_n_barcodes_of_previously_written_matrices if flag_renumber_feature_index else len( l_t_entry )} {int_total_n_records}\n"
            )
        bk.OS_Run(
            ["cat", path_file_header] + list(df_file.path.values),
            path_file_stdout=f"{path_folder_mtx_10x_output}matrix.mtx.gz",
            stdout_binary=True,
            return_output=False,
        )  # combine the output mtx files in the order

        if not flag_split_mtx:
            # delete temporary files if 'flag_split_mtx' is False
            for path_file in df_file.path.values:
                os.remove(path_file)

        # write a flag indicating that the current output directory contains split mtx files
        with open(f"{path_folder_mtx_10x_output}matrix.mtx.gz.split.flag", "w") as file:
            file.write("completed")
    else:
        """normal operation mode perfoming count merging operations"""
        l_df_mtx, l_df_feature = [], []
        for path_folder_mtx_10x in l_path_folder_mtx_10x_input:
            df_mtx, df_feature = Read_10X(path_folder_mtx_10x)
            l_df_mtx.append(df_mtx), l_df_feature.append(df_feature)

        # combine mtx
        df_mtx = pd.concat(l_df_mtx)
        df_mtx = df_mtx.groupby(["barcode", "id_feature"]).sum()
        df_mtx.reset_index(drop=False, inplace=True)

        # combine features
        df_feature = pd.concat(l_df_feature)
        df_feature.drop_duplicates(inplace=True)

        Write_10X(df_mtx, df_feature, path_folder_mtx_10x_output)

        # split a matrix file into multiple files
        MTX_10X_Split(
            path_folder_mtx_10x_output,
            int_max_num_entries_for_chunk=int_max_num_entries_for_chunk,
        )


def __Combine_Dictionaries__(path_folder_mtx_10x_input, name_dict):
    """# 2022-03-06 00:06:23
    combined dictionaries processed from individual files
    """
    import collections

    if filesystem_operations(
        "exists", f"{path_folder_mtx_10x_input}{name_dict}.tsv.gz"
    ):
        """if an output file already exists, read the file and return the combined dictionary"""
        dict_combined = (
            pd.read_csv(
                f"{path_folder_mtx_10x_input}{name_dict}.tsv.gz",
                sep="\t",
                header=None,
                index_col=0,
            )
            .iloc[:, 0]
            .to_dict()
        )
    else:
        """combine summarized results"""
        l_path_file = filesystem_operations(
            "glob", f"{path_folder_mtx_10x_input}{name_dict}.*"
        )
        try:
            counter = collections.Counter(
                pd.read_csv(l_path_file[0], sep="\t", header=None, index_col=0)
                .iloc[:, 0]
                .to_dict()
            )  # initialize counter object with the dictionary from the first file
        except pd.errors.EmptyDataError:
            counter = (
                collections.Counter()
            )  # when an error (possibly because the file is empty) occur, use an empty counter
        for path_file in l_path_file[1:]:
            # when an error (possibly because the file is empty) occur, skip updating the counter
            try:
                counter = counter + collections.Counter(
                    pd.read_csv(path_file, sep="\t", header=None, index_col=0)
                    .iloc[:, 0]
                    .to_dict()
                )  # update counter object using the dictionary from each file
            except pd.errors.EmptyDataError:
                pass
        dict_combined = dict(counter)  # retrieve a combined dictionary
        """remove temporary files """
        for path_file in l_path_file:
            filesystem_operations("rm", path_file)
        """ save dictionary as a file """
        pd.Series(dict_combined).to_csv(
            f"{path_folder_mtx_10x_input}{name_dict}.tsv.gz", sep="\t", header=None
        )
    return dict_combined  # returns a combined dictionary


def __MTX_10X_Summarize_Counts__summarize_counts_for_each_mtx_10x__(
    path_file_input: str, path_folder_mtx_10x_input: str, str_data_type: str
):
    """
    internal function for MTX_10X_Summarize_Count
    # 2023-09-10 16:37:13

    str_data_type : str # matrix market datatype in string format
    """
    """ survey the metrics """
    """ for each split mtx file, count number of umi and n_feature for each cells or the number of cells for each feature """
    """ initialize the dictionaries that will be handled by the current function """
    dict_id_column_to_count = dict()
    dict_id_column_to_n_features = dict()
    dict_id_row_to_count = dict()
    dict_id_row_to_n_cells = dict()
    dict_id_row_to_log_transformed_count = dict()

    flag_matrix_contain_float_values = (
        str_data_type == "real"
    )  # retrieve a flag indicating the matrix is containing float values

    global dict_name_set_feature_to_set_id_row  # use global read-only object
    dict_name_set_feature_to_dict_id_column_to_count = dict(
        (name_set_feature, dict())
        for name_set_feature in dict_name_set_feature_to_set_id_row
    )  # initialize 'dict_name_set_feature_to_dict_id_column_to_count'
    for path_file_input_mtx in pd.read_csv(
        path_file_input, sep="\t", header=None
    ).values.ravel():
        with gzip.open(path_file_input_mtx, "rb") as file:
            """read the first line"""
            line = file.readline().decode()
            """ if the first line of the file contains a comment line, read all comment lines and a description line following the comments. """
            if len(line) > 0 and line[0] == "%":
                # read comment and the description line
                while True:
                    if line[0] != "%":
                        break
                    line = file.readline().decode()  # read the next line
                # process the description line
                int_num_rows, int_num_columns, int_num_entries = tuple(
                    int(e) for e in line.strip().split()
                )  # retrieve the number of rows, number of columns and number of entries
                line = file.readline().decode()  # read the next line
            """ process entries"""
            while True:
                if len(line) == 0:
                    break
                """ parse a record, and update metrics """
                l_str = (
                    line.strip().split()
                )  # parse a record of a matrix-market format file
                if flag_matrix_contain_float_values:  # parse float data type
                    id_row, id_column, value = (
                        int(l_str[0]),
                        int(l_str[1]),
                        float(l_str[2]),
                    )
                else:  # parse integer data type
                    id_row, id_column, value = tuple(int(float(e)) for e in l_str)

                """ 1-based > 0-based coordinates """
                id_row -= 1
                id_column -= 1
                """ update umi count for each cell """
                if id_column not in dict_id_column_to_count:
                    dict_id_column_to_count[id_column] = 0
                dict_id_column_to_count[id_column] += value
                """ update umi count of specific set of features for each cell """
                for (
                    name_set_feature
                ) in dict_name_set_feature_to_dict_id_column_to_count:
                    if id_row in dict_name_set_feature_to_set_id_row[name_set_feature]:
                        if (
                            id_column
                            not in dict_name_set_feature_to_dict_id_column_to_count[
                                name_set_feature
                            ]
                        ):
                            dict_name_set_feature_to_dict_id_column_to_count[
                                name_set_feature
                            ][id_column] = 0
                        dict_name_set_feature_to_dict_id_column_to_count[
                            name_set_feature
                        ][id_column] += value
                """ update n_features for each cell """
                if id_column not in dict_id_column_to_n_features:
                    dict_id_column_to_n_features[id_column] = 0
                dict_id_column_to_n_features[id_column] += 1
                """ update umi count for each feature """
                if id_row not in dict_id_row_to_count:
                    dict_id_row_to_count[id_row] = 0
                dict_id_row_to_count[id_row] += value
                """ update n_cells for each feature """
                if id_row not in dict_id_row_to_n_cells:
                    dict_id_row_to_n_cells[id_row] = 0
                dict_id_row_to_n_cells[id_row] += 1
                """ update log transformed counts, calculated by 'X_new = log_10(X_old + 1)', for each feature """
                if id_row not in dict_id_row_to_log_transformed_count:
                    dict_id_row_to_log_transformed_count[id_row] = 0
                dict_id_row_to_log_transformed_count[id_row] += math.log10(value + 1)

                """ read the next line """
                line = (
                    file.readline().decode()
                )  # binary > uncompressed string # read the next line

    # save collected count as tsv files
    str_uuid_process = bk.UUID()  # retrieve uuid of the current process
    pd.Series(dict_id_column_to_count).to_csv(
        f"{path_folder_mtx_10x_input}dict_id_column_to_count.{str_uuid_process}.tsv.gz",
        sep="\t",
        header=None,
    )
    pd.Series(dict_id_column_to_n_features).to_csv(
        f"{path_folder_mtx_10x_input}dict_id_column_to_n_features.{str_uuid_process}.tsv.gz",
        sep="\t",
        header=None,
    )
    pd.Series(dict_id_row_to_count).to_csv(
        f"{path_folder_mtx_10x_input}dict_id_row_to_count.{str_uuid_process}.tsv.gz",
        sep="\t",
        header=None,
    )
    pd.Series(dict_id_row_to_n_cells).to_csv(
        f"{path_folder_mtx_10x_input}dict_id_row_to_n_cells.{str_uuid_process}.tsv.gz",
        sep="\t",
        header=None,
    )
    pd.Series(dict_id_row_to_log_transformed_count).to_csv(
        f"{path_folder_mtx_10x_input}dict_id_row_to_log_transformed_count.{str_uuid_process}.tsv.gz",
        sep="\t",
        header=None,
    )

    # save collected counts as tsv files for 'dict_name_set_feature_to_dict_id_column_to_count'
    for name_set_feature in dict_name_set_feature_to_dict_id_column_to_count:
        pd.Series(
            dict_name_set_feature_to_dict_id_column_to_count[name_set_feature]
        ).to_csv(
            f"{path_folder_mtx_10x_input}{name_set_feature}.dict_id_column_to_count.{str_uuid_process}.tsv.gz",
            sep="\t",
            header=None,
        )


def MTX_10X_Summarize_Counts(
    path_folder_mtx_10x_input,
    verbose=False,
    int_num_threads=15,
    flag_split_mtx=True,
    int_max_num_entries_for_chunk=10000000,
    dict_name_set_feature_to_l_id_feature=dict(),
    flag_split_mtx_again=False,
):
    """# 2022-04-28 06:53:45
    Summarize
    (1) UMI and Feature counts for each cell,
    (2) UMI and Cell counts for each feature, and
    (3) log10-transformed values of UMI counts (X_new = log_10(X_old + 1)) for each feature
    (4) UMI counts for the optionally given lists of features for each cell
    and save these metrics as TSV files

    Inputs:
    'dict_name_set_feature_to_l_id_feature' : (Default: None)
                                            a dictionary with 'name_set_features' as key and a list of id_feature as value for each set of id_features.
                                            If None is given, only the basic metrics will be summarized.
                                            'name_set_features' should be compatible as a Linux file system (should not contain '/' and other special characters, such as newlines).
                                            (for Scarab short_read outputs)
                                            If 'atac' is given, 'promoter_and_gene_body', 'promoter' features will be summarized.
                                            If 'multiome' is given, total 'atac' counts will be summarized separately in addition to 'atac' mode

    Returns:
    a dictionary containing the following and other additional dictionaries: dict_id_column_to_count, dict_id_column_to_n_features, dict_id_row_to_count, dict_id_row_to_n_cells, dict_id_row_to_log_transformed_count
    """

    """ the name of the dictionaries handled by this function (basic) """
    l_name_dict = [
        "dict_id_column_to_count",
        "dict_id_column_to_n_features",
        "dict_id_row_to_count",
        "dict_id_row_to_n_cells",
        "dict_id_row_to_log_transformed_count",
    ]

    """ handle inputs """
    if path_folder_mtx_10x_input[-1] != "/":
        path_folder_mtx_10x_input += "/"

    # define flag and check whether the flag exists
    path_file_flag = f"{path_folder_mtx_10x_input}counts_summarized.flag"
    if not filesystem_operations("exists", path_file_flag):
        # define input file directories
        path_file_input_bc = f"{path_folder_mtx_10x_input}barcodes.tsv.gz"
        path_file_input_feature = f"{path_folder_mtx_10x_input}features.tsv.gz"
        path_file_input_mtx = f"{path_folder_mtx_10x_input}matrix.mtx.gz"

        # check whether all required files are present
        if sum(
            list(
                not filesystem_operations("exists", path_folder)
                for path_folder in [
                    path_file_input_bc,
                    path_file_input_feature,
                    path_file_input_mtx,
                ]
            )
        ):
            if verbose:
                logger.info(f"required file(s) is not present at {path_folder_mtx_10x}")

        """ split input mtx file into multiple files """
        l_path_file_mtx_10x = MTX_10X_Split(
            path_folder_mtx_10x_input,
            int_max_num_entries_for_chunk=int_max_num_entries_for_chunk,
            flag_split_mtx=flag_split_mtx,
            flag_split_mtx_again=flag_split_mtx_again,
        )
        str_data_type = _MTX_Detect_data_type(
            path_file_input_mtx
        )  # retrieve the data type of the matrix in string format

        """ prepare 'dict_name_set_feature_to_set_id_row' for summarizing total counts for given sets of features """
        global dict_name_set_feature_to_set_id_row
        dict_name_set_feature_to_set_id_row = (
            dict()
        )  # initialize 'dict_name_set_feature_to_set_id_row'
        if dict_name_set_feature_to_l_id_feature is not None:
            arr_id_feature = pd.read_csv(
                path_file_input_feature, sep="\t", usecols=[0], header=None
            ).values.ravel()  # retrieve array of id_features
            dict_id_feature_to_id_row = dict(
                (e, i) for i, e in enumerate(arr_id_feature)
            )  # retrieve id_feature -> id_row mapping

            """ handle presets for 'dict_name_set_feature_to_l_id_feature' """
            if isinstance(dict_name_set_feature_to_l_id_feature, str):
                str_preset = dict_name_set_feature_to_l_id_feature  # retrieve preset
                dict_name_set_feature_to_l_id_feature = (
                    dict()
                )  # initialize the dictionary
                if str_preset in ["multiome", "atac"]:
                    if str_preset == "multiome":
                        arr_id_feature_atac = (
                            bk.Search_list_of_strings_with_multiple_query(
                                arr_id_feature, "|mode=atac"
                            )
                        )
                        dict_name_set_feature_to_l_id_feature["atac_all"] = (
                            arr_id_feature_atac
                        )
                    elif str_preset == "atac":
                        arr_id_feature_atac = arr_id_feature
                    # add sets of promoter and gene_body features
                    arr_id_feature_atac_promoter_and_gene_body = (
                        bk.Search_list_of_strings_with_multiple_query(
                            arr_id_feature_atac,
                            "-genomic_region|",
                            "-repeatmasker_ucsc|",
                            "-regulatory_element|",
                        )
                    )
                    arr_id_feature_atac_promoter = (
                        bk.Search_list_of_strings_with_multiple_query(
                            arr_id_feature_atac_promoter_and_gene_body, "promoter|"
                        )
                    )
                    dict_name_set_feature_to_l_id_feature[
                        "atac_promoter_and_gene_body"
                    ] = arr_id_feature_atac_promoter_and_gene_body
                    dict_name_set_feature_to_l_id_feature["atac_promoter"] = (
                        arr_id_feature_atac_promoter
                    )

            # make sure that 'name_set_feature' does not contains characters incompatible with linux file path
            for name_set_feature in dict_name_set_feature_to_l_id_feature:
                assert not ("/" in name_set_feature or "\n" in name_set_feature)

            dict_name_set_feature_to_set_id_row = dict(
                (
                    name_set_feature,
                    set(
                        dict_id_feature_to_id_row[id_feature]
                        for id_feature in dict_name_set_feature_to_l_id_feature[
                            name_set_feature
                        ]
                    ),
                )
                for name_set_feature in dict_name_set_feature_to_l_id_feature
            )
            # bk.PICKLE_Write( f"{path_folder_mtx_10x_input}dict_name_set_feature_to_set_id_row.binary.pickle", dict_name_set_feature_to_set_id_row ) # write the dictionary as a pickle

        """ summarize each split mtx file """
        bk.Multiprocessing(
            l_path_file_mtx_10x,
            __MTX_10X_Summarize_Counts__summarize_counts_for_each_mtx_10x__,
            n_threads=int_num_threads,
            global_arguments=[path_folder_mtx_10x_input, str_data_type],
        )

        """ combine summarized results """
        # update the list of the names of dictionaries
        l_name_dict += list(
            f"{name_set_feature}.dict_id_column_to_count"
            for name_set_feature in bk.GLOB_Retrive_Strings_in_Wildcards(
                f"{path_folder_mtx_10x_input}*.dict_id_column_to_count.*.tsv.gz"
            ).wildcard_0.unique()
        )

        dict_dict = dict()
        for name_dict in l_name_dict:
            dict_dict[name_dict] = __Combine_Dictionaries__(
                path_folder_mtx_10x_input, name_dict
            )
        # write the flag
        with open(path_file_flag, "w") as newfile:
            newfile.write("completed at " + bk.TIME_GET_timestamp(True))
    else:
        """read summarized results"""
        # update the list of the names of dictionaries
        l_name_dict += list(
            f"{name_set_feature}.dict_id_column_to_count"
            for name_set_feature in bk.GLOB_Retrive_Strings_in_Wildcards(
                f"{path_folder_mtx_10x_input}*.dict_id_column_to_count.tsv.gz"
            ).wildcard_0.unique()
        )

        dict_dict = dict()
        for name_dict in l_name_dict:
            try:
                dict_dict[name_dict] = (
                    pd.read_csv(
                        f"{path_folder_mtx_10x_input}{name_dict}.tsv.gz",
                        sep="\t",
                        header=None,
                        index_col=0,
                    )
                    .iloc[:, 0]
                    .to_dict()
                )
            except (
                pd.errors.EmptyDataError
            ):  # handle when the current dictionary is empty
                dict_dict[name_dict] = dict()

    # return summarized metrics
    return dict_dict


def MTX_10X_Retrieve_number_of_rows_columns_and_records(path_folder_mtx_10x_input):
    """# 2022-03-05 19:58:32
    Retrieve the number of rows, columns, and entries from the matrix with the matrix market format

    'path_folder_mtx_10x_input' : a folder mtx file resides or path to mtx file

    Returns:
    int_num_rows, int_num_columns, int_num_entries
    """
    """ handle inputs """
    if (
        path_folder_mtx_10x_input[-3:].lower() == ".gz"
    ):  # when a path to mtx file was given
        path_file_input_mtx = path_folder_mtx_10x_input
    else:  # when a folder where mtx file resides was given
        if path_folder_mtx_10x_input[-1] != "/":
            path_folder_mtx_10x_input += "/"

        # define input file directories
        path_file_input_mtx = f"{path_folder_mtx_10x_input}matrix.mtx.gz"

        # check whether all required files are present
        if sum(
            list(
                not filesystem_operations("exists", path_folder)
                for path_folder in [path_file_input_mtx]
            )
        ):
            return None

    # read the input matrix
    with gzip.open(path_file_input_mtx, "rb") as file:
        """read the first line"""
        line = file.readline().decode().strip()
        """ if the first line of the file contains a comment line, read all comment lines and a description line following the comments. """
        if len(line) > 0 and line[0] == "%":
            # read comment and the description line
            while True:
                if line[0] != "%":
                    break
                line = file.readline().decode().strip()  # read the next line
            # process the description line
            int_num_rows, int_num_columns, int_num_entries = tuple(
                int(e) for e in line.strip().split()
            )  # retrieve the number of rows, number of columns and number of entries
        else:
            """the first line does not contain a comment, assumes it contains a description line"""
            int_num_rows, int_num_columns, int_num_entries = tuple(
                int(e) for e in line.strip().split()
            )  # retrieve the number of rows, number of columns and number of entries
    return int_num_rows, int_num_columns, int_num_entries


(
    dict_id_column_to_count,
    dict_id_row_to_avg_count,
    dict_id_row_to_avg_log_transformed_count,
    dict_id_row_to_avg_normalized_count,
    dict_id_row_to_avg_log_transformed_normalized_count,
) = (
    dict(),
    dict(),
    dict(),
    dict(),
    dict(),
)  # global variables # total UMI counts for each cell, average feature counts for each feature


def __MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr__first_pass__(
    path_file_input, path_folder_mtx_10x_input, int_target_sum
):
    """# 2022-03-06 01:21:07
    internal function for MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr
    """
    global dict_id_column_to_count, dict_id_row_to_avg_count, dict_id_row_to_avg_log_transformed_count  # use data in read-only global variables
    """ initialize the dictionaries that will be handled by the current function """
    dict_id_row_to_deviation_from_mean_count = dict()
    dict_id_row_to_deviation_from_mean_log_transformed_count = dict()
    dict_id_row_to_normalized_count = dict()
    dict_id_row_to_log_transformed_normalized_count = dict()

    for path_file_input_mtx in pd.read_csv(
        path_file_input, sep="\t", header=None
    ).values.ravel():
        with gzip.open(path_file_input_mtx, "rb") as file:
            """read the first line"""
            line = file.readline().decode()
            """ if the first line of the file contains a comment line, read all comment lines and a description line following the comments. """
            if len(line) > 0 and line[0] == "%":
                # read comment and the description line
                while True:
                    if line[0] != "%":
                        break
                    line = file.readline().decode()  # read the next line
                # process the description line
                int_num_rows, int_num_columns, int_num_entries = tuple(
                    int(e) for e in line.strip().split()
                )  # retrieve the number of rows, number of columns and number of entries
                line = file.readline().decode()  # read the next line
            """ process entries"""
            while True:
                if len(line) == 0:
                    break
                """ parse a record, and update metrics """
                id_row, id_column, int_value = tuple(
                    int(e) for e in line.strip().split()
                )  # parse a record of a matrix-market format file
                """ 1-based > 0-based coordinates """
                id_row -= 1
                id_column -= 1

                """ update deviation from mean umi count for count of each feature """
                if id_row not in dict_id_row_to_deviation_from_mean_count:
                    dict_id_row_to_deviation_from_mean_count[id_row] = 0
                dict_id_row_to_deviation_from_mean_count[id_row] += (
                    int_value - dict_id_row_to_avg_count[id_row]
                ) ** 2
                """ update deviation from mean log transformed umi count for log_transformed count of each feature """
                if (
                    id_row
                    not in dict_id_row_to_deviation_from_mean_log_transformed_count
                ):
                    dict_id_row_to_deviation_from_mean_log_transformed_count[id_row] = 0
                dict_id_row_to_deviation_from_mean_log_transformed_count[id_row] += (
                    math.log10(int_value + 1)
                    - dict_id_row_to_avg_log_transformed_count[id_row]
                ) ** 2
                """ calculate normalized target sum """
                int_value_normalized = (
                    int_value / dict_id_column_to_count[id_column] * int_target_sum
                )
                """ update normalized counts, calculated by 'X_new = X_old / total_umi * int_target_sum', for each feature """
                if id_row not in dict_id_row_to_normalized_count:
                    dict_id_row_to_normalized_count[id_row] = 0
                dict_id_row_to_normalized_count[id_row] += int_value_normalized
                """ update log transformed normalized counts, calculated by 'X_new = log_10(X_old / total_umi * int_target_sum + 1)', for each feature """
                if id_row not in dict_id_row_to_log_transformed_normalized_count:
                    dict_id_row_to_log_transformed_normalized_count[id_row] = 0
                dict_id_row_to_log_transformed_normalized_count[id_row] += math.log10(
                    int_value_normalized + 1
                )

                line = (
                    file.readline().decode()
                )  # binary > uncompressed string # read the next line

    # save collected count as tsv files
    str_uuid_process = bk.UUID()  # retrieve uuid of the current process
    pd.Series(dict_id_row_to_deviation_from_mean_count).to_csv(
        f"{path_folder_mtx_10x_input}dict_id_row_to_deviation_from_mean_count.{str_uuid_process}.tsv.gz",
        sep="\t",
        header=None,
    )
    pd.Series(dict_id_row_to_deviation_from_mean_log_transformed_count).to_csv(
        f"{path_folder_mtx_10x_input}dict_id_row_to_deviation_from_mean_log_transformed_count.{str_uuid_process}.tsv.gz",
        sep="\t",
        header=None,
    )
    pd.Series(dict_id_row_to_normalized_count).to_csv(
        f"{path_folder_mtx_10x_input}dict_id_row_to_normalized_count.{str_uuid_process}.tsv.gz",
        sep="\t",
        header=None,
    )
    pd.Series(dict_id_row_to_log_transformed_normalized_count).to_csv(
        f"{path_folder_mtx_10x_input}dict_id_row_to_log_transformed_normalized_count.{str_uuid_process}.tsv.gz",
        sep="\t",
        header=None,
    )


def __MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr__second_pass__(
    path_file_input, path_folder_mtx_10x_input, int_target_sum
):
    """# 2022-03-06 01:21:14
    internal function for MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr
    """
    global dict_id_column_to_count, dict_id_row_to_avg_normalized_count, dict_id_row_to_avg_log_transformed_normalized_count  # use data in read-only global variables
    """ initialize the dictionaries that will be handled by the current function """
    dict_id_row_to_deviation_from_mean_normalized_count = dict()
    dict_id_row_to_deviation_from_mean_log_transformed_normalized_count = dict()

    for path_file_input_mtx in pd.read_csv(
        path_file_input, sep="\t", header=None
    ).values.ravel():
        with gzip.open(path_file_input_mtx, "rb") as file:
            """read the first line"""
            line = file.readline().decode()
            """ if the first line of the file contains a comment line, read all comment lines and a description line following the comments. """
            if len(line) > 0 and line[0] == "%":
                # read comment and the description line
                while True:
                    if line[0] != "%":
                        break
                    line = file.readline().decode()  # read the next line
                # process the description line
                int_num_rows, int_num_columns, int_num_entries = tuple(
                    int(e) for e in line.strip().split()
                )  # retrieve the number of rows, number of columns and number of entries
                line = file.readline().decode()  # read the next line
            """ process entries"""
            while True:
                if len(line) == 0:
                    break
                """ parse a record, and update metrics """
                id_row, id_column, int_value = tuple(
                    int(e) for e in line.strip().split()
                )  # parse a record of a matrix-market format file
                """ 1-based > 0-based coordinates """
                id_row -= 1
                id_column -= 1

                """ calculate normalized target sum """
                int_value_normalized = (
                    int_value / dict_id_column_to_count[id_column] * int_target_sum
                )
                """ update deviation from mean normalized umi counts, calculated by 'X_new = X_old / total_umi * int_target_sum', for each feature """
                if id_row not in dict_id_row_to_deviation_from_mean_normalized_count:
                    dict_id_row_to_deviation_from_mean_normalized_count[id_row] = 0
                dict_id_row_to_deviation_from_mean_normalized_count[id_row] += (
                    int_value_normalized - dict_id_row_to_avg_normalized_count[id_row]
                ) ** 2
                """ update deviation from mean log transformed normalized umi counts, calculated by 'X_new = log_10(X_old / total_umi * int_target_sum + 1)', for each feature """
                if (
                    id_row
                    not in dict_id_row_to_deviation_from_mean_log_transformed_normalized_count
                ):
                    dict_id_row_to_deviation_from_mean_log_transformed_normalized_count[
                        id_row
                    ] = 0
                dict_id_row_to_deviation_from_mean_log_transformed_normalized_count[
                    id_row
                ] += (
                    math.log10(int_value_normalized + 1)
                    - dict_id_row_to_avg_log_transformed_normalized_count[id_row]
                ) ** 2

                line = (
                    file.readline().decode()
                )  # binary > uncompressed string # read the next line

    # save collected count as tsv files
    str_uuid_process = bk.UUID()  # retrieve uuid of the current process
    pd.Series(dict_id_row_to_deviation_from_mean_normalized_count).to_csv(
        f"{path_folder_mtx_10x_input}dict_id_row_to_deviation_from_mean_normalized_count.{str_uuid_process}.tsv.gz",
        sep="\t",
        header=None,
    )
    pd.Series(
        dict_id_row_to_deviation_from_mean_log_transformed_normalized_count
    ).to_csv(
        f"{path_folder_mtx_10x_input}dict_id_row_to_deviation_from_mean_log_transformed_normalized_count.{str_uuid_process}.tsv.gz",
        sep="\t",
        header=None,
    )


def MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr(
    path_folder_mtx_10x_input,
    int_target_sum=10000,
    verbose=False,
    int_num_threads=15,
    flag_split_mtx=True,
    int_max_num_entries_for_chunk=10000000,
):
    """# 2022-02-23 22:54:35
    Calculate average log transformed normalized expr
    (1) UMI and Feature counts for cells, and
    (2) Cell counts for features,
    and save these metrics as TSV files

    Arguments:
    'int_target_sum' : the target count for the total UMI count for each cell. The counts will normalized to meet the target sum.

    Returns:
    dict_id_column_to_count, dict_id_column_to_n_features, dict_id_row_to_count, dict_id_row_to_n_cells, dict_id_row_to_log_transformed_count
    """

    """ handle inputs """
    if path_folder_mtx_10x_input[-1] != "/":
        path_folder_mtx_10x_input += "/"

    # define flag and check whether the flag exists
    path_file_flag = f"{path_folder_mtx_10x_input}avg_expr_normalized_summarized.int_target_sum__{int_target_sum}.flag"
    if not filesystem_operations("exists", path_file_flag):
        # define input file directories
        path_file_input_bc = f"{path_folder_mtx_10x_input}barcodes.tsv.gz"
        path_file_input_feature = f"{path_folder_mtx_10x_input}features.tsv.gz"
        path_file_input_mtx = f"{path_folder_mtx_10x_input}matrix.mtx.gz"

        # check whether all required files are present
        if sum(
            list(
                not filesystem_operations("exists", path_folder)
                for path_folder in [
                    path_file_input_bc,
                    path_file_input_feature,
                    path_file_input_mtx,
                ]
            )
        ):
            if verbose:
                logger.info(f"required file(s) is not present at {path_folder_mtx_10x}")

        """ split input mtx file into multiple files """
        l_path_file_mtx_10x = MTX_10X_Split(
            path_folder_mtx_10x_input,
            int_max_num_entries_for_chunk=int_max_num_entries_for_chunk,
            flag_split_mtx=flag_split_mtx,
        )

        """ retrieve number of cells, features, and entries from the matrix file """
        (
            int_num_cells,
            int_num_features,
            int_num_entries,
        ) = MTX_10X_Retrieve_number_of_rows_columns_and_records(
            path_folder_mtx_10x_input
        )

        """ summarizes counts """
        global dict_id_column_to_count, dict_id_row_to_avg_count, dict_id_row_to_avg_log_transformed_count, dict_id_row_to_avg_normalized_count, dict_id_row_to_avg_log_transformed_normalized_count  # use global variable
        dict_data = MTX_10X_Summarize_Counts(
            path_folder_mtx_10x_input,
            verbose=verbose,
            int_num_threads=int_num_threads,
            flag_split_mtx=flag_split_mtx,
            int_max_num_entries_for_chunk=int_max_num_entries_for_chunk,
        )
        (
            dict_id_column_to_count,
            dict_id_column_to_n_features,
            dict_id_row_to_count,
            dict_id_row_to_n_cells,
            dict_id_row_to_log_transformed_count,
        ) = (
            dict_data["dict_id_column_to_count"],
            dict_data["dict_id_column_to_n_features"],
            dict_data["dict_id_row_to_count"],
            dict_data["dict_id_row_to_n_cells"],
            dict_data["dict_id_row_to_log_transformed_count"],
        )  # parse 'dict_data'

        """ first pass """
        # calculate mean counts
        dict_id_row_to_avg_count = (
            pd.Series(dict_id_row_to_count) / int_num_cells
        ).to_dict()  # calculate average expression of each feature
        dict_id_row_to_avg_log_transformed_count = (
            pd.Series(dict_id_row_to_log_transformed_count) / int_num_cells
        ).to_dict()  # calculate average log-transformed expression of each feature

        """ calculated average log2 transformed normalized expr for each split mtx file """
        bk.Multiprocessing(
            l_path_file_mtx_10x,
            __MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr__first_pass__,
            n_threads=int_num_threads,
            global_arguments=[path_folder_mtx_10x_input, int_target_sum],
        )

        l_name_dict_first_pass = [
            "dict_id_row_to_deviation_from_mean_count",
            "dict_id_row_to_deviation_from_mean_log_transformed_count",
            "dict_id_row_to_normalized_count",
            "dict_id_row_to_log_transformed_normalized_count",
        ]

        """ combine summarized results """
        dict_dict = dict()
        for name_dict in l_name_dict_first_pass:
            dict_dict[name_dict] = __Combine_Dictionaries__(
                path_folder_mtx_10x_input, name_dict
            )

        """ second pass """
        # calculate mean counts
        dict_id_row_to_avg_normalized_count = (
            pd.Series(dict_dict["dict_id_row_to_normalized_count"]) / int_num_cells
        ).to_dict()  # calculate average expression of each feature
        dict_id_row_to_avg_log_transformed_normalized_count = (
            pd.Series(dict_dict["dict_id_row_to_log_transformed_normalized_count"])
            / int_num_cells
        ).to_dict()  # calculate average log-transformed expression of each feature

        """ calculated average log2 transformed normalized expr for each split mtx file """
        bk.Multiprocessing(
            l_path_file_mtx_10x,
            __MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr__second_pass__,
            n_threads=int_num_threads,
            global_arguments=[path_folder_mtx_10x_input, int_target_sum],
        )

        l_name_dict_second_pass = [
            "dict_id_row_to_deviation_from_mean_normalized_count",
            "dict_id_row_to_deviation_from_mean_log_transformed_normalized_count",
        ]

        """ combine summarized results """
        for name_dict in l_name_dict_second_pass:
            dict_dict[name_dict] = __Combine_Dictionaries__(
                path_folder_mtx_10x_input, name_dict
            )

        """ compose a dataframe containing the summary about the features """
        df_summary = pd.DataFrame(
            {
                "n_cells": pd.Series(dict_id_row_to_n_cells),
                "variance_of_count": pd.Series(
                    dict_dict["dict_id_row_to_deviation_from_mean_count"]
                )
                / (int_num_cells - 1),
                "variance_of_log_transformed_count": pd.Series(
                    dict_dict[
                        "dict_id_row_to_deviation_from_mean_log_transformed_count"
                    ]
                )
                / (int_num_cells - 1),
                "variance_of_normalized_count": pd.Series(
                    dict_dict["dict_id_row_to_deviation_from_mean_normalized_count"]
                )
                / (int_num_cells - 1),
                "variance_of_log_transformed_normalized_count": pd.Series(
                    dict_dict[
                        "dict_id_row_to_deviation_from_mean_log_transformed_normalized_count"
                    ]
                )
                / (int_num_cells - 1),
                "mean_count": pd.Series(dict_id_row_to_avg_count),
                "mean_log_transformed_count": pd.Series(
                    dict_id_row_to_avg_log_transformed_count
                ),
                "mean_normalized_count": pd.Series(dict_id_row_to_avg_normalized_count),
                "mean_log_transformed_normalized_count": pd.Series(
                    dict_id_row_to_avg_log_transformed_normalized_count
                ),
            }
        )
        # read a dataframe containing features
        df_feature = pd.read_csv(path_file_input_feature, sep="\t", header=None)
        df_feature.columns = ["id_feature", "feature", "feature_type"]

        df_summary = df_summary.join(
            df_feature, how="left"
        )  # add df_feature to the df_summary
        df_summary.index.name = "id_row"
        df_summary.reset_index(drop=False, inplace=True)  # retrieve id_row as a column
        df_summary.to_csv(
            f"{path_folder_mtx_10x_input}statistical_summary_of_features.int_target_sum__{int_target_sum}.tsv.gz",
            sep="\t",
            index=False,
        )  # save statistical summary as a text file

        # write the flag
        with open(path_file_flag, "w") as newfile:
            newfile.write("completed at " + bk.TIME_GET_timestamp(True))
    else:
        """if 'MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr' function has been already run on the current folder, read the previously saved result, and return the summary dataframe"""
        df_summary = pd.read_csv(
            f"{path_folder_mtx_10x_input}statistical_summary_of_features.int_target_sum__{int_target_sum}.tsv.gz",
            sep="\t",
        )  # save statistical summary as a text file
    return df_summary


dict_id_column_previous_to_id_column_current, dict_id_row_previous_to_id_row_current = (
    dict(),
    dict(),
)


def __MTX_10X_Filter__filter_mtx_10x__(
    path_file_input: str,
    path_folder_mtx_10x_output: str,
    str_data_type: str,
):
    """# 2023-09-10 16:37:33
    __MTX_10X_Filter__filter_mtx_10x__

    str_data_type : str # matrix market datatype in string format

    Returns:
    int_n_entries = total number of entries written by the current process after filtering
    """
    int_n_entries = (
        0  # total number of entries written by the current process after filtering
    )
    flag_matrix_contain_float_values = (
        str_data_type == "real"
    )  # retrieve a flag indicating the matrix is containing float values
    #     dict_id_column_previous_to_id_column_current = bk.PICKLE_Read( f'{path_folder_mtx_10x_output}dict_id_column_previous_to_id_column_current.pickle' ) # retrieve id_feature to index_feature mapping
    #     dict_id_row_previous_to_id_row_current = bk.PICKLE_Read( f'{path_folder_mtx_10x_output}dict_id_row_previous_to_id_row_current.pickle' ) # retrieve id_feature to index_feature mapping
    """ write a filtered matrix.mtx.gz for each split mtx file """
    for path_file_mtx_10x, index_mtx_10x in pd.read_csv(
        path_file_input, sep="\t"
    ).values:
        # directly write matrix.mtx.gz file without using an external dependency
        with gzip.open(
            f"{path_folder_mtx_10x_output}matrix.mtx.gz.{index_mtx_10x}.gz", "wb"
        ) as newfile:
            with gzip.open(path_file_mtx_10x, "rb") as file:
                """read the first line"""
                line = file.readline().decode()
                """ if the first line of the file contains a comment line, read all comment lines and a description line following the comments. """
                if len(line) > 0 and line[0] == "%":
                    # read comment and the description line
                    while True:
                        if line[0] != "%":
                            break
                        line = file.readline().decode()  # read the next line
                    # process the description line
                    int_num_rows, int_num_columns, int_num_entries = tuple(
                        int(e) for e in line.strip().split()
                    )  # retrieve the number of rows, number of columns and number of entries
                    line = file.readline().decode()  # read the next line
                """ process entries"""
                while True:
                    if len(line) == 0:
                        break
                    """ parse a record """
                    l_str = (
                        line.strip().split()
                    )  # parse a record of a matrix-market format file
                    if flag_matrix_contain_float_values:  # parse float data type
                        id_row, id_column, value = (
                            int(l_str[0]),
                            int(l_str[1]),
                            float(l_str[2]),
                        )
                    else:  # parse integer data type
                        id_row, id_column, value = tuple(int(float(e)) for e in l_str)

                    """ 1-based > 0-based coordinates """
                    id_row -= 1
                    id_column -= 1
                    """ write a record to the new matrix file only when both id_row and id_column belongs to filtered id_rows and id_columns """
                    if (
                        id_row in dict_id_row_previous_to_id_row_current
                        and id_column in dict_id_column_previous_to_id_column_current
                    ):
                        newfile.write(
                            (
                                " ".join(
                                    tuple(
                                        map(
                                            str,
                                            [
                                                dict_id_row_previous_to_id_row_current[
                                                    id_row
                                                ]
                                                + 1,
                                                dict_id_column_previous_to_id_column_current[
                                                    id_column
                                                ]
                                                + 1,
                                                value,
                                            ],
                                        )
                                    )
                                )
                                + "\n"
                            ).encode()
                        )  # map id_row and id_column of the previous matrix to those of the filtered matrix (new matrix) # 0-based > 1-based coordinates
                        int_n_entries += 1  # update the total number of entries written by the current process
                    line = file.readline().decode()  # read the next line
    return int_n_entries  # returns the total number of entries written by the current process


def MTX_10X_Filter(
    path_folder_mtx_10x_input,
    path_folder_mtx_10x_output,
    min_counts=None,
    min_features=None,
    min_cells=None,
    l_features=None,
    l_cells=None,
    verbose=False,
    function_for_adjusting_thresholds=None,
    int_num_threads=15,
    flag_split_mtx=True,
    int_max_num_entries_for_chunk=10000000,
):
    """# 2022-08-20 10:23:28
    # hyunsu-an
    read 10x count matrix and filter matrix based on several thresholds
    'path_folder_mtx_10x_input' : a folder containing files for the input 10x count matrix
    'path_folder_mtx_10x_output' : a folder containing files for the input 10x count matrix

    Only the threshold arguments for either cells ( 'min_counts', 'min_features' ) or features ( 'min_cells' ) can be given at a time.

    'min_counts' : the minimum number of total counts for a cell to be included in the output matrix
    'min_features' : the minimum number of features for a cell to be included in the output matrix
    'min_cells' : the minimum number of cells for a feature to be included in the output matrix
    'l_features' : a list of features (values in the first column of 'features.tsv.gz') to include. All other features will be excluded from the output matrix. (default: None) If None is given, include all features in the output matrix.
    'l_cells' : a list of cells (values in the first column of 'barcodes.tsv.gz') to include. All other cells will be excluded from the output matrix. (default: None) If None is given, include all cells in the output matrix.
    'int_num_threads' : when 'int_num_threads' is 1, does not use the multiprocessing  module for parallel processing
    'function_for_adjusting_thresholds' : a function for adjusting thresholds based on the summarized metrics. Useful when the exact threshold for removing empty droplets are variable across the samples. the function should receive arguments and return values in the following structure:
                                        min_counts_new, min_features_new, min_cells_new = function_for_adjusting_thresholds( path_folder_mtx_10x_output, min_counts, min_features, min_cells )
    """

    """ handle inputs """
    if path_folder_mtx_10x_input[-1] != "/":
        path_folder_mtx_10x_input += "/"
    if path_folder_mtx_10x_output[-1] != "/":
        path_folder_mtx_10x_output += "/"
    if ((min_counts is not None) or (min_features is not None)) and (
        min_cells is not None
    ):  # check whether thresholds for both cells and features were given (thresdholds for either cells or features can be given at a time)
        if verbose:
            logger.info(
                "[MTX_10X_Filter] (error) no threshold is given or more thresholds for both cells and features are given. (Thresdholds for either cells or features can be given at a time.)"
            )
        return -1
    # create an output folder
    filesystem_operations("mkdir", path_folder_mtx_10x_output, exist_ok=True)

    # define input file directories
    path_file_input_bc = f"{path_folder_mtx_10x_input}barcodes.tsv.gz"
    path_file_input_feature = f"{path_folder_mtx_10x_input}features.tsv.gz"
    path_file_input_mtx = f"{path_folder_mtx_10x_input}matrix.mtx.gz"

    str_data_type = _MTX_Detect_data_type(
        path_file_input_mtx
    )  # retrieve the data type of the matrix in string format

    # check whether all required files are present
    if sum(
        list(
            not filesystem_operations("exists", path_folder)
            for path_folder in [
                path_file_input_bc,
                path_file_input_feature,
                path_file_input_mtx,
            ]
        )
    ):
        if verbose:
            logger.info(f"required file(s) is not present at {path_folder_mtx_10x}")

    """ read barcode and feature information """
    df_bc = pd.read_csv(path_file_input_bc, sep="\t", header=None)
    df_bc.columns = ["barcode"]
    df_feature = pd.read_csv(path_file_input_feature, sep="\t", header=None)
    df_feature.columns = ["id_feature", "feature", "feature_type"]

    """ split input mtx file into multiple files """
    l_path_file_mtx_10x = MTX_10X_Split(
        path_folder_mtx_10x_input,
        int_max_num_entries_for_chunk=int_max_num_entries_for_chunk,
        flag_split_mtx=flag_split_mtx,
    )

    """ summarizes counts """
    dict_data = MTX_10X_Summarize_Counts(
        path_folder_mtx_10x_input,
        verbose=verbose,
        int_num_threads=int_num_threads,
        flag_split_mtx=flag_split_mtx,
        int_max_num_entries_for_chunk=int_max_num_entries_for_chunk,
    )
    (
        dict_id_column_to_count,
        dict_id_column_to_n_features,
        dict_id_row_to_count,
        dict_id_row_to_n_cells,
        dict_id_row_to_log_transformed_count,
    ) = (
        dict_data["dict_id_column_to_count"],
        dict_data["dict_id_column_to_n_features"],
        dict_data["dict_id_row_to_count"],
        dict_data["dict_id_row_to_n_cells"],
        dict_data["dict_id_row_to_log_transformed_count"],
    )  # parse 'dict_data'

    """ adjust thresholds based on the summarized metrices (if a function has been given) """
    if function_for_adjusting_thresholds is not None:
        min_counts, min_features, min_cells = function_for_adjusting_thresholds(
            path_folder_mtx_10x_input, min_counts, min_features, min_cells
        )

    """ filter row or column that do not satisfy the given thresholds """
    if min_counts is not None:
        dict_id_column_to_count = dict(
            (k, dict_id_column_to_count[k])
            for k in dict_id_column_to_count
            if dict_id_column_to_count[k] >= min_counts
        )
    if min_features is not None:
        dict_id_column_to_n_features = dict(
            (k, dict_id_column_to_n_features[k])
            for k in dict_id_column_to_n_features
            if dict_id_column_to_n_features[k] >= min_features
        )
    if min_cells is not None:
        dict_id_row_to_n_cells = dict(
            (k, dict_id_row_to_n_cells[k])
            for k in dict_id_row_to_n_cells
            if dict_id_row_to_n_cells[k] >= min_cells
        )

    """ retrieve id_row and id_column that satisfy the given thresholds """
    set_id_column = set(dict_id_column_to_count).intersection(
        set(dict_id_column_to_n_features)
    )
    set_id_row = set(dict_id_row_to_n_cells)

    """ exclude cells and features not present in the input lists (if the lists were given)  """
    if l_cells is not None:
        dict_barcode_to_id_column = dict(
            (barcode, id_column)
            for id_column, barcode in enumerate(df_bc.barcode.values)
        )
        set_id_column = set_id_column.intersection(
            set(
                dict_barcode_to_id_column[barcode]
                for barcode in set(l_cells)
                if barcode in dict_barcode_to_id_column
            )
        )
        del dict_barcode_to_id_column
    if l_features is not None:
        dict_id_feature_to_id_row = dict(
            (id_feature, id_row)
            for id_row, id_feature in enumerate(df_feature.id_feature.values)
        )
        set_id_row = set_id_row.intersection(
            set(
                dict_id_feature_to_id_row[id_feature]
                for id_feature in set(l_features)
                if id_feature in dict_id_feature_to_id_row
            )
        )
        del dict_id_feature_to_id_row

    """ report the number of cells or features that will be filtered out """
    if verbose:
        int_n_bc_filtered = len(df_bc) - len(set_id_column)
        if int_n_bc_filtered > 0:
            logger.info(
                f"{int_n_bc_filtered}/{len( df_bc )} barcodes will be filtered out"
            )
        int_n_feature_filtered = len(df_feature) - len(set_id_row)
        if int_n_feature_filtered > 0:
            logger.info(
                f"{int_n_feature_filtered}/{len( df_feature )} features will be filtered out"
            )

    """ retrieve a mapping between previous id_column to current id_column """
    global dict_id_column_previous_to_id_column_current, dict_id_row_previous_to_id_row_current  # use global variables for multiprocessing
    df_bc = df_bc.loc[list(set_id_column)]
    df_bc.index.name = "id_column_previous"
    df_bc.reset_index(drop=False, inplace=True)
    df_bc["id_column_current"] = np.arange(len(df_bc))
    dict_id_column_previous_to_id_column_current = df_bc.set_index(
        "id_column_previous"
    ).id_column_current.to_dict()
    bk.PICKLE_Write(
        f"{path_folder_mtx_10x_output}dict_id_column_previous_to_id_column_current.pickle",
        dict_id_column_previous_to_id_column_current,
    )  # save id_feature to index_feature mapping
    """ retrieve a mapping between previous id_row to current id_row """
    df_feature = df_feature.loc[list(set_id_row)]
    df_feature.index.name = "id_row_previous"
    df_feature.reset_index(drop=False, inplace=True)
    df_feature["id_row_current"] = np.arange(len(df_feature))
    dict_id_row_previous_to_id_row_current = df_feature.set_index(
        "id_row_previous"
    ).id_row_current.to_dict()
    bk.PICKLE_Write(
        f"{path_folder_mtx_10x_output}dict_id_row_previous_to_id_row_current.pickle",
        dict_id_row_previous_to_id_row_current,
    )  # save id_feature to index_feature mapping

    """ save barcode file """
    df_bc.to_csv(
        f"{path_folder_mtx_10x_output}barcodes.tsv.gz",
        columns=["barcode"],
        sep="\t",
        index=False,
        header=False,
    )
    del df_bc

    """ save feature file """
    df_feature[["id_feature", "feature", "feature_type"]].to_csv(
        f"{path_folder_mtx_10x_output}features.tsv.gz",
        sep="\t",
        index=False,
        header=False,
    )  # save as a file
    del df_feature

    """ write a filtered matrix.mtx.gz for each split mtx file using multiple processes and retrieve the total number of entries written by each process """
    # compose inputs for multiprocessing
    df_input = pd.DataFrame(
        {
            "path_file_mtx_10x": l_path_file_mtx_10x,
            "index_mtx_10x": np.arange(len(l_path_file_mtx_10x)),
        }
    )

    l_int_n_entries = bk.Multiprocessing(
        df_input,
        __MTX_10X_Filter__filter_mtx_10x__,
        int_num_threads,
        global_arguments=[path_folder_mtx_10x_output, str_data_type],
    )
    # retrieve the total number of entries
    int_total_n_entries = sum(l_int_n_entries)

    """ combine parts and add the MTX file header to compose a combined mtx file """
    df_file = bk.GLOB_Retrive_Strings_in_Wildcards(
        f"{path_folder_mtx_10x_output}matrix.mtx.gz.*.gz"
    )
    df_file.wildcard_0 = df_file.wildcard_0.astype(int)
    df_file.sort_values("wildcard_0", inplace=True)

    # write header
    path_file_header = f"{path_folder_mtx_10x_output}matrix.mtx.header.txt.gz"
    with gzip.open(path_file_header, "wb") as newfile:
        newfile.write(
            f"%%MatrixMarket matrix coordinate {str_data_type} general\n%\n{len( dict_id_row_previous_to_id_row_current )} {len( dict_id_column_previous_to_id_column_current )} {int_total_n_entries}\n".encode()
        )
    bk.OS_Run(
        ["cat", path_file_header] + list(df_file.path.values),
        path_file_stdout=f"{path_folder_mtx_10x_output}matrix.mtx.gz",
        stdout_binary=True,
        return_output=False,
    )  # combine the output mtx files in the order # does not delete temporary files if 'flag_split_mtx' is True

    # write a flag indicating that the current output directory contains split mtx files
    with open(f"{path_folder_mtx_10x_output}matrix.mtx.gz.split.flag", "w") as file:
        file.write("completed")


def MTX_10X_Identify_Highly_Variable_Features(
    path_folder_mtx_10x_input,
    int_target_sum=10000,
    verbose=False,
    int_num_threads=15,
    flag_split_mtx=True,
    int_max_num_entries_for_chunk=10000000,
):
    """# 2022-03-16 17:18:44
    calculate variance from log-transformed normalized counts using 'MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr' and rank features based on how each feature is variable compared to other features with similar means.
    Specifically, polynomial of degree 2 will be fitted to variance-mean relationship graph in order to captures the relationship between variance and mean.

    'name_col_for_mean', 'name_col_for_variance' : name of columns of 'df_summary' returned by 'MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr' that will be used to infer highly variable features. By defaults, mean and variance of log-transformed normalized counts will be used.
    """

    # calculate variance and means and load the result
    df_summary = MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr(
        path_folder_mtx_10x_input,
        int_target_sum=int_target_sum,
        int_num_threads=int_num_threads,
        verbose=verbose,
        flag_split_mtx=flag_split_mtx,
    )

    # calculate scores for identifying highly variable features for the selected set of count data types: [ 'log_transformed_normalized_count', 'log_transformed_count' ]
    for name_type in ["log_transformed_normalized_count", "log_transformed_count"]:
        name_col_for_mean, name_col_for_variance = (
            f"mean_{name_type}",
            f"variance_of_{name_type}",
        )
        # retrieve the relationship between mean and variance
        arr_mean = df_summary[name_col_for_mean].values
        arr_var = df_summary[name_col_for_variance].values
        mean_var_relationship_fit = np.polynomial.polynomial.Polynomial.fit(
            arr_mean, arr_var, 2
        )

        # calculate the deviation from the estimated variance from the mean
        arr_ratio_of_variance_to_expected_variance_from_mean = np.zeros(len(df_summary))
        arr_diff_of_variance_to_expected_variance_from_mean = np.zeros(len(df_summary))
        for i in range(len(df_summary)):  # iterate list of means of the features
            var, mean = arr_var[i], arr_mean[i]  # retrieve var and mean
            var_expected = mean_var_relationship_fit(
                mean
            )  # calculate expected variance from the mean
            if (
                var_expected == 0
            ):  # handle the case when the current expected variance is zero
                arr_ratio_of_variance_to_expected_variance_from_mean[i] = 1
                arr_diff_of_variance_to_expected_variance_from_mean[i] = 0
            else:
                arr_ratio_of_variance_to_expected_variance_from_mean[i] = (
                    var / var_expected
                )
                arr_diff_of_variance_to_expected_variance_from_mean[i] = (
                    var - var_expected
                )

        df_summary[
            f"float_ratio_of_variance_to_expected_variance_from_mean_from_{name_type}"
        ] = arr_ratio_of_variance_to_expected_variance_from_mean
        df_summary[
            f"float_diff_of_variance_to_expected_variance_from_mean_{name_type}"
        ] = arr_diff_of_variance_to_expected_variance_from_mean

        # calculate the product of the ratio and difference of variance to expected variance for scoring and sorting highly variable features
        df_summary[f"float_score_highly_variable_feature_from_{name_type}"] = (
            df_summary[
                f"float_ratio_of_variance_to_expected_variance_from_mean_from_{name_type}"
            ]
            * df_summary[
                f"float_diff_of_variance_to_expected_variance_from_mean_{name_type}"
            ]
        )

    df_summary["float_score_highly_variable_feature"] = list(
        np.prod(arr_val) if sum(np.sign(arr_val) < 0) == 0 else 0
        for arr_val in df_summary[
            [
                "float_score_highly_variable_feature_from_log_transformed_normalized_count",
                "float_score_highly_variable_feature_from_log_transformed_count",
            ]
        ].values
    )
    return df_summary


""" newly written functions """


def is_binary_stream(f):
    """# 2022-05-01 01:57:10
    check whether a given stream is a binary stream"""
    if hasattr(f, "mode"):  # if given stream is file
        return "b" in f.mode
    else:
        return isinstance(f, (io.RawIOBase, io.BufferedIOBase))


def MTX_Get_path_essential_files(path_folder_mtx_10x_input):
    """# 2022-04-30 16:28:15
    get paths of essential files for the given matrix folder ('path_folder_mtx_10x_input', currently only supports 10X-GEX-formated output matrix)
    """
    # define input file paths
    path_file_input_bc = f"{path_folder_mtx_10x_input}barcodes.tsv.gz"
    path_file_input_feature = f"{path_folder_mtx_10x_input}features.tsv.gz"
    path_file_input_mtx = f"{path_folder_mtx_10x_input}matrix.mtx.gz"
    # check whether input files exist
    for path_file in [path_file_input_bc, path_file_input_feature, path_file_input_mtx]:
        if not filesystem_operations("exists", path_file):
            raise OSError(f"{path_file} does not exist")
    return path_file_input_bc, path_file_input_feature, path_file_input_mtx


def Merge_Sort_Files(file_output, *l_iterator_decorated_file_input):
    """# 2022-05-01 02:23:09
    Merge sort input files (should be sorted) without loading the complete contents on memory.
    'path_file_output' : output file handle/stream
    'l_iterator_decorated_file_input' : a list of iterators based on input file handles (or streams). each iterator should yield the following tuple: (key_for_sorting, content_that_will_be_written_in_the_output_file). This function does not check whether the datatype of the 'content_that_will_be_written_in_the_output_file' matches that of 'path_file_output'
    """
    import heapq

    # handle invalid case
    if len(l_iterator_decorated_file_input) == 0:
        return -1
    # perform merge sorting
    for r in heapq.merge(*l_iterator_decorated_file_input):
        file_output.write(
            r[1]
        )  # assumes the 'iterator_decorated_file_input' returns appropriate datatype (either bytes or string) for the output file


def __Merge_Sort_MTX_10X__(
    path_file_output,
    *l_path_file_input,
    flag_ramtx_sorted_by_id_feature=True,
    flag_delete_input_file_upon_completion=False,
):
    """# 2022-05-01 02:25:07
    merge sort mtx files
    'path_file_output' and 'l_path_file_input'  : either file path or file handles

    'flag_ramtx_sorted_by_id_feature' : if True, sort by 'id_feature'. if False, sort by 'id_cell'
    """
    # process arguments for input files
    if isinstance(l_path_file_input[0], str):  # if paths are given as input files
        flag_input_binary = (
            l_path_file_input[0].rsplit(".", 1)[1].lower() == "gz"
        )  # automatically detect gzipped input file # determined gzipped status by only looking at the first file
        l_file_input = list(
            gzip.open(path_file, "rb") if flag_input_binary else open(path_file, "r")
            for path_file in l_path_file_input
        )
    else:
        flag_input_binary = is_binary_stream(l_file_input[0])  # detect binary stream
        l_file_input = l_path_file_input
    # process argument for output file
    if isinstance(path_file_output, str):  # if path was given as an output file
        flag_output_is_file = True
        flag_output_binary = (
            path_file_output.rsplit(".", 1)[1].lower() == "gz"
        )  # automatically detect gzipped input file # determined gzipped status by only looking at the first file
        file_output = (
            gzip.open(path_file_output, "wb")
            if flag_output_binary
            else open(path_file_output, "w")
        )
    else:
        flag_output_is_file = False
        flag_output_binary = is_binary_stream(path_file_output)  # detect binary stream
        file_output = path_file_output

    # define a function for decorating mtx record
    def __decorate_mtx_file__(file):
        while True:
            line = file.readline()
            if len(line) == 0:
                break
            """ parse a mtx record """
            line_decoded = line.decode() if flag_input_binary else line
            index_row, index_column, float_value = (
                (line_decoded).strip().split()
            )  # parse a record of a matrix-market format file
            index_row, index_column, float_value = (
                int(index_row),
                int(index_column),
                float(float_value),
            )  # 0-based coordinates
            yield index_row if flag_ramtx_sorted_by_id_feature else index_column, (
                (line if flag_input_binary else line.encode())
                if flag_output_binary
                else line_decoded
            )

    Merge_Sort_Files(
        file_output, *list(__decorate_mtx_file__(file) for file in l_file_input)
    )  # perform merge sorting

    # if the output file is stream, does not close the stream # only close the output if the output file was an actual file
    if flag_output_is_file:
        file_output.close()

    """ delete input files once merge sort is completed if 'flag_delete_input_file_upon_completion' is True """
    if flag_delete_input_file_upon_completion and isinstance(
        l_path_file_input[0], str
    ):  # if paths are given as input files
        for path_file in l_path_file_input:
            filesystem_operations("rm", path_file)


def __Merge_Sort_and_Index_MTX_10X__(
    path_file_output,
    *l_path_file_input,
    flag_ramtx_sorted_by_id_feature=True,
    flag_delete_input_file_upon_completion=False,
):
    """# 2022-05-01 02:25:07
    merge sort mtx files into a single mtx uncompressed file and index entries in the combined mtx file while writing the file
    'path_file_output' : should be a file path, file handle (or stream) for non-binary (text) output
    'l_path_file_input'

    'flag_ramtx_sorted_by_id_feature' : if True, sort by 'id_feature'. if False, sort by 'id_cell'
    """
    import heapq

    # process arguments for input files
    if isinstance(l_path_file_input[0], str):  # if paths are given as input files
        flag_input_binary = (
            l_path_file_input[0].rsplit(".", 1)[1].lower() == "gz"
        )  # automatically detect gzipped input file # determined gzipped status by only looking at the first file
        l_file_input = list(
            gzip.open(path_file, "rb") if flag_input_binary else open(path_file, "r")
            for path_file in l_path_file_input
        )
    else:
        flag_input_binary = is_binary_stream(l_file_input[0])  # detect binary stream
        l_file_input = l_path_file_input
    # process argument for output file
    if isinstance(path_file_output, str):  # if path was given as an output file
        flag_output_is_file = True
        flag_output_binary = (
            path_file_output.rsplit(".", 1)[1].lower() == "gz"
        )  # automatically detect gzipped input file # determined gzipped status by only looking at the first file
        file_output = (
            gzip.open(path_file_output, "wb")
            if flag_output_binary
            else open(path_file_output, "w")
        )
    else:
        flag_output_is_file = False
        flag_output_binary = is_binary_stream(path_file_output)  # detect binary stream
        file_output = path_file_output

    if flag_output_binary:  # the output file should be non-binary stream/file
        raise OSError("the output file should be non-binary stream/file")

    # define and open index output file
    path_file_index_output = f"{path_file_output}.idx.tsv.gz"
    file_index_output = gzip.open(path_file_index_output, "wb")
    file_index_output.write(
        ("\t".join(["index_entry", "int_pos_start", "int_pos_end"]) + "\n").encode()
    )  # write the header of the index file

    # define a function for decorating mtx record
    def __decorate_mtx_file__(file):
        while True:
            line = file.readline()
            if len(line) == 0:
                break
            """ parse a mtx record """
            line_decoded = line.decode() if flag_input_binary else line
            index_row, index_column, float_value = (
                (line_decoded).strip().split()
            )  # parse a record of a matrix-market format file
            index_row, index_column, float_value = (
                int(index_row),
                int(index_column),
                float(float_value),
            )  # 0-based coordinates
            yield index_row if flag_ramtx_sorted_by_id_feature else index_column, (
                (line if flag_input_binary else line.encode())
                if flag_output_binary
                else line_decoded
            )

    # perform merge sorting
    index_entry_currently_being_written = -1
    int_num_character_written_for_index_entry_currently_being_written = 0
    int_total_num_character_written = 0
    for r in heapq.merge(*list(__decorate_mtx_file__(file) for file in l_file_input)):
        if (
            index_entry_currently_being_written != r[0]
        ):  # if current index_entry is different from the previous one, which mark the change of sorted blocks (a block has the same id_entry), record the data for the previous block and initialze data for the next block
            if (
                index_entry_currently_being_written > 0
            ):  # check whether 'index_entry_currently_being_written' is valid (ignore 'dummy' or default value that was used for initialization)
                file_index_output.write(
                    (
                        "\t".join(
                            map(
                                str,
                                [
                                    index_entry_currently_being_written,
                                    int_total_num_character_written,
                                    int_total_num_character_written
                                    + int_num_character_written_for_index_entry_currently_being_written,
                                ],
                            )
                        )
                        + "\n"
                    ).encode()
                )  # write information required for indexing
            int_total_num_character_written += int_num_character_written_for_index_entry_currently_being_written  # update 'int_total_num_character_written'
            # initialize data for index of the next 'index_entry'
            index_entry_currently_being_written = r[0]  # update current index_entry
            int_num_character_written_for_index_entry_currently_being_written = 0  # reset the count of characters (which is equal to the number of bytes for any mtx records, because they only contains numeric characters)
        int_num_character_written_for_index_entry_currently_being_written += file_output.write(
            r[1]
        )  # assumes the 'iterator_decorated_file_input' returns appropriate datatype (either bytes or string) for the output file # count the number of characters written for the current index_row

    # write the record for the last block
    file_index_output.write(
        (
            "\t".join(
                map(
                    str,
                    [
                        index_entry_currently_being_written,
                        int_total_num_character_written,
                        int_total_num_character_written
                        + int_num_character_written_for_index_entry_currently_being_written,
                    ],
                )
            )
            + "\n"
        ).encode()
    )  # write information required for indexing
    # close index file
    file_index_output.close()
    # if the output file is stream, does not close the stream # only close the output if the output file was an actual file
    if flag_output_is_file:
        file_output.close()

    """ delete input files once merge sort is completed if 'flag_delete_input_file_upon_completion' is True """
    if flag_delete_input_file_upon_completion and isinstance(
        l_path_file_input[0], str
    ):  # if paths are given as input files
        for path_file in l_path_file_input:
            filesystem_operations("rm", path_file)


""" methods for handling 10X matrix objects """


def Convert_df_count_to_MTX_10X(
    path_file_df_count,
    path_folder_mtx_10x_output,
    chunksize=500000,
    flag_debugging=False,
    inplace=False,
):
    """# 2022-06-02 01:43:01
    convert df_count (scarab output) to 10X MTX (matrix market) format in a memory-efficient manner.

    'path_file_df_count' : file path to 'df_count'
    """
    # create a temporary output folder
    path_folder_temp = f"{path_folder_mtx_10x_output}temp_{bk.UUID( )}/"
    filesystem_operations("mkdir", path_folder_temp, exist_ok=True)

    # retrieve unique feature/barcode information from df_count
    DF_Deduplicate_without_loading_in_memory(
        path_file_df_count,
        f"{path_folder_temp}_features.tsv.gz",
        l_col_for_identifying_duplicates=["feature", "id_feature"],
        str_delimiter="\t",
    )
    int_num_lines = DF_Deduplicate_without_loading_in_memory(
        path_file_df_count,
        f"{path_folder_temp}_barcodes.tsv.gz",
        l_col_for_identifying_duplicates=["barcode"],
        str_delimiter="\t",
    )  # collect the number of records

    # read features and barcode information
    df_barcode = pd.read_csv(
        f"{path_folder_temp}_barcodes.tsv.gz", sep="\t", usecols=["barcode"]
    )
    df_feature = pd.read_csv(
        f"{path_folder_temp}_features.tsv.gz",
        sep="\t",
        usecols=["feature", "id_feature"],
    )
    df_feature = df_feature.loc[:, ["id_feature", "feature"]]
    df_feature["10X_type"] = "Gene Expression"
    # save feature/cell metadata
    df_barcode.to_csv(
        f"{path_folder_temp}barcodes.tsv.gz", sep="\t", index=False, header=False
    )
    df_feature.to_csv(
        f"{path_folder_temp}features.tsv.gz", sep="\t", index=False, header=False
    )

    # retrieve barcode/feature to integer representation of barcode/feature mapping
    dict_to_int_barcode = dict(
        (e, i + 1) for i, e in enumerate(df_barcode.iloc[:, 0].values)
    )
    dict_to_int_feature = dict(
        (e, i + 1) for i, e in enumerate(df_feature.iloc[:, 0].values)
    )

    int_num_features, int_num_barcodes, int_num_records = (
        len(df_feature),
        len(df_barcode),
        int_num_lines,
    )  # retrieve metadata of the output matrix
    del df_feature, df_barcode  # delete objects

    # write mtx file
    with gzip.open(f"{path_folder_temp}matrix.mtx.gz", "wb") as newfile:
        newfile.write(
            f"""%%MatrixMarket matrix coordinate integer general\n%\n{int_num_features} {int_num_barcodes} {int_num_records}\n""".encode()
        )
        # iterate through each chunk
        for df_chunk in pd.read_csv(
            path_file_df_count,
            iterator=True,
            header=0,
            chunksize=chunksize,
            sep="\t",
            usecols=["id_feature", "barcode", "read_count"],
        ):
            df_chunk = df_chunk[
                ["id_feature", "barcode", "read_count"]
            ]  # reorder columns
            df_chunk["id_feature"] = df_chunk.id_feature.apply(
                bk.Map(dict_to_int_feature).a2b
            )
            df_chunk["barcode"] = df_chunk.barcode.apply(
                bk.Map(dict_to_int_barcode).a2b
            )
            df_chunk.to_csv(newfile, sep=" ", header=None, index=False)

    # export result files
    for name_file in ["features.tsv.gz", "barcodes.tsv.gz", "matrix.mtx.gz"]:
        filesystem_operations(
            "mv",
            f"{path_folder_temp}{name_file}",
            f"{path_folder_mtx_10x_output}{name_file}",
        )
    # delete temporary folder
    filesystem_operations("rm", path_folder_temp)


""" method for compressing and decompressing blocks of data """
# settings
"""
file_formats : [ 
    'mtx_gzipped' : 10X matrix format. (pros) the RAMtx can be read by other program that can read 10X matrix file, small disk size (cons) very slow write speed, slow read speed
    'pickle' : uncompressed python pickle format. (pros) very fast write speed, very fast read speed. (cons) 5~10 times larger disk usage, python-specific data format
    'pickle_gzipped' : gzipped python pickle format. (pros) fast read speed. disk usage is 20~50% smaller than 10X matrix file. the most efficient storage format. (cons) very slow write speed, python-specific data format
    'feather' : uncompressed Apache Arrow feather storage format for DataFrames. (pros) very fast write speed, fast read speed, language-agnostic (R, Python, Julia, JS, etc.). (cons) ~2 times larger disk usage
    'feather_lz4' : LZ4 compressed (a default compression of 'feather') Apache Arrow feather storage format for DataFrames. (pros) very fast write speed, fast read speed, language-agnostic (R, Python, Julia, JS, etc.). (cons) ~2 times larger disk usage.
]
"""
_dict_file_format_to_ext = {
    "mtx_gzipped": "mtx.gz",
    "pickle": "pickle.stacked",
    "pickle_gzipped": "pickle.gz.stacked",
    "feather": "feather.stacked",
    "feather_lz4": "feather_lz4.stacked",
    "index": "idx.tsv.gz",
}


def base64_decode(str_content):
    """# 2022-07-04 23:19:18
    receive base64-encoded string and return decoded bytes
    """
    return base64.b64decode(str_content.encode("ascii"))


def base64_encode(byte_content):
    """# 2022-07-04 23:19:18
    receive bytes and return base64-encoded string
    """
    return base64.b64encode(byte_content).decode("ascii")


def gzip_bytes(bytes_content):
    """# 2022-05-24 23:43:36
    gzip the given bytes

    inputs:
    'bytes_content' : input bytes

    returns:
    'bytes_content_gzipped' : gzipped contents, number of bytes written

    """
    # compress the data
    gzip_file = io.BytesIO()
    with gzip.GzipFile(fileobj=gzip_file, mode="w") as file:
        file.write(bytes_content)

    # retrieve the compressed content
    gzip_file.seek(0)
    bytes_content_gzipped = gzip_file.read()
    gzip_file.close()
    return bytes_content_gzipped


def gunzip_bytes(bytes_content_gzipped):
    """# 2022-05-24 23:43:36
    gzip the given bytes

    inputs:
    'bytes_content_gzipped' : input gzipped bytes

    returns:
    'bytes_content' : unzipped contents, number of bytes written

    """
    # uncompress the gzipped bytes
    with io.BytesIO() as gzip_file_content:
        gzip_file_content.write(bytes_content_gzipped)
        gzip_file_content.seek(0)
        with gzip.GzipFile(fileobj=gzip_file_content, mode="r") as gzip_file:
            bytes_content = gzip_file.read()
    return bytes_content


def _feather_bytes_to_df(bytes_content):
    """# 2022-05-25 01:50:46
    convert bytes to df using pyarrow.feather
    """
    # uncompress the gzipped bytes
    with io.BytesIO() as file:
        file.write(bytes_content)
        file.seek(0)
        df = pyarrow.feather.read_feather(file)
    return df


def _feather_df_to_bytes(df, compression="uncompressed"):
    """# 2022-05-25 01:56:37
    convert a python dataframe to bytes using pyarrow.feather with the given compression
    """
    # compress the data
    file = io.BytesIO()
    pyarrow.feather.write_feather(df, file, compression=compression)

    # retrieve the converted content
    file.seek(0)
    bytes_content = file.read()
    file.close()
    return bytes_content


def _pickle_bytes_to_obj(bytes_content):
    """# 2022-05-25 01:50:46
    convert bytes to df using pickle
    """
    # uncompress the gzipped bytes
    with io.BytesIO() as file:
        file.write(bytes_content)
        file.seek(0)
        obj = pickle.load(file)
    return obj


def _pickle_obj_to_bytes(obj):
    """# 2022-05-25 01:56:37
    convert a python dataframe to bytes using pickle
    """
    # compress the data
    file = io.BytesIO()
    pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    # retrieve the converted content
    file.seek(0)
    bytes_content = file.read()
    file.close()
    return bytes_content


def _bytes_mtx_to_df_mtx(
    bytes_mtx,
    dtype_of_row_and_col_indices=None,
    dtype_of_value=None,
    int_min_num_records_for_pandas_parsing=200,
):
    """# 2022-06-01 13:55:58
    convert bytes of a portion of matrix market file as a dataframe containing (arr_int_feature, arr_int_barcode, arr_value)

    'bytes_mtx': bytes of a portion of matrix market file. separator is ' '
    'dtype_of_row_and_col_indices' : set the dtype of column '0' (row index) and column '1' (col index)
    'dtype_of_value' : set the dtype of value
    """
    if (
        int_min_num_records_for_pandas_parsing > 0
        and bytes_mtx.count(b"\n") < int_min_num_records_for_pandas_parsing
    ):
        # parse mtx bytes using pandas module without using pandas module
        return _arrays_mtx_to_df_mtx(
            _bytes_mtx_to_arrays_mtx(
                bytes_mtx,
                dtype_of_row_and_col_indices,
                dtype_of_value,
                int_min_num_records_for_pandas_parsing,
            )
        )
    else:
        # parse mtx bytes using pandas module
        df = pd.read_csv(io.BytesIO(bytes_mtx), sep=" ", header=None)
        df[0] -= 1
        df[1] -= 1
        # convert dtypes
        df.columns = np.arange(3, dtype=np.uint8)  # set integer index for each columns
        df.index = np.zeros(len(df), dtype=np.uint8)  # ignore 'index' integers
        if (
            dtype_of_row_and_col_indices is not None
        ):  # change dtype of row and col indices
            df[0] = df[0].astype(dtype_of_row_and_col_indices)
            df[1] = df[1].astype(dtype_of_row_and_col_indices)
        if dtype_of_value is not None:  # change dtype of value
            df[2] = df[2].astype(dtype_of_value)
        return df


def _bytes_mtx_to_arrays_mtx(
    bytes_mtx,
    dtype_of_row_and_col_indices=None,
    dtype_of_value=None,
    int_min_num_records_for_pandas_parsing=200,
):
    """# 2022-06-01 13:56:07
    convert bytes of a portion of matrix market file as three arrays (arr_int_feature, arr_int_barcode, arr_value)

    'bytes_mtx': bytes of a portion of matrix market file. separator is ' '
    'dtype_of_row_and_col_indices' : set the dtype of column '0' (row index) and column '1' (col index)
    'dtype_of_value' : set the dtype of value
    'int_min_num_records_for_pandas_parsing' : the minimum number of records for parsing with pandas module.
    """
    if (
        int_min_num_records_for_pandas_parsing > 0
        and bytes_mtx.count(b"\n") < int_min_num_records_for_pandas_parsing
    ):
        # parse without using pandas module to avoid the overhead
        l_f, l_b, l_v = [], [], []
        for r in bytes_mtx.strip().split(b"\n"):
            f, b, v = r.split(b" ")
            l_f.append(f)
            l_b.append(b)
            l_v.append(v)
        # set default dtypes of bytes_mtx, which is mandatory
        if dtype_of_value is None:
            dtype_of_value = np.float64
        if dtype_of_row_and_col_indices is None:
            dtype_of_row_and_col_indices = np.int32
        arr_f = np.array(l_f, dtype=dtype_of_row_and_col_indices)
        arr_b = np.array(l_b, dtype=dtype_of_row_and_col_indices)
        arr_v = np.array(l_v, dtype=dtype_of_value)
        arr_f -= 1
        arr_b -= 1
        return arr_f, arr_b, arr_v
    else:
        return _df_mtx_to_arrays_mtx(
            _bytes_mtx_to_df_mtx(bytes_mtx, int_min_num_records_for_pandas_parsing=0),
            dtype_of_row_and_col_indices,
            dtype_of_value,
        )  # make sure the records are parsed with pandas module


def _arrays_mtx_to_df_mtx(
    arrays_mtx, dtype_of_row_and_col_indices=None, dtype_of_value=None
):
    """# 2022-05-25 16:59:28
    convert arrays mtx formats to dataframe

    dtype_of_row_and_col_indices = None, dtype_of_value = None : conversion of dtypes to use different dtypes in the output data
    """
    arr_int_feature, arr_int_barcode, arr_value = arrays_mtx
    df = pd.DataFrame(
        {0: arr_int_feature, 1: arr_int_barcode, 2: arr_value},
        index=np.zeros(len(arr_int_feature), dtype=np.uint8),
    )  # ignore 'index' integers # this will preserve the original data types
    df.columns = np.arange(3, dtype=np.uint8)  # set integer index for each columns
    # convert dtypes
    if dtype_of_row_and_col_indices is not None:  # change dtype of row and col indices
        df[0] = df[0].astype(dtype_of_row_and_col_indices)
        df[1] = df[1].astype(dtype_of_row_and_col_indices)
    if dtype_of_value is not None:  # change dtype of value
        df[2] = df[2].astype(dtype_of_value)
    return df


def _df_mtx_to_arrays_mtx(
    df_mtx, dtype_of_row_and_col_indices=None, dtype_of_value=None
):
    """# 2022-05-25 16:59:32
    convert dataframe mtx format to arrays mtx objects

    dtype_of_row_and_col_indices = None, dtype_of_value = None : conversion of dtypes to use different dtypes in the output data
    """
    # parse df as arrays # parsing individual columns will preserve dtypes, and will be much faster
    arr_int_feature = df_mtx[0].values
    arr_int_barcode = df_mtx[1].values
    arr_value = df_mtx[2].values
    # convert dtypes
    if dtype_of_row_and_col_indices is not None:  # change dtype of row and col indices
        arr_int_feature = arr_int_feature.astype(dtype_of_row_and_col_indices)
        arr_int_barcode = arr_int_barcode.astype(dtype_of_row_and_col_indices)
    if dtype_of_value is not None:  # change dtype of value
        arr_value = arr_value.astype(dtype_of_value)
    return (arr_int_feature, arr_int_barcode, arr_value)


def _arrays_mtx_to_bytes_mtx(arrays_mtx, str_format_value="{}"):
    """# 2022-05-25 10:43:01
    converts arrays of a matrix (0-based coordinates) to bytes_mtx

    'arrays_mtx' : input arrays of a mtx, ( arr_int_feature, arr_int_barcode, arr_value )
    'str_format_value' : a format string to encode value
    """
    arr_int_feature, arr_int_barcode, arr_value = arrays_mtx
    return (
        "\n".join(
            list(
                str(index_row)
                + " "
                + str(index_col)
                + " "
                + str_format_value.format(value)
                for index_row, index_col, value in zip(
                    arr_int_feature + 1, arr_int_barcode + 1, arr_value
                )
            )
        )
        + "\n"
    ).encode()  # 0>1-based coordinates


def _df_mtx_to_bytes_mtx(df_mtx, str_format_value="{}"):
    """# 2022-05-25 10:50:16
    converts arrays of a matrix (0-based coordinates) to bytes_mtx

    'df_mtx' : input dataframe of a mtx, columns: ( int_feature, int_barcode, value )
    'str_format_value' : a format string to encode value
    """
    return _arrays_mtx_to_bytes_mtx(
        _df_mtx_to_arrays_mtx(df_mtx), str_format_value=str_format_value
    )


def _arrays_mtx_to_arrays_mtx(
    arrays_mtx, dtype_of_row_and_col_indices=None, dtype_of_value=None
):
    """#2022-05-25 04:26:08
    change dtypes of arrays_mtx
    """
    # parse df as arrays
    (
        arr_int_feature,
        arr_int_barcode,
        arr_value,
    ) = arrays_mtx  # col = barcode, row = feature
    # convert dtypes
    if dtype_of_row_and_col_indices is not None:  # change dtype of row and col indices
        arr_int_feature = arr_int_feature.astype(dtype_of_row_and_col_indices)
        arr_int_barcode = arr_int_barcode.astype(dtype_of_row_and_col_indices)
    if dtype_of_value is not None:  # change dtype of value
        arr_value = arr_value.astype(dtype_of_value)
    return arr_int_feature, arr_int_barcode, arr_value


""" methods for retrieving appropriate functions based on input file format and the task """


def _get_func_bytes_mtx_to_processed_bytes_and_other_settings_based_on_file_format(
    file_format, dtype_of_row_and_col_indices=None, dtype_of_value=None
):
    """# 2022-05-25 23:23:52
    return a function 'func_arrays_mtx_to_processed_bytes' and relevant settings for the given file_format

    returns:
    str_etx, str_ext_index, func_bytes_mtx_to_processed_bytes
    """
    str_etx = _dict_file_format_to_ext[file_format]
    str_ext_index = _dict_file_format_to_ext["index"]
    if file_format == "mtx_gzipped":
        func_bytes_mtx_to_processed_bytes = gzip_bytes
    elif file_format == "feather_lz4":

        def func_bytes_mtx_to_processed_bytes(bytes_mtx):
            return _feather_df_to_bytes(
                _bytes_mtx_to_df_mtx(
                    bytes_mtx, dtype_of_row_and_col_indices, dtype_of_value
                ),
                "lz4",
            )

    elif file_format == "feather":

        def func_bytes_mtx_to_processed_bytes(bytes_mtx):
            return _feather_df_to_bytes(
                _bytes_mtx_to_df_mtx(
                    bytes_mtx, dtype_of_row_and_col_indices, dtype_of_value
                ),
                "uncompressed",
            )

    elif file_format == "pickle":

        def func_bytes_mtx_to_processed_bytes(bytes_mtx):
            return _pickle_obj_to_bytes(
                _bytes_mtx_to_arrays_mtx(
                    bytes_mtx, dtype_of_row_and_col_indices, dtype_of_value
                )
            )

    elif file_format == "pickle_gzipped":

        def func_bytes_mtx_to_processed_bytes(bytes_mtx):
            return gzip_bytes(
                _pickle_obj_to_bytes(
                    _bytes_mtx_to_arrays_mtx(
                        bytes_mtx, dtype_of_row_and_col_indices, dtype_of_value
                    )
                )
            )

    return str_etx, str_ext_index, func_bytes_mtx_to_processed_bytes


def _get_func_arrays_mtx_to_processed_bytes_and_other_settings_based_on_file_format(
    file_format,
    str_format_value="{}",
    dtype_of_row_and_col_indices=None,
    dtype_of_value=None,
):
    """# 2022-05-25 23:21:44
    return a function 'func_arrays_mtx_to_processed_bytes' and relevant settings for the given file_format

    returns:
    str_etx, str_ext_index, func_arrays_mtx_to_processed_bytes
    """
    str_etx = _dict_file_format_to_ext[file_format]
    str_ext_index = _dict_file_format_to_ext["index"]
    if file_format == "mtx_gzipped":

        def func_arrays_mtx_to_processed_bytes(arrays_mtx):
            return gzip_bytes(_arrays_mtx_to_bytes_mtx(arrays_mtx, str_format_value))

    elif file_format == "feather_lz4":

        def func_arrays_mtx_to_processed_bytes(arrays_mtx):
            return _feather_df_to_bytes(
                _arrays_mtx_to_df_mtx(
                    arrays_mtx, dtype_of_row_and_col_indices, dtype_of_value
                ),
                "lz4",
            )

    elif file_format == "feather":

        def func_arrays_mtx_to_processed_bytes(arrays_mtx):
            return _feather_df_to_bytes(
                _arrays_mtx_to_df_mtx(
                    arrays_mtx, dtype_of_row_and_col_indices, dtype_of_value
                ),
                "uncompressed",
            )

    elif file_format == "pickle":

        def func_arrays_mtx_to_processed_bytes(arrays_mtx):
            return _pickle_obj_to_bytes(
                _arrays_mtx_to_arrays_mtx(
                    arrays_mtx, dtype_of_row_and_col_indices, dtype_of_value
                )
            )

    elif file_format == "pickle_gzipped":

        def func_arrays_mtx_to_processed_bytes(arrays_mtx):
            return gzip_bytes(
                _pickle_obj_to_bytes(
                    _arrays_mtx_to_arrays_mtx(
                        arrays_mtx, dtype_of_row_and_col_indices, dtype_of_value
                    )
                )
            )

    return str_etx, str_ext_index, func_arrays_mtx_to_processed_bytes


def _get_func_processed_bytes_to_arrays_mtx_and_other_settings_based_on_file_format(
    file_format, dtype_of_row_and_col_indices=None, dtype_of_value=None
):
    """# 2022-05-26 00:25:44
    return a function 'func_processed_bytes_to_arrays_mtx' and relevant settings for the given file_format

    returns:
    str_etx, str_ext_index, func_processed_bytes_to_arrays_mtx
    """
    """ retrieve RAMtx format-specific import settings """
    str_ext = _dict_file_format_to_ext[file_format]
    str_ext_index = _dict_file_format_to_ext["index"]
    if file_format == "mtx_gzipped":

        def func_processed_bytes_to_arrays_mtx(bytes_content):
            return _bytes_mtx_to_arrays_mtx(
                gunzip_bytes(bytes_content),
                dtype_of_row_and_col_indices,
                dtype_of_value,
            )

    elif file_format == "feather_lz4":

        def func_processed_bytes_to_arrays_mtx(bytes_content):
            return _df_mtx_to_arrays_mtx(_feather_bytes_to_df(bytes_content))

    elif file_format == "feather":

        def func_processed_bytes_to_arrays_mtx(bytes_content):
            return _df_mtx_to_arrays_mtx(_feather_bytes_to_df(bytes_content))

    elif file_format == "pickle":

        def func_processed_bytes_to_arrays_mtx(bytes_content):
            return _pickle_bytes_to_obj(bytes_content)

    elif file_format == "pickle_gzipped":

        def func_processed_bytes_to_arrays_mtx(bytes_content):
            return _pickle_bytes_to_obj(gunzip_bytes(bytes_content))

    return str_ext, str_ext_index, func_processed_bytes_to_arrays_mtx


""" above functions will be moved below eventually """
""" miscellaneous functions """


def convert_numpy_dtype_number_to_number(e):
    """# 2022-08-22 15:46:33
    convert potentially numpy number to number. useful for JSON serialization, since it cannot serialize numbers in the numpy dtype
    """
    if np.issubdtype(type(e), np.floating):
        return float(e)
    elif np.issubdtype(type(e), np.integer):
        return int(e)
    else:
        return e


""" methods for logging purposes """


def installed_packages():
    """# 2022-12-01 21:24:03
    display the installed packages of scelephant
    """
    df_installed_packages = bk.PD_Select(
        bk.PIP_List_Packages(),
        index=[
            "s3fs",
            "fsspec",
            "umap-learn",
            "tensorflow",
            "igraph",
            "biobookshelf",
            "typing",
            "zarr",
            "numcodecs",
            "anndata",
            "scanpy",
            "shelve",
            "sklearn",
            "tarfile",
            "requests",
            "shutil",
            "numba",
            "tqdm",
            "umap",
            "hdbscan",
            "pgzip",
            "scipy",
            "pynndescent",
            "leidenalg",
            "sys",
            "os",
            "subprocess",
            "subprocess",
            "multiprocessing",
            "ctypes",
            "logging",
            "inspect",
            "copy",
            "collections",
            "ast",
            "pickle",
            "traceback",
            "mmap",
            "itertools",
            "math",
            "uuid",
            "gc",
            "time",
            "heapq",
            "datetime",
            "json",
            "numpy",
            "pandas",
            "matplotlib",
            "requests",
            "ftplib",
            "urllib",
            "importlib",
            "bokeh",
            "pysam",
            "plotly",
            "scanpy",
            "bitarray",
            "intervaltree",
            "statsmodels",
            "scipy",
            "upsetplot",
        ],
    )
    return df_installed_packages


""" methods for jupyter notebook interaction (IPython) """


def html_from_dict(dict_data: dict, name_dict: str = None):
    """# 2022-08-07 23:47:15
    compose a html page displaying the given dicionary by converting the dictionary to JSON format and visualizing JSON format using jsonTreeViewer lightweight javascript package.
    the main purpose of this function is to provide an interactive interface for exploration of an object using jupyter notebook's _repr_html_ method.

    'dict_data' : a dictionary that contains JSON-like data
    'name_dict' : name of the dictionary
    """
    str_uuid = (
        bk.UUID()
    )  # retrieve a unique id for this function call. returned HTML document will contain DOM elements with unique ids
    return (
        """
    <!DOCTYPE html>
    <html>
    <head>
    <link href="https://rawgit.com/summerstyle/jsonTreeViewer/master/libs/jsonTree/jsonTree.css" rel="stylesheet" />
    <script src="https://rawgit.com/summerstyle/jsonTreeViewer/master/libs/jsonTree/jsonTree.js"></script>
    <style>
    #wrapper_"""
        + str_uuid
        + """ li {
      list-style:none;
    }
    </style>
    </head>
    <body>
    <div>
    """
        + ("" if name_dict is None else f"{name_dict}")
        + """
    <div id="wrapper_"""
        + str_uuid
        + """"></div>
    </div>


    <script>
    // Get DOM-element for inserting json-tree
    var wrapper = document.getElementById("wrapper_"""
        + str_uuid
        + """");

    // Get json-data by javascript-object
    var data = """
        + json.dumps(dict_data)
        + """

    var tree = jsonTree.create(data, wrapper);
    </script></body></html>"""
    )


""" methods for handling tar.gz file """


def tar_create(path_file_output, path_folder_input):
    """# 2022-08-05 21:07:53
    create tar.gz file

    'path_file_output' : output tar.gz file
    'path_folder_input' : input folder for creation of a tar.gz file
    """
    import tarfile

    with tarfile.open(path_file_output, "w:gz") as tar:
        tar.add(path_folder_input, arcname=os.path.basename(path_folder_input))


def tar_extract(path_file_input, path_folder_output):
    """# 2022-08-05 21:07:53
    extract tar.gz file

    'path_file_output' : output tar.gz file
    'path_folder_input' : input folder for creation of a tar.gz file
    """
    import tarfile

    with tarfile.open(path_file_input, "r:gz") as tar:
        tar.extractall(path_folder_output)


""" methods for handling remote file """


def is_s3_url(url):
    """# 2022-12-02 18:23:18
    check whether the given url is s3uri (s3url)
    """
    # handle None value
    if url is None:
        return False
    return "s3://" == url[:5]


def is_http_url(url):
    """# 2022-12-02 18:23:18
    check whether the given url is HTTP URL
    """
    return "https://" == url[:8] or "http://" == url[:7]


def is_remote_url(url):
    """# 2022-12-02 18:31:45
    check whether a url is a remote resource
    """
    return is_s3_url(url) or is_http_url(url)


""" remote files over HTTP """


def http_response_code(url):
    """# 2022-08-05 22:27:27
    check http response code
    """
    import requests  # download from url

    status_code = None  # by default, 'status_code' is None
    try:
        r = requests.head(url)
        status_code = r.status_code  # record the status header
    except requests.ConnectionError:
        status_code = None
    return status_code


def http_download_file(url, path_file_local):
    """# 2022-08-05 22:14:30
    download file from the remote location to the local directory
    """
    import requests  # download from url

    with requests.get(url, stream=True) as r:
        with open(path_file_local, "wb") as f:
            shutil.copyfileobj(r.raw, f)


""" remote files over AWS S3 """


def s3_exists(s3url):
    """# 2022-12-02 18:15:49
    check whether a path/file exists in AWS S3
    """
    import s3fs

    fs = s3fs.S3FileSystem()
    return fs.exists(s3url)


def s3_download_file(s3url, path_file_local):
    """# 2022-12-02 18:15:44
    download file from the remote AWS S3 location to the local directory
    """
    import s3fs

    fs = s3fs.S3FileSystem()
    fs.download(s3url, path_file_local)


def s3_rm(s3url, recursive=False, **kwargs):
    """# 2022-12-03 23:48:26
    delete file (or an entire folder) from a AWS S3 location
    """
    import s3fs

    fs = s3fs.S3FileSystem()
    fs.rm(s3url, recursive=recursive, **kwargs)  # delete files


""" method and class for handling file system """


def filesystem_operations(
    method: Literal["exists", "rm", "glob", "mkdir", "mv", "cp", "isdir"],
    path_src: str,
    path_dest: Union[str, None] = None,
    flag_recursive: bool = True,
    dict_kwargs_credentials_s3: dict = dict(),
    **kwargs,
):
    """# 2022-12-04 00:57:45
    perform a file system operation (either Amazon S3 or local file system)

    method : Literal[
        'exists', # check whether a file or folder exists, given through 'path_src' arguments
        'rm', # remove file or folder, given through 'path_src' arguments
        'glob', # retrieve path of files matching the glob pattern, given through 'path_src' arguments
        'mkdir', # create a directory, given through 'path_src' arguments
        'mv', # move file or folder , given through 'path_src' and 'path_dest' arguments
        'cp', # copy file or folder , given through 'path_src' and 'path_dest' arguments
        'isdir', # check whether the given input is a file or directory
    ]

    kwargs :
        exist_ok : for 'mkdir' operation

    dict_kwargs_credentials_s3 : dict = dict( ) # the credentials for the Amazon S3 file system as keyworded arguments

    """
    if is_s3_url(path_src) or is_s3_url(
        path_dest
    ):  # if at least one path is s3 locations
        # %% Amazon s3 file system %%
        # load the file system
        import s3fs

        fs = s3fs.S3FileSystem(**dict_kwargs_credentials_s3)
        if method == "exists":
            return fs.exists(path_src, **kwargs)
        elif method == "rm":
            return fs.rm(path_src, recursive=flag_recursive, **kwargs)  # delete files
        elif method == "glob":
            return list(
                "s3://" + e for e in fs.glob(path_src, **kwargs)
            )  # 's3://' prefix should be added
        elif method == "mkdir":
            # use default 'exist_ok' value
            if "exist_ok" not in kwargs:
                kwargs["exist_ok"] = True
            return fs.makedirs(path_src, **kwargs)
        elif method == "mv":
            if not fs.exists(
                path_dest, **kwargs
            ):  # avoid overwriting of the existing file
                return fs.mv(path_src, path_dest, recursive=flag_recursive, **kwargs)
            else:
                return "destionation file already exists, exiting"
        elif method == "cp":
            if is_s3_url(path_src) and is_s3_url(path_dest):  # copy from s3 to s3
                return fs.copy(path_src, path_dest, recursive=flag_recursive, **kwargs)
            elif is_s3_url(path_src):  # copy from s3 to local
                return fs.get(path_src, path_dest, recursive=flag_recursive, **kwargs)
            elif is_s3_url(path_dest):  # copy from local to s3
                return fs.put(path_src, path_dest, recursive=flag_recursive, **kwargs)
        elif method == "isdir":
            return fs.isdir(path_src)
    elif is_http_url(path_src):  # for http
        # %% HTTP server %%
        if method == "exists":
            return (
                http_response_code(path_src) == 200
            )  # check whether http file (not tested for directory) exists
        else:
            return "not implemented"
    else:
        # %% local file system %%
        if method == "exists":
            return os.path.exists(path_src)
        elif method == "rm":
            if flag_recursive and os.path.isdir(
                path_src
            ):  # when the recursive option is active
                shutil.rmtree(path_src)
            else:
                os.remove(path_src)
        elif method == "glob":
            return glob.glob(path_src)
        elif method == "mkdir":
            # use default 'exist_ok' value
            if "exist_ok" not in kwargs:
                kwargs["exist_ok"] = True
            os.makedirs(path_src, exist_ok=kwargs["exist_ok"])
        elif method == "mv":
            shutil.move(path_src, path_dest)
        elif method == "cp":
            if flag_recursive and os.path.isdir(
                path_src
            ):  # when the recursive option is active
                shutil.copytree(path_src, path_dest)
            else:
                shutil.copyfile(path_src, path_dest)
        elif method == "isdir":
            return os.path.isdir(path_src)


def filesystem_server(
    pipe_receiver_input, pipe_sender_output, dict_kwargs_credentials_s3: dict = dict()
):
    """# 2022-12-05 18:49:06
    This function is for serving file-system operations in a spawned process for fork-safe operation

    dict_kwargs_credentials_s3 : dict = dict( ) # the credentials for the Amazon S3 file system as keyworded arguments
    """

    def __filesystem_operations(
        method: Literal["exists", "rm", "glob", "mkdir", "mv", "cp", "isdir"],
        path_src: str,
        path_dest: Union[str, None] = None,
        flag_recursive: bool = True,
        dict_kwargs_credentials_s3: dict = dict_kwargs_credentials_s3,
        **kwargs,
    ):
        """# 2022-12-04 00:57:45
        perform a file system operation (either Amazon S3 or local file system)

        method : Literal[
            'exists', # check whether a file or folder exists, given through 'path_src' arguments
            'rm', # remove file or folder, given through 'path_src' arguments
            'glob', # retrieve path of files matching the glob pattern, given through 'path_src' arguments
            'mkdir', # create a directory, given through 'path_src' arguments
            'mv', # move file or folder , given through 'path_src' and 'path_dest' arguments
            'cp', # copy file or folder , given through 'path_src' and 'path_dest' arguments
            'isdir', # check whether the given input is a file or directory
        ]

        kwargs :
            exist_ok : for 'mkdir' operation

        dict_kwargs_credentials_s3 : dict = dict( ) # the credentials for the Amazon S3 file system as keyworded arguments

        """
        if is_s3_url(path_src) or is_s3_url(
            path_dest
        ):  # if at least one path is s3 locations
            # %% Amazon s3 file system %%
            # load the file system
            import s3fs

            fs = s3fs.S3FileSystem(**dict_kwargs_credentials_s3)
            if method == "exists":
                return fs.exists(path_src, **kwargs)
            elif method == "rm":
                return fs.rm(
                    path_src, recursive=flag_recursive, **kwargs
                )  # delete files
            elif method == "glob":
                return list(
                    "s3://" + e for e in fs.glob(path_src, **kwargs)
                )  # 's3://' prefix should be added
            elif method == "mkdir":
                # use default 'exist_ok' value
                if "exist_ok" not in kwargs:
                    kwargs["exist_ok"] = True
                return fs.makedirs(path_src, **kwargs)
            elif method == "mv":
                if not fs.exists(
                    path_dest, **kwargs
                ):  # avoid overwriting of the existing file
                    return fs.mv(
                        path_src, path_dest, recursive=flag_recursive, **kwargs
                    )
                else:
                    return "destionation file already exists, exiting"
            elif method == "cp":
                if is_s3_url(path_src) and is_s3_url(path_dest):  # copy from s3 to s3
                    return fs.copy(
                        path_src, path_dest, recursive=flag_recursive, **kwargs
                    )
                elif is_s3_url(path_src):  # copy from s3 to local
                    return fs.get(
                        path_src, path_dest, recursive=flag_recursive, **kwargs
                    )
                elif is_s3_url(path_dest):  # copy from local to s3
                    return fs.put(
                        path_src, path_dest, recursive=flag_recursive, **kwargs
                    )
            elif method == "isdir":
                return fs.isdir(path_src)
        elif is_http_url(path_src):  # for http
            # %% HTTP server %%
            if method == "exists":
                return (
                    http_response_code(path_src) == 200
                )  # check whether http file (not tested for directory) exists
            else:
                return "not implemented"
        else:
            # %% local file system %%
            if method == "exists":
                return os.path.exists(path_src)
            elif method == "rm":
                if flag_recursive and os.path.isdir(
                    path_src
                ):  # when the recursive option is active
                    try:
                        shutil.rmtree(path_src)
                    except:
                        pass
                else:
                    os.remove(path_src)
            elif method == "glob":
                return glob.glob(path_src)
            elif method == "mkdir":
                # use default 'exist_ok' value
                if "exist_ok" not in kwargs:
                    kwargs["exist_ok"] = True
                os.makedirs(path_src, exist_ok=kwargs["exist_ok"])
            elif method == "mv":
                shutil.move(path_src, path_dest)
            elif method == "cp":
                if flag_recursive and os.path.isdir(
                    path_src
                ):  # when the recursive option is active
                    shutil.copytree(path_src, path_dest)
                else:
                    shutil.copyfile(path_src, path_dest)
            elif method == "isdir":
                return os.path.isdir(path_src)

    while True:
        e = pipe_receiver_input.recv()
        if e is None:  # exit if None is received
            break
        args, kwargs = e  # parse input
        pipe_sender_output.send(
            __filesystem_operations(*args, **kwargs)
        )  # return result


class FileSystemServer:
    """# 2022-12-05 18:49:02
    This class is for serving file-system operations ('filesystem_operations' function) in a spawned process or the current process for fork-safe operation

    dict_kwargs_credentials_s3 : dict = dict( ) # the credentials for the Amazon S3 file system as keyworded arguments

    flag_spawn : bool = True # if True, spawn a new process for file system operations. if False, perform file system operations in the current process.
        (both are blocking and synchronous. the difference is that file system operations that are not fork-safe can be performed in forked process by spawning a new process)
    """

    def __init__(
        self, flag_spawn: bool = False, dict_kwargs_credentials_s3: dict = dict()
    ):
        """# 2022-12-05 18:48:59"""
        # set read-only attributes
        self._flag_spawn = flag_spawn  # indicate that a process has been spawned

        # set attributes
        self._flag_is_terminated = False
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            # create pipes for interactions
            mpsp = mp.get_context("spawn")
            pipe_sender_input, pipe_receiver_input = mpsp.Pipe()
            pipe_sender_output, pipe_receiver_output = mpsp.Pipe()

            self._pipe_sender_input = pipe_sender_input
            self._pipe_receiver_output = pipe_receiver_output

            # start the process for file-system operations
            p = mpsp.Process(
                target=filesystem_server,
                args=(
                    pipe_receiver_input,
                    pipe_sender_output,
                    dict_kwargs_credentials_s3,
                ),
            )
            p.start()
            self._p = p

    @property
    def flag_spawn(self):
        """# 2022-12-05 22:26:33
        return a flag indicating whether a process has been spawned and interacting with the current object or not.
        """
        return self._flag_spawn

    def filesystem_operations(self, *args, **kwargs):
        """# 2022-12-05 22:34:49
        a wrapper of 'filesystem_operations' function
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send((args, kwargs))  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            # run file system operations in the current process
            return filesystem_operations(*args, **kwargs)

    def terminate(self):
        """# 2022-09-06 23:16:22
        terminate the server
        """
        if self.flag_spawn and not self._flag_is_terminated:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(None)
            self._p.join()  # wait until the process join the main process
            self._flag_is_terminated = True  # set the flag

    def __enter__(self):
        """# 2022-12-08 02:00:08"""
        return self

    def __exit__(self):
        """# 2022-12-08 02:00:08
        terminate the spawned process when exiting the context
        """
        self.terminate()


""" memory-efficient methods for creating RAMtx/RamData object """


# latest 2022-07-28 11:31:12
# implementation using pipe (~3 times more efficient)
def create_stream_from_a_gzip_file_using_pipe(
    path_file_gzip, pipe_sender, func, int_buffer_size=100
):
    """# 2022-07-27 06:50:29
    parse and decorate mtx record for sorting. the resulting records only contains two values, index of axis that were not indexed and the data value, for more efficient pipe operation
    return a generator yielding ungziped records

    'path_file_gzip' : input file gzip file to create stream of decorated mtx record
    'pipe_sender' : pipe for retrieving decorated mtx records. when all records are parsed, None will be given.
    'func' : a function for transforming each 'line' in the input gzip file to a (decorated) record. if None is returned, the line will be ignored and will not be included in the output stream.
    'int_buffer_size' : the number of entries for each batch that will be given to 'pipe_sender'. increasing this number will reduce the overhead associated with interprocess-communication through pipe, but will require more memory usage

    returns:
    return the process that will be used for unzipping the input gzip file and creating a stream.
    """
    # handle arguments
    int_buffer_size = int(max(1, int_buffer_size))

    # define a function for doing the work
    def __gunzip(path_file_gzip, pipe_sender, int_buffer_size):
        """# 2022-07-25 22:22:33
        unzip gzip file and create a stream using the given pipe
        """
        with gzip.open(path_file_gzip, "rt") as file:
            l_buffer = []  # initialize the buffer
            for line in file:
                rec = func(line)  # convert gzipped line into a decorated record
                if rec is not None:  # if the transformed record is valid
                    l_buffer.append(rec)  # add a parsed record to the buffer

                if len(l_buffer) >= int_buffer_size:  # if the buffer is full
                    pipe_sender.send(
                        l_buffer
                    )  # send a list of record of a given buffer size
                    l_buffer = []  # initialize the next buffer
            if len(l_buffer) > 0:  # flush remaining buffer
                pipe_sender.send(l_buffer)
        pipe_sender.send(None)

    p = mp.Process(target=__gunzip, args=(path_file_gzip, pipe_sender, int_buffer_size))
    return p  # return the process


def concurrent_merge_sort_using_pipe(
    pipe_sender,
    *l_pipe_receiver,
    int_max_num_pipe_for_each_worker=8,
    int_buffer_size=100,
):
    """# 2022-07-27 06:50:22
    inputs:
    'pipe_sender' : a pipe through which sorted decorated mtx records will be send. when all records are parsed, None will be given.
    'l_pipe_receiver' : list of pipes through which decorated mtx records will be received. when all records are parsed, these pipes should return None.
    'int_max_num_pipe_for_each_worker' : maximum number of input pipes for each worker. this argument and the number of input pipes together will determine the number of threads used for sorting
    'int_buffer_size' : the number of entries for each batch that will be given to 'pipe_sender'. increasing this number will reduce the overhead associated with interprocess-communication through pipe, but will require more memory usage

    returns:
    l_p : list of processes for all the workers that will be used for sorting
    """
    import heapq

    # parse arguments
    int_max_num_pipe_for_each_worker = int(int_max_num_pipe_for_each_worker)
    # handle when no input pipes are given
    if len(l_pipe_receiver) == 0:
        pipe_sender.send(None)  # notify the end of records
        return -1

    int_num_merging_layers = int(
        np.ceil(math.log(len(l_pipe_receiver), int_max_num_pipe_for_each_worker))
    )  # retrieve the number of merging layers.

    def __pipe_receiver_to_iterator(pipe_receiver):
        """# 2022-07-25 00:59:22
        convert pipe_receiver to iterator
        """
        while True:
            l_r = pipe_receiver.recv()  # retrieve a batch of records
            # detect pipe_receiver
            if l_r is None:
                break
            for r in l_r:  # iterate through record by record, and yield each record
                yield r

    def __sorter(pipe_sender, *l_pipe_receiver):
        """# 2022-07-25 00:57:56"""
        # handle when no input pipes are given
        if len(l_pipe_receiver) == 0:
            pipe_sender.send(None)

        # perform merge sorting
        l_buffer = []  # initialize a buffer
        for r in heapq.merge(
            *list(map(__pipe_receiver_to_iterator, l_pipe_receiver))
        ):  # convert pipe to iterator
            l_buffer.append(r)  # add a parsed record to the buffer
            # flush the buffer
            if len(l_buffer) >= int_buffer_size:  # if the buffer is full
                pipe_sender.send(l_buffer)  # return record in a sorted order
                l_buffer = []  # initialize the buffer
        if len(l_buffer) > 0:  # if there are some records remaining in the buffer
            pipe_sender.send(l_buffer)  # send the buffer
        pipe_sender.send(None)  # notify the end of records

    l_p = (
        []
    )  # initialize the list that will contain all the processes that will be used for sorting.
    while (
        len(l_pipe_receiver) > int_max_num_pipe_for_each_worker
    ):  # perform merge operations for each layer until all input pipes can be merged using a single worker (perform merge operations for all layers except for the last layer)
        l_pipe_receiver_for_the_next_layer = (
            []
        )  # initialize the list of receiving pipes for the next layer, which will be collected while initializing workers for the current merging layer
        for index_worker_in_a_layer in range(
            int(np.ceil(len(l_pipe_receiver) / int_max_num_pipe_for_each_worker))
        ):  # iterate through the workers of the current merging layer
            pipe_sender_for_a_worker, pipe_receiver_for_a_worker = mp.Pipe()
            l_pipe_receiver_for_the_next_layer.append(
                pipe_receiver_for_a_worker
            )  # collect receiving end of pipes for the initiated workers
            l_p.append(
                mp.Process(
                    target=__sorter,
                    args=[pipe_sender_for_a_worker]
                    + list(
                        l_pipe_receiver[
                            index_worker_in_a_layer
                            * int_max_num_pipe_for_each_worker : (
                                index_worker_in_a_layer + 1
                            )
                            * int_max_num_pipe_for_each_worker
                        ]
                    ),
                )
            )
        l_pipe_receiver = (
            l_pipe_receiver_for_the_next_layer  # initialize the next layer
        )
    # retrieve a worker for the last merging layer
    l_p.append(mp.Process(target=__sorter, args=[pipe_sender] + list(l_pipe_receiver)))
    return l_p  # return the list of processes


def write_stream_as_a_gzip_file_using_pipe(
    pipe_receiver,
    path_file_gzip,
    func,
    compresslevel=6,
    int_num_threads=1,
    int_buffer_size=100,
    header=None,
):
    """# 2022-07-27 06:50:14
    parse and decorate mtx record for sorting. the resulting records only contains two values, index of axis that were not indexed and the data value, for more efficient pipe operation
    return a generator yielding ungziped records

    arguments:
    'pipe_receiver' : pipe for retrieving decorated mtx records. when all records are parsed, None should be given.
    'path_file_gzip' : output gzip file path
    'func' : a function for transforming each '(decorated) record to a line in the original gzip file' in the input gzip file to a. if None is returned, the line will be ignored and will not be included in the output stream.
    'flag_mtx_sorted_by_id_feature' : whether to create decoration with id_feature / id_barcode
    'flag_dtype_is_float' : set this flag to True to export float values to the output mtx matrix
    'compresslevel' : compression level of the output Gzip file. 6 by default
    'header' : a header text to include. if None is given, no header will be written.
    'int_num_threads' : the number of threads for gzip writer. (deprecated, currently only supporting single-thread due to memory-leakage issue of pgzip package)
    'int_buffer_size' : the number of entries for each batch that will be given to 'pipe_sender'. increasing this number will reduce the overhead associated with interprocess-communication through pipe, but will require more memory usage

    returns:
    return the process that will be used for gzipping the input stream
    """
    # handle input arguments
    int_num_threads = int(max(1, int_num_threads))

    # define a function for doing the work
    def __gzip(pipe_receiver, path_file_gzip, func):
        """# 2022-07-25 22:22:33
        unzip gzip file and create a stream using the given pipe
        """
        with gzip.open(
            path_file_gzip, "wt", compresslevel=compresslevel
        ) as newfile:  # open the output file
            if header is not None:  # if a valid header is given, write the header
                newfile.write(header)
            while True:
                l_r = pipe_receiver.recv()  # retrieve record
                if l_r is None:  # handle when all records are parsed
                    break
                l_buffer = []  # initialize the buffer
                for r in l_r:  # iterate through the list of records
                    l_buffer.append(func(r))  # collect the output
                    # write the output file
                    if (
                        len(l_buffer) >= int_buffer_size
                    ):  # when the buffer is full, flush the buffer
                        newfile.write("".join(l_buffer))  # write the output
                        l_buffer = []  # initialize the buffer
                if len(l_buffer) > 0:  # flush the buffer
                    newfile.write("".join(l_buffer))  # flush the buffer

    p = mp.Process(target=__gzip, args=(pipe_receiver, path_file_gzip, func))
    return p  # return the process


def write_stream_as_a_sparse_ramtx_zarr_using_pipe(pipe_receiver, za_mtx, za_mtx_index):
    """# 2022-07-29 09:54:57
    write a stream of decorated mtx records to a sparse ramtx zarr object, sorted by barcodes or features (and its associated index)

    arguments:
    'pipe_receiver' : pipe for retrieving decorated mtx records. when all records are parsed, None should be given.
    'za_mtx', 'za_mtx_index' : output zarr objects

    returns:
    return the list of processes that will be used for building a ramtx zarr from the input stream
    """
    # retrieve settings
    dtype_mtx = za_mtx.dtype
    dtype_mtx_index = za_mtx_index.dtype
    # set buffer size
    int_buffer_size_mtx_index = za_mtx_index.chunks[0] * 10
    int_buffer_size_mtx = za_mtx.chunks[0] * 10

    # define a function for doing the work
    def __write_zarr(pipe_receiver, za):
        """# 2022-07-29 09:41:08
        write an array of a specific coordinates to a given zarr object
        """
        while True:
            r = pipe_receiver.recv()
            if r is None:  # exist once the stream ends
                break
            st, en, arr = r  # parse the received record
            za[st:en] = arr  # write zarr object

    def __compose_array(
        pipe_receiver, pipe_sender_to_mtx_writer, pipe_sender_to_mtx_index_writer
    ):
        """# 2022-07-25 22:22:33
        convert a given stream into ramtx arrays (mtx and mtx index)
        """
        # perform merge sorting
        int_entry_currently_being_written = None  # place holder value
        int_num_mtx_records_written = 0
        l_mtx_record = []
        int_num_mtx_index_records_written = 0
        l_mtx_index = []

        # iterate through records
        while True:
            l_r = pipe_receiver.recv()  # retrieve record
            if l_r is None:  # handle when all records are parsed
                break
            for r in l_r:  # iterate through the list of records
                int_entry_of_the_current_record, mtx_record = r
                if int_entry_currently_being_written is None:
                    int_entry_currently_being_written = (
                        int_entry_of_the_current_record  # update current int_entry
                    )
                elif (
                    int_entry_currently_being_written != int_entry_of_the_current_record
                ):  # if current int_entry is different from the previous one, which mark the change of sorted blocks (a block has the same id_entry), save the data for the previous block to the output zarr and initialze data for the next block
                    """compose index"""
                    l_mtx_index.append(
                        [
                            int_num_mtx_records_written,
                            int_num_mtx_records_written + len(l_mtx_record),
                        ]
                    )  # collect information required for indexing # add records to mtx_index
                    if (
                        int_entry_currently_being_written + 1
                        > int_entry_of_the_current_record
                    ):
                        for int_entry in range(
                            int_entry_currently_being_written + 1,
                            int_entry_of_the_current_record,
                        ):  # for the int_entry that lack count data and does not have indexing data, put place holder values
                            l_mtx_index.append(
                                [0, 0]
                            )  # put place holder values for int_entry lacking count data.
                    int_entry_currently_being_written = (
                        int_entry_of_the_current_record  # update current int_entry
                    )

                    """ check mtx buffer and flush """
                    if len(l_mtx_record) >= int_buffer_size_mtx:  # if buffer is full
                        pipe_sender_to_mtx_writer.send(
                            (
                                int_num_mtx_records_written,
                                int_num_mtx_records_written + len(l_mtx_record),
                                np.array(l_mtx_record, dtype=dtype_mtx),
                            )
                        )  # send data to the zarr mtx writer
                        int_num_mtx_records_written += len(l_mtx_record)
                        l_mtx_record = []  # reset buffer

                    """ check mtx index buffer and flush """
                    if (
                        len(l_mtx_index) >= int_buffer_size_mtx_index
                    ):  # if buffer is full
                        pipe_sender_to_mtx_index_writer.send(
                            (
                                int_num_mtx_index_records_written,
                                int_num_mtx_index_records_written + len(l_mtx_index),
                                np.array(l_mtx_index, dtype=dtype_mtx_index),
                            )
                        )  # send data to the zarr mtx index writer
                        int_num_mtx_index_records_written += len(
                            l_mtx_index
                        )  # update 'int_num_mtx_index_records_written'
                        l_mtx_index = []  # reset buffer
                # collect mtx record
                l_mtx_record.append(mtx_record)

        # write the record for the last entry
        assert len(l_mtx_record) > 0  # there SHOULD be a last entry
        """ compose index """
        l_mtx_index.append(
            [
                int_num_mtx_records_written,
                int_num_mtx_records_written + len(l_mtx_record),
            ]
        )  # collect information required for indexing # add records to mtx_index
        for int_entry in range(
            int_entry_currently_being_written + 1, za_mtx_index.shape[0]
        ):  # for the int_entry that lack count data and does not have indexing data, put place holder values # set 'next' int_entry to the end of the int_entry values so that place holder values can be set to the missing int_entry
            l_mtx_index.append(
                [0, 0]
            )  # put place holder values for int_entry lacking count data.

        """ flush mtx buffer """
        pipe_sender_to_mtx_writer.send(
            (
                int_num_mtx_records_written,
                int_num_mtx_records_written + len(l_mtx_record),
                np.array(l_mtx_record, dtype=dtype_mtx),
            )
        )  # send data to the zarr mtx writer

        """ flush mtx index buffer """
        pipe_sender_to_mtx_index_writer.send(
            (
                int_num_mtx_index_records_written,
                int_num_mtx_index_records_written + len(l_mtx_index),
                np.array(l_mtx_index, dtype=dtype_mtx_index),
            )
        )  # send data to the zarr mtx index writer

        """ send termination signals """
        pipe_sender_to_mtx_writer.send(None)
        pipe_sender_to_mtx_index_writer.send(None)

    # create pipes for communications between the processes
    pipe_sender_to_mtx_writer, pipe_receiver_to_mtx_writer = mp.Pipe()
    pipe_sender_to_mtx_index_writer, pipe_receiver_to_mtx_index_writer = mp.Pipe()
    # compose the list of processes
    l_p = []
    l_p.append(
        mp.Process(
            target=__compose_array,
            args=(
                pipe_receiver,
                pipe_sender_to_mtx_writer,
                pipe_sender_to_mtx_index_writer,
            ),
        )
    )
    l_p.append(
        mp.Process(target=__write_zarr, args=(pipe_receiver_to_mtx_writer, za_mtx))
    )
    l_p.append(
        mp.Process(
            target=__write_zarr, args=(pipe_receiver_to_mtx_index_writer, za_mtx_index)
        )
    )
    return l_p  # return the list of processes


def concurrent_merge_sort_using_pipe_mtx(
    path_file_output=None,
    l_path_file=[],
    flag_mtx_sorted_by_id_feature=True,
    int_buffer_size=100,
    compresslevel=6,
    int_max_num_pipe_for_each_worker=8,
    flag_dtype_is_float=False,
    flag_return_processes=False,
    int_num_threads=1,
    flag_delete_input_files=False,
    header=None,
    za_mtx=None,
    za_mtx_index=None,
):
    """# 2022-07-27 06:50:06

    'path_file_output' : output mtx gzip file path
    'l_path_file' : list of input mtx gzip file paths
    'flag_mtx_sorted_by_id_feature' : whether to create decoration with id_feature / id_barcode
    'int_buffer_size' : the number of entries for each batch that will be given to 'pipe_sender'. increasing this number will reduce the overhead associated with interprocess-communication through pipe, but will require more memory usage
    'compresslevel' : compression level of the output Gzip file. 6 by default
    'int_max_num_pipe_for_each_worker' : maximum number of input pipes for each worker. this argument and the number of input pipes together will determine the number of threads used for sorting
    'flag_dtype_is_float' : set this flag to True to export float values to the output mtx matrix
    'flag_return_processes' : if False, run all the processes. if True, return the processes that can be run to perform the concurrent merge sort operation.
    'int_num_threads' : the number of threads for gzip writer. if 'int_num_threads' > 1, pgzip will be used to write the output gzip file.
    'header' : a header text to include. if None is given, no header will be written.
    'flag_delete_input_files' : delete input files
    'za_mtx', 'za_mtx_index' : to build ramtx zarr object from the input mtx files, please use these arguments to pass over zarr mtx and zarr mtx index objects.
    """
    # handle invalid input
    if len(l_path_file) == 0:  # if the list of input files are empty, exit
        return
    if not (
        path_file_output is not None
        or (za_mtx is not None and za_mtx_index is not None)
    ):  # if all output are invalid, exit
        return

    def __decode_mtx(line):
        """# 2022-07-27 00:28:42
        decode a line and return a parsed line (record)
        """
        """ skip comment lines """
        if line[0] == "%":
            return None
        """ parse a mtx record """
        (
            index_row,
            index_column,
            float_value,
        ) = line.strip().split()  # parse a record of a matrix-market format file
        index_row, index_column, float_value = (
            int(index_row) - 1,
            int(index_column) - 1,
            float(float_value),
        )  # 1-based > 0-based coordinates
        # return record with decoration according to the sorted axis # return 0-based coordinates
        if flag_mtx_sorted_by_id_feature:
            res = index_row, (index_column, float_value)
        else:
            res = index_column, (index_row, float_value)
        return res

    convert_to_output_dtype = float if flag_dtype_is_float else int

    def __encode_mtx(r):
        """# 2022-07-27 00:31:27
        encode parsed record into a line (in an original format)
        """
        dec, rec = r  # retrieve decorator and the remaining record
        if flag_mtx_sorted_by_id_feature:
            index_row = dec
            index_column, val = rec
        else:
            index_column = dec
            index_row, val = rec
        val = convert_to_output_dtype(val)  # convert to the output dtype
        return (
            str(index_row + 1) + " " + str(index_column + 1) + " " + str(val) + "\n"
        )  # return the output

    # construct and collect the processes for the parsers
    l_p = []
    l_pipe_receiver = []
    for index_file, path_file in enumerate(l_path_file):
        pipe_sender, pipe_receiver = mp.Pipe()
        p = create_stream_from_a_gzip_file_using_pipe(
            path_file, pipe_sender, func=__decode_mtx, int_buffer_size=int_buffer_size
        )
        l_p.append(p)
        l_pipe_receiver.append(pipe_receiver)

    # construct and collect the processes for a concurrent merge sort tree and writer
    pipe_sender, pipe_receiver = mp.Pipe()  # create a link
    l_p.extend(
        concurrent_merge_sort_using_pipe(
            pipe_sender,
            *l_pipe_receiver,
            int_max_num_pipe_for_each_worker=int_max_num_pipe_for_each_worker,
            int_buffer_size=int_buffer_size,
        )
    )
    if path_file_output is not None:  # when an output file is an another gzip file
        l_p.append(
            write_stream_as_a_gzip_file_using_pipe(
                pipe_receiver,
                path_file_output,
                func=__encode_mtx,
                compresslevel=compresslevel,
                int_num_threads=int_num_threads,
                int_buffer_size=int_buffer_size,
                header=header,
            )
        )
    else:  # when the output is a ramtx zarr object
        l_p.extend(
            write_stream_as_a_sparse_ramtx_zarr_using_pipe(
                pipe_receiver, za_mtx, za_mtx_index
            )
        )

    if flag_return_processes:
        # simply return the processes
        return l_p
    else:
        # run the processes
        for p in l_p:
            p.start()
        for p in l_p:
            p.join()

    # delete input files if 'flag_delete_input_files' is True
    if flag_delete_input_files:
        for path_file in l_path_file:
            filesystem_operations("rm", path_file)

    if (
        path_file_output is not None
    ):  # when an output file is an another gzip file # return the path to the output file
        return path_file_output


# for sorting mtx and creating RamData
def create_and_sort_chunk(
    path_file_gzip,
    path_prefix_chunk,
    func_encoding,
    func_decoding,
    pipe_sender,
    func_detect_header=None,
    int_num_records_in_a_chunk=10000000,
    int_num_threads_for_sorting_and_writing=5,
    int_buffer_size=300,
):
    """# 2022-11-28 23:43:38
    split an input gzip file into smaller chunks and sort individual chunks.
    returns a list of processes that will perform the operation.

    'path_file_gzip' : file path of an input gzip file
    'path_prefix_chunk' : a prefix for the chunks that will be written.
    'func_encoding' : a function for transforming a decorated record into a line in a gzip file.
    'func_decoding' : a function for transforming a line in a gzip file into a decorated record. the lines will be sorted according to the first element of the returned records. the first element (key) should be float/integers (numbers)
    'pipe_sender' : a pipe that will be used to return the list of file path of the created chunks. when all chunks are created, None will be given.
    'func_detect_header' : a function for detecting header lines in a gzip file. the opened gzip file will be given ('rw' mode) to the function and the funciton should consume all header lines. optionally, the function can return a line that are the start of the record if the algorithm required to read non-header line to detect the end of header. (i.e. if header line was not present, the consumed line should be returned)
    'int_num_records_in_a_chunk' : the number of maximum records in a chunk
    'int_num_threads_for_sorting_and_writing' : number of workers for sorting and writing operations. the number of worker for reading the input gzip file will be 1.
    'int_buffer_size' : the number of entries for each batch that will be given to 'pipe_sender'. increasing this number will reduce the overhead associated with interprocess-communication through pipe, but will require more memory usage

    """
    # handle arguments
    int_buffer_size = int(max(1, int_buffer_size))

    def __sort_and_write(pipe_receiver_record, pipe_sender_file_path):
        """# 2022-07-27 09:07:26
        receive records for a chunk, sort the records, and write encoded records as a gzip file
        """
        l_r = []  # initialize a list for collecting records
        while True:
            b = pipe_receiver_record.recv()
            if b is None:
                break
            if isinstance(b, str):
                newfile = gzip.open(b, "wt")  # open a new file using the name

                """ sort records """
                int_num_records = len(l_r)
                arr_key = np.zeros(
                    int_num_records, dtype=float
                )  # the first element (key) should be float/integers (numbers)
                for i, r in enumerate(l_r):
                    arr_key[i] = r[0]
                l_r = np.array(l_r, dtype=object)[
                    arr_key.argsort()
                ]  # sort records by keys
                del arr_key

                for r in l_r:
                    newfile.write(func_encoding(r))

                l_r = []  # initialize a list for collecting records
                newfile.close()  # close an output file
                pipe_sender_file_path.send(b)  # send the file path of written chunk
            else:
                # collect record from the buffer
                for r in b:
                    l_r.append(r)

    # initialize
    l_p = []  # initialize the list of processes
    l_pipe_sender_record = []  # collect the pipe for distributing records
    l_pipe_receiver_file_path = []  # collect the pipe for receiving file path
    # compose processes for sorting and writing chunks
    for index_process in range(int_num_threads_for_sorting_and_writing):
        (
            pipe_sender_record,
            pipe_receiver_record,
        ) = mp.Pipe()  # create a pipe for sending parsed records
        (
            pipe_sender_file_path,
            pipe_receiver_file_path,
        ) = mp.Pipe()  # create a pipe for sending file path of written chunks
        l_pipe_sender_record.append(pipe_sender_record)
        l_pipe_receiver_file_path.append(pipe_receiver_file_path)
        l_p.append(
            mp.Process(
                target=__sort_and_write,
                args=(pipe_receiver_record, pipe_sender_file_path),
            )
        )

    # define a function for reading gzip file
    def __gunzip(
        path_file_gzip,
        path_prefix_chunk,
        pipe_sender,
        int_num_records_in_a_chunk,
        int_buffer_size,
    ):
        """# 2022-07-25 22:22:33
        unzip gzip file, distribute records across workers, and collect the file path of written chunks
        """
        int_num_workers = len(l_pipe_sender_record)  # retrieve the number of workers
        arr_num_files = np.zeros(
            int_num_workers, dtype=int
        )  # initialize an array indicating how many files each worker processes need to write

        def __collect_file_path():
            """# 2022-07-27 09:48:04
            collect file paths of written chunks from workers and report the file path using 'pipe_sender'
            """
            while True:
                for index_worker, pipe_receiver_file_path in enumerate(
                    l_pipe_receiver_file_path
                ):
                    if pipe_receiver_file_path.poll():
                        path_file = pipe_receiver_file_path.recv()
                        pipe_sender.send(path_file)
                        arr_num_files[
                            index_worker
                        ] -= 1  # update the number of files for the process
                        assert arr_num_files[index_worker] >= 0

                # if all workers has less than two files to write, does not block
                if (arr_num_files <= 2).sum() == int_num_workers:
                    break
                time.sleep(1)  # sleep for one second before collecting completed works

        # iterate through lines in the input gzip file and assign works to the workers
        with gzip.open(path_file_gzip, "rt") as file:
            l_buffer = []  # initialize the buffer
            int_num_sent_records = 0  # initialize the number of send records
            index_worker = 0  # initialize the worker for receiving records

            # detect header from the start of the file
            if func_detect_header is not None and hasattr(
                func_detect_header, "__call__"
            ):  # if a valid function for detecting header has been given
                line = func_detect_header(
                    file
                )  # consume header lines. if header line was not present, the consumed line will be returned
                if (
                    line is None
                ):  # if exactly header lines are consumed and no actual records were consumed from the file, read the first record
                    line = file.readline()
            # iterate lines of the rest of the gzip file
            while True:
                r = func_decoding(line)  # convert gzipped line into a decorated record
                if r is not None:  # if the transformed record is valid
                    l_buffer.append(r)  # add a parsed record to the buffer

                if len(l_buffer) >= int_buffer_size:  # if the buffer is full
                    l_pipe_sender_record[index_worker].send(
                        l_buffer
                    )  # send a list of record of a given buffer size
                    int_num_sent_records += len(
                        l_buffer
                    )  # update 'int_num_sent_records'
                    l_buffer = []  # initialize the next buffer
                elif int_num_sent_records + len(l_buffer) >= int_num_records_in_a_chunk:
                    l_pipe_sender_record[index_worker].send(
                        l_buffer
                    )  # send a list of records
                    int_num_sent_records += len(
                        l_buffer
                    )  # update 'int_num_sent_records'
                    l_buffer = []  # initialize the next buffer
                    int_num_sent_records = 0  # initialize the next chunk
                    l_pipe_sender_record[index_worker].send(
                        f"{path_prefix_chunk}.{bk.UUID( )}.gz"
                    )  # assign the file path of the chunk
                    arr_num_files[
                        index_worker
                    ] += 1  # update the number of files for the process
                    index_worker = (
                        1 + index_worker
                    ) % int_num_workers  # update the index of the worker
                    __collect_file_path()  # collect and report file path
                line = file.readline()  # read the next line
                if len(line) == 0:
                    break

        if len(l_buffer) > 0:  # if there is some buffer remaining, flush the buffer
            l_pipe_sender_record[index_worker].send(l_buffer)  # send a list of records
            l_pipe_sender_record[index_worker].send(
                f"{path_prefix_chunk}.{bk.UUID( )}.gz"
            )  # assign the file path of the chunk
            arr_num_files[
                index_worker
            ] += 1  # update the number of files for the process
            __collect_file_path()  # collect and report file path

        # wait until all worker processes complete writing files
        while arr_num_files.sum() > 0:
            time.sleep(1)
            __collect_file_path()  # collect and report file path

        # terminate the worker processes
        for pipe_sender_record in l_pipe_sender_record:
            pipe_sender_record.send(None)

        pipe_sender.send(None)  # notify that the process has been completed

    l_p.append(
        mp.Process(
            target=__gunzip,
            args=(
                path_file_gzip,
                path_prefix_chunk,
                pipe_sender,
                int_num_records_in_a_chunk,
                int_buffer_size,
            ),
        )
    )
    return l_p  # return the list of processes


def sort_mtx(
    path_file_gzip,
    path_file_gzip_sorted=None,
    int_num_records_in_a_chunk=10000000,
    int_buffer_size=300,
    compresslevel=6,
    flag_dtype_is_float=False,
    flag_mtx_sorted_by_id_feature=True,
    int_num_threads_for_chunking=5,
    int_num_threads_for_writing=1,
    int_max_num_input_files_for_each_merge_sort_worker=8,
    int_num_chunks_to_combine_before_concurrent_merge_sorting=8,
    za_mtx=None,
    za_mtx_index=None,
):
    """# 2022-07-28 11:07:44
    sort a given mtx file in a very time- and memory-efficient manner

    'path_file_gzip' : file path of an input gzip file
    'int_num_records_in_a_chunk' : the number of maximum records in a chunk
    'int_num_threads_for_chunking' : number of workers for sorting and writing operations. the number of worker for reading the input gzip file will be 1.
    'int_buffer_size' : the number of entries for each batch that will be given to 'pipe_sender'. increasing this number will reduce the overhead associated with interprocess-communication through pipe, but will require more memory usage
    'flag_mtx_sorted_by_id_feature' : whether to create decoration with id_feature / id_barcode
    'compresslevel' : compression level of the output Gzip file. 6 by default
    'int_max_num_input_files_for_each_merge_sort_worker' : maximum number of input pipes for each worker. this argument and the number of input pipes together will determine the number of threads used for sorting
    'flag_dtype_is_float' : set this flag to True to export float values to the output mtx matrix
    'int_num_threads_for_writing' : the number of threads for gzip writer. if 'int_num_threads' > 1, pgzip will be used to write the output gzip file. please note that pgzip (multithreaded version of gzip module) has some memory-leaking issue for large inputs.
    'flag_delete_input_files' : delete input files
    'za_mtx', 'za_mtx_index' : to build ramtx zarr object from the input mtx files, please use these arguments to pass over zarr mtx and zarr mtx index objects.

    """
    # check validity of inputs
    if path_file_gzip_sorted is None:
        assert (
            za_mtx is not None and za_mtx_index is not None
        )  # if ramtx zarr object will be written, both arguments should be valid
    # create a temporary folder
    path_folder = path_file_gzip.rsplit("/", 1)[0] + "/"
    path_folder_temp = f"{path_folder}temp_{bk.UUID( )}/"
    filesystem_operations("mkdir", path_folder_temp, exist_ok=True)

    # create and sort chunks
    def __detect_header_mtx(file):
        """# 2022-07-28 10:21:15
        detect header lines from mtx file
        """
        line = file.readline()
        if len(line) > 0 and line[0] == "%":  # if comment lines are detected
            # read comment and the description line of a mtx file
            while True:
                if (
                    line[0] != "%"
                ):  # once a description line was consumed, exit the function
                    break
                line = file.readline()  # read the next line
            return None
        else:  # if no header was detected, return a consumed line so that it can be processed by downstream application
            return line

    def __decode_mtx(line):
        """# 2022-07-27 00:28:42
        decode a line and return a parsed line (record)
        """
        """ parse a mtx record """
        try:
            (
                index_row,
                index_column,
                float_value,
            ) = line.strip().split()  # parse a record of a matrix-market format file
        except:
            return None
        index_row, index_column, float_value = (
            int(index_row) - 1,
            int(index_column) - 1,
            float(float_value),
        )  # 1-based > 0-based coordinates
        # return record with decoration according to the sorted axis # return 0-based coordinates
        if flag_mtx_sorted_by_id_feature:
            res = index_row, (index_column, float_value)
        else:
            res = index_column, (index_row, float_value)
        return res

    convert_to_output_dtype = float if flag_dtype_is_float else int

    def __encode_mtx(r):
        """# 2022-07-27 00:31:27
        encode parsed record into a line (in an original format)
        """
        dec, rec = r  # retrieve decorator and the remaining record
        if flag_mtx_sorted_by_id_feature:
            index_row = dec
            index_column, val = rec
        else:
            index_column = dec
            index_row, val = rec
        val = convert_to_output_dtype(val)  # convert to the output dtype
        return (
            str(index_row + 1) + " " + str(index_column + 1) + " " + str(val) + "\n"
        )  # return the output

    pipe_sender, pipe_receiver = mp.Pipe()  # create a link
    l_p = create_and_sort_chunk(
        path_file_gzip,
        f"{path_folder_temp}chunk",
        __encode_mtx,
        __decode_mtx,
        pipe_sender,
        func_detect_header=__detect_header_mtx,
        int_num_records_in_a_chunk=int_num_records_in_a_chunk,
        int_num_threads_for_sorting_and_writing=int_num_threads_for_chunking,
        int_buffer_size=int_buffer_size,
    )  # retrieve processes
    for p in l_p:
        p.start()  # start chunking

    l_path_file_for_concurrent_merge_sorting = []
    l_path_file_chunk_for_merging = []
    dict_process = dict()

    def __combine_chunks(path_file_output, l_path_file, pipe_sender):
        """# 2022-07-27 22:05:31
        merge sort a given list of chunks and return a signal once operation has been completed
        """
        pipe_sender.send(
            concurrent_merge_sort_using_pipe_mtx(
                path_file_output=path_file_output,
                l_path_file=l_path_file,
                flag_mtx_sorted_by_id_feature=flag_mtx_sorted_by_id_feature,
                int_buffer_size=int_buffer_size,
                compresslevel=compresslevel,
                int_max_num_pipe_for_each_worker=int_max_num_input_files_for_each_merge_sort_worker,
                flag_dtype_is_float=flag_dtype_is_float,
                flag_return_processes=False,
                int_num_threads=int_num_threads_for_writing,
                flag_delete_input_files=True,
            )
        )

    def __run_combine_chunks(l_path_file):
        pipe_sender, pipe_receiver = mp.Pipe()  # create a link
        path_file_output = f"{path_folder_temp}combined_chunk.{bk.UUID( )}.gz"
        p = mp.Process(
            target=__combine_chunks, args=(path_file_output, l_path_file, pipe_sender)
        )
        dict_process[bk.UUID()] = {
            "p": p,
            "pipe_receiver": pipe_receiver,
            "path_file_output": path_file_output,
        }  # collect the process
        p.start()  # start the process

    def __collect_files_for_concurrent_merge_sorting():
        for id_process in list(dict_process):
            if dict_process[id_process][
                "pipe_receiver"
            ].poll():  # if the process has been completed
                path_file_output = dict_process[id_process][
                    "pipe_receiver"
                ].recv()  # receive output
                assert (
                    path_file_output is not None
                )  # check whether the merge sort was successful
                l_path_file_for_concurrent_merge_sorting.append(
                    path_file_output
                )  # collect the file path of the larger chunk
                dict_process[id_process]["p"].join()
                del dict_process[
                    id_process
                ]  # remove the process from the dictionary of running processes

    """ merge created chunks into larger chunks while chunking is completed """
    while True:
        if not pipe_receiver.poll():
            time.sleep(1)  # sleep for 1 second
        else:  # if an input is available
            path_file_chunk = pipe_receiver.recv()
            # if all chunks are created, exit
            if path_file_chunk is None:
                break
            # collect file path of chunks, and combine these smaller chunks into larger chunks for concurrent merge sort operation
            if (
                int_num_chunks_to_combine_before_concurrent_merge_sorting == 1
            ):  # when 'int_num_chunks_to_combine_before_concurrent_merge_sorting' == 1, the small chunks will be directly used for concurrent merge sort.
                l_path_file_for_concurrent_merge_sorting.append(path_file_chunk)
            else:
                l_path_file_chunk_for_merging.append(
                    path_file_chunk
                )  # collect file path of chunks
                if (
                    len(l_path_file_chunk_for_merging)
                    >= int_num_chunks_to_combine_before_concurrent_merge_sorting
                ):  # if the number of collected chunks reaches the number that could be combined into a larger chunk
                    __run_combine_chunks(
                        l_path_file_chunk_for_merging
                    )  # combine chunks into a larger chunk
                    l_path_file_chunk_for_merging = (
                        []
                    )  # initialize the list for the next run
                __collect_files_for_concurrent_merge_sorting()  # collect file path of chunks for concurrent merge sorting

    if len(l_path_file_chunk_for_merging) > 0:
        l_path_file_for_concurrent_merge_sorting.extend(l_path_file_chunk_for_merging)

    while (
        len(dict_process) > 0
    ):  # wait until all preliminary merge sort operation on created chunks are completed
        __collect_files_for_concurrent_merge_sorting()  # collect files for concurrent merge sorting
        time.sleep(1)  # sleep for 1 second

    if path_file_gzip_sorted is not None:  # if an ouptut file is another mtx.gz file
        # retrieve metadata from the input mtx file
        (
            int_num_rows,
            int_num_columns,
            int_num_records,
        ) = MTX_10X_Retrieve_number_of_rows_columns_and_records(path_file_gzip)
        header = f"""%%MatrixMarket matrix coordinate integer general\n%\n{int_num_rows} {int_num_columns} {int_num_records}\n"""  # compose a header

        # perform merge sorting preliminarily merge-sorted chunks into a single sorted output file
        filesystem_operations(
            "mkdir", path_file_gzip_sorted.rsplit("/", 1)[0], exist_ok=True
        )  # create an output folder
        concurrent_merge_sort_using_pipe_mtx(
            path_file_gzip_sorted,
            l_path_file_for_concurrent_merge_sorting,
            flag_mtx_sorted_by_id_feature=flag_mtx_sorted_by_id_feature,
            int_buffer_size=int_buffer_size,
            compresslevel=compresslevel,
            int_max_num_pipe_for_each_worker=int_max_num_input_files_for_each_merge_sort_worker,
            flag_dtype_is_float=flag_dtype_is_float,
            flag_return_processes=False,
            int_num_threads=int_num_threads_for_writing,
            flag_delete_input_files=True,
            header=header,
        )  # write matrix market file header
    else:  # if an output is a ramtx zarr object
        concurrent_merge_sort_using_pipe_mtx(
            l_path_file=l_path_file_for_concurrent_merge_sorting,
            flag_mtx_sorted_by_id_feature=flag_mtx_sorted_by_id_feature,
            int_buffer_size=int_buffer_size,
            compresslevel=compresslevel,
            int_max_num_pipe_for_each_worker=int_max_num_input_files_for_each_merge_sort_worker,
            flag_dtype_is_float=flag_dtype_is_float,
            flag_return_processes=False,
            int_num_threads=int_num_threads_for_writing,
            flag_delete_input_files=True,
            za_mtx=za_mtx,
            za_mtx_index=za_mtx_index,
        )  # write ramtx zarr object

    # delete temp folder
    filesystem_operations("rm", path_folder_temp)


def create_zarr_from_mtx(
    path_file_input_mtx,
    path_folder_zarr,
    int_buffer_size=1000,
    int_num_workers_for_writing_ramtx=10,
    chunks_dense=(1000, 1000),
    dtype_mtx=np.float64,
    flag_combine_duplicate_records: bool = False,
):
    """# 2023-01-27 12:30:25
    create dense ramtx (dense zarr object) from matrix sorted by barcodes.

    'path_file_input_mtx' : input mtx gzip file
    'path_folder_zarr' : output zarr object folder
    'int_buffer_size' : number of lines for a pipe communcation. larger value will decrease an overhead for interprocess coummuncaiton. however, it will lead to more memory usage.
    'int_num_workers_for_writing_ramtx' : the number of worker for writing zarr object
    'chunks_dense' : chunk size of the output zarr object. smaller number of rows in a chunk will lead to smaller memory consumption, since data of all genes for the cells in a chunk will be collected before writing. ( int_num_barcodes_in_a_chunk, int_num_features_in_a_chunk )
    'dtype_mtx' : zarr object dtype
    flag_combine_duplicate_records : bool = False # by default, it has been set to False to increase the performance. if True, for duplicate records in the given matrix market file, values will be summed. (for example, if ( 1, 2, 10 ) and ( 1, 2, 5 ) records will be combined into ( 1, 2, 15 )).
    """
    int_num_barcodes_in_a_chunk = chunks_dense[0]

    # retrieve dimension of the output dense zarr array
    (
        int_num_features,
        int_num_barcodes,
        int_num_records,
    ) = MTX_10X_Retrieve_number_of_rows_columns_and_records(
        path_file_input_mtx
    )  # retrieve metadata of mtx
    # open persistent zarr arrays to store matrix (dense ramtx)
    za_mtx = zarr.open(
        path_folder_zarr,
        mode="w",
        shape=(int_num_barcodes, int_num_features),
        chunks=chunks_dense,
        dtype=dtype_mtx,
        synchronizer=zarr.ThreadSynchronizer(),
    )  # each mtx record will contains two values instead of three values for more compact storage

    """ assumes input mtx is sorted by id_barcode (sorted by columns of the matrix market formatted matrix) """

    def __gunzip(path_file_input_mtx, pipe_sender):
        """# 2022-11-28 23:39:34
        create a stream of lines from a gzipped mtx file
        """
        with gzip.open(path_file_input_mtx, "rt") as file:
            line = file.readline()
            if len(line) == 0:  # if the file is empty, exit
                pipe_sender.send(None)  # indicates that the file reading is completed
            else:
                # consume comment lines
                while line[0] == "%":
                    line = file.readline()  # read the next line
                line = (
                    file.readline()
                )  # discard the description line (the number of barcodes/features/records) and read the next line

                # use the buffer to reduce the overhead of interprocess communications
                l_buffer = []  # initialize the buffer
                while line:  # iteratre through lines
                    l_buffer.append(line)
                    if (
                        len(l_buffer) >= int_buffer_size
                    ):  # if buffer is full, flush the buffer
                        pipe_sender.send(l_buffer)
                        l_buffer = []
                    line = file.readline()  # read the next line
                # flush remaining buffer
                if len(l_buffer) >= 0:
                    pipe_sender.send(l_buffer)
                pipe_sender.send(None)  # indicates that the file reading is completed

    def __distribute(
        pipe_receiver, int_num_workers_for_writing_ramtx, int_num_barcodes_in_a_chunk
    ):
        """# 2022-07-29 23:28:37"""

        def __write_zarr(pipe_receiver):
            """# 2022-07-29 23:29:00"""
            while True:
                r = pipe_receiver.recv()
                if r is None:  # when all works are completed, exit
                    break
                coords_barcodes, coords_features, values = r  # parse received records

                if (
                    flag_combine_duplicate_records
                ):  # handle duplicate records (combine values of duplicate records)
                    # combine values for each unique set of coordinates
                    dict_t_coords_to_combined_value = dict()
                    for coord_bc, coord_ft, value in zip(
                        coords_barcodes, coords_features, values
                    ):
                        t_coords = (coord_bc, coord_ft)
                        if t_coords in dict_t_coords_to_combined_value:
                            dict_t_coords_to_combined_value[t_coords] += value
                        else:
                            dict_t_coords_to_combined_value[t_coords] = value

                    # construct new, de-duplicated 'coords_barcodes', 'coords_features', and 'values'
                    coords_barcodes, coords_features, values = [], [], []
                    for t_coords in dict_t_coords_to_combined_value:
                        coords_barcodes.append(t_coords[0])
                        coords_features.append(t_coords[1])
                        values.append(dict_t_coords_to_combined_value[t_coords])

                za_mtx.set_coordinate_selection(
                    (coords_barcodes, coords_features), values
                )  # write zarr matrix

        # start workers for writing zarr
        l_p = []
        l_pipe_sender = []
        index_process = 0  # initialize index of process
        for index_worker in range(int_num_workers_for_writing_ramtx):
            (
                pipe_sender_for_a_worker,
                pipe_receiver_for_a_worker,
            ) = mp.Pipe()  # create a pipe to the worker
            p = mp.Process(target=__write_zarr, args=(pipe_receiver_for_a_worker,))
            p.start()  # start process
            l_p.append(p)  # collect process
            l_pipe_sender.append(pipe_sender_for_a_worker)  # collect pipe_sender

        # distribute works to workers
        int_index_chunk_being_collected = None
        l_int_feature, l_int_barcode, l_float_value = [], [], []
        while True:
            l_line = pipe_receiver.recv()
            if l_line is None:  # if all lines are retrieved, exit
                break
            for line in l_line:
                (
                    int_feature,
                    int_barcode,
                    float_value,
                ) = (
                    line.strip().split()
                )  # parse a record of a matrix-market format file
                int_feature, int_barcode, float_value = (
                    int(int_feature) - 1,
                    int(int_barcode) - 1,
                    float(float_value),
                )  # 1-based > 0-based coordinates

                int_index_chunk = (
                    int_barcode // int_num_barcodes_in_a_chunk
                )  # retrieve int_chunk of the current barcode

                # initialize 'int_index_chunk_being_collected'
                if int_index_chunk_being_collected is None:
                    int_index_chunk_being_collected = int_index_chunk

                # flush the chunk
                if int_index_chunk_being_collected != int_index_chunk:
                    l_pipe_sender[index_process].send(
                        (l_int_barcode, l_int_feature, l_float_value)
                    )
                    # change the worker
                    index_process = (
                        index_process + 1
                    ) % int_num_workers_for_writing_ramtx
                    # initialize the next chunk
                    l_int_feature, l_int_barcode, l_float_value = [], [], []
                    int_index_chunk_being_collected = int_index_chunk

                # collect record
                l_int_feature.append(int_feature)
                l_int_barcode.append(int_barcode)
                l_float_value.append(float_value)

        # write the last chunk if valid unwritten chunk exists
        if len(l_int_barcode) > 0:
            l_pipe_sender[index_process].send(
                (l_int_barcode, l_int_feature, l_float_value)
            )

        # terminate the workers
        for pipe_sender in l_pipe_sender:
            pipe_sender.send(None)

    # compose processes
    l_p = []
    pipe_sender, pipe_receiver = mp.Pipe()  # create a link
    l_p.append(mp.Process(target=__gunzip, args=(path_file_input_mtx, pipe_sender)))
    l_p.append(
        mp.Process(
            target=__distribute,
            args=(
                pipe_receiver,
                int_num_workers_for_writing_ramtx,
                int_num_barcodes_in_a_chunk,
            ),
        )
    )

    # run processes
    for p in l_p:
        p.start()
    for p in l_p:
        p.join()


def sort_mtx_10x(
    path_folder_mtx_input: str,
    path_folder_mtx_output: str,
    flag_mtx_sorted_by_id_feature: bool = False,
    **kwargs,
):
    """# 2022-08-21 17:05:55
    a convenient wrapper method of 'sort_mtx' for sorting a matrix in a 10X matrix format. features and barcodes files will be copied, too.

    'path_folder_mtx_input' : path to the input matrix folder
    'path_folder_mtx_output' : path to the output matrix folder
    'flag_mtx_sorted_by_id_feature' : whether to sort mtx file in the feature axis or not

    kwargs: keyworded arguments for 'sort_mtx'
    """
    filesystem_operations(
        "mkdir", path_folder_mtx_output, exist_ok=True
    )  # create output folder
    # copy features and barcodes files
    for name_file in ["features.tsv.gz", "barcodes.tsv.gz"]:
        filesystem_operations(
            "cp",
            f"{path_folder_mtx_input}{name_file}",
            f"{path_folder_mtx_output}{name_file}",
        )
    # sort matrix file
    sort_mtx(
        f"{path_folder_mtx_input}matrix.mtx.gz",
        path_file_gzip_sorted=f"{path_folder_mtx_output}matrix.mtx.gz",
        flag_mtx_sorted_by_id_feature=flag_mtx_sorted_by_id_feature,
        **kwargs,
    )


""" utility functions for handling zarr """


def zarr_exists(
    path_folder_zarr, filesystemserver: Union[None, FileSystemServer] = None
):
    """# 2022-07-20 01:06:09
    check whether the given zarr object path exists.

    filesystemserver
    """
    if filesystemserver is not None and isinstance(filesystemserver, FileSystemServer):
        # if a filesystemserver has been given, use the FileSystemServer object to check whether a zarr object exist in the given directory
        fs = filesystemserver
        # check whether a zarr group exists. if a zarr group exists, return the result (True)
        flag_zarr_group_exists = fs.filesystem_operations(
            "exists", f"{path_folder_zarr}.zgroup"
        )
        if flag_zarr_group_exists:
            return True
        # check whether a zarr array exists. if a zarr array exists, return the result (True)
        flag_zarr_array_exists = fs.filesystem_operations(
            "exists", f"{path_folder_zarr}.zarray"
        )
        return flag_zarr_array_exists
    else:
        # when file system server is not available, use the Zarr module as a fall back
        import zarr

        try:
            zarr.open(path_folder_zarr, "r")
            flag_zarr_exists = True
        except:
            flag_zarr_exists = False
    return flag_zarr_exists


def zarr_copy(
    path_folder_zarr_source, path_folder_zarr_sink, int_num_chunks_per_batch=1000
):
    """# 2022-07-22 01:45:17
    copy a soruce zarr object to a sink zarr object chunks by chunks along the primary axis (axis 0)
    also copy associated attributes, too.

    'path_folder_zarr_source' : source zarr object path
    'path_folder_zarr_sink' : sink zarr object path
    'int_num_chunks_per_batch' : number of chunks along the primary axis (axis 0) to be copied for each loop. for example, when the size of an array is (100, 100), chunk size is (10, 10), and 'int_num_chunks_per_batch' = 1, 10 chunks along the secondary axis (axis = 1) will be saved for each batch.
    """
    # open zarr objects
    za = zarr.open(path_folder_zarr_source)
    za_sink = zarr.open(
        path_folder_zarr_sink,
        mode="w",
        shape=za.shape,
        chunks=za.chunks,
        dtype=za.dtype,
        fill_value=za.fill_value,
        synchronizer=zarr.ThreadSynchronizer(),
    )  # open the output zarr

    # copy count data
    int_total_num_rows = za.shape[0]
    int_num_rows_in_batch = za.chunks[0] * int(int_num_chunks_per_batch)
    for index_batch in range(int(np.ceil(int_total_num_rows / int_num_rows_in_batch))):
        sl = slice(
            index_batch * int_num_rows_in_batch,
            (index_batch + 1) * int_num_rows_in_batch,
        )
        za_sink[sl] = za[sl]  # copy batch by batch

    # copy metadata
    dict_attrs = dict(za.attrs)  # retrieve metadata from the source
    for key in dict_attrs:  # write metadata to sink Zarr object
        za_sink.attrs[key] = dict_attrs[key]


def zarr_start_multiprocessing_write():
    """# 2022-08-07 20:55:26
    change setting for write of a zarr object using multiple processes
    since a zarr object will be modified by multiple processes, setting 'numcodecs.blosc.use_threads' to False as recommended by the zarr documentation
    """
    numcodecs.blosc.use_threads = False


def zarr_end_multiprocessing_write():
    """# 2022-08-07 20:55:26
    revert setting back from the write of a zarr object using multiple processes
    """
    numcodecs.blosc.use_threads = None


""" a class for containing disk-backed AnnData objects """


class AnnDataContainer:
    """# 2022-06-09 18:35:04
    AnnDataContainer
    Also contains utility functions for handling multiple AnnData objects on the disk sharing the same list of cells

    this object will contain AnnData objects and their file paths on the disk, and provide a convenient interface of accessing the items.

    'flag_enforce_name_adata_with_only_valid_characters' : (Default : True). does not allow the use of 'name_adata' containing the following characters { ' ', '/', '-', '"', "'" ";", and other special characters... }
    'path_prefix_default' : a default path of AnnData on disk. f'{path_prefix_default}{name_adata}.h5ad' will be used.
    'mode' : file mode. 'r' for read-only mode and 'a' for mode allowing modifications
    'path_prefix_default_mask' : the LOCAL file system path of 'MASK' where the modifications of the current object will be saved and retrieved. If this attribute has been set, the given RamData in the the given 'path_folder_ramdata' will be used as READ-ONLY. For example, when RamData resides in the HTTP server, data is often read-only (data can be only fetched from the server, and not the other way around). However, by giving a local path through this argument, the read-only RamData object can be analyzed as if the RamData object can be modified. This is possible since all the modifications made on the input RamData will be instead written to the local RamData object 'mask' and data will be fetced from the local copy before checking the availability in the remote RamData object.
    'flag_is_read_only' : read-only status of the storage

    '** args' : a keyworded argument containing name of the AnnData and the path to h5ad file of an AnnData object and an AnnData object (optional) :
            args = {
                    name_adata_1 = { 'adata' : AnnDataObject }, # when AnnDataObject is given and file path is not given, the path composed from 'path_prefix_default' and 'name_adata' will be used.
                    name_adata_2 = { 'path' : 'path/to/adata', 'adata' : AnnDataObject }, # when AnnDataObject is given, the validity of the path will not be checked.
                    name_adata_3 = { 'path' : 'path/to/adata', 'adata' : None }, # when adata is not loaded in memory. in this case, the path should be valid (the path validility will be checked)
                    name_adata_4 = { 'path' : 'path/to/adata' }, # same as the previous example. # when adata is not loaded in memory. in this case, the path should be valid (the path validility will be checked)
                    name_adata_5 = 'path/to/adata', # same as the previous example. # when adata is not loaded in memory. in this case, the path should be valid (the path validility will be checked),
                    name_adata_6 = AnnDataObject, # the default path will be used, but the validity will not be checkec
                    name_adata_7 = None, # when None is given, the default path will be used, and the validity will be checked
            }
                in summary, (1) when AnnData is given, the validity of the path will not be checked, (2) when path is not given, the default path will be used.
    """

    def __init__(
        self,
        flag_enforce_name_adata_with_only_valid_characters=True,
        path_prefix_default=None,
        mode="a",
        path_prefix_default_mask=None,
        flag_is_read_only=False,
        **args,
    ):
        import scanpy

        self.__str_invalid_char = (
            "! @#$%^&*()-=+`~:;[]{}\|,<.>/?" + '"' + "'"
            if flag_enforce_name_adata_with_only_valid_characters
            else ""
        )
        self.path_prefix_default = path_prefix_default
        self._mode = mode
        self._path_prefix_default_mask = path_prefix_default_mask
        self._flag_is_read_only = flag_is_read_only
        self._dict_name_adata_to_namespace = dict()

        # add items
        for name_adata in args:
            self.__setitem__(name_adata, args[name_adata])

    def __getitem__(self, name_adata):
        return self._dict_name_adata_to_namespace[name_adata]["adata"]

    def __setitem__(self, name_adata, args):
        # check whether the given name_adata contains invalid characters(s)
        for char_invalid in self.__str_invalid_char:
            if char_invalid in name_adata:
                raise TypeError(
                    f'the following characters cannot be used in "name_adata": {self.__str_invalid_char}'
                )

        # if the given argument is not a dictionary format, convert it to the dictionary format
        if not isinstance(args, dict):
            # check the type of input value
            if isinstance(args, scanpy.AnnData):
                args = {"adata": args}
            elif isinstance(args, str):
                args = {"path": args}
            else:
                args = dict()

        # set the default file path
        if "path" not in args:
            args["path"] = f"{self.path_prefix_default}{name_adata}.h5ad"
        # check validity of the path if AnnDataObject was not given
        if "adata" not in args:
            if not filesystem_operations("exists", args["path"]):
                raise FileNotFoundError(
                    f"{args[ 'path' ]} does not exist, while AnnData object is not given"
                )
            args["adata"] = None  # put a placeholder value

        self._dict_name_adata_to_namespace[name_adata] = args
        setattr(self, name_adata, args["adata"])

    def __delitem__(self, name_adata):
        """# 2022-06-09 12:47:36
        remove the adata from the memory and the object
        """
        # remove adata attribute from the dictionary
        if name_adata in self._dict_name_adata_to_namespace:
            del self._dict_name_adata_to_namespace[name_adata]
        # remove adata attribute from the current object
        if hasattr(self, name_adata):
            delattr(self, name_adata)

    def __contains__(self, name_adata):
        return name_adata in self._dict_name_adata_to_namespace

    def __iter__(self):
        return iter(self._dict_name_adata_to_namespace)

    def __repr__(self):
        return f"<AnnDataContainer object with the following items: {list( self._dict_name_adata_to_namespace )}\n\t default prefix is {self.path_prefix_default}>"

    def load(self, *l_name_adata):
        """# 2022-05-24 02:33:36
        load given anndata object(s) of the given list of 'name_adata'
        """
        import scanpy

        for name_adata in l_name_adata:
            if (
                name_adata not in self
            ):  # skip the 'name_adata' that does not exist in the current container
                continue
            args = self._dict_name_adata_to_namespace[name_adata]
            # if current AnnData has not been loaded
            if args["adata"] is None:
                args["adata"] = scanpy.read_h5ad(args["path"])
                self[name_adata] = args  # update the current name_adata

    def unload(self, *l_name_adata):
        """# 2022-06-09 12:47:42
        remove the adata object from the memory
        similar to __delitem__, but does not remove the attribute from the current 'AnnDataContainer' object
        """
        for name_adata in l_name_adata:
            if (
                name_adata not in self
            ):  # skip the 'name_adata' that does not exist in the current container
                continue
            args = self._dict_name_adata_to_namespace[name_adata]
            # if current AnnData has been loaded
            if args["adata"] is not None:
                args["adata"] = None
                self[name_adata] = args  # update the current name_adata

    def delete(self, *l_name_adata):
        """# 2022-06-09 12:58:45
        remove the adata from the memory, the current object, and from the disk
        """
        for name_adata in l_name_adata:
            if (
                name_adata not in self
            ):  # skip the 'name_adata' that does not exist in the current container
                continue
            # remove file on disk if exists
            path_file = self._dict_name_adata_to_namespace[name_adata]["path"]
            if filesystem_operations("exists", path_file):
                filesystem_operations("rm", path_file)
            del self[name_adata]  # delete element from the current object

    def update(self, *l_name_adata):
        """# 2022-06-09 18:13:21
        save the given AnnData objects to disk
        """
        import scanpy

        for name_adata in l_name_adata:
            if (
                name_adata not in self
            ):  # skip the 'name_adata' that does not exist in the current container
                continue
            # if current AnnData is a valid AnnData object
            args = self._dict_name_adata_to_namespace[name_adata]
            if isinstance(args["adata"], scanpy.AnnData):
                args["adata"].write(args["path"])  # write AnnData object

    def empty(self, *l_name_adata):
        """# 2022-06-09 18:23:44
        empty the count matrix of the given AnnData objects
        """
        import scanpy

        for name_adata in l_name_adata:
            if (
                name_adata not in self
            ):  # skip the 'name_adata' that does not exist in the current container
                continue
            # if current AnnData is a valid AnnData object
            adata = self._dict_name_adata_to_namespace[name_adata]["adata"]
            if isinstance(adata, scanpy.AnnData):
                adata.X = scipy.sparse.csr_matrix(
                    scipy.sparse.coo_matrix(
                        ([], ([], [])), shape=(len(adata.obs), len(adata.var))
                    )
                )  # empty the anndata object

    def transfer_attributes(self, name_adata, adata, flag_ignore_var=True):
        """# 2022-06-06 01:44:00
        transfer attributes of the given AnnDAta 'adata' to the current AnnData data containined in this object 'name_adata'
        'flag_ignore_var' : ignore attributes related to 'var' (var, varm, varp)
        """
        adata_current = self[name_adata]  # retrieve current AnnData

        # transfer uns and obs-related elements
        for name_attr in ["obs", "uns", "obsm", "obsp"]:
            if hasattr(adata, name_attr):
                setattr(adata_current, name_attr, getattr(adata, name_attr))

        # transfer var-related elements if 'flag_ignore_var' is True
        if not flag_ignore_var:
            for name_attr in ["var", "varm", "varp"]:
                if hasattr(adata, name_attr):
                    setattr(adata_current, name_attr, getattr(adata, name_attr))


""" a class for wrapping shelve-backed persistent dictionary """


class ShelveContainer:
    """# 2022-07-14 20:29:42
    a convenient wrapper of 'shelve' module-backed persistant dictionary to increase the memory-efficiency of a shelve-based persistent dicitonary, enabling the use of shelve dictionary without calling close( ) function to remove the added elements from the memory.

    'path_prefix_shelve' : a prefix of the persisent dictionary
    'mode' : file mode. 'r' for read-only mode and 'a' for mode allowing modifications
    'path_prefix_shelve_mask' : the LOCAL file system path of 'MASK' where the modifications of the current object will be saved and retrieved. If this attribute has been set, the given RamData in the the given 'path_folder_ramdata' will be used as READ-ONLY. For example, when RamData resides in the HTTP server, data is often read-only (data can be only fetched from the server, and not the other way around). However, by giving a local path through this argument, the read-only RamData object can be analyzed as if the RamData object can be modified. This is possible since all the modifications made on the input RamData will be instead written to the local RamData object 'mask' and data will be fetced from the local copy before checking the availability in the remote RamData object.
    'flag_is_read_only' : read-only status of the storage

    """

    def __init__(
        self,
        path_prefix_shelve,
        mode="a",
        path_prefix_shelve_mask=None,
        flag_is_read_only=False,
    ):
        """# 2022-07-20 22:06:15"""
        # set attributes
        self.path_prefix_shelve = path_prefix_shelve
        self._mode = mode
        self._path_prefix_shelve_mask = path_prefix_shelve_mask
        self._flag_is_read_only = flag_is_read_only

    @property
    def keys(self):
        """# 2022-07-14 20:43:24
        return keys
        """
        # if keys has not been loaded
        if not hasattr(self, "_set_keys"):
            self.update_keys()  # update keys
        return self._set_keys

    def update_keys(self):
        """# 2022-07-14 20:41:20
        update keys of 'shelve'
        """
        with shelve.open(self.path_prefix_shelve) as ns:
            self._set_keys = set(ns.keys())

    def __contains__(self, x) -> bool:
        """# 2022-07-14 21:14:47"""
        return x in self.keys

    def __iter__(self):
        """# 2022-07-14 21:15:18"""
        return iter(self.keys)

    def __getitem__(self, x):
        """# 2022-07-14 21:18:50"""
        if x in self:
            with shelve.open(self.path_prefix_shelve, "r") as ns:
                item = ns[str(x)]  # only string is avaiable as a key
            return item

    def __setitem__(self, x, val):
        """# 2022-07-14 21:22:54"""
        with shelve.open(self.path_prefix_shelve) as ns:
            ns[str(x)] = val  # only string is avaiable as a key
            self._set_keys = set(ns.keys())  # update keys

    def __delitem__(self, x):
        """# 2022-07-14 21:37:25"""
        with shelve.open(self.path_prefix_shelve) as ns:
            del ns[str(x)]  # only string is avaiable as a key
            self._set_keys = set(ns.keys())  # update keys

    def __repr__(self):
        """# 2022-07-14 21:37:28"""
        return f"<shelve-backed namespace: {self.keys}>"


""" a class for serving zarr object from remote source in multiple forked processes """


def zarr_object_server(
    path_folder_zarr: Union[None, str],
    pipe_receiver_input,
    pipe_sender_output,
    mode: str = "r",
    shape=None,
    chunks=None,
    dtype=np.int32,
    fill_value=0,
    path_process_synchronizer: Union[str, None] = None,
):
    """# 2023-04-19 01:33:09
    open a zarr object and serve various operations

    path_folder_zarr : Union[ None, str ] # if None is given, does not open a zarr object
    'mode' : zarr object mode
    shape = None, chunks = None, dtype = None, fill_value = None # arguments for initializing the output zarr object when mode = 'w' and the output zarr object does not exist
    path_process_synchronizer : Union[ str, None ] = None # path to the process synchronizer. if None is given, does not use any synchronizer
    """
    flag_zarr_opened = False  # flag indicating a zarr has been opened
    flag_zarr_array = False  # flag indicating a zarr array has been opened
    if (
        path_folder_zarr is not None
    ):  # if a valid path to the Zarr object has been given
        # %% OPEN ZARR %% # open a zarr object
        ps_za = (
            None
            if path_process_synchronizer is None
            else zarr.ProcessSynchronizer(path_process_synchronizer)
        )  # initialize process synchronizer (likely based on a local file system)
        if mode != "r":  # create a new zarr object
            za = zarr.open(
                path_folder_zarr,
                mode,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                fill_value=fill_value,
                synchronizer=ps_za,
            )
        else:  # use existing zarr object
            za = zarr.open(path_folder_zarr, mode, synchronizer=ps_za)
        # return information about the opened zarr object
        if hasattr(za, "shape"):
            pipe_sender_output.send(
                (za.shape, za.chunks, za.dtype, za.fill_value)
            )  # send basic information about the zarr array
            flag_zarr_array = True  # set the flag to True
        else:
            pipe_sender_output.send(
                (None, None, None, None)
            )  # return None values for a zarr group
        flag_zarr_opened = True  # set the flag to True
    else:
        pipe_sender_output.send((None, None, None, None))  # return None values

    while True:
        e = pipe_receiver_input.recv()
        if e is None:  # exit if None is received
            break
        name_func, args, kwargs = e  # parse the input
        if name_func == "open":  # open a new zarr object
            (
                path_folder_zarr,
                mode,
                shape,
                chunks,
                dtype,
                fill_value,
                path_process_synchronizer,
            ) = args  # parse arguments
            # %% OPEN ZARR %% # open a zarr object
            ps_za = (
                None
                if path_process_synchronizer is None
                else zarr.ProcessSynchronizer(path_process_synchronizer)
            )  # initialize process synchronizer (likely based on a local file system)
            if mode != "r":  # create a new zarr object
                if (
                    shape is None or chunks is None
                ):  # if one of the arguments for opening zarr array is invalid, open zarr group instead
                    za = zarr.open(path_folder_zarr, mode)
                else:  # open zarr array
                    za = zarr.open(
                        path_folder_zarr,
                        mode,
                        shape=shape,
                        chunks=chunks,
                        dtype=dtype,
                        fill_value=fill_value,
                        synchronizer=ps_za,
                    )
            else:  # use existing zarr object
                za = zarr.open(path_folder_zarr, mode, synchronizer=ps_za)
            # return information about the opened zarr object
            if hasattr(za, "shape"):
                pipe_sender_output.send(
                    (za.shape, za.chunks, za.dtype, za.fill_value)
                )  # send basic information about the zarr array
                flag_zarr_array = True  # set the flag to True
            else:
                pipe_sender_output.send(
                    (None, None, None, None)
                )  # return None values for a zarr group
                flag_zarr_array = False  # set the flag to False
            flag_zarr_opened = True  # set the flag to True
        elif flag_zarr_opened:  # when a zarr object has been opened
            if name_func == "get_attrs":
                set_keys = set(za.attrs)  # retrieve a list of keys
                pipe_sender_output.send(
                    dict((key, za.attrs[key]) for key in args if key in set_keys)
                )  # return a subset of metadata using the list of key values given as 'args'
            elif name_func == "set_attrs":
                # update the metadata
                for key in kwargs:
                    za.attrs[key] = kwargs[key]
                pipe_sender_output.send(
                    None
                )  # indicate the operation has been completed
            elif flag_zarr_array:  # when the opened Zarr object is Zarr array
                if name_func == "__getitem__":
                    pipe_sender_output.send(
                        getattr(za, name_func)(args)
                    )  # perform the '__getitem__' operation
                elif name_func == "__setitem__":
                    pipe_sender_output.send(
                        getattr(za, name_func)(args, kwargs)
                    )  # perform the '__setitem__' operation # for '__setitem__' operation, 'kwargs' represents 'values'
                else:
                    pipe_sender_output.send(
                        getattr(za, name_func)(*args, **kwargs)
                    )  # perform other operations, and return the result
            else:  # if zarr object is not array, does not perform any operations other than get/set attributes operations
                pipe_sender_output.send(None)  # return None value
        else:  # if zarr has not been opened, does not perform any operations
            pipe_sender_output.send(None)  # return None value


class ZarrServer:
    """# 2023-04-19 01:33:17
    This class is for serving zarr object in a spawned process or the current process for thread-safe operation.
    API functions calls mimic those of a zarr object for seamless replacement of a zarr object.

    path_folder_zarr : str # a path to a (remote) zarr object
    mode : str = 'r' # mode

    flag_spawn : bool = True # if True, spawn a new process for zarr operations. if False, perform zarr operations in the current process.
        (both are blocking and synchronous. the difference is that zarr operations that are not fork-safe can be performed in forked process by spawning a new process and interacting with the process using pipes)

    path_process_synchronizer : Union[ str, None ] = None # path to the process synchronizer. if None is given, does not use any synchronizer

    """

    def __init__(
        self,
        path_folder_zarr,
        mode="r",
        shape=None,
        chunks=None,
        dtype=np.int32,
        fill_value=0,
        flag_spawn: bool = True,
        path_process_synchronizer: Union[str, None] = None,
    ):
        """# 2023-04-19 15:02:17"""
        # set read-only attributes
        self._flag_spawn = flag_spawn  # indicate that a process has been spawned

        # set attributes
        self.is_zarr_server = True  # indicate that current object is ZarrServer
        self._mode = mode
        self._path_folder_zarr = path_folder_zarr
        self._path_process_synchronizer = path_process_synchronizer
        self._flag_is_terminated = False

        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            # create pipes for interactions
            mpsp = mp.get_context("spawn")
            pipe_sender_input, pipe_receiver_input = mpsp.Pipe()
            pipe_sender_output, pipe_receiver_output = mpsp.Pipe()

            self._pipe_sender_input = pipe_sender_input
            self._pipe_receiver_output = pipe_receiver_output

            # start the process hosting a zarr object
            p = mpsp.Process(
                target=zarr_object_server,
                args=(
                    path_folder_zarr,
                    pipe_receiver_input,
                    pipe_sender_output,
                    mode,
                    shape,
                    chunks,
                    dtype,
                    fill_value,
                    path_process_synchronizer,
                ),
            )
            p.start()
            self._p = p

            # retrieve attributes of a zarr object
            (
                self.shape,
                self.chunks,
                self.dtype,
                self.fill_value,
            ) = self._pipe_receiver_output.recv()  # set attributes
        else:
            if path_folder_zarr is None:  # if 'path_folder_zarr' is invalid
                self.shape, self.chunks, self.dtype, self.fill_value, self._za = (
                    None,
                    None,
                    None,
                    None,
                    None,
                )  # set attributes to None values to indicate the Zarr object has not been opened
            else:  # if 'path_folder_zarr' is valid
                # open a zarr object
                if mode != "r":  # create a new zarr object
                    if (
                        shape is None or chunks is None
                    ):  # if one of the arguments for opening zarr array is invalid, open zarr group instead
                        za = zarr.open(path_folder_zarr, mode)
                    else:  # open zarr array
                        za = zarr.open(
                            path_folder_zarr,
                            mode,
                            shape=shape,
                            chunks=chunks,
                            dtype=dtype,
                            fill_value=fill_value,
                        )
                else:  # use existing zarr object
                    za = zarr.open(path_folder_zarr, mode)
                self._za = za  # set the zarr object as an attribute
                # retrieve attributes of a zarr object
                self.shape, self.chunks, self.dtype, self.fill_value = (
                    self._za.shape,
                    self._za.chunks,
                    self._za.dtype,
                    self._za.fill_value,
                )

    @property
    def path_folder(self):
        """# 2023-04-19 17:33:21"""
        return self._path_folder_zarr

    @property
    def flag_spawn(self):
        """# 2022-12-05 22:26:33
        return a flag indicating whether a process has been spawned and interacting with the current object or not.
        """
        return self._flag_spawn

    def __repr__(self):
        """# 2023-04-20 01:06:16"""
        return f"<ZarrServer of {path_folder_zarr}>"

    @property
    def path_process_synchronizer(self):
        """# 2022-12-07 00:19:29
        return a path of the folder used for process synchronization
        """
        return self._path_process_synchronizer

    def open(
        self,
        path_folder_zarr,
        mode="r",
        shape=None,
        chunks=None,
        dtype=np.int32,
        fill_value=0,
        path_process_synchronizer: Union[str, None] = None,
        reload: bool = False,
    ):
        """# 2023-04-20 02:08:57
        open a new zarr in a ZarrServer object

        reload : bool = False # if True, reload the zarr object even if the 'path_folder' and 'mode' are identical to the currently opened Zarr object. (useful when folder has been updated using the external methods.)
        """
        # if the zarr object is already opened in the same mode, exit, unless 'reload' flag has been set to True.
        if not reload and path_folder_zarr == self.path_folder and self._mode == mode:
            return
        # open Zarr object
        args = (
            path_folder_zarr,
            mode,
            shape,
            chunks,
            dtype,
            fill_value,
            path_process_synchronizer,
        )  # compose arguments
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(("open", args, None))  # send input
            # retrieve attributes of a zarr object
            (
                self.shape,
                self.chunks,
                self.dtype,
                self.fill_value,
            ) = self._pipe_receiver_output.recv()  # set attributes
        else:
            # open a zarr object
            if mode != "r":  # create a new zarr object
                if (
                    shape is None or chunks is None
                ):  # if one of the arguments for opening zarr array is invalid, open zarr group instead
                    za = zarr.open(path_folder_zarr, mode)
                else:  # open zarr array
                    za = zarr.open(
                        path_folder_zarr,
                        mode,
                        shape=shape,
                        chunks=chunks,
                        dtype=dtype,
                        fill_value=fill_value,
                    )
            else:  # use existing zarr object
                za = zarr.open(path_folder_zarr, mode)
            self._za = za  # set the zarr object as an attribute
            # retrieve attributes of a zarr array
            if hasattr(za, "shape"):  # if zarr object is an array
                self.shape, self.chunks, self.dtype, self.fill_value = (
                    self._za.shape,
                    self._za.chunks,
                    self._za.dtype,
                    self._za.fill_value,
                )
            else:  # if zarr object is a group
                self.shape, self.chunks, self.dtype, self.fill_value = (
                    None,
                    None,
                    None,
                    None,
                )
        # update the attributes
        self._path_folder_zarr = path_folder_zarr
        self._mode = mode

    def get_attrs(self, *keys):
        """# 2023-04-19 15:00:04
        get an attribute of the currently opened zarr object using the list of key values
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(("get_attrs", keys, None))  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            set_keys = set(self._za.attrs)  # retrieve a list of keys
            return dict(
                (key, self._za.attrs[key]) for key in keys if key in set_keys
            )  # return a subset of metadata using the list of key values given as 'args'

    def get_attr(self, key):
        """# 2023-04-20 01:08:59
        a wrapper of 'get_attrs' for a single key value
        """
        dict_attrs = self.get_attrs(key)  # retrieve the attributes
        if key not in dict_attrs:
            raise KeyError(
                f"attribute {key} does not exist in the zarr object."
            )  # raise a key error if the key does not exist
        return dict_attrs[key]

    def set_attrs(self, **kwargs):
        """# 2023-04-19 15:00:00
        update the attributes of the currently opened zarr object using the keyworded arguments
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(("set_attrs", None, kwargs))  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            # update the metadata
            for key in kwargs:
                self._za.attrs[key] = kwargs[key]

    def get_coordinate_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'get_coordinate_selection' zarr operation using a spawned process.
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(
                ("get_coordinate_selection", args, kwargs)
            )  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            # run a zarr operation in the current process
            return self._za.get_coordinate_selection(*args, **kwargs)

    def get_basic_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'get_basic_selection' zarr operation using a spawned process.
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(
                ("get_basic_selection", args, kwargs)
            )  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            # run a zarr operation in the current process
            return self._za.get_basic_selection(*args, **kwargs)

    def get_orthogonal_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'get_orthogonal_selection' zarr operation using a spawned process.
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(
                ("get_orthogonal_selection", args, kwargs)
            )  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            # run a zarr operation in the current process
            return self._za.get_orthogonal_selection(*args, **kwargs)

    def get_mask_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'get_mask_selection' zarr operation using a spawned process.
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(
                ("get_mask_selection", args, kwargs)
            )  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            # run a zarr operation in the current process
            return self._za.get_mask_selection(*args, **kwargs)

    def set_coordinate_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'set_coordinate_selection' zarr operation using a spawned process.
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(
                ("set_coordinate_selection", args, kwargs)
            )  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            # run a zarr operation in the current process
            return self._za.set_coordinate_selection(*args, **kwargs)

    def set_basic_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'set_basic_selection' zarr operation using a spawned process.
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(
                ("set_basic_selection", args, kwargs)
            )  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            # run a zarr operation in the current process
            return self._za.set_basic_selection(*args, **kwargs)

    def set_orthogonal_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'set_orthogonal_selection' zarr operation using a spawned process.
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(
                ("set_orthogonal_selection", args, kwargs)
            )  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            # run a zarr operation in the current process
            return self._za.set_orthogonal_selection(*args, **kwargs)

    def set_mask_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'set_mask_selection' zarr operation using a spawned process.
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(
                ("set_mask_selection", args, kwargs)
            )  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            # run a zarr operation in the current process
            return self._za.set_mask_selection(*args, **kwargs)

    def resize(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'resize' zarr operation using a spawned process.
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(("resize", args, kwargs))  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            # run a zarr operation in the current process
            return self._za.resize(*args, **kwargs)

    def __getitem__(self, args):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the '__getitem__' zarr operation using a spawned process.
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(
                ("__getitem__", args, None)
            )  # send input # no 'kwargs' arguments
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            return self._za.__getitem__(args)

    def __setitem__(self, args, values):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the '__setitem__' zarr operation using a spawned process.
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(
                ("__setitem__", args, values)
            )  # send input # put 'values' in the place for 'kwargs'
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            return self._za.__setitem__(args, values)

    def terminate(self):
        """# 2022-09-06 23:16:22
        terminate the server
        """
        if self.flag_spawn and not self._flag_is_terminated:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(None)
            self._p.join()  # wait until the process join the main process
            self._flag_is_terminated = True  # set the flag

    def __enter__(self):
        """# 2022-12-08 02:00:08"""
        return self

    def __exit__(self):
        """# 2022-12-08 02:00:08
        terminate the spawned process when exiting the context
        """
        self.terminate()


def zarr_metadata_server(
    pipe_receiver_input, pipe_sender_output, dict_kwargs_credentials_s3: dict = dict()
):
    """# 2022-12-07 18:19:13
    a function for getting and setting zarr object metadata dictionaries

    dict_kwargs_credentials_s3 : dict = dict( ) # the credentials for the Amazon S3 file system as keyworded arguments

    'method' :
        'get_or_set_metadata' : set the metadata. if key does not exist in the metadata, return None. if a key is already available, return the value instead.
    """
    import zarr

    while True:
        ins = pipe_receiver_input.recv()
        if ins is None:  # exit if None is received
            break
        method, path_folder_zarr, key, value, mode = ins  # parse the input

        outs = None  # set default output
        try:
            za = zarr.open(path_folder_zarr, mode=mode)
        except:
            pipe_sender_output.send(outs)  # return the result
            continue

        if method == "get_or_set_metadata":
            if key in za.attrs:
                outs = za.attrs[key]
            else:
                za.attrs[key] = value
                outs = value  # return the value that was successfully set
        if method == "set_metadata":
            za.attrs[key] = value
            outs = value  # return the value that was successfully set
        elif method == "get_metadata":
            if key in za.attrs:  # try to retrieve the value of the 'key'
                outs = za.attrs[key]
        pipe_sender_output.send(outs)  # return the result


class ZarrMetadataServer:
    """# 2022-12-07 18:57:38
    This class is for getting and setting zarr object metadata in a spawned process or the current process for thread-safe operation.
    API functions calls mimic those of a zarr object for seamless replacement of a zarr object

    dict_kwargs_credentials_s3 : dict = dict( ) # the credentials for the Amazon S3 file system as keyworded arguments
    flag_spawn : bool = True # if True, spawn a new process for zarr operations. if False, perform zarr operations in the current process.
        (both are blocking and synchronous. the difference is that zarr operations that are not fork-safe can be performed in forked process by spawning a new process and interacting with the process using pipes)
    """

    def __init__(
        self, flag_spawn: bool = True, dict_kwargs_credentials_s3: dict = dict()
    ):
        """# 2022-12-07 18:55:04"""
        # set read-only attributes
        self._flag_spawn = flag_spawn  # indicate that a process has been spawned

        # set attributes
        self._flag_is_terminated = False
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            # create pipes for interactions
            mpsp = mp.get_context("spawn")
            pipe_sender_input, pipe_receiver_input = mpsp.Pipe()
            pipe_sender_output, pipe_receiver_output = mpsp.Pipe()

            self._pipe_sender_input = pipe_sender_input
            self._pipe_receiver_output = pipe_receiver_output

            # start the process hosting a zarr object
            p = mpsp.Process(
                target=zarr_metadata_server,
                args=(
                    pipe_receiver_input,
                    pipe_sender_output,
                    dict_kwargs_credentials_s3,
                ),
            )
            p.start()
            self._p = p

    @property
    def flag_spawn(self):
        """# 2022-12-05 22:26:33
        return a flag indicating whether a process has been spawned and interacting with the current object or not.
        """
        return self._flag_spawn

    def get_or_set_metadata(
        self, path_folder_zarr: str, key: str, value, mode: str = "a"
    ):
        """# 2022-12-07 18:59:43
        a (possibly) fork-safe method for getting or setting zarr group metadata

        === return ===
        None : None will be returned if the operation has failed.
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(
                ("get_or_set_metadata", path_folder_zarr, key, value, mode)
            )  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            # run a zarr operation in the current process
            outs = None  # set default output
            try:
                za = zarr.open(path_folder_zarr, mode=mode)
            except:  # if opening the zarr object fails, return None
                return outs

            # 'get_or_set_metadata' operation
            if key in za.attrs:
                outs = za.attrs[key]
            else:
                za.attrs[key] = value
                outs = value  # return the value that was successfully set
            return outs

    def get_metadata(self, path_folder_zarr: str, key: str, mode: str = "r"):
        """# 2022-12-07 18:59:43
        a (possibly) fork-safe method for getting zarr group metadata

        === return ===
        None : None will be returned if the operation has failed.
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(
                ("get_metadata", path_folder_zarr, key, None, mode)
            )  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            # run a zarr operation in the current process
            outs = None  # set default output
            try:
                za = zarr.open(path_folder_zarr, mode=mode)
            except:  # if opening the zarr object fails, return None
                return outs

            # 'get_metadata' operation
            if key in za.attrs:  # try to retrieve the value of the 'key'
                outs = za.attrs[key]
            return outs

    def set_metadata(self, path_folder_zarr: str, key: str, value, mode: str = "a"):
        """# 2022-12-07 18:59:43
        a (possibly) fork-safe method for setting zarr group metadata

        === return ===
        None : None will be returned if the operation has failed.
        """
        if self.flag_spawn:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(
                ("set_metadata", path_folder_zarr, key, value, mode)
            )  # send input
            return self._pipe_receiver_output.recv()  # retrieve result and return
        else:
            # run a zarr operation in the current process
            outs = None  # set default output
            try:
                za = zarr.open(path_folder_zarr, mode=mode)
            except:  # if opening the zarr object fails, return None
                return outs

            # 'set_metadata' operation
            za.attrs[key] = value
            outs = value  # return the value that was successfully set
            return outs

    def terminate(self):
        """# 2022-09-06 23:16:22
        terminate the server
        """
        if self.flag_spawn and not self._flag_is_terminated:
            # %% PROCESS SPAWNING %%
            self._pipe_sender_input.send(None)
            self._p.join()  # wait until the process join the main process
            self._flag_is_terminated = True  # set the flag

    def __enter__(self):
        """# 2022-12-08 02:00:08"""
        return self

    def __exit__(self):
        """# 2022-12-08 02:00:08
        terminate the spawned process when exiting the context
        """
        self.terminate()


""" a class for file-system-backed synchronization of zarr objects """


class ZarrSpinLockServer:
    """# 2022-12-11 19:36:14
    A class for acquiring, waiting, releasing for a spin-lock based on a file system and the Zarr format

    === arguments ===
    flag_spawn : bool = False # when used in a forked process and path to the lock object is remote (e.g. Amazon S3), please set this flag to True to avoid runtime error. it will create appropriate ZarrMetadataServer and FileSystemServer objects to handle lock operations in a fork-safe manner.
    dict_kwargs_credentials_s3 : dict = dict( ) # the credentials for the Amazon S3 file system as keyworded arguments
    filesystem_server : Union[ None, FileSystemServer ] = None # a FileSystemServer object to use. if None is given, start the new server based on the setting
    zarrmetadata_server : Union[ None, ZarrMetadataServer ] = None # a FileSystemServer object to use. if None is given, start the new server based on the setting
    verbose : bool = False # an arugment for debugging purpose

    flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock : bool = False # if True, does not wait and raise 'RuntimeError' when a modification of a RamData cannot be made due to the resource that need modification is temporarily unavailable, locked by other processes
    float_second_to_wait_before_checking_availability_of_a_spin_lock : float = 0.5 # number of seconds to wait before repeatedly checking the availability of a spin lock if the lock has been acquired by other operations.
    """

    def __init__(
        self,
        flag_spawn: bool = False,
        dict_kwargs_credentials_s3: dict = dict(),
        flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock: bool = False,
        float_second_to_wait_before_checking_availability_of_a_spin_lock: float = 0.1,
        filesystem_server: Union[None, FileSystemServer] = None,
        zarrmetadata_server: Union[None, ZarrMetadataServer] = None,
        template=None,
        verbose: bool = False,
    ):
        """# 2022-12-11 14:03:53"""
        # set read-only attributes
        self._flag_spawn = flag_spawn  # indicate that a process has been spawned
        self._str_uuid_lock = (
            bk.UUID()
        )  # a unique id of the current ZarrSpinLockServer object. This id will be used to acquire and release locks so that lock can only be released by the object that acquired the lock
        self.verbose = verbose

        # set attributes that can be changed anytime during the lifetime of the object
        if isinstance(template, ZarrSpinLockServer):
            # retrieve attributes from the template
            self.flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock = (
                template.flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock
            )
            self.float_second_to_wait_before_checking_availability_of_a_spin_lock = (
                template.float_second_to_wait_before_checking_availability_of_a_spin_lock
            )
        else:
            self.flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock = flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock
            self.float_second_to_wait_before_checking_availability_of_a_spin_lock = (
                float_second_to_wait_before_checking_availability_of_a_spin_lock
            )

        # set attributes
        self._flag_is_terminated = False

        # start servers required for operations
        self.fs = (
            filesystem_server
            if isinstance(filesystem_server, FileSystemServer)
            else FileSystemServer(
                flag_spawn=flag_spawn,
                dict_kwargs_credentials_s3=dict_kwargs_credentials_s3,
            )
        )
        self.zms = (
            zarrmetadata_server
            if isinstance(zarrmetadata_server, ZarrMetadataServer)
            else ZarrMetadataServer(
                flag_spawn=flag_spawn,
                dict_kwargs_credentials_s3=dict_kwargs_credentials_s3,
            )
        )

        # initialize a set for saving the list of lock objects current ZarrSpinLockServer has acquired in order to ignore additional attempts to acquire the lock that has been already acquired
        self._set_path_folder_lock = set()

    @property
    def flag_spawn(self):
        """# 2022-12-11 14:04:21
        return whether spawned processes are used to perform fork-safe operations
        """
        return self._flag_spawn

    @property
    def str_uuid_lock(self):
        """# 2022-12-11 14:04:21
        return a unique id of the current ZarrSpinLockServer object
        """
        return self._str_uuid_lock

    @property
    def currently_held_locks(self):
        """# 2022-12-11 16:56:33
        return a copy of a set containing path_folder_lock of all the lock objects current ZarrSpinLockServer has acquired.
        """
        return set(self._set_path_folder_lock)

    def terminate(self):
        """# 2022-09-06 23:16:22
        terminate the server
        """
        if (
            len(self.currently_held_locks) > 0
        ):  # if unreleased locks are present, raise a RuntimeError
            raise RuntimeError(
                f"there are unreleased locks held by current ZarrSpinLockServer object being terminated. the list of the acquired locks are the following: {self.currently_held_locks}."
            )
        self.fs.terminate()
        self.zms.terminate()

    def __enter__(self):
        """# 2022-12-08 02:00:08"""
        return self

    def __exit__(self):
        """# 2022-12-08 02:00:08
        terminate the spawned process when exiting the context
        """
        self.terminate()

    def process_path_folder_lock(self, path_folder_lock):
        """# 2022-12-11 22:40:37
        process the given 'process_path_folder_lock'
        """
        # add '/' at the end of the 'path_folder_lock'
        if path_folder_lock[-1] != "/":
            path_folder_lock += "/"
        return path_folder_lock

    def check_lock(self, path_folder_lock: str):
        """# 2022-12-10 21:32:38
        check whether the lock currently exists, based on the file system where the current lock object resides.

        path_folder_lock : str # an absolute (full-length) path to the lock (an absolute path to the zarr object, representing a spin lock)
        """
        # process 'path_folder_lock'
        path_folder_lock = self.process_path_folder_lock(path_folder_lock)
        # return the flag indicating whether the lock exists
        return self.fs.filesystem_operations("exists", f"{path_folder_lock}.zattrs")

    def wait_lock(self, path_folder_lock: str):
        """# 2022-12-10 21:32:38
        wait for the lock, based on the file system where the current lock object resides.

        path_folder_lock : str # an absolute (full-length) path to the lock (an absolute path to the zarr object, representing a spin lock)
        """
        if self.verbose:
            logger.info(
                f"the current ZarrSpinLockServer ({self.str_uuid_lock}) is trying to wait for the lock '{path_folder_lock}', with currently_held_locks '{self.currently_held_locks}'"
            )
        # process 'path_folder_lock'
        path_folder_lock = self.process_path_folder_lock(path_folder_lock)

        # if a lock for 'path_folder_lock' has been already acquired, does not wait for the lock
        if path_folder_lock in self.currently_held_locks:
            return

        # if lock is available and 'flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock' is True, raise a RuntimeError
        if (
            self.check_lock(path_folder_lock)
            and self.flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock
        ):
            raise RuntimeError(f"a lock is present at ({path_folder_lock}), exiting")
        # implement a spin lock using the sleep function
        while self.check_lock(path_folder_lock):  # until a lock is released
            time.sleep(
                self.float_second_to_wait_before_checking_availability_of_a_spin_lock
            )  # wait for 'float_second_to_wait_before_checking_availability_of_a_spin_lock' second

    def acquire_lock(self, path_folder_lock: str):
        """# 2022-12-29 03:12:52
        acquire the lock, based on the file system where the current lock object resides.

        === arguments ===
        path_folder_lock : str # an absolute (full-length) path to the lock (an absolute path to the zarr object, representing a spin lock)

        === returns ===
        return True if a lock has been acquired (a lock object was created).
        """
        if self.verbose:
            logger.info(
                f"the current ZarrSpinLockServer ({self.str_uuid_lock}) is trying to acquire the lock '{path_folder_lock}', with currently_held_locks '{self.currently_held_locks}'"
            )
        # process 'path_folder_lock'
        path_folder_lock = self.process_path_folder_lock(path_folder_lock)
        if (
            path_folder_lock not in self.currently_held_locks
        ):  # if the lock object has not been previously acquired by the current object
            # create the lock zarr object
            while True:
                # attempts to acquire a lock
                res = self.zms.set_metadata(
                    path_folder_lock,
                    "dict_metadata",
                    {"str_uuid_lock": self.str_uuid_lock, "time": int(time.time())},
                    "w-",
                )  # if the lock object already exists, acquiring lock would fail
                # if a lock appear to be acquired (a positive response that a zarr object has been create), check again that the written lock belongs to the current object
                # when two processes attempts to acquire the same lock object within a time window of 1 ms, two processes will write the same lock object, but one will be overwritten by another.
                # therefore, the content of the lock should be checked again in order to ensure then the lock has been actually acquired.
                if res is not None:
                    # wait 'sufficient' time to ensure the acquired lock became visible to all other processes that have attempted to acquire the same lock when the lock has not been acquired by any other objects. (waiting the file-system judge's decision)
                    time.sleep(
                        self.float_second_to_wait_before_checking_availability_of_a_spin_lock
                    )  # wait for 'float_second_to_wait_before_checking_availability_of_a_spin_lock' second
                    lock_metadata = self.zms.get_metadata(
                        path_folder_lock, "dict_metadata"
                    )  # read the content of the written lock
                    if (
                        lock_metadata is not None
                        and "str_uuid_lock" in lock_metadata
                        and lock_metadata["str_uuid_lock"] == self.str_uuid_lock
                    ):  # if the lock has been written by the current object
                        break  # consider the lock has been acquired by the current object

                # wait until the lock becomes available
                # if lock is available and 'flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock' is True, raise a RuntimeError
                if (
                    self.flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock
                ):
                    raise RuntimeError(
                        f"a lock is present at ({path_folder_lock}), exiting"
                    )
                # implement a spin lock using the sleep function
                time.sleep(
                    self.float_second_to_wait_before_checking_availability_of_a_spin_lock
                )  # wait for 'float_second_to_wait_before_checking_availability_of_a_spin_lock' second

            # record the 'path_folder_lock' of the acquired lock object
            self._set_path_folder_lock.add(path_folder_lock)
            if self.verbose:
                logger.info(
                    f"the current ZarrSpinLockServer ({self.str_uuid_lock}) acquired the lock '{path_folder_lock}', with currently_held_locks '{self.currently_held_locks}'"
                )
            return True  # return True if a lock has been acquired
        else:
            return False  # return False if a lock has been already acquired prior to this function call

    def release_lock(self, path_folder_lock: str):
        """# 2022-12-10 21:32:38
        release the lock, based on the file system where the current lock object resides

        path_folder_lock : str # an absolute (full-length) path to the lock (an absolute path to the zarr object, representing a spin lock)
        """
        if self.verbose:
            logger.info(
                f"the current ZarrSpinLockServer ({self.str_uuid_lock}) is trying to release_lock the lock '{path_folder_lock}', with currently_held_locks '{self.currently_held_locks}'"
            )
        # process 'path_folder_lock'
        path_folder_lock = self.process_path_folder_lock(path_folder_lock)
        if (
            path_folder_lock in self.currently_held_locks
        ):  # if the lock object has been previously acquired by the current object
            lock_metadata = self.zms.get_metadata(
                path_folder_lock, "dict_metadata", "r"
            )  # retrieve the lock metadata
            if lock_metadata is not None and "str_uuid_lock" in lock_metadata:
                if (
                    lock_metadata["str_uuid_lock"] == self.str_uuid_lock
                ):  # if the lock has been acquired by the current object
                    self.fs.filesystem_operations(
                        "rm", path_folder_lock
                    )  # release the lock
                    if self.verbose:
                        logger.info(
                            f"the current ZarrSpinLockServer ({self.str_uuid_lock}) released the lock '{path_folder_lock}'"
                        )
                else:
                    logger.error(
                        f"the current ZarrSpinLockServer ({self.str_uuid_lock}) have acquired the lock {path_folder_lock} but it appears the lock belongs to another ZarrSpinLockServer ({lock_metadata[ 'str_uuid_lock' ]})."
                    )
                    # raise KeyError( f"the current ZarrSpinLockServer ({self.str_uuid_lock}) have acquired the lock {path_folder_lock} but it appears the lock belongs to another ZarrSpinLockServer ({lock_metadata[ 'str_uuid_lock' ]})." )
            else:
                logger.error(
                    f"the current ZarrSpinLockServer ({self.str_uuid_lock}) have acquired the lock {path_folder_lock} but the lock object does not exist."
                )
                # raise FileNotFoundError( f"the current ZarrSpinLockServer ({self.str_uuid_lock}) have acquired the lock {path_folder_lock} but the lock object does not exist." )
            self._set_path_folder_lock.remove(
                path_folder_lock
            )  # remove the released lock's 'path_folder_lock' from the list of the acquired lock objects

    """ </Methods for Locking> """


# other utility functions
def get_path_compatible_str(
    str_input: str,
    str_invalid_char: str = "<>:/\|?*"
    + '"',  # reserved characters in Windows file system # a string containing all the characters incompatible with a file path string
    int_max_num_bytes_in_a_folder_name: int = 255,  # the maximum number of bytes for a folder name in Linux
):
    """# 2023-05-05 22:51:50
    get path-compatible string from the input string, by replacing file-system-incompatible characters with escape characters.
    if the input string is incompatible with operating system (Linux, specifically), a 'FileNotFoundError' error will be raised.

    str_invalid_char : str = '/', # a string containing all the characters incompatible with a file path string
    int_max_num_bytes_in_a_folder_name : int = 255, # the maximum number of bytes for a folder name in Linux
    """
    str_input_original = str_input  # record the input column name
    # replace invalid characters in the column name
    for chr_invalid in str_invalid_char:
        if chr_invalid in str_input:
            str_input = str_input.replace(
                chr_invalid, "((" + str(ord(chr_invalid)) + "))"
            )

    # check the length limit
    if len(str_input.encode()) > int_max_num_bytes_in_a_folder_name:
        raise FileNotFoundError(
            f"the number of bytes for the path-compatible string '{str_input}' exceeded the maximum number of bytes, {int_max_num_bytes_in_a_folder_name}. '{str_input_original}' cannot be used."
        )

    return str_input


def MTX_Convert_10x_MEX_to_10x_HDF5_Format(
    path_folder_matrix_input_mex_format: str,
    path_file_matrix_output_hdf5_format: str,
    name_genome: str = "unknown",
    flag_round_float: bool = True,
):
    """# 2023-09-15 01:36:49
    path_folder_matrix_input_mex_format : str # the path of the input 10x MEX matrix folder
    path_file_matrix_output_hdf5_format : str # the path of the output 10x HDF5 matrix file
    name_genome : str = 'unknown' # the name of the genome
    flag_round_float : bool = True, # if True, if float values in the matrix is detected, round the values to convert to the integer values. However, when the value becomes 0 after being rounded, change the values back to 1 (to avoid exporting records with zero count values).
    """
    """ import libaries """
    import h5py

    """ read 10x MEX format """
    # read mtx file as a tabular format
    df_mtx = pd.read_csv(
        f"{path_folder_matrix_input_mex_format}matrix.mtx.gz", sep=" ", comment="%"
    )
    df_mtx.columns = ["id_row", "id_column", "read_count"]
    df_mtx.sort_values("id_column", inplace=True)  # sort using id_cell
    # read barcodes
    arr_bc = pd.read_csv(
        f"{path_folder_matrix_input_mex_format}barcodes.tsv.gz", sep="\t", header=None
    ).values.ravel()
    # read feature tables
    df_feature = pd.read_csv(
        f"{path_folder_matrix_input_mex_format}features.tsv.gz", sep="\t", header=None
    )
    df_feature.columns = ["id_feature", "feature", "feature_type"]

    """ write hdf5 file """
    newfile = h5py.File(path_file_matrix_output_hdf5_format, "w")  # open new HDF5 file

    def _write_string_array(handle, name_array: str, arr_str: List[str]):
        """# 2023-09-14 21:41:14
        write a string array to a HDF5 object
        """
        handle.create_dataset(
            name_array,
            (len(arr_str),),
            dtype="S" + str(np.max(list(len(e) for e in arr_str))),
            data=list(e.encode("ascii", "ignore") for e in arr_str),
        )  # writing string dtype array

    # create matrix group
    mtx = newfile.create_group("matrix")

    # write barcodes
    _write_string_array(mtx, "barcodes", arr_bc)

    # # write id/names

    # write data
    arr = df_mtx.read_count.values
    flag_dtype_is_integer = np.issubdtype(arr.dtype, np.integer)  # check integer dtype
    """ round the values in the matrix to convert float to integer """
    if (
        flag_round_float and not flag_dtype_is_integer
    ):  # if dtype is float, and 'flag_round_float' flag is set to True, round the values
        arr = np.rint(arr).astype(
            int
        )  # convert to integer type (round to the nearest integer)
        arr[arr == 0] = (
            1  # convert entries with 0 values counts to 1, so that the minimum value in the matrix is 1
        )
        flag_dtype_is_integer = True  # update the flag
    mtx.create_dataset("data", (len(arr),), "i8" if flag_dtype_is_integer else "f", arr)

    # write indices
    arr = df_mtx.id_row.values - 1  # 1 -> 0-based coordinates
    mtx.create_dataset("indices", (len(arr),), "i8", arr)

    # write shape
    mtx.create_dataset("shape", (2,), "i8", [len(df_feature), len(arr_bc)])

    # write indptr
    arr = df_mtx.id_column.values
    arr = arr - 1  # 1>0-based coordinate
    int_num_bc = len(arr_bc)  # retrieve the number of barcodes
    int_num_records = len(arr)  # retrieve the number of records
    arr_indptr = np.zeros(int_num_bc + 1, dtype="i8")  # initialize 'arr_indptr'
    arr_indptr[-1] = int_num_records  # last entry should be the number of the records
    id_col_current = arr[0]  # initialize 'id_col_current'
    for i, id_col in enumerate(arr):
        if id_col_current != id_col:
            if (
                id_col_current + 1 < id_col
            ):  # if there are some skipped columns ('barcodes' with zero number of records)
                for id_col_with_no_records in range(id_col_current + 1, id_col):
                    arr_indptr[id_col_with_no_records] = (
                        i  # add 'indptr' for the 'barcodes' with zero number of records
                    )
            id_col_current = id_col  # update 'id_col_current'
            arr_indptr[id_col] = i
    if id_col_current + 1 < int_num_bc:
        for id_col_with_no_records in range(id_col_current + 1, int_num_bc):
            arr_indptr[id_col_with_no_records] = (
                int_num_records  # add 'indptr' for the 'barcodes' with zero number of records
            )
    mtx.create_dataset("indptr", (len(arr_indptr),), "i8", arr_indptr)

    # create matrix group
    ft = mtx.create_group("features")

    # write features/id, features/name, features/feature_type
    _write_string_array(ft, "id", df_feature.id_feature.values)
    _write_string_array(ft, "name", df_feature.feature.values)
    _write_string_array(ft, "feature_type", df_feature.feature_type.values)
    _write_string_array(
        ft, "genome", [name_genome] * len(df_feature)
    )  # add genome data type (required for scanpy)

    # close the file
    newfile.close()


def MTX_Convert_10x_HDF5_to_10x_MEX_Format(
    path_file_matrix_input_hdf5_format: str,
    path_folder_matrix_output_mex_format: str,
):
    """# 2023-10-04 13:31:40
    path_file_matrix_input_hdf5_format: str, # the path of the input 10x HDF5 matrix file
    path_folder_matrix_output_mex_format: str, # the path of the output 10x MEX matrix folder
    """
    """ import libaries """
    import h5py
    import scipy.io

    """ read hdf5 file """
    newfile = h5py.File(path_file_matrix_input_hdf5_format, "r")  # open new HDF5 file
    mtx = newfile["matrix"]  # retrieve the group

    """ read barcodes """
    arr_bc = mtx["barcodes"][:]  # retrieve col (cell) boundary positions
    arr_bc = np.array(
        list(e.decode() for e in arr_bc), dtype=object
    )  # change dtype of the barcode

    """ read features """
    ft = mtx["features"]  # retrieve the group
    arr_id_ft = np.array(
        list(e.decode() for e in ft["id"][:]), dtype=object
    )  # change dtype of the barcode
    arr_id_name = np.array(
        list(e.decode() for e in ft["name"][:]), dtype=object
    )  # change dtype of the barcode
    arr_id_feature_type = np.array(
        list(e.decode() for e in ft["feature_type"][:]), dtype=object
    )  # change dtype of the barcode
    arr_genome = np.array(
        list(e.decode() for e in ft["genome"][:]), dtype=object
    )  # change dtype of the barcode
    # compose feature dataframe
    df_feature = pd.DataFrame(
        {
            "id_feature": arr_id_ft,
            "feature": arr_id_name,
            "feature_type": arr_id_feature_type,
            "genome": arr_genome,
        }
    )

    """ read count values """
    arr_data = mtx["data"][:]
    arr_row_indices = mtx["indices"][:]  # retrieve row (gene) indices
    arr_col_index_boundary = mtx["indptr"][:]  # retrieve col (cell) boundary positions
    # compose arr_col_indices
    arr_col_indices = np.zeros_like(arr_row_indices)  # initialize 'arr_col_indices'
    for idx_bc in range(len(arr_bc)):  # for each barcode index (0-based coordinates)
        arr_col_indices[
            arr_col_index_boundary[idx_bc] : arr_col_index_boundary[idx_bc + 1]
        ] = idx_bc
    # compose 'df_mtx'
    df_mtx = pd.DataFrame(
        {
            "id_row": arr_row_indices,  # 0 based coordinates
            "id_column": arr_col_indices,  # 0 based coordinates
            "read_count": arr_data,
        }
    )

    """ write the output MEX files """
    # create the output directory
    os.makedirs(path_folder_matrix_output_mex_format, exist_ok=True)

    # save barcodes
    pd.DataFrame(arr_bc).to_csv(
        f"{path_folder_matrix_output_mex_format}barcodes.tsv.gz",
        sep="\t",
        index=False,
        header=False,
    )

    # save features
    # compose a feature dataframe
    df_feature[["id_feature", "feature", "feature_type"]].to_csv(
        f"{path_folder_matrix_output_mex_format}features.tsv.gz",
        sep="\t",
        index=False,
        header=False,
    )  # save as a file
    # retrieve list of features

    # save count matrix as a gzipped matrix market format
    row, col, data = df_mtx[["id_row", "id_column", "read_count"]].values.T
    sm = scipy.sparse.coo_matrix(
        (data, (row, col)), shape=(len(df_feature), len(arr_bc))
    )
    scipy.io.mmwrite(f"{path_folder_matrix_output_mex_format}matrix", sm)

    # remove previous output file to overwrite the file
    path_file_mtx_output = f"{path_folder_matrix_output_mex_format}matrix.mtx.gz"
    if os.path.exists(path_file_mtx_output):
        os.remove(path_file_mtx_output)
    bk.OS_Run(
        ["gzip", f"{path_folder_matrix_output_mex_format}matrix.mtx"]
    )  # gzip the mtx file


"""
classes and functions for file system operations usimg managers
"""
from multiprocessing.managers import BaseManager
import zarr


class ManagerFileSystem(BaseManager):
    pass


class HostedFileSystemOperator:
    """# 2023-01-08 23:00:20
    A class intended for performing asynchronous file system operations in a separate, managed process. By using multiple managers, concurrent, asynchronous operations can be performed in multiple processes. These managers can be used multiple times.

    dict_kwargs_s3 : dict = dict( ) # s3 credentials to use
    """

    # constructor
    def __init__(self, dict_kwargs_s3: dict = dict()):
        import s3fs
        import zarr

        # save the settings
        self._dict_kwargs_s3 = dict_kwargs_s3

        # open async/sync version of s3fs
        self._as3 = s3fs.S3FileSystem(asynchronous=True, **dict_kwargs_s3)
        self._s3 = s3fs.S3FileSystem(**dict_kwargs_s3)
        print(self._s3)

    def exists(self, path_src: str, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        """
        return self._s3.exists(path_src, **kwargs)

    def rm(self, path_src: str, flag_recursive: bool = True, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        """
        return self._s3.rm(path_src, recursive=flag_recursive, **kwargs)  # delete files

    def glob(self, path_src: str, flag_recursive: bool = True, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        """
        return list(
            "s3://" + e for e in self._s3.glob(path_src, **kwargs)
        )  # 's3://' prefix should be added

    def mkdir(self, path_src: str, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        """
        # use default 'exist_ok' value
        if "exist_ok" not in kwargs:
            kwargs["exist_ok"] = True
        return self._s3.makedirs(path_src, **kwargs)

    def mv(self, path_src: str, path_dest: str, flag_recursive: bool = True, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        """
        if not self._s3.exists(
            path_dest, **kwargs
        ):  # avoid overwriting of the existing file
            return self._s3.mv(path_src, path_dest, recursive=flag_recursive, **kwargs)
        else:
            return "destionation file already exists, exiting"

    def cp(self, path_src: str, path_dest: str, flag_recursive: bool = True, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        """
        if is_s3_url(path_src) and is_s3_url(path_dest):  # copy from s3 to s3
            return self._s3.copy(
                path_src, path_dest, recursive=flag_recursive, **kwargs
            )
        elif is_s3_url(path_src):  # copy from s3 to local
            return self._s3.get(path_src, path_dest, recursive=flag_recursive, **kwargs)
        elif is_s3_url(path_dest):  # copy from local to s3
            return self._s3.put(path_src, path_dest, recursive=flag_recursive, **kwargs)

    def isdir(self, path_src: str, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        """
        return self._s3.isdir(path_src)

    def get_zarr_metadata(self, path_src: str, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        ❤️ test
        """
        return dict(zarr.open(path_src).attrs)

    def say_hello(self, path_src: str, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        ❤️ test
        """
        return "hello"


# register the manager
ManagerFileSystem.register("HostedFileSystemOperator", HostedFileSystemOperator)
