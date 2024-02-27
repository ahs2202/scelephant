from typing import Union, List, Literal, Dict
import numpy as np
import pandas as pd
import os
from io import StringIO  # for converting a string to a file-like stream
import time
import uuid
import glob
import datetime
import pickle
import matplotlib.pyplot as plt
from uuid import uuid4
import gzip
import subprocess
import shutil
import csv
from multiprocessing import (
    Pool,
    get_context,
    set_start_method,
)  # for multiple processing  # with get_context("spawn").Pool() as pool:
import multiprocessing
import multiprocessing as mp
import collections
from copy import copy, deepcopy


def Wide(int_percent_html_code_cell_width=95):
    """
    # 20210224
    widen jupyter notebook cell
    """
    from IPython.core.display import display, HTML

    display(HTML("<style>.container { width:100% !important; }</style>"))
    display(
        HTML(
            "<style>div.cell{width:"
            + str(int_percent_html_code_cell_width)
            + "%;margin-left:"
            + str(100 - int_percent_html_code_cell_width)
            + "%;margin-right:auto;}</style>"
        )
    )


def PICKLE_Write(path_file, data_object):
    """write binary pickle file of a given data_object"""
    with open(path_file, "wb") as handle:
        pickle.dump(data_object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def PICKLE_Read(path_file):
    """write binary pickle file of a given data_object"""
    with open(path_file, "rb") as handle:
        data_object = pickle.load(handle)
    return data_object


def TIME_GET_timestamp(flag_human_readable=False):
    """Get timestamp of current time in "%Y%m%d_%H%M" format"""
    cur_time = datetime.datetime.now()  # retrieve current time
    return (
        cur_time.strftime("%Y/%m/%d %H:%M")
        if flag_human_readable
        else cur_time.strftime("%Y%m%d_%H%M")
    )


def To_window_path_compatible_str(a_string):
    """
        replace following characters to '_' so that a given string will be compatible for Window file system :
    : (colon)    " (double quote)    / (forward slash)    \ (backslash)    | (vertical bar or pipe)    ? (question mark)    * (asterisk)
        Also, replace new line character into '_'
    """
    return (
        a_string.replace("\n", "_")
        .replace(":", "_")
        .replace('"', "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("|", "_")
        .replace("?", "_")
        .replace("*", "_")
    )


def Search_list_of_strings_with_multiple_query(
    l_str,
    *l_query,
    flag_ignore_case: bool = True,
    return_mask=False,
    return_position=False,
):
    """# 2023-04-28 01:04:38
    Search list of strings with multiple query. for negative query, add '-' in front of the query

    flag_ignore_case : bool = True # ignore cases by default
    """

    arr_mask_matched = np.ones(len(l_str), dtype=bool)
    l_str_input = l_str  # save the reference to the original object
    if flag_ignore_case:
        l_str = deepcopy(l_str)  # copy l_str 'before' converting to lower cases
        for i in range(len(l_str)):
            l_str[i] = l_str[i].lower()
    for query in l_query:
        bool_query_positive = True
        if query[0] == "-":
            bool_query_positive, query = False, query[1:]
        if flag_ignore_case:
            query = query.lower()  # convert the query to lower case
        l_mask_matched_for_a_query = (
            list(True if query in entry else False for entry in l_str)
            if bool_query_positive
            else list(False if query in entry else True for entry in l_str)
        )
        arr_mask_matched = arr_mask_matched & np.array(
            l_mask_matched_for_a_query, dtype="bool"
        )
    if return_position:
        return np.where(arr_mask_matched)[0]
    return (
        arr_mask_matched
        if return_mask
        else np.array(l_str_input, dtype=object)[arr_mask_matched]
    )  # return a subset of the list of input strings


def MATPLOTLIB_savefig(title, dpi=200, folder=None, close_fig=True, format=".png"):
    if "." not in format:
        format = "." + format
    plt.savefig(
        folder + To_window_path_compatible_str(title) + format,
        dpi=200,
        bbox_inches="tight",
    )  # prevent x or y labels from cutting off
    if close_fig:
        plt.close()


def MATPLOTLIB_basic_configuration(
    font_size=None,
    font_size_axes_title=None,
    font_size_axes_label=None,
    font_size_xtick=None,
    font_size_ytick=None,
    font_size_legend=None,
    font_size_figure_title=None,
    x_label=None,
    y_label=None,
    title=None,
    x_scale=None,
    y_scale=None,
    show_grid=True,
    show_legend=False,
    savefig=False,
    y_lim=None,
    x_lim=None,
    save_file_name=None,
    folder=None,
    format=".png",
    show_colorbar=False,
):
    """A basic function for confiquring a matplotlib plot"""
    # set font sizes
    if font_size is not None:
        plt.rc("font", size=20)  # controls default text sizes
    if font_size_axes_title is not None:
        plt.rc("axes", titlesize=20)  # fontsize of the axes title
    if font_size_axes_label is not None:
        plt.rc("axes", labelsize=20)  # fontsize of the x and y labels
    if font_size_xtick is not None:
        plt.rc("xtick", labelsize=20)  # fontsize of the x tick labels
    if font_size_ytick is not None:
        plt.rc("ytick", labelsize=20)  # fontsize of the y tick labels
    if font_size_legend is not None:
        plt.rc("legend", fontsize=20)  # legend fontsize
    if font_size_figure_title is not None:
        plt.rc("figure", titlesize=50)  # fontsize of the figure title
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    if x_scale is not None:
        plt.xscale(x_scale)
    if y_scale is not None:
        plt.yscale(y_scale)
    if x_lim is not None:
        if isinstance(x_lim, (tuple, list)):
            plt.xlim(left=x_lim[0], right=x_lim[1])
        elif isinstance(x_lim, dict):
            plt.xlim(**x_lim)
    if y_lim is not None:
        if isinstance(y_lim, (tuple, list)):
            plt.ylim(bottom=y_lim[0], top=y_lim[1])
        elif isinstance(y_lim, dict):
            plt.ylim(**y_lim)
    plt.grid(show_grid)
    if show_legend:
        plt.legend()
    if savefig:
        if save_file_name is None:  # if 'save_file_name' is not given
            if title is None:
                title = (
                    "Unnamed Plot_" + TIME_GET_timestamp()
                )  # if title is not given, put a default title to save a plot
            MATPLOTLIB_savefig(title=title, folder=folder, format=format)
        else:
            MATPLOTLIB_savefig(title=save_file_name, folder=folder, format=format)
    if show_colorbar:
        plt.colorbar()


MPL_basic_configuration = MATPLOTLIB_basic_configuration


class Map(object):
    def __init__(self, dict_a2b):
        self.dict_a2b = dict_a2b

    def a2b(self, a):
        if a in self.dict_a2b:
            return self.dict_a2b[a]
        else:
            return np.nan

    def a2b_if_mapping_available_else_Map_a2a(self, a):
        if a in self.dict_a2b:
            return self.dict_a2b[a]
        else:
            return a


def DICTIONARY_Build_from_arr(arr, order_index_entry=True, index_start=0):
    if order_index_entry:
        return dict(
            (index, entry)
            for entry, index in zip(arr, np.arange(index_start, len(arr) + index_start))
        )
    else:
        return dict(
            (entry, index)
            for entry, index in zip(arr, np.arange(index_start, len(arr) + index_start))
        )


def UUID():
    """return a 128bit universal unique identifier"""
    return uuid4().hex


def OS_Run(
    l_args,
    path_file_stdout=None,
    path_file_stderr=None,
    return_output=True,
    remove_default_output_files=True,
    stdout_binary=False,
):
    """# 2021-03-30 19:41:16
    Run a process and save stdout and stderr as a file.

    'return_output' : return the output as dictionary of strings
    'remove_default_output_files' : remove default stdout and stderr files containing the output of the process when 'path_file_stdout' and 'path_file_stderr' were not given.
    'stdout_binary' : set this flag to True if stdout is binary.
    """
    from subprocess import call as subprocess_call

    uuid_process = UUID()  # set uuid of the process
    # define default stdout and stdin files and set approproate flags
    flag_path_file_stdout_was_given = path_file_stdout is not None
    flag_path_file_stderr_was_given = path_file_stderr is not None

    # default stdout/stderr files will be written to the current working directory
    path_cwd = os.getcwd()
    if not flag_path_file_stdout_was_given:
        path_file_stdout = f"{path_cwd}/{uuid_process}.out.txt"
    if not flag_path_file_stderr_was_given:
        path_file_stderr = f"{path_cwd}/{uuid_process}.err.txt"

    with open(
        path_file_stdout, "w+b" if stdout_binary else "w+"
    ) as fout:  # excute and read std output and std errors of a process
        with open(path_file_stderr, "w+") as ferr:
            out = subprocess_call(l_args, stdout=fout, stderr=ferr)
            fout.seek(0)
            stdout = fout.read()
            ferr.seek(0)
            stderr = ferr.read()
    # remove default output files
    if not flag_path_file_stdout_was_given:
        os.remove(path_file_stdout)
    if not flag_path_file_stderr_was_given:
        os.remove(path_file_stderr)

    return {"stdout": stdout, "stderr": stderr} if return_output else None


def LIST_Split(
    l=None,
    n_split=0,
    return_slice=False,
    flag_contiguous_chunk=False,
    arr_weight_for_load_balancing=None,
    return_split_arr_weight=False,
):
    """# 2022-05-26 10:14:31
    split a list into 'n_split' number of chunks. if 'return_slice' is True, return slice() instances instead of actually spliting the given list-like object.
    performs load balancing based on given list of weights (the 'arr_weight_for_load_balancing' argument)

    'flag_contiguous_chunk' : split the list in a continguous manner so that each chunk contains a region of a original list
    'arr_weight_for_load_balancing' : only valid when 'flag_contiguous_chunk' is True. 'arr_weight_for_load_balancing' should contains the list of weights for each element for load balancing
    'return_split_arr_weight' : return split arr_weights, too (only valid if 'flag_contiguous_chunk' == True and valid arr_weight is given through the 'arr_weight_for_load_balancing' element)
    """
    # retrieve slice
    if flag_contiguous_chunk:
        if (
            arr_weight_for_load_balancing is None
        ):  # process equal number of entries for each chunk
            int_num_entries_for_each_chunk = int(np.ceil(len(l) / n_split))
            l_slice = list(
                slice(
                    index_split * int_num_entries_for_each_chunk,
                    (index_split + 1) * int_num_entries_for_each_chunk,
                )
                for index_split in np.arange(n_split)
            )
        else:  # if an array of weights are given, use the weights to balance the load for each chunk
            # convert dtype of the array to increase the resolution and prevent error due to small resolution of np.float32 # 2022-05-26 10:07:10 by ahs2202
            arr_weight_for_load_balancing = np.array(
                arr_weight_for_load_balancing, dtype=np.float64
            )
            # calculate total weaight for each chunk
            int_total_weight = np.sum(arr_weight_for_load_balancing)
            int_total_weight_for_each_chunk = np.ceil(int_total_weight / n_split)

            # collect the start positions of each chunk
            index_chunk = 0
            l_index_start_of_chunk = [0]
            for index, accumulated_weight in enumerate(
                np.cumsum(arr_weight_for_load_balancing)
            ):
                if (
                    int_total_weight_for_each_chunk * (index_chunk + 1)
                    < accumulated_weight
                ):  # if the accumulated bytes is larger than the 'int_total_weight_for_each_chunk' times the number of chunk written, record a chunk boundary.
                    l_index_start_of_chunk.append(
                        index
                    )  # mark the current position as the start of the chunk (and thus the start of the next chunk)
                    index_chunk += 1  # update the index of the chunk
            if (
                len(l_index_start_of_chunk) > n_split
            ):  # when a possible overflow/errors from too low resolution was detected, correct the boundary
                l_index_start_of_chunk[n_split] = len(arr_weight_for_load_balancing)
                l_pos_start_chunk = l_index_start_of_chunk[: n_split + 1]
            else:
                l_pos_start_chunk = l_index_start_of_chunk + [
                    len(arr_weight_for_load_balancing)
                ]
            l_slice = list(
                slice(
                    l_pos_start_chunk[index_split], l_pos_start_chunk[index_split + 1]
                )
                for index_split in np.arange(n_split)
            )
    else:
        l_slice = list(
            slice(index_split, None, n_split) for index_split in np.arange(n_split)
        )
    if return_slice:
        return l_slice  # if 'return_slice' is True, return slice() instances instead of actually spliting the given list-like object
    else:
        if (
            flag_contiguous_chunk
            and arr_weight_for_load_balancing is not None
            and return_split_arr_weight
        ):  # return split input list and the weights
            return list(l[a_slice] for a_slice in l_slice), list(
                arr_weight_for_load_balancing[a_slice] for a_slice in l_slice
            )
        else:
            return list(l[a_slice] for a_slice in l_slice)


def PD_Select(df, deselect=False, **dict_select):
    """Select and filter rows of df according to the given dict_select. If 'deselect' is set to True, deselect rows according to the given dict_select  Usage example : PANDAS_Select( df_meta_imid_ubi, dict(  Data_Type = [ 'Proteome', 'Ubi_Profiling' ], Value_Type = 'log2fc' ) )"""
    for col, query in dict_select.items():
        if type(df) is pd.Series:
            data_values = (
                df.index.values if col == "index" else df.values
            )  # select values or indices of a given pd.Series
        elif type(df) is pd.DataFrame:
            if col not in df.columns.values and col != "index":
                print("'{}' does not exist in columns of a given DataFrame".format(col))
                continue
            data_values = df.index.values if col == "index" else df[col].values
        else:
            print("[INVALID INPUT]: Inputs should be DataFrame or Series")
            return -1
        if isinstance(
            query, (list, tuple, np.ndarray, set)
        ):  # if data to be selected is iterable
            query = (
                set(query) if isinstance(query, (list, tuple, np.ndarray)) else query
            )  # convert query into set
            df = (
                df[
                    list(
                        False if data_value in query else True
                        for data_value in data_values
                    )
                ]
                if deselect
                else df[
                    list(
                        True if data_value in query else False
                        for data_value in data_values
                    )
                ]
            )
        else:
            df = df[data_values != query] if deselect else df[data_values == query]
    return df


def PD_Threshold(df, AND_operation=True, **dict_thresholds):
    """Select rows of a given DataFrame or indices of Series based on a given threshold for each given column or the given series.
    Add 'b' or 'B' at the end of column_label to select rows below the threshold, or add 'a' or 'A' to select rows above the threshold.
    If 'AND_operation' is true, filter generated from the given threshold will be combined with AND operation before filtering rows of a given dataframe
    """
    set_df_columns = set(df.columns.values) if type(df) is pd.DataFrame else set([""])
    mask_filter = (
        np.ones(len(df), dtype=bool) if AND_operation else np.zeros(len(df), dtype=bool)
    )
    for col_direction, threshold in dict_thresholds.items():
        col, direction = col_direction[:-1], col_direction[-1]
        if col not in set_df_columns:
            print("'{}' column_label does not exist in the given DataFrame".format(col))
            continue
        data = df[col].values if type(df) is pd.DataFrame else df.values
        if direction.lower() == "a":
            current_mask = data > threshold
        elif direction.lower() == "b":
            current_mask = data < threshold
        else:
            print(
                "'{}' direction is not either 'a' or 'b' and thus invalid".format(
                    direction
                )
            )
            continue
        mask_filter = (
            current_mask & mask_filter if AND_operation else current_mask | mask_filter
        )
    return df[mask_filter]


# functions related to the command line interface
def Parse_Printed_Table(str_output):
    """# 2022-08-08 11:39:01
    Parse printed table by identifying the position of columns and inferring datatypes using pandas module
    """
    l_line = str_output.split("\n")
    # survey the number of ' ' space characters in each line
    int_max_characters_in_each_line = max(
        len(line) for line in l_line
    )  # retrieve the maximum number of characters a line contains
    arr_space_counter = np.zeros(int_max_characters_in_each_line, dtype=int)
    for line in l_line:
        # add padding containing ' ' to the end of the sentence for accurate detection of columns
        if len(line) < int_max_characters_in_each_line:
            line += " " * (int_max_characters_in_each_line - len(line))
        for i, c in enumerate(line):
            if c == " ":
                arr_space_counter[i] += 1

    arr_pos_col_space = np.where(arr_space_counter == max(arr_space_counter))[
        0
    ]  # retrieve positions of the columns of ' ' space characters

    arr_pos_of_columns_marking_the_boundary = (
        [-1, arr_pos_col_space[0]]
        + list(arr_pos_col_space[1:][np.diff(arr_pos_col_space) != 1])
        + [int_max_characters_in_each_line + 1]
    )  # retrieve positions of the columns marking the boundaries

    # collect values
    l_l = []
    for line in l_line:
        l = []
        for i in range(len(arr_pos_of_columns_marking_the_boundary) - 1):
            col_before, col_after = arr_pos_of_columns_marking_the_boundary[i : i + 2]
            l.append(line[col_before + 1 : col_after].strip())
        l_l.append("\t".join(l))
    df = pd.read_csv(StringIO("\n".join(l_l)), sep="\t")
    return df


def PIP_List_Packages():
    """# 2022-08-08 11:32:04
    list installed packages
    """
    return (
        Parse_Printed_Table(os.popen("pip list").read().strip())
        .drop(index=[0])
        .set_index("Package")
    )


# slice functions
def Slice_to_Range(sl, length):
    """# 2022-06-28 21:47:51
    iterate indices from the given slice
    """
    assert isinstance(sl, slice)  # make sure sl is slice
    # convert slice to integer indices
    for i in range(*sl.indices(length)):
        yield i


def COUNTER(l_values, dict_counter=None, ignore_float=True):  # 2020-07-29 23:49:51
    """Count values in l_values and return a dictionary containing count values. if 'dict_counter' is given, countinue counting by using the 'dict_counter'. if 'ignore_float' is True, ignore float values, including np.nan"""
    if dict_counter is None:
        dict_counter = dict()
    if ignore_float:  # if 'ignore_float' is True, ignore float values, including np.nan
        for value in l_values:
            if isinstance(value, float):
                continue  # ignore float values
            if value in dict_counter:
                dict_counter[value] += 1
            else:
                dict_counter[value] = 1
    else:  # faster counting by not checking type of value
        for value in l_values:
            if value in dict_counter:
                dict_counter[value] += 1
            else:
                dict_counter[value] = 1
    return dict_counter


def LIST_COUNT(
    iterable,
    return_series=True,
    duplicate_filter=2,
    dropna=True,
    sort_series_by_values=True,
    convert_tuple_to_string=False,
):
    """
    # 20210224
    return a dictionary where key = each element in a given list, value = counts of the element in the list. if 'duplicate_filter' is not None, return entries that are duplicated 'duplicate_filter' times or more.
    """
    if dropna and isinstance(iterable, pd.Series):
        iterable = (
            iterable.dropna()
        )  # if dropna is set to 'True', dropn NaN values before counting
    if isinstance(next(iterable.__iter__()), (np.ndarray, list)):
        iterable = list(
            map(tuple, iterable)
        )  # if value is non-hashable list of numpy array, convert a value to a hashable format, tuple
    dict_counted = COUNTER(iterable)
    if (
        convert_tuple_to_string
    ):  # if 'convert_tuple_to_string' is True and values in a given list are tuples, convert tuples into string
        dict_counted__tuple_converted_to_string = dict()
        for key in dict_counted:
            value = dict_counted[key]
            if isinstance(key, (tuple)):
                dict_counted__tuple_converted_to_string[
                    ("{}, " * len(key))[:-2].format(*key)
                ] = value  # convert tuple into string concatanated with ', '
            else:
                dict_counted__tuple_converted_to_string[key] = value
        dict_counted = dict_counted__tuple_converted_to_string
    if return_series:
        s_counted = pd.Series(dict_counted)
        if duplicate_filter is not None:
            s_counted = s_counted[s_counted >= duplicate_filter]
        if sort_series_by_values:
            s_counted = s_counted.sort_values(ascending=False)
        return s_counted
    else:
        return dict_counted


def DICTIONARY_Find_keys_with_max_value(dict_value):
    """# 2021-11-24 20:44:07
    find a list of key values with the maximum value in a given dictionary, and return 'l_key_max', 'value_max'
    """
    value_max = None  # initialize max value
    l_key_max = []  # list of key with max_values
    if len(dict_value) != 0:  # if the dictionary is not empty
        for key in dict_value:
            value = dict_value[key]
            if value_max is None:
                value_max = value
                l_key_max.append(key)
            elif value_max > value:
                continue
            elif value_max < value:
                l_key_max = [key]
                value_max = value
            elif (
                value_max == value
            ):  # if the another key contains the current max value, add the key to the list of keys with max values
                l_key_max.append(key)
    return l_key_max, value_max


def GLOB_Retrive_Strings_in_Wildcards(
    str_glob,
    l_path_match=None,
    return_dataframe=True,
    retrieve_file_size=False,
    retrieve_last_modified_time=False,
    time_offset_in_seconds=3600 * 9,
):  # 2020-11-16 18:20:52
    """# 2022-01-09 23:25:48
    retrieve strings in '*' wildcards in list of matched directories for the given string containing '*' wildcards. return strings in wildcards as a nested lists. Consecutive wildcards should not be used ('**' should not be used in the given string)
    'retrieve_file_size': if 'return_dataframe' is True, return file sizes in bytes by using os.stat( path_match ).st_size
    'retrieve_last_modified_time': return the last modified time with pandas datetime datatype
    'time_offset_in_seconds': offset in seconds to Coordinated Universal Time (UTC)"""
    l_path_match = (
        glob.glob(str_glob) if l_path_match is None else l_path_match
    )  # retrive matched directories using glob.glob if 'l_path_match' is not given
    l_intervening_str = str_glob.split(
        "*"
    )  # retrive intervening strings in a glob string containing '*' wildcards
    l_l_str_in_wildcard = list()
    for (
        path_match
    ) in l_path_match:  # retrive strings in wildcards for each matched directory
        path_match_subset = path_match.split(l_intervening_str[0], 1)[1]
        l_str_in_wildcard = list()
        for intervening_str in l_intervening_str[1:]:
            if len(intervening_str) > 0:
                str_in_wildcard, path_match_subset = path_match_subset.split(
                    intervening_str, 1
                )
            else:
                str_in_wildcard, path_match_subset = (
                    path_match_subset,
                    "",
                )  # for the wildcard at the end of the given string, put remaining string into 'str_in_wildcard' and empties 'path_match_subset'
            l_str_in_wildcard.append(str_in_wildcard)
        l_l_str_in_wildcard.append(l_str_in_wildcard)
    if (
        return_dataframe
    ):  # return dataframe containing strings in wildcards and matched directory
        df = pd.DataFrame(
            l_l_str_in_wildcard,
            columns=list(
                "wildcard_" + str(index) for index in range(str_glob.count("*"))
            ),
        )
        df["path"] = l_path_match
        if retrieve_file_size:
            df["size_in_bytes"] = list(
                os.stat(path_match).st_size for path_match in l_path_match
            )
            df["size_in_gigabytes"] = df["size_in_bytes"] / 2**30
        if retrieve_last_modified_time:
            df["time_last_modified"] = list(
                datetime.datetime.utcfromtimestamp(
                    os.path.getmtime(path_file) + time_offset_in_seconds
                ).strftime("%Y-%m-%d %H:%M:%S")
                for path_file in df.path.values
            )
            df.time_last_modified = pd.to_datetime(
                df.time_last_modified
            )  # convert to datetime datatype
        return df
    else:
        return l_l_str_in_wildcard


def Multiprocessing_Batch_Generator_and_Workers(
    gen_batch,
    process_batch,
    post_process_batch=None,
    int_num_threads: int = 15,
    int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop: float = 0.2,
    flag_wait_for_a_response_from_worker_after_sending_termination_signal: bool = True,  # wait until all worker exists before resuming works in the main process
):
    """# 2023-10-06 01:52:29
    'Multiprocessing_Batch_Generator_and_Workers' : multiprocessing using batch generator and workers.
    all worker process will be started using the default ('fork' in UNIX) method.
    perform batch-based multiprocessing using the three components, (1) gen_batch, (2) process_batch, (3) post_process_batch. (3) will be run in the main process, while (1) and (2) will be offloaded to worker processes.
    the 'batch' and result returned by 'process_batch' will be communicated to batch processing workers through pipes.

    'gen_batch' : a generator object returning batches
    'process_batch( pipe_receiver, pipe_sender )' : a function that can process batch from 'pipe_receiver'. should terminate itself when None is received. 'pipe_sender' argument is to deliver the result to the main process, and should be used at the end of code to notify the main process that the work has been completed. sending 'None' through 'pipe_sender' will terminate the block and the main process will be unblocked (however, the works will be continued to be distributed and performed by the child processes).
    'post_process_batch( result )' : a function that can process return value from 'process_batch' function in the main process. operations that are not thread/process-safe can be done here, as these works will be serialized in the main thread.
    'int_num_threads' : the number of threads(actually processes) including the main process. For example, when 'int_num_threads' is 3, 2 worker processes will be used. one thread is reserved for batch generation.
    'int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop' : number of seconds to wait for each loop before checking which running processes has been completed
    flag_wait_for_a_response_from_worker_after_sending_termination_signal : bool = True, # wait until all worker exists before resuming works in the main process
    """

    def __batch_generating_worker(
        gen_batch,
        l_pipe_sender_input,
        l_pipe_receiver_output,
        pipe_sender_output_to_main_process,
    ):
        """# 2023-10-06 01:52:24
        define a worker for generating batch and distributing batches across the workers, receives results across the workers, and send result back to the main process
        """
        # hard coded setting
        int_max_num_batches_in_a_queue_for_each_worker = 2  # 2 batches distributed to each process should be optimal, while preventing pipe buffer overloading.

        q_batch = collections.deque()  # initialize queue of batchs
        int_num_batch_processing_workers = len(l_pipe_sender_input)
        flag_batch_generation_completed = False  # flag indicating whether generating batchs for the current input sam file was completed
        arr_num_batch_being_processed = np.zeros(
            int_num_batch_processing_workers, dtype=int
        )  # retrieve the number of batches currently being processed in each worker. if this number becomes 0, assumes the worker is available
        while True:
            """retrieve batch (if available)"""
            if (
                not flag_batch_generation_completed
                and len(q_batch)
                < int_num_batch_processing_workers
                * int_max_num_batches_in_a_queue_for_each_worker
            ):  # if batch generation has not been completed, or the number of batches that have been generated are not large, continue to generate batches.
                try:
                    batch = next(gen_batch)  # retrieve the next barcode
                    q_batch.appendleft(batch)  # append batch
                except StopIteration:
                    flag_batch_generation_completed = True
            else:
                # if batch generation has been completed, all batches have been distributed, and processed, exit the loop
                if (
                    flag_batch_generation_completed
                    and len(q_batch) == 0
                    and arr_num_batch_being_processed.sum() == 0
                ):
                    break
                # sleep for a while
                time.sleep(
                    int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop
                )  # sleep

            """ collect completed works """
            for index_worker in range(int_num_batch_processing_workers):
                while l_pipe_receiver_output[
                    index_worker
                ].poll():  # until results are available
                    pipe_sender_output_to_main_process.send(
                        l_pipe_receiver_output[index_worker].recv()
                    )  # retrieve result, and send the result back to the main process
                    arr_num_batch_being_processed[
                        index_worker
                    ] -= 1  # update the number of batches being processed by the worker

            """ if workers are available and there are remaining works to be distributed, distribute works """
            index_worker = 0  # initialize the index of the worker
            while (
                len(q_batch) > 0
                and (
                    arr_num_batch_being_processed
                    < int_max_num_batches_in_a_queue_for_each_worker
                ).sum()
                > 0
            ):  # if there is remaining batch to be distributed or at least at least one worker should be available
                if (
                    arr_num_batch_being_processed[index_worker]
                    <= arr_num_batch_being_processed.mean()
                    and arr_num_batch_being_processed[index_worker]
                    < int_max_num_batches_in_a_queue_for_each_worker
                ):  # the load of the current worker should be below the threshold # if the load for the current worker is below the pool average, assign the work to the process (load-balancing)
                    l_pipe_sender_input[index_worker].send(q_batch.pop())
                    arr_num_batch_being_processed[index_worker] += 1
                index_worker = (
                    1 + index_worker
                ) % int_num_batch_processing_workers  # retrieve index_worker of the next worker

        # notify batch-processing workers that all workers are completed
        for pipe_s in l_pipe_sender_input:
            pipe_s.send(None)
        # notify the main process that all batches have been processed
        pipe_sender_output_to_main_process.send(None)
        return

    int_num_batch_processing_workers = max(
        1, int_num_threads - 2
    )  # retrieve the number of workers for processing batches # minimum number of worker is 1
    # compose pipes
    l_pipes_input = list(mp.Pipe() for i in range(int_num_batch_processing_workers))
    l_pipes_output = list(mp.Pipe() for i in range(int_num_batch_processing_workers))
    pipe_sender_output_to_main_process, pipe_receiver_output_to_main_process = mp.Pipe()
    # compose workers
    l_batch_processing_workers = list(
        mp.Process(
            target=process_batch, args=(l_pipes_input[i][1], l_pipes_output[i][0])
        )
        for i in range(int_num_batch_processing_workers)
    )  # compose a list of batch processing workers
    p_batch_generating_worker = mp.Process(
        target=__batch_generating_worker,
        args=(
            gen_batch,
            list(s for s, r in l_pipes_input),
            list(r for s, r in l_pipes_output),
            pipe_sender_output_to_main_process,
        ),
    )
    # start workers
    for p in l_batch_processing_workers:
        p.start()
    p_batch_generating_worker.start()

    # post-process batches
    while True:
        res = pipe_receiver_output_to_main_process.recv()
        if res is None:
            break
        if post_process_batch is not None:
            post_process_batch(
                res
            )  # process the result returned by the 'process_batch' function in the 'MAIN PROCESS', serializing potentially not thread/process-safe operations in the main thread.

    # if 'flag_wait_for_a_response_from_worker_after_sending_termination_signal' is True, wait until a response is received from the worker
    if flag_wait_for_a_response_from_worker_after_sending_termination_signal:
        for s, r in l_pipes_output:  # pipe receiving responses from batch workers
            r.recv()


def Workers(
    func,
    *args,
    int_num_workers_for_Workers: int = 8,
    **kwargs,
):
    """
    func, # a function to run across workers
    * args, # arguments to the function
    int_num_workers_for_Workers : int = 8, # number of workers for the current function ('Workers')
    ** kwargs, # keyworded arguments to the function

    For the arguments or keyworded arguments that are unique to individual runs, Python List should be used to list the arguments for each run. If the original argument is Python List, the argument should be wrapped inside another Python List, repeated N times where N is the total number of runs.

    For example, the following usage is possible,
    >>> def func( d, e, a = 0, b = 1, c = 3 ) :
    >>>     print( f"{d + e + a + b + c = }" )
    >>> Workers( func, [ 10, 20, 30, 40, 50 ], 30, int_num_workers_for_Workers = 8, a = [ 1, 2, 3, 4, 5, 6, 7, 8 ], b = 10, c = [ 7, 8, 9, 10, 11, 12, 13 ] )
    d + e + a + b + c = 58d + e + a + b + c = 70d + e + a + b + c = 94d + e + a + b + c = 106d + e + a + b + c = 82

    # 2024-01-03 17:33:49
    """

    def _gen_arguments(args_workers, kwargs_workers):
        """
        generate arguments with given args and kwargs
        # 2024-01-03 16:23:42
        """
        """ survey the number of independent runs """
        l_int_num_runs = []
        for arg_workers in args_workers:  # survey 'args_workers'
            if isinstance(arg_workers, list):
                l_int_num_runs.append(len(arg_workers))
        for k in kwargs_workers:  # survey 'kwargs_workers'
            kwarg_workers = kwargs_workers[k]  # retrieve the argument
            if isinstance(kwarg_workers, list):
                l_int_num_runs.append(len(kwarg_workers))
        int_num_runs = np.min(
            l_int_num_runs
        )  # determin the number of runs (minimum of the length of arguments)

        """ generate arguments for individual runs """
        for idx_run in range(int_num_runs):  # for 'idx_run'
            """compose arguments for a run"""
            args_run, kwargs_run = [], dict()
            for arg_workers in args_workers:  # compose 'args_run'
                args_run.append(
                    arg_workers[idx_run]
                    if isinstance(arg_workers, list)
                    else arg_workers
                )

            for k in list(kwargs_workers):  # modify 'kwargs_workers'
                kwarg_workers = kwargs_workers[k]  # retrieve the argument
                kwargs_run[k] = (
                    kwarg_workers[idx_run]
                    if isinstance(kwarg_workers, list)
                    else kwarg_workers
                )
            yield args_run, kwargs_run  # yield generated arguments and key-worded arguments

    def _run_func(p_i, p_o):
        """
        run the given function with given arguments
        # 2024-01-03 16:23:42
        """
        while True:
            ins = p_i.recv()
            if ins is None:
                break
            args, kwargs = ins  # parse the input

            # run function
            func(
                *args,
                **kwargs,
            )
            p_o.send("completed")  # notify the work has been completed
        p_o.send("completed")  # notify all works have been completed

    Multiprocessing_Batch_Generator_and_Workers(
        gen_batch=_gen_arguments(args, kwargs),  # generate arguments
        process_batch=_run_func,
        int_num_threads=int_num_workers_for_Workers + 2,
        flag_wait_for_a_response_from_worker_after_sending_termination_signal=True,
    )


def Multiprocessing(
    arr,
    Function,
    n_threads=12,
    path_temp="/tmp/",
    Function_PostProcessing=None,
    global_arguments=[],
    col_split_load=None,
):
    """# 2022-02-23 10:55:34
    split a given iterable (array, dataframe) containing inputs for a large number of jobs given by 'arr' into 'n_threads' number of temporary files, and folks 'n_threads' number of processes running a function given by 'Function' by givning a directory of each temporary file as an argument. if arr is DataFrame, the temporary file will be split DataFrame (tsv format) with column names, and if arr is 1d or 2d array, the temporary file will be tsv file without header
    By default, given inputs will be randomly distributed into multiple files. In order to prevent separation of a set of inputs sharing common input variable(s), use 'col_split_load' to group such inputs together.

    'Function_PostProcessing' : if given, Run the function before removing temporary files at the given temp folder. uuid of the current session and directory of the temporary folder are given as arguments to the function.
    'global_arguments' : a sort of environment variables (read only) given to each process as a list of additional arguments in addition to the directory of the input file. should be used to use local variables inside main( ) function if this function is called inside the main( ) function.
                         'global_arguments' will be passed to 'Function_PostProcessing', too.
    'col_split_load' : a name of column or a list of column names (or integer index of column or list of integer indices of columns if 'arr' is not a dataframe) for grouping given inputs when spliting the inputs into 'n_threads' number of dataframes. Each unique tuple in the column(s) will be present in only one of split dataframes.
    'n_threads' : if 'n_threads' is 1, does not use multiprocessing module, but simply run the function with the given input. This behavior is for enabiling using functions running Multiprocessing in another function using Multiprocessing, since multiprocessing.Pool module does not allow nested pooling.
    """
    from multiprocessing import Pool

    if isinstance(
        arr, (list)
    ):  # if a list is given, convert the list into a numpy array
        arr = np.array(arr)
    str_uuid = UUID()  # create a identifier for making temporary files
    l_path_file = (
        list()
    )  # split inputs given by 'arr' into 'n_threads' number of temporary files
    if (
        col_split_load is not None
    ):  # (only valid when 'arr' is dataframe) a name of column for spliting a given dataframe into 'n_threads' number of dataframes. Each unique value in the column will be present in only one split dataframe.
        if isinstance(
            arr, pd.DataFrame
        ):  # if arr is DataFrame, the temporary file will be split DataFrame (tsv format) with column names
            if isinstance(
                col_split_load, (str)
            ):  # if only single column name is given, put it in a list
                col_split_load = [col_split_load]

            # randomly distribute distinct tuples into 'n_threads' number of lists
            dict_index = DF_Build_Index_Using_Dictionary(
                arr, col_split_load
            )  # retrieve index according to the tuple contained by 'col_split_load'
            l_t = list(dict_index)
            n_t = len(l_t)  # retrieve number of tuples
            if (
                n_t < n_threads
            ):  # if the given number of thread is larger than the existing number of tuples, set the number of tuples as the number of threads
                n_threads = n_t
            np.random.shuffle(l_t)
            l_l_t = list(l_t[i::n_threads] for i in range(n_threads))

            arr_df = arr.values
            l_col = arr.columns.values

            for index_chunk in range(n_threads):
                l_t_for_the_chunk = l_l_t[index_chunk]
                # retrieve integer indices of the original array for composing array of the curreht chunk
                l_index = []
                for t in l_t_for_the_chunk:
                    l_index.extend(dict_index[t])
                path_file_temp = (
                    path_temp + str_uuid + "_" + str(index_chunk) + ".tsv.gz"
                )
                pd.DataFrame(arr_df[np.sort(l_index)], columns=l_col).to_csv(
                    path_file_temp, sep="\t", index=False
                )  # split a given dataframe containing inputs with groupping with a given list of 'col_split_load' columns
                l_path_file.append(path_file_temp)
        else:
            print(
                "'col_split_load' option is only valid when the given 'arr' is dataframe, exiting"
            )
            return -1
    else:
        # if number of entries is larger than the number of threads, reduce the n_threads
        if len(arr) < n_threads:
            n_threads = len(arr)
        if isinstance(
            arr, pd.DataFrame
        ):  # if arr is DataFrame, the temporary file will be split DataFrame (tsv format) with column names
            for index_chunk in range(n_threads):
                path_file_temp = (
                    path_temp + str_uuid + "_" + str(index_chunk) + ".tsv.gz"
                )
                arr.iloc[index_chunk::n_threads].to_csv(
                    path_file_temp, sep="\t", index=False
                )
                l_path_file.append(path_file_temp)
        else:  # if arr is 1d or 2d array, the temporary file will be tsv file without header
            l_chunk = LIST_Split(arr, n_threads)
            for index, arr in enumerate(
                l_chunk
            ):  # save temporary files containing inputs
                path_file_temp = path_temp + str_uuid + "_" + str(index) + ".tsv"
                if len(arr.shape) == 1:
                    df = pd.DataFrame(arr.reshape(arr.shape[0], 1))
                elif len(arr.shape) == 2:
                    df = pd.DataFrame(arr)
                else:
                    print("invalid inputs: input array should be 1D or 2D")
                    return -1
                df.to_csv(path_file_temp, sep="\t", header=None, index=False)
                l_path_file.append(path_file_temp)

    if n_threads > 1:
        with Pool(n_threads) as p:
            l = p.starmap(
                Function,
                list([path_file] + list(global_arguments) for path_file in l_path_file),
            )  # use multiple process to run the given function
    else:
        """if n_threads == 1, does not use multiprocessing module"""
        l = [Function(l_path_file[0], *list(global_arguments))]

    if Function_PostProcessing is not None:
        Function_PostProcessing(str_uuid, path_temp, *global_arguments)

    for path_file in glob.glob(path_temp + str_uuid + "*"):
        os.remove(path_file)  # remove temporary files

    return l  # return mapped results
