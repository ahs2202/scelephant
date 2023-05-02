from biobookshelf.main import *
from biobookshelf import *
import biobookshelf as bk


def SemiSimulate_scData(
    path_folder_mtx_10x_input,
    path_folder_mtx_10x_output,
    float_prop_blur=0.5,
    float_ratio_of_standard_deviation_to_expression_count=0.3,
    int_num_copies=30,
    int_num_threads=30,
):
    """# 2022-07-16 00:44:53
    semi-simulate single-cell count/expression data in a very memory-efficient manner

    float_prop_blur = 0.5 : a proportion of reads that will be 'perturbed' to generate semi-simulated single-cell count data
    float_ratio_of_standard_deviation_to_expression_count = 0.3 : the larger this ratio is, the more variable from the simulated count data from the original count data
    int_num_copies : number of simulated copies of the original dataset in the output simulated single-cell dataset
    int_num_threads : number of threads, including the main process distributing and collecting results
    """
    # define temp folder
    path_folder_temp = f"{path_folder_mtx_10x_output}temp_{UUID( )}/"
    os.makedirs(path_folder_temp, exist_ok=True)

    # define matrix file pathes
    path_file_mtx, path_file_feature, path_file_barcode = (
        f"{path_folder_mtx_10x_input}matrix.mtx.gz",
        f"{path_folder_mtx_10x_input}features.tsv.gz",
        f"{path_folder_mtx_10x_input}barcodes.tsv.gz",
    )

    # retrieve matrix metadata
    (
        int_num_features,
        int_num_barcodes,
        int_num_records,
    ) = bk.SC.MTX_10X_Retrieve_number_of_rows_columns_and_entries(
        path_folder_mtx_10x_input
    )

    """
    generate semi-simulated cells
    """

    def process_batch(index_batch, pipe_sender):
        """process batch"""
        # write barcode file with a batch-specific suffix
        bk.SC.MTX_10X_Barcode_add_prefix_or_suffix(
            path_file_barcode,
            f"{path_folder_temp}barcodes.{index_batch}.tsv.gz",
            barcode_suffix=f"|sim_{index_batch}",
        )  # add a batch-specific suffix

        # retrieve int_total_num_cells_of_previous_batch for renumbering
        int_total_num_cells_of_previous_batch = index_batch * int_num_barcodes

        # change coordinates of barcodes and blur count data
        newfile = gzip.open(
            f"{path_folder_temp}matrix.{index_batch}.mtx.gz", "wb"
        )  # open an output matrix file
        with gzip.open(path_file_mtx, "rb") as file:
            line = file.readline().decode()  # read the first line
            # if the first line of the file contains a comment line, read all comment lines and a description line following the comments.
            if line[0] == "%":
                # read comment and the description line
                while True:
                    if line[0] != "%":
                        break
                    line = file.readline().decode()  # read the next line
                line = (
                    file.readline().decode()
                )  # discard the description line and read the next line
            # process entries
            while len(line) > 0:
                index_row, index_col, val = tuple(
                    map(int, line.strip().split())
                )  # parse each entry of the current matrix
                if (
                    np.random.random() < float_prop_blur
                ):  # create random variation using the given probability
                    val = max(
                        int(
                            val
                            * (
                                1
                                + np.random.normal()
                                * float_ratio_of_standard_deviation_to_expression_count
                            )
                        ),
                        1,
                    )  # min count is 1 (creating random variation should not delete/create feature/cells)
                newfile.write(
                    (
                        " ".join(
                            tuple(
                                map(
                                    str,
                                    [
                                        index_row,
                                        index_col
                                        + int_total_num_cells_of_previous_batch,
                                        val,
                                    ],
                                )
                            )
                        )
                        + "\n"
                    ).encode()
                )  # renumber barcode index
                line = file.readline().decode()  # read the next line
        newfile.close()

        # notify that generation of simulation data has been completed
        pipe_sender.send(index_batch)

    bk.Multiprocessing_Batch(
        np.arange(int_num_copies).__iter__(),
        process_batch,
        int_num_threads=int_num_threads,
    )  # use 'int_num_copies' number of workers

    """
    combine output matrix files
    """

    def combine_file(path_folder, name_file, name_ext, flag_run_in_bash_shell=True):
        """# 2022-07-11 18:59:44
        combine files in order

        path_folder, : folder where files reside
        name_file, : name of files
        name_ext : name of extension
        'flag_run_in_bash_shell' : run concatenation command in bash
        """
        df = GLOB_Retrive_Strings_in_Wildcards(f"{path_folder}{name_file}.*.{name_ext}")
        df.wildcard_0 = df.wildcard_0.astype(int)
        df.sort_values("wildcard_0", inplace=True)
        if flag_run_in_bash_shell:
            os.system(
                f"cat {' '.join( list( df.path.values ) )} > {path_folder_temp}{name_file}.{name_ext}"
            )  # combine file using bash shell
        else:
            OS_Run(
                ["cat"] + list(df.path.values),
                path_file_stdout=f"{path_folder_temp}{name_file}.{name_ext}",
                stdout_binary=True,
            )  # somehow using python pipe to save concatenated file takes a lot of memory.. therefore, 'flag_run_in_bash_shell' should be used

    # write MatrixMarket format header lines
    with gzip.open(f"{path_folder_temp}matrix.-1.mtx.gz", "wb") as newfile:
        newfile.write(
            f"%%MatrixMarket matrix coordinate integer general\n%\n{int_num_features} {int_num_barcodes * int_num_copies} {int_num_records * int_num_copies}\n".encode()
        )

    combine_file(path_folder_temp, "barcodes", "tsv.gz")  # combine barcode files
    combine_file(path_folder_temp, "matrix", "mtx.gz")  # combine matrix files

    """
    prepare output files
    """
    os.rename(
        f"{path_folder_temp}barcodes.tsv.gz",
        f"{path_folder_mtx_10x_output}barcodes.tsv.gz",
    )
    os.rename(
        f"{path_folder_temp}matrix.mtx.gz", f"{path_folder_mtx_10x_output}matrix.mtx.gz"
    )
    shutil.copyfile(path_file_feature, f"{path_folder_mtx_10x_output}features.tsv.gz")

    # remove temp folder
    shutil.rmtree(path_folder_temp)
