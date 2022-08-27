# core
# from memory_profiler import profile # for testing

import typing # import typing
from typing import Union
from biobookshelf.main import *
from biobookshelf import *
import biobookshelf as bk
pd.options.mode.chained_assignment = None  # default='warn' # to disable worining
import math
import zarr # SCElephant is currently implemented using Zarr
import numcodecs
import anndata
import scanpy
import shelve # for persistent database (key-value based database)
import sklearn as skl

import tarfile # tar.gz
import requests # download from url 
import shutil

from numba import jit # for speed up

from tqdm import tqdm as progress_bar # for progress bar

# dimension reduction / clustering
import umap.parametric_umap as pumap # parametric UMAP
import hdbscan # for clustering
import leidenalg

# for fast gzip compression/decompression
import pgzip # - deprecated

# might not be used
import scipy.spatial.distance as dist # for distance metrics calculation
import sklearn.cluster as skc # K-means

# for fast approximate kNN search
import pynndescent 

# for leiden clustering
import igraph as ig

# define version
_version_ = '0.0.5'
_scelephant_version_ = _version_
_last_modified_time_ = '2022-08-27 10:26:26'

""" # 2022-07-21 10:35:42  realease note

HTTP hosted RamData subclustering tested and verified.
identified problems: Zarr HTTP Store is not thread-safe (crashes the program on run time) nor process-safe (dead lock... why?).
Therefore, serializing access to Zarr HTTP Store is required 

TODO: do not use fork when store is Zarr HTTP Store
TODO: use Pynndescent for efficient HDBSCAN cluster label assignment, even when program is single-threaded (when Zarr HTTP store is used)
TODO: use Pynndescent for better subsampling of cells before UMAP embedding

if these implementations are in place, subclustering on HTTP-hosted RamData will be efficient and accurate.

# 2022-07-28 16:00:50 

Hard dependency on entries sorted by string representations ('id_feature' and 'id_cell') will be dropped. 
This decision was made to make integration of datasets (especially datasets fetched from the web with dataset exists locally) more efficient.
Also, this change will make the RamData construction process more time- and memory-efficient

Also, hard dependency on feature/barcode sorted ramtx will be dropped, and a new ramtx, dense ramtx will be introduced

# 2022-07-30 17:24:25 
metadata naming convention changed, removing backward compatibility.

dense ramtx was introduced, and the dependency on the existence of a paired sparse ramtx object was dropped. 
below is a short description about the different types of ramtx objects:

    # 'mode' of RAMtx objects
    There are three valid 'mode' (or internal structures) for RAMtx object : {'dense' or 'sparse_sorted_by_barcodes', 'sparse_sorted_by_features'}
    
    (sparse_sorted_by_barcodes) <---> (dense) <---> (sparse_sorted_by_features)
    *fast barcode data retrieval                    *fast feature data retrieval
    
    As shown above, RAMtx objects are interconvertible. Of note, for the converion between sparse ramtx sorted by barcodes and features, 'dense' ramtx object should be used in the conversion process.
    - dense ramtx object can be used to retrieve data of a single barcode or feature (with moderate efficiency)
    - sparse ramtx object can only be used data of either a single barcode ('sparse_sorted_by_barcodes') or a single feature ('sparse_sorted_by_features') very efficiently.
    
# 2022-08-01 20:38:54 

RamData.apply has been implemented
RAMtx.__getitem__ function has been improved to allow read data respecting chunk boundaries
RAMtx.batch_generator now respect chunk boundaries when creating batches

# 2022-08-03 00:39:12 
wrote a method in RamData for fast exploratory analysis of count data stored in dense, which enables normalizations/scaling of highly/variable genes of 200k single-cell data in 3~4 minutes from the count data
now ZarrDataFrame supports multi-dimensional data as a 'column'
also, mask, coordinate array, and orthogonal selections (all methods supported by Zarr) is also supported in ZarrDataFrame.

# 2022-08-05 01:53:22 
RAMtx.batch_generator was modified so that for dense matrix, appropriate batch that does not overwhelm the system memory
RamData.apply was modified to allow synchronization across processes. it is a bit slow than writing chunks by chunk, but consums far less memory.

# 2022-08-05 17:24:17 
improved RamData.__getitem__ method to set obsm and varm properties in the returned AnnData object

# 2022-08-05 17:42:03 
RamData dimension reduction methods were re-structured

# 2022-08-06 22:24:19 
RamData.pca was split into RamData.train_pca and RamData.apply_pca

# 2022-08-07 03:15:06 
RAMtx.batch_generator was modified to support progress_bar functionality (backed by 'tqdm')
RAMtx.get_total_num_records method was added to support progress_bar functionality (backed by 'tqdm')

# 2022-08-07 17:00:53 
RamData.umap has been split into RamData.train_umap and RamData.apply_umap

# 2022-08-08 00:49:58 
RamData._repr_html_ was added for more interactive visualization of RamData in an interactive IPython environment (such as Jupyter Notebook).

# 2022-08-08 23:14:16 
implemented PyNNdescent-backed implementation of cluster label assignments using labels assigned to subsampled barcodes, adding RamData.

# 2022-08-09 03:13:15 
implemented leiden clustering algorithm using 'leidenalg.find_partition' method

# 2022-08-10 04:51:54 
implemented subsampling method using iterative community detection and density-based subsampling algorithms (RamData.subsample)

# 2022-08-10 10:02:18 
RamData.delete_model error corrected

# 2022-08-12 01:51:27 
resolved error in RamData.summarize

# 2022-08-16 02:32:51 
resolved RAMtx.__getitem__ performance issues for dense RAMtx data when the length of axis not for querying is extremely large.
New algorithm uses sub-batches to convert dense matrix to sparse matrix in a memory-efficient conversion.

# 2022-08-16 11:21:53 
resolved an issue in RAMtx.survey_number_of_records_for_each_entry

# 2022-08-18 15:30:22 
bumped version to v0.0.5

# 2022-08-21 16:13:08 
RamData.get_expr function was implemented.
    possible usages: (1) calculating gene_set/pathway activities across cells, which can subsequently used for filtering cells for subclustering
                     (2) calculating pseudo-bulk expression profiles of a subset of cells across all active features 

# 2022-08-23 11:36:32 
RamData.find_markers : finding markers for each cluster using AUROC, log2fc, and statistical test methods in memory-efficient method, built on top of RamData.summarize
RamData.get_marker_table : retrieve markers as table from the results produced by 'RamData.find_markers'

# 2022-08-25 16:18:04 
initialized ZarrDataFrame class with an additional functionality (combined mode)

# 2022-08-27 10:21:45 
ZarrDataFrame class now support combined mode, where data are retrieved across multiple zdf conmponents, either sharing rows ('interleaved' type) or each has unique rows ('stacked' type).
currently supported operations are __getitem__
Also, fill_value of categorical column is now -1, interpreting empty values as np.nan by default

"""

''' previosuly written for biobookshelf '''
def CB_Parse_list_of_id_cell( l_id_cell, dropna = True ) :
    ''' # 2022-03-25 16:35:23 
    parse a given list of id_cells into a dataframe using 'SC.CB_detect_cell_barcode_from_id_cell' function
    'dropna' : drop id_cells that does not contains cell barcodes 
    '''
    df = pd.DataFrame( list( [ e ] + list( CB_detect_cell_barcode_from_id_cell( e ) ) for e in l_id_cell ), columns = [ 'id_cell', 'CB', 'id_sample_from_id_cell' ] ).set_index( 'id_cell' )
    return df
def CB_Build_dict_id_sample_to_set_cb( l_id_cell ) :
    ''' # 2022-03-28 22:24:30 
    Build a set of cell barcode for each id_sample from the given list of id_cells 
    '''
    df = CB_Parse_list_of_id_cell( l_id_cell )
    dict_id_sample_to_set_cb = dict( )
    for cb, id_sample in df.values :
        if id_sample not in dict_id_sample_to_set_cb :
            dict_id_sample_to_set_cb[ id_sample ] = set( )
        dict_id_sample_to_set_cb[ id_sample ].add( cb )
    return dict_id_sample_to_set_cb
def CB_Match_Batch( l_id_cell_1, l_id_cell_2, flag_calculate_proportion_using_l_id_cell_2 = True ) :
    ''' # 2022-03-28 23:10:43 
    Find matching batches between two given lists of id_cells by finding the batches sharing the largest number of cell barcodes
    
    'l_id_cell_1' : first list of id_cells 
    'l_id_cell_2' : second list of id_cells
    'flag_calculate_proportion_using_l_id_cell_2' : if True, finding matching batches using the shared proportion calculated using the cell barcodes from 'l_id_cell_2'. if False, proportion of the matching barcodes will be calculated using the cell barcodes from 'l_id_cell_1'
    
    return:
    df_id_cell_matched, df_sample_matched
    '''
    # retrieve set of cell barcodes 
    df_id_cell_1 = CB_Parse_list_of_id_cell( l_id_cell_1 )
    df_id_cell_2 = CB_Parse_list_of_id_cell( l_id_cell_2 )
    dict_id_sample_to_set_cb_1 = CB_Build_dict_id_sample_to_set_cb( l_id_cell_1 )
    dict_id_sample_to_set_cb_2 = CB_Build_dict_id_sample_to_set_cb( l_id_cell_2 )

    # Find matching id_samples of the two given list of id_cells
    # calculate the proportion of matching cell barcodes between each pair of samples from the two given list of id_cells
    l_l = [ ]
    for id_sample_1 in dict_id_sample_to_set_cb_1 :
        for id_sample_2 in dict_id_sample_to_set_cb_2 :
            set_cb_1 = dict_id_sample_to_set_cb_1[ id_sample_1 ]
            set_cb_2 = dict_id_sample_to_set_cb_2[ id_sample_2 ]
            float_prop_matching_cb = len( set_cb_1.intersection( set_cb_2 ) ) / ( len( set_cb_2 ) if flag_calculate_proportion_using_l_id_cell_2 else len( set_cb_1 ) )
            l_l.append( [ id_sample_1, id_sample_2, float_prop_matching_cb ] )
    df = pd.DataFrame( l_l, columns = [ 'id_sample_1', 'id_sample_2', 'float_prop_matching_cb' ] ) # search result
    df_sample_matched = df.sort_values( 'float_prop_matching_cb', ascending = False ).drop_duplicates( 'id_sample_2', keep = 'first' ).drop_duplicates( 'id_sample_1', keep = 'first' ) # retrieve the best matches between samples so that a unique mapping exists for every sample

    # Find matching id_cells of given two list of id_cells
    df_id_cell_1.reset_index( inplace = True, drop = False )
    df_id_cell_2.reset_index( inplace = True, drop = False )
    df_id_cell_1.rename( columns = { 'id_sample_from_id_cell' : 'id_sample_from_id_cell_1' }, inplace = True )
    df_id_cell_2.rename( columns = { 'id_sample_from_id_cell' : 'id_sample_from_id_cell_2' }, inplace = True )
    df_id_cell_1[ 'id_sample_from_id_cell_2' ] = df_id_cell_1.id_sample_from_id_cell_1.apply( MAP.Map( df_sample_matched.set_index( 'id_sample_1' ).id_sample_2.to_dict( ) ).a2b )
    df_id_cell_1.dropna( subset = [ 'id_sample_from_id_cell_2' ], inplace = True ) # ignore cells without matching id_sample from the '2' batch
    df_id_cell_1.set_index( [ 'CB', 'id_sample_from_id_cell_2' ], inplace = True )
    df_id_cell_matched = df_id_cell_1.join( df_id_cell_2[ ~ pd.isnull( df_id_cell_2.id_sample_from_id_cell_2 ) ].set_index( [ 'CB', 'id_sample_from_id_cell_2' ] ), lsuffix = '_1', rsuffix = '_2' ) # match id_cells from two given list of id_cells
    df_id_cell_matched.reset_index( drop = False, inplace = True )
    df_id_cell_matched = df_id_cell_matched[ [ 'id_cell_1', 'id_cell_2', 'CB', 'id_sample_from_id_cell_1', 'id_sample_from_id_cell_2' ] ] # reorder columns
    
    return df_id_cell_matched, df_sample_matched
def SCANPY_Detect_cell_barcode_from_cell_id( adata ) :
    ''' # 2022-03-24 20:35:22 
    Detect cell barcodes from id_cell (index of adata.obs), and add new two columns to the adata.obs [ 'CB', 'id_sample_from_id_cell' ]
    '''
    adata.obs = adata.obs.join( pd.DataFrame( list( [ e ] + list( CB_detect_cell_barcode_from_id_cell( e ) ) for e in adata.obs.index.values ), columns = [ 'id_cell', 'CB', 'id_sample_from_id_cell' ] ).set_index( 'id_cell' ) )
def SCANPY_Retrieve_Markers_as_DataFrame( adata ) :
    ''' # 2022-02-15 14:40:02 
    receive scanpy anndata and return a dataframe contianing marker genes 
    
    --- return --- 
    df_marker : a dataframe contianing marker genes 
    '''
    l_df = [ ]
    for index_clus, name_clus in enumerate( adata.uns["rank_genes_groups"]['names'].dtype.names ) : 
        df = pd.DataFrame( dict( ( name_col, adata.uns["rank_genes_groups"][ name_col ][ name_clus ] ) for name_col in ['logfoldchanges', 'names', 'pvals', 'pvals_adj', 'scores' ] ) )
        df[ "name_clus" ] = name_clus 
        df[ "index_clus" ] = index_clus
        l_df.append( df )
    df_marker = pd.concat( l_df )
    return df_marker
def CB_detect_cell_barcode_from_id_cell( id_cell, int_number_atgc_in_cell_barcode = 16 ) :
    ''' # 2022-02-21 00:03:34 
    retrieve cell_barcode from id_cell 
    'int_number_atgc_in_cell_barcode' : number of ATGC characters in the cell barcode
    '''
    int_count_atgc = 0
    int_start_appearance_of_atgc = None
    set_atgc = set( "ATGC" )
    
    def __retrieve_cell_barcode_and_id_channel_from_id_cell__( id_cell, int_start_appearance_of_atgc, int_number_atgc_in_cell_barcode ) :
        ''' __retrieve_cell_barcode_and_id_channel_from_id_cell__ '''
        int_cb_start = int_start_appearance_of_atgc
        int_cb_end = int_start_appearance_of_atgc + int_number_atgc_in_cell_barcode
        return [ id_cell[ int_cb_start : int_cb_end ], id_cell[ : int_cb_start ] + '|' + id_cell[ int_cb_end : ] ] # return cell_barcode, id_channel
        
    for index_c, c in enumerate( id_cell.upper( ) ) : # case-insensitive detection of cell-barcodes
        if c in set_atgc :
            if int_start_appearance_of_atgc is None:
                int_start_appearance_of_atgc = index_c
            int_count_atgc += 1
        else :
            ''' identify cell barcode and return the cell barcode '''
            if int_start_appearance_of_atgc is not None:
                if int_count_atgc == int_number_atgc_in_cell_barcode :
                    return __retrieve_cell_barcode_and_id_channel_from_id_cell__( id_cell, int_start_appearance_of_atgc, int_number_atgc_in_cell_barcode )
            # initialize the next search
            int_count_atgc = 0 
            int_start_appearance_of_atgc = None
    ''' identify cell barcode and return the cell barcode '''
    if int_start_appearance_of_atgc is not None:
        if int_count_atgc == int_number_atgc_in_cell_barcode :
            return __retrieve_cell_barcode_and_id_channel_from_id_cell__( id_cell, int_start_appearance_of_atgc, int_number_atgc_in_cell_barcode )
    ''' return None when cell_barcode was not found '''
    return [ None, None ]

def Read_10X( path_folder_mtx_10x, verbose = False ) :
    ''' # 2021-11-24 13:00:13 
    read 10x count matrix
    'path_folder_mtx_10x' : a folder containing files for 10x count matrix
    return df_mtx, df_feature
    '''
    # handle inputs
    if path_folder_mtx_10x[ -1 ] != '/' :
        path_folder_mtx_10x += '/'
    
    # define input file directories
    path_file_bc = f'{path_folder_mtx_10x}barcodes.tsv.gz'
    path_file_feature = f'{path_folder_mtx_10x}features.tsv.gz'
    path_file_mtx = f'{path_folder_mtx_10x}matrix.mtx.gz'

    # check whether all required files are present
    if sum( list( not os.path.exists( path_folder ) for path_folder in [ path_file_bc, path_file_feature, path_file_mtx ] ) ) :
        if verbose :
            print( f'required file(s) is not present at {path_folder_mtx_10x}' )

    # read mtx file as a tabular format
    df_mtx = pd.read_csv( path_file_mtx, sep = ' ', comment = '%' )
    df_mtx.columns = [ 'id_row', 'id_column', 'read_count' ]

    # read barcode and feature information
    df_bc = pd.read_csv( path_file_bc, sep = '\t', header = None )
    df_bc.columns = [ 'barcode' ]
    df_feature = pd.read_csv( path_file_feature, sep = '\t', header = None )
    df_feature.columns = [ 'id_feature', 'feature', 'feature_type' ]

    # mapping using 1 based coordinates (0->1 based coordinate )
    df_mtx[ 'barcode' ] = df_mtx.id_column.apply( MAP.Map( DICTIONARY_Build_from_arr( df_bc.barcode.values, index_start = 1 ) ).a2b ) # mapping using 1 based coordinates (0->1 based coordinate )
    df_mtx[ 'id_feature' ] = df_mtx.id_row.apply( MAP.Map( DICTIONARY_Build_from_arr( df_feature.id_feature.values, index_start = 1 ) ).a2b ) 
    df_mtx.drop( columns = [ 'id_row', 'id_column' ], inplace = True ) # drop unnecessary columns
    
    return df_mtx, df_feature
def Write_10X( df_mtx, df_feature, path_folder_output_mtx_10x ) :
    """ # 2021-11-24 12:57:30 
    'df_feature' should contains the following column names : [ 'id_feature', 'feature', 'feature_type' ]
    'df_mtx' should contains the following column names : [ 'id_feature', 'barcode', 'read_count' ]
    'path_folder_output_mtx_10x' : an output folder directory where the mtx_10x files will be written

    """
    df_mtx = deepcopy( df_mtx ) # create a copy of df_mtx before modification

    # create an output folder
    os.makedirs( path_folder_output_mtx_10x, exist_ok = True )

    ''' save barcode file '''
    # retrieve list of barcodes
    arr_barcode = LIST_COUNT( df_mtx.barcode, duplicate_filter = None ).index.values
    pd.DataFrame( arr_barcode ).to_csv( f"{path_folder_output_mtx_10x}barcodes.tsv.gz", sep = '\t', index = False, header = False ) 

    ''' save feature file '''
    # compose a feature dataframe
    df_feature[ [ 'id_feature', 'feature', 'feature_type' ] ].to_csv( f"{path_folder_output_mtx_10x}features.tsv.gz", sep = '\t', index = False, header = False ) # save as a file
    # retrieve list of features
    arr_id_feature = df_feature.id_feature.values

    ''' save matrix file '''
    # convert feature and barcode to integer indices
    df_mtx.id_feature = df_mtx.id_feature.apply( MAP.Map( DICTIONARY_Build_from_arr( arr_id_feature, order_index_entry = False ) ).a2b ) # 0-based coordinates
    df_mtx.barcode = df_mtx.barcode.apply( MAP.Map( DICTIONARY_Build_from_arr( arr_barcode, order_index_entry = False ) ).a2b ) # 0-based coordinates
    # save count matrix as a gzipped matrix market format
    row, col, data = df_mtx[ [ 'id_feature', 'barcode', 'read_count' ] ].values.T
    sm = scipy.sparse.coo_matrix( ( data, ( row, col ) ), shape = ( len( arr_id_feature ), len( arr_barcode ) ) )
    scipy.io.mmwrite( f"{path_folder_output_mtx_10x}matrix", sm )
    # remove previous output file to overwrite the file
    path_file_mtx_output = f"{path_folder_output_mtx_10x}matrix.mtx.gz"
    if os.path.exists( path_file_mtx_output ) :
        os.remove( path_file_mtx_output )
    OS_Run( [ 'gzip', f"{path_folder_output_mtx_10x}matrix.mtx" ] ) # gzip the mtx file
def __function_for_adjusting_thresholds_for_filtering_empty_droplets__( path_folder_mtx_10x_output, min_counts, min_features, min_cells ) :
    ''' # 2022-02-23 14:26:07 
    This function is intended for the use in 'MTX_10X_Filter' function for filtering cells from the 10X dataset (before chromium X, 10,000 cells per channel)
    
    Assuming a typical number of droplets in a experiment is 100,000, adjust 'min_counts' to reduce the number of filtered cells below 'int_max_num_cells' 
    '''
    s_count = pd.read_csv( f"{path_folder_mtx_10x_output}dict_id_column_to_count.before_filtering.tsv.gz", sep = '\t', header = None, index_col = 0 )[ 1 ].sort_values( ascending = False ).iloc[ : 100000 ]
    
    int_max_num_cells = 20000 # maximum number of allowed cells
    min_counts_maximum = 2000
    def function_for_increasing_min_counts( min_counts ) :
        return min_counts * 2
    while True :
        ''' increase threshold if the number of filtered cells is larger than 'int_max_num_cells' '''
        if len( s_count[ s_count > min_counts ] ) > int_max_num_cells and min_counts < min_counts_maximum :
            min_counts = function_for_increasing_min_counts( min_counts )
        else :
            break
    return min_counts, min_features, min_cells
def MTX_10X_Split( path_folder_mtx_10x_output, int_max_num_entries_for_chunk = 10000000, flag_split_mtx = True, flag_split_mtx_again = False ) :
    ''' # 2022-04-28 01:16:35 
    split input mtx file into multiple files and write a flag file indicating the splitting has been completed. 
    return the list of split mtx files
    
    'flag_split_mtx' : if 'flag_split_mtx' is True, split input mtx file into multiple files. if False, does not split the input matrix, and just return the list containing a single path pointing to the input matrix. This flag exists for the compatibility with single-thread operations
    'flag_split_mtx_again' : split the input matrix again even if it has beem already split. It will remove previously split files.
    '''
    # 'flag_split_mtx' : if False, does not split the input matrix, and just return the list containing a single path pointing to the input matrix
    if not flag_split_mtx :
        return [ f"{path_folder_mtx_10x_output}matrix.mtx.gz" ]
    
    ''' if 'flag_split_mtx_again' flag is True, remove previously split files '''
    path_file_flag = f"{path_folder_mtx_10x_output}matrix.mtx.gz.split.flag"
    if flag_split_mtx_again :
        os.remove( path_file_flag ) # remove the flag
        # remove previously split files
        for path_file in glob.glob( f'{path_folder_mtx_10x_output}matrix.mtx.gz.*.gz' ) :
            os.remove( path_file )

    ''' split input matrix file '''
    if not os.path.exists( path_file_flag ) : # check whether the flag exists
        index_mtx_10x = 0
        newfile = gzip.open( f"{path_folder_mtx_10x_output}matrix.mtx.gz.{index_mtx_10x}.gz", 'wb' )
        l_path_file_mtx_10x = [ f"{path_folder_mtx_10x_output}matrix.mtx.gz.{index_mtx_10x}.gz" ]
        int_num_entries_written_for_the_current_chunk = 0
        with gzip.open( f"{path_folder_mtx_10x_output}matrix.mtx.gz", 'rb' ) as file :
            while True :
                line = file.readline( ) # binary string
                if len( line ) == 0 :
                    newfile.close( ) # close the output file
                    break
                ''' write the line to the current chunk and update the number of entries written for the current chunk '''
                newfile.write( line )
                int_num_entries_written_for_the_current_chunk += 1
                ''' initialize the next chunk if a sufficient number of entries were written '''
                if int_num_entries_written_for_the_current_chunk >= int_max_num_entries_for_chunk :
                    newfile.close( ) # close the output file
                    # initialize the next chunk
                    index_mtx_10x += 1
                    int_num_entries_written_for_the_current_chunk = 0
                    newfile = gzip.open( f"{path_folder_mtx_10x_output}matrix.mtx.gz.{index_mtx_10x}.gz", 'wb' )
                    l_path_file_mtx_10x.append( f"{path_folder_mtx_10x_output}matrix.mtx.gz.{index_mtx_10x}.gz" )
        with open( path_file_flag, 'w' ) as file :
            file.write( 'completed' )
    else :
        ''' retrieve the list of split mtx files '''
        df = GLOB_Retrive_Strings_in_Wildcards( f"{path_folder_mtx_10x_output}matrix.mtx.gz.*.gz" )
        df.wildcard_0 = df.wildcard_0.astype( int )
        df.sort_values( 'wildcard_0', ascending = True, inplace = True )
        l_path_file_mtx_10x = df.path.values
    return l_path_file_mtx_10x
dict_id_feature_to_index_feature = dict( )
def __MTX_10X_Combine__renumber_feature_mtx_10x__( path_file_input, path_folder_mtx_10x_output ) :
    ''' # deprecated
    internal function for MTX_10X_Combine
    # 2022-02-22 00:38:33 
    '''
#     dict_id_feature_to_index_feature = PICKLE_Read( f'{path_folder_mtx_10x_output}dict_id_feature_to_index_feature.pickle' ) # retrieve id_feature to index_feature mapping 
    for path_folder_mtx_10x, int_total_n_barcodes_of_previously_written_matrices, index_mtx_10x in pd.read_csv( path_file_input, sep = '\t' ).values :
        # directly write matrix.mtx.gz file without header
        with gzip.open( f"{path_folder_mtx_10x_output}matrix.mtx.gz.{index_mtx_10x}.gz", 'wb' ) as newfile :
            arr_id_feature = pd.read_csv( f'{path_folder_mtx_10x}features.tsv.gz', sep = '\t', header = None ).values[ :, 0 ] # retrieve a list of id_feature for the current dataset
            with gzip.open( f'{path_folder_mtx_10x}matrix.mtx.gz', 'rb' ) as file : # retrieve a list of features
                line = file.readline( ).decode( ) # read the first line
                # if the first line of the file contains a comment line, read all comment lines and a description line following the comments.
                if len( line ) > 0 and line[ 0 ] == '%' :
                    # read comment and the description line
                    while True :
                        if line[ 0 ] != '%' :
                            break
                        line = file.readline( ).decode( ) # read the next line
                    line = file.readline( ).decode( ) # discard the description line and read the next line
                # process entries
                while True :
                    if len( line ) == 0 :
                        break
                    index_row, index_col, int_value = tuple( map( int, line.strip( ).split( ) ) ) # parse each entry of the current matrix 
                    newfile.write( ( ' '.join( tuple( map( str, [ dict_id_feature_to_index_feature[ arr_id_feature[ index_row - 1 ] ], index_col + int_total_n_barcodes_of_previously_written_matrices, int_value ] ) ) ) + '\n' ).encode( ) ) # translate indices of the current matrix to that of the combined matrix            
                    line = file.readline( ).decode( ) # read the next line
def Read_SPLiT_Seq( path_folder_mtx ) :
    ''' # 2022-04-22 07:10:50 
    Read SPLiT-Seq pipeline output 
    return:
    df_feature, df_mtx
    '''
    path_file_bc = f'{path_folder_mtx}cell_metadata.csv'
    path_file_feature = f'{path_folder_mtx}genes.csv'
    path_file_mtx = f'{path_folder_mtx}DGE.mtx'

    # read mtx file as a tabular format
    df_mtx = pd.read_csv( path_file_mtx, sep = ' ', comment = '%' )
    df_mtx.columns = [ 'id_row', 'id_column', 'read_count' ]

    # read barcode and feature information
    df_bc = pd.read_csv( path_file_bc )[ [ 'cell_barcode' ] ]
    df_bc.columns = [ 'barcode' ]
    df_feature = pd.read_csv( path_file_feature, index_col = 0 )
    df_feature.columns = [ 'id_feature', 'feature', 'genome' ]

    # mapping using 1 based coordinates (0->1 based coordinate )
    df_mtx[ 'barcode' ] = df_mtx.id_row.apply( MAP.Map( DICTIONARY_Build_from_arr( df_bc.barcode.values, index_start = 1 ) ).a2b ) # mapping using 1 based coordinates (0->1 based coordinate )
    df_mtx[ 'id_feature' ] = df_mtx.id_column.apply( MAP.Map( DICTIONARY_Build_from_arr( df_feature.id_feature.values, index_start = 1 ) ).a2b ) 
    df_mtx.drop( columns = [ 'id_row', 'id_column' ], inplace = True ) # drop unnecessary columns
    return df_feature, df_mtx
def MTX_10X_Barcode_add_prefix_or_suffix( path_file_barcodes_input, path_file_barcodes_output = None, barcode_prefix = '', barcode_suffix = '' ) :
    ''' # 2022-05-13 17:54:13 
    Add prefix or suffix to the 'barcode' of a given 'barcodes.tsv.gz' file
    'path_file_barcodes_output' : default: None. by default, the input 'path_file_barcodes_input' file will be overwritten with the modified barcodes
    '''
    flag_replace_input_file = path_file_barcodes_output is None # retrieve a flag indicating the replacement of original input file with modified input file
    if flag_replace_input_file :
        path_file_barcodes_output = f'{path_file_barcodes_input}.writing.tsv.gz' # set a temporary output file 
    newfile = gzip.open( path_file_barcodes_output, 'wb' ) # open an output file
    with gzip.open( path_file_barcodes_input, 'rb' ) as file :
        while True :
            line = file.readline( )
            if len( line ) == 0 :
                break
            barcode = line.decode( ).strip( ) # parse a barcode
            barcode_new = barcode_prefix + barcode + barcode_suffix # compose new barcode
            newfile.write( ( barcode_new + '\n' ).encode( ) ) # write a new barcode
    newfile.close( ) # close the output file
    # if the output file path was not given, replace the original file with modified file
    if flag_replace_input_file :
        os.remove( path_file_barcodes_input )
        os.rename( path_file_barcodes_output, path_file_barcodes_input )
def MTX_10X_Feature_add_prefix_or_suffix( path_file_features_input, path_file_features_output = None, id_feature_prefix = '', id_feature_suffix = '', name_feature_prefix = '', name_feature_suffix = '' ) :
    ''' # 2022-05-13 17:54:17 
    Add prefix or suffix to the id_feature and name_feature of a given 'features.tsv.gz' file
    'path_file_features_output' : default: None. by default, the input 'path_file_features_input' file will be overwritten with the modified features
    '''
    flag_replace_input_file = path_file_features_output is None # retrieve a flag indicating the replacement of original input file with modified input file
    if flag_replace_input_file :
        path_file_features_output = f'{path_file_features_input}.writing.tsv.gz' # set a temporary output file 
    newfile = gzip.open( path_file_features_output, 'wb' ) # open an output file
    with gzip.open( path_file_features_input, 'rb' ) as file :
        while True :
            line = file.readline( )
            if len( line ) == 0 :
                break
            id_feature, name_feature, type_feature = line.decode( ).strip( ).split( '\t' ) # parse a feature
            id_feature_new = id_feature_prefix + id_feature + id_feature_suffix # compose new id_feature
            name_feature_new = name_feature_prefix + name_feature + name_feature_suffix # compose new name_feature
            newfile.write( ( '\t'.join( [ id_feature_new, name_feature_new, type_feature ] ) + '\n' ).encode( ) ) # write a new feature
    newfile.close( ) # close the output file
    # if the output file path was not given, replace the original file with modified file
    if flag_replace_input_file :
        os.remove( path_file_features_input )
        os.rename( path_file_features_output, path_file_features_input )
def __MTX_10X_Combine__renumber_barcode_or_feature_index_mtx_10x__( path_file_input, path_folder_mtx_10x_output, flag_renumber_feature_index ) :
    '''
    internal function for MTX_10X_Combine
    # 2022-04-21 12:10:53 
    
    'flag_renumber_feature_index' : if True, assumes barcodes are not shared between matrices and renumber features only. If False, assumes features are not shared between matrices and renumber barcodes only.
    '''
    global dict_id_entry_to_index_entry
    for path_folder_mtx_10x, int_total_n_entries_of_previously_written_matrices, index_mtx_10x in pd.read_csv( path_file_input, sep = '\t' ).values :
        # directly write matrix.mtx.gz file without header
        with gzip.open( f"{path_folder_mtx_10x_output}matrix.mtx.gz.{index_mtx_10x}.gz", 'wb' ) as newfile :
            arr_id_entry = pd.read_csv( f"{path_folder_mtx_10x}{'features' if flag_renumber_feature_index else 'barcodes'}.tsv.gz", sep = '\t', header = None ).values[ :, 0 ] # retrieve a list of id_feature for the current dataset
            with gzip.open( f'{path_folder_mtx_10x}matrix.mtx.gz', 'rb' ) as file : # retrieve a list of features
                line = file.readline( ).decode( ) # read the first line
                # if the first line of the file contains a comment line, read all comment lines and a description line following the comments.
                if len( line ) > 0 and line[ 0 ] == '%' :
                    # read comment and the description line
                    while True :
                        if line[ 0 ] != '%' :
                            break
                        line = file.readline( ).decode( ) # read the next line
                    line = file.readline( ).decode( ) # discard the description line and read the next line
                # process entries
                while True :
                    if len( line ) == 0 :
                        break
                    index_row, index_col, int_value = tuple( map( int, map( float, line.strip( ).split( ) ) ) ) # parse each entry of the current matrix 
                    
                    newfile.write( ( ' '.join( tuple( map( str, ( [ dict_id_entry_to_index_entry[ arr_id_entry[ index_row - 1 ] ], index_col + int_total_n_entries_of_previously_written_matrices ] if flag_renumber_feature_index else [ index_row + int_total_n_entries_of_previously_written_matrices, dict_id_entry_to_index_entry[ arr_id_entry[ index_col - 1 ] ] ] ) + [ int_value ] ) ) ) + '\n' ).encode( ) ) # translate indices of the current matrix to that of the combined matrix            
                    line = file.readline( ).decode( ) # read the next line
def MTX_10X_Combine( path_folder_mtx_10x_output, * l_path_folder_mtx_10x_input, int_num_threads = 15, flag_split_mtx = True, flag_split_mtx_again = False, int_max_num_entries_for_chunk = 10000000, flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs = None, flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs = None, verbose = False ) :
    '''
    # 2022-08-20 01:04:36 
    Combine 10X count matrix files from the given list of folders and write combined output files to the given output folder 'path_folder_mtx_10x_output'
    If there are no shared cells between matrix files, a low-memory mode will be used. The output files will be simply combined since no count summing operation is needed. Only feature matrix will be loaded and updated in the memory.
    'id_feature' should be unique across all features. if id_feature is not unique, features with duplicated id_features will lead to combining of the features into a single feature (with combined counts/values).
    
    'int_num_threads' : number of threads to use when combining datasets. multiple threads will be utilized only when datasets does not share cells and thus can be safely concatanated.
    'flag_split_mtx' : split the resulting mtx file so that the contents in the output mtx file can be processed in parallel without ungzipping the mtx.gz file and spliting the file.
    'flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs' : a flag for entering low-memory mode when there is no shared cells between given input matrices. By default (when None is given), matrices will be examined and the flag will be set automatically by the program. To reduce running time and memory, this flag can be manually set by users. Explicitly setting this flag will dramatically reduce the memory consumption. 
    'flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs' : a flag for entering low-memory mode when there is no shared features between given input matrices. By default (when None is given), matrices will be examined and the flag will be set automatically by the program. To reduce running time and memory, this flag can be manually set by users. Explicitly setting this flag will dramatically reduce the memory consumption. 
    '''
    
    # create an output folder
    os.makedirs( path_folder_mtx_10x_output, exist_ok = True ) 

    if not flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs and flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs is None :
        """ retrieve cell barcodes of all 10X matrices and check whether cell barcodes are not shared between matrices """
        int_total_n_barcodes_of_previously_written_matrices = 0 # follow the number of barcodes that are previously written
        l_int_total_n_barcodes_of_previously_written_matrices = [ ] # calculate the number of barcodes of the previous dataset in the combined mtx.
        set_barcode = set( ) # update a set of unique barcodes
        for path_folder_mtx_10x in l_path_folder_mtx_10x_input :
            arr_barcode = pd.read_csv( f'{path_folder_mtx_10x}barcodes.tsv.gz', sep = '\t', header = None ).squeeze( "columns" ).values # retrieve a list of features
            set_barcode.update( arr_barcode ) # update a set of barcodes
            l_int_total_n_barcodes_of_previously_written_matrices.append( int_total_n_barcodes_of_previously_written_matrices )
            int_total_n_barcodes_of_previously_written_matrices += len( arr_barcode ) # update the number of barcodes 
        ''' check whether there are shared cell barcodes between matrices and set a flag for entering a low-memory mode '''
        flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs = len( set_barcode ) == int_total_n_barcodes_of_previously_written_matrices # update flag
    elif flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs :
        """ retrieve cell barcodes of all 10X matrices and check whether cell barcodes are not shared between matrices """
        int_total_n_barcodes_of_previously_written_matrices = 0 # follow the number of barcodes that are previously written
        l_int_total_n_barcodes_of_previously_written_matrices = [ ] # calculate the number of barcodes of the previous dataset in the combined mtx.
        for path_folder_mtx_10x in l_path_folder_mtx_10x_input :
            l_int_total_n_barcodes_of_previously_written_matrices.append( int_total_n_barcodes_of_previously_written_matrices )
            int_total_n_barcodes_of_previously_written_matrices += len( pd.read_csv( f'{path_folder_mtx_10x}barcodes.tsv.gz', sep = '\t', header = None ) ) # retrieve a list of barcodes and # update the number of barcodes 
    
    if not flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs and flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs is None :
        """ retrieve features of all 10X matrices and check whether features are not shared between matrices """
        int_total_n_features_of_previously_written_matrices = 0 # follow the number of features that are previously written
        l_int_total_n_features_of_previously_written_matrices = [ ] # calculate the number of features of the previous dataset in the combined mtx.
        set_feature = set( ) # update a set of unique features
        for path_folder_mtx_10x in l_path_folder_mtx_10x_input :
            arr_feature = pd.read_csv( f'{path_folder_mtx_10x}features.tsv.gz', sep = '\t', header = None, usecols = [ 0 ] ).squeeze( "columns" ).values # retrieve a list of features
            set_feature.update( arr_feature ) # update a set of features
            l_int_total_n_features_of_previously_written_matrices.append( int_total_n_features_of_previously_written_matrices )
            int_total_n_features_of_previously_written_matrices += len( arr_feature ) # update the number of features 
        ''' check whether there are shared features between matrices and set a flag for entering a low-memory mode '''
        flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs = len( set_feature ) == int_total_n_features_of_previously_written_matrices # update flag
    elif flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs :
        """ retrieve features of all 10X matrices and check whether features are not shared between matrices """
        int_total_n_features_of_previously_written_matrices = 0 # follow the number of features that are previously written
        l_int_total_n_features_of_previously_written_matrices = [ ] # calculate the number of features of the previous dataset in the combined mtx.
        for path_folder_mtx_10x in l_path_folder_mtx_10x_input :
            l_int_total_n_features_of_previously_written_matrices.append( int_total_n_features_of_previously_written_matrices )
            int_total_n_features_of_previously_written_matrices += len( pd.read_csv( f'{path_folder_mtx_10x}features.tsv.gz', sep = '\t', header = None, usecols = [ 0 ] ) ) # retrieve a list of features and update the number of features 
        

    flag_low_memory_mode = flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs or flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs # retrieve flag for low-memory mode
    if flag_low_memory_mode :
        """ low-memory mode """
        flag_renumber_feature_index = flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs # retrieve a flag for renumbering features
        if verbose :
            print( f"entering low-memory mode, re-numbering {'features' if flag_renumber_feature_index else 'barcodes'} index because {'barcodes' if flag_renumber_feature_index else 'features'} are not shared across the matrices." )
        
        """ write a combined barcodes/features.tsv.gz - that are not shared between matrices """
        OS_FILE_Combine_Files_in_order( list( f"{path_folder_mtx_10x}{'barcodes' if flag_renumber_feature_index else 'features'}.tsv.gz" for path_folder_mtx_10x in l_path_folder_mtx_10x_input ), f"{path_folder_mtx_10x_output}{'barcodes' if flag_renumber_feature_index else 'features'}.tsv.gz", overwrite_existing_file = True )

        ''' collect a set of unique entries and a list of entries for each 10X matrix '''
        set_t_entry = set( ) # update a set unique id_entry (either id_cell or id_entry)
        for path_folder_mtx_10x in l_path_folder_mtx_10x_input :
            set_t_entry.update( list( map( tuple, pd.read_csv( f"{path_folder_mtx_10x}{'features' if flag_renumber_feature_index else 'barcodes'}.tsv.gz", sep = '\t', header = None ).values ) ) ) # update a set of feature tuples

        """ write a combined features/barcodes.tsv.gz - that are shared between matrices """
        l_t_entry = list( set_t_entry ) # convert set to list
        with gzip.open( f"{path_folder_mtx_10x_output}{'features' if flag_renumber_feature_index else 'barcodes'}.tsv.gz", 'wb' ) as newfile :
            for t_entry in l_t_entry :
                newfile.write( ( '\t'.join( t_entry ) + '\n' ).encode( ) )

        """ build a mapping of id_entry to index_entry, which will be consistent across datasets - for features/barcodes that are shared between matrices """
        global dict_id_entry_to_index_entry # use global variable for multiprocessing
        dict_id_entry_to_index_entry = dict( ( t_entry[ 0 ], index_entry + 1 ) for index_entry, t_entry in enumerate( l_t_entry ) ) # 0>1 based index
        PICKLE_Write( f'{path_folder_mtx_10x_output}dict_id_entry_to_index_entry.pickle', dict_id_entry_to_index_entry ) # save id_feature to index_feature mapping as a pickle file

        ''' collect the number of records for each 10X matrix '''
        int_total_n_records = 0 
        for path_folder_mtx_10x in l_path_folder_mtx_10x_input :
            with gzip.open( f'{path_folder_mtx_10x}matrix.mtx.gz', 'rt' ) as file : # retrieve a list of features
                line = file.readline( )
                while line[ 0 ] == '%' :
                    line = file.readline( )
                int_total_n_records += int( line.strip( ).split( )[ 2 ] ) # update the total number of entries

        """ write a part of a combined matrix.mtx.gz for each dataset using multiple processes """
        # compose inputs for multiprocessing
        df_input = pd.DataFrame( { 'path_folder_input_mtx_10x' : l_path_folder_mtx_10x_input, 'int_total_n_barcodes_of_previously_written_matrices' : ( l_int_total_n_barcodes_of_previously_written_matrices if flag_renumber_feature_index else l_int_total_n_features_of_previously_written_matrices ), 'index_mtx_10x' : np.arange( len( l_int_total_n_barcodes_of_previously_written_matrices ) if flag_renumber_feature_index else len( l_int_total_n_features_of_previously_written_matrices ) ) } )
        Multiprocessing( df_input, __MTX_10X_Combine__renumber_barcode_or_feature_index_mtx_10x__, int_num_threads, global_arguments = [ path_folder_mtx_10x_output, flag_renumber_feature_index ] )
#         os.remove( f'{path_folder_mtx_10x_output}dict_id_entry_to_index_entry.pickle' ) # remove pickle file
        
        """ combine parts and add the MTX file header to compose a combined mtx file """
        df_file = GLOB_Retrive_Strings_in_Wildcards( f"{path_folder_mtx_10x_output}matrix.mtx.gz.*.gz" )
        df_file.wildcard_0 = df_file.wildcard_0.astype( int )
        df_file.sort_values( 'wildcard_0', inplace = True )
        # if 'flag_split_mtx' is True, does not delete the split mtx files
        OS_FILE_Combine_Files_in_order( df_file.path.values, f"{path_folder_mtx_10x_output}matrix.mtx.gz", delete_input_files = not flag_split_mtx, header = f"%%MatrixMarket matrix coordinate integer general\n%\n{len( l_t_entry ) if flag_renumber_feature_index else int_total_n_features_of_previously_written_matrices} {int_total_n_barcodes_of_previously_written_matrices if flag_renumber_feature_index else len( l_t_entry )} {int_total_n_records}\n" ) # combine the output mtx files in the given order
        # write a flag indicating that the current output directory contains split mtx files
        with open( f"{path_folder_mtx_10x_output}matrix.mtx.gz.split.flag", 'w' ) as file :
            file.write( 'completed' )
        
    else :
        ''' normal operation mode perfoming count merging operations '''
        l_df_mtx, l_df_feature = [ ], [ ]
        for path_folder_mtx_10x in l_path_folder_mtx_10x_input :
            df_mtx, df_feature = Read_10X( path_folder_mtx_10x )
            l_df_mtx.append( df_mtx ), l_df_feature.append( df_feature )

        # combine mtx
        df_mtx = pd.concat( l_df_mtx )
        df_mtx = df_mtx.groupby( [ 'barcode', 'id_feature' ] ).sum( )
        df_mtx.reset_index( drop = False, inplace = True )

        # combine features
        df_feature = pd.concat( l_df_feature )
        df_feature.drop_duplicates( inplace = True )

        Write_10X( df_mtx, df_feature, path_folder_mtx_10x_output )
        
        # split a matrix file into multiple files
        MTX_10X_Split( path_folder_mtx_10x_output, int_max_num_entries_for_chunk = int_max_num_entries_for_chunk )
def __Combine_Dictionaries__( path_folder_mtx_10x_input, name_dict ) :
    """ # 2022-03-06 00:06:23 
    combined dictionaries processed from individual files
    """
    if os.path.exists( f'{path_folder_mtx_10x_input}{name_dict}.tsv.gz' ) :
        ''' if an output file already exists, read the file and return the combined dictionary '''
        dict_combined = pd.read_csv( f'{path_folder_mtx_10x_input}{name_dict}.tsv.gz', sep = '\t', header = None, index_col = 0 ).iloc[ :, 0 ].to_dict( )
    else :
        ''' combine summarized results '''
        l_path_file = glob.glob( f"{path_folder_mtx_10x_input}{name_dict}.*" )
        try :
            counter = collections.Counter( pd.read_csv( l_path_file[ 0 ], sep = '\t', header = None, index_col = 0 ).iloc[ :, 0 ].to_dict( ) ) # initialize counter object with the dictionary from the first file
        except pd.errors.EmptyDataError :
            counter = collections.Counter( ) # when an error (possibly because the file is empty) occur, use an empty counter
        for path_file in l_path_file[ 1 : ] :
            # when an error (possibly because the file is empty) occur, skip updating the counter
            try :
                counter = counter + collections.Counter( pd.read_csv( path_file, sep = '\t', header = None, index_col = 0 ).iloc[ :, 0 ].to_dict( ) ) # update counter object using the dictionary from each file
            except pd.errors.EmptyDataError :
                pass
        dict_combined = dict( counter ) # retrieve a combined dictionary
        '''remove temporary files '''
        for path_file in l_path_file :
            os.remove( path_file )
        ''' save dictionary as a file '''
        pd.Series( dict_combined ).to_csv( f'{path_folder_mtx_10x_input}{name_dict}.tsv.gz', sep = '\t', header = None )
    return dict_combined # returns a combined dictionary
def __MTX_10X_Summarize_Counts__summarize_counts_for_each_mtx_10x__( path_file_input, path_folder_mtx_10x_input ) :
    '''
    internal function for MTX_10X_Summarize_Count
    # 2022-04-28 04:26:57 
    '''
    ''' survey the metrics '''
    ''' for each split mtx file, count number of umi and n_feature for each cells or the number of cells for each feature '''
    ''' initialize the dictionaries that will be handled by the current function '''
    dict_id_column_to_count = dict( )
    dict_id_column_to_n_features = dict( )
    dict_id_row_to_count = dict( )
    dict_id_row_to_n_cells = dict( )
    dict_id_row_to_log_transformed_count = dict( )
    
    global dict_name_set_feature_to_set_id_row # use global read-only object
    dict_name_set_feature_to_dict_id_column_to_count = dict( ( name_set_feature, dict( ) ) for name_set_feature in dict_name_set_feature_to_set_id_row ) # initialize 'dict_name_set_feature_to_dict_id_column_to_count'
    for path_file_input_mtx in pd.read_csv( path_file_input, sep = '\t', header = None ).values.ravel( ) :
        with gzip.open( path_file_input_mtx, 'rb' ) as file :
            ''' read the first line '''
            line = file.readline( ).decode( ) 
            ''' if the first line of the file contains a comment line, read all comment lines and a description line following the comments. '''
            if len( line ) > 0 and line[ 0 ] == '%' :
                # read comment and the description line
                while True :
                    if line[ 0 ] != '%' :
                        break
                    line = file.readline( ).decode( ) # read the next line
                # process the description line
                int_num_rows, int_num_columns, int_num_entries = tuple( int( e ) for e in line.strip( ).split( ) ) # retrieve the number of rows, number of columns and number of entries
                line = file.readline( ).decode( ) # read the next line
            ''' process entries'''
            while True :
                if len( line ) == 0 :
                    break
                ''' parse a record, and update metrics '''
                id_row, id_column, int_value = tuple( int( e ) for e in line.strip( ).split( ) ) # parse a record of a matrix-market format file
                ''' 1-based > 0-based coordinates '''
                id_row -= 1
                id_column -= 1
                ''' update umi count for each cell '''
                if id_column not in dict_id_column_to_count :
                    dict_id_column_to_count[ id_column ] = 0
                dict_id_column_to_count[ id_column ] += int_value
                ''' update umi count of specific set of features for each cell '''
                for name_set_feature in dict_name_set_feature_to_dict_id_column_to_count :
                    if id_row in dict_name_set_feature_to_set_id_row[ name_set_feature ] :
                        if id_column not in dict_name_set_feature_to_dict_id_column_to_count[ name_set_feature ] :
                            dict_name_set_feature_to_dict_id_column_to_count[ name_set_feature ][ id_column ] = 0
                        dict_name_set_feature_to_dict_id_column_to_count[ name_set_feature ][ id_column ] += int_value
                ''' update n_features for each cell '''
                if id_column not in dict_id_column_to_n_features :
                    dict_id_column_to_n_features[ id_column ] = 0
                dict_id_column_to_n_features[ id_column ] += 1
                ''' update umi count for each feature '''
                if id_row not in dict_id_row_to_count :
                    dict_id_row_to_count[ id_row ] = 0
                dict_id_row_to_count[ id_row ] += int_value
                ''' update n_cells for each feature '''
                if id_row not in dict_id_row_to_n_cells :
                    dict_id_row_to_n_cells[ id_row ] = 0
                dict_id_row_to_n_cells[ id_row ] += 1
                ''' update log transformed counts, calculated by 'X_new = log_10(X_old + 1)', for each feature '''
                if id_row not in dict_id_row_to_log_transformed_count :
                    dict_id_row_to_log_transformed_count[ id_row ] = 0
                dict_id_row_to_log_transformed_count[ id_row ] += math.log10( int_value + 1 )
                
                ''' read the next line '''
                line = file.readline( ).decode( ) # binary > uncompressed string # read the next line
    
    # save collected count as tsv files
    str_uuid_process = UUID( ) # retrieve uuid of the current process
    pd.Series( dict_id_column_to_count ).to_csv( f'{path_folder_mtx_10x_input}dict_id_column_to_count.{str_uuid_process}.tsv.gz', sep = '\t', header = None )
    pd.Series( dict_id_column_to_n_features ).to_csv( f'{path_folder_mtx_10x_input}dict_id_column_to_n_features.{str_uuid_process}.tsv.gz', sep = '\t', header = None )
    pd.Series( dict_id_row_to_count ).to_csv( f'{path_folder_mtx_10x_input}dict_id_row_to_count.{str_uuid_process}.tsv.gz', sep = '\t', header = None )
    pd.Series( dict_id_row_to_n_cells ).to_csv( f'{path_folder_mtx_10x_input}dict_id_row_to_n_cells.{str_uuid_process}.tsv.gz', sep = '\t', header = None )
    pd.Series( dict_id_row_to_log_transformed_count ).to_csv( f'{path_folder_mtx_10x_input}dict_id_row_to_log_transformed_count.{str_uuid_process}.tsv.gz', sep = '\t', header = None )
    
    # save collected counts as tsv files for 'dict_name_set_feature_to_dict_id_column_to_count'
    for name_set_feature in dict_name_set_feature_to_dict_id_column_to_count :
        pd.Series( dict_name_set_feature_to_dict_id_column_to_count[ name_set_feature ] ).to_csv( f'{path_folder_mtx_10x_input}{name_set_feature}.dict_id_column_to_count.{str_uuid_process}.tsv.gz', sep = '\t', header = None )
def MTX_10X_Summarize_Counts( path_folder_mtx_10x_input, verbose = False, int_num_threads = 15, flag_split_mtx = True, int_max_num_entries_for_chunk = 10000000, dict_name_set_feature_to_l_id_feature = dict( ), flag_split_mtx_again = False ) :
    """ # 2022-04-28 06:53:45 
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

    ''' the name of the dictionaries handled by this function (basic) '''
    l_name_dict = [ 'dict_id_column_to_count', 'dict_id_column_to_n_features', 'dict_id_row_to_count', 'dict_id_row_to_n_cells', 'dict_id_row_to_log_transformed_count' ]

    ''' handle inputs '''
    if path_folder_mtx_10x_input[ -1 ] != '/' :
        path_folder_mtx_10x_input += '/'

    # define flag and check whether the flag exists
    path_file_flag = f"{path_folder_mtx_10x_input}counts_summarized.flag"
    if not os.path.exists( path_file_flag ) :
        # define input file directories
        path_file_input_bc = f'{path_folder_mtx_10x_input}barcodes.tsv.gz'
        path_file_input_feature = f'{path_folder_mtx_10x_input}features.tsv.gz'
        path_file_input_mtx = f'{path_folder_mtx_10x_input}matrix.mtx.gz'

        # check whether all required files are present
        if sum( list( not os.path.exists( path_folder ) for path_folder in [ path_file_input_bc, path_file_input_feature, path_file_input_mtx ] ) ) :
            if verbose :
                print( f'required file(s) is not present at {path_folder_mtx_10x}' )

        ''' split input mtx file into multiple files '''
        l_path_file_mtx_10x = MTX_10X_Split( path_folder_mtx_10x_input, int_max_num_entries_for_chunk = int_max_num_entries_for_chunk, flag_split_mtx = flag_split_mtx, flag_split_mtx_again = flag_split_mtx_again )

        ''' prepare 'dict_name_set_feature_to_set_id_row' for summarizing total counts for given sets of features '''
        global dict_name_set_feature_to_set_id_row
        dict_name_set_feature_to_set_id_row = dict( ) # initialize 'dict_name_set_feature_to_set_id_row'
        if dict_name_set_feature_to_l_id_feature is not None :
            arr_id_feature = pd.read_csv( path_file_input_feature, sep = '\t', usecols = [ 0 ], header = None ).values.ravel( ) # retrieve array of id_features
            dict_id_feature_to_id_row = dict( ( e, i ) for i, e in enumerate( arr_id_feature ) ) # retrieve id_feature -> id_row mapping

            ''' handle presets for 'dict_name_set_feature_to_l_id_feature' '''
            if isinstance( dict_name_set_feature_to_l_id_feature, str ) :    
                str_preset = dict_name_set_feature_to_l_id_feature # retrieve preset
                dict_name_set_feature_to_l_id_feature = dict( ) # initialize the dictionary
                if str_preset in [ 'multiome', 'atac' ] :
                    if str_preset == 'multiome' :
                        arr_id_feature_atac = Search_list_of_strings_with_multiple_query( arr_id_feature, '|mode=atac' )
                        dict_name_set_feature_to_l_id_feature[ 'atac_all' ] = arr_id_feature_atac
                    elif str_preset == 'atac' :
                        arr_id_feature_atac = arr_id_feature
                    # add sets of promoter and gene_body features
                    arr_id_feature_atac_promoter_and_gene_body = Search_list_of_strings_with_multiple_query( arr_id_feature_atac, '-genomic_region|', '-repeatmasker_ucsc|', '-regulatory_element|' )
                    arr_id_feature_atac_promoter = Search_list_of_strings_with_multiple_query( arr_id_feature_atac_promoter_and_gene_body, 'promoter|' )
                    dict_name_set_feature_to_l_id_feature[ 'atac_promoter_and_gene_body' ] = arr_id_feature_atac_promoter_and_gene_body    
                    dict_name_set_feature_to_l_id_feature[ 'atac_promoter' ] = arr_id_feature_atac_promoter

            # make sure that 'name_set_feature' does not contains characters incompatible with linux file path
            for name_set_feature in dict_name_set_feature_to_l_id_feature :
                assert not( '/' in name_set_feature or '\n' in name_set_feature )

            dict_name_set_feature_to_set_id_row = dict( ( name_set_feature, set( dict_id_feature_to_id_row[ id_feature ] for id_feature in dict_name_set_feature_to_l_id_feature[ name_set_feature ] ) ) for name_set_feature in dict_name_set_feature_to_l_id_feature )
            # PICKLE_Write( f"{path_folder_mtx_10x_input}dict_name_set_feature_to_set_id_row.binary.pickle", dict_name_set_feature_to_set_id_row ) # write the dictionary as a pickle

        ''' summarize each split mtx file '''
        Multiprocessing( l_path_file_mtx_10x, __MTX_10X_Summarize_Counts__summarize_counts_for_each_mtx_10x__, n_threads = int_num_threads, global_arguments = [ path_folder_mtx_10x_input ] )

        ''' combine summarized results '''
        # update the list of the names of dictionaries
        l_name_dict += list( f"{name_set_feature}.dict_id_column_to_count" for name_set_feature in GLOB_Retrive_Strings_in_Wildcards( f'{path_folder_mtx_10x_input}*.dict_id_column_to_count.*.tsv.gz' ).wildcard_0.unique( ) ) 

        dict_dict = dict( )
        for name_dict in l_name_dict :
            dict_dict[ name_dict ] = __Combine_Dictionaries__( path_folder_mtx_10x_input, name_dict )
        # write the flag
        with open( path_file_flag, 'w' ) as newfile :
            newfile.write( 'completed at ' + TIME_GET_timestamp( True ) )
    else :
        ''' read summarized results '''
        # update the list of the names of dictionaries
        l_name_dict += list( f"{name_set_feature}.dict_id_column_to_count" for name_set_feature in GLOB_Retrive_Strings_in_Wildcards( f'{path_folder_mtx_10x_input}*.dict_id_column_to_count.tsv.gz' ).wildcard_0.unique( ) ) 

        dict_dict = dict( )
        for name_dict in l_name_dict :
            try :
                dict_dict[ name_dict ] = pd.read_csv( f'{path_folder_mtx_10x_input}{name_dict}.tsv.gz', sep = '\t', header = None, index_col = 0 ).iloc[ :, 0 ].to_dict( )
            except pd.errors.EmptyDataError : # handle when the current dictionary is empty
                dict_dict[ name_dict ] = dict( )

    # return summarized metrics
    return dict_dict
def MTX_10X_Retrieve_number_of_rows_columns_and_records( path_folder_mtx_10x_input ) :
    """ # 2022-03-05 19:58:32 
    Retrieve the number of rows, columns, and entries from the matrix with the matrix market format 
    
    'path_folder_mtx_10x_input' : a folder mtx file resides or path to mtx file
    
    Returns:
    int_num_rows, int_num_columns, int_num_entries
    """
    ''' handle inputs '''
    if path_folder_mtx_10x_input[ -3 : ].lower( ) == '.gz' : # when a path to mtx file was given
        path_file_input_mtx = path_folder_mtx_10x_input
    else : # when a folder where mtx file resides was given
        if path_folder_mtx_10x_input[ -1 ] != '/' :
            path_folder_mtx_10x_input += '/'

        # define input file directories
        path_file_input_mtx = f'{path_folder_mtx_10x_input}matrix.mtx.gz'
    
        # check whether all required files are present
        if sum( list( not os.path.exists( path_folder ) for path_folder in [ path_file_input_mtx ] ) ) :
            return None
    
    # read the input matrix
    with gzip.open( path_file_input_mtx, 'rb' ) as file :
        ''' read the first line '''
        line = file.readline( ).decode( ).strip( )
        ''' if the first line of the file contains a comment line, read all comment lines and a description line following the comments. '''
        if len( line ) > 0 and line[ 0 ] == '%' :
            # read comment and the description line
            while True :
                if line[ 0 ] != '%' :
                    break
                line = file.readline( ).decode( ).strip( ) # read the next line
            # process the description line
            int_num_rows, int_num_columns, int_num_entries = tuple( int( e ) for e in line.strip( ).split( ) ) # retrieve the number of rows, number of columns and number of entries
        else :
            ''' the first line does not contain a comment, assumes it contains a description line '''
            int_num_rows, int_num_columns, int_num_entries = tuple( int( e ) for e in line.strip( ).split( ) ) # retrieve the number of rows, number of columns and number of entries
    return int_num_rows, int_num_columns, int_num_entries
dict_id_column_to_count, dict_id_row_to_avg_count, dict_id_row_to_avg_log_transformed_count, dict_id_row_to_avg_normalized_count, dict_id_row_to_avg_log_transformed_normalized_count = dict( ), dict( ), dict( ), dict( ), dict( ) # global variables # total UMI counts for each cell, average feature counts for each feature
def __MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr__first_pass__( path_file_input, path_folder_mtx_10x_input, int_target_sum ) :
    ''' # 2022-03-06 01:21:07 
    internal function for MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr
    '''
    global dict_id_column_to_count, dict_id_row_to_avg_count, dict_id_row_to_avg_log_transformed_count # use data in read-only global variables 
    ''' initialize the dictionaries that will be handled by the current function '''
    dict_id_row_to_deviation_from_mean_count = dict( )
    dict_id_row_to_deviation_from_mean_log_transformed_count = dict( )
    dict_id_row_to_normalized_count = dict( )
    dict_id_row_to_log_transformed_normalized_count = dict( )
    
    for path_file_input_mtx in pd.read_csv( path_file_input, sep = '\t', header = None ).values.ravel( ) :
        with gzip.open( path_file_input_mtx, 'rb' ) as file :
            ''' read the first line '''
            line = file.readline( ).decode( ) 
            ''' if the first line of the file contains a comment line, read all comment lines and a description line following the comments. '''
            if len( line ) > 0 and line[ 0 ] == '%' :
                # read comment and the description line
                while True :
                    if line[ 0 ] != '%' :
                        break
                    line = file.readline( ).decode( ) # read the next line
                # process the description line
                int_num_rows, int_num_columns, int_num_entries = tuple( int( e ) for e in line.strip( ).split( ) ) # retrieve the number of rows, number of columns and number of entries
                line = file.readline( ).decode( ) # read the next line
            ''' process entries'''
            while True :
                if len( line ) == 0 :
                    break
                ''' parse a record, and update metrics '''
                id_row, id_column, int_value = tuple( int( e ) for e in line.strip( ).split( ) ) # parse a record of a matrix-market format file
                ''' 1-based > 0-based coordinates '''
                id_row -= 1
                id_column -= 1
                
                ''' update deviation from mean umi count for count of each feature '''
                if id_row not in dict_id_row_to_deviation_from_mean_count :
                    dict_id_row_to_deviation_from_mean_count[ id_row ] = 0
                dict_id_row_to_deviation_from_mean_count[ id_row ] += ( int_value - dict_id_row_to_avg_count[ id_row ] ) ** 2
                ''' update deviation from mean log transformed umi count for log_transformed count of each feature '''
                if id_row not in dict_id_row_to_deviation_from_mean_log_transformed_count :
                    dict_id_row_to_deviation_from_mean_log_transformed_count[ id_row ] = 0
                dict_id_row_to_deviation_from_mean_log_transformed_count[ id_row ] += ( math.log10( int_value + 1 ) - dict_id_row_to_avg_log_transformed_count[ id_row ] ) ** 2
                ''' calculate normalized target sum '''
                int_value_normalized = int_value / dict_id_column_to_count[ id_column ] * int_target_sum 
                ''' update normalized counts, calculated by 'X_new = X_old / total_umi * int_target_sum', for each feature '''
                if id_row not in dict_id_row_to_normalized_count :
                    dict_id_row_to_normalized_count[ id_row ] = 0
                dict_id_row_to_normalized_count[ id_row ] += int_value_normalized
                ''' update log transformed normalized counts, calculated by 'X_new = log_10(X_old / total_umi * int_target_sum + 1)', for each feature '''
                if id_row not in dict_id_row_to_log_transformed_normalized_count :
                    dict_id_row_to_log_transformed_normalized_count[ id_row ] = 0
                dict_id_row_to_log_transformed_normalized_count[ id_row ] += math.log10( int_value_normalized + 1 ) 
                
                line = file.readline( ).decode( ) # binary > uncompressed string # read the next line
    
    # save collected count as tsv files
    str_uuid_process = UUID( ) # retrieve uuid of the current process
    pd.Series( dict_id_row_to_deviation_from_mean_count ).to_csv( f'{path_folder_mtx_10x_input}dict_id_row_to_deviation_from_mean_count.{str_uuid_process}.tsv.gz', sep = '\t', header = None )
    pd.Series( dict_id_row_to_deviation_from_mean_log_transformed_count ).to_csv( f'{path_folder_mtx_10x_input}dict_id_row_to_deviation_from_mean_log_transformed_count.{str_uuid_process}.tsv.gz', sep = '\t', header = None )
    pd.Series( dict_id_row_to_normalized_count ).to_csv( f'{path_folder_mtx_10x_input}dict_id_row_to_normalized_count.{str_uuid_process}.tsv.gz', sep = '\t', header = None )
    pd.Series( dict_id_row_to_log_transformed_normalized_count ).to_csv( f'{path_folder_mtx_10x_input}dict_id_row_to_log_transformed_normalized_count.{str_uuid_process}.tsv.gz', sep = '\t', header = None )
def __MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr__second_pass__( path_file_input, path_folder_mtx_10x_input, int_target_sum ) :
    ''' # 2022-03-06 01:21:14 
    internal function for MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr
    '''
    global dict_id_column_to_count, dict_id_row_to_avg_normalized_count, dict_id_row_to_avg_log_transformed_normalized_count # use data in read-only global variables 
    ''' initialize the dictionaries that will be handled by the current function '''
    dict_id_row_to_deviation_from_mean_normalized_count = dict( )
    dict_id_row_to_deviation_from_mean_log_transformed_normalized_count = dict( )
    
    for path_file_input_mtx in pd.read_csv( path_file_input, sep = '\t', header = None ).values.ravel( ) :
        with gzip.open( path_file_input_mtx, 'rb' ) as file :
            ''' read the first line '''
            line = file.readline( ).decode( ) 
            ''' if the first line of the file contains a comment line, read all comment lines and a description line following the comments. '''
            if len( line ) > 0 and line[ 0 ] == '%' :
                # read comment and the description line
                while True :
                    if line[ 0 ] != '%' :
                        break
                    line = file.readline( ).decode( ) # read the next line
                # process the description line
                int_num_rows, int_num_columns, int_num_entries = tuple( int( e ) for e in line.strip( ).split( ) ) # retrieve the number of rows, number of columns and number of entries
                line = file.readline( ).decode( ) # read the next line
            ''' process entries'''
            while True :
                if len( line ) == 0 :
                    break
                ''' parse a record, and update metrics '''
                id_row, id_column, int_value = tuple( int( e ) for e in line.strip( ).split( ) ) # parse a record of a matrix-market format file
                ''' 1-based > 0-based coordinates '''
                id_row -= 1
                id_column -= 1
                
                ''' calculate normalized target sum '''
                int_value_normalized = int_value / dict_id_column_to_count[ id_column ] * int_target_sum 
                ''' update deviation from mean normalized umi counts, calculated by 'X_new = X_old / total_umi * int_target_sum', for each feature '''
                if id_row not in dict_id_row_to_deviation_from_mean_normalized_count :
                    dict_id_row_to_deviation_from_mean_normalized_count[ id_row ] = 0
                dict_id_row_to_deviation_from_mean_normalized_count[ id_row ] += ( int_value_normalized - dict_id_row_to_avg_normalized_count[ id_row ] ) ** 2
                ''' update deviation from mean log transformed normalized umi counts, calculated by 'X_new = log_10(X_old / total_umi * int_target_sum + 1)', for each feature '''
                if id_row not in dict_id_row_to_deviation_from_mean_log_transformed_normalized_count :
                    dict_id_row_to_deviation_from_mean_log_transformed_normalized_count[ id_row ] = 0
                dict_id_row_to_deviation_from_mean_log_transformed_normalized_count[ id_row ] += ( math.log10( int_value_normalized + 1 ) - dict_id_row_to_avg_log_transformed_normalized_count[ id_row ] ) ** 2
                
                line = file.readline( ).decode( ) # binary > uncompressed string # read the next line
    
    # save collected count as tsv files
    str_uuid_process = UUID( ) # retrieve uuid of the current process
    pd.Series( dict_id_row_to_deviation_from_mean_normalized_count ).to_csv( f'{path_folder_mtx_10x_input}dict_id_row_to_deviation_from_mean_normalized_count.{str_uuid_process}.tsv.gz', sep = '\t', header = None )
    pd.Series( dict_id_row_to_deviation_from_mean_log_transformed_normalized_count ).to_csv( f'{path_folder_mtx_10x_input}dict_id_row_to_deviation_from_mean_log_transformed_normalized_count.{str_uuid_process}.tsv.gz', sep = '\t', header = None )
def MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr( path_folder_mtx_10x_input, int_target_sum = 10000, verbose = False, int_num_threads = 15, flag_split_mtx = True, int_max_num_entries_for_chunk = 10000000 ) :
    """ # 2022-02-23 22:54:35 
    Calculate average log transformed normalized expr
    (1) UMI and Feature counts for cells, and
    (2) Cell counts for features,
    and save these metrics as TSV files
    
    Arguments:
    'int_target_sum' : the target count for the total UMI count for each cell. The counts will normalized to meet the target sum.
    
    Returns:
    dict_id_column_to_count, dict_id_column_to_n_features, dict_id_row_to_count, dict_id_row_to_n_cells, dict_id_row_to_log_transformed_count
    """

    ''' handle inputs '''
    if path_folder_mtx_10x_input[ -1 ] != '/' :
        path_folder_mtx_10x_input += '/'

    # define flag and check whether the flag exists
    path_file_flag = f"{path_folder_mtx_10x_input}avg_expr_normalized_summarized.int_target_sum__{int_target_sum}.flag"
    if not os.path.exists( path_file_flag ) :
        # define input file directories
        path_file_input_bc = f'{path_folder_mtx_10x_input}barcodes.tsv.gz'
        path_file_input_feature = f'{path_folder_mtx_10x_input}features.tsv.gz'
        path_file_input_mtx = f'{path_folder_mtx_10x_input}matrix.mtx.gz'

        # check whether all required files are present
        if sum( list( not os.path.exists( path_folder ) for path_folder in [ path_file_input_bc, path_file_input_feature, path_file_input_mtx ] ) ) :
            if verbose :
                print( f'required file(s) is not present at {path_folder_mtx_10x}' )

        ''' split input mtx file into multiple files '''
        l_path_file_mtx_10x = MTX_10X_Split( path_folder_mtx_10x_input, int_max_num_entries_for_chunk = int_max_num_entries_for_chunk, flag_split_mtx = flag_split_mtx )

        ''' retrieve number of cells, features, and entries from the matrix file '''
        int_num_cells, int_num_features, int_num_entries = MTX_10X_Retrieve_number_of_rows_columns_and_records( path_folder_mtx_10x_input )
        
        ''' summarizes counts '''
        global dict_id_column_to_count, dict_id_row_to_avg_count, dict_id_row_to_avg_log_transformed_count, dict_id_row_to_avg_normalized_count, dict_id_row_to_avg_log_transformed_normalized_count # use global variable
        dict_data = MTX_10X_Summarize_Counts( path_folder_mtx_10x_input, verbose = verbose, int_num_threads = int_num_threads, flag_split_mtx = flag_split_mtx, int_max_num_entries_for_chunk = int_max_num_entries_for_chunk )
        dict_id_column_to_count, dict_id_column_to_n_features, dict_id_row_to_count, dict_id_row_to_n_cells, dict_id_row_to_log_transformed_count = dict_data[ 'dict_id_column_to_count' ], dict_data[ 'dict_id_column_to_n_features' ], dict_data[ 'dict_id_row_to_count' ], dict_data[ 'dict_id_row_to_n_cells' ], dict_data[ 'dict_id_row_to_log_transformed_count' ] # parse 'dict_data'

        """ first pass """
        # calculate mean counts
        dict_id_row_to_avg_count = ( pd.Series( dict_id_row_to_count ) / int_num_cells ).to_dict( ) # calculate average expression of each feature
        dict_id_row_to_avg_log_transformed_count = ( pd.Series( dict_id_row_to_log_transformed_count ) / int_num_cells ).to_dict( ) # calculate average log-transformed expression of each feature
        
        ''' calculated average log2 transformed normalized expr for each split mtx file '''
        Multiprocessing( l_path_file_mtx_10x, __MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr__first_pass__, n_threads = int_num_threads, global_arguments = [ path_folder_mtx_10x_input, int_target_sum ] )

        l_name_dict_first_pass = [ 'dict_id_row_to_deviation_from_mean_count', 'dict_id_row_to_deviation_from_mean_log_transformed_count', 'dict_id_row_to_normalized_count', 'dict_id_row_to_log_transformed_normalized_count' ]
        
        ''' combine summarized results '''
        dict_dict = dict( )
        for name_dict in l_name_dict_first_pass :
            dict_dict[ name_dict ] = __Combine_Dictionaries__( path_folder_mtx_10x_input, name_dict )
            
        """ second pass """
        # calculate mean counts
        dict_id_row_to_avg_normalized_count = ( pd.Series( dict_dict[ 'dict_id_row_to_normalized_count' ] ) / int_num_cells ).to_dict( ) # calculate average expression of each feature
        dict_id_row_to_avg_log_transformed_normalized_count = ( pd.Series( dict_dict[ 'dict_id_row_to_log_transformed_normalized_count' ] ) / int_num_cells ).to_dict( ) # calculate average log-transformed expression of each feature
        
        ''' calculated average log2 transformed normalized expr for each split mtx file '''
        Multiprocessing( l_path_file_mtx_10x, __MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr__second_pass__, n_threads = int_num_threads, global_arguments = [ path_folder_mtx_10x_input, int_target_sum ] )

        l_name_dict_second_pass = [ 'dict_id_row_to_deviation_from_mean_normalized_count', 'dict_id_row_to_deviation_from_mean_log_transformed_normalized_count' ]
        
        ''' combine summarized results '''
        for name_dict in l_name_dict_second_pass :
            dict_dict[ name_dict ] = __Combine_Dictionaries__( path_folder_mtx_10x_input, name_dict )
            
        ''' compose a dataframe containing the summary about the features '''
        df_summary = pd.DataFrame( { 
            'n_cells' : pd.Series( dict_id_row_to_n_cells ),
            'variance_of_count' : pd.Series( dict_dict[ 'dict_id_row_to_deviation_from_mean_count' ] ) / ( int_num_cells - 1 ),
            'variance_of_log_transformed_count' : pd.Series( dict_dict[ 'dict_id_row_to_deviation_from_mean_log_transformed_count' ] ) / ( int_num_cells - 1 ),
            'variance_of_normalized_count' : pd.Series( dict_dict[ 'dict_id_row_to_deviation_from_mean_normalized_count' ] ) / ( int_num_cells - 1 ),
            'variance_of_log_transformed_normalized_count' : pd.Series( dict_dict[ 'dict_id_row_to_deviation_from_mean_log_transformed_normalized_count' ] ) / ( int_num_cells - 1 ),
            'mean_count' : pd.Series( dict_id_row_to_avg_count ),
            'mean_log_transformed_count' : pd.Series( dict_id_row_to_avg_log_transformed_count ),
            'mean_normalized_count' : pd.Series( dict_id_row_to_avg_normalized_count ),
            'mean_log_transformed_normalized_count' : pd.Series( dict_id_row_to_avg_log_transformed_normalized_count ),
        } )
        # read a dataframe containing features
        df_feature = pd.read_csv( path_file_input_feature, sep = '\t', header = None )
        df_feature.columns = [ 'id_feature', 'feature', 'feature_type' ]
        
        df_summary = df_summary.join( df_feature, how = 'left' ) # add df_feature to the df_summary
        df_summary.index.name = 'id_row' 
        df_summary.reset_index( drop = False, inplace = True ) # retrieve id_row as a column
        df_summary.to_csv( f'{path_folder_mtx_10x_input}statistical_summary_of_features.int_target_sum__{int_target_sum}.tsv.gz', sep = '\t', index = False ) # save statistical summary as a text file
        
        # write the flag
        with open( path_file_flag, 'w' ) as newfile :
            newfile.write( 'completed at ' + TIME_GET_timestamp( True ) )
    else :
        ''' if 'MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr' function has been already run on the current folder, read the previously saved result, and return the summary dataframe '''
        df_summary = pd.read_csv( f'{path_folder_mtx_10x_input}statistical_summary_of_features.int_target_sum__{int_target_sum}.tsv.gz', sep = '\t' ) # save statistical summary as a text file
    return df_summary
dict_id_column_previous_to_id_column_current, dict_id_row_previous_to_id_row_current = dict( ), dict( )
def __MTX_10X_Filter__filter_mtx_10x__( path_file_input, path_folder_mtx_10x_output ) :
    """ # 2022-02-22 02:06:03 
    __MTX_10X_Filter__filter_mtx_10x__
    
    Returns:
    int_n_entries = total number of entries written by the current process after filtering
    """
    int_n_entries = 0 # total number of entries written by the current process after filtering
#     dict_id_column_previous_to_id_column_current = PICKLE_Read( f'{path_folder_mtx_10x_output}dict_id_column_previous_to_id_column_current.pickle' ) # retrieve id_feature to index_feature mapping 
#     dict_id_row_previous_to_id_row_current = PICKLE_Read( f'{path_folder_mtx_10x_output}dict_id_row_previous_to_id_row_current.pickle' ) # retrieve id_feature to index_feature mapping 
    """ write a filtered matrix.mtx.gz for each split mtx file """
    for path_file_mtx_10x, index_mtx_10x in pd.read_csv( path_file_input, sep = '\t' ).values :
        # directly write matrix.mtx.gz file without using an external dependency
        with gzip.open( f"{path_folder_mtx_10x_output}matrix.mtx.gz.{index_mtx_10x}.gz", 'wb' ) as newfile :
            with gzip.open( path_file_mtx_10x, 'rb' ) as file : 
                ''' read the first line '''
                line = file.readline( ).decode( ) 
                ''' if the first line of the file contains a comment line, read all comment lines and a description line following the comments. '''
                if len( line ) > 0 and line[ 0 ] == '%' :
                    # read comment and the description line
                    while True :
                        if line[ 0 ] != '%' :
                            break
                        line = file.readline( ).decode( ) # read the next line
                    # process the description line
                    int_num_rows, int_num_columns, int_num_entries = tuple( int( e ) for e in line.strip( ).split( ) ) # retrieve the number of rows, number of columns and number of entries
                    line = file.readline( ).decode( ) # read the next line
                ''' process entries'''
                while True :
                    if len( line ) == 0 :
                        break
                    id_row, id_column, int_value = tuple( map( int, line.strip( ).split( ) ) ) # parse each entry of the current matrix 
                    ''' 1-based > 0-based coordinates '''
                    id_row -= 1
                    id_column -= 1
                    ''' write a record to the new matrix file only when both id_row and id_column belongs to filtered id_rows and id_columns '''
                    if id_row in dict_id_row_previous_to_id_row_current and id_column in dict_id_column_previous_to_id_column_current :
                        newfile.write( ( ' '.join( tuple( map( str, [ dict_id_row_previous_to_id_row_current[ id_row ] + 1, dict_id_column_previous_to_id_column_current[ id_column ] + 1, int_value ] ) ) ) + '\n' ).encode( ) ) # map id_row and id_column of the previous matrix to those of the filtered matrix (new matrix) # 0-based > 1-based coordinates
                        int_n_entries += 1 # update the total number of entries written by the current process
                    line = file.readline( ).decode( ) # read the next line
    return int_n_entries # returns the total number of entries written by the current process
def MTX_10X_Filter( path_folder_mtx_10x_input, path_folder_mtx_10x_output, min_counts = None, min_features = None, min_cells = None, l_features = None, l_cells = None, verbose = False, function_for_adjusting_thresholds = None, int_num_threads = 15, flag_split_mtx = True, int_max_num_entries_for_chunk = 10000000 ) :
    ''' # 2022-08-20 10:23:28 
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
    '''

    ''' handle inputs '''
    if path_folder_mtx_10x_input[ -1 ] != '/' :
        path_folder_mtx_10x_input += '/'
    if path_folder_mtx_10x_output[ -1 ] != '/' :
        path_folder_mtx_10x_output += '/'
    if ( ( min_counts is not None ) or ( min_features is not None ) ) and ( min_cells is not None ) : # check whether thresholds for both cells and features were given (thresdholds for either cells or features can be given at a time)
        if verbose :
            print( '[MTX_10X_Filter] (error) no threshold is given or more thresholds for both cells and features are given. (Thresdholds for either cells or features can be given at a time.)' )
        return -1
    # create an output folder
    os.makedirs( path_folder_mtx_10x_output, exist_ok = True )

    # define input file directories
    path_file_input_bc = f'{path_folder_mtx_10x_input}barcodes.tsv.gz'
    path_file_input_feature = f'{path_folder_mtx_10x_input}features.tsv.gz'
    path_file_input_mtx = f'{path_folder_mtx_10x_input}matrix.mtx.gz'

    # check whether all required files are present
    if sum( list( not os.path.exists( path_folder ) for path_folder in [ path_file_input_bc, path_file_input_feature, path_file_input_mtx ] ) ) :
        if verbose :
            print( f'required file(s) is not present at {path_folder_mtx_10x}' )

    ''' read barcode and feature information '''
    df_bc = pd.read_csv( path_file_input_bc, sep = '\t', header = None )
    df_bc.columns = [ 'barcode' ]
    df_feature = pd.read_csv( path_file_input_feature, sep = '\t', header = None )
    df_feature.columns = [ 'id_feature', 'feature', 'feature_type' ]

    ''' split input mtx file into multiple files '''
    l_path_file_mtx_10x = MTX_10X_Split( path_folder_mtx_10x_input, int_max_num_entries_for_chunk = int_max_num_entries_for_chunk, flag_split_mtx = flag_split_mtx )
    
    ''' summarizes counts '''
    dict_data = MTX_10X_Summarize_Counts( path_folder_mtx_10x_input, verbose = verbose, int_num_threads = int_num_threads, flag_split_mtx = flag_split_mtx, int_max_num_entries_for_chunk = int_max_num_entries_for_chunk )
    dict_id_column_to_count, dict_id_column_to_n_features, dict_id_row_to_count, dict_id_row_to_n_cells, dict_id_row_to_log_transformed_count = dict_data[ 'dict_id_column_to_count' ], dict_data[ 'dict_id_column_to_n_features' ], dict_data[ 'dict_id_row_to_count' ], dict_data[ 'dict_id_row_to_n_cells' ], dict_data[ 'dict_id_row_to_log_transformed_count' ] # parse 'dict_data'
    
    ''' adjust thresholds based on the summarized metrices (if a function has been given) '''
    if function_for_adjusting_thresholds is not None :
        min_counts, min_features, min_cells = function_for_adjusting_thresholds( path_folder_mtx_10x_input, min_counts, min_features, min_cells )
    
    ''' filter row or column that do not satisfy the given thresholds '''
    if min_counts is not None :
        dict_id_column_to_count = dict( ( k, dict_id_column_to_count[ k ] ) for k in dict_id_column_to_count if dict_id_column_to_count[ k ] >= min_counts ) 
    if min_features is not None :
        dict_id_column_to_n_features = dict( ( k, dict_id_column_to_n_features[ k ] ) for k in dict_id_column_to_n_features if dict_id_column_to_n_features[ k ] >= min_features )
    if min_cells is not None :
        dict_id_row_to_n_cells = dict( ( k, dict_id_row_to_n_cells[ k ] ) for k in dict_id_row_to_n_cells if dict_id_row_to_n_cells[ k ] >= min_cells )

    ''' retrieve id_row and id_column that satisfy the given thresholds '''    
    set_id_column = set( dict_id_column_to_count ).intersection( set( dict_id_column_to_n_features ) )
    set_id_row = set( dict_id_row_to_n_cells )
    
    ''' exclude cells and features not present in the input lists (if the lists were given)  '''
    if l_cells is not None :        
        dict_barcode_to_id_column = dict( ( barcode, id_column ) for id_column, barcode in enumerate( df_bc.barcode.values ) )
        set_id_column = set_id_column.intersection( set( dict_barcode_to_id_column[ barcode ] for barcode in set( l_cells ) if barcode in dict_barcode_to_id_column ) )
        del dict_barcode_to_id_column
    if l_features is not None :
        dict_id_feature_to_id_row = dict( ( id_feature, id_row ) for id_row, id_feature in enumerate( df_feature.id_feature.values ) )
        set_id_row = set_id_row.intersection( set( dict_id_feature_to_id_row[ id_feature ] for id_feature in set( l_features ) if id_feature in dict_id_feature_to_id_row ) )
        del dict_id_feature_to_id_row

    ''' report the number of cells or features that will be filtered out '''
    if verbose :
        int_n_bc_filtered = len( df_bc ) - len( set_id_column )
        if int_n_bc_filtered > 0 :
            print( f"{int_n_bc_filtered}/{len( df_bc )} barcodes will be filtered out" )
        int_n_feature_filtered = len( df_feature ) - len( set_id_row )
        if int_n_feature_filtered > 0 :
            print( f"{int_n_feature_filtered}/{len( df_feature )} features will be filtered out" )

    """ retrieve a mapping between previous id_column to current id_column """
    global dict_id_column_previous_to_id_column_current, dict_id_row_previous_to_id_row_current # use global variables for multiprocessing
    df_bc = df_bc.loc[ list( set_id_column ) ]
    df_bc.index.name = 'id_column_previous'
    df_bc.reset_index( drop = False, inplace = True )
    df_bc[ 'id_column_current' ] = np.arange( len( df_bc ) )
    dict_id_column_previous_to_id_column_current = df_bc.set_index( 'id_column_previous' ).id_column_current.to_dict( ) 
    PICKLE_Write( f'{path_folder_mtx_10x_output}dict_id_column_previous_to_id_column_current.pickle', dict_id_column_previous_to_id_column_current ) # save id_feature to index_feature mapping 
    """ retrieve a mapping between previous id_row to current id_row """
    df_feature = df_feature.loc[ list( set_id_row ) ]
    df_feature.index.name = 'id_row_previous'
    df_feature.reset_index( drop = False, inplace = True )
    df_feature[ 'id_row_current' ] = np.arange( len( df_feature ) )
    dict_id_row_previous_to_id_row_current = df_feature.set_index( 'id_row_previous' ).id_row_current.to_dict( ) 
    PICKLE_Write( f'{path_folder_mtx_10x_output}dict_id_row_previous_to_id_row_current.pickle', dict_id_row_previous_to_id_row_current ) # save id_feature to index_feature mapping 

    ''' save barcode file '''
    df_bc.to_csv( f"{path_folder_mtx_10x_output}barcodes.tsv.gz", columns = [ 'barcode' ], sep = '\t', index = False, header = False ) 
    del df_bc

    ''' save feature file '''
    df_feature[ [ 'id_feature', 'feature', 'feature_type' ] ].to_csv( f"{path_folder_mtx_10x_output}features.tsv.gz", sep = '\t', index = False, header = False ) # save as a file
    del df_feature

    """ write a filtered matrix.mtx.gz for each split mtx file using multiple processes and retrieve the total number of entries written by each process """
    # compose inputs for multiprocessing
    df_input = pd.DataFrame( { 'path_file_mtx_10x' : l_path_file_mtx_10x, 'index_mtx_10x' : np.arange( len( l_path_file_mtx_10x ) ) } )
    l_int_n_entries = Multiprocessing( df_input, __MTX_10X_Filter__filter_mtx_10x__, int_num_threads, global_arguments = [ path_folder_mtx_10x_output ] ) 
    # retrieve the total number of entries
    int_total_n_entries = sum( l_int_n_entries )
    
    """ combine parts and add the MTX file header to compose a combined mtx file """
    df_file = GLOB_Retrive_Strings_in_Wildcards( f"{path_folder_mtx_10x_output}matrix.mtx.gz.*.gz" )
    df_file.wildcard_0 = df_file.wildcard_0.astype( int )
    df_file.sort_values( 'wildcard_0', inplace = True )
    
    # write header 
    path_file_header = f"{path_folder_mtx_10x_output}matrix.mtx.header.txt.gz"
    with gzip.open( path_file_header, 'wb' ) as newfile :
        newfile.write( f"%%MatrixMarket matrix coordinate integer general\n%\n{len( dict_id_row_previous_to_id_row_current )} {len( dict_id_column_previous_to_id_column_current )} {int_total_n_entries}\n".encode( ) )
    OS_Run( [ 'cat', path_file_header ] + list( df_file.path.values ), path_file_stdout = f"{path_folder_mtx_10x_output}matrix.mtx.gz", stdout_binary = True, return_output = False ) # combine the output mtx files in the order # does not delete temporary files if 'flag_split_mtx' is True
    
    # write a flag indicating that the current output directory contains split mtx files
    with open( f"{path_folder_mtx_10x_output}matrix.mtx.gz.split.flag", 'w' ) as file :
        file.write( 'completed' )
def MTX_10X_Identify_Highly_Variable_Features( path_folder_mtx_10x_input, int_target_sum = 10000, verbose = False, int_num_threads = 15, flag_split_mtx = True, int_max_num_entries_for_chunk = 10000000 ) :
    ''' # 2022-03-16 17:18:44 
    calculate variance from log-transformed normalized counts using 'MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr' and rank features based on how each feature is variable compared to other features with similar means.
    Specifically, polynomial of degree 2 will be fitted to variance-mean relationship graph in order to captures the relationship between variance and mean. 
    
    'name_col_for_mean', 'name_col_for_variance' : name of columns of 'df_summary' returned by 'MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr' that will be used to infer highly variable features. By defaults, mean and variance of log-transformed normalized counts will be used.
    '''
    
    # calculate variance and means and load the result
    df_summary = MTX_10X_Calculate_Average_Log10_Transformed_Normalized_Expr( path_folder_mtx_10x_input, int_target_sum = int_target_sum, int_num_threads = int_num_threads, verbose = verbose, flag_split_mtx = flag_split_mtx )
    
    # calculate scores for identifying highly variable features for the selected set of count data types: [ 'log_transformed_normalized_count', 'log_transformed_count' ]
    for name_type in [ 'log_transformed_normalized_count', 'log_transformed_count' ] :
        name_col_for_mean, name_col_for_variance = f'mean_{name_type}', f'variance_of_{name_type}'
        # retrieve the relationship between mean and variance
        arr_mean = df_summary[ name_col_for_mean ].values
        arr_var = df_summary[ name_col_for_variance ].values
        mean_var_relationship_fit = np.polynomial.polynomial.Polynomial.fit( arr_mean, arr_var, 2 )

        # calculate the deviation from the estimated variance from the mean
        arr_ratio_of_variance_to_expected_variance_from_mean = np.zeros( len( df_summary ) )
        arr_diff_of_variance_to_expected_variance_from_mean = np.zeros( len( df_summary ) )
        for i in range( len( df_summary ) ) : # iterate list of means of the features
            var, mean = arr_var[ i ], arr_mean[ i ] # retrieve var and mean
            var_expected = mean_var_relationship_fit( mean ) # calculate expected variance from the mean
            if var_expected == 0 : # handle the case when the current expected variance is zero 
                arr_ratio_of_variance_to_expected_variance_from_mean[ i ] = 1
                arr_diff_of_variance_to_expected_variance_from_mean[ i ] = 0
            else :
                arr_ratio_of_variance_to_expected_variance_from_mean[ i ] = var / var_expected
                arr_diff_of_variance_to_expected_variance_from_mean[ i ] = var - var_expected

        df_summary[ f'float_ratio_of_variance_to_expected_variance_from_mean_from_{name_type}' ] = arr_ratio_of_variance_to_expected_variance_from_mean
        df_summary[ f'float_diff_of_variance_to_expected_variance_from_mean_{name_type}' ] = arr_diff_of_variance_to_expected_variance_from_mean

        # calculate the product of the ratio and difference of variance to expected variance for scoring and sorting highly variable features
        df_summary[ f'float_score_highly_variable_feature_from_{name_type}' ] = df_summary[ f'float_ratio_of_variance_to_expected_variance_from_mean_from_{name_type}' ] * df_summary[ f'float_diff_of_variance_to_expected_variance_from_mean_{name_type}' ]

    df_summary[ 'float_score_highly_variable_feature' ] = list( np.prod( arr_val ) if sum( np.sign( arr_val ) < 0 ) == 0 else 0 for arr_val in df_summary[ [ 'float_score_highly_variable_feature_from_log_transformed_normalized_count', 'float_score_highly_variable_feature_from_log_transformed_count' ] ].values )
    return df_summary

''' newly written functions '''
def is_binary_stream( f ) :
    ''' # 2022-05-01 01:57:10 
    check whether a given stream is a binary stream '''
    if hasattr( f, 'mode' ) : # if given stream is file
        return 'b' in f.mode
    else :
        return isinstance( f, ( io.RawIOBase, io.BufferedIOBase ) ) 
def __Get_path_essential_files__( path_folder_mtx_10x_input ) :
    ''' # 2022-04-30 16:28:15 
    get paths of essential files for the given matrix folder ('path_folder_mtx_10x_input', currently only supports 10X-GEX-formated output matrix)
    '''
    # define input file paths
    path_file_input_bc = f'{path_folder_mtx_10x_input}barcodes.tsv.gz'
    path_file_input_feature = f'{path_folder_mtx_10x_input}features.tsv.gz'
    path_file_input_mtx = f'{path_folder_mtx_10x_input}matrix.mtx.gz'
    # check whether input files exist
    for path_file in [ path_file_input_bc, path_file_input_feature, path_file_input_mtx ] :
        if not os.path.exists( path_file ) :
            raise OSError( f'{path_file} does not exist' )
    return path_file_input_bc, path_file_input_feature, path_file_input_mtx 
def Merge_Sort_Files( file_output, * l_iterator_decorated_file_input ) :
    """ # 2022-05-01 02:23:09 
    Merge sort input files (should be sorted) without loading the complete contents on memory.
    'path_file_output' : output file handle/stream
    'l_iterator_decorated_file_input' : a list of iterators based on input file handles (or streams). each iterator should yield the following tuple: (key_for_sorting, content_that_will_be_written_in_the_output_file). This function does not check whether the datatype of the 'content_that_will_be_written_in_the_output_file' matches that of 'path_file_output'
    """
    # handle invalid case
    if len( l_iterator_decorated_file_input ) == 0 :
        return - 1
    # perform merge sorting
    for r in heapq.merge( * l_iterator_decorated_file_input ) :
        file_output.write( r[ 1 ] ) # assumes the 'iterator_decorated_file_input' returns appropriate datatype (either bytes or string) for the output file
def __Merge_Sort_MTX_10X__( path_file_output, * l_path_file_input, flag_ramtx_sorted_by_id_feature = True, flag_delete_input_file_upon_completion = False ) :
    """ # 2022-05-01 02:25:07 
    merge sort mtx files 
    'path_file_output' and 'l_path_file_input'  : either file path or file handles
    
    'flag_ramtx_sorted_by_id_feature' : if True, sort by 'id_feature'. if False, sort by 'id_cell'
    """
    # process arguments for input files
    if isinstance( l_path_file_input[ 0 ], str ) : # if paths are given as input files
        flag_input_binary = l_path_file_input[ 0 ].rsplit( '.', 1 )[ 1 ].lower( ) == 'gz' # automatically detect gzipped input file # determined gzipped status by only looking at the first file
        l_file_input = list( gzip.open( path_file, 'rb' ) if flag_input_binary else open( path_file, 'r' ) for path_file in l_path_file_input )
    else :
        flag_input_binary = is_binary_stream( l_file_input[ 0 ] ) # detect binary stream 
        l_file_input = l_path_file_input
    # process argument for output file
    if isinstance( path_file_output, str ) : # if path was given as an output file
        flag_output_is_file = True
        flag_output_binary = path_file_output.rsplit( '.', 1 )[ 1 ].lower( ) == 'gz' # automatically detect gzipped input file # determined gzipped status by only looking at the first file
        file_output = gzip.open( path_file_output, 'wb' ) if flag_output_binary else open( path_file_output, 'w' )
    else :
        flag_output_is_file = False
        flag_output_binary = is_binary_stream( path_file_output ) # detect binary stream 
        file_output = path_file_output
    # define a function for decorating mtx record
    def __decorate_mtx_file__( file ) :
        while True :
            line = file.readline( )
            if len( line ) == 0 :
                break
            ''' parse a mtx record '''
            line_decoded = line.decode( ) if flag_input_binary else line
            index_row, index_column, float_value = ( line_decoded ).strip( ).split( ) # parse a record of a matrix-market format file
            index_row, index_column, float_value = int( index_row ), int( index_column ), float( float_value ) # 0-based coordinates
            yield index_row if flag_ramtx_sorted_by_id_feature else index_column, ( line if flag_input_binary else line.encode( ) ) if flag_output_binary else line_decoded
    
    Merge_Sort_Files( file_output, * list( __decorate_mtx_file__( file ) for file in l_file_input ) ) # perform merge sorting
    
    # if the output file is stream, does not close the stream # only close the output if the output file was an actual file
    if flag_output_is_file :
        file_output.close( )
    
    ''' delete input files once merge sort is completed if 'flag_delete_input_file_upon_completion' is True '''
    if flag_delete_input_file_upon_completion and isinstance( l_path_file_input[ 0 ], str ) : # if paths are given as input files
        for path_file in l_path_file_input :
            os.remove( path_file )
def __Merge_Sort_and_Index_MTX_10X__( path_file_output, * l_path_file_input, flag_ramtx_sorted_by_id_feature = True, flag_delete_input_file_upon_completion = False ) :
    """ # 2022-05-01 02:25:07 
    merge sort mtx files into a single mtx uncompressed file and index entries in the combined mtx file while writing the file
    'path_file_output' : should be a file path, file handle (or stream) for non-binary (text) output
    'l_path_file_input' 
    
    'flag_ramtx_sorted_by_id_feature' : if True, sort by 'id_feature'. if False, sort by 'id_cell'
    """
    # process arguments for input files
    if isinstance( l_path_file_input[ 0 ], str ) : # if paths are given as input files
        flag_input_binary = l_path_file_input[ 0 ].rsplit( '.', 1 )[ 1 ].lower( ) == 'gz' # automatically detect gzipped input file # determined gzipped status by only looking at the first file
        l_file_input = list( gzip.open( path_file, 'rb' ) if flag_input_binary else open( path_file, 'r' ) for path_file in l_path_file_input )
    else :
        flag_input_binary = is_binary_stream( l_file_input[ 0 ] ) # detect binary stream 
        l_file_input = l_path_file_input
    # process argument for output file
    if isinstance( path_file_output, str ) : # if path was given as an output file
        flag_output_is_file = True
        flag_output_binary = path_file_output.rsplit( '.', 1 )[ 1 ].lower( ) == 'gz' # automatically detect gzipped input file # determined gzipped status by only looking at the first file
        file_output = gzip.open( path_file_output, 'wb' ) if flag_output_binary else open( path_file_output, 'w' )
    else :
        flag_output_is_file = False
        flag_output_binary = is_binary_stream( path_file_output ) # detect binary stream 
        file_output = path_file_output
        
    if flag_output_binary : # the output file should be non-binary stream/file
        raise OSError( 'the output file should be non-binary stream/file' )

    # define and open index output file
    path_file_index_output = f"{path_file_output}.idx.tsv.gz"
    file_index_output = gzip.open( path_file_index_output, 'wb' )
    file_index_output.write( ( '\t'.join( [ 'index_entry', 'int_pos_start', 'int_pos_end' ] ) + '\n' ).encode( ) ) # write the header of the index file
        
    # define a function for decorating mtx record
    def __decorate_mtx_file__( file ) :
        while True :
            line = file.readline( )
            if len( line ) == 0 :
                break
            ''' parse a mtx record '''
            line_decoded = line.decode( ) if flag_input_binary else line
            index_row, index_column, float_value = ( line_decoded ).strip( ).split( ) # parse a record of a matrix-market format file
            index_row, index_column, float_value = int( index_row ), int( index_column ), float( float_value ) # 0-based coordinates
            yield index_row if flag_ramtx_sorted_by_id_feature else index_column, ( line if flag_input_binary else line.encode( ) ) if flag_output_binary else line_decoded

    # perform merge sorting
    index_entry_currently_being_written = -1
    int_num_character_written_for_index_entry_currently_being_written = 0
    int_total_num_character_written = 0
    for r in heapq.merge( * list( __decorate_mtx_file__( file ) for file in l_file_input ) ) :
        if index_entry_currently_being_written != r[ 0 ] : # if current index_entry is different from the previous one, which mark the change of sorted blocks (a block has the same id_entry), record the data for the previous block and initialze data for the next block 
            if index_entry_currently_being_written > 0 : # check whether 'index_entry_currently_being_written' is valid (ignore 'dummy' or default value that was used for initialization)
                file_index_output.write( ( '\t'.join( map( str, [ index_entry_currently_being_written, int_total_num_character_written, int_total_num_character_written + int_num_character_written_for_index_entry_currently_being_written ] ) ) + '\n' ).encode( ) ) # write information required for indexing
            int_total_num_character_written += int_num_character_written_for_index_entry_currently_being_written # update 'int_total_num_character_written'
            # initialize data for index of the next 'index_entry'
            index_entry_currently_being_written = r[ 0 ] # update current index_entry
            int_num_character_written_for_index_entry_currently_being_written = 0 # reset the count of characters (which is equal to the number of bytes for any mtx records, because they only contains numeric characters)
        int_num_character_written_for_index_entry_currently_being_written += file_output.write( r[ 1 ] ) # assumes the 'iterator_decorated_file_input' returns appropriate datatype (either bytes or string) for the output file # count the number of characters written for the current index_row
    
    # write the record for the last block
    file_index_output.write( ( '\t'.join( map( str, [ index_entry_currently_being_written, int_total_num_character_written, int_total_num_character_written + int_num_character_written_for_index_entry_currently_being_written ] ) ) + '\n' ).encode( ) ) # write information required for indexing
    # close index file
    file_index_output.close( )
    # if the output file is stream, does not close the stream # only close the output if the output file was an actual file
    if flag_output_is_file :
        file_output.close( )
    
    ''' delete input files once merge sort is completed if 'flag_delete_input_file_upon_completion' is True '''
    if flag_delete_input_file_upon_completion and isinstance( l_path_file_input[ 0 ], str ) : # if paths are given as input files
        for path_file in l_path_file_input :
            os.remove( path_file )

''' methods for handling 10X matrix objects '''
def Convert_df_count_to_MTX_10X( path_file_df_count, path_folder_mtx_10x_output, chunksize = 500000, flag_debugging = False, inplace = False ) :
    ''' # 2022-06-02 01:43:01 
    convert df_count (scarab output) to 10X MTX (matrix market) format in a memory-efficient manner.
    
    'path_file_df_count' : file path to 'df_count'
    '''
    # create a temporary output folder
    path_folder_temp = f'{path_folder_mtx_10x_output}temp_{UUID( )}/' 
    os.makedirs( path_folder_temp, exist_ok = True ) 

    # retrieve unique feature/barcode information from df_count
    DF_Deduplicate_without_loading_in_memory( path_file_df_count, f'{path_folder_temp}_features.tsv.gz', l_col_for_identifying_duplicates = [ 'feature', 'id_feature' ], str_delimiter = '\t' )
    int_num_lines = DF_Deduplicate_without_loading_in_memory( path_file_df_count, f'{path_folder_temp}_barcodes.tsv.gz', l_col_for_identifying_duplicates = [ 'barcode' ], str_delimiter = '\t' ) # collect the number of records

    # read features and barcode information
    df_barcode = pd.read_csv( f'{path_folder_temp}_barcodes.tsv.gz', sep = '\t', usecols = [ 'barcode' ] )
    df_feature = pd.read_csv( f'{path_folder_temp}_features.tsv.gz', sep = '\t', usecols = [ 'feature', 'id_feature' ] )
    df_feature = df_feature.loc[ :, [ 'id_feature', 'feature' ] ]
    df_feature[ '10X_type' ] = 'Gene Expression'
    # save feature/cell metadata
    df_barcode.to_csv( f'{path_folder_temp}barcodes.tsv.gz', sep = '\t', index = False, header = False )
    df_feature.to_csv( f'{path_folder_temp}features.tsv.gz', sep = '\t', index = False, header = False )

    # retrieve barcode/feature to integer representation of barcode/feature mapping
    dict_to_int_barcode = dict( ( e, i + 1 ) for i, e in enumerate( df_barcode.iloc[ :, 0 ].values ) )
    dict_to_int_feature = dict( ( e, i + 1 ) for i, e in enumerate( df_feature.iloc[ :, 0 ].values ) )

    int_num_features, int_num_barcodes, int_num_records = len( df_feature ), len( df_barcode ), int_num_lines # retrieve metadata of the output matrix
    del df_feature, df_barcode # delete objects

    # write mtx file
    with gzip.open( f'{path_folder_temp}matrix.mtx.gz', 'wb' ) as newfile :
        newfile.write( f"""%%MatrixMarket matrix coordinate integer general\n%\n{int_num_features} {int_num_barcodes} {int_num_records}\n""".encode( ) )
        # iterate through each chunk
        for df_chunk in pd.read_csv( path_file_df_count, iterator = True, header = 0, chunksize = chunksize, sep = '\t', usecols = [ 'id_feature', 'barcode', 'read_count' ] ) :
            df_chunk = df_chunk[ [ 'id_feature', 'barcode', 'read_count' ] ] # reorder columns
            df_chunk[ 'id_feature' ] = df_chunk.id_feature.apply( MAP.Map( dict_to_int_feature ).a2b )
            df_chunk[ 'barcode' ] = df_chunk.barcode.apply( MAP.Map( dict_to_int_barcode ).a2b )
            df_chunk.to_csv( newfile, sep = ' ', header = None, index = False )

    # export result files
    for name_file in [ 'features.tsv.gz', 'barcodes.tsv.gz', 'matrix.mtx.gz' ] :
        os.rename( f"{path_folder_temp}{name_file}", f"{path_folder_mtx_10x_output}{name_file}" )
    # delete temporary folder
    shutil.rmtree( path_folder_temp )
            
''' method for compressing and decompressing blocks of data '''
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
    'mtx_gzipped' :'mtx.gz',
    'pickle' :'pickle.stacked',
    'pickle_gzipped' :'pickle.gz.stacked',
    'feather' :'feather.stacked',
    'feather_lz4' :'feather_lz4.stacked',
    'index' :'idx.tsv.gz',
}
def _base64_decode( str_content ) :
    """ # 2022-07-04 23:19:18 
    receive base64-encoded string and return decoded bytes
    """
    return base64.b64decode( str_content.encode( 'ascii' ) )
def _base64_encode( byte_content ) :
    """ # 2022-07-04 23:19:18 
    receive bytes and return base64-encoded string
    """
    return base64.b64encode( byte_content ).decode( 'ascii' )
def _gzip_bytes( bytes_content ) :
    """ # 2022-05-24 23:43:36 
    gzip the given bytes 
    
    inputs:
    'bytes_content' : input bytes
    
    returns:
    'bytes_content_gzipped' : gzipped contents, number of bytes written
    
    """
    # compress the data
    gzip_file = io.BytesIO( )
    with gzip.GzipFile( fileobj = gzip_file, mode = 'w' ) as file :
        file.write( bytes_content ) 

    # retrieve the compressed content
    gzip_file.seek( 0 )
    bytes_content_gzipped = gzip_file.read( )
    gzip_file.close( )
    return bytes_content_gzipped
def _gunzip_bytes( bytes_content_gzipped ) :
    """ # 2022-05-24 23:43:36 
    gzip the given bytes 
    
    inputs:
    'bytes_content_gzipped' : input gzipped bytes
    
    returns:
    'bytes_content' : unzipped contents, number of bytes written
    
    """
    # uncompress the gzipped bytes
    with io.BytesIO( ) as gzip_file_content :
        gzip_file_content.write( bytes_content_gzipped )
        gzip_file_content.seek( 0 )
        with gzip.GzipFile( fileobj = gzip_file_content, mode = 'r' ) as gzip_file :
            bytes_content = gzip_file.read( )
    return bytes_content
def _feather_bytes_to_df( bytes_content ) :
    ''' # 2022-05-25 01:50:46 
    convert bytes to df using pyarrow.feather
    '''
    # uncompress the gzipped bytes
    with io.BytesIO( ) as file :
        file.write( bytes_content )
        file.seek( 0 )
        df = pyarrow.feather.read_feather( file )
    return df
def _feather_df_to_bytes( df, compression = 'uncompressed' ) :
    ''' # 2022-05-25 01:56:37 
    convert a python dataframe to bytes using pyarrow.feather with the given compression
    '''
    # compress the data
    file = io.BytesIO( )
    pyarrow.feather.write_feather( df, file, compression = compression )

    # retrieve the converted content
    file.seek( 0 )
    bytes_content = file.read( )
    file.close( )
    return bytes_content
def _pickle_bytes_to_obj( bytes_content ) :
    ''' # 2022-05-25 01:50:46 
    convert bytes to df using pickle
    '''
    # uncompress the gzipped bytes
    with io.BytesIO( ) as file :
        file.write( bytes_content )
        file.seek( 0 )
        obj = pickle.load( file ) 
    return obj
def _pickle_obj_to_bytes( obj ) :
    ''' # 2022-05-25 01:56:37 
    convert a python dataframe to bytes using pickle
    '''
    # compress the data
    file = io.BytesIO( )
    pickle.dump( obj, file, protocol = pickle.HIGHEST_PROTOCOL )

    # retrieve the converted content
    file.seek( 0 )
    bytes_content = file.read( )
    file.close( )
    return bytes_content
def _bytes_mtx_to_df_mtx( bytes_mtx, dtype_of_row_and_col_indices = None, dtype_of_value = None, int_min_num_records_for_pandas_parsing = 200 ) :
    ''' # 2022-06-01 13:55:58 
    convert bytes of a portion of matrix market file as a dataframe containing (arr_int_feature, arr_int_barcode, arr_value)
    
    'bytes_mtx': bytes of a portion of matrix market file. separator is ' '
    'dtype_of_row_and_col_indices' : set the dtype of column '0' (row index) and column '1' (col index)
    'dtype_of_value' : set the dtype of value
    '''
    if int_min_num_records_for_pandas_parsing > 0 and bytes_mtx.count( b'\n' ) < int_min_num_records_for_pandas_parsing : 
        # parse mtx bytes using pandas module without using pandas module
        return _arrays_mtx_to_df_mtx( _bytes_mtx_to_arrays_mtx( bytes_mtx, dtype_of_row_and_col_indices, dtype_of_value, int_min_num_records_for_pandas_parsing ) )
    else :
        # parse mtx bytes using pandas module
        df = pd.read_csv( io.BytesIO( bytes_mtx ), sep = ' ', header = None )
        df[ 0 ] -= 1
        df[ 1 ] -= 1
        # convert dtypes
        df.columns = np.arange( 3, dtype = np.uint8 ) # set integer index for each columns 
        df.index = np.zeros( len( df ), dtype = np.uint8 ) # ignore 'index' integers
        if dtype_of_row_and_col_indices is not None : # change dtype of row and col indices
            df[ 0 ] = df[ 0 ].astype( dtype_of_row_and_col_indices )
            df[ 1 ] = df[ 1 ].astype( dtype_of_row_and_col_indices )
        if dtype_of_value is not None : # change dtype of value
            df[ 2 ] = df[ 2 ].astype( dtype_of_value )
        return df
def _bytes_mtx_to_arrays_mtx( bytes_mtx, dtype_of_row_and_col_indices = None, dtype_of_value = None, int_min_num_records_for_pandas_parsing = 200 ) :
    ''' # 2022-06-01 13:56:07 
    convert bytes of a portion of matrix market file as three arrays (arr_int_feature, arr_int_barcode, arr_value)
    
    'bytes_mtx': bytes of a portion of matrix market file. separator is ' '
    'dtype_of_row_and_col_indices' : set the dtype of column '0' (row index) and column '1' (col index)
    'dtype_of_value' : set the dtype of value
    'int_min_num_records_for_pandas_parsing' : the minimum number of records for parsing with pandas module. 
    '''
    if int_min_num_records_for_pandas_parsing > 0 and bytes_mtx.count( b'\n' ) < int_min_num_records_for_pandas_parsing : 
        # parse without using pandas module to avoid the overhead
        l_f, l_b, l_v = [ ], [ ], [ ]
        for r in bytes_mtx.strip( ).split( b'\n' ) :
            f, b, v = r.split( b' ' )
            l_f.append( f )
            l_b.append( b )
            l_v.append( v )
        # set default dtypes of bytes_mtx, which is mandatory
        if dtype_of_value is None :
            dtype_of_value = np.float64
        if dtype_of_row_and_col_indices is None :
            dtype_of_row_and_col_indices = np.int32
        arr_f = np.array( l_f, dtype = dtype_of_row_and_col_indices )
        arr_b = np.array( l_b, dtype = dtype_of_row_and_col_indices )
        arr_v = np.array( l_v, dtype = dtype_of_value )
        arr_f -= 1
        arr_b -= 1
        return arr_f, arr_b, arr_v
    else :
        return _df_mtx_to_arrays_mtx( _bytes_mtx_to_df_mtx( bytes_mtx, int_min_num_records_for_pandas_parsing = 0 ), dtype_of_row_and_col_indices, dtype_of_value ) # make sure the records are parsed with pandas module
def _arrays_mtx_to_df_mtx( arrays_mtx, dtype_of_row_and_col_indices = None, dtype_of_value = None ) :
    ''' # 2022-05-25 16:59:28 
    convert arrays mtx formats to dataframe 
    
    dtype_of_row_and_col_indices = None, dtype_of_value = None : conversion of dtypes to use different dtypes in the output data
    '''
    arr_int_feature, arr_int_barcode, arr_value = arrays_mtx
    df = pd.DataFrame( { 0 : arr_int_feature, 1 : arr_int_barcode, 2 : arr_value }, index = np.zeros( len( arr_int_feature ), dtype = np.uint8 ) ) # ignore 'index' integers # this will preserve the original data types
    df.columns = np.arange( 3, dtype = np.uint8 ) # set integer index for each columns 
    # convert dtypes
    if dtype_of_row_and_col_indices is not None : # change dtype of row and col indices
        df[ 0 ] = df[ 0 ].astype( dtype_of_row_and_col_indices )
        df[ 1 ] = df[ 1 ].astype( dtype_of_row_and_col_indices )
    if dtype_of_value is not None : # change dtype of value
        df[ 2 ] = df[ 2 ].astype( dtype_of_value )
    return df
def _df_mtx_to_arrays_mtx( df_mtx, dtype_of_row_and_col_indices = None, dtype_of_value = None ) :
    ''' # 2022-05-25 16:59:32 
    convert dataframe mtx format to arrays mtx objects
    
    dtype_of_row_and_col_indices = None, dtype_of_value = None : conversion of dtypes to use different dtypes in the output data
    '''
    # parse df as arrays # parsing individual columns will preserve dtypes, and will be much faster
    arr_int_feature = df_mtx[ 0 ].values
    arr_int_barcode = df_mtx[ 1 ].values
    arr_value = df_mtx[ 2 ].values
    # convert dtypes
    if dtype_of_row_and_col_indices is not None : # change dtype of row and col indices
        arr_int_feature = arr_int_feature.astype( dtype_of_row_and_col_indices )
        arr_int_barcode = arr_int_barcode.astype( dtype_of_row_and_col_indices )
    if dtype_of_value is not None : # change dtype of value
        arr_value = arr_value.astype( dtype_of_value )     
    return ( arr_int_feature, arr_int_barcode, arr_value )
def _arrays_mtx_to_bytes_mtx( arrays_mtx, str_format_value = "{}" ) :
    ''' # 2022-05-25 10:43:01 
    converts arrays of a matrix (0-based coordinates) to bytes_mtx
    
    'arrays_mtx' : input arrays of a mtx, ( arr_int_feature, arr_int_barcode, arr_value )
    'str_format_value' : a format string to encode value
    '''
    arr_int_feature, arr_int_barcode, arr_value = arrays_mtx
    return ( '\n'.join( list( str( index_row ) + ' ' + str( index_col ) + ' ' + str_format_value.format( value ) for index_row, index_col, value in zip( arr_int_feature + 1, arr_int_barcode + 1, arr_value ) ) ) + '\n' ).encode( ) # 0>1-based coordinates
def _df_mtx_to_bytes_mtx( df_mtx, str_format_value = "{}" ) :    
    ''' # 2022-05-25 10:50:16 
    converts arrays of a matrix (0-based coordinates) to bytes_mtx
    
    'df_mtx' : input dataframe of a mtx, columns: ( int_feature, int_barcode, value )
    'str_format_value' : a format string to encode value
    '''
    return _arrays_mtx_to_bytes_mtx( _df_mtx_to_arrays_mtx( df_mtx ), str_format_value = str_format_value )
def _arrays_mtx_to_arrays_mtx( arrays_mtx, dtype_of_row_and_col_indices = None, dtype_of_value = None ) :
    ''' #2022-05-25 04:26:08 
    change dtypes of arrays_mtx
    '''
    # parse df as arrays
    arr_int_feature, arr_int_barcode, arr_value = arrays_mtx # col = barcode, row = feature
    # convert dtypes
    if dtype_of_row_and_col_indices is not None : # change dtype of row and col indices
        arr_int_feature = arr_int_feature.astype( dtype_of_row_and_col_indices )
        arr_int_barcode = arr_int_barcode.astype( dtype_of_row_and_col_indices )
    if dtype_of_value is not None : # change dtype of value
        arr_value = arr_value.astype( dtype_of_value )
    return arr_int_feature, arr_int_barcode, arr_value
''' methods for retrieving appropriate functions based on input file format and the task '''
def _get_func_bytes_mtx_to_processed_bytes_and_other_settings_based_on_file_format( file_format, dtype_of_row_and_col_indices = None, dtype_of_value = None ) :
    ''' # 2022-05-25 23:23:52 
    return a function 'func_arrays_mtx_to_processed_bytes' and relevant settings for the given file_format
    
    returns:
    str_etx, str_ext_index, func_bytes_mtx_to_processed_bytes
    '''
    str_etx = _dict_file_format_to_ext[ file_format ]
    str_ext_index = _dict_file_format_to_ext[ 'index' ]
    if file_format == 'mtx_gzipped' :
        func_bytes_mtx_to_processed_bytes = _gzip_bytes
    elif file_format == 'feather_lz4' : 
        def func_bytes_mtx_to_processed_bytes( bytes_mtx ) :
            return _feather_df_to_bytes( _bytes_mtx_to_df_mtx( bytes_mtx, dtype_of_row_and_col_indices, dtype_of_value ), 'lz4' )
    elif file_format == 'feather' :
        def func_bytes_mtx_to_processed_bytes( bytes_mtx ) :
            return _feather_df_to_bytes( _bytes_mtx_to_df_mtx( bytes_mtx, dtype_of_row_and_col_indices, dtype_of_value ), 'uncompressed' )
    elif file_format == 'pickle' :
        def func_bytes_mtx_to_processed_bytes( bytes_mtx ) :
            return _pickle_obj_to_bytes( _bytes_mtx_to_arrays_mtx( bytes_mtx, dtype_of_row_and_col_indices, dtype_of_value ) )
    elif file_format == 'pickle_gzipped' :
        def func_bytes_mtx_to_processed_bytes( bytes_mtx ) :
            return _gzip_bytes( _pickle_obj_to_bytes( _bytes_mtx_to_arrays_mtx( bytes_mtx, dtype_of_row_and_col_indices, dtype_of_value ) ) )
    return str_etx, str_ext_index, func_bytes_mtx_to_processed_bytes
def _get_func_arrays_mtx_to_processed_bytes_and_other_settings_based_on_file_format( file_format, str_format_value = '{}', dtype_of_row_and_col_indices = None, dtype_of_value = None ) :
    ''' # 2022-05-25 23:21:44 
    return a function 'func_arrays_mtx_to_processed_bytes' and relevant settings for the given file_format
    
    returns:
    str_etx, str_ext_index, func_arrays_mtx_to_processed_bytes
    '''
    str_etx = _dict_file_format_to_ext[ file_format ]
    str_ext_index = _dict_file_format_to_ext[ 'index' ]
    if file_format == 'mtx_gzipped' :
        def func_arrays_mtx_to_processed_bytes( arrays_mtx ) :
            return _gzip_bytes( _arrays_mtx_to_bytes_mtx( arrays_mtx, str_format_value ) )
    elif file_format == 'feather_lz4' :
        def func_arrays_mtx_to_processed_bytes( arrays_mtx ) :
            return _feather_df_to_bytes( _arrays_mtx_to_df_mtx( arrays_mtx, dtype_of_row_and_col_indices, dtype_of_value ), 'lz4' )
    elif file_format == 'feather' :
        def func_arrays_mtx_to_processed_bytes( arrays_mtx ) :
            return _feather_df_to_bytes( _arrays_mtx_to_df_mtx( arrays_mtx, dtype_of_row_and_col_indices, dtype_of_value ), 'uncompressed' )
    elif file_format == 'pickle' :
        def func_arrays_mtx_to_processed_bytes( arrays_mtx ) :
            return _pickle_obj_to_bytes( _arrays_mtx_to_arrays_mtx( arrays_mtx, dtype_of_row_and_col_indices, dtype_of_value ) )
    elif file_format == 'pickle_gzipped' :
        def func_arrays_mtx_to_processed_bytes( arrays_mtx ) :
            return _gzip_bytes( _pickle_obj_to_bytes( _arrays_mtx_to_arrays_mtx( arrays_mtx, dtype_of_row_and_col_indices, dtype_of_value ) ) )
    return str_etx, str_ext_index, func_arrays_mtx_to_processed_bytes
def _get_func_processed_bytes_to_arrays_mtx_and_other_settings_based_on_file_format( file_format, dtype_of_row_and_col_indices = None, dtype_of_value = None ) :
    ''' # 2022-05-26 00:25:44 
    return a function 'func_processed_bytes_to_arrays_mtx' and relevant settings for the given file_format
    
    returns:
    str_etx, str_ext_index, func_processed_bytes_to_arrays_mtx
    ''' 
    ''' retrieve RAMtx format-specific import settings '''
    str_ext = _dict_file_format_to_ext[ file_format ]
    str_ext_index = _dict_file_format_to_ext[ 'index' ]
    if file_format == 'mtx_gzipped' :
        def func_processed_bytes_to_arrays_mtx( bytes_content ) :
            return _bytes_mtx_to_arrays_mtx( _gunzip_bytes( bytes_content ), dtype_of_row_and_col_indices, dtype_of_value )
    elif file_format == 'feather_lz4' :
        def func_processed_bytes_to_arrays_mtx( bytes_content ) :
            return _df_mtx_to_arrays_mtx( _feather_bytes_to_df( bytes_content ) )
    elif file_format == 'feather' :
        def func_processed_bytes_to_arrays_mtx( bytes_content ) :
            return _df_mtx_to_arrays_mtx( _feather_bytes_to_df( bytes_content ) )
    elif file_format == 'pickle' :
        def func_processed_bytes_to_arrays_mtx( bytes_content ) :
            return _pickle_bytes_to_obj( bytes_content )
    elif file_format == 'pickle_gzipped' :
        def func_processed_bytes_to_arrays_mtx( bytes_content ) :
            return _pickle_bytes_to_obj( _gunzip_bytes( bytes_content ) )
    return str_ext, str_ext_index, func_processed_bytes_to_arrays_mtx

''' deprecated methods for creating RAMtx objects '''
def __Merge_Sort_MTX_10X_and_Write_and_Index_Zarr__( za_mtx, za_mtx_index, * l_path_file_input, flag_ramtx_sorted_by_id_feature = True, flag_delete_input_file_upon_completion = False, dtype_mtx = np.float64, dtype_mtx_index = np.float64, int_size_buffer_for_mtx_index = 1000 ) :
    """ # deprecated
    # 2022-07-02 11:37:05 
    merge sort mtx files into a single mtx uncompressed file and index entries in the combined mtx file while writing the file
    
    # 2022-07-02 11:37:13 
    za_mtx format change. now each mtx record contains two values instead of three values for more compact storage
    
    'za_mtx' : output Zarr matrix object (persistant array is recommended)
    'za_mtx_index' : output Zarr matrix index object (persistant array is recommended)
    'flag_ramtx_sorted_by_id_feature' : if True, sort by 'id_feature'. if False, sort by 'id_cell'
    'flag_delete_input_file_upon_completion' : delete input files after the output files are generated
    'int_size_buffer_for_mtx_index' : size of buffer for writing mtx index data to the input 'za_mtx_index' data object
    """
    # process arguments for input files
    if isinstance( l_path_file_input[ 0 ], str ) : # if paths are given as input files
        flag_input_binary = l_path_file_input[ 0 ].rsplit( '.', 1 )[ 1 ].lower( ) == 'gz' # automatically detect gzipped input file # determined gzipped status by only looking at the first file
        l_file_input = list( gzip.open( path_file, 'rb' ) if flag_input_binary else open( path_file, 'r' ) for path_file in l_path_file_input )
    else :
        flag_input_binary = is_binary_stream( l_file_input[ 0 ] ) # detect binary stream 
        l_file_input = l_path_file_input
    # process argument for output file
        
    # define a function for decorating mtx record
    def __decorate_mtx_file__( file ) :
        ''' parse and decorate mtx record for sorting. the resulting records only contains two values, index of axis that were not indexed and the data value, for more compact storage and efficient retrival of the data. '''
        while True :
            line = file.readline( )
            if len( line ) == 0 :
                break
            ''' parse a mtx record '''
            line_decoded = line.decode( ) if flag_input_binary else line
            index_row, index_column, float_value = ( line_decoded ).strip( ).split( ) # parse a record of a matrix-market format file
            index_row, index_column, float_value = int( index_row ) - 1, int( index_column ) - 1, float( float_value ) # 1-based > 0-based coordinates
            # return record with decoration according to the sorted axis # return 0-based coordinates
            if flag_ramtx_sorted_by_id_feature :
                yield index_row, ( index_column, float_value ) 
            else :
                yield index_column, ( index_row, float_value ) 
    # perform merge sorting
    int_entry_currently_being_written = None # place holder value
    int_num_mtx_records_written = 0
    l_mtx_record = [ ]
    int_num_mtx_index_records_written = 0
    l_mtx_index = [ ]
    def flush_matrix_index( int_num_mtx_index_records_written ) :
        ''' # 2022-06-21 16:26:09 
        flush matrix index data and update 'int_num_mtx_index_records_written' '''
        int_num_newly_added_mtx_index_records = len( l_mtx_index )
        if int_num_newly_added_mtx_index_records > 0 :
            za_mtx_index[ int_num_mtx_index_records_written : int_num_mtx_index_records_written + int_num_newly_added_mtx_index_records ] = np.array( l_mtx_index, dtype = dtype_mtx_index ) # add data to the zarr data sink
            int_num_mtx_index_records_written += int_num_newly_added_mtx_index_records # update 'int_num_mtx_index_records_written'
        return int_num_mtx_index_records_written
    def flush_matrix( int_num_mtx_records_written ) : 
        ''' # 2022-06-21 16:26:09 
        flush a block of matrix data of a single entry (of the axis used for sorting) to Zarr and index the block, and update 'int_num_mtx_records_written' '''
        int_num_newly_added_mtx_records = len( l_mtx_record )
        if int_num_newly_added_mtx_records > 0 : # if there is valid record to be flushed
            # add records to mtx_index
            l_mtx_index.append( [ int_num_mtx_records_written, int_num_mtx_records_written + int_num_newly_added_mtx_records ] ) # collect information required for indexing
            for int_entry in range( int_entry_currently_being_written + 1, int_entry_of_the_current_record ) : # for the int_entry that lack count data and does not have indexing data, put place holder values
                l_mtx_index.append( [ -1, -1 ] ) # put place holder values for int_entry lacking count data.
            
            za_mtx[ int_num_mtx_records_written : int_num_mtx_records_written + int_num_newly_added_mtx_records ] = np.array( l_mtx_record, dtype = dtype_mtx ) # add data to the zarr data sink
            int_num_mtx_records_written += int_num_newly_added_mtx_records
        return int_num_mtx_records_written
    
    for int_entry_of_the_current_record, mtx_record in heapq.merge( * list( __decorate_mtx_file__( file ) for file in l_file_input ) ) : # retrieve int_entry and mtx_record of the current mtx_record
        if int_entry_currently_being_written is None :
            int_entry_currently_being_written = int_entry_of_the_current_record # update current int_entry
        elif int_entry_currently_being_written != int_entry_of_the_current_record : # if current int_entry is different from the previous one, which mark the change of sorted blocks (a block has the same id_entry), save the data for the previous block to the output zarr and initialze data for the next block 
            int_num_mtx_records_written = flush_matrix( int_num_mtx_records_written ) # flush data
            l_mtx_record = [ ] # reset buffer
            int_entry_currently_being_written = int_entry_of_the_current_record # update current int_entry
            # flush matrix index once the buffer is full
            if len( l_mtx_index ) >= int_size_buffer_for_mtx_index :
                int_num_mtx_index_records_written = flush_matrix_index( int_num_mtx_index_records_written )
                l_mtx_index = [ ] # reset buffer
        # append the record to the data
        l_mtx_record.append( mtx_record )
        
    # write the record for the last block
    int_entry_of_the_current_record = za_mtx_index.shape[ 0 ] # set 'next' int_entry to the end of the int_entry values so that place holder values can be set to the missing int_entry 
    int_num_mtx_records_written = flush_matrix( int_num_mtx_records_written ) # flush the last block
    int_num_mtx_index_records_written = flush_matrix_index( int_num_mtx_index_records_written ) # flush matrix index data
    
    ''' delete input files once merge sort is completed if 'flag_delete_input_file_upon_completion' is True '''
    if flag_delete_input_file_upon_completion and isinstance( l_path_file_input[ 0 ], str ) : # if paths are given as input files
        for path_file in l_path_file_input :
            os.remove( path_file )
def Convert_MTX_10X_to_RAMtx( path_folder_mtx_10x_input, path_folder_ramtx_output, flag_ramtx_sorted_by_id_feature = True, int_num_threads = 15, int_num_threads_for_splitting = 3, int_max_num_entries_for_chunk = 10000000, int_max_num_files_for_each_merge_sort = 5, dtype_mtx = np.float64, dtype_mtx_index = np.float64, int_num_of_records_in_a_chunk_zarr_matrix = 20000, int_num_of_entries_in_a_chunk_zarr_matrix_index = 1000, verbose = False, flag_debugging = False ) :
    '''  # deprecated
    # 2022-07-08 01:56:32 
    sort and index a given 'path_folder_mtx_10x_input' containing a 10X matrix file, constructing a random read access matrix (RAMtx).
    'path_folder_mtx_10x_input' folder will be used as temporary folder
    During the operation, the input matrix will be unzipped in two different formats before indexing, which might require extradisk space (expected peak disk usage: 4~8 times of the gzipped mtx file size).
    if the 'path_folder_ramtx_output' folder contains RAMtx object of any format, this funciton will exit
    Assumes the values stored in the 10X data is integer.

    change in 2022-05-15 19:43:07 :
    barcodes and features will be sorted, and both will be renumbered in the output matrix.mtx.gz file.

    2022-05-16 15:05:32 :
    barcodes and features with zero counts will not be removed to give flexibillity for the RAMtx object.

    2022-06-10 20:04:06 :
    memory leak issue detected and resolved

    'int_num_threads' : number of threads for multiprocessing/multithreading. 
    'path_folder_ramtx_output' : an empty folder should be given.
    'flag_ramtx_sorted_by_id_feature' : if True, sort by 'id_feature'. if False, sort by 'id_cell'
    'int_max_num_files_for_each_merge_sort' : (default: 5) the maximun number of files that can be merged in a merge sort run. reducing this number will increase the number of threads that can be used for sorting but increase the number of iteration for iterative merge sorting. (3~10 will produce an ideal performance.)
    'dtype_mtx' (default: np.float64), dtype of the output zarr array for storing matrix
    'dtype_mtx_index' (default: np.float64) : dtype of the output zarr array for storing matrix indices
    'flag_debugging' : if True, does not delete temporary files
    'int_num_of_records_in_a_chunk_zarr_matrix' : chunk size for output zarr objects
    'int_num_of_entries_in_a_chunk_zarr_matrix_index' : chunk size for output zarr objects
    'int_num_threads_for_splitting' : minimum number of threads for spliting and sorting of the input matrix.
    
    ❤️❤️❤️ # 2022-07-12 20:05:10  improve memory efficiency by using array for coordinate conversion and read chunking for 'Axis' generation
    '''

    path_file_flag = f"{path_folder_ramtx_output}ramtx.completed.flag"
    if not os.path.exists( path_file_flag ) : # check flag
        # create a temporary folder
        path_temp = f"{path_folder_mtx_10x_input}temp_{UUID( )}/" 
        os.makedirs( path_temp, exist_ok = True )

        path_file_input_bc, path_file_input_feature, path_file_input_mtx = __Get_path_essential_files__( path_folder_mtx_10x_input )
        int_num_features, int_num_barcodes, int_num_records = MTX_10X_Retrieve_number_of_rows_columns_and_records( path_folder_mtx_10x_input ) # retrieve metadata of mtx
        int_max_num_entries_for_chunk = min( int( np.ceil( int_num_records / int_num_threads ) ), int_max_num_entries_for_chunk ) # update the 'int_max_num_entries_for_chunk' based on the total number of entries and the number of given threads

        ''' retrieve mappings of previous index_entry to new index_entry after sorting (1-based) for both barcodes and features '''
        # retrieve sorting indices of barcodes and features
        dict_name_file_to_dict_index_entry_to_index_entry_new_after_sorting = dict( ( name_file, dict( ( e, i + 1 ) for i, e in enumerate( pd.read_csv( f"{path_folder_mtx_10x_input}{name_file}", sep = '\t', header = None, usecols = [ 0 ] ).squeeze( ).sort_values( ).index.values + 1 ) ) ) for name_file in [ 'barcodes.tsv.gz', 'features.tsv.gz' ] ) # 0>1 based coordinates. index_entry is in 1-based coordinate format (same as mtx format) # since rank will be used as a new index, it should be 1-based, and 1 will be added to the rank in 0-based coordinates
        dict_name_file_to_mo_from_index_entry_to_index_entry_new_after_sorting = dict( ( name_file, MAP.Map( dict_name_file_to_dict_index_entry_to_index_entry_new_after_sorting[ name_file ] ).a2b ) for name_file in dict_name_file_to_dict_index_entry_to_index_entry_new_after_sorting ) # retrieve mapping objects for each name_file

        ''' sort small chunks based on the mapped ranks and write them as small files '''
        def __mtx_record_distributor__( * l_pipe_to_and_from_mtx_record_receiver ) :
            ''' # 2022-06-10 21:15:45 
            distribute matrix records from the input matrix file 

            'l_pipe_to_and_from_mtx_record_receiver' : [ pipe_to_receiver_1, pipe_to_receiver_2, ..., pipe_from_receiver_1, pipe_from_receiver_2, ... ]
            '''
            # parse inputs
            int_num_receiver = int( len( l_pipe_to_and_from_mtx_record_receiver ) / 2 ) # retrieve the number of receivers
            l_pipe_to_mtx_record_receiver = l_pipe_to_and_from_mtx_record_receiver[ : int_num_receiver ]
            l_pipe_from_mtx_record_receiver = l_pipe_to_and_from_mtx_record_receiver[ int_num_receiver : ]

            with gzip.open( path_file_input_mtx, 'rb' ) as file : # assumes that the input fastq file is gzipped.
                ''' read the first line '''
                line = file.readline( ) # binary string
                ''' if the first line of the file contains a comment line, read all comment lines and a description line following the comments. '''
                if line.decode( )[ 0 ] == '%' :
                    # read comment and the description line
                    while True :
                        if line.decode( )[ 0 ] != '%' :
                            break
                        line = file.readline( ) # try to read the description line
                    line = file.readline( ) # read a mtx record 
                ''' distribute mtx records '''
                index_receiver = 0 
                arr_num_mtx_record_need_flushing = np.zeros( len( l_pipe_to_mtx_record_receiver ) ) # record the number of mtx records that need flushing
                l_line = [ ] # initialize the contents 
                while True :
                    if len( line ) == 0 : # if all records have been read
                        l_pipe_to_mtx_record_receiver[ index_receiver ].send( b''.join( l_line ) ) # send the collected records
                        break
                    ''' if the current receiver is marked unavailable, check whether it is available now, and if not, wait a little and use the next receiver  '''
                    if arr_num_mtx_record_need_flushing[ index_receiver ] < 0 :
                        if not l_pipe_from_mtx_record_receiver[ index_receiver ].poll( ) : # if the receiver is still unavailable
                            time.sleep( 1 ) # sleep for 1 second
                            index_receiver = ( index_receiver + 1 ) % int_num_receiver # use the next receiver instead
                            continue
                        else :
                            ''' if the receiver can become available, mark it as an available receiver '''
                            l_pipe_from_mtx_record_receiver[ index_receiver ].recv( ) # clear the signal that indicates the receiver is not available
                            arr_num_mtx_record_need_flushing[ index_receiver ] = 0 # remove the mark indicating the current receiver is 'unavailable'
                    """ for the currently available receiver, add a record in the queue for the receiver and if the size of the queue is full, distribute the records to the receiver """
                    ''' add a record in the queue for the receiver '''
                    l_line.append( line ) # collect the content
                    arr_num_mtx_record_need_flushing[ index_receiver ] += 1 # update the count
                    ''' if the size of the queue is full, distribute the records to the receiver '''
                    if arr_num_mtx_record_need_flushing[ index_receiver ] >= int_max_num_entries_for_chunk : # if the chunk is full, send collected content, start sorting and writing as a file
                        l_pipe_to_mtx_record_receiver[ index_receiver ].send( b''.join( l_line ) ) # send the collected records
                        l_line = [ ] # initialize the content for the next batch
                        arr_num_mtx_record_need_flushing[ index_receiver ] = -1 # mark the current receiver as 'unavailable'
                        index_receiver = ( index_receiver + 1 ) % int_num_receiver # use the next receiver
                    ''' read the next record '''
                    line = file.readline( ) # read a mtx record 
            # notify each receiver that all records were parsed and receivers should exit after completing their remaining jobs
            for pipe in l_pipe_to_mtx_record_receiver :
                pipe.send( -1 )
        def __mtx_record_sorter__( pipe_from_mtx_record_parser, pipe_to_mtx_record_parser ) :
            ''' # 2022-07-08 01:56:26 
            receive matrix record from the mtx record parser and write a sorted matrix file for each chunk

            'pipe_from_mtx_record_parser' : if bytes is received, parse the bytes. if 0 is received, write the bytes to file. if -1 is received, write the bytes to file and exit
            '''
            while True :
                byte_content = pipe_from_mtx_record_parser.recv( )
                if byte_content == -1 or isinstance( byte_content, int ) : # complete the process
                    break
                # if there is valid content to be written, write the sorted records as a file
                if len( byte_content ) > 0 :
                    try :
                        with io.BytesIO( byte_content ) as file :
                            df = pd.read_csv( file, sep = ' ', header = None )
                        del byte_content
                        df.columns = [ 'index_row', 'index_col', 'float_value' ]
                        for name_col, name_file in zip( [ 'index_row', 'index_col' ], [ 'features.tsv.gz', 'barcodes.tsv.gz' ] ) :
                            df[ name_col ] = df[ name_col ].apply( dict_name_file_to_mo_from_index_entry_to_index_entry_new_after_sorting[ name_file ] ) # retrieve ranks of the entries, or new indices after sorting
                        df.sort_values( 'index_row' if flag_ramtx_sorted_by_id_feature else 'index_col', inplace = True ) # sort by row if the matrix is sorted by features and sort by column if the matrix is sorted by barcodes
                        df.to_csv( f"{path_temp}0.{UUID( )}.mtx.gz", sep = ' ', header = None, index = False ) # save the sorted mtx records as a file
                        del df
                    except :
                        print( byte_content.decode( ).split( '\n', 1 )[ 0 ] )
                        break
                pipe_to_mtx_record_parser.send( 'flush completed' ) # send signal that flushing the received data has been completed, and now ready to export matrix again

        int_n_workers_for_sorting = min( int_num_threads_for_splitting, max( int_num_threads - 1, 1 ) ) # one thread for distributing records. Minimum numbers of workers for sorting is 1 # the number of worker for sorting should not exceed 3
        l_pipes_from_distributor_to_sorter = list( mp.Pipe( ) for _ in range( int_n_workers_for_sorting ) ) # create pipes for sending records from the distributor to the sorter
        l_pipes_from_sorter_to_distributor = list( mp.Pipe( ) for _ in range( int_n_workers_for_sorting ) ) # create pipes for sending records from the sorter to the distributor

        l_processes = [ mp.Process( target = __mtx_record_distributor__, args = ( list( pipe_in for pipe_in, pipe_out in l_pipes_from_distributor_to_sorter ) + list( pipe_out for pipe_in, pipe_out in l_pipes_from_sorter_to_distributor ) ) ) ] # add a process for distributing fastq records
        for index in range( int_n_workers_for_sorting ) :
            l_processes.append( mp.Process( target = __mtx_record_sorter__, args = ( l_pipes_from_distributor_to_sorter[ index ][ 1 ], l_pipes_from_sorter_to_distributor[ index ][ 0 ] ) ) ) # add process for receivers
        # start works and wait until all works are completed.
        for p in l_processes :
            p.start( )
        for p in l_processes :
            p.join( )

        """
        Iterative merge sort of MTX files
        """
        int_max_num_files_for_each_merge_sort = 5 # set the maximum number of files that can be merge-sorted at a single time
        int_number_of_recursive_merge_sort = 0 # the number of merge sort steps that has been performed

        ''' prepare workers '''
        int_num_workers =  int_num_threads # the main thread will not takes a lot of computing time, since it will only be used to distribute jobs
        l_pipes_from_main_process_to_worker = list( mp.Pipe( ) for _ in range( int_num_workers ) ) # create pipes for sending records from the main process to workers
        l_pipes_from_worker_to_main_process = list( mp.Pipe( ) for _ in range( int_num_workers ) ) # create pipes for sending records from workers to the main process
        arr_availability = np.full( int_num_workers, 'available', dtype = object ) # an array that will store availability status for each process
        def __merge_sorter__( pipe_from_main_process, pipe_to_main_process ) :
            while True :
                args = pipe_from_main_process.recv( ) # receive work from the main process
                if isinstance( args, int ) : # if a termination signal is received from the main process, exit 
                    break
                __Merge_Sort_MTX_10X__( * args, flag_ramtx_sorted_by_id_feature = flag_ramtx_sorted_by_id_feature, flag_delete_input_file_upon_completion = False if flag_debugging else True )
                pipe_to_main_process.send( args ) # notify main process that the work has been done
        l_worker = list( mp.Process( target = __merge_sorter__, args = ( l_pipes_from_main_process_to_worker[ index ][ 1 ], l_pipes_from_worker_to_main_process[ index ][ 0 ] ) ) for index in range( int_num_workers ) )
        index_worker = 0 # index of current worker
        ''' start workers '''
        for p in l_worker :
            p.start( ) # start workers

        ''' until files can be merge-sorted only using a single process '''
        while len( glob.glob( f"{path_temp}{int_number_of_recursive_merge_sort}.*.mtx.gz" ) ) > int_max_num_files_for_each_merge_sort :
            l_path_file_to_be_merged = glob.glob( f"{path_temp}{int_number_of_recursive_merge_sort}.*.mtx.gz" ) # retrieve the paths of the files that will be merged
            int_num_files_to_be_merged = len( l_path_file_to_be_merged ) # retrieve the number of files that will be merged.
            if verbose :
                print( f'current recursive merge sort step is {int_number_of_recursive_merge_sort + 1}, merging {int_num_files_to_be_merged} files' )

            ''' distribute works to workers '''
            l_l_path_file_input = LIST_Split( l_path_file_to_be_merged, int( np.ceil( int_num_files_to_be_merged / int_max_num_files_for_each_merge_sort ) ) )
            l_l_path_file_input_for_distribution = deepcopy( l_l_path_file_input )[ : : -1 ] # a list containing inputs
            l_l_path_file_input_completed = [ ]
            while len( l_l_path_file_input_completed ) != len( l_l_path_file_input ) : # until all works are distributed and completed
                if arr_availability[ index_worker ] == 'available' : # if the current worker is available
                    if len( l_l_path_file_input_for_distribution ) > 0 : # if there are works remain to be given, give a work to the current worker
                        path_file_output = f"{path_temp}{int_number_of_recursive_merge_sort + 1}.{UUID( )}.mtx.gz"
                        l_pipes_from_main_process_to_worker[ index_worker ][ 0 ].send( [ path_file_output ] + list( l_l_path_file_input_for_distribution.pop( ) ) ) # give a work to the current worker
                        arr_availability[ index_worker ] = 'working' # update the status of the worker to 'working'
                else : # if the current worker is still working
                    time.sleep( 1 ) # give time to workers until the work is completed

                if arr_availability[ index_worker ] == 'working' and l_pipes_from_worker_to_main_process[ index_worker ][ 1 ].poll( ) : # if the previous work given to the current thread has been completed
                    l_l_path_file_input_completed.append( l_pipes_from_worker_to_main_process[ index_worker ][ 1 ].recv( ) ) # collect the information about the completed work
                    arr_availability[ index_worker ] = 'available' # update the status of the worker

                index_worker = ( index_worker + 1 ) % int_num_workers # set the next worker as the current worker
            if verbose :
                print( f"all works for the 'int_number_of_recursive_merge_sort'={int_number_of_recursive_merge_sort} completed" )

            ''' update the number of recursive merge sort steps '''
            int_number_of_recursive_merge_sort += 1

        ''' send termination signals to the workers '''
        for index in range( int_num_workers ) :
            l_pipes_from_main_process_to_worker[ index ][ 0 ].send( -1 )
        if verbose :
            print( 'termination signal given' )

        ''' dismiss workers '''
        for p in l_worker :
            p.join( )    

        """
        Final merge sort step and construct RAMTx (Zarr) matrix
        """
        # create an output directory
        os.makedirs( path_folder_ramtx_output, exist_ok = True )

        # open a persistent zarray to store matrix and matrix index
        za_mtx = zarr.open( f'{path_folder_ramtx_output}matrix.zarr', mode = 'w', shape = ( int_num_records, 2 ), chunks = ( int_num_of_records_in_a_chunk_zarr_matrix, 2 ), dtype = dtype_mtx ) # each mtx record will contains two values instead of three values for more compact storage 
        za_mtx_index = zarr.open( f'{path_folder_ramtx_output}matrix.index.zarr', mode = 'w', shape = ( int_num_features if flag_ramtx_sorted_by_id_feature else int_num_barcodes, 2 ), chunks = ( int_num_of_entries_in_a_chunk_zarr_matrix_index, 2 ), dtype = dtype_mtx_index ) # max number of matrix index entries is 'int_num_records' of the input matrix. this will be resized # dtype of index should be np.float64 to be compatible with Zarr.js, since Zarr.js currently does not support np.int64...

        # merge-sort the remaining files into the output zarr data sink and index the zarr
        __Merge_Sort_MTX_10X_and_Write_and_Index_Zarr__( za_mtx, za_mtx_index, * glob.glob( f"{path_temp}{int_number_of_recursive_merge_sort}.*.mtx.gz" ), flag_ramtx_sorted_by_id_feature = flag_ramtx_sorted_by_id_feature, flag_delete_input_file_upon_completion = False if flag_debugging else True, int_size_buffer_for_mtx_index = int_num_of_entries_in_a_chunk_zarr_matrix_index ) # matrix index buffer size is 'int_num_of_entries_in_a_chunk_zarr_matrix_index'
            
        ''' write sorted barcodes and features files to zarr objects'''
        for name_axis, flag_used_for_sorting in zip( [ 'barcodes', 'features' ], [ not flag_ramtx_sorted_by_id_feature, flag_ramtx_sorted_by_id_feature ] ) : # retrieve a flag whether the entry was used for sorting.
            df = pd.read_csv( f"{path_folder_mtx_10x_input}{name_axis}.tsv.gz", sep = '\t', header = None )
            df.sort_values( 0, inplace = True ) # sort by id_barcode or id_feature
            
            # annotate and split the dataframe into a dataframe containing string representations and another dataframe containing numbers and categorical data, and save to Zarr objects separately
            df.columns = list( f"{name_axis}_{i}" for i in range( len( df.columns ) ) ) # name the columns using 0-based indices
            df_str = df.iloc[ :, : 2 ]
            df_num_and_cat = df.iloc[ :, 2 : ]
            del df 
            
            # write zarr object for random access of string representation of features/barcodes
            za = zarr.open( f'{path_folder_ramtx_output}{name_axis}.str.zarr', mode = 'w', shape = df_str.shape, chunks = ( int_num_of_entries_in_a_chunk_zarr_matrix_index, 1 ), dtype = str, synchronizer = zarr.ThreadSynchronizer( ) ) # multithreading? # string object # individual columns will be chucked, so that each column can be retrieved separately.
            za[ : ] = df_str.values
            # write random-access compatible format for web applications (#2022-06-20 10:57:51 currently there is no javascript packages supporting string zarr objects)
            if flag_used_for_sorting :
                WEB.Index_Chunks_and_Base64_Encode( df_to_be_chunked_and_indexed = df_str, int_num_rows_for_each_chunk = int_num_of_entries_in_a_chunk_zarr_matrix_index, path_prefix_output = f"{path_folder_ramtx_output}{name_axis}.str", path_folder_temp = path_temp, flag_delete_temp_folder = True, flag_include_header = False )
            del df_str
            
            # build a ZarrDataFrame object for random access of number and categorical data of features/barcodes
            zdf = ZarrDataFrame( f'{path_folder_ramtx_output}{name_axis}.num_and_cat.zdf', df = df_num_and_cat, int_num_rows = len( df_num_and_cat ), int_num_rows_in_a_chunk = int_num_of_entries_in_a_chunk_zarr_matrix_index, flag_store_string_as_categorical = True, flag_retrieve_categorical_data_as_integers = True, flag_enforce_name_col_with_only_valid_characters = False, flag_load_data_after_adding_new_column = False ) # use the same chunk size for all feature/barcode objects
            del df_num_and_cat

        ''' write metadata '''
        root = zarr.group( f'{path_folder_ramtx_output}' )
        root.attrs[ 'dict_metadata' ] = { 
            'path_folder_mtx_10x_input' : path_folder_mtx_10x_input,
            'flag_ramtx_sorted_by_id_feature' : flag_ramtx_sorted_by_id_feature,
            'str_completed_time' : TIME_GET_timestamp( True ),
            'int_num_features' : int_num_features,
            'int_num_barcodes' : int_num_barcodes,
            'int_num_records' : int_num_records,
            'int_num_of_records_in_a_chunk_zarr_matrix' : int_num_of_records_in_a_chunk_zarr_matrix,
            'int_num_of_entries_in_a_chunk_zarr_matrix_index' : int_num_of_entries_in_a_chunk_zarr_matrix_index,
            'version' : _version_,
        }

        ''' delete temporary folder '''
        if not flag_debugging :
            shutil.rmtree( path_temp )

        ''' write a flag indicating the export has been completed '''
        with open( path_file_flag, 'w' ) as file :
            file.write( TIME_GET_timestamp( True ) )
def Convert_MTX_10X_to_RamData( path_folder_mtx_10x_input, path_folder_ramdata_output, name_layer = 'raw', int_num_threads = 15, int_num_threads_for_splitting = 3, int_max_num_entries_for_chunk = 10000000, int_max_num_files_for_each_merge_sort = 5, dtype_mtx = np.float64, dtype_mtx_index = np.float64, int_num_of_records_in_a_chunk_zarr_matrix = 10000, int_num_of_entries_in_a_chunk_zarr_matrix_index = 1000, flag_simultaneous_indexing_of_cell_and_barcode = True, verbose = False, flag_debugging = False ) :
    """  # deprecated
    # 2022-07-08 02:00:14 
    convert 10X count matrix data to the two RAMtx object, one sorted by features and the other sorted by barcodes, and construct a RamData data object on disk, backed by Zarr persistant arrays

    inputs:
    ========
    'path_folder_mtx_10x_input' : an input directory of 10X matrix data
    'path_folder_ramdata_output' an output directory of RamData
    'name_layer' : a name of the given data layer
    'int_num_threads' : the number of threads for multiprocessing
    'dtype_mtx' (default: np.float64), dtype of the output zarr array for storing matrix
    'dtype_mtx_index' (default: np.float64) : dtype of the output zarr array for storing matrix indices
    'flag_simultaneous_indexing_of_cell_and_barcode' : if True, create cell-sorted RAMtx and feature-sorted RAMtx simultaneously using two worker processes with the half of given 'int_num_threads'. it is generally recommended to turn this feature on, since the last step of the merge-sort is always single-threaded.
    """
    # build barcode- and feature-sorted RAMtx objects
    path_folder_data = f"{path_folder_ramdata_output}{name_layer}/" # define directory of the output data
    if flag_simultaneous_indexing_of_cell_and_barcode :
        l_process = list( mp.Process( target = Convert_MTX_10X_to_RAMtx, args = ( path_folder_mtx_10x_input, path_folder_ramtx_output, flag_ramtx_sorted_by_id_feature, int_num_threads_for_the_current_process, int_num_threads_for_splitting, int_max_num_entries_for_chunk, int_max_num_files_for_each_merge_sort, dtype_mtx, dtype_mtx_index, int_num_of_records_in_a_chunk_zarr_matrix, int_num_of_entries_in_a_chunk_zarr_matrix_index, verbose, flag_debugging ) ) for path_folder_ramtx_output, flag_ramtx_sorted_by_id_feature, int_num_threads_for_the_current_process in zip( [ f"{path_folder_data}sorted_by_barcode/", f"{path_folder_data}sorted_by_feature/" ], [ False, True ], [ int( np.floor( int_num_threads / 2 ) ), int( np.ceil( int_num_threads / 2 ) ) ] ) )
        for p in l_process : p.start( )
        for p in l_process : p.join( )
    else :
        Convert_MTX_10X_to_RAMtx( path_folder_mtx_10x_input, path_folder_ramtx_output = f"{path_folder_data}sorted_by_barcode/", flag_ramtx_sorted_by_id_feature = False, int_num_threads = int_num_threads, int_num_threads_for_splitting = int_num_threads_for_splitting, int_max_num_entries_for_chunk = int_max_num_entries_for_chunk, int_max_num_files_for_each_merge_sort = int_max_num_files_for_each_merge_sort, dtype_mtx = dtype_mtx, dtype_mtx_index = dtype_mtx_index, int_num_of_records_in_a_chunk_zarr_matrix = int_num_of_records_in_a_chunk_zarr_matrix, int_num_of_entries_in_a_chunk_zarr_matrix_index = int_num_of_entries_in_a_chunk_zarr_matrix_index, verbose = verbose, flag_debugging = flag_debugging )
        Convert_MTX_10X_to_RAMtx( path_folder_mtx_10x_input, path_folder_ramtx_output = f"{path_folder_data}sorted_by_feature/", flag_ramtx_sorted_by_id_feature = True, int_num_threads = int_num_threads, int_num_threads_for_splitting = int_num_threads_for_splitting, int_max_num_entries_for_chunk = int_max_num_entries_for_chunk, int_max_num_files_for_each_merge_sort = int_max_num_files_for_each_merge_sort, dtype_mtx = dtype_mtx, dtype_mtx_index = dtype_mtx_index, int_num_of_records_in_a_chunk_zarr_matrix = int_num_of_records_in_a_chunk_zarr_matrix, int_num_of_entries_in_a_chunk_zarr_matrix_index = int_num_of_entries_in_a_chunk_zarr_matrix_index, verbose = verbose, flag_debugging = flag_debugging )

    # copy features/barcode.tsv.gz random access files for the web (stacked base64 encoded tsv.gz files)
    # copy features/barcode string representation zarr objects
    # copy features/barcode ZarrDataFrame containing number/categorical data
    for name_axis_singular in [ 'feature', 'barcode' ] :
        for str_suffix in [ 's.str.tsv.gz.base64.concatenated.txt', 's.str.index.tsv.gz.base64.txt', 's.str.zarr', 's.num_and_cat.zdf' ] :
            OS_Run( [ 'cp', '-r', f"{path_folder_data}sorted_by_{name_axis_singular}/{name_axis_singular}{str_suffix}", f"{path_folder_ramdata_output}{name_axis_singular}{str_suffix}" ] )
            
    # write metadata 
    int_num_features, int_num_barcodes, int_num_records = MTX_10X_Retrieve_number_of_rows_columns_and_records( path_folder_mtx_10x_input ) # retrieve metadata of the input 10X mtx
    root = zarr.group( path_folder_ramdata_output )
    root.attrs[ 'dict_metadata' ] = { 
        'path_folder_mtx_10x_input' : path_folder_mtx_10x_input,
        'str_completed_time' : TIME_GET_timestamp( True ),
        'int_num_features' : int_num_features,
        'int_num_barcodes' : int_num_barcodes,
        'int_num_of_records_in_a_chunk_zarr_matrix' : int_num_of_records_in_a_chunk_zarr_matrix,
        'int_num_of_entries_in_a_chunk_zarr_matrix_index' : int_num_of_entries_in_a_chunk_zarr_matrix_index,
        'layers' : [ name_layer ],
        'version' : _version_,
    }
    
""" above functions will be moved below eventually """

''' miscellaneous functions '''
def convert_numpy_dtype_number_to_number( e ) :
    """ # 2022-08-22 15:46:33 
    convert potentially numpy number to number. useful for JSON serialization, since it cannot serialize numbers in the numpy dtype  
    """
    if np.issubdtype( type( e ), np.floating ) :
        return float( e )
    elif np.issubdtype( type( e ), np.integer ) :
        return int( e )
    else :
        return e

''' methods for logging purposes '''
def installed_packages( ) :
    """ # 2022-08-09 02:09:07 
    display the installed packages of scelephant
    """
    df_installed_packages = bk.PD_Select( bk.PIP_List_Packages( ), index = [ 'igraph', 'biobookshelf', 'typing', 'zarr', 'numcodecs', 'anndata', 'scanpy', 'shelve', 'sklearn', 'tarfile', 'requests', 'shutil', 'numba', 'tqdm', 'umap', 'hdbscan', 'pgzip', 'scipy', 'pynndescent', 'leidenalg', 'sys', 'os', 'subprocess', 'subprocess', 'multiprocessing', 'ctypes', 'logging', 'inspect', 'copy', 'collections', 'ast', 'pickle', 'traceback', 'mmap', 'itertools', 'math', 'uuid', 'gc', 'time', 'heapq', 'datetime', 'json', 'numpy', 'pandas', 'matplotlib', 'requests', 'ftplib', 'urllib', 'importlib', 'bokeh', 'pysam', 'plotly', 'scanpy', 'bitarray', 'intervaltree', 'statsmodels', 'scipy', 'upsetplot' ] )
    display( df_installed_packages )
    return df_installed_packages 

''' methods for jupyter notebook interaction (IPython) '''
def html_from_dict( dict_data : dict, name_dict : str = None ) :
    """ # 2022-08-07 23:47:15 
    compose a html page displaying the given dicionary by converting the dictionary to JSON format and visualizing JSON format using jsonTreeViewer lightweight javascript package.
    the main purpose of this function is to provide an interactive interface for exploration of an object using jupyter notebook's _repr_html_ method.
    
    'dict_data' : a dictionary that contains JSON-like data
    'name_dict' : name of the dictionary
    """
    str_uuid = UUID( ) # retrieve a unique id for this function call. returned HTML document will contain DOM elements with unique ids
    return """
    <!DOCTYPE html>
    <html>
    <head>
    <link href="https://rawgit.com/summerstyle/jsonTreeViewer/master/libs/jsonTree/jsonTree.css" rel="stylesheet" />
    <script src="https://rawgit.com/summerstyle/jsonTreeViewer/master/libs/jsonTree/jsonTree.js"></script>
    <style>
    #wrapper_""" + str_uuid + """ li {
      list-style:none;
    }
    </style>
    </head>
    <body>
    <div>
    """ + ( '' if name_dict is None else f"{name_dict}" ) + """
    <div id="wrapper_""" + str_uuid +  """"></div>
    </div>


    <script>
    // Get DOM-element for inserting json-tree
    var wrapper = document.getElementById("wrapper_""" + str_uuid + """");

    // Get json-data by javascript-object
    var data = """ + json.dumps( dict_data ) +  """

    var tree = jsonTree.create(data, wrapper);
    </script></body></html>"""
    
''' methods for handling tar.gz file '''
def tar_create( path_file_output, path_folder_input ) :
    ''' # 2022-08-05 21:07:53 
    create tar.gz file
    
    'path_file_output' : output tar.gz file
    'path_folder_input' : input folder for creation of a tar.gz file
    '''
    with tarfile.open( path_file_output, "w:gz" ) as tar :
        tar.add( path_folder_input, arcname = os.path.basename( path_folder_input ) )
def tar_extract( path_file_input, path_folder_output ) :
    ''' # 2022-08-05 21:07:53 
    extract tar.gz file
    
    'path_file_output' : output tar.gz file
    'path_folder_input' : input folder for creation of a tar.gz file
    '''
    with tarfile.open( path_file_input, "r:gz" ) as tar :
        tar.extractall( path_folder_output )

''' methods for handling remote file '''
def http_response_code( url ) :
    """ # 2022-08-05 22:27:27 
    check http response code
    """
    status_code = None # by default, 'status_code' is None
    try:
        r = requests.head( url )
        status_code = r.status_code # record the status header
    except requests.ConnectionError:
        status_code = None
    return status_code
def download_file( url, path_file_local ) :
    """ # 2022-08-05 22:14:30 
    download file from the remote location to the local directory
    """
    with requests.get( url, stream = True ) as r :
        with open( local_filename, 'wb' ) as f :
            shutil.copyfileobj( r.raw, f )

''' memory-efficient methods for creating RAMtx/RamData object '''
# latest 2022-07-28 11:31:12 
# implementation using pipe (~3 times more efficient)
def create_stream_from_a_gzip_file_using_pipe( path_file_gzip, pipe_sender, func, int_buffer_size = 100 ) :
    ''' # 2022-07-27 06:50:29 
    parse and decorate mtx record for sorting. the resulting records only contains two values, index of axis that were not indexed and the data value, for more efficient pipe operation
    return a generator yielding ungziped records
    
    'path_file_gzip' : input file gzip file to create stream of decorated mtx record 
    'pipe_sender' : pipe for retrieving decorated mtx records. when all records are parsed, None will be given.
    'func' : a function for transforming each 'line' in the input gzip file to a (decorated) record. if None is returned, the line will be ignored and will not be included in the output stream.
    'int_buffer_size' : the number of entries for each batch that will be given to 'pipe_sender'. increasing this number will reduce the overhead associated with interprocess-communication through pipe, but will require more memory usage
    
    returns:
    return the process that will be used for unzipping the input gzip file and creating a stream.
    '''
    # handle arguments
    int_buffer_size = int( max( 1, int_buffer_size ) ) 
    
    # define a function for doing the work
    def __gunzip( path_file_gzip, pipe_sender, int_buffer_size ) :
        """ # 2022-07-25 22:22:33 
        unzip gzip file and create a stream using the given pipe
        """
        with gzip.open( path_file_gzip, "rt" ) as file :
            l_buffer = [ ] # initialize the buffer
            for line in file :
                rec = func( line ) # convert gzipped line into a decorated record
                if rec is not None : # if the transformed record is valid
                    l_buffer.append( rec ) # add a parsed record to the buffer
                    
                if len( l_buffer ) >= int_buffer_size : # if the buffer is full
                    pipe_sender.send( l_buffer ) # send a list of record of a given buffer size
                    l_buffer = [ ] # initialize the next buffer
            if len( l_buffer ) > 0 : # flush remaining buffer
                pipe_sender.send( l_buffer ) 
        pipe_sender.send( None )
    p = mp.Process( target = __gunzip, args = ( path_file_gzip, pipe_sender, int_buffer_size ) )
    return p # return the process
def concurrent_merge_sort_using_pipe( pipe_sender, * l_pipe_receiver, int_max_num_pipe_for_each_worker = 8, int_buffer_size = 100 ) :
    """ # 2022-07-27 06:50:22 
    inputs:
    'pipe_sender' : a pipe through which sorted decorated mtx records will be send. when all records are parsed, None will be given.
    'l_pipe_receiver' : list of pipes through which decorated mtx records will be received. when all records are parsed, these pipes should return None.
    'int_max_num_pipe_for_each_worker' : maximum number of input pipes for each worker. this argument and the number of input pipes together will determine the number of threads used for sorting
    'int_buffer_size' : the number of entries for each batch that will be given to 'pipe_sender'. increasing this number will reduce the overhead associated with interprocess-communication through pipe, but will require more memory usage

    returns:
    l_p : list of processes for all the workers that will be used for sorting
    """
    # parse arguments
    int_max_num_pipe_for_each_worker = int( int_max_num_pipe_for_each_worker )
    # handle when no input pipes are given
    if len( l_pipe_receiver ) == 0 :
        pipe_sender.send( None ) # notify the end of records
        return -1
    
    int_num_merging_layers = int( np.ceil( math.log( len( l_pipe_receiver ), int_max_num_pipe_for_each_worker ) ) ) # retrieve the number of merging layers.
    def __pipe_receiver_to_iterator( pipe_receiver ) :
        ''' # 2022-07-25 00:59:22 
        convert pipe_receiver to iterator 
        '''
        while True :
            l_r = pipe_receiver.recv( ) # retrieve a batch of records
            # detect pipe_receiver
            if l_r is None :
                break
            for r in l_r : # iterate through record by record, and yield each record
                yield r
    def __sorter( pipe_sender, * l_pipe_receiver ) :
        """ # 2022-07-25 00:57:56 
        """
        # handle when no input pipes are given
        if len( l_pipe_receiver ) == 0 :
            pipe_sender.send( None )
            
        # perform merge sorting
        l_buffer = [ ] # initialize a buffer
        for r in heapq.merge( * list( map( __pipe_receiver_to_iterator, l_pipe_receiver ) ) ) : # convert pipe to iterator
            l_buffer.append( r ) # add a parsed record to the buffer
            # flush the buffer
            if len( l_buffer ) >= int_buffer_size : # if the buffer is full
                pipe_sender.send( l_buffer ) # return record in a sorted order
                l_buffer = [ ] # initialize the buffer
        if len( l_buffer ) > 0 : # if there are some records remaining in the buffer
            pipe_sender.send( l_buffer ) # send the buffer
        pipe_sender.send( None ) # notify the end of records
    l_p = [ ] # initialize the list that will contain all the processes that will be used for sorting.
    while len( l_pipe_receiver ) > int_max_num_pipe_for_each_worker : # perform merge operations for each layer until all input pipes can be merged using a single worker (perform merge operations for all layers except for the last layer)
        l_pipe_receiver_for_the_next_layer = [ ] # initialize the list of receiving pipes for the next layer, which will be collected while initializing workers for the current merging layer 
        for index_worker_in_a_layer in range( int( np.ceil( len( l_pipe_receiver ) / int_max_num_pipe_for_each_worker ) ) ) : # iterate through the workers of the current merging layer
            pipe_sender_for_a_worker, pipe_receiver_for_a_worker = mp.Pipe( )
            l_pipe_receiver_for_the_next_layer.append( pipe_receiver_for_a_worker ) # collect receiving end of pipes for the initiated workers
            l_p.append( mp.Process( target = __sorter, args = [ pipe_sender_for_a_worker ] + list( l_pipe_receiver[ index_worker_in_a_layer * int_max_num_pipe_for_each_worker : ( index_worker_in_a_layer + 1 ) * int_max_num_pipe_for_each_worker ] ) ) )
        l_pipe_receiver = l_pipe_receiver_for_the_next_layer # initialize the next layer
    # retrieve a worker for the last merging layer
    l_p.append( mp.Process( target = __sorter, args = [ pipe_sender ] + list( l_pipe_receiver ) ) )
    return l_p # return the list of processes
def write_stream_as_a_gzip_file_using_pipe( pipe_receiver, path_file_gzip, func, compresslevel = 6, int_num_threads = 1, int_buffer_size = 100, header = None ) :
    ''' # 2022-07-27 06:50:14 
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
    'int_num_threads' : the number of threads for gzip writer. if 'int_num_threads' > 1, pgzip will be used to write the output gzip file. please note that pgzip (multithreaded version of gzip module) has some memory-leaking issue.
    'int_buffer_size' : the number of entries for each batch that will be given to 'pipe_sender'. increasing this number will reduce the overhead associated with interprocess-communication through pipe, but will require more memory usage
    
    returns:
    return the process that will be used for gzipping the input stream
    '''
    # handle input arguments
    int_num_threads = int( max( 1, int_num_threads ) )
    # define a function for doing the work
    def __gzip( pipe_receiver, path_file_gzip, func ) :
        """ # 2022-07-25 22:22:33 
        unzip gzip file and create a stream using the given pipe
        """
        with gzip.open( path_file_gzip, "wt", compresslevel = compresslevel ) if int_num_threads <= 1 else pgzip.open( path_file_gzip, "wt", compresslevel = compresslevel, thread = int_num_threads ) as newfile : # open the output file
            if header is not None : # if a valid header is given, write the header
                newfile.write( header )
            while True :
                l_r = pipe_receiver.recv( ) # retrieve record
                if l_r is None : # handle when all records are parsed
                    break
                l_buffer = [ ] # initialize the buffer
                for r in l_r : # iterate through the list of records
                    l_buffer.append( func( r ) ) # collect the output
                    # write the output file
                    if len( l_buffer ) >= int_buffer_size : # when the buffer is full, flush the buffer
                        newfile.write( ''.join( l_buffer ) ) # write the output
                        l_buffer = [ ] # initialize the buffer
                if len( l_buffer ) > 0 : # flush the buffer
                    newfile.write( ''.join( l_buffer ) ) # flush the buffer
    p = mp.Process( target = __gzip, args = ( pipe_receiver, path_file_gzip, func ) )
    return p # return the process
def write_stream_as_a_sparse_ramtx_zarr_using_pipe( pipe_receiver, za_mtx, za_mtx_index ) :
    ''' # 2022-07-29 09:54:57 
    write a stream of decorated mtx records to a sparse ramtx zarr object, sorted by barcodes or features (and its associated index)
        
    arguments:
    'pipe_receiver' : pipe for retrieving decorated mtx records. when all records are parsed, None should be given.
    'za_mtx', 'za_mtx_index' : output zarr objects

    returns:
    return the list of processes that will be used for building a ramtx zarr from the input stream
    '''
    # retrieve settings
    dtype_mtx = za_mtx.dtype
    dtype_mtx_index = za_mtx_index.dtype
    # set buffer size
    int_buffer_size_mtx_index = za_mtx_index.chunks[ 0 ] * 10
    int_buffer_size_mtx = za_mtx.chunks[ 0 ] * 10
    
    # define a function for doing the work
    def __write_zarr( pipe_receiver, za ) :
        """ # 2022-07-29 09:41:08 
        write an array of a specific coordinates to a given zarr object
        """
        while True :
            r = pipe_receiver.recv( )
            if r is None : # exist once the stream ends
                break
            st, en, arr = r # parse the received record
            za[ st : en ] = arr # write zarr object
    def __compose_array( pipe_receiver, pipe_sender_to_mtx_writer, pipe_sender_to_mtx_index_writer ) :
        """ # 2022-07-25 22:22:33 
        convert a given stream into ramtx arrays (mtx and mtx index)
        """
        # perform merge sorting
        int_entry_currently_being_written = None # place holder value
        int_num_mtx_records_written = 0
        l_mtx_record = [ ]
        int_num_mtx_index_records_written = 0
        l_mtx_index = [ ]

        # iterate through records
        while True :
            l_r = pipe_receiver.recv( ) # retrieve record
            if l_r is None : # handle when all records are parsed
                break
            for r in l_r : # iterate through the list of records
                int_entry_of_the_current_record, mtx_record = r
                if int_entry_currently_being_written is None :
                    int_entry_currently_being_written = int_entry_of_the_current_record # update current int_entry
                elif int_entry_currently_being_written != int_entry_of_the_current_record : # if current int_entry is different from the previous one, which mark the change of sorted blocks (a block has the same id_entry), save the data for the previous block to the output zarr and initialze data for the next block 
                    ''' compose index '''
                    l_mtx_index.append( [ int_num_mtx_records_written, int_num_mtx_records_written + len( l_mtx_record ) ] ) # collect information required for indexing # add records to mtx_index
                    if int_entry_currently_being_written + 1 > int_entry_of_the_current_record :
                        for int_entry in range( int_entry_currently_being_written + 1, int_entry_of_the_current_record ) : # for the int_entry that lack count data and does not have indexing data, put place holder values
                            l_mtx_index.append( [ 0, 0 ] ) # put place holder values for int_entry lacking count data.
                    int_entry_currently_being_written = int_entry_of_the_current_record # update current int_entry    
                    
                    ''' check mtx buffer and flush '''
                    if len( l_mtx_record ) >= int_buffer_size_mtx : # if buffer is full
                        pipe_sender_to_mtx_writer.send( ( int_num_mtx_records_written, int_num_mtx_records_written + len( l_mtx_record ), np.array( l_mtx_record, dtype = dtype_mtx ) ) ) # send data to the zarr mtx writer
                        int_num_mtx_records_written += len( l_mtx_record )
                        l_mtx_record = [ ] # reset buffer
                    
                    ''' check mtx index buffer and flush '''
                    if len( l_mtx_index ) >= int_buffer_size_mtx_index : # if buffer is full
                        pipe_sender_to_mtx_index_writer.send( ( int_num_mtx_index_records_written, int_num_mtx_index_records_written + len( l_mtx_index ), np.array( l_mtx_index, dtype = dtype_mtx_index ) ) ) # send data to the zarr mtx index writer
                        int_num_mtx_index_records_written += len( l_mtx_index ) # update 'int_num_mtx_index_records_written'
                        l_mtx_index = [ ] # reset buffer
                # collect mtx record
                l_mtx_record.append( mtx_record )

        # write the record for the last entry
        assert len( l_mtx_record ) > 0 # there SHOULD be a last entry
        ''' compose index '''
        l_mtx_index.append( [ int_num_mtx_records_written, int_num_mtx_records_written + len( l_mtx_record ) ] ) # collect information required for indexing # add records to mtx_index
        for int_entry in range( int_entry_currently_being_written + 1, za_mtx_index.shape[ 0 ] ) : # for the int_entry that lack count data and does not have indexing data, put place holder values # set 'next' int_entry to the end of the int_entry values so that place holder values can be set to the missing int_entry 
            l_mtx_index.append( [ 0, 0 ] ) # put place holder values for int_entry lacking count data.

        ''' flush mtx buffer '''
        pipe_sender_to_mtx_writer.send( ( int_num_mtx_records_written, int_num_mtx_records_written + len( l_mtx_record ), np.array( l_mtx_record, dtype = dtype_mtx ) ) ) # send data to the zarr mtx writer

        ''' flush mtx index buffer '''
        pipe_sender_to_mtx_index_writer.send( ( int_num_mtx_index_records_written, int_num_mtx_index_records_written + len( l_mtx_index ), np.array( l_mtx_index, dtype = dtype_mtx_index ) ) ) # send data to the zarr mtx index writer

        ''' send termination signals '''
        pipe_sender_to_mtx_writer.send( None )
        pipe_sender_to_mtx_index_writer.send( None )
        
    # create pipes for communications between the processes
    pipe_sender_to_mtx_writer, pipe_receiver_to_mtx_writer = mp.Pipe( )
    pipe_sender_to_mtx_index_writer, pipe_receiver_to_mtx_index_writer = mp.Pipe( )
    # compose the list of processes
    l_p = [ ]
    l_p.append( mp.Process( target = __compose_array, args = ( pipe_receiver, pipe_sender_to_mtx_writer, pipe_sender_to_mtx_index_writer ) ) )
    l_p.append( mp.Process( target = __write_zarr, args = ( pipe_receiver_to_mtx_writer, za_mtx ) ) )
    l_p.append( mp.Process( target = __write_zarr, args = ( pipe_receiver_to_mtx_index_writer, za_mtx_index ) ) )
    return l_p # return the list of processes
def concurrent_merge_sort_using_pipe_mtx( path_file_output = None, l_path_file = [ ], flag_mtx_sorted_by_id_feature = True, int_buffer_size = 100, compresslevel = 6, int_max_num_pipe_for_each_worker = 8, flag_dtype_is_float = False, flag_return_processes = False, int_num_threads = 1, flag_delete_input_files = False, header = None, za_mtx = None, za_mtx_index = None ) :
    """ # 2022-07-27 06:50:06 
    
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
    if len( l_path_file ) == 0 : # if the list of input files are empty, exit
        return
    if not ( path_file_output is not None or ( za_mtx is not None and za_mtx_index is not None ) ) : # if all output are invalid, exit
        return
    def __decode_mtx( line ) :
        """ # 2022-07-27 00:28:42 
        decode a line and return a parsed line (record)
        """
        ''' skip comment lines '''
        if line[ 0 ] == '%' :
            return None
        ''' parse a mtx record '''
        index_row, index_column, float_value = line.strip( ).split( ) # parse a record of a matrix-market format file
        index_row, index_column, float_value = int( index_row ) - 1, int( index_column ) - 1, float( float_value ) # 1-based > 0-based coordinates
        # return record with decoration according to the sorted axis # return 0-based coordinates
        if flag_mtx_sorted_by_id_feature :
            res = index_row, ( index_column, float_value ) 
        else :
            res = index_column, ( index_row, float_value )
        return res
    convert_to_output_dtype = float if flag_dtype_is_float else int
    def __encode_mtx( r ) :
        """ # 2022-07-27 00:31:27 
        encode parsed record into a line (in an original format)
        """
        dec, rec = r # retrieve decorator and the remaining record
        if flag_mtx_sorted_by_id_feature : 
            index_row = dec
            index_column, val = rec
        else :
            index_column = dec
            index_row, val = rec
        val = convert_to_output_dtype( val ) # convert to the output dtype
        return str( index_row + 1 ) + ' ' + str( index_column + 1 ) + ' ' + str( val ) + '\n' # return the output
    
    # construct and collect the processes for the parsers
    l_p = [ ]
    l_pipe_receiver = [ ]
    for index_file, path_file in enumerate( l_path_file ) :
        pipe_sender, pipe_receiver = mp.Pipe( )
        p = create_stream_from_a_gzip_file_using_pipe( path_file, pipe_sender, func = __decode_mtx, int_buffer_size = int_buffer_size )
        l_p.append( p )
        l_pipe_receiver.append( pipe_receiver )
    
    # construct and collect the processes for a concurrent merge sort tree and writer
    pipe_sender, pipe_receiver = mp.Pipe( ) # create a link
    l_p.extend( concurrent_merge_sort_using_pipe( pipe_sender, * l_pipe_receiver, int_max_num_pipe_for_each_worker = int_max_num_pipe_for_each_worker, int_buffer_size = int_buffer_size ) )
    if path_file_output is not None : # when an output file is an another gzip file
        l_p.append( write_stream_as_a_gzip_file_using_pipe( pipe_receiver, path_file_output, func = __encode_mtx, compresslevel = compresslevel, int_num_threads = int_num_threads, int_buffer_size = int_buffer_size, header = header ) )
    else : # when the output is a ramtx zarr object
        l_p.extend( write_stream_as_a_sparse_ramtx_zarr_using_pipe( pipe_receiver, za_mtx, za_mtx_index ) )

    if flag_return_processes : 
        # simply return the processes
        return l_p
    else :
        # run the processes
        for p in l_p : p.start( )
        for p in l_p : p.join( )
            
    # delete input files if 'flag_delete_input_files' is True
    if flag_delete_input_files :
        for path_file in l_path_file :
            os.remove( path_file )
    
    if path_file_output is not None : # when an output file is an another gzip file # return the path to the output file
        return path_file_output

# for sorting mtx and creating RamData
def create_and_sort_chunk( path_file_gzip, path_prefix_chunk, func_encoding, func_decoding, pipe_sender, func_detect_header = None, int_num_records_in_a_chunk = 10000000, int_num_threads_for_sorting_and_writing = 5, int_buffer_size = 300 ) :
    """ # 2022-07-28 11:07:36 
    split an input gzip file into smaller chunks and sort individual chunks.
    returns a list of processes that will perform the operation.
    
    'path_file_gzip' : file path of an input gzip file
    'path_prefix_chunk' : a prefix for the chunks that will be written.
    'func_encoding' : a function for transforming a decorated record into a line in a gzip file.
    'func_decoding' : a function for transforming a line in a gzip file into a decorated record. the lines will be sorted according to the first element of the returned records. the first element (key) should be float/integers (numbers)
    'pipe_sender' : a pipe that will be used to return the list of file path of the created chunks. when all chunks are created, None will be given.
    'func_detect_header' : a function for detecting header lines in a gzip file. the opened gzip file will be given ('rw' mode) to the function and the funciton should consume all header lines. optionally, the function can return a line that are the start of the record if the algorithm required to read non-header line to detect the end of header.
    'int_num_records_in_a_chunk' : the number of maximum records in a chunk
    'int_num_threads_for_sorting_and_writing' : number of workers for sorting and writing operations. the number of worker for reading the input gzip file will be 1.
    'int_buffer_size' : the number of entries for each batch that will be given to 'pipe_sender'. increasing this number will reduce the overhead associated with interprocess-communication through pipe, but will require more memory usage

    """
    # handle arguments
    int_buffer_size = int( max( 1, int_buffer_size ) ) 
    
    def __sort_and_write( pipe_receiver_record, pipe_sender_file_path ) :
        """ # 2022-07-27 09:07:26 
        receive records for a chunk, sort the records, and write encoded records as a gzip file
        """
        l_r = [ ] # initialize a list for collecting records
        while True :
            b = pipe_receiver_record.recv( )
            if b is None : 
                break
            if isinstance( b, str ) :
                newfile = gzip.open( b, 'wt' ) # open a new file using the name
                
                ''' sort records '''
                int_num_records = len( l_r )
                arr_key = np.zeros( int_num_records, dtype = float ) # the first element (key) should be float/integers (numbers)
                for i, r in enumerate( l_r ) :
                    arr_key[ i ] = r[ 0 ]
                l_r = np.array( l_r, dtype = object )[ arr_key.argsort( ) ] # sort records by keys
                del arr_key
                
                for r in l_r :
                    newfile.write( func_encoding( r ) )
                
                l_r = [ ] # initialize a list for collecting records
                newfile.close( ) # close an output file
                pipe_sender_file_path.send( b ) # send the file path of written chunk
            else :
                # collect record from the buffer
                for r in b :
                    l_r.append( r )
                    
    # initialize
    l_p = [ ] # initialize the list of processes
    l_pipe_sender_record = [ ] # collect the pipe for distributing records
    l_pipe_receiver_file_path = [ ] # collect the pipe for receiving file path
    # compose processes for sorting and writing chunks
    for index_process in range( int_num_threads_for_sorting_and_writing ) :
        pipe_sender_record, pipe_receiver_record = mp.Pipe( ) # create a pipe for sending parsed records
        pipe_sender_file_path, pipe_receiver_file_path = mp.Pipe( ) # create a pipe for sending file path of written chunks
        l_pipe_sender_record.append( pipe_sender_record )
        l_pipe_receiver_file_path.append( pipe_receiver_file_path )
        l_p.append( mp.Process( target = __sort_and_write, args = ( pipe_receiver_record, pipe_sender_file_path ) ) )
        
    # define a function for reading gzip file
    def __gunzip( path_file_gzip, path_prefix_chunk, pipe_sender, int_num_records_in_a_chunk, int_buffer_size ) :
        """ # 2022-07-25 22:22:33 
        unzip gzip file, distribute records across workers, and collect the file path of written chunks
        """
        int_num_workers = len( l_pipe_sender_record ) # retrieve the number of workers
        arr_num_files = np.zeros( int_num_workers, dtype = int ) # initialize an array indicating how many files each worker processes need to write
        
        def __collect_file_path( ) :
            ''' # 2022-07-27 09:48:04 
            collect file paths of written chunks from workers and report the file path using 'pipe_sender' 
            '''
            while True : 
                for index_worker, pipe_receiver_file_path in enumerate( l_pipe_receiver_file_path ) :
                    if pipe_receiver_file_path.poll( ) :
                        path_file = pipe_receiver_file_path.recv( )
                        pipe_sender.send( path_file )
                        arr_num_files[ index_worker ] -= 1 # update the number of files for the process                    
                        assert arr_num_files[ index_worker ] >= 0
                        
                # if all workers has less than two files to write, does not block
                if ( arr_num_files <= 2 ).sum( ) == int_num_workers :
                    break
                time.sleep( 1 ) # sleep for one second before collecting completed works
        
        # iterate through lines in the input gzip file and assign works to the workers
        with gzip.open( path_file_gzip, "rt" ) as file :
            l_buffer = [ ] # initialize the buffer
            int_num_sent_records = 0 # initialize the number of send records
            index_worker = 0 # initialize the worker for receiving records
            
            # detect header from the start of the file
            if func_detect_header is not None and hasattr( func_detect_header, '__call__' ) : # if a valid function for detecting header has been given
                line = func_detect_header( file ) # consume header lines
                if line is None : # if exactly header lines are consumed and no actual records were consumed from the file, read the first record
                    line = file.readline( )
            # iterate lines of the rest of the gzip file    
            while True :
                r = func_decoding( line ) # convert gzipped line into a decorated record
                if r is not None : # if the transformed record is valid
                    l_buffer.append( r ) # add a parsed record to the buffer
                
                if len( l_buffer ) >= int_buffer_size : # if the buffer is full
                    l_pipe_sender_record[ index_worker ].send( l_buffer ) # send a list of record of a given buffer size
                    int_num_sent_records += len( l_buffer ) # update 'int_num_sent_records'
                    l_buffer = [ ] # initialize the next buffer
                elif int_num_sent_records + len( l_buffer ) >= int_num_records_in_a_chunk :
                    l_pipe_sender_record[ index_worker ].send( l_buffer ) # send a list of records
                    int_num_sent_records += len( l_buffer ) # update 'int_num_sent_records'
                    l_buffer = [ ] # initialize the next buffer
                    int_num_sent_records = 0 # initialize the next chunk
                    l_pipe_sender_record[ index_worker ].send( f"{path_prefix_chunk}.{UUID( )}.gz" ) # assign the file path of the chunk
                    arr_num_files[ index_worker ] += 1 # update the number of files for the process
                    index_worker = ( 1 + index_worker ) % int_num_workers  # update the index of the worker
                    __collect_file_path( ) # collect and report file path
                line = file.readline( ) # read the next line
                if len( line ) == 0 :
                    break
                
        if len( l_buffer ) > 0 : # if there is some buffer remaining, flush the buffer
            l_pipe_sender_record[ index_worker ].send( l_buffer ) # send a list of records
            l_pipe_sender_record[ index_worker ].send( f"{path_prefix_chunk}.{UUID( )}.gz" ) # assign the file path of the chunk
            arr_num_files[ index_worker ] += 1 # update the number of files for the process
            __collect_file_path( ) # collect and report file path

        # wait until all worker processes complete writing files
        while arr_num_files.sum( ) > 0 :
            time.sleep( 1 )
            __collect_file_path( ) # collect and report file path
            
        # terminate the worker processes
        for pipe_sender_record in l_pipe_sender_record :
            pipe_sender_record.send( None )
            
        pipe_sender.send( None ) # notify that the process has been completed
    l_p.append( mp.Process( target = __gunzip, args = ( path_file_gzip, path_prefix_chunk, pipe_sender, int_num_records_in_a_chunk, int_buffer_size ) ) )
    return l_p # return the list of processes
def sort_mtx( path_file_gzip, path_file_gzip_sorted = None, int_num_records_in_a_chunk = 10000000, int_buffer_size = 300, compresslevel = 6, flag_dtype_is_float = False, flag_mtx_sorted_by_id_feature = True, int_num_threads_for_chunking = 5, int_num_threads_for_writing = 1, int_max_num_input_files_for_each_merge_sort_worker = 8, int_num_chunks_to_combine_before_concurrent_merge_sorting = 8, za_mtx = None, za_mtx_index = None ) :
    """ # 2022-07-28 11:07:44 
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
    if path_file_gzip_sorted is None :
        assert za_mtx is not None and za_mtx_index is not None # if ramtx zarr object will be written, both arguments should be valid
    # create a temporary folder
    path_folder = path_file_gzip.rsplit( '/', 1 )[ 0 ] + '/'
    path_folder_temp = f"{path_folder}temp_{UUID( )}/"
    os.makedirs( path_folder_temp, exist_ok = True )
    
    # create and sort chunks
    def __detect_header_mtx( file ) :
        """ # 2022-07-28 10:21:15 
        detect header lines from mtx file
        """
        line = file.readline( )
        if len( line ) > 0 and line[ 0 ] == '%' : # if comment lines are detected 
            # read comment and the description line of a mtx file
            while True :
                if line[ 0 ] != '%' : # once a description line was consumed, exit the function
                    break
                line = file.readline( ) # read the next line
            return None
        else : # if no header was detected, return a consumed line
            return line
    def __decode_mtx( line ) :
        """ # 2022-07-27 00:28:42 
        decode a line and return a parsed line (record)
        """
        ''' parse a mtx record '''
        try :
            index_row, index_column, float_value = line.strip( ).split( ) # parse a record of a matrix-market format file
        except :
            return None
        index_row, index_column, float_value = int( index_row ) - 1, int( index_column ) - 1, float( float_value ) # 1-based > 0-based coordinates
        # return record with decoration according to the sorted axis # return 0-based coordinates
        if flag_mtx_sorted_by_id_feature :
            res = index_row, ( index_column, float_value ) 
        else :
            res = index_column, ( index_row, float_value )
        return res
    convert_to_output_dtype = float if flag_dtype_is_float else int
    def __encode_mtx( r ) :
        """ # 2022-07-27 00:31:27 
        encode parsed record into a line (in an original format)
        """
        dec, rec = r # retrieve decorator and the remaining record
        if flag_mtx_sorted_by_id_feature : 
            index_row = dec
            index_column, val = rec
        else :
            index_column = dec
            index_row, val = rec
        val = convert_to_output_dtype( val ) # convert to the output dtype
        return str( index_row + 1 ) + ' ' + str( index_column + 1 ) + ' ' + str( val ) + '\n' # return the output
    pipe_sender, pipe_receiver = mp.Pipe( ) # create a link
    l_p = create_and_sort_chunk( path_file_gzip, f"{path_folder_temp}chunk", __encode_mtx, __decode_mtx, pipe_sender, func_detect_header = __detect_header_mtx, int_num_records_in_a_chunk = int_num_records_in_a_chunk, int_num_threads_for_sorting_and_writing = int_num_threads_for_chunking, int_buffer_size = int_buffer_size ) # retrieve processes
    for p in l_p : p.start( ) # start chunking
    
    l_path_file_for_concurrent_merge_sorting = [ ]
    l_path_file_chunk_for_merging = [ ]
    dict_process = dict( )
    
    def __combine_chunks( path_file_output, l_path_file, pipe_sender ) :
        """ # 2022-07-27 22:05:31 
        merge sort a given list of chunks and return a signal once operation has been completed
        """
        pipe_sender.send( concurrent_merge_sort_using_pipe_mtx( path_file_output = path_file_output, l_path_file = l_path_file, flag_mtx_sorted_by_id_feature = flag_mtx_sorted_by_id_feature, int_buffer_size = int_buffer_size, compresslevel = compresslevel, int_max_num_pipe_for_each_worker = int_max_num_input_files_for_each_merge_sort_worker, flag_dtype_is_float = flag_dtype_is_float, flag_return_processes = False, int_num_threads = int_num_threads_for_writing, flag_delete_input_files = True ) )
    def __run_combine_chunks( l_path_file ) :
        pipe_sender, pipe_receiver = mp.Pipe( ) # create a link
        path_file_output = f"{path_folder_temp}combined_chunk.{UUID( )}.gz"
        p = mp.Process( target = __combine_chunks, args = ( path_file_output, l_path_file, pipe_sender ) )
        dict_process[ UUID( ) ] = { 'p' : p, 'pipe_receiver' : pipe_receiver, 'path_file_output' : path_file_output } # collect the process
        p.start( ) # start the process
    def __collect_files_for_concurrent_merge_sorting( ) :
        for id_process in list( dict_process ) :
            if dict_process[ id_process ][ 'pipe_receiver' ].poll( ) : # if the process has been completed
                path_file_output = dict_process[ id_process ][ 'pipe_receiver' ].recv( ) # receive output
                assert path_file_output is not None # check whether the merge sort was successful
                l_path_file_for_concurrent_merge_sorting.append( path_file_output ) # collect the file path of the larger chunk
                dict_process[ id_process ][ 'p' ].join( )
                del dict_process[ id_process ] # remove the process from the dictionary of running processes
    ''' merge created chunks into larger chunks while chunking is completed '''
    while True :
        if not pipe_receiver.poll( ) :
            time.sleep( 1 ) # sleep for 1 second
        else : # if an input is available
            path_file_chunk = pipe_receiver.recv( )
            # if all chunks are created, exit
            if path_file_chunk is None :
                break
            # collect file path of chunks, and combine these smaller chunks into larger chunks for concurrent merge sort operation
            if int_num_chunks_to_combine_before_concurrent_merge_sorting == 1 : # when 'int_num_chunks_to_combine_before_concurrent_merge_sorting' == 1, the small chunks will be directly used for concurrent merge sort.
                l_path_file_for_concurrent_merge_sorting.append( path_file_chunk )
            else :
                l_path_file_chunk_for_merging.append( path_file_chunk ) # collect file path of chunks
                if len( l_path_file_chunk_for_merging ) >= int_num_chunks_to_combine_before_concurrent_merge_sorting : # if the number of collected chunks reaches the number that could be combined into a larger chunk
                    __run_combine_chunks( l_path_file_chunk_for_merging ) # combine chunks into a larger chunk
                    l_path_file_chunk_for_merging = [ ] # initialize the list for the next run
                __collect_files_for_concurrent_merge_sorting( ) # collect file path of chunks for concurrent merge sorting
                    
    if len( l_path_file_chunk_for_merging ) > 0 :
        l_path_file_for_concurrent_merge_sorting.extend( l_path_file_chunk_for_merging )
        
    while len( dict_process ) > 0 : # wait until all preliminary merge sort operation on created chunks are completed
        __collect_files_for_concurrent_merge_sorting( ) # collect files for concurrent merge sorting
        time.sleep( 1 ) # sleep for 1 second
    
    if path_file_gzip_sorted is not None : # if an ouptut file is another mtx.gz file
        # retrieve metadata from the input mtx file
        int_num_rows, int_num_columns, int_num_records = MTX_10X_Retrieve_number_of_rows_columns_and_records( path_file_gzip )
        header = f"""%%MatrixMarket matrix coordinate integer general\n%\n{int_num_rows} {int_num_columns} {int_num_records}\n""" # compose a header

        # perform merge sorting preliminarily merge-sorted chunks into a single sorted output file
        os.makedirs( path_file_gzip_sorted.rsplit( '/', 1 )[ 0 ], exist_ok = True ) # create an output folder
        concurrent_merge_sort_using_pipe_mtx( path_file_gzip_sorted, l_path_file_for_concurrent_merge_sorting, flag_mtx_sorted_by_id_feature = flag_mtx_sorted_by_id_feature, int_buffer_size = int_buffer_size, compresslevel = compresslevel, int_max_num_pipe_for_each_worker = int_max_num_input_files_for_each_merge_sort_worker, flag_dtype_is_float = flag_dtype_is_float, flag_return_processes = False, int_num_threads = int_num_threads_for_writing, flag_delete_input_files = True, header = header ) # write matrix market file header
    else : # if an output is a ramtx zarr object
        concurrent_merge_sort_using_pipe_mtx( l_path_file = l_path_file_for_concurrent_merge_sorting, flag_mtx_sorted_by_id_feature = flag_mtx_sorted_by_id_feature, int_buffer_size = int_buffer_size, compresslevel = compresslevel, int_max_num_pipe_for_each_worker = int_max_num_input_files_for_each_merge_sort_worker, flag_dtype_is_float = flag_dtype_is_float, flag_return_processes = False, int_num_threads = int_num_threads_for_writing, flag_delete_input_files = True, za_mtx = za_mtx, za_mtx_index = za_mtx_index ) # write ramtx zarr object
    
    # delete temp folder
    shutil.rmtree( path_folder_temp )
def create_zarr_from_mtx( path_file_input_mtx, path_folder_zarr, int_buffer_size = 1000, int_num_workers_for_writing_ramtx = 10, chunks_dense = ( 1000, 1000 ), dtype_mtx = np.float64 ) :
    """ # 2022-07-30 03:52:08 
    create dense ramtx (dense zarr object) from matrix sorted by barcodes.
    
    'path_file_input_mtx' : input mtx gzip file
    'path_folder_zarr' : output zarr object folder
    'int_buffer_size' : number of lines for a pipe communcation. larger value will decrease an overhead for interprocess coummuncaiton. however, it will lead to more memory usage.
    'int_num_workers_for_writing_ramtx' : the number of worker for writing zarr object
    'chunks_dense' : chunk size of the output zarr object. smaller number of rows in a chunk will lead to smaller memory consumption, since data of all genes for the cells in a chunk will be collected before writing. ( int_num_barcodes_in_a_chunk, int_num_features_in_a_chunk )
    'dtype_mtx' : zarr object dtype
    """
    int_num_barcodes_in_a_chunk = chunks_dense[ 0 ]

    # retrieve dimension of the output dense zarr array
    int_num_features, int_num_barcodes, int_num_records = MTX_10X_Retrieve_number_of_rows_columns_and_records( path_file_input_mtx ) # retrieve metadata of mtx
    # open persistent zarr arrays to store matrix (dense ramtx)
    za_mtx = zarr.open( path_folder_zarr, mode = 'w', shape = ( int_num_barcodes, int_num_features ), chunks = chunks_dense, dtype = dtype_mtx, synchronizer = zarr.ThreadSynchronizer( ) ) # each mtx record will contains two values instead of three values for more compact storage 

    """ assumes input mtx is sorted by id_barcode (sorted by columns of the matrix market formatted matrix) """
    def __gunzip( path_file_input_mtx, pipe_sender ) :
        ''' # 2022-07-29 23:25:36 
        create a stream of lines from a gzipped mtx file
        '''
        with gzip.open( path_file_input_mtx, 'rt' ) as file :
            line = file.readline( )
            if len( line ) == 0 : # if the file is empty, exit
                pipe_sender.send( None ) # indicates that the file reading is completed
            else :
                # consume comment lines
                while line[ 0 ] == '%' : 
                    line = file.readline( ) # read the next line

                # use the buffer to reduce the overhead of interprocess communications
                l_buffer = [ ] # initialize the buffer
                while line : # iteratre through lines
                    l_buffer.append( line ) 
                    if len( l_buffer ) >= int_buffer_size : # if buffer is full, flush the buffer
                        pipe_sender.send( l_buffer )
                        l_buffer = [ ]
                    line = file.readline( ) # read the next line
                # flush remaining buffer
                if len( l_buffer ) >= 0 :
                    pipe_sender.send( l_buffer )
                pipe_sender.send( None ) # indicates that the file reading is completed
    def __distribute( pipe_receiver, int_num_workers_for_writing_ramtx, int_num_barcodes_in_a_chunk ) :
        ''' # 2022-07-29 23:28:37 
        '''
        def __write_zarr( pipe_receiver ) :
            """ # 2022-07-29 23:29:00 
            """
            while True :
                r = pipe_receiver.recv( )
                if r is None : # when all works are completed, exit
                    break
                coords_barcodes, coords_features, values = r # parse received records
                za_mtx.set_coordinate_selection( ( coords_barcodes, coords_features ), values ) # write zarr matrix

        # start workers for writing zarr
        l_p = [ ]
        l_pipe_sender = [ ]
        index_process = 0 # initialize index of process 
        for index_worker in range( int_num_workers_for_writing_ramtx ) :
            pipe_sender_for_a_worker, pipe_receiver_for_a_worker = mp.Pipe( ) # create a pipe to the worker
            p = mp.Process( target = __write_zarr, args = ( pipe_receiver_for_a_worker, ) )
            p.start( ) # start process
            l_p.append( p ) # collect process
            l_pipe_sender.append( pipe_sender_for_a_worker ) # collect pipe_sender

        # distribute works to workers
        int_index_chunk_being_collected = None
        l_int_feature, l_int_barcode, l_float_value = [ ], [ ], [ ]
        while True :
            l_line = pipe_receiver.recv( )
            if l_line is None : # if all lines are retrieved, exit
                break
            for line in l_line :
                int_feature, int_barcode, float_value = line.strip( ).split( ) # parse a record of a matrix-market format file
                int_feature, int_barcode, float_value = int( int_feature ) - 1, int( int_barcode ) - 1, float( float_value ) # 1-based > 0-based coordinates

                int_index_chunk = int_barcode // int_num_barcodes_in_a_chunk # retrieve int_chunk of the current barcode

                # initialize 'int_index_chunk_being_collected'
                if int_index_chunk_being_collected is None :
                    int_index_chunk_being_collected = int_index_chunk

                # flush the chunk
                if int_index_chunk_being_collected != int_index_chunk :
                    l_pipe_sender[ index_process ].send( ( l_int_barcode, l_int_feature, l_float_value ) )
                    # change the worker
                    index_process = ( index_process + 1 ) % int_num_workers_for_writing_ramtx
                    # initialize the next chunk
                    l_int_feature, l_int_barcode, l_float_value = [ ], [ ], [ ]
                    int_index_chunk_being_collected = int_index_chunk

                # collect record
                l_int_feature.append( int_feature )
                l_int_barcode.append( int_barcode )
                l_float_value.append( float_value )

        # write the last chunk if valid unwritten chunk exists
        if len( l_int_barcode ) > 0 :
            l_pipe_sender[ index_process ].send( ( l_int_barcode, l_int_feature, l_float_value ) )

        # terminate the workers
        for pipe_sender in l_pipe_sender :
            pipe_sender.send( None )

    # compose processes
    l_p = [ ]
    pipe_sender, pipe_receiver = mp.Pipe( ) # create a link
    l_p.append( mp.Process( target = __gunzip, args = ( path_file_input_mtx, pipe_sender ) ) )
    l_p.append( mp.Process( target = __distribute, args = ( pipe_receiver, int_num_workers_for_writing_ramtx, int_num_barcodes_in_a_chunk ) ) )
    
    # run processes
    for p in l_p : 
        p.start( )
    for p in l_p : 
        p.join( )
def create_ramtx_from_mtx( path_folder_mtx_10x_input, path_folder_output, mode = 'dense', int_num_records_in_a_chunk = 10000000, int_buffer_size = 300, compresslevel = 6, flag_dtype_is_float = False, int_num_threads_for_chunking = 5, int_num_threads_for_writing = 1, int_max_num_input_files_for_each_merge_sort_worker = 8, int_num_chunks_to_combine_before_concurrent_merge_sorting = 8, dtype_dense_mtx = np.uint32, dtype_sparse_mtx = np.float64, dtype_sparse_mtx_index = np.float64, int_num_of_records_in_a_chunk_zarr_matrix = 20000, int_num_of_entries_in_a_chunk_zarr_matrix_index = 1000, chunks_dense = ( 2000, 1000 ), int_num_of_entries_in_a_chunk_metadata = 1000, verbose = False, flag_debugging = False ) :
    """ # 2022-07-30 15:05:58 
    sort a given mtx file in a very time- and memory-efficient manner, and create sparse (sorted by barcode/feature).
    when 'type' == 'dense', create a dense ramtx object in the given output folder without sorting the input mtx file in the given axis ('flag_mtx_sorted_by_id_feature')
    
    Arguments:
    -- basic arguments --
    'path_folder_mtx_10x_input' : a folder where mtx/feature/barcode files reside.
    'path_folder_output' : folder directory of the output folder that will contains zarr representation of a mtx file
    'mode' : {'dense' or 'sparse_for_querying_barcodes', 'sparse_for_querying_features'} : whether to create dense ramtx or sparse ramtx. for sparse ramtx, please set the appropriate 'flag_mtx_sorted_by_id_feature' flag argument for sorting. When building a dense ramtx, the chunk size can be set using 'chunks_dense' arguments
    'int_buffer_size' : the number of entries for each batch that will be given to 'pipe_sender'. increasing this number will reduce the overhead associated with interprocess-communication through pipe, but will require more memory usage
    'flag_debugging' : if True, does not delete temporary files
    
    -- for sparse ramtx creation --
    'compresslevel' : compression level of the output Gzip file. 6 by default
    'int_max_num_input_files_for_each_merge_sort_worker' : maximum number of input pipes for each worker. this argument and the number of input pipes together will determine the number of threads used for sorting
    'flag_dtype_is_float' : set this flag to True to export float values to the output mtx matrix
    'int_num_threads_for_writing' : the number of threads for gzip writer. if 'int_num_threads' > 1, pgzip will be used to write the output gzip file. please note that pgzip (multithreaded version of gzip module) has some memory-leaking issue for large inputs.
    'int_num_records_in_a_chunk' : the number of maximum records in a chunk
    'int_num_threads_for_chunking' : number of workers for sorting and writing operations. the number of worker for reading the input gzip file will be 1.
    'dtype_sparse_mtx' (default: np.float64), dtype of the output zarr array for storing sparse matrix
    'dtype_sparse_mtx_index' (default: np.float64) : dtype of the output zarr array for storing sparse matrix indices
    'int_num_of_records_in_a_chunk_zarr_matrix' : chunk size for output zarr mtx object (sparse ramtx)
    'int_num_of_entries_in_a_chunk_zarr_matrix_index' : chunk size for output zarr mtx index object (sparse ramtx)

    -- for dense ramtx creation --
    'dtype_dense_mtx' (default: np.float64), dtype of the output zarr array for storing dense matrix
    'chunks_dense' : chunk size for dense ramtx object. if None is given, a dense ramtx object will be created. when dense ramtx object is created, the number of threads for chunking can be set using the 'int_num_threads_for_chunking' argument ( int_num_barcodes_in_a_chunk, int_num_features_in_a_chunk )
    
    -- for metadata creation --
    'int_num_of_entries_in_a_chunk_metadata' : chunk size for output ramtx metadata
    
    """
    # check flag
    path_file_flag_completion = f"{path_folder_output}ramtx.completed.flag"
    if os.path.exists( path_file_flag_completion ) : # exit if a flag indicating the pipeline was completed previously.
        return
    
    ''' prepare '''
    mode = mode.lower( ) # handle mode argument
    
    # retrieve file pathes
    path_file_input_bc, path_file_input_feature, path_file_input_mtx = __Get_path_essential_files__( path_folder_mtx_10x_input )
    # retrieve metadata from the input mtx file
    int_num_features, int_num_barcodes, int_num_records = MTX_10X_Retrieve_number_of_rows_columns_and_records( path_folder_mtx_10x_input ) # retrieve metadata of mtx
    # create an output directory
    os.makedirs( path_folder_output, exist_ok = True )
    path_folder_temp = f"{path_folder_output}temp_{UUID( )}/"
    os.makedirs( path_folder_temp, exist_ok = True )
    
    """
    construct RAMTx (Zarr) matrix
    """
    if mode.lower( ) == 'dense' : # build dense ramtx based on the setting.
        create_zarr_from_mtx( path_file_input_mtx, f'{path_folder_output}matrix.zarr', int_buffer_size = int_buffer_size, chunks_dense = chunks_dense, dtype_mtx = dtype_dense_mtx, int_num_workers_for_writing_ramtx = int_num_threads_for_chunking )
    else : # build sparse ramtx
        flag_mtx_sorted_by_id_feature = 'feature' in mode # retrieve a flag whether to sort ramtx by id_feature or id_barcode. 
        # open persistent zarr arrays to store matrix and matrix index
        za_mtx = zarr.open( f'{path_folder_output}matrix.zarr', mode = 'w', shape = ( int_num_records, 2 ), chunks = ( int_num_of_records_in_a_chunk_zarr_matrix, 2 ), dtype = dtype_sparse_mtx ) # each mtx record will contains two values instead of three values for more compact storage 
        za_mtx_index = zarr.open( f'{path_folder_output}matrix.index.zarr', mode = 'w', shape = ( int_num_features if flag_mtx_sorted_by_id_feature else int_num_barcodes, 2 ), chunks = ( int_num_of_entries_in_a_chunk_zarr_matrix_index, 2 ), dtype = dtype_sparse_mtx_index ) # max number of matrix index entries is 'int_num_records' of the input matrix. this will be resized # dtype of index should be np.float64 to be compatible with Zarr.js, since Zarr.js currently does not support np.int64...
        # build RAMtx matrix from the input matrix file
        sort_mtx( path_file_input_mtx, int_num_records_in_a_chunk = int_num_records_in_a_chunk, int_buffer_size = int_buffer_size, compresslevel = compresslevel, flag_dtype_is_float = flag_dtype_is_float, flag_mtx_sorted_by_id_feature = flag_mtx_sorted_by_id_feature, int_num_threads_for_chunking = int_num_threads_for_chunking, int_num_threads_for_writing = int_num_threads_for_writing, int_max_num_input_files_for_each_merge_sort_worker = int_max_num_input_files_for_each_merge_sort_worker, int_num_chunks_to_combine_before_concurrent_merge_sorting = int_num_chunks_to_combine_before_concurrent_merge_sorting, za_mtx = za_mtx, za_mtx_index = za_mtx_index )
    
    """
    prepare data for the axes (features/barcodes)
    """
    ''' write sorted barcodes and features files to zarr objects'''
    for name_axis, int_num_entries in zip( [ 'barcodes', 'features' ], [ int_num_barcodes, int_num_features ] ) : # retrieve a flag whether the entry was used for sorting.
        flag_axis_initialized = False # initialize the flag to False 
        for index_chunk, df in enumerate( pd.read_csv( f"{path_folder_mtx_10x_input}{name_axis}.tsv.gz", sep = '\t', header = None, chunksize = int_num_of_entries_in_a_chunk_metadata ) ) : # read chunk by chunk
            if not flag_axis_initialized :
                l_col = list( f"{name_axis}_{i}" for i in range( len( df.columns ) ) ) # name the columns using 0-based indices
                
                # write zarr object for random access of string representation of features/barcodes
                za = zarr.open( f'{path_folder_output}{name_axis}.str.zarr', mode = 'w', shape = ( int_num_entries, min( 2, df.shape[ 1 ] ) ), chunks = ( int_num_of_entries_in_a_chunk_metadata, 1 ), dtype = str, synchronizer = zarr.ThreadSynchronizer( ) ) # multithreading? # string object # individual columns will be chucked, so that each column can be retrieved separately.
                
                # build a ZarrDataFrame object for random access of number and categorical data of features/barcodes
                zdf = ZarrDataFrame( f'{path_folder_output}{name_axis}.num_and_cat.zdf', int_num_rows = int_num_entries, int_num_rows_in_a_chunk = int_num_of_entries_in_a_chunk_metadata, flag_store_string_as_categorical = True, flag_retrieve_categorical_data_as_integers = True, flag_enforce_name_col_with_only_valid_characters = False, flag_load_data_after_adding_new_column = False ) # use the same chunk size for all feature/barcode objects
                
                # create a folder to save a chunked string representations
                path_folder_str_chunks = f'{path_folder_output}{name_axis}.str.chunks/'
                os.makedirs( path_folder_str_chunks, exist_ok = True )
                za_str_chunks = zarr.group( path_folder_str_chunks )
                za_str_chunks.attrs[ 'dict_metadata' ] = { 'int_num_entries' : int_num_entries, 'int_num_of_entries_in_a_chunk' : int_num_of_entries_in_a_chunk_metadata } # write essential metadata for str.chunks
                
                flag_axis_initialized = True # set the flag to True
                
            values = df.values # retrieve values
            
            sl_chunk = slice( index_chunk * int_num_of_entries_in_a_chunk_metadata, ( index_chunk + 1 ) * int_num_of_entries_in_a_chunk_metadata )
            values_str = values[ :, : 2 ] # retrieve string representations
            za[ sl_chunk ] = values_str # set str.zarr
            # set str.chunks
            for index_col, arr_val in enumerate( values_str.T ) :
                with open( f"{path_folder_str_chunks}{index_chunk}.{index_col}", 'wt' ) as newfile : # similar organization to zarr
                    newfile.write( _base64_encode( _gzip_bytes( ( '\n'.join( arr_val ) + '\n' ).encode( ) ) ) )
            # set num_and_cat.zdf
            if values.shape[ 1 ] > 2 :
                values_num_and_cat = values[ :, 2 : ] # retrieve number and categorical data
                for arr_val, name_col in zip( values[ :, 2 : ].T, l_col[ 2 : ] ) :
                    zdf[ name_col, sl_chunk ] = arr_val

    ''' write metadata '''
    # compose metadata
    dict_metadata = { 
        'path_folder_mtx_10x_input' : path_folder_mtx_10x_input,
        'mode' : mode,
        'str_completed_time' : TIME_GET_timestamp( True ),
        'int_num_features' : int_num_features,
        'int_num_barcodes' : int_num_barcodes,
        'int_num_records' : int_num_records,
        'version' : _version_,
    }
    if mode.lower( ) != 'dense' :
        dict_metadata[ 'flag_ramtx_sorted_by_id_feature' ] = flag_mtx_sorted_by_id_feature
    root = zarr.group( f'{path_folder_output}' )
    root.attrs[ 'dict_metadata' ] = dict_metadata
    
    # delete temp folder
    shutil.rmtree( path_folder_temp )
    
    ''' write a flag indicating the export has been completed '''
    with open( path_file_flag_completion, 'w' ) as file :
        file.write( TIME_GET_timestamp( True ) )
def create_ramdata_from_mtx( path_folder_mtx_10x_input, path_folder_ramdata_output, set_modes = { 'dense' }, name_layer = 'raw', int_num_records_in_a_chunk = 10000000, int_buffer_size = 300, compresslevel = 6, flag_dtype_is_float = False, int_num_threads_for_chunking = 5, int_num_threads_for_writing = 1, int_max_num_input_files_for_each_merge_sort_worker = 8, int_num_chunks_to_combine_before_concurrent_merge_sorting = 8, dtype_dense_mtx = np.uint32, dtype_sparse_mtx = np.float64, dtype_sparse_mtx_index = np.float64, int_num_of_records_in_a_chunk_zarr_matrix = 20000, int_num_of_entries_in_a_chunk_zarr_matrix_index = 1000, chunks_dense = ( 2000, 1000 ), int_num_of_entries_in_a_chunk_metadata = 1000, flag_multiprocessing = True, verbose = False, flag_debugging = False ) :
    """ # 2022-08-05 21:41:07 
    sort a given mtx file in a very time- and memory-efficient manner, and create sparse (sorted by barcode/feature).
    when 'type' == 'dense', create a dense ramtx object in the given output folder without sorting the input mtx file in the given axis ('flag_mtx_sorted_by_id_feature')
    
    Arguments:
    -- basic arguments --
    'path_folder_mtx_10x_input' : a folder where mtx/feature/barcode files reside.
    'path_folder_ramdata_output' : an output folder directory of the RamData object
    'set_modes' : a set of {'dense', 'sparse_for_querying_barcodes', 'sparse_for_querying_features'} : modes of ramtxs to build. 
                'dense' : dense ramtx. When building a dense ramtx, the chunk size can be set using 'chunks_dense' arguments
                'sparse_for_querying_barcodes/features' : sparse ramtx sorted by each axis
    'int_buffer_size' : the number of entries for each batch that will be given to 'pipe_sender'. increasing this number will reduce the overhead associated with interprocess-communication through pipe, but will require more memory usage
    'flag_debugging' : if True, does not delete temporary files
    
    -- for sparse ramtx creation --
    'compresslevel' : compression level of the output Gzip file. 6 by default
    'int_max_num_input_files_for_each_merge_sort_worker' : maximum number of input pipes for each worker for concurrent-merge-sorting. this argument and the number of input pipes together will determine the number of threads used for sorting.
    'flag_dtype_is_float' : set this flag to True to export float values to the output mtx matrix
    'int_num_threads_for_writing' : the number of threads for gzip writer. if 'int_num_threads' > 1, pgzip will be used to write the output gzip file. please note that pgzip (multithreaded version of gzip module) has some memory-leaking issue for large inputs.
    'int_num_records_in_a_chunk' : the number of maximum records in a chunk
    'int_num_threads_for_chunking' : number of workers for sorting and writing operations. the number of worker for reading the input gzip file will be 1.
    'int_num_chunks_to_combine_before_concurrent_merge_sorting' : preliminary merge-sorting step. in a typical system, a single process can open upto ~1000 files at once. therefore, merge-sorting more than 1000 files cannot be done in a single concurrent merge-sort step. therefore, sorted chunks will be combined into a larger chunk before they can be merge-sorted into a single output file. 
            for very large single-cell data (>10 million single-cells or > 50 GB of matrix), increase the number given through this argument to avoid too-many-files-opened error.
    'dtype_sparse_mtx' (default: np.float64), dtype of the output zarr array for storing sparse matrix
    'dtype_sparse_mtx_index' (default: np.float64) : dtype of the output zarr array for storing sparse matrix indices
    'int_num_of_records_in_a_chunk_zarr_matrix' : chunk size for output zarr mtx object (sparse ramtx)
    'int_num_of_entries_in_a_chunk_zarr_matrix_index' : chunk size for output zarr mtx index object (sparse ramtx)

    -- for dense ramtx creation --
    'dtype_dense_mtx' (default: np.float64), dtype of the output zarr array for storing dense matrix
    'chunks_dense' : chunk size for dense ramtx object. if None is given, a dense ramtx object will be created. when dense ramtx object is created, the number of threads for chunking can be set using the 'int_num_threads_for_chunking' argument ( int_num_barcodes_in_a_chunk, int_num_features_in_a_chunk )
    
    -- for metadata creation --
    'int_num_of_entries_in_a_chunk_metadata' : chunk size for output ramtx metadata
    
    -- for RamData creation --
    'name_layer' : a name of the ramdata layer to create (default: raw)
    """
    ''' handle arguments '''
    set_valid_modes = { 'dense', 'sparse_for_querying_barcodes', 'sparse_for_querying_features' }
    set_modes = set( e for e in set( e.lower( ).strip( ) for e in set_modes ) if e in set_valid_modes ) # retrieve valid mode
    assert len( set_modes ) > 0 # at least one valid mode should exists
    
    # build RAMtx objects
    path_folder_ramdata_layer = f"{path_folder_ramdata_output}{name_layer}/" # define directory of the output data layer
    
    # define keyword arguments for ramtx building
    kwargs_ramtx = { 
        'int_num_records_in_a_chunk' : int_num_records_in_a_chunk, 
        'int_buffer_size' : int_buffer_size, 
        'compresslevel' : compresslevel, 
        'flag_dtype_is_float' : flag_dtype_is_float, 
        'int_num_threads_for_chunking' : int_num_threads_for_chunking, 
        'int_num_threads_for_writing' : int_num_threads_for_writing, 
        'int_max_num_input_files_for_each_merge_sort_worker' : int_max_num_input_files_for_each_merge_sort_worker, 
        'int_num_chunks_to_combine_before_concurrent_merge_sorting' : int_num_chunks_to_combine_before_concurrent_merge_sorting, 
        'dtype_dense_mtx' : dtype_dense_mtx, 
        'dtype_sparse_mtx' : dtype_sparse_mtx, 
        'dtype_sparse_mtx_index' : dtype_sparse_mtx_index, 
        'int_num_of_records_in_a_chunk_zarr_matrix' : int_num_of_records_in_a_chunk_zarr_matrix, 
        'int_num_of_entries_in_a_chunk_zarr_matrix_index' : int_num_of_entries_in_a_chunk_zarr_matrix_index, 
        'chunks_dense' : chunks_dense, 
        'int_num_of_entries_in_a_chunk_metadata' : int_num_of_entries_in_a_chunk_metadata, 
        'verbose' : verbose, 
        'flag_debugging' : flag_debugging
    }
    
    # compose processes 
    l_p = [ ]
    for mode in set_modes : 
        l_p.append( mp.Process( target = create_ramtx_from_mtx, args = ( path_folder_mtx_10x_input, f"{path_folder_ramdata_layer}{mode}/", mode ), kwargs = kwargs_ramtx ) )        
    # run processes
    for p in l_p : p.start( )
    for p in l_p : p.join( )
            
    # copy features/barcode.tsv.gz random access files for the web (stacked base64 encoded tsv.gz files)
    # copy features/barcode string representation zarr objects
    # copy features/barcode ZarrDataFrame containing number/categorical data
    for name_axis in [ 'features', 'barcodes' ] :
        for str_suffix in [ '.str.chunks', '.str.zarr', '.num_and_cat.zdf' ] :
            OS_Run( [ 'cp', '-r', f"{path_folder_ramdata_layer}{mode}/{name_axis}{str_suffix}", f"{path_folder_ramdata_output}{name_axis}{str_suffix}" ] )
            
    # write ramdata metadata 
    int_num_features, int_num_barcodes, int_num_records = MTX_10X_Retrieve_number_of_rows_columns_and_records( path_folder_mtx_10x_input ) # retrieve metadata of the input 10X mtx
    root = zarr.group( path_folder_ramdata_output )
    root.attrs[ 'dict_metadata' ] = { 
        'path_folder_mtx_10x_input' : path_folder_mtx_10x_input,
        'str_completed_time' : TIME_GET_timestamp( True ),
        'int_num_features' : int_num_features,
        'int_num_barcodes' : int_num_barcodes,
        'layers' : [ name_layer ],
        'models' : dict( ),
        'version' : _version_,
    }
    # write layer metadata
    lay = zarr.group( path_folder_ramdata_layer )
    lay.attrs[ 'dict_metadata' ] = { 
        'set_modes' : list( set_modes ) + ( [ 'dense_for_querying_barcodes', 'dense_for_querying_features' ] if 'dense' in set_modes else [ ] ), # dense ramtx can be operated for querying either barcodes/features
        'version' : _version_,
    }
def sort_mtx_10x( path_folder_mtx_input : str, path_folder_mtx_output : str, flag_mtx_sorted_by_id_feature : bool = False, ** kwargs ) :
    """ # 2022-08-21 17:05:55 
    a convenient wrapper method of 'sort_mtx' for sorting a matrix in a 10X matrix format. features and barcodes files will be copied, too.
    
    'path_folder_mtx_input' : path to the input matrix folder
    'path_folder_mtx_output' : path to the output matrix folder
    'flag_mtx_sorted_by_id_feature' : whether to sort mtx file in the feature axis or not
    
    kwargs: keyworded arguments for 'sort_mtx'
    """
    os.makedirs( path_folder_mtx_output, exist_ok = True ) # create output folder
    # copy features and barcodes files
    for name_file in [ 'features.tsv.gz', 'barcodes.tsv.gz' ] :
        shutil.copyfile( f"{path_folder_mtx_input}{name_file}", f"{path_folder_mtx_output}{name_file}" )
    # sort matrix file
    sort_mtx( f"{path_folder_mtx_input}matrix.mtx.gz", path_file_gzip_sorted = f"{path_folder_mtx_output}matrix.mtx.gz", flag_mtx_sorted_by_id_feature = flag_mtx_sorted_by_id_feature, ** kwargs )

''' utility functions for handling zarr '''
def zarr_exists( path_folder_zarr ) :
    """ # 2022-07-20 01:06:09 
    check whether the given zarr object path exists
    """
    try : 
        zarr.open( path_folder_zarr, 'r' )
        flag_zarr_exists = True
    except :
        flag_zarr_exists = False
    return flag_zarr_exists
def zarr_copy( path_folder_zarr_source, path_folder_zarr_sink, int_num_chunks_per_batch = 1000 ) :
    """ # 2022-07-22 01:45:17 
    copy a soruce zarr object to a sink zarr object chunks by chunks along the primary axis (axis 0)
    also copy associated attributes, too.
    
    'path_folder_zarr_source' : source zarr object path
    'path_folder_zarr_sink' : sink zarr object path
    'int_num_chunks_per_batch' : number of chunks along the primary axis (axis 0) to be copied for each loop. for example, when the size of an array is (100, 100), chunk size is (10, 10), and 'int_num_chunks_per_batch' = 1, 10 chunks along the secondary axis (axis = 1) will be saved for each batch.
    """
    # open zarr objects
    za = zarr.open( path_folder_zarr_source )
    za_sink = zarr.open( path_folder_zarr_sink, mode = 'w', shape = za.shape, chunks = za.chunks, dtype = za.dtype, fill_value = za.fill_value, synchronizer = zarr.ThreadSynchronizer( ) ) # open the output zarr
    
    # copy count data
    int_total_num_rows = za.shape[ 0 ]
    int_num_rows_in_batch = za.chunks[ 0 ] * int( int_num_chunks_per_batch )
    for index_batch in range( int( np.ceil( int_total_num_rows / int_num_rows_in_batch ) ) ) :
        sl = slice( index_batch * int_num_rows_in_batch, ( index_batch + 1 ) * int_num_rows_in_batch )
        za_sink[ sl ] = za[ sl ] # copy batch by batch
    
    # copy metadata
    dict_attrs = dict( za.attrs ) # retrieve metadata from the source
    for key in dict_attrs : # write metadata to sink Zarr object
        za_sink.attrs[ key ] = dict_attrs[ key ]
def zarr_start_multiprocessing_write( ) :
    """ # 2022-08-07 20:55:26 
    change setting for write of a zarr object using multiple processes
    since a zarr object will be modified by multiple processes, setting 'numcodecs.blosc.use_threads' to False as recommended by the zarr documentation
    """
    numcodecs.blosc.use_threads = False
def zarr_end_multiprocessing_write( ) :
    """ # 2022-08-07 20:55:26 
    revert setting back from the write of a zarr object using multiple processes
    """
    numcodecs.blosc.use_threads = None
        
''' a class for containing disk-backed AnnData objects '''
class AnnDataContainer( ) :
    """ # 2022-06-09 18:35:04 
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
    def __init__( self, flag_enforce_name_adata_with_only_valid_characters = True, path_prefix_default = None, mode = 'a', path_prefix_default_mask = None, flag_is_read_only = False, ** args ) :
        self.__str_invalid_char = '! @#$%^&*()-=+`~:;[]{}\|,<.>/?' + '"' + "'" if flag_enforce_name_adata_with_only_valid_characters else ''
        self.path_prefix_default = path_prefix_default
        self._mode = mode
        self._path_prefix_default_mask = path_prefix_default_mask
        self._flag_is_read_only = flag_is_read_only
        self._dict_name_adata_to_namespace = dict( )

        # add items
        for name_adata in args :
            self.__setitem__( name_adata, args[ name_adata ] )
    def __getitem__( self, name_adata ) :
        return self._dict_name_adata_to_namespace[ name_adata ][ 'adata' ]
    def __setitem__( self, name_adata, args ) :
        # check whether the given name_adata contains invalid characters(s)
        for char_invalid in self.__str_invalid_char :
            if char_invalid in name_adata :
                raise TypeError( f'the following characters cannot be used in "name_adata": {self.__str_invalid_char}' )
        
        # if the given argument is not a dictionary format, convert it to the dictionary format
        if not isinstance( args, dict ) : 
            # check the type of input value
            if isinstance( args, scanpy.AnnData ) :
                args = { 'adata' : args }
            elif isinstance( args, str ) :
                args = { 'path' : args }
            else :
                args = dict( )
        
        # set the default file path
        if 'path' not in args :
            args[ 'path' ] = f"{self.path_prefix_default}{name_adata}.h5ad"
        # check validity of the path if AnnDataObject was not given
        if 'adata' not in args :
            if not os.path.exists( args[ 'path' ] ) :
                raise FileNotFoundError( f"{args[ 'path' ]} does not exist, while AnnData object is not given" )
            args[ 'adata' ] = None # put a placeholder value
        
        self._dict_name_adata_to_namespace[ name_adata ] = args
        setattr( self, name_adata, args[ 'adata' ] )
    def __delitem__( self, name_adata ) :
        ''' # 2022-06-09 12:47:36 
        remove the adata from the memory and the object
        '''
        # remove adata attribute from the dictionary
        if name_adata in self._dict_name_adata_to_namespace :
            del self._dict_name_adata_to_namespace[ name_adata ]
        # remove adata attribute from the current object
        if hasattr( self, name_adata ) :
            delattr( self, name_adata )
    def __contains__( self, name_adata ) :
        return name_adata in self._dict_name_adata_to_namespace
    def __iter__( self ) :
        return iter( self._dict_name_adata_to_namespace )
    def __repr__( self ) :
        return f"<AnnDataContainer object with the following items: {list( self._dict_name_adata_to_namespace )}\n\t default prefix is {self.path_prefix_default}>"
    def load( self, * l_name_adata ) :
        ''' # 2022-05-24 02:33:36 
        load given anndata object(s) of the given list of 'name_adata' 
        '''
        for name_adata in l_name_adata :
            if name_adata not in self : # skip the 'name_adata' that does not exist in the current container
                continue
            args = self._dict_name_adata_to_namespace[ name_adata ]
            # if current AnnData has not been loaded
            if args[ 'adata' ] is None :
                args[ 'adata' ] = scanpy.read_h5ad( args[ 'path' ] )
                self[ name_adata ] = args # update the current name_adata
    def unload( self, * l_name_adata ) :
        """ # 2022-06-09 12:47:42 
        remove the adata object from the memory
        similar to __delitem__, but does not remove the attribute from the current 'AnnDataContainer' object
        """
        for name_adata in l_name_adata :
            if name_adata not in self : # skip the 'name_adata' that does not exist in the current container
                continue
            args = self._dict_name_adata_to_namespace[ name_adata ]
            # if current AnnData has been loaded
            if args[ 'adata' ] is not None :
                args[ 'adata' ] = None
                self[ name_adata ] = args # update the current name_adata
    def delete( self, * l_name_adata ) :
        ''' # 2022-06-09 12:58:45 
        remove the adata from the memory, the current object, and from the disk
        '''
        for name_adata in l_name_adata :
            if name_adata not in self : # skip the 'name_adata' that does not exist in the current container
                continue
            # remove file on disk if exists
            path_file = self._dict_name_adata_to_namespace[ name_adata ][ 'path' ]
            if os.path.exists( path_file ) :
                os.remove( path_file )
            del self[ name_adata ] # delete element from the current object
    def update( self, * l_name_adata ) :
        """ # 2022-06-09 18:13:21 
        save the given AnnData objects to disk
        """
        for name_adata in l_name_adata :
            if name_adata not in self : # skip the 'name_adata' that does not exist in the current container
                continue
            # if current AnnData is a valid AnnData object
            args = self._dict_name_adata_to_namespace[ name_adata ]
            if isinstance( args[ 'adata' ], scanpy.AnnData ) :
                args[ 'adata' ].write( args[ 'path' ] ) # write AnnData object
    def empty( self, * l_name_adata ) :
        """ # 2022-06-09 18:23:44 
        empty the count matrix of the given AnnData objects 
        """
        for name_adata in l_name_adata :
            if name_adata not in self : # skip the 'name_adata' that does not exist in the current container
                continue
            # if current AnnData is a valid AnnData object
            adata = self._dict_name_adata_to_namespace[ name_adata ][ 'adata' ]
            if isinstance( adata, scanpy.AnnData ) :
                adata.X = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( ( [], ( [], [] ) ), shape = ( len( adata.obs ), len( adata.var ) ) ) ) # empty the anndata object
    def transfer_attributes( self, name_adata, adata, flag_ignore_var = True ) :
        ''' # 2022-06-06 01:44:00 
        transfer attributes of the given AnnDAta 'adata' to the current AnnData data containined in this object 'name_adata'  
        'flag_ignore_var' : ignore attributes related to 'var' (var, varm, varp)
        '''
        adata_current = self[ name_adata ] # retrieve current AnnData
        
        # transfer uns and obs-related elements
        for name_attr in [ 'obs', 'uns', 'obsm', 'obsp' ] :
            if hasattr( adata, name_attr ) :
                setattr( adata_current, name_attr, getattr( adata, name_attr ) )
        
        # transfer var-related elements if 'flag_ignore_var' is True
        if not flag_ignore_var  :
            for name_attr in [ 'var', 'varm', 'varp' ] :
                if hasattr( adata, name_attr ) :
                    setattr( adata_current, name_attr, getattr( adata, name_attr ) )

''' a class for wrapping shelve-backed persistent dictionary '''
class ShelveContainer( ) :
    """ # 2022-07-14 20:29:42 
    a convenient wrapper of 'shelve' module-backed persistant dictionary to increase the memory-efficiency of a shelve-based persistent dicitonary, enabling the use of shelve dictionary without calling close( ) function to remove the added elements from the memory.
    
    'path_prefix_shelve' : a prefix of the persisent dictionary
    'mode' : file mode. 'r' for read-only mode and 'a' for mode allowing modifications
    'path_prefix_shelve_mask' : the LOCAL file system path of 'MASK' where the modifications of the current object will be saved and retrieved. If this attribute has been set, the given RamData in the the given 'path_folder_ramdata' will be used as READ-ONLY. For example, when RamData resides in the HTTP server, data is often read-only (data can be only fetched from the server, and not the other way around). However, by giving a local path through this argument, the read-only RamData object can be analyzed as if the RamData object can be modified. This is possible since all the modifications made on the input RamData will be instead written to the local RamData object 'mask' and data will be fetced from the local copy before checking the availability in the remote RamData object.
    'flag_is_read_only' : read-only status of the storage
    
    """
    def __init__( self, path_prefix_shelve, mode = 'a', path_prefix_shelve_mask = None, flag_is_read_only = False ) :
        """ # 2022-07-20 22:06:15 
        """
        # set attributes
        self.path_prefix_shelve = path_prefix_shelve
        self._mode = mode
        self._path_prefix_shelve_mask = path_prefix_shelve_mask
        self._flag_is_read_only = flag_is_read_only
    @property
    def keys( self ) :
        """ # 2022-07-14 20:43:24 
        return keys
        """
        # if keys has not been loaded
        if not hasattr( self, '_set_keys' ) :
            self.update_keys( ) # update keys
        return self._set_keys
    def update_keys( self ) :
        """ # 2022-07-14 20:41:20 
        update keys of 'shelve'
        """
        with shelve.open( self.path_prefix_shelve ) as ns :
            self._set_keys = set( ns.keys( ) )
    def __contains__( self, x ) -> bool :
        """ # 2022-07-14 21:14:47 
        """
        return x in self.keys
    def __iter__( self ) :
        """ # 2022-07-14 21:15:18 
        """
        return iter( self.keys )
    def __getitem__( self, x ) :
        """ # 2022-07-14 21:18:50 
        """
        if x in self :
            with shelve.open( self.path_prefix_shelve, 'r' ) as ns :
                item = ns[ str( x ) ] # only string is avaiable as a key
            return item
    def __setitem__( self, x, val ) :
        """ # 2022-07-14 21:22:54 
        """
        with shelve.open( self.path_prefix_shelve ) as ns :
            ns[ str( x ) ] = val # only string is avaiable as a key
            self._set_keys = set( ns.keys( ) ) # update keys
    def __delitem__( self, x ) :
        """ # 2022-07-14 21:37:25 
        """
        with shelve.open( self.path_prefix_shelve ) as ns :
            del ns[ str( x ) ] # only string is avaiable as a key
            self._set_keys = set( ns.keys( ) ) # update keys
    def __repr__( self ) :
        """ # 2022-07-14 21:37:28 
        """
        return f"<shelve-backed namespace: {self.keys}>"

''' a class for Zarr-based DataFrame object '''
class ZarrDataFrame( ) :
    """ # 2022-08-27 10:21:20 
    storage-based persistant DataFrame backed by Zarr persistent arrays.
    each column can be separately loaded, updated, and unloaded.
    a filter can be set, which allows updating and reading ZarrDataFrame as if it only contains the rows indicated by the given filter.
    currently supported dtypes are bool, float, int, strings (will be interpreted as categorical data).
    the one of the functionality of this class is to provide a Zarr-based dataframe object that is compatible with Zarr.js (javascript implementation of Zarr), with a categorical data type (the format used in zarr is currently not supported in zarr.js) compatible with zarr.js.
    
    Of note, secondary indexing (row indexing) is always applied to unfiltered columns, not to a subset of column containing filtered rows.
    '__getitem__' function is thread and process-safe, while '__setitem__' is not thread nor prosess-safe. 
    
    # 2022-07-04 10:40:14 implement handling of categorical series inputs/categorical series output. Therefore, convertion of ZarrDataFrame categorical data to pandas categorical data should occurs only when dataframe was given as input/output is in dataframe format.
    # 2022-07-04 10:40:20 also, implement a flag-based switch for returning series-based outputs
    # 2022-07-20 22:29:41 : masking functionality was added for the analysis of remote, read-only ZarrDataFrame
    # 2022-08-02 02:17:32 : will enable the use of multi-dimensional data as the 'column'. the primary axis of the data should be same as the length of ZarrDataFrame (the number of rows when no filter is active). By default, the chunking will be only available along the primary axis.
    
    === arguments ===
    'path_folder_zdf' : a folder to store persistant zarr dataframe.
    'df' : input dataframe.
    
    === settings that cannot be changed after initialization ===
    'int_num_rows_in_a_chunk' : chunk size.
    'flag_enforce_name_col_with_only_valid_characters' : if True, every column name should not contain any of the following invalid characters, incompatible with attribute names: '! @#$%^&*()-=+`~:;[]{}\|,<.>/?' + '"' + "'", if False, the only invalid character will be '/', which is incompatible with linux file system as file/folder name.
    'flag_store_string_as_categorical' : if True, for string datatypes, it will be converted to categorical data type.
    
    === settings that can be changed anytime after initialization ===
    'ba_filter' : a bitarray object for applying filter for the ZarrDataFrame. 1 meaning the row is included, 0 meaning the row is excluded
                  If None is given (which is default), the filter is not applied, and the returned data will have the same number of items as the number of rows of the ZarrDataFrame
                  when a filter is active, setting new data and retrieving existing data will work as if the actual ZarrDataFrame is filtered and has the smaller number of rows. 
                  A filter can be changed anytime after initialization of a ZarrDataFrame, but changing (including removing or newly applying filters) will remove all cached data, since the cache only contains data after apply filter to be memory-efficient.
                  A filter can be removed by setting 'filter' attribute to None
                  
                  current implementation has following limits, which can be improved further:
                      - when a filter is active, slicing will load an entire column (after applying filter) (when filter is inactive, slicing will only load the data corresponding to the sliced region)
    'flag_retrieve_categorical_data_as_integers' : if True, accessing categorical data will return integer representations of the categorical values. 
    'flag_load_data_after_adding_new_column' : if True, automatically load the newly added data to the object cache. if False, the newly added data will not be added to the object cache, and accessing the newly added column will cause reading the newly written data from the disk. It is recommended to set this to False if Zdf is used as a data sink
    'mode' : file mode. 'r' for read-only mode and 'a' for mode allowing modifications
    'path_folder_mask' : a local (local file system) path to the mask of the current ZarrDataFrame that allows modifications to be written without modifying the source. if a valid local path to a mask is given, all modifications will be written to the mask
    'flag_is_read_only' : read-only status of the storage
    'flag_use_mask_for_caching' : use mask for not only storing modifications, but also save retrieved data from (remote) sources for faster access. this behavior can be turned on/off at any time
    
    === settings for controlling buffer (batch) size ===
    'int_max_num_entries_per_batch' = 1000000 # the maximum number of entries to be processed in a batch (determines memory usage)
    
    === settings for combined ZarrDataFrame ===
    settings for combined ZarrDataFrames.
    When a list of ZarrDataFrame objects were given through the 'l_zdf' argument, 'combined' ZarrDataFrame mode will be activated.
    Briefly, combined ZarrDataFrame works by retrieving data from individual zdf objects, combined the data, and write a combined data as a column of 'combined' ZarrDataFrame.
    A combined column will be written to the current ZarrDataFrame every time data values were retrieved across the given list of zdf objects and combined (which can serve as a local cache, if one of the zdf object reside in remote location). 
    
    There are two types of combined ZarrDataFrames, 'stacked' and 'interleaved'
        * 'stacked' : rows of each ZarrDataFrame stacked on top of each other without interleaving
            ----------
            rows of ZDF-0
            ----------
            rows of ZDF-1
            ----------
            rows of ZDF-2
            ...
            
        * 'interleaved' : rows of each ZarrDataFrame can be mapped to those of each other.
    
    'l_zdf' : a list of ZarrDataFrame objects that will be combined
    'index_zdf_data_source_when_interleaved' : the index of the zdf to retrieve data when combining mode is interleaved (rows shared between ZDFs)
    'l_dict_index_mapping_interleaved' : list of dictionaries mapping row indices of the combined ZarrDataFrame to row indices of each individual ZarrDataFrame. this argument should be set to non-None value only when the current combined ZarrDataFrame is interleaved. if None is given, the combined ZarrDataFrame will be considered 'stacked'
    
    
    # arguments that works differently in combined zdf object
    'path_folder_zdf' : a path to the 'combined' ZarrDataFrame object.
    'int_num_rows' : when ZarrDataFrame is in combined mode and 'int_num_rows' argument is not given, 'int_num_rows' will be inferred from the given list of ZarrDataFrame 'l_zdf' and 'l_dict_index_mapping_interleaved'
    
    
    """
    def __init__( self, path_folder_zdf : str, l_zdf : Union[ list, np.ndarray, None ] = None, index_zdf_data_source_when_interleaved : int = 0, l_dict_index_mapping_interleaved : Union[ list, None ] = None, int_max_num_entries_per_batch = 1000000, df = None, int_num_rows = None, int_num_rows_in_a_chunk = 10000, ba_filter = None, flag_enforce_name_col_with_only_valid_characters = False, flag_store_string_as_categorical = True, flag_retrieve_categorical_data_as_integers = False, flag_load_data_after_adding_new_column = True, mode = 'a', path_folder_mask = None, flag_is_read_only = False, flag_use_mask_for_caching = True, verbose = True ) :
        """ # 2022-07-20 10:50:23 
        """
        # handle path
        if '://' not in path_folder_zdf : # does not retrieve abspath if the given path is remote path
            path_folder_zdf = os.path.abspath( path_folder_zdf ) # retrieve absolute path
        if path_folder_zdf[ -1 ] != '/' : # add '/' to the end of path to mark that this is a folder directory
            path_folder_zdf += '/'
            
        self._path_folder_zdf = path_folder_zdf
        self._mode = mode
        self._flag_is_read_only = flag_is_read_only
        self._path_folder_mask = path_folder_mask
        self.flag_use_mask_for_caching = flag_use_mask_for_caching
        self._flag_retrieve_categorical_data_as_integers = flag_retrieve_categorical_data_as_integers
        self._flag_load_data_after_adding_new_column = flag_load_data_after_adding_new_column
        self._ba_filter = None # initialize the '_ba_filter' attribute
        self.verbose = verbose
        self.int_max_num_entries_per_batch = int_max_num_entries_per_batch
        
        # %% COMBINED MODE %%
        self._l_zdf = l_zdf
        self.index_zdf_data_source_when_interleaved = index_zdf_data_source_when_interleaved
        self._l_dict_index_mapping_interleaved = l_dict_index_mapping_interleaved

        # %% COMBINED = STACKED %%
        if self.is_combined and not self.is_interleaved :
            # retrieve the number of unfiltered rows for each zdf
            self._l_n_rows_unfiltered = list( zdf._n_rows_unfiltered for zdf in self._l_zdf )
        
        if self.is_combined and int_num_rows is None : # infer 'int_num_rows' from the given arguments
            if self.is_interleaved : # infer 'int_num_rows' for interleaved czdf
                int_num_rows = max( max( dict_index ) for dict_index in self._l_dict_index_mapping_interleaved ) + 1 # 0 based -> 1 based length
            else : # infer 'int_num_rows' for stacked czdf (combined zdf)
                int_num_rows = sum( self._l_n_rows_unfiltered ) # assumes given zdf has valid number of rows
                
        # apply filter once the zdf is properly initialized
        self.filter = ba_filter

        # open or initialize zdf and retrieve associated metadata
        if not zarr_exists( path_folder_zdf ) : # if the object does not exist, initialize ZarrDataFrame
            # create the output folder
            os.makedirs( path_folder_zdf, exist_ok = True )
            
            self._root = zarr.open( path_folder_zdf, mode = 'a' )
            self._dict_metadata = { 'version' : _version_, 'columns' : set( ), 'int_num_rows_in_a_chunk' : int_num_rows_in_a_chunk, 'flag_enforce_name_col_with_only_valid_characters' : flag_enforce_name_col_with_only_valid_characters, 'flag_store_string_as_categorical' : flag_store_string_as_categorical, 'is_interleaved' : self.is_interleaved, 'is_combined' : self.is_combined } # to reduce the number of I/O operations from lookup, a metadata dictionary will be used to retrieve/update all the metadata
            # if 'int_num_rows' has been given, add it to the metadata
            if int_num_rows is not None :
                self._dict_metadata[ 'int_num_rows' ] = int_num_rows
            self._save_metadata_( ) # save metadata
        else :
            # read existing zdf object
            self._root = zarr.open( path_folder_zdf, mode = 'a' )
                
            # retrieve metadata
            self._dict_metadata = self._root.attrs[ 'dict_metadata' ] 
            # convert 'columns' list to set
            if 'columns' in self._dict_metadata :
                self._dict_metadata[ 'columns' ] = set( self._dict_metadata[ 'columns' ] )
        
        # if a mask is given, open the mask zdf
        self._mask = None # initialize 'mask'
        if path_folder_mask is not None : # if a mask is given
            self._mask = ZarrDataFrame( 
                path_folder_mask, 
                df = df, 
                int_num_rows = self._n_rows_unfiltered,
                int_num_rows_in_a_chunk = self.metadata[ 'int_num_rows_in_a_chunk' ], 
                ba_filter = ba_filter, 
                flag_enforce_name_col_with_only_valid_characters = self.metadata[ 'flag_enforce_name_col_with_only_valid_characters' ], 
                flag_store_string_as_categorical = self.metadata[ 'flag_store_string_as_categorical' ], 
                flag_retrieve_categorical_data_as_integers = flag_retrieve_categorical_data_as_integers, 
                flag_load_data_after_adding_new_column = flag_load_data_after_adding_new_column, 
                mode = 'a', 
                path_folder_mask = None, 
                flag_is_read_only = False,
                l_zdf = l_zdf,
                index_zdf_data_source_when_interleaved = index_zdf_data_source_when_interleaved,
                dict_index_mapping_interleaved = dict_index_mapping_interleaved,
            ) # the mask ZarrDataFrame shoud not have mask, should be modifiable, and not mode == 'r'.
        
        # handle input arguments
        self._str_invalid_char = '! @#$%^&*()-=+`~:;[]{}\|,<.>/?' + '"' + "'" if self._dict_metadata[ 'flag_enforce_name_col_with_only_valid_characters' ] else '/' # linux file system does not allow the use of linux'/' character in the folder/file name

        # initialize loaded data
        self._loaded_data = dict( ) # containing filtered data (if filter is active) or unfiltered data (if filter is not active)
        
        if isinstance( df, pd.DataFrame ) : # if a valid pandas.dataframe has been given
            # update zdf with the given dataframe
            self.update( df )
            
        # initialize attribute storing columns as dictionaries
        self.dict = dict( )
    @property
    def is_combined( self ) :
        """ # 2022-08-25 13:53:20 
        return True if current zdf is in 'combined' mode
        """
        return self._l_zdf is not None
    @property
    def is_interleaved( self ) :
        """ # 2022-08-25 14:03:34 
        return True if current zdf is interleaved 'combined' zdf
        """
        return self._l_dict_index_mapping_interleaved is not None
    @property
    def int_num_rows_in_a_chunk( self ) :
        """ # 2022-08-02 13:01:53 
        return the length of chunk in the primary axis
        """
        return self._dict_metadata[ 'int_num_rows_in_a_chunk' ]
    @int_num_rows_in_a_chunk.setter
    def int_num_rows_in_a_chunk( self, val ) :
        """ # 2022-08-02 13:01:53 
        setting the length of chunk in the primary axis
        """
        self._dict_metadata[ 'int_num_rows_in_a_chunk' ] = val
        self._save_metadata_( ) # save metadata
        # update the settings of the mask, if available.
        if self._mask is not None :
            self.int_num_rows_in_a_chunk = val
    @property
    def metadata( self ) :
        ''' # 2022-07-21 02:38:31 
        '''
        return self._dict_metadata
    @property
    def _n_rows_unfiltered( self ) :
        """ # 2022-06-22 23:12:09 
        retrieve the number of rows in unfiltered ZarrDataFrame. return None if unavailable.
        """
        if 'int_num_rows' not in self._dict_metadata : # if 'int_num_rows' has not been set, return None
            return None
        else : # if 'int_num_rows' has been set
            return self._dict_metadata[ 'int_num_rows' ]
    @property
    def n_rows( self ) :
        """ # 2022-06-22 16:36:54 
        retrieve the number of rows after applying filter. if the filter is not active, return the number of rows of the unfiltered ZarrDataFrame
        """
        if self.filter is None : # if the filter is not active, return the number of rows of the unfiltered ZarrDataFrame
            return self._n_rows_unfiltered
        else : # if a filter is active
            return self._n_rows_after_applying_filter # return the number of active rows in the filter
    @property
    def filter( self ) :
        ''' # 2022-06-22 16:36:22 
        return filter bitarray  '''
        return self._ba_filter
    @filter.setter
    def filter( self, ba_filter ) :
        """ # 2022-08-25 17:17:58 
        change filter, and empty the cache
        """
        if ba_filter is None : # if filter is removed, 
            # if the filter was present before the filter was removed, empty the cache and the temp folder
            if self.filter is not None :
                self._loaded_data = dict( ) # empty the cache
                self.dict = dict( ) # empty the cache for columns stored as dictionaries
            self._ba_filter = None
            self._n_rows_after_applying_filter = None
        else :
            # check whether the given filter is bitarray
            if isinstance( ba_filter, np.ndarray ) : # convert numpy array to bitarray
                ba_filter = bk.BA.to_bitarray( ba_filter )
            assert isinstance( ba_filter, bitarray ) # make sure that the input value is a bitarray object
            
            # check the length of filter bitarray
            if 'int_num_rows' not in self._dict_metadata : # if 'int_num_rows' has not been set, set 'int_num_rows' using the length of the filter bitarray
                self._dict_metadata[ 'int_num_rows' ] = len( ba_filter )
                self._save_metadata_( ) # save metadata
            else :
                # check the length of filter bitarray
                assert len( ba_filter ) == self._dict_metadata[ 'int_num_rows' ]

            self._loaded_data = dict( ) # empty the cache
            self.dict = dict( ) # empty the cache for columns stored as dictionaries
            self._n_rows_after_applying_filter = ba_filter.count( ) # retrieve the number of rows after applying the filter

            self._ba_filter = ba_filter # set bitarray filter
        # set filter of mask
        if hasattr( self, '_mask' ) and self._mask is not None : # propagate filter change to the mask ZDF
            self._mask.filter = ba_filter
            
        if self.is_combined :
            # %% COMBINED %%
            if ba_filter is None : # if filter is removed
                # remove filter from all zdf objects
                for zdf in self._l_zdf : 
                    zdf.filter = None
            else : # if filter is being set
                if self.is_interleaved :
                    # %% COMBINED - INTERLEAVED %%
                    for zdf, dict_index_mapping_interleaved in zip( self._l_zdf, self._l_dict_index_mapping_interleaved ) :
                        # initialize filter for the current zdf object
                        ba_zdf = bitarray( zdf._n_rows_unfiltered )
                        ba_zdf.setall( 0 )
                        # compose filter
                        for int_entry_combined in bk.BA.find( ba_filter ) : # iterate active entries in the combined axis
                            if int_entry_combined in dict_index_mapping_interleaved : # if the active entry also exists in the current axis, update the filter
                                ba_zdf[ dict_index_mapping_interleaved[ int_entry_combined ] ] = 1
                                
                        zdf.filter = ba_zdf # set filter
                else :
                    # %% COMBINED - STACKED %%
                    # for stacked czdf, split the given filter into smaller filters for each zdf
                    int_pos = 0
                    for zdf in self._l_zdf :
                        zdf.filter = ba_filter[ int_pos : int_pos + zdf._n_rows_unfiltered ] # apply a subset of filter
                        int_pos += zdf._n_rows_unfiltered # update 'int_pos'
    def get_column_metadata( self, name_col ) :
        """ # 2022-08-02 11:21:24 
        get metadata of a given column
        """
        if name_col in self : # if the current column is valid
            # if mask is available return the metadata from the mask
            if self._mask is not None and name_col in self._mask : # if the column is available in the mask
                return self._mask.get_column_metadata( name_col = name_col )
            
            # read metadata
            za = zarr.open( f"{self._path_folder_zdf}{name_col}/", mode = 'r' ) # read data from the Zarr object
            dict_col_metadata = za.attrs[ 'dict_col_metadata' ] # retrieve metadata of the current column
            return dict_col_metadata
    def _set_column_metadata( self, name_col, dict_col_metadata ) :
        """ # 2022-08-22 12:36:13 
        a semi-private method for setting metadata of a given column (metadata is supposed to be read-only)
        """
        if name_col in self : # if the current column is valid
            # if mask is available return the metadata from the mask
            if self._mask is not None and name_col in self._mask : # if the column is available in the mask
                return self._mask._set_column_metadata( name_col = name_col, dict_col_metadata = dict_col_metadata )
            
            # read metadata
            za = zarr.open( f"{self._path_folder_zdf}{name_col}/", mode = 'a' ) # read data from the Zarr object
            za.attrs[ 'dict_col_metadata' ] = dict_col_metadata # retrieve metadata of the current column
    def _get_column_path( self, name_col ) :
        """ # 2022-08-26 10:34:35 
        if 'name_col' column exists in the current ZDF object, return the path of the column
        
        === arguments ===
        'name_col' : the name of the column to search
        
        the column will be searched in the following order: main zdf object --> mask zdf object --> component zdf objects, in the order specified in the list.
        """
        path_col = None # set default value
        if name_col is not None and name_col in self : # use 'name_col' as a template if valid name_col has been given
            if name_col in self._dict_metadata[ 'columns' ] : # search the current zdf
                path_col = f"{self._path_folder_zdf}{name_col}/" 
            elif self._mask is not None and name_col in self._mask._dict_metadata[ 'columns' ] : # search mask (if available)
                path_col = f"{self._mask._path_folder_zdf}{name_col}/"
            elif self.is_combined : # search component zdf(s) (if combined mode is active)
                if self.is_interleaved :
                    # %% COMBINED INTERLEAVED %%
                    zdf = self._l_zdf[ self.index_zdf_data_source_when_interleaved ] # retrieve data source zdf
                    path_col = f"{zdf._path_folder_zdf}{name_col}/"
                else :
                    # %% COMBINED STACKED %%
                    for zdf in self._l_zdf : # for each zdf component
                        if name_col in zdf : # when the column was identified
                            path_col = f"{zdf._path_folder_zdf}{name_col}/"
                            break
        return path_col # return the path of the matched column
    def initialize_column( self, name_col, dtype = np.float64, shape_not_primary_axis = ( ), chunks = ( ), categorical_values = None, fill_value = 0, zdf_template : Union[ None, ZarrDataFrame ] = None, name_col_template : Union[ str, None ] = None, path_col_template : Union[ str, None ] = None ) : 
        """ # 2022-08-22 16:30:48 
        initialize columns with a given shape and given dtype
        'dtype' : initialize the column with this 'dtype'
        'shape_not_primary_axis' : initialize the column with this shape excluding the dimension of the primary axis. if an empty tuple or None is given, a 1D array will be initialized. 
                for example, for ZDF with length 100, 
                'shape_not_primary_axis' = ( ) will lead to creation of zarr object of (100,)
                'shape_not_primary_axis' = ( 10, 10 ) will lead to creation of zarr object of (100, 10, 10)
                
        'chunks' : chunk size of the zarr object. length of the chunk along the primary axis can be skipped, which will be replaced by 'int_num_rows_in_a_chunk' of the current ZarrDataFrame attribute
        'categorical_values' : if 'categorical_values' has been given, set the column as a column containing categorical data
        'fill_value' : fill value of zarr object
        'zdf_template' : zdf object from which to search 'name_col_template' column. by default, 
        'name_col_template' : the name of the column to use as a template. if given, 'path_col_template' will be ignored, and use the column as a template to initialize the current column. the column will be searched in the following order: main zdf object --> mask zdf object --> component zdf objects, in the order specified in the list.
        'path_col_template' : the (remote) path to the column to be used as a template. if given, the metadata available in the path will be used to initialize the current column
        """
        # hand over to mask if mask is available
        if self._mask is not None : # if mask is available, save new data to the mask # overwriting on the mask
            self._mask.initialize_column( name_col, dtype = dtype, shape_not_primary_axis = shape_not_primary_axis, chunks = chunks, categorical_values = categorical_values, fill_value = fill_value, zdf_template = zdf_template, name_col_template = name_col_template, path_col_template = path_col_template )
            return
        
        if self._n_rows_unfiltered is None : # if length of zdf has not been set, exit
            return
        
        if name_col in self.columns_excluding_components : # if the column exists in the current ZarrDataFrame (excluding component zdf objects), ignore the call and exit
            return
        
        # retrieve metadata information from the template column object
        if not isinstance( zdf_template, ZarrDataFrame ) : # by default, use self as template zdf object
            zdf_template = self
        if name_col_template is not None and name_col_template in zdf_template : # use 'name_col_template' as a template if valid name_col has been given
            path_col_template = zdf_template._get_column_path( name_col_template ) # retrieve path of the template column from the template zdf               
        if path_col_template is not None and zarr_exists( path_col_template ) : # use 'path_col_template' as a template, retrieve settings
            za = zarr.open( path_col_template ) # open zarr object
            dtype = za.dtype
            shape_not_primary_axis = za.shape[ 1 : ]
            chunks = za.chunks
            fill_value = za.fill_value
            # retrieve column metadata
            dict_metadata = za.attrs[ 'dict_col_metadata' ]
            categorical_values = dict_metadata[ 'l_value_unique' ] if 'flag_categorical' in dict_metadata and dict_metadata[ 'flag_categorical' ] else None # retrieve categorical values
            
        if name_col not in self.columns_excluding_components : # if the column does not exists in the current ZarrDataFrame (excluding component zdf objects )
            # check whether the given name_col contains invalid characters(s)
            for char_invalid in self._str_invalid_char :
                if char_invalid in name_col :
                    raise TypeError( f"the character '{char_invalid}' cannot be used in 'name_col'. Also, the 'name_col' cannot contains the following characters: {self._str_invalid_char}" )
            
            # compose metadata
            dict_col_metadata = { 'flag_categorical' : False } # set a default value for 'flag_categorical' metadata attribute
            dict_col_metadata[ 'flag_filtered' ] = self.filter is not None # mark the column containing filtered data
            
            # initialize a column containing categorical data
            if categorical_values is not None : # if 'categorical_values' has been given
                dict_col_metadata[ 'flag_categorical' ] = True # set metadata for categorical datatype
                set_value_unique = set( categorical_values ) # retrieve a set of unique values
                # handle when np.nan value exist 
                if np.nan in set_value_unique : # when np.nan value was detected
                    if 'flag_contains_nan' not in dict_col_metadata : # update metadata
                        dict_col_metadata[ 'flag_contains_nan' ] = True # mark that the column contains np.nan values
                    set_value_unique.remove( np.nan ) # removes np.nan from the category

                if 'l_value_unique' not in dict_col_metadata :
                    l_value_unique = list( set_value_unique ) # retrieve a list of unique values # can contain mixed types (int, float, str)
                    dict_col_metadata[ 'l_value_unique' ] = list( str( e ) for e in l_value_unique ) # update metadata # convert entries to string (so that all values with mixed types can be interpreted as strings)
                else :
                    set_value_unique_previously_set = set( dict_col_metadata[ 'l_value_unique' ] )
                    l_value_unique = dict_col_metadata[ 'l_value_unique' ] + list( val for val in list( set_value_unique ) if val not in set_value_unique_previously_set ) # extend 'l_value_unique'
                    dict_col_metadata[ 'l_value_unique' ] = l_value_unique # update metadata
                
                # retrieve appropriate datatype for encoding unique categorical values
                int_min_number_of_bits = int( np.ceil( math.log2( len( l_value_unique ) ) ) ) + 1 # since signed int will be used, an additional bit is required to encode the data
                if int_min_number_of_bits <= 8 :
                    dtype = np.int8
                elif int_min_number_of_bits <= 16 :
                    dtype = np.int16
                else :
                    dtype = np.int32
                    
            # initialize the zarr objects
            path_folder_col = f"{self._path_folder_zdf}{name_col}/"
            shape = tuple( [ self._n_rows_unfiltered ] + list( shape_not_primary_axis ) ) # compose 'shape' of the zarr object
            chunks = tuple( chunks ) if len( chunks ) == len( shape ) else tuple( [ self.int_num_rows_in_a_chunk ] + list( chunks ) ) # compose 'chunks' of the zarr object
            assert len( chunks ) == len( shape ) # the length of chunks and shape should be the same
            za = zarr.open( path_folder_col, mode = 'w', shape = shape, chunks = chunks, dtype = dtype, fill_value = fill_value, synchronizer = zarr.ThreadSynchronizer( ) ) # create a new Zarr object if the object does not exist.
            
            # write metadata
            za.attrs[ 'dict_col_metadata' ] = dict_col_metadata

            # add column to zdf (and update the associated metadata)
            self._add_column( name_col )
    def __getitem__( self, args ) :
        ''' # 2022-08-26 14:23:38 
        retrieve data of a column.
        partial read is allowed through indexing (slice/integer index/boolean mask/bitarray is supported)
        if mask is set, retrieve data from the mask if the column is available in the mask. 
        also, when the 'flag_use_mask_for_caching' setting is active, use mask for caching data from source data (possibly remote source).
        
        when combined mode is active, all data of the queried column will be retrieved across the component columns, and saved as a combined column in the current zdf object. Then, the query will be used to retrieve data from the combined column
        '''
        """
        # parse arguments
        
        'name_col' : the name of the column
        'coords' : coordinates/slice/mask for the primary axis
        'coords_rest' : coordinates/slices for axis other than the primary axis
        """
        # initialize indexing
        flag_indexing_primary_axis = False # a boolean flag indicating whether an indexing is active
        flag_coords_in_bool_mask = False
        flag_coords_in_coordinate_arrays = False
        # parse arguments
        if isinstance( args, tuple ) and args[ 1 ] is not None  : # when indexing on the primary axis is active
            flag_indexing_primary_axis = True # update the flag
            # parse the args
            if len( args ) == 2 :
                name_col, coords, coords_rest = args[ 0 ], args[ 1 ], None 
            elif len( args ) > 2 :
                name_col, coords, coords_rest = args[ 0 ], args[ 1 ], args[ 2 : ]
            # detect boolean mask
            flag_coords_in_bool_mask = bk.BA.detect_boolean_mask( coords )
            # convert boolean masks to np.ndarray object
            if flag_coords_in_bool_mask :
                # handle np.ndarray mask
                if isinstance( coords, np.ndarray ) and coords.dtype != bool :
                    coords = coords.astype( bool ) # change dtype
                else : # handle other masks
                    coords = bk.BA.convert_mask_to_array( coords )
            elif isinstance( coords, tuple ) : # if a tuple is given as coords, assumes it contains a list of coordinate arrays
                flag_coords_in_coordinate_arrays = True
        else : 
            # when indexing on the primary axis is not active
            coords = slice( None, None, None ) if self.filter is None else bk.BA.to_array( self.filter ) # retrieve selection filter for the primary axis according to the self.filter
            if isinstance( args, tuple ) : # if indexing in non-primary axis is active
                name_col, coords_rest = args[ 0 ], args[ 2 : ]
            else : # only column name was given
                name_col, coords_rest = args, None 
        flag_indexing_in_non_primary_axis = coords_rest is not None # a flag indicating indexing in non-primary axis is active
        
        """
        # retrieve data
        """
        if name_col not in self : # if name_col is not valid (name_col does not exists in current ZDF, including the mask), exit by returning None
            return None
        # off load to mask
        if self._mask is not None : # if mask is available
            if self.flag_use_mask_for_caching and name_col not in self._mask : # if 'flag_use_mask_for_caching' option is active and the column is not available in the mask, copy the column from the source to the mask
                zarr_copy( f"{self._path_folder_zdf}{name_col}/", f"{self._mask._path_folder_zdf}{name_col}/" ) # copy zarr object from the source to the mask
                self._mask._add_column( name_col ) # manually add column label to the mask
            if name_col in self._mask : # if 'name_col' is available in the mask, retrieve data from the mask.
                return self._mask[ args ]
        # collect data from component zdfs and compose a combined column
        if self.is_combined and name_col not in self.columns_excluding_components : # if data reside only in the component zdf objects, retrieve data from the component objects and save as a column in the current zdf object 
            # %% COMBINED MODE %%
            # if the queried column does not exist in the current zdf or mask zdf, fetch data from component zdf objects and save the data to the current zdf or the mask of it
            if self.is_interleaved :
                # %% COMBINED INTERLEAVED %%
                zdf = self._l_zdf[ self.index_zdf_data_source_when_interleaved ] # retrieve source zdf
                dict_index_mapping_interleaved = self._l_dict_index_mapping_interleaved[ self.index_zdf_data_source_when_interleaved ] # retrieve index-mapping dictionary of source zdf
                assert name_col in zdf # the name_col should exist in the zdf object
                # initialize the 'name_col' column of the current zdf object using the column from the current component zdf
                self.initialize_column( name_col = name_col, zdf_template = zdf, name_col_template = name_col ) 
                
                # transfer data from the source zdf to the combined column of the current zdf 
                l_int_entry_combined, l_int_entry_component = [ ], [ ] # initialize the array 
                for int_entry_combined in dict_index_mapping_interleaved :
                    l_int_entry_combined.append( int_entry_combined )
                    l_int_entry_component.append( dict_index_mapping_interleaved[ int_entry_combined ] )
                    if len( l_int_entry_component ) >= self.int_max_num_entries_per_batch : # if a batch is full, flush the buffer
                        self[ name_col, l_int_entry_combined ] = zdf[ name_col, l_int_entry_component ] # transfer data from the source zdf to the combined column of the current zdf for the current batch
                        l_int_entry_combined, l_int_entry_component = [ ], [ ] # initialize the next batch
                if len( l_int_entry_component ) >= 0 : # if the current batch is not empty, flush the buffer
                    self[ name_col, l_int_entry_combined ] = zdf[ name_col, l_int_entry_component ] # transfer data from the source zdf to the combined column of the current zdf for the current batch
                del l_int_entry_combined, l_int_entry_component
            else :
                # %% COMBINED STACKED %%
                # collect data from stacked czdf 
                int_pos = 0 # initialize the position
                for zdf in self._l_zdf :
                    if name_col in zdf : # if the 'name_col' exist in the current component zdf
                        # initialize the 'name_col' column of the current zdf object using the column from the current component zdf
                        self.initialize_column( name_col = name_col, zdf_template = zdf, name_col_template = name_col ) 
                        
                        # transfer data from component zdf to the combined column of the current zdf object batch by batch
                        st, en = int_pos, int_pos + zdf._n_rows_unfiltered # retrieve start and end coordinates for the current component
                        int_pos_current_component = 0 # initialize the position of current entry in the current batch
                        while st + int_pos_current_component < en : # until all entries for the current component have been processed
                            # transfer data from component zdf to the combined column of the current zdf object for the current batch
                            self[ name_col, slice( st + int_pos_current_component, min( en, st + int_pos_current_component + self.int_max_num_entries_per_batch ) ) ] =  zdf[ name_col, slice( int_pos_current_component, min( en - st, int_pos_current_component + self.int_max_num_entries_per_batch ) ) ] # update the combined column using the values retrieved from the current zdf object
                            int_pos_current_component += self.int_max_num_entries_per_batch # update 'int_pos_current_component' for the next batch
                    int_pos += zdf._n_rows_unfiltered # update 'int_pos'
        
        # retrieve data from zdf objects excluding components (current zdf and mask zdf)
        if name_col in self : # if name_col is valid
            if name_col in self._loaded_data and not flag_indexing_primary_axis : # if a loaded data (filtered/unfiltered, according to the self.filter) is available and indexing is not active, return the cached data
                """ if (filtered), preloaded data is available """
                data = self._loaded_data[ name_col ] # retrieve memory-cached data
                return data[ :, coords_rest ] if flag_indexing_in_non_primary_axis else data # return a subset of result if 'flag_indexing_in_non_primary_axis' is True
            else : 
                """ read data from zarr object """
                
                # open the zarr object
                za = zarr.open( f"{self._path_folder_zdf}{name_col}/", mode = 'r' ) 
                
                if flag_coords_in_bool_mask and isinstance( coords, np.ndarray ) and za.shape == coords.shape :
                    # use mask selection
                    values = za.get_mask_selection( coords )
                elif flag_coords_in_coordinate_arrays :
                    # coordinate array selection
                    values = za.get_coordinate_selection( coords )
                else :
                    # use orthogonal selection as a default
                    values = za.get_orthogonal_selection( tuple( [ coords ] + list( coords_rest ) ) ) if flag_indexing_in_non_primary_axis else za.get_orthogonal_selection( coords )
                
                # check whether the current column contains categorical data
                l_value_unique = self.get_categories( name_col ) # non-categorical data will get an empty list
                if len( l_value_unique ) == 0 or self._flag_retrieve_categorical_data_as_integers : # handle non-categorical data
                    return values
                else : # decode categorical data
                    values = values.astype( object ) # prepare data for storing categorical data
                    # perform decoding 
                    for t_coord, val in np.ndenumerate( values ) :
                        values[ t_coord ] = l_value_unique[ val ] if val >= 0 else np.nan # convert integer representations to its original string values # -1 (negative integers) encodes np.nan
                    return values
    def __setitem__( self, args, values ) :
        ''' # 2022-08-27 10:21:01 
        save/update a column at indexed positions.
        when a filter is active, only active entries will be saved/updated automatically.
        boolean mask/integer arrays/slice indexing is supported. However, indexing will be applied to the original column with unfiltered rows (i.e., when indexing is active, filter will be ignored)
        if mask is set, save data to the mask
        
        automatically detect dtype of the input array/list, including that of categorical data (all string data will be interpreted as categorical data). when the original dtype and dtype inferred from the updated values are different, an error will occur.
        
        # 2022-08-03 00:32:19 multi-dimensional data 'columns' is now supported. now complex selection approach can be used to set/view values of a given column as shown below
        
        zdf[ 'new_col', ( [ 0, 1, 2 ], [ 0, 0, 1 ] ) ] = [ 'three', 'new', 'values' ] # coordinate selection when a TUPLE is given
        zdf[ 'new_col', : 10 ] = np.arange( 1000 ).reshape( ( 10, 100 ) ) # orthogonal selection 
        
        '''
        """
        1) parse arguments
        """
        if self._mode == 'r' : # if mode == 'r', ignore __setitem__ method calls
            return 
        
        # initialize indexing
        flag_indexing_primary_axis = False # a boolean flag indicating whether an indexing is active
        flag_coords_in_bool_mask = False
        flag_coords_in_coordinate_arrays = False
        # parse arguments
        if isinstance( args, tuple ) and args[ 1 ] is not None  : # when indexing on the primary axis is active
            flag_indexing_primary_axis = True # update the flag
            # parse the args
            if len( args ) == 2 :
                name_col, coords, coords_rest = args[ 0 ], args[ 1 ], None 
            elif len( args ) > 2 :
                name_col, coords, coords_rest = args[ 0 ], args[ 1 ], args[ 2 : ]
            # detect boolean mask
            flag_coords_in_bool_mask = bk.BA.detect_boolean_mask( coords )
            # convert boolean masks to np.ndarray object
            if flag_coords_in_bool_mask :
                # handle np.ndarray mask
                if isinstance( coords, np.ndarray ) and coords.dtype != bool :
                    coords = coords.astype( bool ) # change dtype
                else : # handle other masks
                    coords = bk.BA.convert_mask_to_array( coords )
            elif isinstance( coords, tuple ) : # if a tuple is given as coords, assumes it contains a list of coordinate arrays
                flag_coords_in_coordinate_arrays = True
        else : 
            # when indexing on the primary axis is not active
            coords = slice( None, None, None ) if self.filter is None else bk.BA.to_array( self.filter ) # retrieve selection filter for the primary axis according to the self.filter
            if isinstance( args, tuple ) : # if indexing in non-primary axis is active
                name_col, coords_rest = args[ 0 ], args[ 2 : ]
            else : # only column name was given
                name_col, coords_rest = args, None 
        flag_indexing_in_non_primary_axis = coords_rest is not None # a flag indicating indexing in non-primary axis is active
        
        # check whether the given name_col contains invalid characters(s)
        for char_invalid in self._str_invalid_char :
            if char_invalid in name_col :
                raise TypeError( f"the character '{char_invalid}' cannot be used in 'name_col'. Also, the 'name_col' cannot contains the following characters: {self._str_invalid_char}" )
        
        """
        2) set data
        """
        # if mask is available, save new data/modify existing data to the mask # overwriting on the mask
        if self._mask is not None : # if mask is available, save new data to the mask
            if name_col in self and name_col not in self._mask : # if the 'name_col' exists in the current ZarrDataFrame and not in mask, copy the column to the mask
                zarr_copy( f"{self._path_folder_zdf}{name_col}/", f"{self._mask._path_folder_zdf}{name_col}/" ) # copy zarr object from the source to the mask
                self._mask._add_column( name_col ) # manually add column label to the mask
            self._mask[ args ] = values # set values to the mask
            return # exit
    
        if self._flag_is_read_only : # if current store is read-only (and mask is not set), exit
            return # exit
        
        """
        retrieve metadata and infer dtypes
        """
        # set default fill_value
        fill_value = 0 # set default fill_value
        # define zarr object directory
        path_folder_col = f"{self._path_folder_zdf}{name_col}/" # compose the output folder
        # retrieve/initialize metadata
        flag_col_already_exists = zarr_exists( path_folder_col ) # retrieve a flag indicating that the column already exists
        if flag_col_already_exists :
            ''' read settings from the existing columns '''
            za = zarr.open( path_folder_col, 'a' ) # open Zarr object
            dict_col_metadata = za.attrs[ 'dict_col_metadata' ] # load previous written metadata
            
            # retrieve dtype
            dtype = str if dict_col_metadata[ "flag_categorical" ] else za.dtype # dtype of cetegorical data columns should be str
        else :
            dtype = None # define default dtype
            ''' create a metadata of the new column '''
            dict_col_metadata = { 'flag_categorical' : False } # set a default value for 'flag_categorical' metadata attribute
            dict_col_metadata[ 'flag_filtered' ] = self.filter is not None # mark the column containing filtered data
            
            # infer the data type of input values
            # if values is numpy.ndarray, use the dtype of the array
            if isinstance( values, np.ndarray ) :
                dtype = values.dtype
                
            # if values is not numpy.ndarray or the dtype is object datatype, use the type of the data returned by the type( ) python function.
            if not isinstance( values, np.ndarray ) or dtype is np.dtype( 'O' ) : 
                
                # extract the first entry from the array
                val = values
                while hasattr( val, '__iter__' ) and not isinstance( val, str ) :
                    val = next( val.__iter__() ) # get the first element of the current array
                dtype = type( val )
                
                # check whether the array contains strings with np.nan values
                if dtype is float and val is np.nan :
                    for t_coord, val in np.ndenumerate( values ) : # np.ndenumerate can handle nexted lists
                        if type( val ) is str :
                            dtype = str
                            break
                            
            # update the length of zdf if it has not been set.
            if self._n_rows_unfiltered is None : # if a valid information about the number of rows is available
                self._dict_metadata[ 'int_num_rows' ] = len( values ) # retrieve the length of the primary axis
                self._save_metadata_( ) # save metadata
        
        """ convert data to np.ndarray """
        # retrieve data values from the 'values' 
        if isinstance( values, bitarray ) :
            values = bk.BA.to_array( values ) # retrieve boolean values from the input bitarray
        if isinstance( values, pd.Series ) :
            values = values.values
        # convert values that is not numpy.ndarray to numpy.ndarray object (for the consistency of the loaded_data)
        if not isinstance( values, np.ndarray ) :
            values = np.array( values, dtype = object if dtype is str else dtype ) # use 'object' dtype when converting values to a numpy.ndarray object if dtype is 'str'
            
        # retrieve shape and chunk sizes of the object
        shape = tuple( [ self._n_rows_unfiltered ] + list( values.shape )[ 1 : ] )
        chunks = tuple( [ self._dict_metadata[ 'int_num_rows_in_a_chunk' ] ] + list( values.shape )[ 1 : ] )
        
        # print( shape, chunks, dtype, self._dict_metadata[ 'flag_store_string_as_categorical' ] )
        # write categorical data
        if dtype is str and self._dict_metadata[ 'flag_store_string_as_categorical' ] : # storing categorical data   
            # default fill_value for categorical data is -1 (representing np.nan values)
            fill_value = -1
            # compose metadata of the column
            dict_col_metadata[ 'flag_categorical' ] = True # set metadata for categorical datatype
            
            ''' retrieve unique values for categorical data '''
            set_value_unique = set( e[ 1 ] for e in np.ndenumerate( values ) ) # retrieve a set of unique values in the input array
            
            # handle when np.nan value exist 
            if np.nan in set_value_unique : # when np.nan value was detected
                if 'flag_contains_nan' not in dict_col_metadata : # update metadata
                    dict_col_metadata[ 'flag_contains_nan' ] = True # mark that the column contains np.nan values
                set_value_unique.remove( np.nan ) # removes np.nan from the category
            
            if 'l_value_unique' not in dict_col_metadata :
                l_value_unique = list( set_value_unique ) # retrieve a list of unique values # can contain mixed types (int, float, str)
                dict_col_metadata[ 'l_value_unique' ] = list( str( e ) for e in l_value_unique ) # update metadata # convert entries to string (so that all values with mixed types can be interpreted as strings)
            else :
                set_value_unique_previously_set = set( dict_col_metadata[ 'l_value_unique' ] )
                l_value_unique = dict_col_metadata[ 'l_value_unique' ] + list( val for val in list( set_value_unique ) if val not in set_value_unique_previously_set ) # extend 'l_value_unique'
                dict_col_metadata[ 'l_value_unique' ] = l_value_unique # update metadata
            
            # retrieve appropriate datatype for encoding unique categorical values
            int_min_number_of_bits = int( np.ceil( math.log2( len( l_value_unique ) ) ) ) + 1 # since signed int will be used, an additional bit is required to encode the data
            if int_min_number_of_bits <= 8 :
                dtype = np.int8
            elif int_min_number_of_bits <= 16 :
                dtype = np.int16
            else :
                dtype = np.int32
            
            # open Zarr object representing the current column
            za = zarr.open( path_folder_col, mode = 'a', synchronizer = zarr.ThreadSynchronizer( ) ) if zarr_exists( path_folder_col ) else zarr.open( path_folder_col, mode = 'w', shape = shape, chunks = chunks, dtype = dtype, fill_value = fill_value, synchronizer = zarr.ThreadSynchronizer( ) ) # create a new Zarr object if the object does not exist.
            
            # if dtype changed from the previous zarr object, re-write the entire Zarr object with changed dtype. (this will happens very rarely, and will not significantly affect the performance)
            if dtype != za.dtype : # dtype should be larger than za.dtype if they are not equal (due to increased number of bits required to encode categorical data)
                if self.verbose :
                    print( f'{za.dtype} will be changed to {dtype}' )
                path_folder_col_new = f"{self._path_folder_zdf}{name_col}_{UUID( )}/" # compose the new output folder
                za_new = zarr.open( path_folder_col_new, mode = 'w', shape = ( self._n_rows_unfiltered, ), chunks = ( self._dict_metadata[ 'int_num_rows_in_a_chunk' ], ), dtype = dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # create a new Zarr object using the new dtype
                za_new[ : ] = za[ : ] # copy the data 
                shutil.rmtree( path_folder_col ) # delete the previous Zarr object
                os.rename( path_folder_col_new, path_folder_col ) # replace the previous Zarr object with the new object
                za = zarr.open( path_folder_col, mode = 'a', synchronizer = zarr.ThreadSynchronizer( ) ) # open the new Zarr object
            
            # encode data
            dict_encode_category = dict( ( e, i ) for i, e in enumerate( l_value_unique ) ) # retrieve a dictionary encoding value to integer representation of the value
            
            # perform encoding 
            values_before_encoding = values
            values = np.zeros_like( values_before_encoding, dtype = dtype ) # initialize encoded values
            for t_coord, val in np.ndenumerate( values_before_encoding ) : # np.ndarray object can be encoded.
                values[ t_coord ] = dict_encode_category[ val ] if val in dict_encode_category else -1 # encode strings into integer representations # -1 (negative integers) encodes np.nan, which is a fill_value for zarr object containing categorical data
            
        # open zarr object and write data
        za = zarr.open( path_folder_col, mode = 'a', synchronizer = zarr.ThreadSynchronizer( ) ) if zarr_exists( path_folder_col ) else zarr.open( path_folder_col, mode = 'w', shape = shape, chunks = chunks, dtype = dtype, fill_value = fill_value, synchronizer = zarr.ThreadSynchronizer( ) ) # create a new Zarr object if the object does not exist.
        
        if flag_coords_in_bool_mask and isinstance( coords, np.ndarray ) and za.shape == coords.shape : 
            # use mask selection
            za.set_mask_selection( coords, values )
        elif flag_coords_in_coordinate_arrays :
            # coordinate array selection
            za.set_coordinate_selection( coords, values )
        else :
            # use orthogonal selection as a default
            za.set_orthogonal_selection( tuple( [ coords ] + list( coords_rest ) ), values ) if flag_indexing_in_non_primary_axis else za.set_orthogonal_selection( coords, values )
            
        # save column metadata
        za.attrs[ 'dict_col_metadata' ] = dict_col_metadata
        
        # update zdf metadata
        if name_col not in self._dict_metadata[ 'columns' ] :
            self._dict_metadata[ 'columns' ].add( name_col )
            self._save_metadata_( )
        
        # if indexing was used to partially update the data, remove the cache, because it can cause inconsistency
        if flag_indexing_primary_axis and name_col in self._loaded_data :
            del self._loaded_data[ name_col ]
        # add data to the loaded data dictionary (object cache) if 'self._flag_load_data_after_adding_new_column' is True and indexing was not used
        if self._flag_load_data_after_adding_new_column and not flag_indexing_primary_axis and coords_rest is None :  # no indexing through secondary axis, too
            self._loaded_data[ name_col ] = values_before_encoding if dict_col_metadata[ 'flag_categorical' ] else values
    def __delitem__( self, name_col ) :
        ''' # 2022-06-20 21:57:38 
        remove the column from the memory and the object on disk
        if mask is set, delete the column of the mask, and does not delete columns of the original ZarrDataFrame
        '''
        if self._mode == 'r' : # if mode == 'r', ignore __delitem__ method calls
            return # exit
        
        # if mask is available, delete the column from the mask
        if self._mask is not None : # if mask is available
            if name_col in self._mask : # if the 'name_col' exists in the mask ZarrDataFrame
                del self._mask[ name_col ] # delete the column from the mask
            return # exit
    
        if self._flag_is_read_only : # if current store is read-only (and mask is not set), exit
            return # exit
        
        if name_col in self : # if the given name_col is valid
            # remove column from the memory
            self.unload( name_col ) 
            # remove the column from metadata
            self._dict_metadata[ 'columns' ].remove( name_col )
            self._save_metadata_( ) # update metadata
            # delete the column from the disk ZarrDataFrame object
            shutil.rmtree( f"{self._path_folder_zdf}{name_col}/" ) #             OS_Run( [ 'rm', '-rf', f"{self._path_folder_zdf}{name_col}/" ] )
    def __repr__( self ) :
        """ # 2022-07-20 23:00:15 
        """
        return f"<ZarrDataFrame object{'' if self._n_rows_unfiltered is None else ' containing '}{'' if self.filter is None else f'{self.n_rows}/'}{'' if self._n_rows_unfiltered is None else f'{self._n_rows_unfiltered} rows'} stored at {self._path_folder_zdf}\n\twith the following columns: {sorted( self._dict_metadata[ 'columns' ] )}" + ( '\n\t[combined]-' + ( '(interleaved)' if self.is_interleaved else '(stacked)' ) + f" ZarrDataFrame, composed of the following ZarrDataFrame objects:\n[" + '\n'.join( str( zdf ) for zdf in self._l_zdf ) + "]" if self.is_combined else '' ) +  ">"
    @property
    def columns( self ) :
        ''' # 2022-08-25 17:33:18 
        return available column names (including mask and components) as a set
        '''
        # retrieve columns
        columns = self._dict_metadata[ 'columns' ]
        # add columns of mask
        if self._mask is not None : # if mask is available :
            columns = columns.union( self._mask._dict_metadata[ 'columns' ] ) # return the column names of the current ZDF and the mask ZDF
        # add columns from zdf components
        if self.is_combined :
            if self.is_interleaved :
                # %% COMBINED INTERLEAVED %%
                zdf = self._l_zdf[ self.index_zdf_data_source_when_interleaved ] # retrieve data source zdf
                columns = columns.union( zdf.columns ) # add columns of the data source zdf
            else :
                # %% COMBINED STACKED %%
                for zdf in self._l_zdf : # for each zdf component
                    columns = columns.union( zdf.columns ) # add columns of the zdf component
        return columns
    @property
    def columns_excluding_components( self ) :
        ''' # 2022-08-26 14:23:27 
        return available column names (including mask but excluding components) as a set
        '''
        # retrieve columns
        columns = self._dict_metadata[ 'columns' ]
        # add columns of mask
        if self._mask is not None : # if mask is available :
            columns = columns.union( self._mask._dict_metadata[ 'columns' ] ) # return the column names of the current ZDF and the mask ZDF
        return columns
    def __contains__( self, name_col ) :
        """ # 2022-08-25 17:33:22 
        check whether a column name exists in the given ZarrDataFrame
        """
        return name_col in self.columns
    def __iter__( self ) :
        """ # 2022-07-20 22:57:19 
        iterate name of columns in the current ZarrDataFrame
        """
        if self._mask is not None : # if mask is available :
            return iter( self.columns.union( self._mask.columns ) ) # iterate over the column names of the current ZDF and the mask ZDF
        else :
            return iter( self.columns )
    @property
    def df( self ) :
        ''' # 2022-07-01 22:32:00 
        return loaded data as a dataframe, with properly indexed rows
        '''
        arr_index = np.arange( self._n_rows_unfiltered ) if self.filter is None else bk.BA.to_integer_indices( self.filter ) # retrieve integer indices of the rows
        if len( self._loaded_data ) > 0 : # if a cache is not empty
            df = pd.DataFrame( self._loaded_data )
            df.index = arr_index # add integer indices of the rows
        else :
            df = pd.DataFrame( index = arr_index ) # build an empty dataframe using the integer indices
        return df
    def _add_column( self, name_col ) :
        """ # 2022-08-06 13:28:42 
        a semi-private method for adding column label to the metadata of the current ZarrDataFrame (not added to the metadata of the mask)
        """
        if name_col not in self :
            self._dict_metadata[ 'columns' ].add( name_col )
            self._save_metadata_( )
    def _save_metadata_( self ) :
        ''' # 2022-07-20 10:31:39 
        save metadata of the current ZarrDataFrame
        '''
        if not self._flag_is_read_only : # save metadata only when it is not in the read-only mode
            # convert 'columns' to list before saving attributes
            temp = self._dict_metadata[ 'columns' ]
            self._dict_metadata[ 'columns' ] = list( temp )
            self._root.attrs[ 'dict_metadata' ] = self._dict_metadata # update metadata
            self._dict_metadata[ 'columns' ] = temp # revert 'columns' to set
    def get_categories( self, name_col ) :
        """ # 2022-06-21 00:57:37 
        for columns with categorical data, return categories. if the column contains non-categorical data, return an empty list
        """
        if name_col in self : # if the current column is valid
            # if the column is available in the mask, return the result of the mask
            if self._mask is not None and name_col in self._mask :
                return self._mask.get_categories( name_col = name_col )
            
            za = zarr.open( f"{self._path_folder_zdf}{name_col}/", mode = 'r' ) # read data from the Zarr object
            dict_col_metadata = za.attrs[ 'dict_col_metadata' ] # retrieve metadata of the current column
            if dict_col_metadata[ 'flag_categorical' ] : # if the current column contains categorical data
                return dict_col_metadata[ 'l_value_unique' ]
            else :
                return [ ]
    def update( self, df, flag_use_index_as_integer_indices = True ) :
        """ # 2022-06-20 21:36:55 
        update ZarrDataFrame with the given 'df'
        
        'flag_use_index_as_integer_indices' : partial update is possible by setting indices of the rows of input DataFrame as the integer indices of the rows of the current ZarrDataFrame and setting 'flag_use_index_as_integer_indices' to True
        """
        # retrieve coordinates for partial 
        coords = df.index.values if flag_use_index_as_integer_indices else slice( None, None, None )
        
        # update each column
        for name_col in df.columns.values :
            self[ name_col, coords ] = df[ name_col ]
    def load( self, * l_name_col ) :
        ''' # 2022-06-20 22:09:42 
        load given column(s) into the memory
        '''
        for name_col in l_name_col :
            if name_col not in self : # skip invalid column
                continue
            if name_col not in self._loaded_data : # if the data has not been loaded
                self._loaded_data[ name_col ] = self[ name_col ]
    def unload( self, * l_name_col ) :
        """ # 2022-06-20 22:09:37 
        remove the column from the memory.
        if no column names were given, unload an entire cache
        """
        # if no column names were given, unload an entire cache
        if len( l_name_col ) == 0 :
            self._loaded_data = dict( )
        # if more than one column name was given, unload data of a subset of cache
        for name_col in l_name_col :
            if name_col not in self : # skip invalid column
                continue
            if name_col in self._loaded_data :
                del self._loaded_data[ name_col ]
    def delete( self, * l_name_col ) :
        ''' # 2022-06-20 22:09:31 
        remove the column from the memory and from the disk
        '''
        for name_col in l_name_col :
            if name_col not in self : # skip invalid column
                continue
            del self[ name_col ] # delete element from the current object
    def get_df( self, * l_name_col ) :
        """ # 2022-08-05 13:54:36 
        get dataframe of a given list of columns, and empty the cache
        """
        set_name_col = set( e for e in l_name_col if isinstance( e, str ) and e in self ) # convert 'l_name_col' to set # use only hashable strings
        self.unload( * list( name_col for name_col in self if name_col not in set_name_col ) ) # drop the columns that do not belonging to 'l_name_col'
        self.load( * set_name_col ) # load the given list of columns
        df = self.df # retrieve dataframe
        self.unload( ) # empty the cache
        return df
    def get_shape( self, name_col ) :
        """ # 2022-08-07 16:01:12 
        return the shape of the given column except for the dimension along the primary axis.
        """
        # the column should exist
        if name_col not in self :
            if self.verbose :
                print( f'{name_col} not available in the current ZarrDataFrame, exiting' )
            return
        
        if self._mask is not None : # if mask is available
            if name_col in self._mask : # if the column is available in the mask
                return self._mask.get_shape( name_col ) # return the result of the mask object
        
        # open a zarr object, and access the shape
        path_folder_zarr = f"{self._path_folder_zdf}{name_col}/"
        za = zarr.open( path_folder_zarr, mode = 'r' ) 
        return za.shape[ 1 : ] # return the shape including the dimension of the primary axis
    def save( self, path_folder_zdf, * l_name_col ) :
        """ # 2022-07-04 21:09:34 
        save data contained in the ZarrDataFrame object to the new path.
        if a filter is active, filtered ZarrDataFrame will be saved.
        
        'path_folder_zdf' : the output ZarrDataFrame object
        'l_name_col' : the list of names of columns to save. if no column name is given, copy all columns in the current ZarrDataFrame
        """
        # check validity of the path
        path_folder_zdf = os.path.abspath( path_folder_zdf ) + '/' # retrieve abspath of the output object
        assert self._path_folder_zdf != path_folder_zdf # the output folder should not be same as the folder of the current ZarrDataFrame

        zdf = ZarrDataFrame( path_folder_zdf, flag_retrieve_categorical_data_as_integers = self._flag_retrieve_categorical_data_as_integers, flag_load_data_after_adding_new_column = self._flag_load_data_after_adding_new_column ) # open a new zdf using the same setting as the current ZarrDataFrame
        
        # handle empty 'l_name_col'
        if len( l_name_col ) == 0 :
            l_name_col = self.columns # if no column name is given, copy all columns in the current ZarrDataFrame to the new ZarrDataFrame
        
        for name_col in self.columns.intersection( l_name_col ) : # copy column by column to the output ZarrDataFrame object
            zdf.initialize_column( name_col, zdf_template = self, name_col_template = name_col ) # initialize the column using the column of the current zdf object 
            zdf[ name_col ] = self[ name_col ] # copy data (with filter applied)
    def load_as_dict( self, * l_name_col, float_min_proportion_of_active_rows_for_using_array_as_dict = 0.1 ) :
        """ # 2022-07-06 01:29:51 
        load columns as dictionaries, which is accessible through the self.dict attribute, where keys are integer representation of rows and values are data values
        
        'float_min_proportion_of_active_rows_for_using_array_as_dict' : A threshold for the transition from dictionary to array for the conversion of coordinates. empirically, dictionary of the same length takes about ~10 times more memory than the array. 
                                                                        By default, when the number of active entries in an exis > 10% (or above any proportion that can set by 'float_min_proportion_of_active_rows_for_using_array_as_dict'), an array representing all rows will be used for the conversion of coordinates.
        """
        set_name_col = self.columns.intersection( l_name_col ) # retrieve a set of valid column names
        if len( set_name_col ) == 0 : # exit if there is no valid column names
            return
        
        n = self._n_rows_unfiltered # retrieve the number of rows in the unfiltered ZarrDataFrame
        arr_index = np.arange( n, dtype = int ) if self.filter is None else bk.BA.to_integer_indices( self.filter ) # retrieve integer indices of the rows
        for name_col in set_name_col : 
            if name_col in self.dict : # ignore columns that were already loaded
                continue
            values = self[ name_col ] # retrieve values of the given column
            dict_data = np.zeros( n, dtype = values.dtype ) if ( self.n_rows / n ) > float_min_proportion_of_active_rows_for_using_array_as_dict else dict( ) # implement a dictionary using an array if the proportion of active rows of ZarrDataFrame is larger than the given threshold to reduce the memory footprint and increase the efficiency of access
            for int_index_row, val in zip( arr_index, values ) : # iterate through data values of the active rows
                dict_data[ int_index_row ] = val
            del values
            self.dict[ name_col ] = dict_data # add column loaded as a dictionary to the cache    
    def get_zarr( self, name_col ) :
        """ # 2022-08-06 11:29:58 
        get multiprocessing-enabled (with filesystem-lock) zarr object of the given column. 
        
        
        """
        # the column should exist
        if name_col not in self :
            if self.verbose :
                print( f'{name_col} not available in the current ZarrDataFrame, exiting' )
            return
        
        if self._mask is not None : # if mask is available
            if name_col not in self._mask : # if the column is not available in the mask, copy the column from the source to the mask
                zarr_copy( f"{self._path_folder_zdf}{name_col}/", f"{self._mask._path_folder_zdf}{name_col}/" ) # copy zarr object from the source to the mask
                self._mask._add_column( name_col ) # manually add column label to the mask
            return self._mask.get_zarr( name_col ) # return the result of the mask object
                
        # define pathes
        path_folder_lock = f"{self._path_folder_zdf}{name_col}_lock.sync/" # define path to locks for parallel processing with multiple processes
        path_folder_zarr = f"{self._path_folder_zdf}{name_col}/"
        
        # if lock already exists, exit
        if os.path.exists( path_folder_lock ) :
            if self.verbose :
                print( f'current column {name_col} appear to be used in another processes, exiting' )
            return None, None
            
        # if mode == 'r', return read-only object
        if self._mode == 'r' :
            def __delete_nothing( ) :
                """ # 2022-08-06 13:36:10 
                place-holding dummy function
                """
                pass
            return zarr.open( path_folder_zarr, 'r' ), __delete_nothing # when array is read-only, it is safe to read from multiple processes
                
        # open a zarr object, write-from-multiple-processes-enabled
        za = zarr.open( path_folder_zarr, mode = 'a', synchronizer = zarr.ProcessSynchronizer( path_folder_lock ) ) # use process-sync lock

        def __delete_locks( ) :
            """ # 2022-08-06 13:20:57 
            destroy the locks used for multiprocessing-enabled modification of a zarr object
            """
            shutil.rmtree( path_folder_lock )
        return za, __delete_locks
        
''' a class for accessing Zarr-backed count matrix data (RAMtx, Random-Access matrix) '''
class RAMtx( ) :
    """ # 2022-08-16 02:47:08 
    This class represent a random-access mtx format for memory-efficient exploration of extremely large single-cell transcriptomics/genomics data.
    This class use a count matrix data stored in a random read-access compatible format, called RAMtx, enabling exploration of a count matrix with hundreds of millions cells with hundreds of millions of features.
    Also, the RAMtx format is supports multi-processing, and provide convenient interface for parallel processing of single-cell data
    Therefore, for exploration of count matrix produced from 'scarab count', which produces dozens of millions of features extracted from both coding and non coding regions, this class provides fast front-end application for exploration of exhaustive data generated from 'scarab count'
    
    
    # 'mode' of RAMtx objects
    There are three valid 'mode' (or internal structures) for RAMtx object : {'dense' or 'sparse_for_querying_barcodes', 'sparse_for_querying_features'}
    
    (sparse_for_querying_barcodes) <---> (dense) <---> (sparse_for_querying_features)
    *fast barcode data retrieval                       *fast feature data retrieval
    
    As shown above, RAMtx objects are interconvertible. Of note, for the converion between sparse ramtx sorted by barcodes and features, 'dense' ramtx object should be used in the conversion process.
    - dense ramtx object can be used to retrieve data of a single barcode or feature (with moderate efficiency)
    - sparse ramtx object can only be used data of either a single barcode ('sparse_for_querying_barcodes') or a single feature ('sparse_for_querying_features') very efficiently.
    
    
    arguments:
    'path_folder_ramtx' : a folder containing RAMtx object
    'ba_filter_features' : a bitarray filter object for features. active element is marked by 1 and inactive element is marked by 0
    'ba_filter_barcodes' : a bitarray filter object for barcodes. active element is marked by 1 and inactive element is marked by 0
    'dtype_of_feature_and_barcode_indices' : dtype of feature/barcode indices 
    'dtype_of_values' : dtype of values. 
    'int_num_cpus' : the number of processes that will be used for random accessing of the data
    'mode' : file mode. 'r' for read-only mode and 'a' for a mode allowing modifications
    'flag_is_read_only' : read-only status of RamData
    'path_folder_ramtx_mask' : a local (local file system) path to the mask of the RAMtx object that allows modifications to be written without modifying the source. if a valid local path to a mask is given, all modifications will be written to the mask
    'is_for_querying_features' : a flag for indicating whether the current RAMtx will be used for querying features. for sparse matrix, this attribute will be fixed. However, for dense matrix, this atrribute can be changed any time.
    'int_total_number_of_values_in_a_batch_for_dense_matrix' : the total number of values that will be loaded in dense format for each minibatch (subbatch) when retrieving sparse data from the dense matrix using multiple number of processes. this setting can be changed later
    
    """
    def __init__( self, path_folder_ramtx, ramdata = None, dtype_of_feature_and_barcode_indices = np.uint32, dtype_of_values = np.float64, int_num_cpus = 1, verbose = False, flag_debugging = False, mode = 'a', flag_is_read_only = False, path_folder_ramtx_mask = None, is_for_querying_features = True, int_total_number_of_values_in_a_batch_for_dense_matrix = 10000000 ) :
        """ # 2022-07-31 00:49:59 
        """
        # read metadata
        self._root = zarr.open( path_folder_ramtx, 'a' )
        self._dict_metadata = self._root.attrs[ 'dict_metadata' ] # retrieve the metadata
        
        # parse the metadata of the RAMtx object
        self._int_num_features, self._int_num_barcodes, self._int_num_records = self._dict_metadata[ 'int_num_features' ], self._dict_metadata[ 'int_num_barcodes' ], self._dict_metadata[ 'int_num_records' ]
        
        # set attributes 
        self._dtype_of_feature_and_barcode_indices = dtype_of_feature_and_barcode_indices
        self._dtype_of_values = dtype_of_values
        self._path_folder_ramtx = path_folder_ramtx
        self.verbose = verbose 
        self.flag_debugging = flag_debugging
        self.int_num_cpus = int_num_cpus
        self._ramdata = ramdata
        self._mode = mode
        self._flag_is_read_only = flag_is_read_only
        self._path_folder_ramtx_mask = path_folder_ramtx_mask
        self.int_total_number_of_values_in_a_batch_for_dense_matrix = int_total_number_of_values_in_a_batch_for_dense_matrix
        
        # set filters using RamData
        self.ba_filter_features = ramdata.ft.filter if ramdata is not None else None
        self.ba_filter_barcodes = ramdata.bc.filter if ramdata is not None else None
        
        self._is_sparse = self.mode != 'dense' # retrieve a flag indicating whether ramtx is dense
        if self.is_sparse :
            self._is_for_querying_features = self._dict_metadata[ 'flag_ramtx_sorted_by_id_feature' ] # for sparse matrix, this attribute is fixed
            # open Zarr object containing matrix and matrix indices
            self._za_mtx_index = zarr.open( f'{self._path_folder_ramtx}matrix.index.zarr', 'r' )
            self._za_mtx = zarr.open( f'{self._path_folder_ramtx}matrix.zarr', 'r' )
        else :
            self.is_for_querying_features = is_for_querying_features # set this attribute
            self._za_mtx = zarr.open( f'{self._path_folder_ramtx}matrix.zarr', 'r' )
    @property
    def is_sparse( self ) :
        """ # 2022-08-04 13:59:15 
        """
        return self._is_sparse
    @property
    def mode( self ) :
        """ # 2022-07-30 20:13:32 
        """
        return self._dict_metadata[ 'mode' ]
    @property
    def _path_folder_ramtx_modifiable( self ) :
        """ # 2022-07-21 09:04:28 
        return the path to the modifiable RAMtx object
        """
        return ( None if self._flag_is_read_only else self._path_folder_ramtx ) if self._path_folder_ramtx_mask is None else self._path_folder_ramtx_mask
    @property
    def ba_active_entries( self ) :
        """ # 2022-08-04 23:42:23 
        return a bitarray filter of the indexed axis where all the entries with valid count data is marked '1'
        """
        # retrieve axis of current ramtx
        axis = 'features' if self.is_for_querying_features else 'barcodes'
        
        # skip if result is already available
        flag_available = False # initialize
        for path_folder in [ self._path_folder_ramtx, self._path_folder_ramtx_modifiable ] :
            if path_folder is not None and zarr_exists( f'{path_folder}matrix.{axis}.active_entries.zarr/' ) :
                path_folder_zarr = f"{path_folder}matrix.{axis}.active_entries.zarr/" # define an existing zarr object path
                flag_available = True
        if not flag_available and self._path_folder_ramtx_modifiable is not None : # if zarr object does not exists and modifiable ramtx path is available
            # try constructing the zarr object 
            path_folder_zarr = f"{self._path_folder_ramtx_modifiable}matrix.{axis}.active_entries.zarr/" # define zarr object path
            self.survey_number_of_records_for_each_entry( ) # survey the number of records for each entry using default settings
            if not zarr_exists( path_folder_zarr ) : # if the zarr object still does not exists
                # create a full bitarray mask as a fall back
                ba = bitarray( self.len_axis_for_querying )
                ba.setall( 1 )
                return ba
            
        za = zarr.open( path_folder_zarr, mode = 'r', synchronizer = zarr.ThreadSynchronizer( ) ) # open zarr object of the current RAMtx object
        ba = bk.BA.to_bitarray( za[ : ] ) # return the boolean array of active entries as a bitarray object

        # if metadata of the number of active entries is not available, update the metadata
        if 'n_active_entries' in self._dict_metadata :
            self._n_active_entries = ba.count( ) # calculate the number of active entries

            # update metadata
            self._dict_metadata[ 'n_active_entries' ] = self._n_active_entries 
            self._save_metadata_( )

        # return the mask
        return ba
    def survey_number_of_records_for_each_entry( self, axes = [ 'barcodes', 'features' ], int_num_chunks_in_a_batch_for_index_of_sparse_matrix = 100, int_num_chunks_in_a_batch_for_axis_for_querying_dense = 1, int_total_number_of_values_in_a_batch_for_dense_matrix = None, int_size_chunk = 1000, flag_ignore_dense = False, int_num_threads = 20 ) :
        """ # 2022-08-16 11:20:41 
        survey the number of records for each entry in the existing axis
        'axes' : a list of axes to use for surveying the number of records for each entry
        
        === batch size control ===
        'int_num_chunks_in_a_batch_for_index_of_sparse_matrix' : the number of chunks of the index zarr object to be processed once for sparse matrix RAMtx formats
        'int_num_chunks_in_a_batch_for_axis_for_querying_dense' : the number of chunks in the axis for the axis for querying that will be processed together in a batch for each process
        'int_total_number_of_values_in_a_batch_for_dense_matrix' : the total number of values that will be loaded during the suvery for each process for surveying dense matrix with multiple number of processes.
        
        'int_size_chunk' : chunk size for the output zarr objects
        'flag_ignore_dense' : if True, does not survey the dense ramtx.
        'int_num_threads' : the number of threads for surveying
        """
        # use default value when 'int_total_number_of_values_in_a_batch_for_dense_matrix' is None
        if int_total_number_of_values_in_a_batch_for_dense_matrix is None :
            int_total_number_of_values_in_a_batch_for_dense_matrix = self.int_total_number_of_values_in_a_batch_for_dense_matrix
        
        # for each axis 
        for axis in axes :  
            # check validity of the axis name
            if axis not in { 'barcodes', 'features' } :
                continue
            flag_axis_is_barcode = axis == 'barcodes'
            # skip if result is already available
            flag_res_already_available = False # initialize
            for path_folder in [ self._path_folder_ramtx, self._path_folder_ramtx_modifiable ] :
                if path_folder is not None and zarr_exists( f'{path_folder}matrix.{axis}.number_of_records_for_each_entry.zarr/' ) :
                    flag_res_already_available = True
                    break
            if flag_res_already_available :
                continue
                
            # if no modifiable ramtx object is available, exit
            if self._path_folder_ramtx_modifiable is None :
                continue
            
            # if the sparse matrix not for querying with the current axis, continue
            if self.is_sparse and axis not in self.mode : 
                continue
            
            # ignore dense ramtx object if 'flag_ignore_dense' is True
            if flag_ignore_dense and not self.is_sparse :
                continue
            
            # perform survey
            len_axis = self._int_num_barcodes if flag_axis_is_barcode else self._int_num_features # retrieve the length of the axis for querying
            
            # open zarr objects
            za = zarr.open( f'{self._path_folder_ramtx_modifiable}matrix.{axis}.number_of_records_for_each_entry.zarr/', mode = 'w', shape = ( len_axis, ), chunks = ( int_size_chunk, ), dtype = np.float64, synchronizer = zarr.ThreadSynchronizer( ) ) # open zarr object of the current RAMtx object
            za_bool = zarr.open( f"{self._path_folder_ramtx_modifiable}matrix.{axis}.active_entries.zarr/", mode = 'w', shape = ( len_axis, ), chunks = ( int_size_chunk, ), dtype = bool, synchronizer = zarr.ThreadSynchronizer( ) ) # open zarr object of the current RAMtx object
            
            # start worker
            def __write_result( pipe_receiver ) :
                """ # 2022-08-16 11:20:34 
                write survey results as zarr objects
                """
                while True :
                    l_r = pipe_receiver.recv( )
                    if l_r is None :
                        break
                    int_num_entries = sum( len( r[ 1 ] ) for r in l_r ) # retrieve the total number of entries in the 'l_r'
                    pos = 0 # initialize 
                    arr_coord_combined, arr_num_records_combined = np.zeros( int_num_entries, dtype = np.int64 ), np.zeros( int_num_entries, dtype = np.int64 )
                    for r in l_r :
                        sl, arr_num_records = r # parse r
                        int_num_entries_in_r = len( arr_num_records )
                        arr_coord_combined[ pos : pos + int_num_entries_in_r ] = np.arange( sl.start, sl.stop ) # update 'arr_coord_combined'
                        arr_num_records_combined[ pos : pos + int_num_entries_in_r ] = arr_num_records # update 'arr_coord_combined'
                        pos += int_num_entries_in_r # update 'pos'
                    del l_r, r
                    
                    za.set_coordinate_selection( ( arr_coord_combined, ), arr_num_records_combined ) # save the number of records 
                    za_bool.set_coordinate_selection( ( arr_coord_combined, ), arr_num_records_combined > 0 ) # active entry is defined by finding entries with at least one count record
                    del arr_coord_combined, arr_num_records_combined
            pipe_sender, pipe_receiver = mp.Pipe( )
            p = mp.Process( target = __write_result, args = ( pipe_receiver, ) )
            p.start( )
            ns = { 'pipe_sender' : pipe_sender, 'l_buffer' : [ ], 'int_size_buffer' : 20 } # a namespace that will be shared between different scopes
            
            if self.is_sparse : # survey for sparse matrix
                """ %% Sparse matrix %% """
                # surveying on the axis of the sparse matrix
                int_num_entries_processed = 0
                int_num_entries_to_retrieve = int( self._za_mtx_index.chunks[ 0 ] * int_num_chunks_in_a_batch_for_index_of_sparse_matrix )
                while int_num_entries_processed < len_axis :
                    sl = slice( int_num_entries_processed, min( len_axis, int_num_entries_processed + int_num_entries_to_retrieve ) )
                    arr_num_records = self._za_mtx_index[ sl ][ :, 1 ] - self._za_mtx_index[ sl ][ :, 0 ] # retrieve the number of records
                    int_num_entries_processed += int_num_entries_to_retrieve # update the position
                    # flush buffer
                    ns[ 'l_buffer' ].append( ( sl, arr_num_records ) )
                    if len( ns[ 'l_buffer' ] ) >= ns[ 'int_size_buffer' ] :
                        ns[ 'pipe_sender' ].send( ns[ 'l_buffer' ] ) # send result to worker
                        ns[ 'l_buffer' ] = [ ] # initialize the buffer
                    del arr_num_records
            else : # survey for dense matrix (multi-processed)
                """ %% Dense matrix %% """
                # prepare
                len_axis_secondary = self._int_num_features if flag_axis_is_barcode else self._int_num_barcodes # retrieve the length of the axis not for querying
                
                # retrieve chunk size for each axis
                int_size_chunk_axis_for_querying, int_size_chunk_axis_not_for_querying = self._za_mtx.chunks[ 0 if flag_axis_is_barcode else 1 ], self._za_mtx.chunks[ 1 if flag_axis_is_barcode else 0 ] 
                
                # retrieve entries for each axis for batch and a subbatch
                int_num_entries_in_a_batch_in_axis_for_querying = int_size_chunk_axis_for_querying * int_num_chunks_in_a_batch_for_axis_for_querying_dense
                int_num_entries_in_a_subbatch_in_axis_not_for_querying = max( 1, int( np.floor( int_total_number_of_values_in_a_batch_for_dense_matrix / int_num_entries_in_a_batch_in_axis_for_querying ) ) )
                    
                def __gen_batch( ) :
                    """ # 2022-08-15 20:16:51 
                    generate batch on the primary axis
                    """
                    # initialize looping through axis for querying (primary axis)
                    int_num_entries_processed_in_axis_for_querying = 0
                    while int_num_entries_processed_in_axis_for_querying < len_axis :
                        sl = slice( int_num_entries_processed_in_axis_for_querying, min( len_axis, int_num_entries_processed_in_axis_for_querying + int_num_entries_in_a_batch_in_axis_for_querying ) ) # retrieve a slice along the primary axis
                        yield { 'sl' : sl, 'int_num_entries_processed_in_axis_for_querying' : int_num_entries_processed_in_axis_for_querying }
                        int_num_entries_processed_in_axis_for_querying += int_num_entries_in_a_batch_in_axis_for_querying # update the position
                def __process_batch( batch, pipe_result_sender ) :
                    """ # 2022-08-15 20:46:05 
                    process batches containing entries on the primary axis
                    """
                    # parse batch
                    sl, int_num_entries_processed_in_axis_for_querying = batch[ 'sl' ], batch[ 'int_num_entries_processed_in_axis_for_querying' ]
                    
                    # initialize looping through axis not for querying (secondary axis)
                    int_num_entries_processed_in_axis_not_for_querying = 0
                    arr_num_records = np.zeros( sl.stop - sl.start, dtype = np.int64 ) # initialize the list of the number of records for the entries in the current batch
                    while int_num_entries_processed_in_axis_not_for_querying < len_axis_secondary :
                        sl_secondary = slice( int_num_entries_processed_in_axis_not_for_querying, min( len_axis_secondary, int_num_entries_processed_in_axis_not_for_querying + int_num_entries_in_a_subbatch_in_axis_not_for_querying ) ) # retrieve a slice along the secondary axis
                        arr_num_records += ( ( self._za_mtx.get_orthogonal_selection( ( sl, sl_secondary ) ).T if flag_axis_is_barcode else self._za_mtx.get_orthogonal_selection( ( sl_secondary, sl ) ) ) > 0 ).sum( axis = 0 ) # update 'arr_num_records'
                        int_num_entries_processed_in_axis_not_for_querying += int_num_entries_in_a_subbatch_in_axis_not_for_querying # update the position
                    # send the result
                    pipe_result_sender.send( ( sl, arr_num_records ) )
                def __post_process_batch( res ) :
                    """ # 2022-08-15 21:03:59 
                    process result from a batch
                    """
                    sl, arr_num_records = res # parse the result
                    # flush buffer
                    ns[ 'l_buffer' ].append( ( sl, arr_num_records ) )
                    if len( ns[ 'l_buffer' ] ) >= ns[ 'int_size_buffer' ] :
                        ns[ 'pipe_sender' ].send( ns[ 'l_buffer' ] ) # send result to worker
                        ns[ 'l_buffer' ] = [ ] # initialize the buffer
                    
                # process batch by batch
                bk.Multiprocessing_Batch( gen_batch = __gen_batch( ), process_batch = __process_batch, post_process_batch = __post_process_batch, int_num_threads = int_num_threads )
                    
            # flush the buffer
            if len( ns[ 'l_buffer' ] ) > 0 :
                pipe_sender.send( ns[ 'l_buffer' ] ) # send result to worker
            # dismiss the worker
            pipe_sender.send( None )
            p.join( )
    def _save_metadata_( self ) :
        ''' # 2022-07-31 00:40:33 
        a method for saving metadata to the disk 
        '''
        if not self._flag_is_read_only : # update metadata only when the current RamData object is not read-only
            if hasattr( self, '_dict_metadata' ) : # if metadata has been loaded
                # convert 'columns' to list before saving attributes
                self._root.attrs[ 'dict_metadata' ] = self._dict_metadata # update metadata
    @property
    def n_active_entries( self ) :
        ''' # 2022-07-31 00:46:40 
        calculate the number of active entries
        '''
        # calculate the number of active entries if it has not been calculated.
        if not 'n_active_entries' in self._dict_metadata :
            self.ba_active_entries
        return self._dict_metadata[ 'n_active_entries' ] if 'n_active_entries' in self._dict_metadata else self.len_axis_for_querying # if the number of active entries cannot be calculated, return the number of all entries
    def __repr__( self ) :
        return f"<RAMtx object ({self.mode}) for querying '{'features' if self.is_for_querying_features else  'barcodes'}' containing {self._int_num_records} records of {self._int_num_features} features X {self._int_num_barcodes} barcodes\n\tRAMtx path: {self._path_folder_ramtx}>"
    @property
    def ba_filter_features( self ) :
        ''' # 2022-06-23 17:05:55 
        returns 'ba_filter'
        '''
        return self._ba_filter_features
    @ba_filter_features.setter
    def ba_filter_features( self, ba_filter ) :
        """ # 2022-06-23 17:06:36 
        check whether the filter is valid
        """
        if ba_filter is not None :
            assert len( ba_filter ) == self._int_num_features
        self._ba_filter_features = ba_filter
    @property
    def ba_filter_barcodes( self ) :
        ''' # 2022-06-23 17:18:56 
        returns 'ba_filter'
        '''
        return self._ba_filter_barcodes
    @ba_filter_barcodes.setter
    def ba_filter_barcodes( self, ba_filter ) :
        """ # 2022-06-23 17:19:00 
        check whether the filter is valid
        """
        if ba_filter is not None :
            assert len( ba_filter ) == self._int_num_barcodes
        self._ba_filter_barcodes = ba_filter
    @property
    def is_for_querying_features( self ) :
        """ # 2022-07-30 20:37:02 
        """
        return self._is_for_querying_features
    @is_for_querying_features.setter
    def is_for_querying_features( self, flag ) :
        """ # 2022-07-30 20:37:37 
        """
        if self.is_sparse : # if current RAMtx is in sparse format, this property is fixed
            return 
        self._is_for_querying_features = flag # update property
    @property
    def flag_ramtx_sorted_by_id_feature( self ) :
        ''' # 2022-06-23 09:06:51 
        retrieve 'flag_ramtx_sorted_by_id_feature' setting from the RAMtx metadata
        '''
        return self._dict_metadata[ 'flag_ramtx_sorted_by_id_feature' ] if self.is_sparse else None
    @property
    def len_axis_for_querying( self ) :
        ''' # 2022-06-23 09:08:44 
        retrieve number of elements of the indexed axis
        '''
        return self._int_num_features if self.is_for_querying_features else self._int_num_barcodes
    @property
    def len_axis_not_for_querying( self ) :
        ''' # 2022-06-23 09:08:44 
        retrieve number of elements of the not indexed axis
        '''
        return self._int_num_barcodes if self.is_for_querying_features else self._int_num_features
    @property
    def axis_for_querying( self ) :
        """
        # 2022-06-30 21:45:48 
        return 'Axis' object of the indexed axis
        """
        return None if self._ramdata is None else ( self._ramdata.ft if self.is_for_querying_features else self._ramdata.bc )
    @property
    def ba_filter_axis_for_querying( self ) :
        ''' # 2022-06-23 09:08:44 
        retrieve filter of the indexed axis
        '''
        return self.ba_filter_features if self.is_for_querying_features else self.ba_filter_barcodes
    def __contains__( self, x ) -> bool :
        ''' # 2022-06-23 09:13:31 
        check whether an integer representation of indexed entry is available in the index. if filter is active, also check whether the entry is active '''
        return ( 0 <= x < self.len_axis_for_querying ) and ( self.ba_filter_axis_for_querying is None or self.ba_filter_axis_for_querying[ x ] ) # x should be in valid range, and if it is, check whether x is an active element in the filter (if filter has been set)
    def __iter__( self ) :
        ''' # 2022-06-23 09:13:46 
        yield each entry in the index upon iteration. if filter is active, ignore inactive elements '''
        if self.ba_filter_axis_for_querying is None : # if filter is not set, iterate over all elements
            return iter( range( self.len_axis_for_querying ) )
        else : # if filter is active, yield indices of active elements only
            return iter( bk.BA.to_integer_indices( self.ba_filter_axis_for_querying ) )
    def __getitem__( self, l_int_entry ) : 
        """ # 2022-08-16 01:54:36 
        Retrieve data of a given list of entries from RAMtx as lists of values and arrays (i.e. sparse matrix), each value and array contains data of a single 'int_entry' of the indexed axis
        '__getitem__' can be used to retrieve minimal number of values required to build a sparse matrix or dense matrix from it
        
        Returns:
        l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying, l_arr_value :
            'l_int_entry_of_axis_for_querying' only contains int_entry of valid entries
        """
        # retrieve settings
        int_total_number_of_values_in_a_batch_for_dense_matrix = self.int_total_number_of_values_in_a_batch_for_dense_matrix
        
        # initialize the output data structures
        l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying, l_arr_value = [ ], [ ], [ ]
        
        # wrap in a list if a single entry was queried
        if isinstance( l_int_entry, ( int, np.int64, np.int32, np.int16, np.int8 ) ) : # check whether the given entry is an integer
            l_int_entry = [ l_int_entry ]
        ''' retrieve filters '''
        is_for_querying_features = self.is_for_querying_features
        ba_filter_axis_for_querying, ba_filter_not_axis_for_querying = ( self.ba_filter_features, self.ba_filter_barcodes ) if is_for_querying_features else ( self.ba_filter_barcodes, self.ba_filter_features )
            
        ''' filter 'int_entry', if a filter has been set '''
        ''' handle when empty 'l_int_entry' has been given and filter has been set  '''
        if ba_filter_axis_for_querying is not None :
            l_int_entry = bk.BA.to_integer_indices( ba_filter_axis_for_querying ) if len( l_int_entry ) == 0 else list( int_entry for int_entry in l_int_entry if ba_filter_axis_for_querying[ int_entry ] ) # filter 'l_int_entry' or use the entries in the given filter (if no int_entry was given, use all active entries in the filter)
                
        # if no valid entries are available, return an empty result
        if len( l_int_entry ) == 0 :
            return l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying, l_arr_value
            
        ''' sort 'int_entry' so that closely located entries can be retrieved together '''
        # sort indices of entries so that the data access can occur in the same direction across threads
        int_num_entries = len( l_int_entry )
        if int_num_entries > 30 : # use numpy sort function only when there are sufficiently large number of indices of entries to be sorted
            l_int_entry = np.sort( l_int_entry )
        else :
            l_int_entry = sorted( l_int_entry )
        
        """
        retrieve data from RAMtx data structure
        """
        # retrieve flags for dtype conversions
        flag_change_dtype_of_values = self._za_mtx.dtype != self._dtype_of_values
        
        # retrieve dictionaries for changing coordinates
        dict_change_int_entry_of_axis_for_querying, dict_change_int_entry_of_axis_not_for_querying = None, None # initialize the dictionaries
        if self._ramdata is not None : # if RAMtx has been attached to RamData, retrieve dictionaries that can be used to change coordinate
            if self.is_for_querying_features :
                dict_change_int_entry_of_axis_for_querying = self._ramdata.ft.dict_change
                dict_change_int_entry_of_axis_not_for_querying = self._ramdata.bc.dict_change
            else :
                dict_change_int_entry_of_axis_for_querying = self._ramdata.bc.dict_change
                dict_change_int_entry_of_axis_not_for_querying = self._ramdata.ft.dict_change
        # compose a vectorized function for the conversion of int_entries of the non-indexed axis.
        def f( i ) :
            return dict_change_int_entry_of_axis_not_for_querying[ i ]
        vchange_int_entry_of_axis_not_for_querying = np.vectorize( f ) if dict_change_int_entry_of_axis_not_for_querying is not None else None
        
        ''' internal settings '''
        int_num_chunks_for_a_batch = 2
        
        def __retrieve_data( pipe_from_main_thread = None, pipe_to_main_thread = None, flag_as_a_worker = True ) :
            """ # 2022-08-16 01:54:31 
            retrieve data as a worker in a worker process or in the main processs (in single-process mode)
            """
            ''' initialize '''
            # handle inputs
            l_int_entry = pipe_from_main_thread.recv( ) if flag_as_a_worker else pipe_from_main_thread  # receive work if 'flag_as_a_worker' is True or use 'pipe_from_main_thread' as a list of works
            # for each int_entry, retrieve data and collect records
            l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying, l_arr_value = [ ], [ ], [ ]
            
            def __process_entry( int_entry, arr_int_entry_of_axis_not_for_querying, arr_value ) :
                """ # 2022-07-30 22:07:46 
                process retrieve data. apply filter and change coordinates
                """
                ''' if a filter for not-indexed axis has been set, apply the filter to the retrieved records '''
                if ba_filter_not_axis_for_querying is not None :
                    arr_int_entry_of_axis_not_for_querying = arr_int_entry_of_axis_not_for_querying.astype( np.int64 ) # convert to integer type
                    arr_mask = np.zeros( len( arr_int_entry_of_axis_not_for_querying ), dtype = bool ) # initialize the mask for filtering records
                    for i, int_entry_of_axis_not_for_querying in enumerate( arr_int_entry_of_axis_not_for_querying ) : # iterate through each record
                        if ba_filter_not_axis_for_querying[ int_entry_of_axis_not_for_querying ] : # check whether the current int_entry is included in the filter
                            arr_mask[ i ] = True # include the record
                    # if no valid data exists (all data were filtered out), continue to the next 'int_entry'
                    if arr_mask.sum( ) == 0 :
                        return

                    # filter records using the mask
                    arr_int_entry_of_axis_not_for_querying = arr_int_entry_of_axis_not_for_querying[ arr_mask ]
                    arr_value = arr_value[ arr_mask ]

                    # convert int_entry for the non-indexed axis if a mapping has been given
                    if vchange_int_entry_of_axis_not_for_querying is not None :
                        arr_int_entry_of_axis_not_for_querying = vchange_int_entry_of_axis_not_for_querying( arr_int_entry_of_axis_not_for_querying )

                ''' convert dtypes of retrieved data '''
                if flag_change_dtype_of_feature_and_barcode_indices :
                    arr_int_entry_of_axis_not_for_querying = arr_int_entry_of_axis_not_for_querying.astype( self._dtype_of_feature_and_barcode_indices )
                if flag_change_dtype_of_values :
                    arr_value = arr_value.astype( self._dtype_of_values )

                ''' append the retrieved data to the output results '''
                l_int_entry_of_axis_for_querying.append( int_entry if dict_change_int_entry_of_axis_for_querying is None else dict_change_int_entry_of_axis_for_querying[ int_entry ] ) # convert int_entry for the indexed axis if a mapping has been given 
                l_arr_int_entry_of_axis_not_for_querying.append( arr_int_entry_of_axis_not_for_querying )
                l_arr_value.append( arr_value )
            def __fetch_from_sparse_ramtx( l_int_entry_in_a_batch, l_index_in_a_batch ) :
                """ # 2022-07-30 22:32:14 
                fetch data from sparse ramtx for a batch
                """
                arr_index_of_a_batch = np.array( l_index_in_a_batch ) # convert index of the batch to a numpy array
                st_batch, en_batch = arr_index_of_a_batch[ 0, 0 ], arr_index_of_a_batch[ - 1, 1 ] # retrieve start and end positions of the current batch
                arr_int_entry_of_axis_not_for_querying, arr_value = self._za_mtx[ st_batch : en_batch ].T # fetch data from the Zarr object
                
                for int_entry, index in zip( l_int_entry_in_a_batch, arr_index_of_a_batch - st_batch ) : # substract the start position of the batch to retrieve the local index
                    st, en = index
                    sl = slice( st, en )
                    __process_entry( int_entry, arr_int_entry_of_axis_not_for_querying[ sl ], arr_value[ sl ] )
            def __fetch_from_dense_ramtx( l_int_entry_in_a_batch ) :
                """ # 2022-08-16 01:54:21 
                fetch data from dense ramtx for a batch in a memory-efficient manner using subbatches
                """
                # initialize the sparse data container for each entry
                dict_data = dict( )
                
                # retrieve entries for a subbatch
                int_num_entries_in_a_subbatch_in_axis_not_for_querying = max( 1, int( np.floor( int_total_number_of_values_in_a_batch_for_dense_matrix / len( l_int_entry_in_a_batch ) ) ) ) # minimum number of entries in a subbatch is 1
                
                # initialize looping through axis not for querying (secondary axis)
                int_num_entries_processed_in_axis_not_for_querying = 0
                while int_num_entries_processed_in_axis_not_for_querying < self.len_axis_not_for_querying : # iterate through each subbatch
                    sl_secondary = slice( int_num_entries_processed_in_axis_not_for_querying, min( self.len_axis_not_for_querying, int_num_entries_processed_in_axis_not_for_querying + int_num_entries_in_a_subbatch_in_axis_not_for_querying ) ) # retrieve a slice along the secondary axis
                    
                    # iterate through each entry on the axis for querying for the current subbatch
                    for int_entry, arr_data in zip( l_int_entry_in_a_batch, self._za_mtx.get_orthogonal_selection( ( sl_secondary, l_int_entry_in_a_batch ) ).T if is_for_querying_features else self._za_mtx.get_orthogonal_selection( ( l_int_entry_in_a_batch, sl_secondary ) ) ) : # fetch data from the Zarr object for the current subbatch and iterate through each entry and its data
                        arr_int_entry_of_axis_not_for_querying = np.where( arr_data )[ 0 ] # retrieve coordinates of non-zero records
                        if len( arr_int_entry_of_axis_not_for_querying ) == 0 : # if no non-zero records exist, continue to the next entry
                            continue
                        arr_value = arr_data[ arr_int_entry_of_axis_not_for_querying ] # retrieve non-zero records
                        arr_int_entry_of_axis_not_for_querying += int_num_entries_processed_in_axis_not_for_querying # add offset 'int_num_entries_processed_in_axis_not_for_querying' to the coordinates retrieved from the subbatch
                        del arr_data
                        
                        # initialize 'int_entry' for the sparse data container
                        if int_entry not in dict_data :
                            dict_data[ int_entry ] = { 'l_arr_int_entry_of_axis_not_for_querying' : [ ], 'l_arr_value' : [ ] }
                        # add retrieved data from the subbatch to the sparse data container
                        dict_data[ int_entry ][ 'l_arr_int_entry_of_axis_not_for_querying' ].append( arr_int_entry_of_axis_not_for_querying )
                        dict_data[ int_entry ][ 'l_arr_value' ].append( arr_value )
                        del arr_value, arr_int_entry_of_axis_not_for_querying
                    
                    int_num_entries_processed_in_axis_not_for_querying += int_num_entries_in_a_subbatch_in_axis_not_for_querying # update the position

                for int_entry in dict_data : # iterate each entry
                    __process_entry( int_entry, np.concatenate( dict_data[ int_entry ][ 'l_arr_int_entry_of_axis_not_for_querying' ] ), np.concatenate( dict_data[ int_entry ][ 'l_arr_value' ] ) ) # concatenate list of arrays into a single array
                del dict_data
            
            ''' retrieve data '''
            if self.is_sparse : # handle sparse ramtx
                ''' %% Sparse ramtx %% '''
                # prepare
                int_num_records_in_a_chunk = self._za_mtx.chunks[ 0 ] # retrieve the number of records in a chunk
                # retrieve flags for dtype conversions
                flag_change_dtype_mtx_index = self._za_mtx_index.dtype != np.int64
                flag_change_dtype_of_feature_and_barcode_indices = self._za_mtx.dtype != self._dtype_of_feature_and_barcode_indices
                
                # retrieve mtx_index data and remove invalid entries
                arr_index = self._za_mtx_index.get_orthogonal_selection( l_int_entry ) # retrieve mtx_index data 
                if flag_change_dtype_mtx_index : # convert dtype of retrieved mtx_index data
                    arr_index = arr_index.astype( np.int64 )
                
                index_chunk_start_current_batch = None # initialize the index of the chunk at the start of the batch
                l_int_entry_in_a_batch, l_index_in_a_batch = [ ], [ ] # several entries will be processed together as a batch if they reside in the same or nearby chunk ('int_num_chunks_for_a_batch' setting)
                # iterate through each 'int_entry'
                for int_entry, index in zip( l_int_entry, arr_index ) : # iterate through each entry
                    st, en = index
                    if st == en : # if there is no count data for the 'int_entry', continue on to the next 'int_entry' # drop 'int_entry' lacking count data (when start and end index is the same, the 'int_entry' does not contain any data)
                        continue
                    
                    ''' if batch is full, flush the batch '''
                    index_chunk_end = en - 1 // int_num_records_in_a_chunk # retrieve the index of the last chunk
                    if index_chunk_start_current_batch is not None and index_chunk_end >= index_chunk_start_current_batch + int_num_chunks_for_a_batch : # if start has been set 
                        __fetch_from_sparse_ramtx( l_int_entry_in_a_batch, l_index_in_a_batch )
                        l_int_entry_in_a_batch, l_index_in_a_batch = [ ], [ ] # initialize the next batch
                        index_chunk_start_current_batch = None # reset start
                        
                    ''' start the batch '''
                    # if start has not been set, set the start of the current batch
                    if index_chunk_start_current_batch is None : # start the batch
                        index_chunk_start_current_batch = st // int_num_records_in_a_chunk 
                    
                    # add int_entry to the batch 
                    l_int_entry_in_a_batch.append( int_entry ) 
                    l_index_in_a_batch.append( [ st, en ] )
                
                if len( l_int_entry_in_a_batch ) > 0 : # if some entries remains unprocessed, flush the buffer
                    __fetch_from_sparse_ramtx( l_int_entry_in_a_batch, l_index_in_a_batch )
            else :
                ''' %% Dense ramtx %% '''
                # prepare
                int_num_entries_in_a_chunk = self._za_mtx.chunks[ 1 ] if is_for_querying_features else self._za_mtx.chunks[ 0 ] # retrieve the number of entries in a chunk
                flag_change_dtype_of_feature_and_barcode_indices = False # since indices from dense ramtx (return values of np.where) will be in np.int64 format, there will be no need to change dtype of indices
                
                index_chunk_start_current_batch = None # initialize the index of the chunk at the start of the batch
                l_int_entry_in_a_batch = [ ] # several entries will be processed together as a batch if they reside in the same or nearby chunk ('int_num_chunks_for_a_batch' setting)
                # iterate through each 'int_entry'
                for int_entry in l_int_entry : # iterate through each entry
                    ''' if batch is full, flush the batch '''
                    index_chunk = int_entry // int_num_entries_in_a_chunk # retrieve the index of the chunk of the current entry
                    if index_chunk_start_current_batch is not None and index_chunk >= index_chunk_start_current_batch + int_num_chunks_for_a_batch : # if batch has been started
                        __fetch_from_dense_ramtx( l_int_entry_in_a_batch )
                        l_int_entry_in_a_batch = [ ] # initialize the next batch
                        index_chunk_start_current_batch = None # reset start
                        
                    ''' start the batch '''
                    # if start has not been set, set the start of the current batch
                    if index_chunk_start_current_batch is None : # start the batch
                        index_chunk_start_current_batch = index_chunk
                    
                    # add int_entry to the batch 
                    l_int_entry_in_a_batch.append( int_entry ) 
                
                if len( l_int_entry_in_a_batch ) > 0 : # if some entries remains unprocessed, flush the buffer
                    __fetch_from_dense_ramtx( l_int_entry_in_a_batch )

            
            ''' return the retrieved data '''
            # compose a output value
            output = ( l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying, l_arr_value )
            # if 'flag_as_a_worker' is True, send the result or return the result
            if flag_as_a_worker :
                pipe_to_main_thread.send( output ) # send unzipped result back
            else :
                return output
        
        # load data using multiprocessing
        if self.int_num_cpus > 1 and int_num_entries > 1 : # enter multi-processing mode only more than one entry should be retrieved
            # initialize workers
            int_n_workers = min( self.int_num_cpus, int_num_entries ) # one thread for distributing records. Minimum numbers of workers for sorting is 1 # the number of workers should not be larger than the number of entries to retrieve.
            l_l_int_entry_for_each_worker = list( l for l in LIST_Split( l_int_entry, int_n_workers, flag_contiguous_chunk = True ) if len( l ) > 0 ) # retrieve a list of valid work loads for the workers
            int_n_workers = min( int_n_workers, len( l_l_int_entry_for_each_worker ) ) # adjust the number of workers according to the number of distributed workloads
            
            l_pipes_from_main_process_to_worker = list( mp.Pipe( ) for _ in range( int_n_workers ) ) # create pipes for sending records to workers # add process for receivers
            l_pipes_from_worker_to_main_process = list( mp.Pipe( ) for _ in range( int_n_workers ) ) # create pipes for collecting results from workers
            l_processes = list( mp.Process( target = __retrieve_data, args = ( l_pipes_from_main_process_to_worker[ index_worker ][ 1 ], l_pipes_from_worker_to_main_process[ index_worker ][ 0 ] ) ) for index_worker in range( int_n_workers ) ) # add a process for distributing fastq records
            for p in l_processes :
                p.start( )
            # distribute works
            for index_worker, l_int_entry_for_each_worker in enumerate( l_l_int_entry_for_each_worker ) : # distribute works # no load balacing for now
                l_pipes_from_main_process_to_worker[ index_worker ][ 0 ].send( l_int_entry_for_each_worker )
            # wait until all works are completed
            int_num_workers_completed = 0
            l_output = list( [ ] for i in range( int_n_workers ) ) # initialize a list that will collect output results
            while int_num_workers_completed < int_n_workers : # until all works are completed
                for index_worker, cxn in enumerate( l_pipes_from_worker_to_main_process ) :
                    _, pipe_reciver = cxn # parse a connection
                    if pipe_reciver.poll( ) :
                        l_output[ index_worker ] = pipe_reciver.recv( ) # collect output 
                        int_num_workers_completed += 1 # update the number of completed workers
                time.sleep( 0.1 )
            # dismiss workers once all works are completed
            for p in l_processes :
                p.join( )
                
            # ravel retrieved records
            for output in l_output :
                l_int_entry_of_axis_for_querying.extend( output[ 0 ] )
                l_arr_int_entry_of_axis_not_for_querying.extend( output[ 1 ] )
                l_arr_value.extend( output[ 2 ] )
            del output, l_output
        else : # single thread mode
            l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying, l_arr_value = __retrieve_data( l_int_entry, flag_as_a_worker = False )
        
        return l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying, l_arr_value
    def get_sparse_matrix( self, l_int_entry, flag_return_as_arrays = False ) :
        """ # 2022-08-21 15:35:09 
        
        get sparse matrix for the given list of integer representations of the entries.
        
        'l_int_entry' : list of int_entries for query
        'flag_return_as_arrays' : if True, return three arrays and a single list, 'l_int_barcodes', 'l_int_features', 'l_values', 'l_int_num_records'. 
                'l_int_barcodes', 'l_int_features', 'l_values' : for building a sparse matrix
                'l_int_num_records' : for building an index
                if False, return a scipy.csr sparse matrix
        """
        l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying, l_arr_value = self[ l_int_entry ] # parse retrieved result
        
        # combine the arrays
        arr_int_entry_of_axis_not_for_querying = np.concatenate( l_arr_int_entry_of_axis_not_for_querying )
        arr_value = np.concatenate( l_arr_value )
        del l_arr_value # delete intermediate objects
        
        # compose 'arr_int_entry_of_axis_for_querying'
        arr_int_entry_of_axis_for_querying = np.zeros( len( arr_int_entry_of_axis_not_for_querying ), dtype = self._dtype_of_feature_and_barcode_indices ) # create an empty array
        int_pos = 0
        for int_entry_of_axis_for_querying, a in zip( l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying ) :
            n = len( a )
            arr_int_entry_of_axis_for_querying[ int_pos : int_pos + n ] = int_entry_of_axis_for_querying # compose 'arr_int_entry_of_axis_for_querying'
            int_pos += n # update the current position
        if flag_return_as_arrays :
            l_int_num_records = list( len( a ) for a in l_arr_int_entry_of_axis_not_for_querying ) # when returning arrays instead of a scipy sparse matrix, additionally prepare records that can be utilized for build an index of the data
        del l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying # delete intermediate objects
        
        # get 'arr_int_barcode' and 'arr_int_feature' based on 'self.is_for_querying_features'
        if self.is_for_querying_features :
            arr_int_barcode = arr_int_entry_of_axis_not_for_querying
            arr_int_feature = arr_int_entry_of_axis_for_querying
        else :
            arr_int_barcode = arr_int_entry_of_axis_for_querying
            arr_int_feature = arr_int_entry_of_axis_not_for_querying
        del arr_int_entry_of_axis_for_querying, arr_int_entry_of_axis_not_for_querying # delete intermediate objects
        
        if flag_return_as_arrays : # if 'flag_return_as_arrays' is True, return data as arrays
            return arr_int_barcode, arr_int_feature, arr_value, l_int_num_records
        # return data as a sparse matrix
        n_bc, n_ft = ( self._int_num_barcodes, self._int_num_features ) if self._ramdata is None else ( len( self._ramdata.bc ), len( self._ramdata.ft ) ) # detect whether the current RAMtx has been attached to a RamData and retrieve the number of barcodes and features accordingly
        X = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( ( arr_value, ( arr_int_barcode, arr_int_feature ) ), shape = ( n_bc, n_ft ) ) ) # convert count data to a sparse matrix
        return X # return the composed sparse matrix 
    def get_total_num_records( self, ba = None, int_num_entries_for_each_weight_calculation_batch = 1000, flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx = True ) :
        """ # 2022-08-07 01:38:02 
        get total number of records in the current RAMtx for the given entries ('ba' filter).
        this function is mainly for the estimation of the total number of records to process for displaying progress information in the progress bar.
        """
        # set defaule arguments
        if ba is None :
            ba = self.ba_filter_axis_for_querying # if None is given, ba_filter of the currently indexed axis will be used.
            if ba is None : # if filter is not set or the current RAMtx has not been attached to a RamData object, use the active entries
                ba = self.ba_active_entries # if None is given, self.ba_active_entries bitarray will be used.
        # initialize
        # a namespace that can safely shared between function scopes
        ns = { 'int_num_records' : 0, 'l_int_entry_for_weight_calculation_batch' : [ ] }
        
        # check if pre-calculated weights are available
        axis = 'features' if self.is_for_querying_features else 'barcodes' # retrieve axis of current ramtx
        flag_weight_available = False # initialize
        for path_folder in [ self._path_folder_ramtx, self._path_folder_ramtx_modifiable ] :
            if path_folder is not None and zarr_exists( f'{path_folder}matrix.{axis}.number_of_records_for_each_entry.zarr/' ) :
                path_folder_zarr_weight = f"{path_folder}matrix.{axis}.number_of_records_for_each_entry.zarr/" # define an existing zarr object path
                flag_weight_available = True
                za_weight = zarr.open( path_folder_zarr_weight ) # open zarr object containing weights if available
        
        def __update_total_num_records( ) :
            """ # 2022-08-05 00:56:12 
            retrieve indices of the current 'weight_current_batch', calculate weights, and yield a batch
            """
            ''' retrieve weights '''
            if flag_weight_available and ( self.mode != 'dense' or not flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx ) : # if weight is available
                # load weight for the batch
                arr_weight = za_weight.get_orthogonal_selection( ns[ 'l_int_entry_for_weight_calculation_batch' ] ) # retrieve weights
            else : # if weight is not available
                arr_weight = np.full( len( ns[ 'l_int_entry_for_weight_calculation_batch' ] ), self.len_axis_not_for_querying ) # if weight is not available, assumes all records are available (number of entries in non-indexed axis) for each entry
            ''' update total number of records '''
            ns[ 'int_num_records' ] += arr_weight.sum( ) # update total number of records
            ns[ 'l_int_entry_for_weight_calculation_batch' ] = [ ] # empty the weight calculation batch

        for int_entry in bk.BA.find( ba ) : # iterate through active entries of the given bitarray
            ns[ 'l_int_entry_for_weight_calculation_batch' ].append( int_entry ) # collect int_entry for the current 'weight_calculation_batch'
            # once 'weight_calculation' batch is full, process the 'weight_calculation' batch
            if len( ns[ 'l_int_entry_for_weight_calculation_batch' ] ) == int_num_entries_for_each_weight_calculation_batch :
                __update_total_num_records( ) # update total number of records
        __update_total_num_records( )
        return int( ns[ 'int_num_records' ] ) # return the total number of records
    def batch_generator( self, ba = None, int_num_entries_for_each_weight_calculation_batch = 1000, int_total_weight_for_each_batch = 10000000, int_chunk_size_for_checking_boundary = None, flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx = False ) :
        ''' # 2022-08-05 23:37:03 
        generate batches of list of integer indices of the active entries in the given bitarray 'ba'. 
        Each bach has the following characteristics:
            monotonous: active entries in a batch are in an increasing order
            the total number of records of a batch is around (but not exactly) 'int_total_weight_for_each_batch'
        
        'ba' : (default None) if None is given, self.ba_active_entries bitarray will be used.
        'int_chunk_size_for_checking_boundary' : if this argument is given, each batch will respect the chunk boundary of the given chunk size so that different batches share the same 'chunk'. setting this argument will override 'int_total_weight_for_each_batch' argument
        'int_total_weight_for_each_batch' : total number of records in a batch. 
        'flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx' : when iterating through a dense matrix, interpret the length of the axis not for querying as the total number of records for every entry in the axis for querying. This will be more useful for restricting the memory usage when analysing dense RAMtx matrix.
        '''
        # set defaule arguments
        if ba is None :
            ba = self.ba_filter_axis_for_querying # if None is given, ba_filter of the currently indexed axis will be used.
            if ba is None : # if filter is not set or the current RAMtx has not been attached to a RamData object, use the active entries
                ba = self.ba_active_entries # if None is given, self.ba_active_entries bitarray will be used.
        # initialize
        # a namespace that can safely shared between functions
        ns = { 'int_accumulated_weight_current_batch' : 0, 'l_int_entry_current_batch' : [ ], 'l_int_entry_for_weight_calculation_batch' : [ ], 'index_chunk_end' : None, 'index_batch' : 0, 'int_num_of_previously_returned_entries' : 0 } # initialize 'index_batch'
        
        # check if pre-calculated weights are available
        axis = 'features' if self.is_for_querying_features else 'barcodes' # retrieve axis of current ramtx
        flag_weight_available = False # initialize
        for path_folder in [ self._path_folder_ramtx, self._path_folder_ramtx_modifiable ] :
            if path_folder is not None and zarr_exists( f'{path_folder}matrix.{axis}.number_of_records_for_each_entry.zarr/' ) :
                path_folder_zarr_weight = f"{path_folder}matrix.{axis}.number_of_records_for_each_entry.zarr/" # define an existing zarr object path
                flag_weight_available = True
                za_weight = zarr.open( path_folder_zarr_weight ) # open zarr object containing weights if available
        
        def __compose_batch( ) :
            """ # 2022-08-05 23:34:28 
            compose batch from the values available in the namespace 'ns'
            """
            return { 'index_batch' : ns[ 'index_batch' ], 'l_int_entry_current_batch' : ns[ 'l_int_entry_current_batch' ], 'int_num_of_previously_returned_entries' : ns[ 'int_num_of_previously_returned_entries' ], 'int_accumulated_weight_current_batch' : ns[ 'int_accumulated_weight_current_batch' ] }
        def find_batch( ) :
            """ # 2022-08-05 00:56:12 
            retrieve indices of the current 'weight_current_batch', calculate weights, and yield a batch
            """
            ''' retrieve weights '''
            if flag_weight_available and ( self.mode != 'dense' or not flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx ) : # if weight is available and if dense, 'flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx' is False
                # load weight for the batch
                arr_weight = za_weight.get_orthogonal_selection( ns[ 'l_int_entry_for_weight_calculation_batch' ] ) # retrieve weights
            else : # if weight is not available
                arr_weight = np.full( len( ns[ 'l_int_entry_for_weight_calculation_batch' ] ), self.len_axis_not_for_querying ) # if weight is not available, assumes all records are available (number of entries in non-indexed axis) for each entry
            ''' search for batch '''
            for int_entry, weight in zip( ns[ 'l_int_entry_for_weight_calculation_batch' ], arr_weight ) :
                if ns[ 'index_chunk_end' ] is not None and ns[ 'index_chunk_end' ] != int_entry // int_chunk_size_for_checking_boundary : # if the chunk boundary has been set and the boundary has reached
                    yield __compose_batch( ) # return a batch
                    # initialize the next batch
                    ns[ 'index_batch' ] += 1
                    ns[ 'int_num_of_previously_returned_entries' ] += len( ns[ 'l_int_entry_current_batch' ] ) # update the total number of entries returned
                    ns[ 'l_int_entry_current_batch' ] = [ ] 
                    ns[ 'int_accumulated_weight_current_batch' ] = 0
                    ns[ 'index_chunk_end' ] = None
                        
                # update the current batch
                ns[ 'l_int_entry_current_batch' ].append( int_entry )
                ns[ 'int_accumulated_weight_current_batch' ] += weight

                # check whether the current batch is full
                if ns[ 'int_accumulated_weight_current_batch' ] >= int_total_weight_for_each_batch and ns[ 'index_chunk_end' ] is None : # a current batch is full, yield the batch # when chunk boundary has not been set
                    if int_chunk_size_for_checking_boundary is not None : # when chunk boundary checking is active
                        ns[ 'index_chunk_end' ] = int_entry // int_chunk_size_for_checking_boundary # set the chunk boundary
                    else :
                        yield __compose_batch( ) # return a batch
                        # initialize the next batch
                        ns[ 'index_batch' ] += 1
                        ns[ 'int_num_of_previously_returned_entries' ] += len( ns[ 'l_int_entry_current_batch' ] ) # update the total number of entries returned
                        ns[ 'l_int_entry_current_batch' ] = [ ] 
                        ns[ 'int_accumulated_weight_current_batch' ] = 0
                
            # initialize next 'weight_calculation_batch'
            ns[ 'l_int_entry_for_weight_calculation_batch' ] = [ ]
                

        for int_entry in bk.BA.find( ba ) : # iterate through active entries of the given bitarray
            ns[ 'l_int_entry_for_weight_calculation_batch' ].append( int_entry ) # collect int_entry for the current 'weight_calculation_batch'
            # once 'weight_calculation' batch is full, process the 'weight_calculation' batch
            if len( ns[ 'l_int_entry_for_weight_calculation_batch' ] ) == int_num_entries_for_each_weight_calculation_batch :
                for e in find_batch( ) : # generate batch from the 'weight_calculation' batch
                    yield e
        for e in find_batch( ) : # generate batch from the last 'weight_calculation_batch'
            yield e
        # return the remaining int_entries as the last batch (if available)
        if len( ns[ 'l_int_entry_current_batch' ] ) > 0 :
            yield __compose_batch( ) # return a batch
''' a class for representing axis of RamData (barcodes/features) '''
class RamDataAxis( ) :
    """ # 2022-08-03 00:55:04 
    a memory-efficient container of features/barcodes and associated metadata for a given RamData object.
    
    'path_folder' : a folder containing the axis
    'name_axis' : ['barcodes', 'features'] 
    'int_index_str_rep' : a integer index for the column for the string representation of the axis in the string Zarr object (the object storing strings) of the axis
    'mode' : file mode. 'r' for read-only mode and 'a' for mode allowing modifications
    'path_folder_mask' : a local (local file system) path to the mask of the current Axis that allows modifications to be written without modifying the source. if a valid local path to a mask is given, all modifications will be written to the mask
    'flag_is_read_only' : read-only status of RamData
    """
    def __init__( self, path_folder, name_axis, ba_filter = None, ramdata = None, int_index_str_rep = 0, mode = 'a', path_folder_mask = None, flag_is_read_only = False, dict_kw_zdf = { 'flag_retrieve_categorical_data_as_integers' : False, 'flag_load_data_after_adding_new_column' : True, 'flag_enforce_name_col_with_only_valid_characters' : True }, dict_kw_view = { 'float_min_proportion_of_active_entries_in_an_axis_for_using_array' : 0.1, 'dtype' : np.int32 }, verbose = True ) :
        """ # 2022-07-16 17:10:26 
        """
        # set attributes
        self._mode = mode
        self._flag_is_read_only = flag_is_read_only
        self._path_folder_mask = path_folder_mask
        self.verbose = verbose
        self._name_axis = name_axis
        self._path_folder = path_folder
        self.meta = ZarrDataFrame( f"{path_folder}{name_axis}.num_and_cat.zdf", ba_filter = ba_filter, mode = mode, path_folder_mask = None if path_folder_mask is None else f"{path_folder_mask}{name_axis}.num_and_cat.zdf", flag_is_read_only = self._flag_is_read_only, ** dict_kw_zdf ) # open a ZarrDataFrame with a given filter
        self.int_num_entries = self.meta._n_rows_unfiltered # retrieve number of entries
        self.int_index_str_rep = int_index_str_rep # it can be changed later
        # initialize the mapping dictionaries
        self._dict_str_to_i = None 
        self._dict_i_to_str = None 
        self._ramdata = ramdata # initialize RamData reference
        self.filter = ba_filter # set filter
        
        # initialize viewer (coordinate converter, a dictionary for converting coordinates) 
        # set viewer settings
        self._dict_kw_view = dict_kw_view
        self.dict_change = None # initialize view
        self._dict_change_backup = None
    def _convert_to_bitarray( self, ba_filter ) :
        ''' # 2022-08-03 02:21:21 
        handle non-None filter objects and convert these formats to the bitarray filter object
        '''
        assert self.int_num_entries == len( ba_filter )

        ''' handle non-bitarray input types '''
        # handle when a list type has been given (convert it to np.ndarray)
        if isinstance( ba_filter, list ) or ( isinstance( ba_filter, np.ndarray ) and ba_filter.dtype != bool ) : # change to target dtype
            ba_filter = np.array( ba_filter, dtype = bool )
        # handle when a numpy ndarray has been given (convert it to bitarray)
        if isinstance( ba_filter, np.ndarray ) :
            ba_filter = bk.BA.to_bitarray( ba_filter )
        assert isinstance( ba_filter, bitarray ) # check the return is bitarray object
        return ba_filter
    def __iter__( self ) :
        """ # 2022-07-02 22:16:56 
        iterate through valid entries in the axis, according to the filter and whether the string representations are loaded or not. if string representations were loaded, iterate over string representations.
        """
        return ( ( bk.BA.to_integer_indices( self.filter ) if self.filter is not None else np.arange( len( self ) ) ) if self._dict_str_to_i is None else self._dict_str_to_i ).__iter__( )
    def __len__( self ) :
        ''' # 2022-08-21 15:31:48 
        returns the number of entries in the Axis. when view is active, the length after applying the view will be returned. when view is absent, the number of all entries will be returned, regardless of whether a filter is active or not.
        '''
        return self.meta.n_rows if self.is_view_active else self.int_num_entries
    @property
    def is_view_active( self ) :
        """ # 2022-08-21 15:31:44 
        return true if a view is active
        """
        return self.dict_change is not None
    def create_view( self ) :
        """ # 2022-07-16 15:17:17 
        build 'dict_change' (dictionaries for conversion of coordinates) from the given filter, creating a view of the current 'Axis'
        automatically set filter using the mask containing all active entries with valid data if filter is not active
        
        for example, when filter is 
         0123456789  - index
        '1000101110' - filter 
        
        then, dict_change will be { 0 : 0, 4 : 1, 6 : 2, 7 : 3, 8 : 4 }
        when the number of active entries in an exis > 10% (or above any proportion that can set by 'float_min_proportion_of_active_entries_in_an_axis_for_using_array'), an array with the same length will be used for the conversion of coordinates
        
        'float_min_proportion_of_active_entries_in_an_axis_for_using_array' : a threshold for the transition from dictionary to array for the conversion of coordinates. empirically, dictionary of the same length takes about ~10 times more memory than the array
        'dtype' : dtype of array that will be used as 'dictionary'
        """
        # retrieve settings to create a view
        float_min_proportion_of_active_entries_in_an_axis_for_using_array = 0.1 if 'float_min_proportion_of_active_entries_in_an_axis_for_using_array' not in self._dict_kw_view else self._dict_kw_view[ 'float_min_proportion_of_active_entries_in_an_axis_for_using_array' ]
        dtype = np.int32 if 'dtype' not in self._dict_kw_view else self._dict_kw_view[ 'dtype' ]
        
        # automatically set filter using the mask containing all active entries with valid data if filter is not active
        if self.filter is None :
            self.filter = self.all( flag_return_valid_entries_in_the_currently_active_layer = True )
        
        dict_change = None # initialize 'dict_change'
        ba = self.filter
        if ba is not None and ba.count( ) < self.int_num_entries : # only build 'dict_change' if a filter is active or at least one entry is not active
            n = len( ba )
            n_active_entries = ba.count( )
            # initialize dictionary
            dict_change = np.zeros( n, dtype = dtype ) if ( n_active_entries / n ) > float_min_proportion_of_active_entries_in_an_axis_for_using_array else dict( ) # implement a dictionary using an array if the proportion of active entries in the axis is larger than the given threshold to reduce the memory footprint and increase the efficiency of conversion process
            for i, e in enumerate( bk.BA.to_integer_indices( ba ) ) : # iterate through 'int_entry' of the active entries
                dict_change[ e ] = i
        self.dict_change = dict_change # load 'dict_change'
    def destroy_view( self ) :
        """ # 2022-07-16 15:23:01 
        unload 'self.dict_change' (dictionaries for conversion of coordinates), destroying the current view
        """
        self.dict_change = None
    def backup_view( self ) :
        """ # 2022-08-20 17:25:12 
        backup view
        """
        self._dict_change_backup = self.dict_change # back up view
        self.destroy_view( ) # destroy view
    def restore_view( self ) :
        """ # 2022-08-20 17:25:12 
        restore view
        """
        self.dict_change = self._dict_change_backup
        self._dict_change_backup = None
    @property
    def filter( self ) :
        """ # 2022-06-24 22:20:43 
        return a bitarray filter 
        """
        return self._ba_filter
    @filter.setter
    def filter( self, ba_filter ) :
        """ # 2022-07-05 23:12:34 
        set a new bitarray filter on the Axis and the RamData object to which the current axis belongs to.
        
        a given mask will be further masked so that only entries with a valid count data is included in the resulting filter
        
        """
        ''' convert other formats to bitarray if a filter has been given '''
        self.destroy_view( ) # if a filter has been updated, 'dict_change' will be unloaded
        
        if ba_filter is not None :
            ba_filter = self._convert_to_bitarray( ba_filter ) # convert mask to bitarray filter
            ba_filter &= self.all( flag_return_valid_entries_in_the_currently_active_layer = True ) # only use the entries with a valid count data
        
        # propagate the filter
        self.meta.filter = ba_filter # change filter of metadata zdf
        self._ba_filter = ba_filter # set the filter of current axis object
        # set the filter of layer object of the RamData to which the current axis object has been attached.
        if self._ramdata is not None and self._ramdata.layer is not None : # if a layor object has been loaded in the RamData to which the current Axis object belongs to.
            setattr( self._ramdata._layer, f'ba_filter_{self._name_axis}', ba_filter )
    @property
    def ba_active_entries( self ) :
        """ # 2022-07-16 17:38:04 
        
        return a bitarray object containing currently active entries in the Axis. 
        if a filter is active, return the current filter
        if a filter is not active, return the return value of Axis.all( flag_return_valid_entries_in_the_currently_active_layer = True )
        """
        return self.all( flag_return_valid_entries_in_the_currently_active_layer = True ) if self.filter is None else self.filter
    def get_str( self, queries, int_index_col = None ) :
        """ # 2022-08-23 08:53:49 
        get string representations of the queries
        """
        # set default value for 'int_index_col'
        if int_index_col is None :
            int_index_col = self.int_index_str_rep
        # check whether string representation of the entries of the given axis is available 
        path_folder_str_zarr = f"{self._path_folder}{self._name_axis}.str.zarr"
        if not zarr_exists( path_folder_str_zarr ) :  # if the zarr object containing string representations are not available, exits
            return None
        
        # open a zarr object containing the string representation of the entries
        za = zarr.open( path_folder_str_zarr, 'r' )
        return za.get_orthogonal_selection( ( queries, int_index_col ) )
    def load_str( self, int_index_col = None ) : 
        ''' # 2022-06-24 22:38:18 
        load string representation of all the active entries of the current axis, and retrieve a mapping from string representation to integer representation
        
        'int_index_col' : default value is 'self.int_index_str_rep'
        '''
        # set default value for 'int_index_col'
        if int_index_col is None :
            int_index_col = self.int_index_str_rep
        # check whether string representation of the entries of the given axis is available 
        path_folder_str_zarr = f"{self._path_folder}{self._name_axis}.str.zarr"
        if not zarr_exists( path_folder_str_zarr ) :  # if the zarr object containing string representations are not available, exits
            return None
        
        # open a zarr object containing the string representation of the entries
        za = zarr.open( path_folder_str_zarr, 'r' ) 
        
        # retrieve string representations from the Zarr object
        if self.filter is None : # when the filter is inactive
            arr_str = za[ :, int_index_col ]
        else : # when a filter has been applied
            # perform mask selection to retrieve filter-applied string values
            int_num_entries = self.int_num_entries
            assert int_num_entries == za.shape[ 0 ] # make sure the number of rows of the Zarr object is same as number of element in a filter
            arr_filter = bk.BA.to_array( self.filter ) # retrieve mask for a column
            if za.shape[ 1 ] == 1 : # if there is a single column
                arr_mask = arr_filter.reshape( int_num_entries, 1 ) # retrieve mask from filter 
            else : # if there are more than one column, since Zarr mask selection requires the use of mask with same shape, compose such a array
                arr_mask = np.zeros( za.shape, dtype = bool )
                arr_false = np.zeros( int_num_entries, dtype = bool ) # placeholder values to make a mask with the same shape as Zarr object
                for i in range( za.shape[ 1 ] ) :
                    if i == int_index_col :
                        arr_mask[ :, i ] = arr_filter 
                    else :
                        arr_mask[ :, i ] = arr_false 
                del arr_false
            arr_str = za.get_mask_selection( arr_mask )
            del arr_mask, arr_filter
            
        # compose a pair of dictionaries for the conversion
        arr_int_entry = np.arange( len( arr_str ) ) if self.filter is None else bk.BA.to_integer_indices( self.filter ) # retrieve integer representations of the entries
        self._dict_str_to_i = dict( ( e, i ) for e, i in zip( arr_str, arr_int_entry ) ) 
        self._dict_i_to_str = dict( ( i, e ) for e, i in zip( arr_str, arr_int_entry ) ) 
        if self.verbose :
            print( f'[Axis {self._name_axis}] completed loading of {len( arr_str )} number of strings' )
        return arr_str # return loaded strings
    def unload_str( self ) :
        """ # 2022-06-25 09:36:59 
        unload a mapping between string representations and integer representations.
        """
        self._dict_str_to_i = None
        self._dict_i_to_str = None
    @property
    def map_str( self ) :
        """ # 2022-06-25 09:31:32 
        return a dictionary for mapping string representation to integer representation
        """
        return self._dict_str_to_i
    @property
    def map_int( self ) :
        """ # 2022-06-25 09:31:32 
        return a dictionary for mapping integer representation to string representation
        """
        return self._dict_i_to_str
    def __getitem__( self, l ) :
        """ # 2022-07-14 00:42:20 
        a main functionality of 'Axis' class
        translate a given list of entries / slice / mask (bitarray/boolean_array), and return a bitarray mask containing valid entries
        
        inputs:
        [list of entries / slice / mask (bitarray/boolean_array)]
        
        returns:
        [a list of valid integer representation]
        """
        ''' initialize '''
        n = self.int_num_entries # retrieve the number of entries
        # initialize the output object
        # initialize the bitarray for the valid entries
        ba_filter_of_selected_entries = bitarray( n )
        ba_filter_of_selected_entries.setall( 0 ) # an initialized output has no active entries
        
        # retrieve bitarray filter (or a filter of all active entries in the current layer)
        ba_filter = self.filter if self.filter is not None else self.all( flag_return_valid_entries_in_the_currently_active_layer = True )
        
        ''' handle slices '''
        if isinstance( l, slice ) :
            for i in bk.Slice_to_Range( l, n ) :
                if ba_filter[ i ] :
                    ba_filter_of_selected_entries[ i ] = True
            return ba_filter_of_selected_entries 
        
        ''' handle 'None' '''
        if l is None :
            return ba_filter # if None is given, return all active entries in the filter (or all active entries in the layer if a filter has not been set).
        
        ''' handle a single value input '''
        if not hasattr( l, '__iter__' ) or isinstance( l, str ) : # if a given input is not iterable or a string, wrap the element in a list
            l = [ l ]
            
        ''' handle an empty list input '''
        # handle empty inputs
        if len( l ) == 0 :
            return ba_filter_of_selected_entries # return results
        
        ''' handle string list input '''
        if isinstance( l[ 0 ], str ) : # when string representations were given
            flag_unload_str = False # a flag to unload string representations before exiting
            # if str has not been loaded, load the data temporarily
            if self.map_str is None :
                flag_unload_str = True
                self.load_str( )
            
            dict_mapping = self.map_str # retrieve a dictionary for mapping str to int
            for e in l :
                if e in dict_mapping :
                    i = dict_mapping[ e ]
                    if ba_filter[ i ] : # if the entry is acitve in the filter (or filter objec containing all active entries)
                        ba_filter_of_selected_entries[ i ] = True
            
            # unload str data
            if flag_unload_str :
                self.unload_str( )
            return ba_filter_of_selected_entries 
        
        ''' handle mask (bitarray / boolean array) '''
        if len( l ) == n and set( COUNTER( l[ : 10 ] ) ).issubset( { 0, 1, True, False } ) : # detect boolean array
            ba = self._convert_to_bitarray( l ) # convert mask to bitarray
            ba &= ba_filter # apply filter
            return ba
        
        ''' handle integer index list input '''
        for i in l :
            if 0 <= i < n and ba_filter[ i ] :
                ba_filter_of_selected_entries[ i ] = True
        return ba_filter_of_selected_entries
    def save( self, path_folder ) :
        """ # 2022-07-05 00:52:39 
        save data contained in the Axis object (and metadata saved as ZarrDataFrame) to the new path.
        if a filter is active, filtered data will be saved.
        
        'path_folder' : the path of the output Axis object
        """
        # check validity of the path
        path_folder = os.path.abspath( path_folder ) + '/' # retrieve abspath of the output object
        assert self._path_folder != path_folder
        
        # create output folder
        os.makedirs( path_folder, exist_ok = True )
        
        # save number and categorical data
        self.meta.save( f"{path_folder}{self._name_axis}.num_and_cat.zdf" ) # save all columns
        
        # save string data
        # initialize
        za = zarr.open( f"{self._path_folder}{self._name_axis}.str.zarr", mode = 'r', synchronizer = zarr.ThreadSynchronizer( ) ) # open a zarr object containing the string representation of the entries
        za_new = zarr.open( f"{path_folder}{self._name_axis}.str.zarr", mode = 'w', shape = ( self.meta.n_rows, za.shape[ 1 ] ), chunks = za.chunks, dtype = str, synchronizer = zarr.ThreadSynchronizer( ) ) # writing a new zarr object
        # open Base64-encoded data
        newfile_chunked_data = open( f'{path_folder}{self._name_axis}s.str.tsv.gz.base64.concatenated.txt', 'w' ) # base64-encodec concatenated data files
        l_index_chunked_data = [ ] # a list that will contains indices of concatenated chunks 
        
        int_size_buffer = za.chunks[ 0 ] # use the chunk size as the size of the buffer
        ns = dict( ) # namespace that can be safely modified across the scopes of the functions
        ns[ 'int_num_entries_written' ] = 0 # initialize the last position of written entries (after filter applied)
        ns[ 'int_num_bytes_written' ] = 0 # initialize the last position of written entries (after filter applied)
        ns[ 'l_buffer' ] = [ ] # initialize the buffer
        
        def flush_buffer( ) :
            ''' # 2022-07-04 23:34:40 
            transfer string representations of entries to output Zarr object and the base64-encoded chunked data
            '''
            # retrieve data of the entries in the buffer, and empty the buffer
            n = len( ns[ 'l_buffer' ] ) # retrieve number of entries in the buffer
            data = za.get_orthogonal_selection( ns[ 'l_buffer' ] ) # retrieve data from the Zarr object
            ns[ 'l_buffer' ] = [ ] # empty the buffer
            
            # write Zarr object
            za_new[ ns[ 'int_num_entries_written' ] : ns[ 'int_num_entries_written' ] + n, : ] = data # transfer data to the new Zarr object
            ns[ 'int_num_entries_written' ] += n # update the number of entries written
            
            # write a chunk for a file containing concatenated chunks (for javascript access)
            str_content = ' ' + _base64_encode( _gzip_bytes( ( '\n'.join( '\t'.join( row ) for row in data.T ) + '\n' ).encode( ) ) ) # retrieved base64-encoded, padded string
            int_num_bytes = len( str_content ) # retrieve number of bytes of the current 
            newfile_chunked_data.write( str_content )
            l_index_chunked_data.append( [ ns[ 'int_num_bytes_written' ], ns[ 'int_num_bytes_written' ] + int_num_bytes ] )
            ns[ 'int_num_bytes_written' ] += int_num_bytes # update the number of bytes written for the concatenated chunks
            
        # process entries using a buffer
        for i in range( len( self ) ) if self.filter is None else bk.BA.find( self.filter, val = 1 ) : # iteratre through active integer representations of the entries
            ns[ 'l_buffer' ].append( i )
            if len( ns[ 'l_buffer' ] ) >= int_size_buffer : # flush the buffer if it is full
                flush_buffer( )
        if len( ns[ 'l_buffer' ] ) >= 0 : # empty the buffer
            flush_buffer( )
        newfile_chunked_data.close( ) # close file
        
        # write index of the file containing concatenated chunks
        df_index = pd.DataFrame( l_index_chunked_data, columns = [ 'index_byte_start', 'index_byte_end' ] )
        df_index[ 'index_chunk' ] = np.arange( len( df_index ) ) # retrieve 'index_chunk'
        with io.BytesIO( ) as file :
            df_index[ [ 'index_chunk', 'index_byte_start', 'index_byte_end' ] ].T.to_csv( file, sep = '\t', index = True, header = False )
            file.seek( 0 )
            with open( f'{path_folder}{self._name_axis}s.str.index.tsv.gz.base64.txt', 'w' ) as newfile :
                newfile.write( _base64_encode( _gzip_bytes( file.read( ) ) ) )
    def __repr__( self ) :
        """ # 2022-07-20 23:12:47 
        """
        return f"<Axis '{self._name_axis}' containing {'' if self.filter is None else f'{self.meta.n_rows}/'}{self.meta._n_rows_unfiltered} entries available at {self._path_folder}\n\tavailable metadata columns are {sorted( self.meta.columns )}>"
    def all( self, flag_return_valid_entries_in_the_currently_active_layer = True ) :
        """ # 2022-06-27 21:41:38  
        return bitarray filter with all entries marked 'active'
        
        'flag_return_valid_entries_in_the_currently_active_layer' : return bitarray filter containing only the active entries in the current layer 
        """
        rtx = self._ramdata.layer.get_ramtx( flag_is_for_querying_features = self._name_axis == 'features' ) # retrieve associated ramtx object
        if flag_return_valid_entries_in_the_currently_active_layer and self._ramdata.layer is not None and rtx is not None : # if RamData has an active layer and 'flag_return_valid_entries_in_the_currently_active_layer' setting is True, return bitarray where entries with valid count data is marked as '1' # if valid ramtx data is available
            ba = rtx.ba_active_entries
        else :
            # if layer is empty or 'flag_return_valid_entries_in_the_currently_active_layer' is False, just return a bitarray filled with '1'
            ba = bitarray( self.int_num_entries )
            ba.setall( 1 ) # set all entries as 'active' 
        return ba # return the bitarray filter
    def AND( self, * l_filter ) :
        """ # 2022-06-27 21:37:31 
        perform AND operations for the given list of filters (bitarray/np.ndarray objects)
        """
        if len( l_filter ) == 0 :
            return self.all( )
        ba_result = self._convert_to_bitarray( l_filter[ 0 ] )
        for ba in l_filter[ 1 : ] :
            ba_result &= self._convert_to_bitarray( ba ) # perform AND operation
        return ba_result # return resulting filter
    def OR( self, * l_filter ) :
        """ # 2022-06-28 20:16:42 
        perform OR operations for the given list of filters (bitarray/np.ndarray objects)
        """
        if len( l_filter ) == 0 : # if no inputs are given, return bitarray filter for all entries
            return self.all( )
        ba_result = self._convert_to_bitarray( l_filter[ 0 ] )
        for ba in l_filter[ 1 : ] :
            ba_result |= self._convert_to_bitarray( ba ) # perform OR operation
        return ba_result # return resulting filter
    def NOT( self, filter = None ) :
        """ # 2022-06-28 20:19:34 
        reverse (not operation) the bitarray filter
        if no 'filter' is given, return empty bitarray
        """
        if filter is not None :
            ba = ~ self._convert_to_bitarray( filter ) # perform 'not' operation
        else :
            ba = bitarray( self.int_num_entries )
            ba.setall( 1 ) # set all entries as 'active' 
        return ba
    def XOR( self, filter_1, filter_2 ) :
        """ # 2022-06-28 20:24:20 
        perform XOR operation between 'filter_1' and 'filter_2'
        """
        return self._convert_to_bitarray( filter_1 ) ^ self._convert_to_bitarray( filter_2 )
    def batch_generator( self, ba = None, int_num_entries_for_batch = 1000, flag_mix_randomly = False ) :
        ''' # 2022-07-16 22:57:23 
        generate batches of list of integer indices of the active entries in the given bitarray 'ba'. 
        Each bach has the following characteristics:
            monotonous: active entries in a batch are in an increasing order
            same size: except for the last batch, each batch has the same number of active entries 'int_num_entries_for_batch'.
        This function is simialr to RAMtx.batch_generator, except that the number of records for each entries ('weights') will not be considered when constructing a batch
        
        'ba' : (default None) if None is given, self.filter bitarray will be used.
        'flag_mix_randomly' : generate batches of entries after mixing randomly 
        '''
        # set defaule arguments
        # set default filter
        if ba is None :
            ba = self.ba_active_entries # iterate through an active entries
            
        # initialize
        # a namespace that can safely shared between functions
        ns = { 'index_batch' : 0, 'l_int_entry_current_batch' : [ ], 'int_num_of_previously_returned_entries' : 0 }
        
        def __compose_batch( ) :
            """ # 2022-08-05 23:34:28 
            compose batch from the values available in the namespace 'ns'
            """
            return { 'index_batch' : ns[ 'index_batch' ], 'l_int_entry_current_batch' : ns[ 'l_int_entry_current_batch' ], 'int_num_of_previously_returned_entries' : ns[ 'int_num_of_previously_returned_entries' ] }

        
        if flag_mix_randomly : # randomly select barcodes across the 
            int_num_active_entries = ba.count( ) # retrieve the total number of active entries 
            float_ratio_batch_size_to_total_size = int_num_entries_for_batch / int_num_active_entries # retrieve approximate number of batches to generate
            # initialize
            int_num_entries_added = 0
            ba_remaining = ba.copy( ) # create a copy of the bitarray of active entries to mark the remaining entries
            float_prob_selection = int_num_entries_for_batch / max( 1, int_num_active_entries - int_num_of_previously_returned_entries ) # calculate the initial probability for selection of entries
            while int_num_entries_added < int_num_active_entries : # repeat entry selection process until all entries are selected
                for int_entry in bk.BA.find( ba_remaining ) : # iterate through remaining active entries
                    if np.random.random( ) < float_prob_selection : # randomly make a decision whether to include the current entry or not
                        ns[ 'l_int_entry_current_batch' ].append( int_entry ) # collect 'int_entry' of the selected entry
                        ba_remaining[ int_entry ] = False # remove the entry from the 'ba_remaining' bitarray
                        int_num_entries_added += 1
                    # once the batch is full, yield the batch
                    if len( ns[ 'l_int_entry_current_batch' ] ) >= int_num_entries_for_batch :
                        ns[ 'l_int_entry_current_batch' ] = np.sort( ns[ 'l_int_entry_current_batch' ] ) # sort the list of int_entries
                        yield __compose_batch( ) # return batch
                        float_prob_selection = int_num_entries_for_batch / max( 1, int_num_active_entries - int_num_of_previously_returned_entries ) # update the probability for selection of an entry
                        ns[ 'int_num_of_previously_returned_entries' ] += len( ns[ 'l_int_entry_current_batch' ] ) # update the number of returned entries
                        ns[ 'l_int_entry_current_batch' ] = [ ] # initialize the next batch
                        ns[ 'index_batch' ] += 1
            # return the remaining int_entries as the last batch (if available)
            if len( ns[ 'l_int_entry_current_batch' ] ) > 0 :
                ns[ 'l_int_entry_current_batch' ] = np.sort( ns[ 'l_int_entry_current_batch' ] ) # sort the list of int_entries
                yield __compose_batch( ) # return batch
        else : # return barcodes in a batch sequentially
            for int_entry in bk.BA.find( ba ) : # iterate through active entries of the given bitarray
                ns[ 'l_int_entry_current_batch' ].append( int_entry ) # collect int_entry for the current batch
                # once the batch is full, yield the batch
                if len( ns[ 'l_int_entry_current_batch' ] ) >= int_num_entries_for_batch :
                    yield __compose_batch( ) # return batch
                    ns[ 'int_num_of_previously_returned_entries' ] += len( ns[ 'l_int_entry_current_batch' ] ) # update the number of returned entries
                    ns[ 'l_int_entry_current_batch' ] = [ ] # initialize the next batch
                    ns[ 'index_batch' ] += 1
            # return the remaining int_entries as the last batch (if available)
            if len( ns[ 'l_int_entry_current_batch' ] ) > 0 :
                yield __compose_batch( ) # return batch
    def change_filter( self, name_col_filter ) :
        """ # 2022-07-16 17:17:29 
        change filter using the filter saved in the metadata with 'name_col_filter' column name. if 'name_col_filter' is not available, current filter setting will not be changed.
        
        'name_col_filter' : name of the column of the metadata ZarrDataFrame containing the filter
        """
        if name_col_filter in self.meta : # if a given column name exists in the current metadata ZarrDataFrame
            self.filter = self.meta[ name_col_filter, : ] # retrieve filter from the storage and apply the filter to the axis
    def save_filter( self, name_col_filter ) :
        """ # 2022-08-06 22:37:49 
        save current filter using the filter to the metadata with 'name_col_filter' column name. if a filter is not active, the metadata will not be updated.
        
        'name_col_filter' : name of the column of the metadata ZarrDataFrame that will contain the filter
        """
        if name_col_filter is not None : # if a given filter name is valid
            self.meta[ name_col_filter, : ] = bk.BA.to_array( self.ba_active_entries ) # save filter to the storage # when a filter is not active, save filter of all active entries of the RAMtx
    def change_or_save_filter( self, name_col_filter ) :
        """ # 2022-08-06 22:36:48 
        change filter to 'name_col_filter' if 'name_col_filter' exists in the metadata, or save the currently active entries (filter) to the metadata using the name 'name_col_filter'
        
        'name_col_filter' : name of the column of the metadata ZarrDataFrame that will contain the filter
        """
        if name_col_filter is not None : # if valid 'name_col_filter' has been given
            if name_col_filter in self.meta : 
                self.change_filter( name_col_filter ) # change filter to 'name_col_filter' if 'name_col_filter' exists in the metadata
            else :
                self.save_filter( name_col_filter ) # save the currently active entries (filter) to the metadata using the name 'name_col_filter'
    def subsample( self, float_prop_subsampling = 1 ) :
        """ # 2022-07-16 17:12:19 
        subsample active entries in the current filter (or all the active entries with valid data) using the proportion of subsampling ratio 'float_prop_subsampling'
        """
        # retrieve bitarray of active entries
        ba_active_entries = self.ba_active_entries
        
        # return the bitarray of all active entries if no subsampling is required
        if float_prop_subsampling is None or float_prop_subsampling == 1 :
            return ba_active_entries
        
        # initialize the output bitarray filter that will contain subsampled entries
        ba_subsampled = bitarray( self.int_num_entries )
        ba_subsampled.setall( 0 ) # exclude all entries by default
        
        # perform subsampling
        for int_entry in bk.BA.find( ba_active_entries ) :
            if np.random.random( ) < float_prop_subsampling :
                ba_subsampled[ int_entry ] = True
        
        # return subsampled entries
        return ba_subsampled
''' a class for representing a layer of RamData '''
class RamDataLayer( ) :
    """ # 2022-07-31 14:39:50 
    A class for interactions with a pair of RAMtx objects of a count matrix. 
    
    'path_folder_ramdata' : location of RamData
    'int_num_cpus' : number of CPUs for RAMtx objects
    'mode' : file mode. 'r' for read-only mode and 'a' for mode allowing modifications
    'flag_is_read_only' : read-only status of RamData
    'path_folder_ramdata_mask' : a local (local file system) path to the mask of the RamData object that allows modifications to be written without modifying the source. if a valid local path to a mask is given, all modifications will be written to the mask
    """
    def __init__( self, path_folder_ramdata, name_layer, ramdata = None, dtype_of_feature_and_barcode_indices = np.int32, dtype_of_values = np.float64, int_num_cpus = 1, verbose = False, mode = 'a', path_folder_ramdata_mask = None, flag_is_read_only = False ) :
        """ # 2022-07-31 14:33:46 
        """
        # set attributes
        self._path_folder_ramdata = path_folder_ramdata
        self._name_layer = name_layer
        self._path_folder_ramdata_layer = f"{path_folder_ramdata}{name_layer}/"
        self._ramdata = ramdata
        self._mode = mode
        self.verbose = verbose
        self._int_num_cpus = int_num_cpus
        self._path_folder_ramdata_mask = path_folder_ramdata_mask
        if path_folder_ramdata_mask is not None : # set path to the mask of the layer if ramdata mask has been given
            self._path_folder_ramdata_layer_mask = f"{self._path_folder_ramdata_mask}{name_layer}/"
        self._flag_is_read_only = flag_is_read_only
        self._dtype_of_values = dtype_of_values
        self._dtype_of_feature_and_barcode_indices = dtype_of_feature_and_barcode_indices
        
        # read metadata
        self._root = zarr.open( self._path_folder_ramdata_layer, 'a' )
        self._dict_metadata = self._root.attrs[ 'dict_metadata' ] # retrieve the metadata 
        self._dict_metadata[ 'set_modes' ] = set( self._dict_metadata[ 'set_modes' ] ) # convert modes to set
        
        # retrieve filters from the axes
        ba_filter_features = ramdata.ft.filter if ramdata is not None else None
        ba_filter_barcodes = ramdata.bc.filter if ramdata is not None else None
        
        # set filters of the current layer
        self.ba_filter_features = ba_filter_features
        self.ba_filter_barcodes = ba_filter_barcodes
        
        # load ramtx
        self._load_ramtx_objects( )
    def _load_ramtx_objects( self ) :
        """ # 2022-08-01 10:57:28 
        load all ramtx present in the layer
        """
        # load RAMtx objects without filters
        # define arguments for opening RAMtx objects
        dict_kwargs = {
            'ramdata' : self._ramdata, 
            'dtype_of_feature_and_barcode_indices' : self._dtype_of_feature_and_barcode_indices, 
            'dtype_of_values' : self._dtype_of_values, 
            'int_num_cpus' : self._int_num_cpus, 
            'verbose' : self.verbose, 
            'flag_debugging' : False, 
            'mode' : self._mode, 
            'path_folder_ramtx_mask' : f'{self._path_folder_ramdata_layer_mask}{mode}/' if self._mask_available else None, 
            'flag_is_read_only' : self._flag_is_read_only
        }
        # load ramtx
        for mode in self.modes : # iterate through each mode
            if not hasattr( self, f"ramtx_{mode}" ) : # if the ramtx object of the current mode has not been load
                if 'dense_for_querying_' in mode :
                    rtx = RAMtx( f'{self._path_folder_ramdata_layer}dense/', is_for_querying_features = mode.rsplit( 'dense_for_querying_', 1 )[ 1 ] == 'features', ** dict_kwargs ) # open dense ramtx in querying_features/querying_barcodes modes
                else :
                    rtx = RAMtx( f'{self._path_folder_ramdata_layer}{mode}/', ** dict_kwargs )
                setattr( self, f"ramtx_{mode}", rtx ) # set ramtx as an attribute
                
        # set filters of the loaded RAMtx objects
        self.ba_filter_features = self.ba_filter_features
        self.ba_filter_barcodes = self.ba_filter_barcodes
    def __repr__( self ) :
        """ # 2022-07-31 01:03:21 
        """
        return f"<RamDataLayer object '{self.name}' containing {self.modes} RAMtx objects\n\tRamDataLayer path: {self._path_folder_ramdata_layer}>"
    @property
    def name( self ) :
        """ # 2022-07-31 01:10:00 
        return the name of the layer
        """
        return self._name_layer
    def _save_metadata_( self ) :
        ''' # 2022-07-20 10:31:39 
        save metadata of the current ZarrDataFrame
        '''
        if not self._flag_is_read_only : # save metadata only when it is not in the read-only mode
            # convert to list before saving attributes
            temp = self._dict_metadata[ 'set_modes' ]
            self._dict_metadata[ 'set_modes' ] = list( temp )
            self._root.attrs[ 'dict_metadata' ] = self._dict_metadata # update metadata
            self._dict_metadata[ 'set_modes' ] = temp # revert to set
    @property
    def _mask_available( self ) :
        """ # 2022-07-30 18:38:30 
        """
        return self._path_folder_ramdata_mask is not None
    @property
    def modes( self ) :
        """ # 2022-07-30 18:29:54 
        return a subst of {'dense' or 'sparse_for_querying_barcodes', 'sparse_for_querying_features'}
        """
        return self._dict_metadata[ 'set_modes' ]
    @property
    def int_num_cpus( self ) :
        """ # 2022-07-21 23:22:24 
        """
        return self._int_num_cpus
    @int_num_cpus.setter
    def int_num_cpus( self, val ) :
        """ # 2022-07-21 23:22:24 
        """
        self._int_num_cpus = max( 1, int( val ) ) # set integer values
        for rtx in self : # iterate through ramtxs
            rtx.int_num_cpus = self._int_num_cpus # update 'int_num_cpus' attributes of RAMtxs
    @property
    def int_num_features( self ) :
        """ # 2022-06-28 21:39:20 
        return the number of features
        """
        return self[ list( self.modes )[ 0 ] ]._int_num_features # return an attribute of the first ramtx of the current layer
    @property
    def int_num_barcodes( self ) :
        """ # 2022-06-28 21:39:20 
        return the number of features
        """
        return self[ list( self.modes )[ 0 ] ]._int_num_barcodes # return an attribute of the first ramtx of the current layer
    @property
    def int_num_records( self ) :
        """ # 2022-06-28 21:39:20 
        return the number of features
        """
        return self[ list( self.modes )[ 0 ] ]._int_num_records # return an attribute of the first ramtx of the current layer
    @property
    def ba_filter_features( self ) :
        """ # 2022-06-26 01:31:24 
        """
        return self._ba_filter_features
    @ba_filter_features.setter
    def ba_filter_features( self, ba_filter ) :
        """ # 2022-06-26 01:31:24 
        """
        # set/update the filter for the associated RAMtx objects
        self._ba_filter_features = ba_filter
        for rtx in self :
            rtx.ba_filter_features = ba_filter
    @property
    def ba_filter_barcodes( self ) :
        """ # 2022-06-26 01:31:24 
        """
        return self._ba_filter_barcodes
    @ba_filter_barcodes.setter
    def ba_filter_barcodes( self, ba_filter ) :
        """ # 2022-06-26 01:31:24 
        """
        # set/update the filter for the associated RAMtx objects
        self._ba_filter_barcodes = ba_filter
        for rtx in self :
            rtx.ba_filter_barcodes = ba_filter
    def __contains__( self, x ) :
        """ # 2022-07-30 18:41:51 
        check whether mode 'x' is available in the layer
        """
        return x in self.modes
    def __iter__( self ) :
        """ # 2022-07-30 18:42:50 
        iterate through ramtx of the modes available in the layer
        """
        return iter( list( self[ mode ] for mode in self.modes if self[ mode ] ) ) # return ramtx object that has been loaded in the current layer
    def select_ramtx( self, ba_entry_bc, ba_entry_ft ) :
        """ # 2022-07-31 11:46:33 
        select appropriate ramtx based on the queryed barcode and features, given as a bitarray filters 'ba_entry_bc', 'ba_entry_ft'
        """
        # count the number of valid queried entries
        int_num_entries_queried_bc = ba_entry_bc.count( )
        int_num_entries_queried_ft = ba_entry_ft.count( )
        
        # detect and handle the cases when one of the axes is empty
        if int_num_entries_queried_bc == 0 or int_num_entries_queried_ft == 0 :
            if self.verbose :
                print( f"Warning: currently queried view is (barcode x features) {int_num_entries_queried_bc} x {int_num_entries_queried_ft}. please change the filter or queries in order to retrieve a valid data" )

        # choose which ramtx object to use
        flag_use_ramtx_for_querying_feature = int_num_entries_queried_bc >= int_num_entries_queried_ft # select which axis to use. if there is more number of barcodes than features, use ramtx for querying 'features'
        
        rtx = self.get_ramtx( flag_is_for_querying_features = flag_use_ramtx_for_querying_feature ) # retrieve ramtx
        if rtx is None :
            return self[ list( self.modes )[ 0 ] ] # return any ramtx as a fallback
        return rtx
    def get_ramtx( self, flag_is_for_querying_features = True, flag_prefer_dense = False ) :
        """ # 2022-07-31 11:55:06 
        retrieve ramtx for querying feature/barcodes
        """
        mode_dense = f"dense_for_querying_{'features' if flag_is_for_querying_features else 'barcodes'}" # retrieve mode name for dense ramtx based on 'flag_is_for_querying_features'
        mode_sparse = f"sparse_for_querying_{'features' if flag_is_for_querying_features else 'barcodes'}" # retrieve mode name for sparse ramtx based on 'flag_is_for_querying_features'
        for mode in [ mode_dense, mode_sparse ] if flag_prefer_dense else [ mode_sparse, mode_dense ] : # search ramtx considering 'flag_prefer_dense'
            if mode in self :
                return self[ mode ]
        if self.verbose :
            print( f"ramtx for querying {'features' if flag_is_for_querying_features else 'barcodes'} efficiently is not available for layer {self.name}, containing the following modes: {self.modes}" )
        return None
    def __getitem__( self, mode ) :
        """ # 2022-07-30 18:44:49 
        """
        if mode in self : # if a given mode is available
            if hasattr( self, f"ramtx_{mode}" ) : # if a given mode has been loaded
                return getattr( self, f"ramtx_{mode}" ) # return the ramtx of the given mode
    def __delitem__( self, mode ) :
        """ # 2022-08-24 20:31:28 
        """
        # ignore if current mode is 'read-only'
        if self._mode == 'r' :
            return
        if mode in self : # if a given mode is available
            if hasattr( self, f"ramtx_{mode}" ) : # if a given mode has been loaded
                # delete from memory
                if 'dense' in mode : # handle 'dense' mode
                    for mode_to_delete in [ 'dense_for_querying_features', 'dense_for_querying_barcodes', 'dense' ] :
                        delattr( self, f"ramtx_{mode_to_delete}" )
                        self._dict_metadata[ 'set_modes' ].remove( mode_to_delete )
                    mode = 'dense'
                else :
                    delattr( self, f"ramtx_{mode}" ) # return the ramtx of the given mode
                    self._dict_metadata[ 'set_modes' ].remove( mode )
                self._save_metadata_( ) # update metadata
                
                # delete a RAMtx
                shutil.rmtree( f'{self._path_folder_ramdata_layer_mask}{mode}/' )
''' class for storing RamData '''
class RamData( ) :
    """ # 2022-08-21 00:43:30 
    This class provides frameworks for single-cell transcriptomic/genomic data analysis, utilizing RAMtx data structures, which is backed by Zarr persistant arrays.
    Extreme lazy loading strategies used by this class allows efficient parallelization of analysis of single cell data with minimal memory footprint, loading only essential data required for analysis. 
    
    'path_folder_ramdata' : a local folder directory or a remote location (https://, s3://, etc.) containing RamData object
    'int_num_cpus' : number of CPUs (processes) to use to distribute works.
    'int_num_cpus_for_fetching_data' : number of CPUs (processes) for individual RAMtx object for retrieving data from the data source. 
    'int_index_str_rep_for_barcodes', 'int_index_str_rep_for_features' : a integer index for the column for the string representation of 'barcodes'/'features' in the string Zarr object (the object storing strings) of 'barcodes'/'features'
    'dict_kw_zdf' : settings for 'Axis' metadata ZarrDataFrame
    'dict_kw_view' : settings for 'Axis' object for creating a view based on the active filter.
    'mode' : file mode. 'r' for read-only mode and 'a' for mode allowing modifications
    'path_folder_ramdata_mask' : the LOCAL file system path where the modifications of the RamData ('MASK') object will be saved and retrieved. If this attribute has been set, the given RamData in the the given 'path_folder_ramdata' will be used as READ-ONLY. For example, when RamData resides in the HTTP server, data is often read-only (data can be only fetched from the server, and not the other way around). However, by giving a local path through this argument, the read-only RamData object can be analyzed as if the RamData object can be modified. This is possible since all the modifications made on the input RamData will be instead written to the local RamData object 'mask' and data will be fetced from the local copy before checking the availability in the remote RamData object.
    
    === batch generation ===
    'int_num_entries_for_each_weight_calculation_batch' : the number of entries in a small batch for generating load-balanced batches.
    'int_total_weight_for_each_batch' : the argument controlling total number of records to be processed in each batch (and thus for each process for parallelized works). it determines memory usage/operation efficiency. higher weight will lead to more memory usage but more operation efficiency, and vice versa
    'flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx' : when iterating through a dense matrix, interpret the length of the axis not for querying as the total number of records for every entry in the axis for querying. This will be more useful for restricting the memory usage when analysing dense RAMtx matrix.

    ==== AnnDataContainer ====
    'flag_enforce_name_adata_with_only_valid_characters' : enforce valid characters in the name of AnnData
    """
    def __init__( self, path_folder_ramdata, name_layer = 'raw', int_num_cpus = 64, int_num_cpus_for_fetching_data = 1, dtype_of_feature_and_barcode_indices = np.int32, dtype_of_values = np.float64, int_index_str_rep_for_barcodes = 0, int_index_str_rep_for_features = 1, int_num_entries_for_each_weight_calculation_batch = 2000, int_total_weight_for_each_batch = 10000000, flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx = False, mode = 'a', path_folder_ramdata_mask = None, dict_kw_zdf = { 'flag_retrieve_categorical_data_as_integers' : False, 'flag_load_data_after_adding_new_column' : True, 'flag_enforce_name_col_with_only_valid_characters' : True }, dict_kw_view = { 'float_min_proportion_of_active_entries_in_an_axis_for_using_array' : 0.1, 'dtype' : np.int32 }, flag_enforce_name_adata_with_only_valid_characters = True, verbose = True, flag_debugging = False ) :
        """ # 2022-07-21 23:32:54 
        """
        ''' hard-coded settings  '''
        # define a set of picklable models :
        self._set_type_model_picklable = { 'ipca', 'hdbscan', 'knn_classifier', 'knngraph' }
        
        ''' modifiable settings '''
        # handle input object paths
        if path_folder_ramdata[ - 1 ] != '/' : # add '/' at the end of path to indicate it is a directory
            path_folder_ramdata += '/'
        if '://' not in path_folder_ramdata : # do not call 'os.path.abspath' on remote path
            path_folder_ramdata = os.path.abspath( path_folder_ramdata ) + '/' # retrieve abs path
        if path_folder_ramdata_mask is not None : # if 'path_folder_ramdata_mask' is given, assumes it is a local path
            path_folder_ramdata_mask = os.path.abspath( path_folder_ramdata_mask ) + '/' # retrieve abs path
            
        # set attributes
        self._mode = mode
        self._path_folder_ramdata = path_folder_ramdata
        self._path_folder_ramdata_mask = path_folder_ramdata_mask
        self.verbose = verbose
        self.flag_debugging = flag_debugging
        self.int_num_cpus = int_num_cpus
        self._int_num_cpus_for_fetching_data = int_num_cpus_for_fetching_data
        self._dtype_of_feature_and_barcode_indices = dtype_of_feature_and_barcode_indices
        self._dtype_of_values = dtype_of_values
        # batch-generation associated settings, which can be changed later
        self.int_num_entries_for_each_weight_calculation_batch = int_num_entries_for_each_weight_calculation_batch
        self.int_total_weight_for_each_batch = int_total_weight_for_each_batch
        self.flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx = flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx
        
        ''' check read-only status of the given RamData '''
        try :
            zarr.open( f'{self._path_folder_ramdata}modification.test.zarr/', 'w' )
            self._flag_is_read_only = False
        except :
            # if test zarr data cannot be written to the source, consider the given RamData object as read-only
            self._flag_is_read_only = True # indicates current RamData object is read-only (however, Mask can be give, and RamData can be modified by modifying the mask)
            if self._path_folder_ramdata_mask is None : # if mask is not given, automatically change the mode to 'r'
                self._mode = 'r' # indicate current RamData cannot be modified
                if self.verbose :
                    print( 'The current RamData object cannot be modified yet no mask location is given. Therefore, the current RamData object will be "read-only"' )
        
        # initialize axis objects
        self.bc = RamDataAxis( path_folder_ramdata, 'barcodes', ba_filter = None, ramdata = self, dict_kw_zdf = dict_kw_zdf, dict_kw_view = dict_kw_view, int_index_str_rep = int_index_str_rep_for_barcodes, verbose = verbose, mode = self._mode, path_folder_mask = self._path_folder_ramdata_mask, flag_is_read_only = self._flag_is_read_only )
        self.ft = RamDataAxis( path_folder_ramdata, 'features', ba_filter = None, ramdata = self, dict_kw_zdf = dict_kw_zdf, dict_kw_view = dict_kw_view, int_index_str_rep = int_index_str_rep_for_features, verbose = verbose, mode = self._mode, path_folder_mask = self._path_folder_ramdata_mask, flag_is_read_only = self._flag_is_read_only )
        
        # initialize the layor object
        self.layer = name_layer
        
        # initialize databases
        if self._path_folder_ramdata_local is not None : # retrieve ramdata object in the local file system, and if the object is available, load/initialize anndatacontainer and shelvecontainer in the local file system
            # set AnnDataContainer attribute for containing various AnnData objects associated with the current RamData
            self.ad = AnnDataContainer( path_prefix_default = self._path_folder_ramdata_local, flag_enforce_name_adata_with_only_valid_characters = flag_enforce_name_adata_with_only_valid_characters, ** GLOB_Retrive_Strings_in_Wildcards( f'{self._path_folder_ramdata_local}*.h5ad' ).set_index( 'wildcard_0' ).path.to_dict( ), mode = self._mode, path_prefix_default_mask = self._path_folder_ramdata_mask, flag_is_read_only = self._flag_is_read_only ) # load locations of AnnData objects stored in the RamData directory

            # open a shelve-based persistent dictionary to save/retrieve arbitrary picklable python objects associated with the current RamData in a memory-efficient manner
            self.ns = ShelveContainer( f"{self._path_folder_ramdata_local}ns", mode = self._mode, path_prefix_shelve_mask = f"{self._path_folder_ramdata_mask}ns", flag_is_read_only = self._flag_is_read_only )
        else : # initialize anndatacontainer and shelvecontainer in the memory using a dicitonary (a fallback)
            self.ad = dict( )
            self.ns = dict( )
            
        # 🔴🔴🔴 TEMP 🔴🔴🔴
        self.models # add model attributes
    @property
    def metadata( self ) :
        """ # 2022-07-21 00:45:12 
        implement lazy-loading of metadata
        """
        # implement lazy-loading of metadata
        if not hasattr( self, '_root' ) :
            # open RamData as a Zarr object (group)
            self._root = zarr.open( self._path_folder_ramdata ) 
            
            if self._path_folder_ramdata_mask is not None : # if mask is given, open the mask object as a zarr group to save/retrieve metadata
                root_mask = zarr.open( self._path_folder_ramdata_mask ) # open the mask object as a zarr group
                if len( list( root_mask.attrs ) ) == 0 : # if mask object does not have a ramdata attribute
                    root_mask.attrs[ 'dict_metadata' ] = self._root.attrs[ 'dict_metadata' ] # copy the ramdata attribute of the current RamData to the mask object
                self._root = root_mask # use the mask object zarr group to save/retrieve ramdata metadata
                
            # retrieve metadata 
            self._dict_metadata = self._root.attrs[ 'dict_metadata' ]
            self._dict_metadata[ 'layers' ] = set( self._dict_metadata[ 'layers' ] )
        # return metadata
        return self._dict_metadata
    def _save_metadata_( self ) :
        ''' # 2022-07-21 00:45:03 
        a semi-private method for saving metadata to the disk 
        '''
        if not self._mode == 'r' : # update metadata only when the current RamData object is not read-only
            if hasattr( self, '_dict_metadata' ) : # if metadata has been loaded
                # convert 'columns' to list before saving attributes
                temp = self._dict_metadata[ 'layers' ] # save the set as a temporary variable 
                self._dict_metadata[ 'layers' ] = list( temp ) # convert to list
                self._root.attrs[ 'dict_metadata' ] = self._dict_metadata # update metadata
                self._dict_metadata[ 'layers' ] = temp # revert to set
    @property
    def layers( self ) :
        ''' # 2022-06-24 00:14:45 
        return a set of available layers
        '''
        return self.metadata[ 'layers' ]
    def __contains__( self, x ) -> bool :
        ''' # 2022-06-24 00:15:04 
        check whether an 'name_layer' is available in the current RamData '''
        return x in self.layers
    def __iter__( self ) :
        ''' # 2022-06-24 00:15:19 
        yield each 'name_layer' upon iteration '''
        return iter( self.layers )
    @property
    def int_num_cpus_for_fetching_data( self ) :
        """ # 2022-07-21 23:32:24 
        """
        return self._int_num_cpus_for_fetching_data
    @int_num_cpus_for_fetching_data.setter
    def int_num_cpus_for_fetching_data( self, val ) :
        """ # 2022-07-21 23:32:35 
        """
        self._int_num_cpus_for_fetching_data = max( 1, int( val ) ) # set an integer value
        if self.layer is not None :
            self.layer.int_num_cpus = self._int_num_cpus_for_fetching_data # update 'int_num_cpus' attributes of RAMtxs
    @property
    def layer( self ) :
        """ # 2022-06-24 00:16:56 
        retrieve the name of layer from the layer object if it has been loaded.
        """
        return self._layer if hasattr( self, '_layer' ) else None # if no layer is set, return None
    @layer.setter
    def layer( self, name_layer ) :
        """ # 2022-07-20 23:29:23 
        change layer to the layer 'name_layer'
        """
        # if None is given as name_layer, remove the current layer from the memory
        if name_layer is None :
            # unload layer if None is given
            if hasattr( self, '_layer' ) :
                delattr( self, '_layer' )
        else :
            # check 'name_layer' is valid
            if name_layer not in self.layers :
                raise KeyError( f"'{name_layer}' data does not exists in the current RamData" )

            if self.layer is None or name_layer != self.layer.name : # if no layer has been loaded or new layer name has been given, load the new layer
                self._layer = RamDataLayer( self._path_folder_ramdata, name_layer, ramdata = self, dtype_of_feature_and_barcode_indices = self._dtype_of_feature_and_barcode_indices, dtype_of_values = self._dtype_of_values, int_num_cpus = self._int_num_cpus_for_fetching_data, verbose = self.verbose, mode = self._mode, path_folder_ramdata_mask = self._path_folder_ramdata_mask, flag_is_read_only = self._flag_is_read_only )

                if self.verbose :
                    print( f"'{name_layer}' layer has been loaded" )
    def delete_layer( self, * l_name_layer ) :
        """ # 2022-08-24 19:25:25 
        delete a given list of layers from the current RamData
        """
        # ignore if current mode is read-only
        if self._mode == 'r' :
            return
        for name_layer in l_name_layer : # for each name_layer
            # ignore invalid 'name_layer'
            if name_layer not in self.layers :
                continue
            # delete an entire layer
            shutil.rmtree( f"{self._path_folder_ramdata}{name_layer}/" )
            
            # remove the current layer from the metadata
            self.layers.remove( name_layer ) 
            self._save_metadata_( )
    def __repr__( self ) :
        """ # 2022-07-20 00:38:24 
        display RamData
        """
        return f"<{'' if not self._mode == 'r' else '(read-only) '}RamData object ({'' if self.bc.filter is None else f'{self.bc.meta.n_rows}/'}{self.metadata[ 'int_num_barcodes' ]} barcodes X {'' if self.ft.filter is None else f'{self.ft.meta.n_rows}/'}{self.metadata[ 'int_num_features' ]} features" + ( '' if self.layer is None else f", {self.layer.int_num_records} records in the currently active layer '{self.layer.name}'" ) + f") stored at {self._path_folder_ramdata}{'' if self._path_folder_ramdata_mask is None else f' with local mask available at {self._path_folder_ramdata_mask}'}\n\twith the following layers : {self.layers}\n\t\tcurrent layer is '{self.layer.name}'>" # show the number of records of the current layer if available.
    def _repr_html_( self ) :
        """ # 2022-08-07 23:56:55 
        display RamData in an interactive environment
        """
        f"<{'' if not self._mode == 'r' else '(read-only) '}RamData object ({'' if self.bc.filter is None else f'{self.bc.meta.n_rows}/'}{self.metadata[ 'int_num_barcodes' ]} barcodes X {'' if self.ft.filter is None else f'{self.ft.meta.n_rows}/'}{self.metadata[ 'int_num_features' ]} features" + ( '' if self.layer is None else f", {self.layer.int_num_records} records in the currently active layer '{self.layer.name}'" ) + f") stored at {self._path_folder_ramdata}{'' if self._path_folder_ramdata_mask is None else f' with local mask available at {self._path_folder_ramdata_mask}'}\n\twith the following layers : {self.layers}\n\t\tcurrent layer is '{self.layer.name}'>" 
        dict_data = {
            'barcodes' : {
                'filter' : self.bc.filter is not None,
                'number_of_entries' : self.bc.meta._n_rows_unfiltered,
                'number_of_entries_after_applying_filter' : self.bc.meta.n_rows,
                'metadata' : {
                    'columns' : list( self.bc.meta.columns ),
                    'settings' : {
                        'path_folder_zdf' : self.bc.meta._path_folder_zdf,
                        'path_folder_mask' : self.bc.meta._path_folder_mask,
                        'flag_use_mask_for_caching' : self.bc.meta.flag_use_mask_for_caching,
                        'flag_retrieve_categorical_data_as_integers' : self.bc.meta._flag_retrieve_categorical_data_as_integers,
                        'flag_load_data_after_adding_new_column' : self.bc.meta._flag_load_data_after_adding_new_column,
                        'int_num_rows_in_a_chunk' : self.bc.meta.int_num_rows_in_a_chunk
                    },
                }
            },
            'features' : {
                'filter' : self.ft.filter is not None,
                'number_of_entries' : self.ft.meta._n_rows_unfiltered,
                'number_of_entries_after_applying_filter' : self.ft.meta.n_rows,
                'metadata' : {
                    'columns' : list( self.ft.meta.columns ),
                    'settings' : {
                        'path_folder_zdf' : self.ft.meta._path_folder_zdf,
                        'path_folder_mask' : self.ft.meta._path_folder_mask,
                        'flag_use_mask_for_caching' : self.ft.meta.flag_use_mask_for_caching,
                        'flag_retrieve_categorical_data_as_integers' : self.ft.meta._flag_retrieve_categorical_data_as_integers,
                        'flag_load_data_after_adding_new_column' : self.ft.meta._flag_load_data_after_adding_new_column,
                        'int_num_rows_in_a_chunk' : self.ft.meta.int_num_rows_in_a_chunk
                    },
                }
            },
            'currently_active_layer' : None if self.layer is None else {
                'name' : self.layer.name,
                'modes' : list( self.layer.modes ),
                'total_number_of_records' : self.layer.int_num_records,
                'settings' : {
                    'int_num_cpus_for_fetching_data' : self.layer.int_num_cpus,
                },
            },
            'layers' : list( self.layers ),
            'models' : self.models,
            'settings' : {
                'read_only' : self._mode == 'r',
                'path_folder_ramdata' : self._path_folder_ramdata,
                'path_folder_ramdata_mask' : self._path_folder_ramdata_mask,
                'verbose' : self.verbose,
                'debugging' : self.flag_debugging,
                'int_num_cpus' : self.int_num_cpus,
                'int_num_cpus_for_fetching_data' : self._int_num_cpus_for_fetching_data,
                'int_num_entries_for_each_weight_calculation_batch' : self.int_num_entries_for_each_weight_calculation_batch,
                'int_total_weight_for_each_batch' : self.int_total_weight_for_each_batch,
                'flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx' : self.flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx,
            }
        }
        return html_from_dict( dict_data = dict_data, name_dict = f'<h2>RamData</h2><div><tt>{self.__repr__( )[ 1 : -1 ]}</tt></div>' )
    def create_view( self ) :
        """  # 2022-07-06 21:17:56 
        create view of the RamData using the current filter settings (load dictionaries for coordinate conversion for filtered barcodes/features)
        """
        self.ft.create_view( )
        self.bc.create_view( )
    def destroy_view( self ) :
        """  # 2022-07-05 22:55:22 
        unload dictionaries for coordinate conversion for filtered barcodes/features, destroying the current view
        """
        self.ft.destroy_view( )
        self.bc.destroy_view( )
    def __enter__( self ) :
        """ # 2022-07-16 15:53:13 
        creating a view of RamData using the current filter settings
        """
        self.create_view( ) # create view
        return self
    def __exit__( self, exc_type, exc_val, exc_tb ) :
        """ # 2022-07-16 15:54:59 
        destroy the current view of RamData
        """
        self.destroy_view( ) # destroy view
    def compose_filters( self, l_entry_bc = [ ], l_entry_ft = [ ], flag_use_str_repr_bc = False, flag_use_str_repr_ft = False ) :
        """ # 2022-07-16 17:10:07 
        for the given 'barcodes'/'features' entries, compose filters containing the entries, and apply the filters.
        
        === inputs ===
        'flag_use_str_repr_bc' = False, 'flag_use_str_repr_ft' = False : flags indicating whether to use string representation of the retrieved entries later
        
        === outputs === 
        bitarray mask of mapped entries and list of string representations (if available. if string representations were not used to retrieve entries, None will be returned for the list object).
        """
        # retrieve flags indicating that the string representations are not loaded
        flag_str_not_loaded_bc = self.bc.map_int is None
        flag_str_not_loaded_ft = self.ft.map_int is None
        
        ''' retrieve filters and string representations for the queried entries '''
        ''' barcode '''
        # load str representation data
        if flag_use_str_repr_bc and flag_str_not_loaded_bc :
            self.bc.load_str( )
        # retrieve filter for the queried entries
        ba_entry_bc = self.bc[ l_entry_bc ]
        # retrieve str representations of the queried entries
        l_str_bc = None
        if flag_use_str_repr_bc :
            dict_map = self.bc.map_int
            l_str_bc = list( dict_map[ i ] for i in bk.BA.to_integer_indices( ba_entry_bc ) )
            del dict_map
        if flag_str_not_loaded_bc : # if 'str' data was not loaded, unload the str data once all necessary data has been retrieved
            self.bc.unload_str( )
        
        ''' feature '''
        # load str representation data
        if flag_use_str_repr_ft and flag_str_not_loaded_ft :
            self.ft.load_str( )
        # retrieve filter for the queried entries
        ba_entry_ft = self.ft[ l_entry_ft ]
        # retrieve str representations of the queried entries
        l_str_ft = None
        if flag_use_str_repr_ft :
            dict_map = self.ft.map_int
            l_str_ft = list( dict_map[ i ] for i in bk.BA.to_integer_indices( ba_entry_ft ) )
            del dict_map
        if flag_str_not_loaded_ft : # if 'str' data was not loaded, unload the str data once all necessary data has been retrieved
            self.ft.unload_str( )
        
        return ba_entry_bc, l_str_bc, ba_entry_ft, l_str_ft # return composed filters and mapped string representations (if available)
    def __getitem__( self, args ) :
        ''' # 2022-08-05 17:18:47 
        please include 'str' in 'barcode_column' and 'feature_column' in order to use string representations in the output AnnData object
        
        possible usages:
        
        [ name_layer, barcode_index, barcode_column, feature_index, feature_column ]
        [ barcode_index, barcode_column, feature_index, feature_column ]
        'barcode_column' and 'feature_column' can include multi-dimensional data 
        for example, 
            [ 'str', { 'X_pca' : slice( 0, 10 ), 'X_umap', : None } ] as 'barcode_column' will include X_umap and X_pca in obsm in the resulting anndata object
            [ 'str', { 'X_pca' : slice( 0, 10 ), { 'X_umap' } ] as 'barcode_column' will also include X_umap and X_pca in obsm in the resulting anndata object
        '''
        assert isinstance( args, tuple ) # more than one arguments should be given
        # if the first argument appears to be 'name_layer', load the layer and drop the argument
        if isinstance( args[ 0 ], str ) and args[ 0 ] in self.layers :
            self.layer = args[ 0 ]
            args = args[ 1 : ] 
        assert len( args ) <= 4 # assumes layer has been loaded, and only arguments are for barcode/feature indexing
        
        args = list( args ) + list( [ ] for i in range( 4 - len( args ) ) ) # make the number of arguments to 4
        l_entry_bc, l_col_bc, l_entry_ft, l_col_ft = args # parse arguments
        
        # backup the filters
        ba_filter_bc_backup = self.bc.filter
        ba_filter_ft_backup = self.ft.filter
        
        # retrieve flags for using string representations in the output
        flag_use_str_repr_bc = 'str' in l_col_bc
        flag_use_str_repr_ft = 'str' in l_col_ft
        
        # compose filters from the queried entries
        ba_entry_bc, l_str_bc, ba_entry_ft, l_str_ft = self.compose_filters( l_entry_bc = l_entry_bc, l_entry_ft = l_entry_ft, flag_use_str_repr_bc = flag_use_str_repr_bc, flag_use_str_repr_ft = flag_use_str_repr_ft )
        
        # retrieve ramtx for retrieving data
        rtx = self.layer.select_ramtx( ba_entry_bc, ba_entry_ft )
        
        # set barcode/feature filters for the queried entries
        self.bc.filter = ba_entry_bc
        self.ft.filter = ba_entry_ft

        # initialize and destroy the view after retrieving the count matrix
        with self as view : # load 'dict_change' for coordinate conversion according to the given filters, creating the view of the RamData
            # retrieve count data
            X = rtx.get_sparse_matrix( [ ] ) # retrieve count data for all entries currently active in the filter
        
        # retrieve meta data as dataframes
        df_obs = self.bc.meta.get_df( * l_col_bc )
        if flag_use_str_repr_bc : # add string representations
            df_obs.index = l_str_bc
            del l_str_bc
        df_var = self.ft.meta.get_df( * l_col_ft )
        if flag_use_str_repr_ft : # add string representations
            df_var.index = l_str_ft
            del l_str_ft
        
        # build output AnnData object
        adata = anndata.AnnData( obs = df_obs, var = df_var ) # in anndata.X, row = barcode, column = feature # set obs and var with integer index values
        adata.X = X # add count data
        del X
        # add obsm/varm
        for ax, name_attr, l in zip( [ self.ft, self.bc ], [ 'varm', 'obsm' ], [ l_col_ft, l_col_bc ] ) :
            for e in l :
                if isinstance( e, set ) : # retrieve all data in the secondary axis
                    for name_col in e :
                        getattr( adata, name_attr )[ name_col ] = ax.meta[ name_col ]
                elif isinstance( e, dict ) : # indexing through secondary axis
                    for name_col in e :
                        getattr( adata, name_attr )[ name_col ] = ax.meta[ name_col ] if e[ name_col ] is None else ax.meta[ name_col, None, e[ name_col ] ] # if e[ name_col ] is None, load all data on the secondary axis
        

        # restore the filters once the data retrieval has been completed
        self.bc.filter = ba_filter_bc_backup
        self.ft.filter = ba_filter_ft_backup
        
        return adata # return resulting AnnData
    def save( self, * l_name_adata ) :
        ''' wrapper of AnnDataContainer.save '''
        self.ad.update( * l_name_adata )
    @property
    def _path_folder_ramdata_modifiable( self ) :
        """ # 2022-07-21 00:07:23 
        return path of the ramdata that is modifiable based on the current RamData settings.
        if mask is given, path to the mask will be returned.
        if current ramdata location cannot be modified and no mask has been given, None will be returned.
        """
        if self._path_folder_ramdata_mask is not None : # if mask is given, use the mask
            return self._path_folder_ramdata_mask
        elif not self._flag_is_read_only : # if current object can be modified, create temp folder inside the current object
            return self._path_folder_ramdata
        else :
            return None
    @property
    def _path_folder_ramdata_local( self ) :
        """ # 2022-07-21 02:08:55 
        return path of the ramdata that is in the local file system based on the current RamData settings.
        if mask is given, path to the mask will be returned, since mask is assumed to be present in the local file system.
        """
        if self._path_folder_ramdata_mask is not None : # if mask is given, use the mask, since mask is assumed to be present in the local file system.
            return self._path_folder_ramdata_mask
        elif '://' not in self._path_folder_ramdata : # if current object appears to be in the local file system, use the current object
            return self._path_folder_ramdata
        else :
            return None
    def create_temp_folder( self ) :
        """ # 2022-07-20 23:36:34 
        create temporary folder and return the path to the temporary folder
        """
        path_folder = self._path_folder_ramdata_modifiable # retrieve path to the modifiable ramdata object
        if path_folder is not None : # if valid location is available (assumes it is a local directory)
            path_folder_temp = f"{path_folder}temp_{UUID( )}/"
            os.makedirs( path_folder_temp, exist_ok = True ) # create the temporary folder
            return path_folder_temp # return the path to the temporary folder
    def summarize( self, name_layer : str, axis : Union[ int, str ], summarizing_func, l_name_col_summarized = None ) :
        ''' # 2022-08-22 11:30:01 
        this function summarize entries of the given axis (0 = barcode, 1 = feature) using the given function
        
        example usage: calculate total sum, standard deviation, pathway enrichment score calculation, etc.
        
        =========
        inputs 
        =========
        'name_layer' : name of the data in the given RamData object to summarize
        'axis': int or str. 
               0, 'bc', 'barcode' or 'barcodes' for applying a given summarizing function for barcodes
               1, 'ft', 'feature' or 'features' for applying a given summarizing function for features
        'summarizing_func' : function object. a function that takes a RAMtx output and return a dictionary containing 'name_of_summarized_data' as key and 'value_of_summarized_data' as value. the resulting summarized outputs will be added as metadata of the given Axis (self.bc.meta or self.ft.meta)
        
                    summarizing_func( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) -> dictionary containing 'key' as summarized metric name and 'value' as a summarized value for the entry. if None is returned, the summary result for the entry will be skipped.
                    
                    a list of pre-defined functions are the followings :
                    'sum' :
                            calculate the total sum (and mean) for each entry
                            useful for initial barcode filtering
                            
                            returns: 'sum', 'mean'
                    
                    'sum_and_dev' :
                            calculate the total sum (and mean) and the total deviation (and variance) for each entry 
                            deviation = sum( ( X - X_mean ) ** 2 )
                            variance = sum( ( X - X_mean ) ** 2 ) / ( total_num_entry - 1 )
                            useful for identifying informative features
                            
                            returns: 'sum', 'mean', 'deviation', 'variance'
                            
                    'count_min_max' :
                            calculate the min and max values, and count the number of active barcodes/features (that were not indexed) for the entry of the indexed axis
                            
                            returns: 'count', 'max', 'min'
                            
        'l_name_col_summarized' : list of column names returned by 'summarizing_func'. by default (when None is given), the list of column names will be inferred by observing output when giving zero values as inputs
        
        =========
        outputs 
        =========
        the summarized metrics will be added to appropriate dataframe attribute of the AnnData of the current RamData (self.adata.obs for axis = 0 and self.adata.var for axis = 1).
        the column names will be constructed as the following :
            f"{name_layer}_{key}"
        if the column name already exist in the dataframe, the values of the columns will be overwritten
        '''
        """
        1) Prepare
        """
        # check the validility of the input arguments
        if name_layer not in self.layers :
            if self.verbose :
                print( f"[ERROR] [RamData.summarize] invalid argument 'name_layer' : '{name_layer}' does not exist." )
            return -1 
        if axis not in { 0, 'barcode', 1, 'feature', 'barcodes', 'features', 'bc', 'ft' } :
            if self.verbose :
                print( f"[ERROR] [RamData.summarize] invalid argument 'axis' : '{name_layer}' is invalid." )
            return -1 
        # set layer
        self.layer = name_layer
        # handle inputs
        flag_summarizing_barcode = axis in { 0, 'barcode', 'barcodes', 'bc' } # retrieve a flag indicating whether the data is summarized for each barcode or not
        
        # retrieve the total number of entries in the axis that was not indexed (if calculating average expression of feature across barcodes, divide expression with # of barcodes, and vice versa.)
        int_total_num_entries_not_indexed = self.ft.meta.n_rows if flag_summarizing_barcode else self.bc.meta.n_rows 

        int_num_threads = self.int_num_cpus # set the number of threads
        if summarizing_func == 'sum' :
            def summarizing_func( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) :
                ''' # 2022-08-01 21:05:06 
                calculate sum of the values of the current entry
                
                assumes 'int_num_records' > 0
                '''
                int_num_records = len( arr_value ) # retrieve the number of records of the current entry
                dict_summary = { 'sum' : np.sum( arr_value ) if int_num_records > 30 else sum( arr_value ), 'num_nonzero_values' : int_num_records } # if an input array has more than 30 elements, use np.sum to calculate the sum
                dict_summary[ 'mean' ] = dict_summary[ 'sum' ] / int_total_num_entries_not_indexed # calculate the mean
                return dict_summary
        elif summarizing_func == 'sum_and_dev' :
            def summarizing_func( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) :
                ''' # 2022-08-01 21:05:02 
                calculate sum and deviation of the values of the current entry
                
                assumes 'int_num_records' > 0
                '''
                int_num_records = len( arr_value ) # retrieve the number of records of the current entry
                dict_summary = { 'sum' : np.sum( arr_value ) if int_num_records > 30 else sum( arr_value ), 'num_nonzero_values' : int_num_records } # if an input array has more than 30 elements, use np.sum to calculate the sum
                dict_summary[ 'mean' ] = dict_summary[ 'sum' ] / int_total_num_entries_not_indexed # calculate the mean
                arr_dev = ( arr_value - dict_summary[ 'mean' ] ) ** 2 # calculate the deviation
                dict_summary[ 'deviation' ] = np.sum( arr_dev ) if int_num_records > 30 else sum( arr_dev )
                dict_summary[ 'variance' ] = dict_summary[ 'deviation' ] / ( int_total_num_entries_not_indexed - 1 ) if int_total_num_entries_not_indexed > 1 else np.nan
                return dict_summary
        elif summarizing_func == 'count_min_max' :
            int_min_num_records_for_numpy = 30
            def summarizing_func( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) :
                ''' # 2022-07-27 15:29:07 
                calculate sum and deviation of the values of the current entry
                
                assumes 'int_num_records' > 0
                '''
                int_num_records = len( arr_value ) # retrieve the number of records of the current entry
                dict_summary = { 'count' : int_num_records, 'max' : np.max( arr_value ) if int_num_records > int_min_num_records_for_numpy else max( arr_value ), 'min' : np.min( arr_value ) if int_num_records > int_min_num_records_for_numpy else min( arr_value ) } # if an input array has more than 'int_min_num_records_for_numpy' elements, use numpy to calculate min/max values
                return dict_summary
        elif not hasattr( summarizing_func, '__call__' ) : # if 'summarizing_func' is not a function, report error message and exit
            if self.verbose :
                print( f"given summarizing_func is not a function, exiting" )
            return -1
        # infer 'l_name_col_summarized'
        if l_name_col_summarized is None :
            # retrieve the list of key values returned by 'summarizing_func' by applying dummy values
            arr_dummy_one, arr_dummy_zero = np.ones( 10, dtype = int ), np.zeros( 10, dtype = int )
            l_name_col_summarized = list( summarizing_func( self, 0, arr_dummy_zero, arr_dummy_one ) )
        l_name_col_summarized = sorted( l_name_col_summarized ) # retrieve the list of key values of an dict_res result returned by 'summarizing_func'
        l_name_col_summarized_with_name_layer_prefix = list( f"{name_layer}_{e}" for e in l_name_col_summarized ) # retrieve the name_col containing summarized data with f'{name_layer}_' prefix 
        
        
        # retrieve RAMtx object to summarize
        rtx = self.layer.get_ramtx( not flag_summarizing_barcode )
        if rtx is None :
            if self.verbose :
                print( f'it appears that the current layer {self.layer.name} appears to be empty, exiting' )
            return
        # retrieve Axis object to summarize 
        ax = self.bc if flag_summarizing_barcode else self.ft
        
        # create a temporary folder
        path_folder_temp = self.create_temp_folder( )
        # handle the case when the temporary folder is not available
        if path_folder_temp is None :
            if self.verbose :
                print( 'failed to create a temporary folder, exiting' )
            return - 1
        
        # define functions for multiprocessing step
        def process_batch( batch, pipe_to_main_process ) :
            ''' # 2022-05-08 13:19:07 
            summarize a given list of entries, and write summarized result as a tsv file, and return the path to the output file
            '''
            int_num_processed_records, l_int_entry_current_batch = batch[ 'int_accumulated_weight_current_batch' ], batch[ 'l_int_entry_current_batch' ] # parse batch
            
            # retrieve the number of index_entries
            int_num_entries_in_a_batch = len( l_int_entry_current_batch )
            
            if int_num_entries_in_a_batch == 0 :
                print( 'empty batch detected' )
            
            # iterate through the data of each entry
            dict_data = dict( ( name_col, [ ] ) for name_col in l_name_col_summarized ) # collect results
            l_int_entry_of_axis_for_querying = [ ] # collect list of queried entries with valid results
            for int_entry_of_axis_for_querying, arr_int_entry_of_axis_not_for_querying, arr_value in zip( * rtx[ l_int_entry_current_batch ] ) : # retrieve data for the current batch
                # retrieve summary for the entry
                dict_res = summarizing_func( self, int_entry_of_axis_for_querying, arr_int_entry_of_axis_not_for_querying, arr_value ) # summarize the data for the entry
                # if the result empty, does not collect the result
                if dict_res is None :
                    continue
                # collect the result
                # collect the int_entry with a valid result
                l_int_entry_of_axis_for_querying.append( int_entry_of_axis_for_querying )
                # collect the result
                for name_col in l_name_col_summarized :
                    dict_data[ name_col ].append( dict_res[ name_col ] if name_col in dict_res else np.nan )
            pipe_to_main_process.send( ( int_num_processed_records, l_int_entry_of_axis_for_querying, dict_data ) ) # send information about the output file
        # initialize the progress bar
        pbar = progress_bar( total = rtx.get_total_num_records( int_num_entries_for_each_weight_calculation_batch = self.int_num_entries_for_each_weight_calculation_batch, flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx = self.flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx ) )
        def post_process_batch( res ) :
            """ # 2022-07-06 03:21:49 
            """
            int_num_processed_records, l_int_entry_of_axis_for_querying, dict_data = res # parse result
            # exit if no result has been collected
            if len( l_int_entry_of_axis_for_querying ) == 0 :
                return
            
            pbar.update( int_num_processed_records ) # update the progress bar

            for name_col, name_col_with_prefix in zip( l_name_col_summarized, l_name_col_summarized_with_name_layer_prefix ) : 
                ax.meta[ name_col_with_prefix, l_int_entry_of_axis_for_querying ] = dict_data[ name_col ]
        # summarize the RAMtx using multiple processes
        bk.Multiprocessing_Batch( rtx.batch_generator( ax.filter, int_num_entries_for_each_weight_calculation_batch = self.int_num_entries_for_each_weight_calculation_batch, int_total_weight_for_each_batch = self.int_total_weight_for_each_batch, flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx = self.flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx ), process_batch, post_process_batch = post_process_batch, int_num_threads = int_num_threads, int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop = 0.2 )
        pbar.close( ) # close the progress bar
        
        # remove temp folder
        shutil.rmtree( path_folder_temp )
    def apply( self, name_layer, name_layer_new, func = None, mode_instructions = 'sparse_for_querying_features', path_folder_ramdata_output = None, dtype_of_row_and_col_indices = np.int32, dtype_of_value = np.float64, int_num_threads = None, flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx = True, int_num_of_records_in_a_chunk_zarr_matrix = 20000, int_num_of_entries_in_a_chunk_zarr_matrix_index = 1000, chunks_dense = ( 2000, 1000 ), dtype_dense_mtx = np.float64, dtype_sparse_mtx = np.float64, dtype_sparse_mtx_index = np.float64 ) :
        ''' # 2022-08-01 10:39:43 
        this function apply a function and/or filters to the records of the given data, and create a new data object with 'name_layer_new' as its name.
        
        example usage: calculate normalized count data, perform log1p transformation, cell filtering, etc.                             
        
        =========
        inputs 
        =========

        'name_layer' : (required) name of the data in the given RamData object to analyze
        'name_layer_new' : (required) name of the new data for the paired RAMtx objects that will contains transformed values (the outputs of the functions applied to previous data values). The disk size of the RAMtx objects can be larger or smaller than the RAMtx objects of 'name_layer'. please make sure that sufficient disk space remains before calling this function.
        'path_folder_ramdata_output' : (Default: store inside the current RamData). The directory of the RamData object that will contain the outputs (paired RAMtx objects). if integer representations of features and barcodes are updated from filtering, the output RAMtx is now incompatible with the current RamData and should be stored as a separate RamData object. The output directory of the new RamData object can be given through this argument.
        'func' : function object or string (Default: identity) a function that takes a tuple of two integers (integer representations of barcode and feature) and another integer or float (value) and returns a modified record. Also, the current RamData object will be given as the first argument (self), and attributes of the current RamData can be used inside the function

                 func( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) -> int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value

                 if None is returned, the entry will be discarded in the output RAMtx object. Therefore, the function can be used to both filter or/and transform values
                 
                 a list of pre-defined functions are the followings:
                 'log1p' :
                          X_new = log_10(X_old + 1)
                 'ident' or None :
                          X_new = X_old
                          
                separate functions for processing feature-by-feature data and barcode-by-barcode data can be given using either dictionary or tuple.
                    func = ( func_bc, func_ft )
                                or
                    func = { 'barcodes' : func_bc, 'features' : func_ft }
        
        'mode_instructions' : instructions for converting modes of the ramtx objects. 
                it is a nested list with the following structure
                [ [ 'ramtx_mode_source', 'ramtx_mode_sink' (or a list of 'ramtx_mode_sink') ], ... ]
                    where 'ramtx_mode_source' is the mode of the ramtx of the 'name_layer' layer from which data will be fetched, and 'ramtx_mode_sink' is the mode of the new ramtx of the 'name_layer_new' layer to which the fetched data will be transformed according to the function and written.
                    
                
                examples:
                
                mode_instructions = 'sparse_for_querying_features' 
                    ... is equivalent to ...
                mode_instructions = [ [ 'sparse_for_querying_features' ] ]
                --> 'sparse_for_querying_features' ramtx object of the 'name_layer' layer converted to 'sparse_for_querying_features' ramtx object of the 'name_layer_new' layer (same ramtx mode)
                
                mode_instructions = [ 'sparse_for_querying_features', [ 'sparse_for_querying_features', 'dense' ] ]
                    ... is equivalent to ...
                mode_instructions = [ [ 'sparse_for_querying_features', [ 'sparse_for_querying_features', 'dense' ] ] ]
                --> 'sparse_for_querying_features' ramtx object of the 'name_layer' layer converted to 'sparse_for_querying_features' and 'dense' ramtx object of the 'name_layer_new' layer 

                mode_instructions = [ [ 'sparse_for_querying_features', 'sparse_for_querying_features' ], 
                                      [ 'sparse_for_querying_features', 'dense' ] ]
                --> 'sparse_for_querying_features' ramtx object of the 'name_layer' layer converted to 'sparse_for_querying_features' and 'dense' ramtx object of the 'name_layer_new' layer. however, for this instruction, 'sparse_for_querying_features' will be read twice, which will be redundant and less efficient.
        
                mode_instructions = [ [ 'sparse_for_querying_features' ],
                                      [ 'dense', [ 'sparse_for_querying_features', 'dense', 'sparse_for_querying_barcodes' ] ] ]
                --> 'sparse_for_querying_features' > 'sparse_for_querying_features'
                    'dense_for_querying_features' > 'sparse_for_querying_features', 'dense' (1 read, 2 write) # by default, dense is set for querying features, but it can be changed so that dense matrix can be constructed by querying barcodes from the source dense ramtx object.
                    'dense_for_querying_barcodes' > 'sparse_for_querying_barcodes' of 'name_layer_new'
                
                mode_instructions = [ [ 'dense_for_querying_features', 'dense_for_querying_barcode' ],
                                      [ 'dense', [ 'sparse_for_querying_features', 'dense', 'sparse_for_querying_barcodes' ] ] ]
                --> 'dense_for_querying_features' > 'dense'
                    'dense_for_querying_features' > 'sparse_for_querying_features' # since 'dense' > 'dense' conversion already took place, (1 read, 1 write) operation will be performed
                    'dense_for_querying_barcodes' > 'sparse_for_querying_barcodes'
                    
                in summary, (1) up to three operations will be performed, to construct three ramtx modes of the resulting layer, (2) the instructions at the front has higher priority, and (3) querying axis of dense can be specified or skipped (in those cases, default will be used)
        
        'int_num_threads' : the number of CPUs to use. by default, the number of CPUs set by the RamData attribute 'int_num_cpus' will be used.
        'dtype_of_row_and_col_indices', 'dtype_of_value' : the dtype of the output matrix
        int_num_of_records_in_a_chunk_zarr_matrix = 20000, int_num_of_entries_in_a_chunk_zarr_matrix_index = 1000, chunks_dense = ( 2000, 1000 ) : determines the chunk size of the output ramtx objects
        dtype_dense_mtx = np.float64, dtype_sparse_mtx = np.float64, dtype_sparse_mtx_index = np.float64 : determines the output dtype
        
        =================
        input attributes 
        =================
        the attributes shown below or any other custom attributes can be used internally as READ-ONLY data objects when executing the given 'func'. 
        
        For example, one can define the following function:
        
        ram = RamData( path_folder_to_ramdata )
        ram.a_variable = 10
        def func( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) :
            return self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value * self.a_variable
        
        # IMPORTANT. since these attributes will be used across multiple processes, only the single RamData 'Apply' operation can be run on Ram data. (or all RamData.Apply operations should use the same attributes containing the same values)
        '''
        # handle inputs
        if int_num_threads is None :
            int_num_threads = self.int_num_cpus
        if name_layer_new is None :
            name_layer_new = name_layer
        flag_new_layer_added_to_the_current_ramdata = False
        if path_folder_ramdata_output is None :
            flag_new_layer_added_to_the_current_ramdata = True # set flag indicating that the new layer will be added to the current ramdata object (or the mask of the current ramdata object)
            path_folder_ramdata_output = self._path_folder_ramdata_modifiable # retrieve path to the modifiable ramdata object
            if path_folder_ramdata_output is None :
                if self.verbose :
                    print( 'current RamData object is not modifiable, exiting' )
                return
        # retrieve flags
        flag_update_a_layer = name_layer_new == name_layer and path_folder_ramdata_output == self._path_folder_ramdata_modifiable # a flag indicating whether a layer of the current ramdata is updated (input ramdata == output ramdata and input layer name == output layer name).
        # retrieve paths
        path_folder_layer_new = f"{path_folder_ramdata_output}{name_layer_new}/" # compose the output directory of the output ramdata layer
        
        # parse 'func' or set default functions, retrieving 'func_bc' and 'func_ft'.
        if hasattr( func, '__call__' ) : # if a single function has been given, use the function for 'func_bc' and 'func_ft'
            func_bc = func
            func_ft = func
        elif isinstance( func, dict ) : # if 'func' is dictionary, parse functions for each axes
            func_bc = func[ 'barcodes' ]
            func_ft = func[ 'features' ]
        elif isinstance( func, tuple ) :
            assert len( func ) == 2 # if 'func' is tuple, the length of 'func' should be 2
            func_bc, func_ft = func
        elif func == 'ident' or func is None  :
            # define identity function if 'func' has not been given
            def func_bc( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) :
                return int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value
            func_ft = func_bc # use the same function for the other axis
        elif func == 'log1p' :
            def func_bc( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) :
                return int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, np.log10( arr_value + 1 )
            func_ft = func_bc # use the same function for the other axis
            
        # check the validility of the input arguments
        if not name_layer in self.layers :
            if verbose :
                print( f"[ERROR] [RamData.Apply] invalid argument 'name_layer' : '{name_layer}' does not exist." )
            return -1 
        elif path_folder_ramdata_output is None and name_layer_new in self.layers : # if the new RAMtx object will be saved to the current RamData and the name of the RAMtx already exists in the current RamData
            if verbose :
                print( f"[ERROR] [RamData.Apply] invalid argument 'name_layer_new' : '{name_layer_new}' is already present in the current RamData." )
            return -1 
        
        ''' set 'name_layer' as a current layer of RamData '''
        self.layer = name_layer
        
        # since a zarr object will be modified by multiple processes, setting 'numcodecs.blosc.use_threads' to False as recommended by the zarr documentation
        zarr_start_multiprocessing_write( )
        
        def RAMtx_Apply( self, rtx, func, flag_dense_ramtx_output, flag_sparse_ramtx_output, int_num_threads ) :
            ''' # 2022-08-01 10:39:38 
            inputs 
            =========

            'rtx': an input RAMtx object
            '''
            ''' prepare '''
            ax = self.ft if rtx.is_for_querying_features else self.bc # retrieve appropriate axis
            ns = dict( ) # create a namespace that can safely shared between different scopes of the functions
            ns[ 'int_num_records_written_to_ramtx' ] = 0 # initlaize the total number of records written to ramtx object
            # create a temporary folder
            path_folder_temp = f'{path_folder_layer_new}temp_{UUID( )}/'
            os.makedirs( path_folder_temp, exist_ok = True )
            
            ''' initialize output ramtx objects '''
            """ %% DENSE %% """
            if flag_dense_ramtx_output : # if dense output is present
                path_folder_ramtx_dense = f"{path_folder_layer_new}dense/"
                os.makedirs( path_folder_ramtx_dense, exist_ok = True ) # create the output ramtx object folder
                path_folder_ramtx_dense_mtx = f"{path_folder_ramtx_dense}matrix.zarr/" # retrieve the folder path of the output RAMtx Zarr matrix object.
                # assert not os.path.exists( path_folder_ramtx_dense_mtx ) # output zarr object should NOT exists!
                path_file_lock_mtx_dense = f'{path_folder_layer_new}lock_{UUID( )}.sync' # define path to locks for parallel processing with multiple processes
                za_mtx_dense = zarr.open( path_folder_ramtx_dense_mtx, mode = 'w', shape = ( rtx._int_num_barcodes, rtx._int_num_features ), chunks = chunks_dense, dtype = dtype_dense_mtx, synchronizer = zarr.ProcessSynchronizer( path_file_lock_mtx_dense ) ) # use the same chunk size of the current RAMtx
            """ %% SPARSE %% """
            if flag_sparse_ramtx_output : # if sparse output is present
                mode_sparse = f"sparse_for_querying_{'features' if rtx.is_for_querying_features else 'barcodes'}"
                path_folder_ramtx_sparse = f"{path_folder_layer_new}{mode_sparse}/"
                os.makedirs( path_folder_ramtx_sparse, exist_ok = True ) # create the output ramtx object folder
                path_folder_ramtx_sparse_mtx = f"{path_folder_ramtx_sparse}matrix.zarr/" # retrieve the folder path of the output RAMtx Zarr matrix object.
                # assert not os.path.exists( path_folder_ramtx_sparse_mtx ) # output zarr object should NOT exists!
                # assert not os.path.exists( f'{path_folder_ramtx_sparse}matrix.index.zarr' ) # output zarr object should NOT exists!
                # define path to locks for parallel processing with multiple processes
                za_mtx_sparse = zarr.open( path_folder_ramtx_sparse_mtx, mode = 'w', shape = ( rtx._int_num_records, 2 ), chunks = ( int_num_of_records_in_a_chunk_zarr_matrix, 2 ), dtype = dtype_sparse_mtx, synchronizer = zarr.ThreadSynchronizer( ) ) # use the same chunk size of the current RAMtx
                za_mtx_sparse_index = zarr.open( f'{path_folder_ramtx_sparse}matrix.index.zarr', mode = 'w', shape = ( rtx.len_axis_for_querying, 2 ), chunks = ( int_num_of_entries_in_a_chunk_zarr_matrix_index, 2 ), dtype = dtype_sparse_mtx_index, synchronizer = zarr.ThreadSynchronizer( ) ) # use the same dtype and chunk size of the current RAMtx
                
                ns[ 'int_num_chunks_written_to_ramtx' ] = 0 # initialize the number of chunks written to ramtx object
                int_num_records_in_a_chunk_of_mtx_sparse = za_mtx_sparse.chunks[ 0 ] # retrieve the number of records in a chunk of output zarr matrix
                
                ns[ 'index_batch_waiting_to_be_written_sparse' ] = 0 # index of the batch currently waiting to be written. 
                ns[ 'l_res_sparse' ] = [ ]
                

            """ convert matrix values and save it to the output RAMtx object """
            # define functions for multiprocessing step
            def process_batch( batch, pipe_to_main_process ) :
                ''' # 2022-05-08 13:19:07 
                retrieve data for a given list of entries, transform values, and save to a Zarr object and index the object, and returns the number of written records and the paths of the written objects (index and Zarr matrix)
                '''
                # initialize
                path_folder_zarr_output_sparse = None
                path_file_index_output_sparse = None
                
                # parse batch
                int_num_processed_records, index_batch, l_int_entry_current_batch = batch[ 'int_accumulated_weight_current_batch' ], batch[ 'index_batch' ], batch[ 'l_int_entry_current_batch' ]
                
                # retrieve the number of index_entries
                int_num_entries = len( l_int_entry_current_batch )
                int_num_records_written = 0 # initialize the record count
                l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying, l_arr_value = [ ], [ ], [ ] # initializes lists for collecting transformed data
                
                """ %% SPARSE %% """
                if flag_sparse_ramtx_output : # if sparse output is present
                    # open an Zarr object
                    path_folder_zarr_output_sparse = f"{path_folder_temp}{UUID( )}.zarr/" # define output Zarr object path
                    za_output_sparse = zarr.open( path_folder_zarr_output_sparse, mode = 'w', shape = ( rtx._int_num_records, 2 ), chunks = za_mtx_sparse.chunks, dtype = dtype_of_value, synchronizer = zarr.ThreadSynchronizer( ) )
                    # define an index file
                    path_file_index_output_sparse = f"{path_folder_temp}{UUID( )}.index.tsv.gz" # define output index file path
                    l_index = [ ] # collect index
                
                # iterate through the data of each entry and transform the data
                for int_entry_of_axis_for_querying, arr_int_entry_of_axis_not_for_querying, arr_value in zip( * rtx[ l_int_entry_current_batch ] ) : # retrieve data for the current batch
                    # transform the values of an entry
                    int_entry_of_axis_for_querying, arr_int_entry_of_axis_not_for_querying, arr_value = func( self, int_entry_of_axis_for_querying, arr_int_entry_of_axis_not_for_querying, arr_value ) 
                    int_num_records = len( arr_value ) # retrieve number of returned records

                    """ %% SPARSE %% """
                    if flag_sparse_ramtx_output : # if sparse output is present
                        # collect index
                        l_index.append( [ int_entry_of_axis_for_querying, int_num_records_written, int_num_records_written + int_num_records ] )
                    
                    # collect transformed data
                    l_int_entry_of_axis_for_querying.append( int_entry_of_axis_for_querying )
                    l_arr_int_entry_of_axis_not_for_querying.append( arr_int_entry_of_axis_not_for_querying )
                    l_arr_value.append( arr_value )
                    int_num_records_written += int_num_records # update the number of records written
                del int_entry_of_axis_for_querying, arr_int_entry_of_axis_not_for_querying, arr_value # delete references
                
                """ combine results """
                # combine the arrays
                arr_int_entry_of_axis_not_for_querying = np.concatenate( l_arr_int_entry_of_axis_not_for_querying )
                arr_value = np.concatenate( l_arr_value )
                del l_arr_value # delete intermediate objects

                # compose 'arr_int_entry_of_axis_for_querying'
                arr_int_entry_of_axis_for_querying = np.zeros( len( arr_int_entry_of_axis_not_for_querying ), dtype = self._dtype_of_feature_and_barcode_indices ) # create an empty array
                int_pos = 0
                for int_entry_of_axis_for_querying, a in zip( l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying ) :
                    n = len( a )
                    arr_int_entry_of_axis_for_querying[ int_pos : int_pos + n ] = int_entry_of_axis_for_querying # compose 'arr_int_entry_of_axis_for_querying'
                    int_pos += n # update the current position
                del l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying # delete intermediate objects
                    
                """ %% DENSE %% """
                if flag_dense_ramtx_output : # if dense output is present
                    za_mtx_dense.set_coordinate_selection( ( arr_int_entry_of_axis_not_for_querying, arr_int_entry_of_axis_for_querying ) if rtx.is_for_querying_features else ( arr_int_entry_of_axis_for_querying, arr_int_entry_of_axis_not_for_querying ), arr_value ) # write dense zarr matrix

                """ %% SPARSE %% """
                if flag_sparse_ramtx_output : # if sparse output is present
                    za_output_sparse[ : int_num_records_written ] = np.vstack( ( arr_int_entry_of_axis_not_for_querying, arr_value ) ).T # save transformed data
                    za_output_sparse.resize( int_num_records_written, 2 ) # resize the output Zarr object
                    pd.DataFrame( l_index ).to_csv( path_file_index_output_sparse, header = None, index = None, sep = '\t' ) # write the index file
                    
                pipe_to_main_process.send( ( index_batch, int_num_processed_records, int_num_records_written, path_folder_zarr_output_sparse, path_file_index_output_sparse ) ) # send information about the output files
            # initialize the progress bar
            pbar = progress_bar( total = rtx.get_total_num_records( int_num_entries_for_each_weight_calculation_batch = self.int_num_entries_for_each_weight_calculation_batch, flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx = self.flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx ) )
            def post_process_batch( res ) :
                """ # 2022-07-06 10:22:05 
                """
                # parse result
                index_batch, int_num_processed_records, int_num_records_written, path_folder_zarr_output, path_file_index_output = res
                ns[ 'int_num_records_written_to_ramtx' ] += int_num_records_written # update the number of records written to the output RAMtx
                
                """ %% SPARSE %% """
                if flag_sparse_ramtx_output : # if sparse output is present
                    ''' collect result of the current batch '''
                    while len( ns[ 'l_res_sparse' ] ) < index_batch + 1 + 1 : # increase the length of ns[ 'l_res_sparse' ] until it can contain the result produced from the current batch. # add a padding (+1) to not raise indexError
                        ns[ 'l_res_sparse' ].append( 0 ) # not completed batch will be marked by 0
                    ns[ 'l_res_sparse' ][ index_batch ] = res # collect the result produced from the current batch
                    
                    ''' process results produced from batches in the order the batches were generated (in an ascending order of 'int_entry') '''
                    while ns[ 'l_res_sparse' ][ ns[ 'index_batch_waiting_to_be_written_sparse' ] ] != 0 : # if the batch referenced by ns[ 'index_batch_waiting_to_be_written_sparse' ] has been completed
                        index_batch, int_num_processed_records, int_num_records_written, path_folder_zarr_output, path_file_index_output = ns[ 'l_res_sparse' ][ ns[ 'index_batch_waiting_to_be_written_sparse' ] ] # parse result
                        int_num_chunks_written_for_a_batch = int( np.ceil( int_num_records_written / int_num_records_in_a_chunk_of_mtx_sparse ) ) # retrieve the number of chunks that were written for a batch
                        int_num_chunks_written_to_ramtx = ns[ 'int_num_chunks_written_to_ramtx' ] # retrieve the number of chunks already present in the output RAMtx zarr matrix object
                        
                        # check size of Zarr matrix object, and increase the size if needed.
                        int_min_num_rows_required = ( int_num_chunks_written_to_ramtx + int_num_chunks_written_for_a_batch ) * int_num_records_in_a_chunk_of_mtx_sparse # calculate the minimal number of rows required in the RAMtx Zarr matrix object
                        if za_mtx_sparse.shape[ 0 ] < int_min_num_rows_required : # check whether the size of Zarr matrix is smaller than the minimum requirement
                            za_mtx_sparse.resize( int_min_num_rows_required, 2 ) # resize the Zarr matrix so that data can be safely added to the matrix

                        # copy Zarr chunks to the sparse RAMtx Zarr matrix object
                        os.chdir( path_folder_zarr_output ) # to reduce the length of file path, change directory to the output folder before retrieving file paths of the chunks
                        for e in glob.glob( '*.0' ) : # to reduce the size of file paths returned by glob, use relative path to retrieve the list of chunk files of the Zarr matrix of the current batch
                            index_chunk = int( e.split( '.0', 1 )[ 0 ] ) # retrieve the integer index of the chunk
                            os.rename( e, path_folder_ramtx_sparse_mtx + str( index_chunk + int_num_chunks_written_to_ramtx ) + '.0' ) # simply rename the chunk to transfer stored values

                        # retrieve index data of the current batch
                        arr_index = pd.read_csv( path_file_index_output, header = None, sep = '\t' ).values.astype( int ) # convert to integer dtype
                        arr_index[ :, 1 : ] += int_num_chunks_written_to_ramtx * int_num_records_in_a_chunk_of_mtx_sparse # match the chunk boundary. if there are empty rows in the chunks currently written to ramtx, these empty rows will be considered as rows containing records, so that Zarr matrix written for a batch can be easily transferred by simply renaming the chunk files
                        za_mtx_sparse_index.set_orthogonal_selection( arr_index[ :, 0 ], arr_index[ :, 1 : ] ) # update the index of the entries of the current batch

                        # update the number of chunks written to RAMtx Zarr matrix object
                        ns[ 'int_num_chunks_written_to_ramtx' ] += int_num_chunks_written_for_a_batch

                        # delete temporary files and folders
                        shutil.rmtree( path_folder_zarr_output )
                        os.remove( path_file_index_output )
                        
                        ns[ 'l_res_sparse' ][ ns[ 'index_batch_waiting_to_be_written_sparse' ] ] = None # remove the result from the list of batch outputs
                        ns[ 'index_batch_waiting_to_be_written_sparse' ] += 1 # start waiting for the next batch to be completed
                        pbar.update( int_num_processed_records ) # update the progress bar
                
            # transform the values of the RAMtx using multiple processes
            bk.Multiprocessing_Batch( rtx.batch_generator( ax.filter, int_num_entries_for_each_weight_calculation_batch = self.int_num_entries_for_each_weight_calculation_batch, int_total_weight_for_each_batch = self.int_total_weight_for_each_batch, flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx = self.flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx ), process_batch, post_process_batch = post_process_batch, int_num_threads = int_num_threads, int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop = 0.2 ) # create batch considering chunk boundaries # return batch index to allow combining sparse matrix in an ascending order.
            pbar.close( ) # close the progress bar
            
            # remove temp folder
            shutil.rmtree( path_folder_temp )
            
            ''' export ramtx settings '''
            """ %% DENSE %% """
            if flag_dense_ramtx_output : # if dense output is present
                shutil.rmtree( path_file_lock_mtx_dense ) # delete file system locks
                root = zarr.group( path_folder_ramtx_dense )
                root.attrs[ 'dict_metadata' ] = { 
                    'mode' : 'dense',
                    'str_completed_time' : TIME_GET_timestamp( True ),
                    'int_num_features' : rtx._int_num_features,
                    'int_num_barcodes' : rtx._int_num_barcodes,
                    'int_num_records' : ns[ 'int_num_records_written_to_ramtx' ],
                    'version' : _version_,
                }

            """ %% SPARSE %% """
            if flag_sparse_ramtx_output : # if sparse output is present
                root = zarr.group( path_folder_ramtx_sparse )
                root.attrs[ 'dict_metadata' ] = { 
                    'mode' : mode_sparse,
                    'flag_ramtx_sorted_by_id_feature' : rtx.is_for_querying_features,
                    'str_completed_time' : TIME_GET_timestamp( True ),
                    'int_num_features' : rtx._int_num_features,
                    'int_num_barcodes' : rtx._int_num_barcodes,
                    'int_num_records' : ns[ 'int_num_records_written_to_ramtx' ],
                    'version' : _version_,
                }
            return 
        
        # initialize the list of processes
        l_p = [ ]
        
        """
        pre-process 'mode_instructions'
        """
        # handle the given argument
        # find the element with mimimum nested list depth, and if it is less than 2, wrap the 'mode_instructions' in a list until the minimum nested list depth becomes 2
        def __find_min_depth( l, current_depth = 0 ) :
            """ # 2022-07-31 15:56:55 
            breadth search for finding the min depth of a given nested list
            """
            if isinstance( l, str ) :
                return current_depth
            else :
                return min( __find_min_depth( e, current_depth = current_depth + 1 ) for e in l )
        for _ in range( 2 - __find_min_depth( mode_instructions ) ) :
            mode_instructions = [ mode_instructions ]
        # { 'dense', 'dense_for_querying_barcodes', 'dense_for_querying_features', 'sparse_for_querying_barcodes', 'sparse_for_querying_features' }
        set_modes_sink_valid = { 'dense', 'sparse_for_querying_barcodes', 'sparse_for_querying_features' }
        set_modes_sink = set( self.layer.modes ) if flag_update_a_layer else set( ) # retrieve the ramtx modes in the output (sink) layer (assumes a data sink layer (that is not data source layer) does not contain any ramtx objects). # to avoid overwriting
        for an_instruction in mode_instructions : # first instructions takes priority
            """
            pre-process each instruction
            """
            # if the 'ramtx_mode_sink' has not been set, 'ramtx_mode_source' will be used as the mode of the ramtx sink, too.
            if len( an_instruction ) == 1 :
                an_instruction = [ an_instruction, an_instruction ]
            ramtx_mode_source, l_ramtx_mode_sink = an_instruction # parse an instruction
            # if 'l_ramtx_mode_sink' is a single 'ramtx_mode_sink', wrap the entry in a list
            if isinstance( l_ramtx_mode_sink, str ) :
                l_ramtx_mode_sink = [ l_ramtx_mode_sink ]
            # if 'ramtx_mode_source' does not exist in the current layer, ignore the current instruction
            if ramtx_mode_source not in self.layer :
                continue
            # compose a valid set of 'ramtx_mode_sink'
            set_ramtx_mode_sink = set( 'dense' if 'dense' in e else e for e in list( e.lower( ) for e in l_ramtx_mode_sink ) ).intersection( set_modes_sink_valid ).difference( set_modes_sink ) # for each given valid 'ramtx_mode_sink' # if 'ramtx_mode_sink' already exists in the output layer (or will exists after running previous instructions), ignore 'ramtx_mode_sink'
            # if there is no valid ramtx sink modes, ignore the instruction
            if len( set_ramtx_mode_sink ) == 0 :
                continue
            
            """
            compose process
            """
            flag_dense_ramtx_output, flag_sparse_ramtx_output = False, False
            if not 'dense' in ramtx_mode_source : # sparse source
                if ramtx_mode_source in set_ramtx_mode_sink : # sparse sink presents
                    flag_sparse_ramtx_output = True
                    set_modes_sink.add( ramtx_mode_source ) # update written (or will be written) sink ramtx modes
                if 'dense' in set_ramtx_mode_sink : # dense sink presents
                    flag_dense_ramtx_output = True
                    set_modes_sink.add( 'dense' ) # update written (or will be written) sink ramtx modes
                
                # add a process if valid output exists
                if flag_sparse_ramtx_output or flag_dense_ramtx_output :
                    l_p.append( mp.Process( target = RAMtx_Apply, args = ( self, self.layer[ ramtx_mode_source ], func_ft if 'features' in ramtx_mode_source else func_bc, flag_dense_ramtx_output, flag_sparse_ramtx_output, int_num_threads ) ) ) # add process based on which axis will be queried for source ramtx
            else : # dense source
                set_modes_sink.update( set_ramtx_mode_sink ) # update written (or will be written) sink ramtx modes. dense source can write all sink modes
                if 'dense' in set_ramtx_mode_sink : # dense sink presents
                    flag_dense_ramtx_output = True
                flag_querying_features = self.layer[ 'dense' ].is_for_querying_features if ramtx_mode_source == 'dense' else 'features' in ramtx_mode_source # retrieve a flag for querying with features
                if 'sparse_for_querying_barcodes' in set_ramtx_mode_sink and 'sparse_for_querying_features' in set_ramtx_mode_sink : # if both sparse sink modes are present, run two processes, and add dense sink to one of the processes based on the given preference
                    l_p.append( mp.Process( target = RAMtx_Apply, args = ( self, self.layer[ 'dense_for_querying_barcodes' ], func_bc, flag_dense_ramtx_output and ( not flag_querying_features ), True, int_num_threads ) ) ) # add a process for querying barcodes
                    l_p.append( mp.Process( target = RAMtx_Apply, args = ( self, self.layer[ 'dense_for_querying_features' ], func_ft, flag_dense_ramtx_output and flag_querying_features, True, int_num_threads ) ) ) # add a process for querying features
                elif 'sparse_for_querying_barcodes' in set_ramtx_mode_sink : # if only a single sparse ramtx (barcodes-indexed) sink is present
                    l_p.append( mp.Process( target = RAMtx_Apply, args = ( self, self.layer[ 'dense_for_querying_barcodes' ], func_bc, flag_dense_ramtx_output, True, int_num_threads ) ) ) # add a process for querying barcodes
                elif 'sparse_for_querying_features' in set_ramtx_mode_sink : # if only a single sparse ramtx (features-indexed) sink is present
                    l_p.append( mp.Process( target = RAMtx_Apply, args = ( self, self.layer[ 'dense_for_querying_features' ], func_ft, flag_dense_ramtx_output, True, int_num_threads ) ) ) # add a process for querying features
                elif flag_dense_ramtx_output : # if only dense sink present, use the axis based on the given preference
                    l_p.append( mp.Process( target = RAMtx_Apply, args = ( self, self.layer[ f"dense_for_querying_{'features' if flag_querying_features else 'barcodes'}" ], func_ft if flag_querying_features else func_bc, True, False, int_num_threads ) ) ) # add a process for querying features
        
        # run processes
        if len( l_p ) == 0 :
            if self.verbose :
                print( 'no operation was performed' )
            return
        for p in l_p : p.start( )
        for p in l_p : p.join( )
        
        # revert the setting 
        zarr_end_multiprocessing_write( )
        
        """
        update the metadata
        """
        if self.verbose :
            print( f'apply operation {name_layer} > {name_layer_new} has been completed' )
            
        # update metadata of the layer added to the current RamData
        if flag_new_layer_added_to_the_current_ramdata :
            if flag_update_a_layer : # update metadata of the current layer
                self.layer._dict_metadata[ 'set_modes' ].update( list( set_modes_sink ) + ( [ 'dense_for_querying_barcodes', 'dense_for_querying_features' ] if 'dense' in set_modes_sink else [ ] ) )
                self.layer._save_metadata_( ) # update metadata
                self.layer._load_ramtx_objects( ) # load written ramtx objects
            else : # update metadata of new layer
                # write layer metadata
                lay = zarr.group( path_folder_layer_new )
                lay.attrs[ 'dict_metadata' ] = { 
                    'set_modes' : list( set_modes_sink ) + ( [ 'dense_for_querying_barcodes', 'dense_for_querying_features' ] if 'dense' in set_modes_sink else [ ] ), # dense ramtx can be operated for querying either barcodes/features
                    'version' : _version_,
                }
            
        # update metadata of current RamData
        # update 'layers' if the layer has been saved in the current RamData object (or the mask of the current RamData object)
        if flag_new_layer_added_to_the_current_ramdata and not flag_update_a_layer :
            self.layers.add( name_layer_new )
            self._save_metadata_( )
    def subset( self, path_folder_ramdata_output, l_name_layer = [ ], int_num_threads = None, ** kwargs ) :
        ''' # 2022-07-05 23:20:02 
        Under Construction!
        this function will create a new RamData object on disk by creating a subset of the current RamData according to the current filters

        =========
        inputs 
        =========
        'path_folder_ramdata_output' : The directory of the RamData object that will contain a subset of the barcodes/features of the current RamData.
        'l_name_layer' : the list of name_layers to subset and transfer to the new RamData object
        'int_num_threads' : the number of CPUs to use. by default, the number of CPUs set by the RamData attribute 'int_num_cpus' will be used.
        '''
        ''' handle inputs '''
        # check invalid input
        if path_folder_ramdata_output == self._path_folder_ramdata:
            if self.verbose :
                print( f'the output RamData object directory is exactly same that of the current RamData object, exiting' )
        # create the RamData output folder
        os.makedirs( path_folder_ramdata_output, exist_ok = True ) 

        # copy axes and associated metadata
        self.bc.save( path_folder_ramdata_output )
        self.ft.save( path_folder_ramdata_output )

        # retrieve valid set of name_layer
        set_name_layer = self.layers.intersection( l_name_layer )
        
        ''' filter each layer '''
        self.load_dict_change( ) # load 'dict_change' for coordinate conversion according to the given filter
        for name_layer in set_name_layer : # for each valid name_layer
            self.apply( name_layer, name_layer_new = None, func = 'ident', path_folder_ramdata_output = path_folder_ramdata_output, int_num_threads = int_num_threads, ** kwargs ) # flag_dtype_output = None : use the same dtype as the input RAMtx object
        self.unload_dict_change( ) # unload 'dict_change' after the conversion process
        
        # compose metadata
        root = zarr.group( path_folder_ramdata_output )
        root.attrs[ 'dict_metadata' ] = { 
            'str_completed_time' : TIME_GET_timestamp( True ),
            'int_num_features' : self.ft.meta.n_rows, # record the number of features/barcodes after filtering
            'int_num_barcodes' : self.bc.meta.n_rows,
            'int_num_of_records_in_a_chunk_zarr_matrix' : self.metadata[ 'int_num_of_records_in_a_chunk_zarr_matrix' ],
            'int_num_of_entries_in_a_chunk_zarr_matrix_index' : self.metadata[ 'int_num_of_entries_in_a_chunk_zarr_matrix_index' ],
            'layers' : list( set_name_layer ),
            'version' : _version_,
        }
    def normalize( self, name_layer = 'raw', name_layer_new = 'normalized', name_col_total_count = 'raw_sum', int_total_count_target = 10000, mode_instructions = [ [ 'sparse_for_querying_features', 'sparse_for_querying_features' ], [ 'sparse_for_querying_barcodes', 'sparse_for_querying_barcodes' ] ], int_num_threads = None, ** kwargs ) :
        ''' # 2022-07-06 23:58:15 
        this function perform normalization of a given data and will create a new data in the current RamData object.

        =========
        inputs 
        =========

        'name_layer' : name of input data
        'name_layer_new' : name of the output (normalized) data
        'name_col_total_count' : name of column of barcode metadata (ZarrDataFrame) to use as total counts of barcodes
        'int_total_count_target' : the target total count. the count data will be normalized according to the total counts of barcodes so that the total counts of each barcode after normalization becomes 'int_total_count_target'.
        'mode_instructions' : please refer to the RamData.apply method
        'int_num_threads' : the number of CPUs to use. by default, the number of CPUs set by the RamData attribute 'int_num_cpus' will be used.
        
        ** kwargs : arguments for 'RamData.apply' method
        '''
        # check validity of inputs
        if name_col_total_count not in self.bc.meta : # 'name_col_total_count' column should be available in the metadata
            if self.verbose :
                print( f"[RamData.normalize] 'name_col_total_count' '{name_col_total_count}' does not exist in the 'barcodes' metadata, exiting" )
            return
        
        # load total count data
        flag_name_col_total_count_already_loaded = name_col_total_count in self.bc.meta.dict # a flag indicating that the total count data of barcodes has been already loaded
        if not flag_name_col_total_count_already_loaded : # load count data of barcodes in memory
            self.bc.meta.load_as_dict( name_col_total_count )
        dict_count = self.bc.meta.dict[ name_col_total_count ] # retrieve total count data as a dictionary
        
        # load layer
        self.layer = name_layer
        
        # define functions for normalization
        def func_norm_barcode_indexed( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) :
            """ # 2022-07-06 23:58:27 
            """
            return int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, ( arr_value / dict_count[ int_entry_of_axis_for_querying ] * int_total_count_target ) # normalize count data of a single barcode
        
        def func_norm_feature_indexed( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) : # normalize count data of a single feature containing (possibly) multiple barcodes
            """ # 2022-07-06 23:58:38 
            """
            # perform normalization in-place
            for i, e in enumerate( arr_int_entries_of_axis_not_for_querying.astype( int ) ) : # iterate through barcodes
                arr_value[ i ] = arr_value[ i ] / dict_count[ e ] # perform normalization of count data for each barcode
            arr_value *= int_total_count_target
            return int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value
        
        ''' normalize the RAMtx matrices '''
        self.apply( name_layer, name_layer_new, func = ( func_norm_barcode_indexed, func_norm_feature_indexed ), int_num_threads = int_num_threads, mode_instructions = mode_instructions, ** kwargs ) # flag_dtype_output = None : use the same dtype as the input RAMtx object
    
        if not flag_name_col_total_count_already_loaded : # unload count data of barcodes from memory if the count data was not loaded before calling this method
            del self.bc.meta.dict[ name_col_total_count ]
    def scale( self, name_layer = 'normalized_log1p', name_layer_new = 'normalized_log1p_scaled', name_col_variance = 'normalized_log1p_variance', name_col_mean = 'normalized_log1p_mean', max_value = 10, mode_instructions = [ [ 'sparse_for_querying_features', 'sparse_for_querying_features' ], [ 'sparse_for_querying_barcodes', 'sparse_for_querying_barcodes' ] ], int_num_threads = None, ** kwargs ) :
        """ # 2022-07-27 16:32:26 
        current implementation only allows output values to be not zero-centered. the zero-value will remain zero, while Z-scores of the non-zero values will be increased by Z-score of zero values, enabling processing of sparse count data

        'name_layer' : the name of the data source layer
        'name_layer_new' : the name of the data sink layer (new layer)
        'name_col_variance' : name of feature metadata containing variance informatin
        'name_col_mean' : name of feature metadata containing mean informatin
        'max_value' : clip values larger than 'max_value' to 'max_value'
        'mode_instructions' : please refer to the RamData.apply method
        'int_num_threads' : the number of CPUs to use. by default, the number of CPUs set by the RamData attribute 'int_num_cpus' will be used.
        
        ** kwargs : arguments for 'RamData.apply' method
        """
        # check validity of inputs
        # column names should be available in the metadata
        if name_col_variance not in self.ft.meta : # 'name_col_variance' column should be available in the metadata
            if self.verbose :
                print( f"[RamData.scale] 'name_col_variance' '{name_col_total_count}' does not exist in the 'barcodes' metadata, exiting" )
            return
    #     assert name_col_mean in self.ft.meta 

        # load feature data
        # retrieve flag indicating whether the data has been already loaded 
        flag_name_col_variance_already_loaded = name_col_variance in self.ft.meta.dict 
        if not flag_name_col_variance_already_loaded : # load data in memory
            self.ft.meta.load_as_dict( name_col_variance )
    #     flag_name_col_mean_already_loaded = name_col_mean in self.ft.meta.dict 
    #     if not flag_name_col_mean_already_loaded : 
    #         self.ft.meta.load_as_dict( name_col_mean )
        # retrieve data as a dictionary
        dict_variance = self.ft.meta.dict[ name_col_variance ]
    #     dict_mean = self.ft.meta.dict[ name_col_mean ]

        # load layer
        self.layer = name_layer

        # define functions for scaling
        def func_feature_indexed( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) :
            """ # 2022-07-27 14:32:21 
            """
            float_std = dict_variance[ int_entry_of_axis_for_querying ] ** 0.5 # retrieve standard deviation from the variance
            if float_std == 0 : # handle when standard deviation is zero (all the data values should be zero)
                return int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value # return result
            arr_value /= float_std # scale count data using the standard deviation (in-place)
            arr_value[ arr_value > max_value ] = max_value # cap exceptionally large values (in-place)
            return int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value # return scaled data

        def func_barcode_indexed( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) : # normalize count data of a single barcode containing (likely) multiple features
            """ # 2022-07-27 16:32:21 
            """
            # perform scaling in-place to reduce memory consumption
            for i, e in enumerate( arr_int_entries_of_axis_not_for_querying.astype( int ) ) : # iterate through barcodes
                float_std = dict_variance[ e ] ** 0.5 # retrieve standard deviation of the current feature from the variance
                arr_value[ i ] = 0 if float_std == 0 else arr_value[ i ] / float_std # perform scaling of data for each feature
            return int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value

        ''' process the RAMtx matrices '''
        self.apply( name_layer, name_layer_new, func = ( func_barcode_indexed, func_feature_indexed ), int_num_threads = int_num_threads, mode_instructions = mode_instructions, ** kwargs ) # flag_dtype_output = None : use the same dtype as the input RAMtx object

        # unload count data of barcodes from memory if the count data was not loaded before calling this method
    #     if not flag_name_col_mean_already_loaded : 
    #         del self.ft.meta.dict[ name_col_mean ]
        if not flag_name_col_variance_already_loaded : 
            del self.ft.meta.dict[ name_col_variance ]
    def identify_highly_variable_features( self, name_layer = 'normalized_log1p', int_num_highly_variable_features = 2000, float_min_mean = 0.01, float_min_variance = 0.01, name_col_filter = 'filter_normalized_log1p_highly_variable', flag_show_graph = True ) :
        """ # 2022-06-07 22:53:55 
        identify highly variable features
        learns mean-variable relationship from the given data, and calculate residual variance to identify highly variable features.
        **Warning** : filters of the current RamData will be reset, and filters based on the identified highly variable features will be set.

        
        'int_num_highly_variable_features' : number of highly variable features to select. if None is given. a threshold for the selection of highly variable features will be set automatically to '0'.
        'float_min_mean' : minimum mean expression for selecting highly variable features
        'float_min_variance' : minimum variance of expression for selecting highly variable features
        'name_col_filter' : the name of column that will contain a feature/barcode filter containing selected highly variable features (and barcode filter for cells that have non-zero expression values for the selected list of highly-variable features)
        'flag_show_graph' : show graphs
        ==========
        returns
        ==========

        new columns will be added to self.ft.meta metadata
        """
        """
        (1) Calculate metrics for identification of highly variable features
        """
        # set the name of the columns that will be used in the current method
        name_col_for_mean, name_col_for_variance = f'{name_layer}_mean', f'{name_layer}_variance'
        
        # check if required metadata (mean and variance data of features) is not available, and if not, calculate and save the data
        if name_col_for_mean not in self.ft.meta or name_col_for_variance not in self.ft.meta :
            self.summarize( name_layer, 'feature', 'sum_and_dev' ) # calculate mean and variance for features

        # load mean and variance data in memory
        arr_mean = self.ft.meta[ name_col_for_mean ]
        arr_var = self.ft.meta[ name_col_for_variance ]

        if flag_show_graph :
            plt.plot( arr_mean[ : : 10 ], arr_var[ : : 10 ], '.', alpha = 0.01 )
            MATPLOTLIB_basic_configuration( x_scale = 'log', y_scale = 'log', x_label = 'mean', y_label = 'variance', title = f"mean-variance relationship\nin '{name_layer}'" )
            plt.show( )
            
        # learn mean-variance relationship for the data
        mask = ~ np.isnan( arr_var ) # exclude values containing np.nan
        if mask.sum( ) == 0 : # exit if no valid data is available for fitting
            return
        mean_var_relationship_fit = np.polynomial.polynomial.Polynomial.fit( arr_mean[ mask ], arr_var[ mask ], 2 ) # fit using polynomial with degree 2
        del mask # delete temporary object

        # initialize output values
        n = len( arr_mean ) # the number of output entries
        arr_ratio_of_variance_to_expected_variance_from_mean = np.full( n, np.nan )
        arr_diff_of_variance_to_expected_variance_from_mean = np.full( n, np.nan )

        for i in range( n ) : # iterate each row
            mean, var = arr_mean[ i ], arr_var[ i ] # retrieve var and mean
            if not np.isnan( var ) : # if current entry is valid
                var_expected = mean_var_relationship_fit( mean ) # calculate expected variance from the mean
                if var_expected == 0 : # handle the case when the current expected variance is zero 
                    arr_ratio_of_variance_to_expected_variance_from_mean[ i ] = np.nan
                    arr_diff_of_variance_to_expected_variance_from_mean[ i ] = np.nan
                else :
                    arr_ratio_of_variance_to_expected_variance_from_mean[ i ] = var / var_expected
                    arr_diff_of_variance_to_expected_variance_from_mean[ i ] = var - var_expected

        # add data to feature metadata
        self.ft.meta[ f'{name_layer}__float_ratio_of_variance_to_expected_variance_from_mean' ] = arr_ratio_of_variance_to_expected_variance_from_mean
        self.ft.meta[ f'{name_layer}__float_diff_of_variance_to_expected_variance_from_mean' ] = arr_diff_of_variance_to_expected_variance_from_mean
        self.ft.meta[ f'{name_layer}__float_score_highly_variable_feature_from_mean' ] = arr_ratio_of_variance_to_expected_variance_from_mean * arr_diff_of_variance_to_expected_variance_from_mean # calculate the product of the ratio and difference of variance to expected variance for scoring and sorting highly variable features

        """
        (2) identify of highly variable features
        """
        # reset the feature filter prior to retrieve the metadata of all features
        ba = self.ft.all( ) if self.ft.filter is None else self.ft.filter # retrieve filter
        
        # filter using variance
        self.ft.filter = None
        ba = self.ft.AND( ba, self.ft.meta[ name_col_for_variance ] > float_min_variance ) 
        self.ft.filter = ba
        
        # calculate a threshold for highly variable score
        arr_scores = self.ft.meta[ f'{name_layer}__float_score_highly_variable_feature_from_mean' ]
        float_min_score_highly_variable = arr_scores[ np.lexsort( ( self.ft.meta[ name_col_for_mean ], arr_scores ) )[ - int_num_highly_variable_features ] ]
        del arr_scores
        
        # filter using highly variable score
        self.ft.filter = None
        ba = self.ft.AND( ba, self.ft.meta[ f'{name_layer}__float_score_highly_variable_feature_from_mean' ] > float_min_score_highly_variable ) 
        self.ft.filter = ba
        self.ft.save_filter( name_col_filter ) # save the feature filter as a metadata
    ''' function for fast exploratory analysis '''
    def prepare_dimension_reduction_from_raw( self, name_layer_raw : str = 'raw', name_layer_normalized : str = 'normalized', name_layer_log_transformed : str = 'normalized_log1p', name_layer_scaled : str = 'normalized_log1p_scaled', name_col_filter_filtered_barcode : str = 'filtered_barcodes', min_counts : int = 500, min_features : int = 100, int_total_count_target : int = 10000, int_num_highly_variable_features : int = 2000, max_value : float = 10, dict_kw_hv : dict = { 'float_min_mean' : 0.01, 'float_min_variance' : 0.01, 'name_col_filter' : 'filter_normalized_log1p_highly_variable' }, flag_use_fast_mode : bool = True ) :
        """ # 2022-08-24 17:02:11 
        assumes raw count data (or the equivalent of it) is available in 'dense' format

        === general ===
        flag_use_fast_mode : bool = True : if True, a fast method designed for fast global exploratory analysis (UMAP projection) of the raw data, removing unncessary layer building operations as much as possible. if False, every layer will be written to disk, unfiltered (containing all barcodes and features). 'slow' mode will be much slower but can be re-analyzed more efficiently later (subclustering, etc.)

        === input/output layers ===
        name_layer_raw : str = 'raw' # the name of the layer containing 'raw' count data
        name_layer_normalized : str = 'normalized' # the name of the layer containing normalized raw count data
        name_layer_log_transformed : str = 'normalized_log1p' # the name of the layer containing log-transformed normalized raw count data
        name_layer_scaled : str = 'normalized_log1p_scaled' : the name of the layer that will contain the log-normalized, scale gene expression data in a 'sparse_for_querying_barcodes' ramtx mode of only the highly variable genes, selected by the current filter settings, 'int_num_highly_variable_features', and 'dict_kw_hv' arguments. data will be scaled and capped according to 'max_value' arguments
        
        === barcode filtering ===
        name_col_filter_filtered_barcode : str = 'filtered_barcodes' # the name of metadata column that will contain filter containing active barcode entries after barcode filtering
        
        'int_total_count_target' : total count target for normalization
        'min_counts' = 500, 'min_features' = 100 : for barcode filtering
        'int_num_highly_variable_features' : the number of highly variable genes to retrieve
        'max_value' = 10 : capping at this value during scaling
        """
        # load a raw count layer
        self.layer = name_layer_raw 
        self.layer[ 'dense' ].survey_number_of_records_for_each_entry( ) # prepare operation on dense RAMtx
        
        # in 'slow' mode, use sparse matrix for more efficient operation
        if not flag_use_fast_mode :
            """ %% SLOW MODE %% """
            if self.verbose :
                print( f"[RamData.prepare_dimension_reduction_from_raw] [SLOW MODE] converting dense to sparse formats ... " )
            # dense -> sparse conversion
            self.apply( name_layer_raw, name_layer_raw, 'ident', mode_instructions = [ [ 'dense', 'sparse_for_querying_features' ], [ 'dense', 'sparse_for_querying_barcodes' ] ] )

        # calculate total counts for each barcode
        if self.verbose :
            print( f"[RamData.prepare_dimension_reduction_from_raw] summarizing total count for each barcode ... " )
        self.summarize( name_layer_raw, 'barcode', 'sum' )

        # filter cells
        if self.verbose :
            print( f"[RamData.prepare_dimension_reduction_from_raw] filtering barcodes ... " )
        self.bc.filter = ( self.bc.all( ) if self.bc.filter is None else self.bc.filter ) & BA.to_bitarray( self.bc.meta[ f'{name_layer_raw}_sum', : ] > min_counts ) & BA.to_bitarray( self.bc.meta[ f'{name_layer_raw}_num_nonzero_values', : ] > min_features )
        self.bc.save_filter( name_col_filter_filtered_barcode ) # save filter for later analysis

        if flag_use_fast_mode :
            """ %% FAST MODE %% """
            # retrieve total raw count data for normalization
            self.bc.meta.load_as_dict( f'{name_layer_raw}_sum' )
            dict_count = self.bc.meta.dict[ f'{name_layer_raw}_sum' ] # retrieve total counts for each barcode as a dictionary

            # retrieve the total number of barcodes
            int_total_num_barcodes = self.bc.meta.n_rows

            def func( self, int_entry_of_axis_for_querying : int, arr_int_entries_of_axis_not_for_querying : np.ndarray, arr_value : np.ndarray ) : # normalize count data of a single feature containing (possibly) multiple barcodes
                """ # 2022-07-06 23:58:38 
                """
                # perform normalization in-place
                for i, e in enumerate( arr_int_entries_of_axis_not_for_querying.astype( int ) ) : # iterate through barcodes
                    arr_value[ i ] = arr_value[ i ] / dict_count[ e ] # perform normalization using the total count data for each barcode
                arr_value *= int_total_count_target

                # perform log1p transformation 
                arr_value = np.log10( arr_value + 1 )

                # calculate deviation
                int_num_records = len( arr_value ) # retrieve the number of records of the current entry
                dict_summary = { 'normalized_log1p_sum' : np.sum( arr_value ) if int_num_records > 30 else sum( arr_value ) } # if an input array has more than 30 elements, use np.sum to calculate the sum
                dict_summary[ 'normalized_log1p_mean' ] = dict_summary[ 'normalized_log1p_sum' ] / int_total_num_barcodes # calculate the mean
                arr_dev = ( arr_value - dict_summary[ 'normalized_log1p_mean' ] ) ** 2 # calculate the deviation
                dict_summary[ 'normalized_log1p_deviation' ] = np.sum( arr_dev ) if int_num_records > 30 else sum( arr_dev )
                dict_summary[ 'normalized_log1p_variance' ] = dict_summary[ 'normalized_log1p_deviation' ] / ( int_total_num_barcodes - 1 ) if int_total_num_barcodes > 1 else np.nan
                return dict_summary    

            # calculate the metric for identifying highly variable genes
            if self.verbose :
                print( f"[RamData.prepare_dimension_reduction_from_raw] [FAST MODE] calculating metrics for highly variable feature detection ... " )
            self.summarize( name_layer_raw, 'feature', func, l_name_col_summarized = [ 'normalized_log1p_sum', 'normalized_log1p_mean', 'normalized_log1p_deviation', 'normalized_log1p_variance' ] )

            # identify highly variable genes
            self.identify_highly_variable_features( 
                name_layer = f'{name_layer_raw}_normalized_log1p', 
                int_num_highly_variable_features = int_num_highly_variable_features, 
                flag_show_graph = True,
                ** dict_kw_hv
            )

            # retrieve variance
            self.ft.meta.load_as_dict( f'{name_layer_raw}_normalized_log1p_variance' )
            dict_var = self.ft.meta.dict[ f'{name_layer_raw}_normalized_log1p_variance' ] # retrieve total counts for each barcode as a dictionary

            # write log-normalized, scaled data for the selected highly variable features
            def func( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) : 
                """ # 2022-07-06 23:58:38 
                """
                # perform normalization 
                arr_value *= int_total_count_target / dict_count[ int_entry_of_axis_for_querying ] # perform normalization using the total count data for each barcode

                # perform log1p transformation 
                arr_value = np.log10( arr_value + 1 )

                # perform scaling in-place
                for i, e in enumerate( arr_int_entries_of_axis_not_for_querying.astype( int ) ) : # iterate through features
                    arr_value[ i ] = arr_value[ i ] / dict_var[ e ] ** 0.5
                arr_value[ arr_value > max_value ] = max_value # capping values above 'max_value'

                # return results
                return int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value
            if self.verbose :
                print( f"[RamData.prepare_dimension_reduction_from_raw] [FAST MODE] write log-normalized, scaled data for the selected highly variable features ... " )
            self.apply( name_layer_raw, name_layer_scaled, func, [ [ 'dense', 'sparse_for_querying_barcodes' ] ] )
        else :
            """ %% SLOW MODE %% """
            self.bc.filter = None # clear barcode filter (in order to preserve every record in the output layers)
            
            # normalize
            self.normalize( 
                name_layer_raw, 
                name_layer_normalized, 
                name_col_total_count = f'{name_layer_raw}_sum',
                int_total_count_target = int_total_count_target,
                mode_instructions = [ [ 'sparse_for_querying_features', 'sparse_for_querying_features' ], [ 'sparse_for_querying_barcodes', 'sparse_for_querying_barcodes' ] ]
            ) 

            # log-transform
            self.apply( 
                name_layer_normalized, 
                name_layer_log_transformed, 
                'log1p',
                mode_instructions = [ [ 'sparse_for_querying_features', 'sparse_for_querying_features' ], [ 'sparse_for_querying_barcodes', 'sparse_for_querying_barcodes' ] ]
            ) 
            
            # load filter for filtered barcodes
            self.bc.change_filter( name_col_filter_filtered_barcode )

            # identify highly variable features (with filtered barcodes)
            self.identify_highly_variable_features( 
                name_layer_log_transformed, 
                int_num_highly_variable_features = int_num_highly_variable_features,
                flag_show_graph = True,
                ** dict_kw_hv
            )

            # scale data (with metrics from the filtered barcodes)
            self.scale( 
                name_layer_log_transformed,
                name_layer_scaled, 
                name_col_variance = f'{name_layer_log_transformed}_variance',
                name_col_mean = f'{name_layer_log_transformed}_mean',
                max_value = max_value,
                mode_instructions = [ [ 'sparse_for_querying_features', 'sparse_for_querying_features' ], [ 'sparse_for_querying_barcodes', 'sparse_for_querying_barcodes' ] ]
            ) # scale data
    ''' utility functions for filter '''
    def change_filter( self, name_col_filter = None, name_col_filter_bc = None, name_col_filter_ft = None ) -> None :
        """ # 2022-07-16 17:27:58 
        retrieve and apply filters for 'barcode' and 'feature' Axes
        
        'name_col_filter_bc', 'name_col_filter_ft' will override 'name_col_filter' when applying filters.
        if all name_cols are invalid, no filters will be retrieved and applied
        """
        # check validity of name_cols for filter
        # bc
        if name_col_filter_bc not in self.bc.meta :
            name_col_filter_bc = name_col_filter if name_col_filter in self.bc.meta else None # use 'name_col_filter' instead if 'name_col_filter_bc' is invalid
        # ft
        if name_col_filter_ft not in self.ft.meta :
            name_col_filter_ft = name_col_filter if name_col_filter in self.ft.meta else None # use 'name_col_filter' instead if 'name_col_filter_ft' is invalid
        
        # apply filters
        self.bc.change_filter( name_col_filter_bc ) # bc
        self.ft.change_filter( name_col_filter_ft ) # ft
    def save_filter( self, name_col_filter = None, name_col_filter_bc = None, name_col_filter_ft = None ) -> None :
        """ # 2022-07-16 17:27:54 
        save filters for 'barcode' and 'feature' Axes
        
        'name_col_filter_bc', 'name_col_filter_ft' will override 'name_col_filter' when saving filters
        for consistency, if filter has not been set, filter containing all active entries (containing valid count data) will be saved instead
        
        if all name_cols are invalid, no filters will be saved
        """
        # save filters
        self.bc.save_filter( name_col_filter if name_col_filter_bc is None else name_col_filter_bc ) # bc
        self.ft.save_filter( name_col_filter if name_col_filter_ft is None else name_col_filter_ft ) # ft
    def change_or_save_filter( self, name_col_filter = None, name_col_filter_bc = None, name_col_filter_ft = None ) -> None :
        """ # 2022-08-07 02:03:53 
        retrieve and apply filters for 'barcode' and 'feature' Axes, and if the filter names do not exist in the metadata and thus cannot be retrieved, save the currently active entries of each axis to its metadata using the given filter name.
        
        'name_col_filter_bc', 'name_col_filter_ft' will override 'name_col_filter' when saving filters
        for consistency, if filter has not been set, filter containing all active entries (containing valid count data) will be saved instead
        
        if all name_cols are invalid, no filters will be saved/retrieved
        """
        # load or save filters ('name_col_filter_bc', 'name_col_filter_ft' get priority over 'name_col_filter')
        self.bc.change_or_save_filter( name_col_filter if name_col_filter_bc is None else name_col_filter_bc ) # bc
        self.ft.change_or_save_filter( name_col_filter if name_col_filter_ft is None else name_col_filter_ft ) # ft
        return
    ''' utility functions for saving/loading models '''
    @property
    def models( self ) :
        """ # 2022-08-05 19:42:00 
        show available models of the RamData
        """
        if 'models' not in self._dict_metadata :
            self._dict_metadata[ 'models' ] = dict( )
            self._save_metadata_( )
        return self._dict_metadata[ 'models' ]
    def load_model( self, name_model, type_model : typing.Literal[ 'ipca', 'pumap' ] ) :
        """ # 2022-08-07 16:55:55 
        load model from RamData
        """
        # if the model does not exists in the RamData, exit
        if type_model not in self._dict_metadata[ 'models' ] :
            return
        if name_model not in self._dict_metadata[ 'models' ][ type_model ] :
            return
        
        # load model only when modifiable ramdata exists (the model should be present in the local storage and should be in the 'modifiable' directory)
        if self._path_folder_ramdata_modifiable is None :
            return
        
        # define a folder for storage of models
        path_folder_models = f"{self._path_folder_ramdata_modifiable}models/" # define a folder to save/load model
        os.makedirs( path_folder_models, exist_ok = True )
        
        def __search_and_download_model_file( name_model_file ) :
            """ # 2022-08-05 22:36:37 
            check availability of models and download model file from the remote location where RamData is being hosted.
            """
            # paths
            path_file_model = f"{self._path_folder_ramdata_modifiable}models/{name_model_file}" 
            if os.path.exists( path_file_model ) : # if the model file exists, return True
                return True
            
            # if file does not exists, try to download from the remote source
            # if the current RamData does not appear to be hosted in the remote location, return None (no remote source to use)
            if 'http' != self._path_folder_ramdata[ : 4 ] :
                return 
            path_file_model_remote = f"{self._path_folder_ramdata}models/{name_model_file}"

            # if the model file appears to be not available from the remote source, return None (no remote file available)
            if http_response_code( path_file_model_remote ) != 200 :
                return None
            # attempts to download the model file
            download_file( path_file_model_remote, path_file_model )
        
        # load model
        if type_model in self._set_type_model_picklable : # handle picklable models
            # define path
            name_model_file = f"{name_model}.{type_model}.pickle"
            path_file_model = f"{path_folder_models}{name_model_file}"
            
            # download the model file
            __search_and_download_model_file( name_model_file )
            
            # exit if the file does not exists
            if not os.path.exists( path_file_model ) :
                return 
                
            model = PICKLE_Read( path_file_model )
        elif type_model == 'pumap' :
            # define paths
            name_model_file = f"{name_model}.pumap.tar.gz"
            path_prefix_model = f"{path_folder_models}{name_model}.pumap"
            path_file_model = path_prefix_model + '.tar.gz'
            
            # download the model file
            __search_and_download_model_file( name_model_file )
            
            # exit if the file does not exists
            if not os.path.exists( path_file_model ) :
                return 
            
            # extract tar.gz
            if not os.path.exists( path_prefix_model ) : # if the model has not been extracted from the tar.gz archive
                tar_extract( path_file_model, path_prefix_model ) # extract tar.gz file of pumap object
            model = pumap.load_ParametricUMAP( path_prefix_model ) # load pumap model
            
            # fix 'load_ParametricUMAP' error ('decoder' attribute does not exist)
            if not hasattr( model, 'decoder' ) : 
                model.decoder = None
        return model # return loaded model
    def save_model( self, model, name_model : str, type_model : typing.Literal[ 'ipca', 'pumap' ] ) :
        """ # 2022-08-07 16:56:07 
        save model to RamData
        
        'model' : input model 
        'name_model' : the name of the model. if the same type of model with the same model name already exists, it will be overwritten
        'type_model' : the type of models. currently [ 'ipca', 'pumap' ], for PCA transformation and UMAP embedding, are supported
        """
        # check validity of the name_model
        assert '/' not in name_model # check validity of 'name_pumap_model'
            
        # save model only when mode != 'r'
        if self._mode == 'r' :
            return
        # save model only when modifiable ramdata exists
        if self._path_folder_ramdata_modifiable is None :
            return
        
        # define a folder for storage of models
        path_folder_models = f"{self._path_folder_ramdata_modifiable}models/" # define a folder to save/load model
        os.makedirs( path_folder_models, exist_ok = True )
        
        # save model
        if type_model in self._set_type_model_picklable : # handle picklable models
            path_file_model = f"{path_folder_models}{name_model}.{type_model}.pickle"
            PICKLE_Write( path_file_model, model )
        elif type_model == 'pumap' :
            path_prefix_model = f"{path_folder_models}{name_model}.pumap"
            path_file_model = path_prefix_model + '.tar.gz'
            model.save( path_prefix_model )
            tar_create( path_file_model, path_prefix_model ) # create tar.gz file of pumap object for efficient retrieval and download
        int_file_size = os.path.getsize( path_file_model ) # retrieve file size of the saved model
        
        # update metadata
        if type_model not in self._dict_metadata[ 'models' ] :
            self._dict_metadata[ 'models' ][ type_model ] = dict( )
        self._dict_metadata[ 'models' ][ type_model ][ name_model ] = int_file_size # record file size of the model # add model to the metadata
        self._save_metadata_( ) # save metadata
        
        # report result
        if self.verbose :
            print( f"{name_model}.{type_model} model saved." )
        return int_file_size # return the number of bytes written
    def delete_model( self, name_model : str, type_model : typing.Literal[ 'ipca', 'pumap' ] ) :
        """ # 2022-08-05 19:44:23 
        delete model of the RamData if the model exists in the RamData
        
        'name_model' : the name of the model. if the same type of model with the same model name already exists, it will be overwritten
        'type_model' : the type of models. currently [ 'ipca', 'pumap' ], for PCA transformation and UMAP embedding, are supported
        """
        # check validity of the name_model
        assert '/' not in name_model # check validity of 'name_pumap_model'
            
        # save model only when mode != 'r'
        if self._mode == 'r' :
            return
        # save model only when modifiable ramdata exists
        if self._path_folder_ramdata_modifiable is None :
            return
        
        # if the model does not exist in the current RamData, exit
        if type_model not in self._dict_metadata[ 'models' ] or name_model not in self._dict_metadata[ 'models' ][ type_model ] :
            return
        
        # define a folder for storage of models
        path_folder_models = f"{self._path_folder_ramdata_modifiable}models/" # define a folder to save/load model
        os.makedirs( path_folder_models, exist_ok = True )
        
        # save model
        if type_model in self._set_type_model_picklable : # handle picklable models 
            path_file_model = f"{path_folder_models}{name_model}.{type_model}.pickle"
        elif type_model == 'pumap' :
            path_prefix_model = f"{path_folder_models}{name_model}.pumap"
            path_file_model = path_prefix_model + '.tar.gz'
            # if an extracted folder exists, delete the folder
            if os.path.exists( path_prefix_model ) :
                shutil.rmtree( path_prefix_model )
        int_file_size = os.path.getsize( path_file_model ) # retrieve file size of the saved model
        os.remove( path_file_model )
        
        # update metadata
        del self._dict_metadata[ 'models' ][ type_model ][ name_model ] # delete model from the metadata
        if len( self._dict_metadata[ 'models' ][ type_model ] ) == 0 : # if list is empty, delete the list, too
            del self._dict_metadata[ 'models' ][ type_model ]
        self._save_metadata_( ) # save metadata
        
        # report result
        if self.verbose :
            print( f"{name_model}.{type_model} model deleted." )
        return int_file_size # return the number of bytes of the deleted model file
    ''' utility functions for retrieving expression values from layer and save them as metadata in axis  '''
    def get_expr( self, name_layer : str, queries, name_new_col = None, axis : Union[ int, str ] = 'barcodes' ) :
        """ # 2022-08-21 16:12:56
        retrieve expression values of given 'queries' from the given layer 'name_layer' in the given axis 'axis' (with currently active filters), 
        and save the total expression values for each entry in the axis metadata using the new column name 'name_new_col'
        
        possible usages: (1) calculating gene_set/pathway activities across cells, which can subsequently used for filtering cells for subclustering
                         (2) calculating pseudo-bulk expression profiles of a subset of cells across all active features 
        
        'name_layer' : name of the layer to retrieve data
        'queries' : queries that will be handed to RamDataAxis.__getitem__ calls. A slice, a bitarray, a single or a list of integer representation(s), a single or a list of string representation(s), boolean array are one of the possible inputs
        'name_new_col' : the name of the new column that will contains retrieved expression values in the metadata of the given axis. if None, does not 
        'axis' : axis for not querying, to which expression values will be summarized. { 0 or 'barcodes' } for operating on the 'barcodes' axis, and { 1 or 'features' } for operating on the 'features' axis
        """
        # handle inputs
        flag_axis_for_querying_is_barcode = axis not in { 0, 'barcode', 'barcodes' } # retrieve a flag indicating whether the data is summarized for each barcode or not
        
        # retrieve the appropriate Axis object
        ax_for_querying, ax_not_for_querying = ( self.bc, self.ft ) if flag_axis_for_querying_is_barcode else ( self.ft, self.bc )
        
        # handle invalid layer
        if name_layer not in self.layers :
            if self.verbose :
                print( f"[RamData.get_expr] the given layer '{name_layer}' does not exist" )
            return 
        self.layer = name_layer # load the target layer
        # retrieve appropriate rtx object
        rtx = ram.layer.get_ramtx( flag_is_for_querying_features = not flag_axis_for_querying_is_barcode )
        # handle when appropriate RAMtx object does not exist
        if rtx is None :
            if self.verbose :
                print( f"[RamData.get_expr] RAMtx appropriate for the given axis does not exist" )
            return
        
        # parse query
        l_int_entry_query = bk.BA.to_integer_indices( ax_for_querying[ queries ] ) # retrieve bitarray of the queried entries, convert to list of integer indices
        
        ax_not_for_querying.backup_view( ) # back-up current view and reset the view of the axis not for querying
        mtx = rtx.get_sparse_matrix( l_int_entry_query ) # retrieve expr matrix of the queries in sparse format
        ax_not_for_querying.restore_view( ) # restore vies
        
        arr_expr = np.array( ( mtx[ l_int_entry_query ] if flag_axis_for_querying_is_barcode else mtx[ :, l_int_entry_query ].T ).sum( axis = 0 ) )[ 0 ] # retrieve summarized expression values of the queried entries # convert it to numpy array of shape (len_axis_not_for_querying, )
        
        if name_new_col is not None : # if valid 'name_new_col' column name has been given, save the retrieved data as a metadata column
            ax_not_for_querying.meta[ name_new_col, : ] = arr_expr
        return arr_expr # return retrieved expression values
    ''' memory-efficient PCA '''
    def train_pca( self, name_model = 'ipca', name_layer = 'normalized_log1p', int_num_components = 50, int_num_barcodes_in_ipca_batch = 50000, name_col_filter = 'filter_pca', float_prop_subsampling = 1, name_col_filter_subsampled = 'filter_pca_subsampled', flag_ipca_whiten = False, int_num_threads = 3, flag_show_graph = True ) :
        """ # 2022-08-08 12:01:00 
        Perform incremental PCA in a very memory-efficient manner.
        the resulting incremental PCA model will be saved in the RamData models database.
        
        arguments:
        'name_model' : the trained incremental PCA model will be saved to RamData models database with this name. if None is given, the model will not be saved.
        'name_layer' : name of the data source layer (the layer from which gene expression data will be retrieved for the barcodes)
        'name_col' : 'name_col' of the PCA data that will be added to Axis.meta ZDF.
        'name_col_filter' : the name of 'feature'/'barcode' Axis metadata column to retrieve selection filter for highly-variable-features. (default: None) if None is given, current feature filter (if it has been set) will be used as-is. if a valid filter is given, filter WILL BE CHANGED.
        'name_col_filter_subsampled' : the name of 'feature'/'barcode' Axis metadata column to retrieve or save mask containing subsampled barcodes. if 'None' is given and 'float_prop_subsampling' is below 1 (i.e. subsampling will be used), the subsampling filter generated for retrieving gene expression data of selected barcodes will not be saved.
        'int_num_components' : number of PCA components.
        'int_num_barcodes_in_ipca_batch' : number of barcodes in an Incremental PCA computation
        'float_prop_subsampling' : proportion of barcodes to used to train representation of single-barcode data using incremental PCA. 1 = all barcodes, 0.1 = 10% of barcodes, etc. subsampling will be performed using a random probability, meaning the actual number of barcodes subsampled will not be same every time.
        'flag_ipca_whiten' : a flag for an incremental PCA computation (Setting this flag to 'True' will reduce the efficiency of model learning, but might make the model more generalizable)
        'int_num_threads' : number of threads for parallel data retrieval/iPCA transformation/ZarrDataFrame update. 3~5 would be ideal. should be larger than 2
        'flag_show_graph' : show graph
        """
        """
        1) Prepare
        """
        # check the validility of the input arguments
        if name_layer not in self.layers :
            if self.verbose :
                print( f"[ERROR] [RamData.train_pca] invalid argument 'name_layer' : '{name_layer}' does not exist." )
            return -1 
        # set layer
        self.layer = name_layer

        # retrieve RAMtx object (sorted by barcodes) to summarize # retrieve 'Barcode' Axis object
        rtx, ax = self.layer.get_ramtx( flag_is_for_querying_features = False ), self.bc
        if rtx is None :
            if self.verbose :
                print( f"[ERROR] [RamData.train_pca] valid ramtx object is not available in the '{self.layer.name}' layer" )

        # set/save filter
        if name_col_filter is not None :
            self.change_or_save_filter( name_col_filter )
        
        # create view for 'feature' Axis
        self.ft.create_view( )
        
        # retrieve a flag indicating whether a subsampling is active
        flag_is_subsampling_active = ( name_col_filter_subsampled in self.bc.meta ) or ( float_prop_subsampling is not None and float_prop_subsampling < 1 ) # perform subsampling if 'name_col_filter_subsampled' is valid or 'float_prop_subsampling' is below 1
        
        # if a subsampling is active, retrieve a filter containing subsampled barcodes and apply the filter to the 'barcode' Axis
        if flag_is_subsampling_active :
            # retrieve barcode filter before subsampling
            ba_filter_bc_before_subsampling = self.bc.filter

            # set barcode filter after subsampling
            if name_col_filter_subsampled in self.bc.meta : # if 'name_col_filter_subsampled' barcode filter is available, load the filter
                self.bc.change_filter( name_col_filter_subsampled )
            else : # if the 'name_col_filter_subsampled' barcode filter is not available, build a filter containing subsampled entries and save the filter
                self.bc.filter = self.bc.subsample( float_prop_subsampling = float_prop_subsampling ) 
                self.bc.save_filter( name_col_filter_subsampled )

        # create view for 'barcode' Axis
        self.bc.create_view( ) 
        """
        2) Fit PCA with/without subsampling of barcodes
        """
        # initialize iPCA object
        ipca = skl.decomposition.IncrementalPCA( n_components = int_num_components, batch_size = int_num_barcodes_in_ipca_batch, copy = False, whiten = flag_ipca_whiten ) # copy = False to increase memory-efficiency

        # define functions for multiprocessing step
        def process_batch( batch, pipe_to_main_process ) :
            ''' # 2022-07-13 22:18:15 
            prepare data as a sparse matrix for the batch
            '''
            # parse the received batch
            int_num_of_previously_returned_entries, l_int_entry_current_batch = batch[ 'int_num_of_previously_returned_entries' ], batch[ 'l_int_entry_current_batch' ]
            int_num_retrieved_entries = len( l_int_entry_current_batch )

            pipe_to_main_process.send( ( int_num_of_previously_returned_entries, int_num_retrieved_entries, rtx.get_sparse_matrix( l_int_entry_current_batch )[ int_num_of_previously_returned_entries : int_num_of_previously_returned_entries + int_num_retrieved_entries ] ) ) # retrieve and send sparse matrix as an input to the incremental PCA # resize sparse matrix
        pbar = progress_bar( total = ax.meta.n_rows ) # initialize the progress bar
        def post_process_batch( res ) :
            """ # 2022-07-13 22:18:18 
            perform partial fit for batch
            """
            int_num_of_previously_returned_entries, int_num_retrieved_entries, X = res # parse the result
            try :
                ipca.partial_fit( X.toarray( ) ) # perform partial fit using the retrieved data # partial_fit only supports dense array
            except ValueError : # handles 'ValueError: n_components=50 must be less or equal to the batch number of samples 14.' error # 2022-07-18 15:09:52 
                if self.verbose :
                    print( f'current batch contains less than {int_num_components} number of barcodes, which is incompatible with iPCA model. therefore, current batch will be skipped.' )
            pbar.update( int_num_retrieved_entries ) # update the progress bar once the training has been completed
            
            if self.verbose : # report
                print( f'fit completed for {int_num_of_previously_returned_entries + 1}-{int_num_of_previously_returned_entries + int_num_retrieved_entries} barcodes' )
        # fit iPCA using multiple processes
        bk.Multiprocessing_Batch( ax.batch_generator( int_num_entries_for_batch = int_num_barcodes_in_ipca_batch ), process_batch, post_process_batch = post_process_batch, int_num_threads = max( min( int_num_threads, 3 ), 2 ), int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop = 0.2 ) # number of threads for multi-processing is 2 ~ 5 # generate batch with fixed number of barcodes
        pbar.close( ) # close the progress bar
        
        # report
        if self.verbose :
            print( 'fit completed' )
        # fix error of ipca object
        if not hasattr( ipca, 'batch_size_' ) :
            ipca.batch_size_ = ipca.batch_size # 'batch_size_' attribute should be set for 'transform' method to work..
        
        # if subsampling has been completed, revert to the original barcode selection filter
        if flag_is_subsampling_active :
            self.bc.filter = ba_filter_bc_before_subsampling
            del ba_filter_bc_before_subsampling

        # destroy the view
        self.destroy_view( )
        
        # save model
        if name_model is not None : # if the given 'name_model' is valid 
            self.save_model( ipca, name_model, 'ipca' ) # save model
        
        # draw graphs
        if flag_show_graph :
            # draw 'explained variance ratio' graph
            fig, ax = plt.subplots( 1, 1 )
            ax.plot( ipca.explained_variance_ratio_, 'o-' )
            bk.MATPLOTLIB_basic_configuration( x_label = 'principal components', y_label = 'explained variance ratio', title = 'PCA result', show_grid = True )
        
        return ipca # return the model
    def apply_pca( self, name_model = 'ipca', name_layer = 'normalized_log1p', name_col = 'X_pca', name_col_filter = 'filter_pca', int_n_components_in_a_chunk = 20, int_num_threads = 5 ) :
        """ # 2022-08-08 22:51:46 
        Apply trained incremental PCA in a memory-efficient manner.
        
        arguments:
        'name_model' : the trained incremental PCA model will be saved to RamData.ns database with this name. if None is given, the model will not be saved.
        'name_layer' : name of the data source layer (the layer from which gene expression data will be retrieved for the barcodes)
        'name_col' : 'name_col' of the PCA data that will be added to Axis.meta ZDF.
        'name_col_filter' : the name of 'feature'/'barcode' Axis metadata column to retrieve selection filter for highly-variable-features. (default: None) if None is given, current feature filter (if it has been set) will be used as-is. if a valid filter is given, filter WILL BE CHANGED.
        
        'int_n_components_in_a_chunk' : deterimines the chunk size for PCA data store
        'int_num_threads' : the number of threads to use for parellel processing. the larger the number of threads are, the larger memory consumed by all the workers.
        """
        """
        1) Prepare
        """
        # check the validility of the input arguments
        if name_layer not in self.layers :
            if self.verbose :
                print( f"[ERROR] [RamData.apply_pca] invalid argument 'name_layer' : '{name_layer}' does not exist." )
            return -1 
        # set layer
        self.layer = name_layer

        # retrieve RAMtx object (sorted by barcodes) to summarize # retrieve 'Barcode' Axis object
        rtx, ax = self.layer.get_ramtx( flag_is_for_querying_features = False ), self.bc
        if rtx is None :
            if self.verbose :
                print( f"[ERROR] [RamData.apply_pca] valid ramtx object is not available in the '{self.layer.name}' layer" )

        # set filters
        if name_col_filter is not None :
            self.change_filter( name_col_filter )
        
        # create view of the RamData
        self.create_view( )

        # exit if the model does not exist
        ipca = self.load_model( name_model, 'ipca' )
        if ipca is None :
            if self.verbose :
                print( f"[ERROR] [RamData.apply_pca] iPCA model '{name_model}' does not exist in the RamData models database" )
            return

        # prepare pca column in the metadata
        ax.meta.initialize_column( name_col, dtype = np.float64, shape_not_primary_axis = ( ipca.n_components, ), chunks = ( int_n_components_in_a_chunk, ), categorical_values = None ) # initialize column
        
        """
        2) Transform Data
        """
        # define functions for multiprocessing step
        def process_batch( batch, pipe_to_main_process ) :
            ''' # 2022-07-13 22:18:22 
            retrieve data and retrieve transformed PCA values for the batch
            '''
            # parse the received batch
            int_num_processed_records, int_num_of_previously_returned_entries, l_int_entry_current_batch = batch[ 'int_accumulated_weight_current_batch' ], batch[ 'int_num_of_previously_returned_entries' ], batch[ 'l_int_entry_current_batch' ]
            int_num_retrieved_entries = len( l_int_entry_current_batch )

            pipe_to_main_process.send( ( int_num_processed_records, l_int_entry_current_batch, rtx.get_sparse_matrix( l_int_entry_current_batch )[ int_num_of_previously_returned_entries : int_num_of_previously_returned_entries + int_num_retrieved_entries ] ) ) # retrieve data as a sparse matrix and send the result of PCA transformation # send the integer representations of the barcodes for PCA value update
        pipe_sender, pipe_receiver = mp.Pipe( ) # create a communication link between the main process and the worker for saving zarr objects
        pbar = progress_bar( total = rtx.get_total_num_records( int_num_entries_for_each_weight_calculation_batch = self.int_num_entries_for_each_weight_calculation_batch, flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx = self.flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx ) )
        def post_process_batch( res ) :
            """ # 2022-07-13 22:18:26 
            perform PCA transformation for each batch
            """
            # parse result 
            int_num_processed_records, l_int_entry_current_batch, X = res
            
            X_transformed = ipca.transform( X ) # perform PCA transformation
            del X
            
            pbar.update( int_num_processed_records ) # update the progress bar
            
            # send result to the worker
            pipe_sender.send( ( int_num_processed_records, l_int_entry_current_batch, X_transformed ) )
        # start the worker 
        def __worker_for_saving_zarr( pipe_receiver ) :
            """ # 2022-08-08 18:00:05 
            save transformed PCA components to the metadata for each batch
            """
            while True :
                res = pipe_receiver.recv( )
                # terminate if None is received
                if res is None :
                    break
                
                int_num_processed_records, l_int_entry_current_batch, X_transformed = res # parse the result

                # update the PCA components for the barcodes of the current batch
                ax.meta[ name_col, l_int_entry_current_batch ] = X_transformed
        p = mp.Process( target = __worker_for_saving_zarr, args = ( pipe_receiver, ) )
        p.start( )

        # transform values using iPCA using multiple processes
        bk.Multiprocessing_Batch( rtx.batch_generator( ax.filter, int_num_entries_for_each_weight_calculation_batch = self.int_num_entries_for_each_weight_calculation_batch, int_total_weight_for_each_batch = self.int_total_weight_for_each_batch, flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx = self.flag_use_total_number_of_entries_of_axis_not_for_querying_as_weight_for_dense_ramtx ), process_batch, post_process_batch = post_process_batch, int_num_threads = int_num_threads, int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop = 0.2 ) 
        pbar.close( ) # close the progress bar
        # dismiss the worker
        pipe_sender.send( None ) # send the termination signal
        p.join( )
        
        # destroy the view
        self.destroy_view( )
        return 
    ''' memory-efficient UMAP '''
    def train_umap( self, name_col_pca = 'X_pca', int_num_components_pca = 20, int_num_components_umap = 2, name_col_filter : Union[ str, None ] = 'filter_umap', name_pumap_model = 'pumap', name_pumap_model_new : Union[ str, None ] = None, dict_kw_pumap : dict = { 'metric' : 'euclidean' } ) :
        """ # 2022-08-07 11:13:38 
        Perform Parametric UMAP to embed cells in reduced dimensions for a scalable analysis of single-cell data
        
        * Parametric UMAP has several advantages over non-parametric UMAP (conventional UMAP), which are 
            (1) GPU can be utilized during training of neural network models
            (2) learned embedding can be applied to other cells not used to build the embedding
            (3) learned embedding can be updated by training with additional cells
        Therefore, parametric UMAP is suited for generating embedding of single-cell data with extremely large number of cells

        arguments:
        'name_col_pca' : 'name_col' of the columns containing PCA transformed values.
        'int_num_components_pca' : number of PCA components to use as inputs for Parametric UMAP learning
        'name_col_umap' : 'name_col' of the columns containing UMAP transformed values.
        'int_num_components_umap' : number of output UMAP components. (default: 2)
        'name_col_filter' : the name of 'feature'/'barcode' Axis metadata column to retrieve selection filter for highly-variable-features. (default: None) if None is given, current feature filter (if it has been set) will be used as-is. if a valid filter is given, filter WILL BE CHANGED.
        'name_pumap_model' = 'pumap' : the name of the parametric UMAP model. if None is given, the trained model will not be saved to the RamData object. if the model already exists, the model will be loaded and trained again.
        'name_pumap_model_new' = 'pumap' : the name of the new parametric UMAP model after the training. if None is given, the new model will not be saved. if 'name_pumap_model' and 'name_pumap_model_new' are the same, the previously written model will be overwritten.
        'dict_kw_pumap' : remaining keyworded arguments of umap.ParametricUMAP
        
        """
        """
        1) Prepare
        """
        # handle arguments
        if name_pumap_model_new is None :
            name_pumap_model_new = name_pumap_model
        # retrieve 'Barcode' Axis object
        ax = self.bc

        # set/save filter
        if name_col_filter is not None :
            self.change_or_save_filter( name_col_filter )
        """
        2) Train Parametric UMAP 
        """
        # load pumap model
        pumap_embedder = self.load_model( name_pumap_model, 'pumap' )
        if pumap_embedder is None :
            pumap_embedder = pumap.ParametricUMAP( low_memory = True, n_components = int_num_components_umap, ** dict_kw_pumap ) # load an empty model if a saved model is not available

        # train parametric UMAP model
        pumap_embedder.fit( self.bc.meta[ name_col_pca, None, : int_num_components_pca ] )

        # report
        if self.verbose :
            print( f'[Info] [RamData.train_umap] training for {ax.meta.n_rows} entries completed' )
        
        # save the model
        int_model_file_size = self.save_model( pumap_embedder, name_pumap_model_new, 'pumap' )
        if int_model_file_size is not None :
            # report the file size of the model if saving of the model was successful
            if self.verbose :
                print( f'[Info] [RamData.train_umap] Parametric UMAP model of {int_model_file_size} Bytes has been saved.' )
        return pumap_embedder # return the model
    def apply_umap( self, name_col_pca : str = 'X_pca', name_col_umap : str = 'X_umap', int_num_barcodes_in_pumap_batch : int = 20000, name_col_filter : Union[ str, None ] = 'filter_umap', name_pumap_model : Union[ str, None ] = 'pumap' ) :
        """ # 2022-08-07 11:27:20 
        Embed barcodes to lower-dimensional space using the trained Parametric UMAP in a scalable way
        
        * Parametric UMAP has several advantages over non-parametric UMAP (conventional UMAP), which are 
            (1) GPU can be utilized during training of neural network models
            (2) learned embedding can be applied to other cells not used to build the embedding
            (3) learned embedding can be updated by training with additional cells
        Therefore, parametric UMAP is suited for generating embedding of single-cell data with extremely large number of cells

        arguments:
        'name_col_pca' : 'name_col' of the columns containing PCA transformed values.
        'name_col_umap' : 'name_col' of the columns containing UMAP transformed values.
        'int_num_barcodes_in_pumap_batch' : number of barcodes in a batch for Parametric UMAP model update.
        'name_col_filter' : the name of 'feature'/'barcode' Axis metadata column to retrieve selection filter for highly-variable-features. (default: None) if None is given, current feature filter (if it has been set) will be used as-is. if a valid filter is given, filter WILL BE CHANGED.
        'name_pumap_model' = 'pumap' : the name of the parametric UMAP model. if None is given, the trained model will not be saved to the RamData object. if the model already exists, the model will be loaded and trained again.
        """
        """
        1) Prepare
        """
        # # retrieve 'Barcode' Axis object
        ax = self.bc

        # set filters
        if name_col_filter is not None :
            self.change_filter( name_col_filter )
        
        # load the model
        pumap_embedder = self.load_model( name_pumap_model, 'pumap' ) # load the model
        if pumap_embedder is None :
            if self.verbose :
                print( f"[Error] [RamData.apply_umap] the parametric UMAP model {name_pumap_model} does not exist in the current RamData, exiting" )
            return
        # retrieve the number of pca components for the input of pumap model
        int_num_components_pca = pumap_embedder.dims[ 0 ]
        if ax.meta.get_shape( name_col_pca )[ 0 ] < int_num_components_pca : # check compatibility between the given PCA data and the given pumap model # if the number of input PCA components is larger than the components available in the input PCA column, exit
            if self.verbose :
                print( f"[Error] [RamData.apply_umap] the number of PCA components of the given parametric UMAP model {name_pumap_model} is {int_num_components_pca}, which is larger than the number of PCA components available in {name_col_pca} data in the 'barcode' metadata, exiting" )
            return
            
        """
        2) Transform Data
        """
        pbar = progress_bar( total = ax.meta.n_rows ) # initialize the progress bar
        # iterate through batches
        for batch in ax.batch_generator( int_num_entries_for_batch = int_num_barcodes_in_pumap_batch ) :
            l_int_entry_current_batch = batch[ 'l_int_entry_current_batch' ] # parse batch
            int_num_retrieved_entries = len( l_int_entry_current_batch ) # retrieve the number of retrieve entries
            pbar.update( int_num_retrieved_entries ) # update the progress bar
            
            # retrieve UMAP embedding of barcodes of the current batch
            X_transformed = pumap_embedder.transform( self.bc.meta[ name_col_pca, l_int_entry_current_batch, : int_num_components_pca ] ) 

            # update the components for the barcodes of the current batch
            ax.meta[ name_col_umap, l_int_entry_current_batch ] = X_transformed
        pbar.close( ) # close the progress bar

        return pumap_embedder # return the model
    ''' for community detection '''
    def hdbscan( self, name_model : str = 'hdbscan', name_col_data : str = 'X_umap', int_num_components_data : int = 2, name_col_label : str = 'hdbscan', min_cluster_size : int = 30, min_samples : int = 30, cut_distance: float = 0.15, flag_reanalysis_of_previous_clustering_result : bool = False, name_col_filter : Union[ str, None ] = 'filter_hdbscan', name_col_embedding : Union[ str, None ] = 'X_umap', dict_kw_scatter : dict = { 's' : 10, 'linewidth' : 0, 'alpha' : 0.05 }, index_col_of_name_col_label : Union[ int, None ] = None ) :
        """ # 2022-08-09 02:19:26 
        Perform HDBSCAN for the currently active barcodes

        arguments:
        'name_model' : name of the model saved/will be saved in RamData.models database. if the model already exists, 'cut_distance' and 'min_cluster_size' arguments will become active.
        
        === data input ===
        'name_col_data' : 'name_col' of the column containing data. UMAP embeddings are recommended (PCA data is not recommended as an input to HDBSCAN clustering, since it is much more sparse and noisy than UMAP embedded data)
        'int_num_components_data' : number of components of the data for clustering (default: 2)

        === clustering arguments ===
        'min_cluster_size', 'min_samples' : arguments for HDBSCAN method. please refer to the documentation of HDBSCAN (https://hdbscan.readthedocs.io/)
        'cut_distance' and 'min_cluster_size' : arguments for the re-analysis of the clustering result for retrieving more fine-grained/coarse-grained cluster labels (for more info., please refer to hdbscan.HDBSCAN.single_linkage_tree_.get_clusters docstring). 
        'flag_reanalysis_of_previous_clustering_result' : if 'flag_reanalysis_of_previous_clustering_result' is True and 'name_model' exists in the RamData.ns database, use the hdbscan model saved in the database to re-analyze the previous hierarchical DBSCAN clustering result. 'cut_distance' and 'min_cluster_size' arguments can be used to re-analyze the clustering result and retrieve more fine-grained/coarse-grained cluster labels (for more info., please refer to hdbscan.HDBSCAN.single_linkage_tree_.get_clusters docstring). To perform hdbscan from the start, change name_model to a new name or delete the model from RamData.ns database
        
        === output ===
        'name_col_label' : 'name_col' of the axis metadata that will contain cluster labels assigned by the current clustering algorithm
        'index_col_of_name_col_label' : index of the secondary axis of the column of 'name_col_label' that will contain cluster labels. if None, 'name_col_label' is assumed to be a 1-dimensional column. 
        
        === cell filter ===
        'name_col_filter' : the name of 'feature'/'barcode' Axis metadata column to retrieve selection filter for running the current method. if None is given, current barcode/feature filters (if it has been set) will be used as-is.
        
        === settings for drawing graph ===
        'name_col_embedding' : 'name_col' of the column containing the embeddings for the visualization of the clustering results. if None is given, the graph will not be drawn
        'dict_kw_scatter' : arguments for 'matplotlib Axes.scatter' that will be used for plotting

        returns:
        arr_cluster_label, clusterer(hdbscan object)
        """
        """
        1) Prepare
        """
        # # retrieve 'Barcode' Axis object
        ax = self.bc

        # set filters for operation
        if name_col_filter is not None :
            self.change_or_save_filter( name_col_filter )

        """
        2) Train model and retrieve cluster labels
        """
        # load the model and retrieve cluster labels
        type_model = 'hdbscan'
        clusterer = self.load_model( name_model, type_model )
        if clusterer is None : # if the model does not exist, initiate the model
            clusterer = hdbscan.HDBSCAN( min_cluster_size = min_cluster_size, min_samples = min_samples ) # initiate the model
            clusterer.fit( self.bc.meta[ name_col_data, None, : int_num_components_data ] ) # retrieve data # clustering embedded barcodes
            arr_cluster_label = clusterer.labels_ # retrieve cluster labels
            # save trained model
            if name_model is not None : # check validity of 'name_model' 
                self.save_model( clusterer, name_model, type_model ) # save model to the RamData
        else : # if 'name_model' hdbscan model exists in the database, use the previously computed clustering results
            if flag_reanalysis_of_previous_clustering_result : # if 'flag_reanalysis_of_previous_clustering_result' is True, perform re-analysis of the clustering result
                arr_cluster_label = clusterer.single_linkage_tree_.get_clusters( cut_distance = cut_distance, min_cluster_size = min_cluster_size ) # re-analyze previous clustering result, and retrieve cluster labels
            else :
                arr_cluster_label = clusterer.labels_ # retrieve previously calculated cluster labels
                
        """
        3) save labels
        """
        if index_col_of_name_col_label is None : # assume 'name_col_label' is 1-dimensional column
            # update all columns
            ax.meta[ name_col_label ] = arr_cluster_label
        else :
            # update a single column in the meatadata column 'name_col_label'
            ax.meta[ name_col_label, None, index_col_of_name_col_label ] = arr_cluster_label
        
        # report
        if self.verbose :
            print( f'[Info] [RamData.hdbscan] clustering completed for {ax.meta.n_rows} number of barcodes' )

        # draw graphs
        if name_col_embedding is not None : # visualize clustering results if 'name_col_embedding' has been given
            color_palette = sns.color_palette( 'Paired', len( set( arr_cluster_label ) ) )
            cluster_colors = [ color_palette[ x ] if x >= 0 else ( 0.5, 0.5, 0.5 ) for x in arr_cluster_label ]
            fig, plt_ax = plt.subplots( 1, 1, figsize = ( 7, 7 ) )
            plt_ax.scatter( * self.bc.meta[ name_col_embedding, None, : 2 ].T, c = cluster_colors, ** dict_kw_scatter ) # retrieve embedding data and draw the graph
            
        # return results
        return arr_cluster_label, clusterer # return the trained model and computed cluster labels
    def leiden( self, name_model : str = 'leiden', name_col_data : str = 'X_pca', int_num_components_data : int = 15, name_col_label : str = 'leiden', resolution: float = 0.2, int_num_clus_expected : Union[ int, None ] = None, directed: bool = True, use_weights: bool = True, dict_kw_leiden_partition : dict = { 'n_iterations' : -1, 'seed' : 0 }, dict_kw_pynndescent_transformer : dict = { 'n_neighbors' : 10, 'metric' : 'euclidean', 'low_memory' : True }, name_col_filter : Union[ str, None ] = 'filter_leiden', name_col_embedding : Union[ str, None ] = 'X_umap', dict_kw_scatter : dict = { 's' : 10, 'linewidth' : 0, 'alpha' : 0.05 }, index_col_of_name_col_label : Union[ int, None ] = None ) -> None :
        """ # 2022-08-09 02:19:31 
        Perform leiden community detection algorithm (clustering) for the currently active barcodes

        arguments:
        'name_model' : name of the model saved/will be saved in RamData.models database. if the model already exists, 'cut_distance' and 'min_cluster_size' arguments will become active.
        
        === data input ===
        'name_col_data' : 'name_col' of the column containing data. PCA data is recommended.
        'int_num_components_data' : number of components of the data for clustering (default: 2)
        
        === output ===
        'name_col_label' : 'name_col' of the axis metadata that will contain cluster labels assigned by the current clustering algorithm
        'index_col_of_name_col_label' : index of the secondary axis of the column of 'name_col_label' that will contain cluster labels. if None, 'name_col_label' is assumed to be a 1-dimensional column. 
        
        === clustering arguments ===
        'resolution' : initial resolution of cluster. please refer to 'resolution_parameter' of 'leidenalg.find_partition' method
        'int_num_clus_expected' : the expected number of clusters in the data to optimize hyperparameters for community detection. if 'int_num_clus_expected' is not None, until the number of detected communities reaches 'int_num_clus_expected', the 'resolution' parameter will be optimized. this argument will be inactive when 'resolution' is None.
        'directed' : create directed graph. it is recommended to set it to True
        'use_weights' : use weights of the kNN graph for the leiden partitioning
        'dict_kw_leiden_partition' : a dictionary containing keyworded arguments for the 'leidenalg.find_partition' method
        'dict_kw_pynndescent_transformer' : a dictionary containing keyworded arguments for the 'pynndescent.PyNNDescentTransformer' method for constructing kNN graph from the data
        
        === cell filter ===
        'name_col_filter' : the name of 'feature'/'barcode' Axis metadata column to retrieve selection filter for running the current method. if None is given, current barcode/feature filters (if it has been set) will be used as-is.
        
        === settings for drawing graph ===
        'name_col_embedding' : 'name_col' of the column containing the embeddings for the visualization of the clustering results. if None is given, the graph will not be drawn
        'dict_kw_scatter' : arguments for 'matplotlib Axes.scatter' that will be used for plotting

        returns:
        """
        """
        1) Prepare
        """
        # # retrieve 'Barcode' Axis object
        ax = self.bc

        # set filters for operation
        if name_col_filter is not None :
            self.change_or_save_filter( name_col_filter )

        """
        2) construct kNN graph
        """
        # load the knn graph
        type_model = 'knngraph'
        conn = self.load_model( name_model, type_model )
        if conn is None : # if the knngraph does not exist, calculate the knngraph
            knnmodel = pynndescent.PyNNDescentTransformer( ** dict_kw_pynndescent_transformer )
            conn = knnmodel.fit_transform( ax.meta[ name_col_data, None, : int_num_components_data ] )
                        
            # save calculated knngraph
            if name_model is not None : # check validity of 'name_model' 
                self.save_model( conn, name_model, type_model ) # save knngraph to the RamData
            
        """
        3) perform leiden clustering
        """
        # construct an igraph object from the knn graph
        def get_igraph_from_adjacency( adjacency, directed = None ) :
            """ # 2022-08-09 02:28:09 
            Get igraph graph from adjacency matrix.
            this code is mostly a copy of a function implemented in scanpy 'https://github.com/scverse/scanpy/blob/536ed15bc73ab5d1131c0d530dd9d4f2dc9aee36/scanpy/_utils/__init__.py'
            """
            import igraph as ig

            sources, targets = adjacency.nonzero( )
            weights = adjacency[ sources, targets ]
            if isinstance( weights, np.matrix ) :
                weights = weights.A1
            g = ig.Graph( directed = directed )
            g.add_vertices( adjacency.shape[ 0 ] )  # this adds adjacency.shape[0] vertices
            g.add_edges( list( zip( sources, targets ) ) ) 
            try:
                g.es[ 'weight' ] = weights
            except KeyError:
                pass
            if g.vcount() != adjacency.shape[ 0 ]:
                if self.verbose :
                    print( f"The constructed graph has only {g.vcount( )} nodes. Your adjacency matrix contained redundant nodes." )
            return g
        g = get_igraph_from_adjacency( conn, directed )
        del conn
        if self.verbose :
            print( f'[Info] [RamData.leiden] knn-graph loaded' )

        # compose partition arguments
        if resolution is not None :
            dict_kw_leiden_partition[ 'resolution_parameter' ] = resolution
        if use_weights :
            dict_kw_leiden_partition[ 'weights' ] = np.array( g.es[ 'weight' ] ).astype( np.float64 )
            

        while True :
            # perform leiden clustering
            arr_cluster_label = np.array( leidenalg.find_partition( g, leidenalg.RBConfigurationVertexPartition, ** dict_kw_leiden_partition ).membership )
            
            # until the desired 
            if resolution is not None and int_num_clus_expected is not None and len( set( arr_cluster_label ) ) < int_num_clus_expected :
                dict_kw_leiden_partition[ 'resolution_parameter' ] *= 1.2
                if self.verbose :
                    print( f"[Info] [RamData.leiden] resolution increased to {dict_kw_leiden_partition[ 'resolution_parameter' ]}" )
            else :
                break
        del g
                
        """
        4) save labels
        """
        if index_col_of_name_col_label is None : # assume 'name_col_label' is 1-dimensional column
            # update all columns
            ax.meta[ name_col_label ] = arr_cluster_label
        else :
            # update a single column in the meatadata column 'name_col_label'
            print( name_col_label, None, index_col_of_name_col_label )
            ax.meta[ name_col_label, None, index_col_of_name_col_label ] = arr_cluster_label
            
        # report
        if self.verbose :
            print( f'[Info] [RamData.leiden] clustering completed for {ax.meta.n_rows} number of barcodes' )

        # draw graphs
        if name_col_embedding is not None : # visualize clustering results if 'name_col_embedding' has been given
            color_palette = sns.color_palette( 'Paired', len( set( arr_cluster_label ) ) )
            cluster_colors = [ color_palette[ x ] if x >= 0 else ( 0.5, 0.5, 0.5 ) for x in arr_cluster_label ]
            fig, plt_ax = plt.subplots( 1, 1, figsize = ( 7, 7 ) )
            plt_ax.scatter( * self.bc.meta[ name_col_embedding, None, : 2 ].T, c = cluster_colors, ** dict_kw_scatter ) # retrieve embedding data and draw the graph
            
        return
    ''' for kNN-bsed label transfer '''
    def train_label( self, name_model : str = 'knn_classifier', n_neighbors : int = 10, name_col_label : str = 'hdbscan', name_col_data : str = 'X_pca', int_num_components_data : int = 20, axis : Union[ int, str ] = 'barcodes', name_col_filter : str = 'filter_label', dict_kw_pynndescent : dict = { 'low_memory' : True, 'n_jobs' : None, 'compressed' : False }, index_col_of_name_col_label : Union[ int, None ] = None ) -> None :
        """ # 2022-08-08 16:42:16 
        build nearest-neighbor search index from the entries of the given axis, and using the labels of the entries, construct a kNN classifier
        
        arguments:
        === general ===
        'axis' : { 0 or 'barcodes' } for operating on the 'barcodes' axis, and { 1 or 'features' } for operating on the 'features' axis
        
        === nearest-neighbor search index ===
        'name_model' : name of the nearest-neighbor index and associated lables of the entries of the index that was saved/will be saved in the RamData.models database. if the model already exists, the index and the associated labels will be loadeded, and will be used to predict labels of the remaining entries.
        'n_neighbors' : the number of neighbors to use for the index
        'dict_kw_pynndescent' : the remaining arguments for constructing the index pynndescent.NNDescen 'model'
        
        === data input ===
        'name_col_filter' : the 'name_col' of the metadata of the given axis containing the filter marking the entries that will be used for trainining (building the index)
        'name_col_label' : the 'name_col' of the metadata of the given axis containing 'labels'.
        'index_col_of_name_col_label' : index of the secondary axis of the column of 'name_col_label' that will contain cluster labels. if None, 'name_col_label' is assumed to be a 1-dimensional column. 
        'name_col_data' : the 'name_col' of the metadata of the given axis containing 'data' for building nearest-neighbor search.
        'int_num_components_data' : the number of components in the 'data' to use. for example, when 'int_num_components_data' is 2 and the 'data' contains 3 components, only the first two components will be used to build the index
        
        returns:
        labels, index 
        """
        # handle inputs
        flag_axis_is_barcode = axis in { 0, 'barcode', 'barcodes' } # retrieve a flag indicating whether the data is summarized for each barcode or not
        
        ax = self.bc if flag_axis_is_barcode else self.ft # retrieve the appropriate Axis object
        
        # set filters for operation
        if name_col_filter is not None :
            self.change_or_save_filter( name_col_filter )
            
        """
        2) Train model and retrieve cluster labels
        """
        # load the model and retrieve cluster labels
        type_model = 'knn_classifier'
        model = self.load_model( name_model, type_model )
        if model is None : # if the model does not exist, initiate the model
            # load training data
            data = ax.meta[ name_col_data, None, : int_num_components_data ]
            labels = ax.meta[ name_col_label ] if index_col_of_name_col_label is None else ax.meta[ name_col_label, None, index_col_of_name_col_label ] # retrieve labels
            
            index = pynndescent.NNDescent( data, n_neighbors = n_neighbors, ** dict_kw_pynndescent )
            index.prepare( ) # prepare index for searching

            # save trained model
            if name_model is not None : # check validity of 'name_model' 
                self.save_model( ( labels, index ), name_model, type_model ) # save model to the RamData # save both labels used for prediction and the index
        else : # if the model with 'name_model' name exists in the database, use the previously saved model
            labels, index = model # parse the model
        
        # report
        if self.verbose :
            print( f"[Info] [RamData.train_label] training of labels completed for {ax.meta.n_rows} number of entries of the axis '{'barcodes' if flag_axis_is_barcode else 'features'}'" )
    def apply_label( self, name_model : str = 'knn_classifier', name_col_label : str = 'hdbscan', name_col_data : str = 'X_pca', int_num_threads : int = 10, int_num_entries_in_a_batch : int = 10000, axis : Union[ int, str ] = 'barcodes', name_col_filter : str = 'filter_pca', index_col_of_name_col_label : Union[ int, None ] = None ) -> dict :
        """ # 2022-08-08 16:42:16 
        using the previously constructed kNN classifier, predict labels of the entries by performing the nearest-neighbor search
        
        arguments:
        === general ===
        'axis' : { 0 or 'barcodes' } for operating on the 'barcodes' axis, and { 1 or 'features' } for operating on the 'features' axis
        'int_num_threads' : the number of threads to use for assigning labels. 3 ~ 10 are recommended.
        'int_num_entries_in_a_batch' : the number of entries to assign labels in each batch
        
        === nearest-neighbor search index ===
        'name_model' : name of the nearest-neighbor index and associated lables of the entries of the index that was saved/will be saved in the RamData.models database. if the model already exists, the index and the associated labels will be loadeded, and will be used to predict labels of the remaining entries.
        
        === data input ===
        'name_col_filter' : the name of 'feature'/'barcode' Axis metadata column to retrieve selection filter for running the current method. if None is given, current barcode/feature filters (if it has been set) will be used as-is.
        'name_col_label' : the 'name_col' of the metadata of the given axis that will contain assigned 'labels' (existing data will be overwritten).
        'index_col_of_name_col_label' : index of the secondary axis of the column of 'name_col_label' that will contain cluster labels. if None, 'name_col_label' is assumed to be a 1-dimensional column. 
        'name_col_data' : the 'name_col' of the metadata of the given axis containing 'data' for building nearest-neighbor search index.
        'int_num_components_data' : the number of components in the 'data' to use. for example, when 'int_num_components_data' is 2 and the 'data' contains 3 components, only the first two components will be used to build the index
        
        returns:
        'dict_label_counter' : a dictionary containing counts of each unique label
        """
        """ prepare """
        # handle inputs
        flag_axis_is_barcode = axis in { 0, 'barcode', 'barcodes' } # retrieve a flag indicating whether the data is summarized for each barcode or not
        
        ax = self.bc if flag_axis_is_barcode else self.ft # retrieve the appropriate Axis object
        
        # set filters for operation
        if name_col_filter is not None :
            self.change_filter( name_col_filter )
            
        """
        load model and the associated data objects
        """
        # load the model and retrieve cluster labels
        type_model = 'knn_classifier'
        model = self.load_model( name_model, type_model )
        if model is None : # if the model does not exist, initiate the model
            if self.verbose :
                print( f"[Error] [RamData.apply_label] the nearest-neighbor search index '{name_model}' does not exist, exiting" )
                return 
        labels, index = model # parse the model

        # retrieve the number of components for the model
        int_num_components_data = index.dim

        """
        assign labels
        """
        if self.verbose :
            print( f"[Info] [RamData.apply_label] the nearest-neighbor search started" )
        # initialize the counter for counting labels
        dict_label_counter = dict( )
        # define functions for multiprocessing step
        def process_batch( batch, pipe_to_main_process ) :
            ''' # 2022-07-13 22:18:22 
            retrieve data and retrieve transformed PCA values for the batch
            '''
            # parse the received batch
            int_num_of_previously_returned_entries, l_int_entry_current_batch = batch[ 'int_num_of_previously_returned_entries' ], batch[ 'l_int_entry_current_batch' ]
            
            # retrieve data from the axis metadata
            data = ax.meta[ name_col_data, l_int_entry_current_batch, : int_num_components_data ]
            
            neighbors, distances = index.query( data ) # retrieve neighbors using the index
            del data, distances
            
            labels_assigned = list( bk.DICTIONARY_Find_keys_with_max_value( bk.COUNTER( labels[ e ] ) )[ 0 ][ 0 ] for e in neighbors ) # assign labels using the labels of nearest neighbors
            del neighbors
            
            pipe_to_main_process.send( ( l_int_entry_current_batch, labels_assigned ) ) # send the result back to the main process
        pbar = progress_bar( total = ax.meta.n_rows ) # initialize the progress bar
        def post_process_batch( res ) :
            """ # 2022-07-13 22:18:26 
            perform PCA transformation for each batch
            """
            # parse result 
            l_int_entry_current_batch, labels_assigned = res
            int_num_retrieved_entries = len( l_int_entry_current_batch )
            
            # write the result to the axis metadata
            if index_col_of_name_col_label is None : # assume 'name_col_label' is 1-dimensional column
                ax.meta[ name_col_label, l_int_entry_current_batch ] = labels_assigned
            else : # update a single column in the meatadata column 'name_col_label'
                ax.meta[ name_col_label, l_int_entry_current_batch, index_col_of_name_col_label ] = labels_assigned
            
            bk.COUNTER( labels_assigned, dict_counter = dict_label_counter ) # count assigned labels
            
            pbar.update( int_num_retrieved_entries ) # update the progress bar
            del labels_assigned
        # transform values using iPCA using multiple processes
        bk.Multiprocessing_Batch( ax.batch_generator( ax.filter, int_num_entries_for_batch = int_num_entries_in_a_batch, flag_mix_randomly = False ), process_batch, post_process_batch = post_process_batch, int_num_threads = int_num_threads, int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop = 0.2 ) 
        pbar.close( ) # close the progress bar
        
        # return the counts of each unique label
        return dict_label_counter
    ''' subsampling method '''
    def subsample( self, name_model = 'leiden', int_num_entries_to_use : int = 30000, int_num_entries_to_subsample : int = 100000, int_num_iterations_for_subsampling : int = 2, name_col_data : str = 'X_pca', int_num_components_data : int = 20, int_num_clus_expected : Union[ int, None ] = 20, name_col_label : str = 'subsampling_label', name_col_avg_dist : str = 'subsampling_avg_dist', axis : typing.Union[ int, str ] = 'barcodes', 
                  name_col_filter : str = 'filter_pca', name_col_filter_subsampled : str = "filter_subsampled", resolution = 0.7, directed : bool = True, use_weights : bool = True, dict_kw_leiden_partition : dict = { 'n_iterations' : -1, 'seed' : 0 }, dict_kw_pynndescent_transformer : dict = { 'n_neighbors' : 10, 'metric' : 'euclidean', 'low_memory' : True }, n_neighbors : int = 20, dict_kw_pynndescent : dict = { 'low_memory' : True, 'n_jobs' : None, 'compressed' : False }, int_num_threads : int = 10, int_num_entries_in_a_batch : int = 10000 ) :
        """ # 2022-08-10 04:15:38 
        subsample informative entries through iterative density-based subsampling combined with community detection algorithm
        
        arguments:
        === general ===
        'int_num_entries_to_use' : the number of entries to use during iterative subsampling
        'int_num_entries_to_subsample' : the number of entries to subsample
        'axis' : { 0 or 'barcodes' } for operating on the 'barcodes' axis, and { 1 or 'features' } for operating on the 'features' axis
        'int_num_threads' : the number of threads to use for nearest neighbor search. 3 ~ 10 are recommended.
        'int_num_entries_in_a_batch' : the number of entries in each batch for nearest neighbor search. 
        
        === data input ===
        'name_col_filter' : the 'name_col' of the metadata of the given axis containing the filter marking the entries that will be used for trainining (building the index)
        'name_col_data' : the 'name_col' of the metadata of the given axis containing 'data' for building nearest-neighbor search index.
        'int_num_components_data' : the number of components in the 'data' to use. for example, when 'int_num_components_data' is 2 and the 'data' contains 3 components, only the first two components will be used to build the index
        
        === nearest-neighbor search index ===
        'n_neighbors' : the number of neighbors to use for the index
        'dict_kw_pynndescent' : the remaining arguments for constructing the index pynndescent.NNDescen 'model'
        
        === clustering arguments ===
        'resolution' : initial resolution of cluster. please refer to 'resolution_parameter' of 'leidenalg.find_partition' method
        'int_num_clus_expected' : the expected number of clusters in the data to optimize hyperparameters for community detection. if 'int_num_clus_expected' is not None, until the number of detected communities reaches 'int_num_clus_expected', the 'resolution' parameter will be optimized. this argument will be inactive when 'resolution' is None.
        'directed' : create directed graph. it is recommended to set it to True
        'use_weights' : use weights of the kNN graph for the leiden partitioning
        'dict_kw_leiden_partition' : a dictionary containing keyworded arguments for the 'leidenalg.find_partition' method
        'dict_kw_pynndescent_transformer' : a dictionary containing keyworded arguments for the 'pynndescent.PyNNDescentTransformer' method for constructing kNN graph from the data
        
        === iterative community-detection-and-density-based subsampling ===
        'name_model' : name of the kNN-clasifier model for predicting labels
        'int_num_iterations_for_subsampling' : the number of interations of subsampling operations
        
        === output ===
        'name_col_label' : the 'name_col' of the axis metadata that will contains clueter lables for each iteration.
        'name_col_avg_dist' : the 'name_col' of the axis metadata that will contains average distance of an entry to its nearest neighbors for each iteration.
        'name_col_filter_subsampled' : the 'name_col' of the metadata of the given axis containing the subsampled entries
        
        returns:
        """
        # handle inputs
        flag_axis_is_barcode = axis in { 0, 'barcode', 'barcodes' } # retrieve a flag indicating whether the data is summarized for each barcode or not
        
        ax = self.bc if flag_axis_is_barcode else self.ft # retrieve the appropriate Axis object

        # initialize output columns in the metadata
        ax.meta.initialize_column( name_col_label, dtype = np.int32, shape_not_primary_axis = ( int_num_iterations_for_subsampling, ), chunks = ( 1, ), categorical_values = None ) 
        ax.meta.initialize_column( name_col_avg_dist, dtype = np.float64, shape_not_primary_axis = ( int_num_iterations_for_subsampling, ), chunks = ( 1, ), categorical_values = None )
        
        # set filters for operation
        if name_col_filter is None :
            if self.verbose  :
                print( f"[Error] [RamData.subsample] 'name_col_filter' should not be None, exiting" )
            return
        self.change_or_save_filter( name_col_filter )
        
        # when the number of entries is below 'int_num_entries_to_subsample'
        if int_num_entries_to_subsample >= ax.meta.n_rows :
            """ if no subsampling is required, save the current filter as the subsampled filter, and exit """
            self.save_filter( name_col_filter_subsampled )
            return
        
        # perform initial random sampling
        ax.filter = ax.subsample( min( 1, int_num_entries_to_use / ax.meta.n_rows ) ) # retrieve subsampling ratio
        self.save_filter( name_col_filter_subsampled ) # save subsampled filter
        
        type_model = 'knn_classifier'
        for index_iteration in range( int_num_iterations_for_subsampling ) : # for each iteration
            if self.verbose :
                print( f"[Info] [RamData.subsample] iteration #{index_iteration} started." )
            """
            community detection - leiden
            """
            # perform leiden clustering
            self.leiden( name_model = None, name_col_data = name_col_data, int_num_components_data = int_num_components_data, name_col_label = name_col_label, resolution = resolution, int_num_clus_expected = int_num_clus_expected, directed = directed, use_weights = use_weights, dict_kw_leiden_partition = dict_kw_leiden_partition, dict_kw_pynndescent_transformer = dict_kw_pynndescent_transformer, name_col_filter = None, name_col_embedding = None, index_col_of_name_col_label = index_iteration ) # clustering result will be saved 'index_iteration' column in the 'name_col_label' # does not save model
            
            # assign labels and retrieve label counts
            self.delete_model( name_model, type_model ) # reset the model before training
            self.train_label( name_model = name_model, n_neighbors = n_neighbors, name_col_label = name_col_label, name_col_data = name_col_data, int_num_components_data = int_num_components_data, axis = axis, name_col_filter = None, dict_kw_pynndescent = dict_kw_pynndescent, index_col_of_name_col_label = index_iteration )
            dict_label_count = self.apply_label( name_model = name_model, name_col_label = name_col_label, name_col_data = name_col_data, int_num_threads = int_num_threads, int_num_entries_in_a_batch = int_num_entries_in_a_batch, axis = axis, name_col_filter = name_col_filter, index_col_of_name_col_label = index_iteration ) # assign labels to 'name_col_filter' entries

            """
            calculate density - build knn search index
            """
            if self.verbose :
                print( f"[Info] [RamData.subsample] iteration #{index_iteration} calculating density information started" )

            # prepare knn search index
            self.change_filter( name_col_filter_subsampled ) # change filter to currently subsampled entries for building knn search index
            index = pynndescent.NNDescent( ax.meta[ name_col_data, None, : int_num_components_data ], n_neighbors = n_neighbors, ** dict_kw_pynndescent )
            index.prepare( ) # prepare index for searching

            """
            calculate density - summarize the distances
            """
            self.change_filter( name_col_filter ) # change filter to all the query entries for estimating density
            dict_label_total_avg_dist = dict( ) # initialize the dictionary for surveying the the total 'average distance' values of the entries belonging to each unique label
            # define functions for multiprocessing step
            def process_batch( batch, pipe_to_main_process ) :
                ''' # 2022-07-13 22:18:22 
                retrieve data and retrieve transformed PCA values for the batch
                '''
                # parse the received batch
                int_num_of_previously_returned_entries, l_int_entry_current_batch = batch[ 'int_num_of_previously_returned_entries' ], batch[ 'l_int_entry_current_batch' ]

                # retrieve data from the axis metadata
                data = ax.meta[ name_col_data, l_int_entry_current_batch, : int_num_components_data ]

                neighbors, distances = index.query( data ) # retrieve neighbors using the index
                del data, neighbors

                pipe_to_main_process.send( ( l_int_entry_current_batch, distances.mean( axis = 1 ) ) ) # calculate average distances of the entries in a batch # send the result back to the main process
            pbar = progress_bar( total = ax.meta.n_rows ) # initialize the progress bar
            def post_process_batch( res ) :
                """ # 2022-07-13 22:18:26 
                perform PCA transformation for each batch
                """
                # parse result 
                l_int_entry_current_batch, arr_avg_dist = res
                int_num_retrieved_entries = len( l_int_entry_current_batch )

                # write the result to the axis metadata
                ax.meta[ name_col_avg_dist, l_int_entry_current_batch, index_iteration ] = arr_avg_dist
                
                # retrieve assigned labels, and summarize calculated average distances
                for label, avg_dist in zip( ax.meta[ name_col_label, l_int_entry_current_batch, index_iteration ], arr_avg_dist ) :
                    if label not in dict_label_total_avg_dist :
                        dict_label_total_avg_dist[ label ] = 0
                    dict_label_total_avg_dist[ label ] += avg_dist # update total avg_dist of the label

                pbar.update( int_num_retrieved_entries ) # update the progress bar
            # transform values using iPCA using multiple processes
            bk.Multiprocessing_Batch( ax.batch_generator( ax.filter, int_num_entries_for_batch = int_num_entries_in_a_batch, flag_mix_randomly = False ), process_batch, post_process_batch = post_process_batch, int_num_threads = int_num_threads, int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop = 0.2 ) 
            pbar.close( ) # close the progress bar
            
            """
            using the summarized metrics, prepare subsampling
            """
            # calculate the number of entries to subsample for each unique label
            int_num_labels = len( dict_label_count ) # retrieve the number of labels
            
            # convert type of 'dict_label_count' to numpy ndarray # leiden labels are 0-based coordinate-based
            arr_label_count = np.zeros( int_num_labels )
            for label in dict_label_count :
                arr_label_count[ label ] = dict_label_count[ label ]
            
            int_num_entries_to_include = int_num_entries_to_subsample if index_iteration == int_num_iterations_for_subsampling - 1 else int_num_entries_to_use # if the current round is the last round, include 'int_num_entries_to_subsample' number of entries in the output filter. if not, include 'int_num_entries_to_use' number of entries for next iteration
            
            int_label_count_current_threshold = int( int_num_entries_to_include / int_num_labels ) # initialize the threshold
            for index_current_label in np.argsort( arr_label_count ) : # from label with the smallest number of entries to label with the largest number of entries
                int_label_count = arr_label_count[ index_current_label ]
                if int_label_count > int_label_count_current_threshold :
                    break
                arr = arr_label_count[ arr_label_count <= int_label_count ]
                int_label_count_current_threshold = int( ( int_num_entries_to_include - arr.sum( ) ) / ( int_num_labels - len( arr ) ) )
                
            # retrieve number of entries to subsample for each label
            arr_label_count_subsampled = deepcopy( arr_label_count )
            arr_label_count_subsampled[ arr_label_count_subsampled > int_label_count_current_threshold ] = int_label_count_current_threshold
            
            # compose name space for subsampling
            dict_ns = dict( ( label, { 'int_num_entries_remaining_to_reject' : dict_label_count[ label ] - arr_label_count_subsampled[ label ], 'int_num_entries_remaining_to_accept' : arr_label_count_subsampled[ label ] } ) for label in dict_label_count )
            
            if self.verbose :
                print( f"[Info] [RamData.subsample] iteration #{index_iteration} subsampling started" )

            # define functions for multiprocessing step
            def process_batch( batch, pipe_to_main_process ) :
                ''' # 2022-07-13 22:18:22 
                retrieve data and retrieve transformed PCA values for the batch
                '''
                # parse the received batch
                int_num_of_previously_returned_entries, l_int_entry_current_batch = batch[ 'int_num_of_previously_returned_entries' ], batch[ 'l_int_entry_current_batch' ]
                
                pipe_to_main_process.send( ( l_int_entry_current_batch, ax.meta[ name_col_label, l_int_entry_current_batch, index_iteration ], ax.meta[ name_col_avg_dist, l_int_entry_current_batch, index_iteration ] ) ) # retrieve data from the axis metadata and # send result back to the main process
            pbar = progress_bar( total = ax.meta.n_rows ) # initialize the progress bar
            def post_process_batch( res ) :
                """ # 2022-07-13 22:18:26 
                perform PCA transformation for each batch
                """
                # parse result 
                l_int_entry_current_batch, arr_labels, arr_avg_dist = res
                int_num_retrieved_entries = len( l_int_entry_current_batch )
                
                # initialize selection result
                arr_selection = np.zeros( len( arr_labels ), dtype = bool ) # no selected entries by default

                for index in range( len( arr_labels ) ) : # iterate through each entry by entry
                    label, avg_dist = arr_labels[ index ], arr_avg_dist[ index ] # retrieve data of an entry
                    
                    # if no entry should be rejected, select 
                    if dict_ns[ label ][ 'int_num_entries_remaining_to_reject' ] == 0 :
                        arr_selection[ index ] = True
                        dict_ns[ label ][ 'int_num_entries_remaining_to_accept' ] -= 1 # update 'int_num_entries_remaining_to_accept' 
                    else :
                        if ( ( dict_ns[ label ][ 'int_num_entries_remaining_to_accept' ] / dict_ns[ label ][ 'int_num_entries_remaining_to_reject' ] ) * avg_dist / dict_label_total_avg_dist[ label ] ) > np.random.random( ) : # determine whether the current entry should be included in the subsampled result
                            arr_selection[ index ] = True
                            dict_ns[ label ][ 'int_num_entries_remaining_to_accept' ] -= 1 # update 'int_num_entries_remaining_to_accept' 
                        else :
                            dict_ns[ label ][ 'int_num_entries_remaining_to_reject' ] -= 1 # update 'int_num_entries_remaining_to_reject'
                    
                # write the subsampled result to the axis metadata
                ax.meta[ name_col_filter_subsampled, l_int_entry_current_batch ] = arr_selection
                
                pbar.update( int_num_retrieved_entries ) # update the progress bar
            # transform values using iPCA using multiple processes
            bk.Multiprocessing_Batch( ax.batch_generator( ax.filter, int_num_entries_for_batch = int_num_entries_in_a_batch, flag_mix_randomly = False ), process_batch, post_process_batch = post_process_batch, int_num_threads = min( 3, int_num_threads ), int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop = 0.2 ) 
            pbar.close( ) # close the progress bar
            
            # prepare next batch
            self.change_filter( name_col_filter_subsampled ) # change filter to currently subsampled entries for the next round
    ''' for marker detection analysis '''
    def find_markers( self, name_layer : str = 'normalized_log1p_scaled', name_col_label : str = 'subsampling_label', index_name_col_label : int = -1, l_name_cluster : Union[ list, np.ndarray, tuple, set, None ] = None, name_col_auroc : str = 'marker_auroc', name_col_log2fc : str = 'marker_log2fc', name_col_pval : str = 'marker_pval', method_pval : str = 'wilcoxon', int_chunk_size_secondary = 10 ) :
        """ # 2022-08-21 13:37:02 
        find marker features for each cluster label by calculating a AUROC metric, log2FC, and Wilcoxon (or alternatively, t-test or Mann-Whitney-U rank test)
        
        name_layer : str = 'normalized_log1p_scaled' : a layer containing expression data to use for finding marker. scaled data is recommended
        name_col_label : str = 'subsampling_label' : the name of the column of 'barcodes' metadata containing cluster labels
        index_name_col_label : int = -1 : the index of the column (secondary axis) of the 'name_col_label' metadata column. if no secondary axis is available, this argument will be ignored.
        l_name_cluster : Union[ list, np.ndarray, tuple, None ] = None : the list of cluster labels that will be included 
        
        
        === output ===
        name_col_auroc : str = 'marker_auroc' : the name of the output column name in the 'features' metadata ZDF. This column contains Area Under Receiver Operating Characteristic (Sensitivity/Specificity) curve values for each cluster and feature pair. if None is given, AUROC will be not calculated
        name_col_log2fc : str = 'marker_log2fc' : the name of the output column name in the 'features' metadata ZDF. This column contains Log_2 fold change values of the cells of the cluster label of interest versus the rest of the cells.
        name_col_pval : str = 'marker_pval' : the name of the output column name in the 'features' metadata ZDF. This column contains uncorrected p-value from the wilcoxon or t-test results
        method_pval : str = 'wilcoxon' : one of the test methods in { 'wilcoxon', 't-test', 'mann-whitney-u' }
        int_chunk_size_secondary : int = 10 : the chunk size of the output columns along the secondary axis
        
        an array with a shape of ( the number of all features ) X ( the number of all cluster labels ), stored in the feature metadata using the given column name
        information about which column of the output array represent which cluster label is available in the column metadata.
        """
        # handle inputs
        if name_col_label not in self.bc.meta : # check input label
            if self.verbose  :
                print( f"[Error] [RamData.find_markers] 'name_col_label' {name_col_label} does not exist in barcode metadata, exiting" )
            return

        if name_layer not in self.layers : # check input layer
            if self.verbose  :
                print( f"[Error] [RamData.find_markers] 'name_layer' {name_layer} does not exist in the layers, exiting" )
            return
        self.layer = name_layer # load layer
        
        # retrieve flags
        flag_calcualte_auroc = name_col_auroc is not None
        flag_calculate_pval = name_col_pval is not None
        assert name_col_log2fc is not None # 'name_col_log2fc' should not be None
        
        # retrieve function for testing p-value
        test = None
        if flag_calculate_pval :
            if method_pval not in { 'wilcoxon', 't-test', 'mann-whitney-u' } :
                if self.verbose  :
                    print( f"[Error] [RamData.find_markers] 'method_pval' {method_pval} is invalid, exiting" )
                return
            if method_pval == 't-test' :
                test = scipy.stats.ttest_ind
            elif method_pval == 'wilcoxon' :
                test = scipy.stats.ranksums
            elif method_pval == 'mann-whitney-u' :
                test = scipy.stats.mannwhitneyu
        
        # compose 'l_name_col_summarized', a list of output column names
        l_name_col_summarized = [ name_col_log2fc ]
        l_fill_value = [ np.nan ] # for 'col_log2fc', fill_value = 0
        if flag_calcualte_auroc :
            l_name_col_summarized.append( name_col_auroc )
            l_fill_value.append( 0 )
        if flag_calculate_pval :
            l_name_col_summarized.append( name_col_pval )
            l_fill_value.append( -1 )
        
        # retrieve labels (considering filters)
        n_dims_non_primary = len( self.bc.meta.get_shape( name_col_label ) )
        arr_cluster_label = self.bc.meta[ name_col_label ] if n_dims_non_primary == 0 else self.bc.meta[ name_col_label, None, index_name_col_label ] # retrieve cluster labels
        int_num_barcodes = len( arr_cluster_label ) # retrieve the number of active barcodes
        l_unique_cluster_label = sorted( convert_numpy_dtype_number_to_number( e ) for e in set( arr_cluster_label ) ) # retrieve a list of unique cluster labels
        dict_cluster_label_to_index = dict( ( e, i ) for i, e in enumerate( l_unique_cluster_label ) ) # map cluster labels to integer indices
        int_num_cluster_labels = len( l_unique_cluster_label ) # retrieve the total number of cluster labels
        
        if l_name_cluster is not None : # if 'l_name_cluster' is given, analyze only the labels in the given list of cluster labels
            set_cluster_label = set( l_name_cluster )
            l_unique_cluster_label_to_analyze = list( e for e in l_unique_cluster_label if e in set_cluster_label )
            del set_cluster_label
        else : # by default, analyze all cluster labels
            l_unique_cluster_label_to_analyze = l_unique_cluster_label
            
        # initialize output columns
        for name_col, fill_value in zip( l_name_col_summarized, l_fill_value ) : # for each column, retrieve name_col and fill_value
            name_col = f"{name_layer}_{name_col}" # compose the name of the destination column
            self.ft.meta.initialize_column( name_col, dtype = np.float64, shape_not_primary_axis = ( int_num_cluster_labels, ), chunks = ( int_chunk_size_secondary, ), fill_value = fill_value ) 
            dict_metadata = self.ft.meta.get_column_metadata( name_col ) # retrieve metadata
            dict_metadata[ 'l_labels_1' ] = l_unique_cluster_label # add cluster label information
            self.ft.meta._set_column_metadata( name_col, dict_metadata ) # update column metadata
            
        # create view
        flag_view_was_not_active = not self.bc.is_view_active # retrieve a flag indicating a view was not active
        if flag_view_was_not_active : # create view
            self.bc.create_view( )
        
        def func( self, int_entry_of_axis_for_querying : int, arr_int_entries_of_axis_not_for_querying : np.ndarray, arr_value : np.ndarray ) : # normalize count data of a single feature containing (possibly) multiple barcodes
            """ # 2022-08-22 11:25:57 
            find markers
            """
            if len( arr_value ) == 0 : # handle when no expression values are available, exit
                return
            
            dict_summary = dict( ( name_col, [ fill_value ] * int_num_cluster_labels ) for name_col, fill_value in zip( l_name_col_summarized, l_fill_value ) ) # initialize output dictionary
            
            arr_expr = np.zeros( int_num_barcodes ) # initialize expression values in dense format
            arr_expr[ arr_int_entries_of_axis_not_for_querying ] = arr_value # convert sparse to dense format
            
            # for each cluster
            for name_clus in l_unique_cluster_label_to_analyze :
                index_clus = dict_cluster_label_to_index[ name_clus ] # retrieve index of the current cluster
                # retrieve expression values of cluster and the rest of the barcodes
                mask = arr_cluster_label == name_clus
                arr_expr_clus = arr_expr[ mask ]
                arr_expr_rest = arr_expr[ ~ mask ]
                
                # calculate log2fc values
                mean_arr_expr_rest = arr_expr_rest.mean( )
                if mean_arr_expr_rest != 0 :
                    try :
                        dict_summary[ name_col_log2fc ][ index_clus ] = math.log2( arr_expr_clus.mean( ) / mean_arr_expr_rest )
                    except ValueError : # catch math.log2 domain error
                        pass
                
                # calculate auroc
                if flag_calcualte_auroc :
                    dict_summary[ name_col_auroc ][ index_clus ] = skl.metrics.roc_auc_score( mask, arr_expr )
                    
                # calculate ttest
                if flag_calculate_pval and test is not None :
                    dict_summary[ name_col_pval ][ index_clus ] = test( arr_expr_clus, arr_expr_rest ).pvalue
            return dict_summary    

        # calculate the metric for identifying marker features
        self.summarize( name_layer, 'features', func, l_name_col_summarized = l_name_col_summarized )

        # destroy view if a view was not active
        if flag_view_was_not_active :
            self.bc.destroy_view( )
    def get_marker_table( self, max_pval : float = 1e-10, min_auroc : float = 0.7, min_log2fc : float = 1, name_col_auroc : Union[ str, None ] = 'normalized_log1p_scaled_marker_auroc', name_col_log2fc : Union[ str, None ] = 'normalized_log1p_scaled_marker_log2fc', name_col_pval : Union[ str, None ] = 'normalized_log1p_scaled_marker_pval', int_num_chunks_in_a_batch : int = 10 ) :
        """ # 2022-08-22 21:59:46 
        retrieve marker table using the given thresholds

        === arguments ===
        max_pval : float = 1e-10 : maximum p-value for identification of marker features
        min_auroc : float = 0.7 : minimum AUROC metric value for identification of marker features
        min_log2fc : float = 1 : minimum Log2 fold change metric value for identification of marker features
        name_col_auroc : Union[ str, None ] = 'normalized_log1p_scaled_marker_auroc' : the name of the column containing AUROC metrics for each feature for each cluster label. if None is given, AUROC metric will be ignored
        name_col_log2fc : Union[ str, None ] = 'normalized_log1p_scaled_marker_log2fc' : the name of the column containing AUROC metrics for each feature for each cluster label. if None is given, Log2FC values will be ignored
        name_col_pval : Union[ str, None ] = 'normalized_log1p_scaled_marker_pval' : the name of the column containing p-value significance for null-hypothesis testing for each feature for each cluster label. if None is given, p-value will be ignored
        int_num_chunks_in_a_batch : int = 10 : the number of chunks in a batch
        """
        # retrieve the maximum number of entries in a batch
        int_num_entries_in_a_batch = self.bc.meta.int_num_rows_in_a_chunk * int_num_chunks_in_a_batch

        # handle inputs
        flag_use_auroc = name_col_auroc is not None
        flag_use_log2fc = name_col_log2fc is not None
        flag_use_pval = name_col_pval is not None

        if not ( flag_use_auroc or flag_use_log2fc or flag_use_pval ) : 
            if self.verbose :
                print( f"[RamData.get_marker_table] at least one metric should be used for filtering markers but none were given, exiting." )
            return

        # retrieve 'features' axis
        ax = self.ft

        # retrieve a list of unique cluster labels
        for name_col in [ name_col_auroc, name_col_log2fc, name_col_pval ] :
            if name_col in ax.meta :
                l_unique_cluster_label = ax.meta.get_column_metadata( name_col )[ 'l_labels_1' ]
                break
        # retrieve the number of cluster labels
        int_num_cluster_labels = len( l_unique_cluster_label )

        # collect the result
        l_l = [ ]

        int_num_entries_searched = 0 # initialize the position
        int_num_entries_total = self.ft.int_num_entries # retrieve the total number of entries to search
        while int_num_entries_searched <= int_num_entries_total : # until all entries were searched
            mask = np.ones( ( min( int_num_entries_in_a_batch, int_num_entries_total - int_num_entries_searched ), int_num_cluster_labels ), dtype = bool ) # initialize the mask # include all records by default
            sl = slice( int_num_entries_searched, int_num_entries_searched + mask.shape[ 0 ] ) # retrieve a slice for the batch
            if flag_use_auroc :
                arr_data = ax.meta[ name_col_auroc, sl, : ] # retrieve data
                mask &= arr_data >= min_auroc # apply filter using AUROC metrics
                arr_data_auroc = arr_data
            if flag_use_log2fc :
                arr_data = ax.meta[ name_col_log2fc, sl, : ] # retrieve data
                mask &= ~ np.isnan( arr_data ) # apply filter using valid log2fc values
                mask &= arr_data >= min_log2fc  # apply filter using log2fc values 
                arr_data_log2fc = arr_data
            if flag_use_pval :
                arr_data = ax.meta[ name_col_pval, sl, : ] # retrieve data
                mask &= arr_data != -1 # apply filter using valid p-values
                mask &= arr_data <= max_pval  # apply filter using p-values
                arr_data_pval = arr_data

            # retrieve coordinates of filtered records
            coords_filtered = np.where( mask )
            int_num_records_filtered = len( coords_filtered[ 0 ] ) # retrieve the number of records after filtering for the current batch

            # retrieve data of filtered records
            arr_data_auroc = arr_data_auroc[ coords_filtered ] if flag_use_auroc else np.full( int_num_records_filtered, np.nan )
            arr_data_log2fc = arr_data_log2fc[ coords_filtered ] if flag_use_log2fc else np.full( int_num_records_filtered, np.nan )
            arr_data_pval = arr_data_pval[ coords_filtered ] if flag_use_pval else np.full( int_num_records_filtered, np.nan )

            arr_int_entry_ft_filtered, arr_int_name_cluster_filtered = coords_filtered # parse coordinates
            arr_int_entry_ft_filtered += int_num_entries_searched # correct 'arr_int_entry_ft_filtered'
            for int_entry_ft, int_name_cluster, value_auroc, value_log2fc, value_pval in zip( arr_int_entry_ft_filtered, arr_int_name_cluster_filtered, arr_data_auroc, arr_data_log2fc, arr_data_pval ) : # for each record
                l_l.append( [ int_entry_ft, l_unique_cluster_label[ int_name_cluster ], value_auroc, value_log2fc, value_pval ] ) # collect metrics and cluster label

            int_num_entries_searched += int_num_entries_in_a_batch # update positions
        df_marker = pd.DataFrame( l_l, columns = [ 'name_feature', 'name_cluster', 'value_auroc', 'value_log2fc', 'value_pval' ] ) # retrieve marker table as a dataframe
        if len( df_marker ) == 0 : # handle the case when no records exist after filtering
            return df_marker

        arr_int_entry_ft = np.sort( df_marker.name_feature.unique( ) ) # retrieve integer representation of features
        arr_str_entry_ft = self.ft.get_str( arr_int_entry_ft ) # retrieve string representations of the features
        dict_mapping = dict( ( i, s ) for i, s in zip( arr_int_entry_ft, arr_str_entry_ft ) ) # retrieve a mapping of int > str repr. of features
        df_marker[ 'name_feature' ] = list( dict_mapping[ i ] for i in df_marker[ 'name_feature' ].values ) # retrieve string representations of the features of the marker table
        return df_marker
    ''' scarab-associated methods for analyzing RamData '''
    def _classify_feature_of_scarab_output_( self, int_min_num_occurrence_to_identify_valid_feature_category = 1000 ) :
        """ # 2022-05-30 12:39:01 
        classify features of count matrix from the scarab output.
        the results will be saved at '_dict_data_for_feature_classification' attribute of the current object. In order to re-run this function with a new setting, please delete the '_dict_data_for_feature_classification' attribute of the current RamData object
        """
        """
        classify scarab features based on a specific format of Scarab output
        """
        if not hasattr( self, '_dict_data_for_feature_classification' ) : # if features were not classified yet, perform classification of features
            ''' retrieve 'name_feature_category_simple' '''
            # retrieve set of int_feature for each simple classification labels
            dict_name_feature_category_simple_to_num_features = dict( ) 
            for int_feature, name_feature in enumerate( self.adata.var.index.values ) : 
                name_feature_category_simple = name_feature.split( '|', 1 )[ 0 ]
                if name_feature_category_simple not in dict_name_feature_category_simple_to_num_features :
                    dict_name_feature_category_simple_to_num_features[ name_feature_category_simple ] = 0
                dict_name_feature_category_simple_to_num_features[ name_feature_category_simple ] += 1 # count current int_feature according to the identified name_feature_category_simple

            # drop name_feature with the number of features smaller than the given threshold
            for name_feature_category_simple in list( dict_name_feature_category_simple_to_num_features ) :
                if dict_name_feature_category_simple_to_num_features[ name_feature_category_simple ] < int_min_num_occurrence_to_identify_valid_feature_category :
                    dict_name_feature_category_simple_to_num_features.pop( name_feature_category_simple )

            # l_name_feaure_category_simple = [ '' ] + list( dict_name_feature_category_simple_to_num_features )
            # dict_name_feature_category_simple_to_int = dict( ( e, i ) for i, e in enumerate( l_name_feaure_category_simple ) )

            ''' retrieve 'name_feature_category_detailed' '''
            l_name_feaure_category_detailed = [ ] # initialize 'l_name_feaure_category_simple'
            dict_name_feaure_category_detailed_to_int = dict( ) # initialize 'dict_name_feaure_category_detailed_to_int'
            arr_int_feature_category_detailed = np.zeros( self._int_num_features, dtype = np.int16 ) # retrieve mapping from int_feature to int_feature_category
            for int_feature, name_feature in enumerate( self.adata.var.index.values ) :
                l_entry = list( e.split( '=', 1 )[ 0 ] for e in name_feature.split( '|' ) ) # retrieve entry composing the name_feature
                if l_entry[ 0 ] not in dict_name_feature_category_simple_to_num_features : # remove the first entry for feature with invalid name_feature_category_simple (features of each gene)
                    l_entry = l_entry[ 1 : ]
                str_category_feature_detailed = '___'.join( l_entry ).replace( 'mode', 'atac_mode' ) # compose str_category_feature_detailed

                if str_category_feature_detailed not in dict_name_feaure_category_detailed_to_int : # append new 'name_feaure_category_detailed' as it is detected 
                    dict_name_feaure_category_detailed_to_int[ str_category_feature_detailed ] = len( l_name_feaure_category_detailed )
                    l_name_feaure_category_detailed.append( str_category_feature_detailed )
                arr_int_feature_category_detailed[ int_feature ] = dict_name_feaure_category_detailed_to_int[ str_category_feature_detailed ]

            # accessory function
            def get_int_feature_category_detailed( int_feature ) :
                return arr_int_feature_category_detailed[ int_feature ]
            vectorized_get_int_feature_category_detailed = np.vectorize( get_int_feature_category_detailed ) # vectorize the function to increase efficiency
            
            # build a mask of features of atac mode
            l_mask_feature_category_of_atac_mode = Search_list_of_strings_Return_mask( l_name_feaure_category_detailed, 'atac_mode' )
            ba_mask_feature_of_atac_mode = bitarray( len( arr_int_feature_category_detailed ) )
            ba_mask_feature_of_atac_mode.setall( 0 )
            for int_feature, int_feature_category in enumerate( arr_int_feature_category_detailed ) :
                if l_mask_feature_category_of_atac_mode[ int_feature_category ] :
                    ba_mask_feature_of_atac_mode[ int_feature ] = 1

                
            # save result and settings to the current object
            self._dict_data_for_feature_classification = {
                'int_min_num_occurrence_to_identify_valid_feature_category' : int_min_num_occurrence_to_identify_valid_feature_category,
                'l_name_feaure_category_detailed' : l_name_feaure_category_detailed,
                'dict_name_feaure_category_detailed_to_int' : dict_name_feaure_category_detailed_to_int,
                'arr_int_feature_category_detailed' : arr_int_feature_category_detailed,
                'get_int_feature_category_detailed' : get_int_feature_category_detailed,
                'vectorized_get_int_feature_category_detailed' : vectorized_get_int_feature_category_detailed,
                'l_mask_feature_category_of_atac_mode' : l_mask_feature_category_of_atac_mode,
                'ba_mask_feature_of_atac_mode' : ba_mask_feature_of_atac_mode,
                'dict_name_feature_category_simple_to_num_features' : dict_name_feature_category_simple_to_num_features,
            } # set the object as an attribute of the object so that the object is available in the child processes consistently                
    def _further_summarize_scarab_output_for_filtering_( self, name_layer = 'raw', name_adata = 'main', flag_show_graph = True ) :
        """ # 2022-06-06 01:01:47 
        (1) calculate the total count in gex mode
        (2) calculate_proportion_of_promoter_in_atac_mode
        assumming Scarab output and the output of 'sum_scarab_feature_category', calculate the ratio of counts of promoter features to the total counts in atac mode.

        'name_layer' : name of the data from which scarab_feature_category summary was generated. by default, 'raw'
        'name_adata' : name of the AnnData of the current RamData object. by default, 'main'
        """
        df = self.ad[ name_adata ].obs # retrieve data of the given AnnData


        # calculate gex metrics
        df[ f'{name_layer}_sum_for_gex_mode' ] = df[ Search_list_of_strings_with_multiple_query( df.columns, f'{name_layer}_sum__', '-atac_mode' ) ].sum( axis = 1 ) # calcualte sum for gex mode outputs

        # calcualte atac metrics
        if f'{name_layer}_sum___category_detailed___atac_mode' in df.columns.values : # check whether the ATAC mode has been used in the scarab output
            df[ f'{name_layer}_sum_for_atac_mode' ] = df[ Search_list_of_strings_with_multiple_query( df.columns, f'{name_layer}_sum__', 'atac_mode' ) ].sum( axis = 1 ) # calcualte sum for atac mode outputs
            df[ f'{name_layer}_sum_for_promoter_atac_mode' ] = df[ list( Search_list_of_strings_with_multiple_query( df.columns, f'{name_layer}_sum___category_detailed___promoter', 'atac_mode' ) ) ].sum( axis = 1 )
            df[ f'{name_layer}_proportion_of_promoter_in_atac_mode' ] = df[ f'{name_layer}_sum_for_promoter_atac_mode' ] / df[ f'{name_layer}_sum_for_atac_mode' ] # calculate the proportion of reads in promoter
            df[ f'{name_layer}_proportion_of_promoter_and_gene_body_in_atac_mode' ] = ( df[ f'{name_layer}_sum_for_promoter_atac_mode' ] + df[ f'{name_layer}_sum___category_detailed___atac_mode' ] ) / df[ f'{name_layer}_sum_for_atac_mode' ] # calculate the proportion of reads in promoter

            # show graphs
            if flag_show_graph :
                MPL_Scatter_Align_Two_Series( df[ f'{name_layer}_sum_for_atac_mode' ], df[ f'{name_layer}_sum_for_gex_mode' ], x_scale = 'log', y_scale = 'log', alpha = 0.005 )
                MPL_Scatter_Align_Two_Series( df[ f'{name_layer}_sum_for_atac_mode' ], df[ f'{name_layer}_proportion_of_promoter_in_atac_mode' ], x_scale = 'log', alpha = 0.01 )

        self.ad[ name_adata ].obs = df # save the result        
    def _filter_cell_scarab_output_( self, path_folder_ramdata_output, name_layer = 'raw', name_adata = 'main', int_min_sum_for_atac_mode = 1500, float_min_proportion_of_promoter_in_atac_mode = 0.22, int_min_sum_for_gex_mode = 250 ) :
        ''' # 2022-06-03 15:25:02 
        filter cells from scarab output 

        'path_folder_ramdata_output' : output directory
        '''
        df = self.ad[ name_adata ].obs # retrieve data of the given AnnData

        # retrieve barcodes for filtering
        set_str_barcode = df[ ( df[ f'{name_layer}_sum_for_atac_mode' ] >= int_min_sum_for_atac_mode ) & ( df[ f'{name_layer}_proportion_of_promoter_in_atac_mode' ] >= float_min_proportion_of_promoter_in_atac_mode ) & ( df[ f'{name_layer}_sum_for_gex_mode' ] >= int_min_sum_for_gex_mode ) ].index.values
        # subset the current RamData for valid cells
        self.subset( path_folder_ramdata_output, set_str_barcode = set_str_barcode )
        if self.verbose :
            print( f'cell filtering completed for {len( set_str_barcode )} cells. A filtered RamData was exported at {path_folder_ramdata_output}' )

# utility functions
# for benchmarking
def draw_result( self, path_folder_graph, dict_kw_scatter = { 's' : 2, 'linewidth' : 0, 'alpha' : 0.1 } ) :
    """ # 2022-08-16 12:00:56 
    draw resulting UMAP graph
    """
    arr_cluster_label = self.bc.meta[ 'subsampling_label', None, 1 ]
    embedding = self.bc.meta[ 'X_umap' ]
    
    # adjust graph settings based on the number of cells to plot
    int_num_bc = embedding.shape[ 0 ]
    size_factor = max( 1, np.log( int_num_bc ) - np.log( 5000 ) )
    dict_kw_scatter = { 's' : 20 / size_factor, 'linewidth' : 0, 'alpha' : 0.5 / size_factor }
    if arr_cluster_label is None :
        arr_cluster_label = [ arr_cluster_label ]
    color_palette = sns.color_palette( 'Paired', len( set( arr_cluster_label ) ) )
    cluster_colors = [ color_palette[ x ] if x is not None and x >= 0 else ( 0.5, 0.5, 0.5 ) for x in arr_cluster_label ]
    fig, plt_ax = plt.subplots( 1, 1, figsize = ( 7, 7 ) )
    plt_ax.scatter( * embedding.T, c = cluster_colors, ** dict_kw_scatter )
    MPL_basic_configuration( x_label = 'UMAP_1', y_label = 'UMAP_1', show_grid = False )
    MPL_SAVE( 'umap.leiden', folder = path_folder_graph )
def umap( adata, l_name_col_for_regression = [ ], float_scale_max_value = 10, int_pca_n_comps = 50, int_neighbors_n_neighbors = 10, int_neighbors_n_pcs = 50, dict_kw_umap = dict( ), dict_leiden = dict( ) ) :
    ''' # 2022-07-06 20:49:44 
    retrieve all expression data of the RamData with current barcodes/feature filters', perform dimension reduction process, and return the new AnnData object with umap coordinate and leiden cluster information

    'adata' : input adata object
    'l_name_col_for_regression' : a list of column names for the regression step. a regression step is often the longest step for demension reduction and clustering process. By simply skipping this step, one can retrieve a brief cluster structures of the cells in a very short time. to skip the regression step, please set this argument to an empty list [ ].
    '''
    # perform dimension reduction and clustering processes
    l_name_col_for_regression = list( set( adata.obs.columns.values ).intersection( l_name_col_for_regression ) ) # retrieve valid column names for regression
    if len( l_name_col_for_regression ) > 1 :
        sc.pp.regress_out( adata, l_name_col_for_regression )
    sc.pp.scale( adata, max_value = float_scale_max_value )
    sc.tl.pca( adata, svd_solver = 'arpack', n_comps = int_pca_n_comps )
    sc.pl.pca_variance_ratio( adata, log = True )
    sc.pp.neighbors( adata, n_neighbors = int_neighbors_n_neighbors, n_pcs = int_neighbors_n_pcs )
    sc.tl.umap( adata, ** dict_kw_umap )
    sc.tl.leiden( adata, ** dict_leiden )

    return adata

''' # not implemented in zarr '''
def Convert_Scanpy_AnnData_to_RAMtx( adata, path_folder_ramtx_output, flag_ramtx_sorted_by_id_feature = True, file_format = 'mtx_gzipped', int_num_threads = 15, verbose = False, flag_overwrite_existing_files = False, flag_sort_and_index_again = False, flag_debugging = False, flag_covert_dense_matrix = False, int_num_digits_after_floating_point_for_export = 5, flag_output_value_is_float = True, dtype_of_row_and_col_indices = np.uint32, dtype_of_value = None ) :
    ''' # 2022-05-25 22:50:27 
    build RAMtx from Scanpy AnnData
    It is recommended to load all data on the memory for efficient conversion. It is often not practical to use the disk-backed AnnData to build RAMtx due to slow access time of the object.
    
    2022-05-16 15:05:32 :
    barcodes and features with zero counts will not be removed to give flexibillity to the RAMtx object.
    
    'path_folder_ramtx_output' an output directory of RAMtx object
    'flag_ramtx_sorted_by_id_feature' : a flag indicating whether the output matrix is sorted and indexed by id_feature.
    'file_format' : file_format of the output RAMtx object. please check 'Convert_MTX_10X_to_RAMtx' function docstring for more details.
    'int_num_threads' : the number of processes to use.
    'dtype_of_row_and_col_indices' (default: np.uint32), 'dtype_of_value' (default: None (auto-detect)) : numpy dtypes for pickle and feather outputs. choosing smaller format will decrease the size of the object on disk
    'flag_covert_dense_matrix' : by default, only sparse matrix will be converted to RAMtx. However, dense matrix can be also converted to RAMtx if this flag is set to True
    '''
    ''' handle inputs '''
    flag_is_data_dense = isinstance( adata.X, ( np.ndarray ) ) # check whether the data matrix is in dense format
    if not flag_covert_dense_matrix and flag_is_data_dense :
        raise IndexError( "currently dense matrix is not allowed to be converted to RAMtx. to allow this behavior, please set 'flag_is_data_dense' to True" )

    """ retrieve RAMtx_file_format specific export settings """
    # prepare
    str_format_value = "{:." + str( int_num_digits_after_floating_point_for_export ) + "f}" if flag_output_value_is_float else "{}" # a string for formating value
    str_etx, str_ext_index, func_arrays_mtx_to_processed_bytes = _get_func_arrays_mtx_to_processed_bytes_and_other_settings_based_on_file_format( file_format, str_format_value, dtype_of_row_and_col_indices, dtype_of_value )
        
    # create an ramtx output folder
    os.makedirs( path_folder_ramtx_output, exist_ok = True )
    
#     2022-05-16 15:05:32 :
#     barcodes and features with zero counts will not be removed to give flexibillity to the RAMtx object. 
#     ''' preprocess input AnnData '''
#     # discard barcodes and features with zero counts
#     arr_mask_valid_barcode = np.array( adata.X.sum( axis = 1 ) ).ravel( ) > 0
#     arr_mask_valid_feature = np.array( adata.X.sum( axis = 0 ) ).ravel( ) > 0
#     if sum( arr_mask_valid_barcode ) != adata.shape[ 0 ] or sum( arr_mask_valid_feature ) != adata.shape[ 1 ] :
#         adata = adata[ adata.X.sum( axis = 1 ) > 0, adata.X.sum( axis = 0 ) > 0 ].copy( )
        
    ''' output metadata associated with anndata to RAMtx '''
    # save metadata (obs & var)
    # save (obs)
    df_obs = deepcopy( adata.obs )
    df_obs.reset_index( drop = False, inplace = True )
    df_obs.rename( columns = { 'index' : 'id_cell' }, inplace = True )
    arr_index_sorting_obs = df_obs.id_cell.argsort( ).values
    df_obs.sort_values( 'id_cell', inplace = True )
    df_obs.to_csv( f'{path_folder_ramtx_output}df_obs.tsv.gz', index = False, sep = '\t' )

    # save (var)
    df_var = deepcopy( adata.var )
    df_var.reset_index( drop = False, inplace = True )
    df_var.rename( columns = { 'index' : 'name_feature', 'gene_ids' : 'id_feature' }, inplace = True )
    df_var = df_var[ [ 'id_feature', 'name_feature', 'feature_types' ] + list( df_var.columns.values[ 3 : ] ) ] # rearrange column names
    arr_index_sorting_var = df_var.id_feature.argsort( ).values
    df_var.sort_values( 'id_feature', inplace = True )
    df_var.to_csv( f"{path_folder_ramtx_output}df_var.tsv.gz", sep = '\t', index = False )

    """ save X as RAMtx """
    # use multiple processes
    # create a temporary folder
    path_folder_temp = f'{path_folder_ramtx_output}temp_{UUID( )}/'
    os.makedirs( path_folder_temp, exist_ok = True )

    # prepare multiprocessing
    arr_index_sorting = arr_index_sorting_var if flag_ramtx_sorted_by_id_feature else arr_index_sorting_obs # retrieve index sorting
    l_arr_index_sorting_for_each_chunk, l_arr_weight_for_each_chunk = LIST_Split( arr_index_sorting, int_num_threads, flag_contiguous_chunk = True, arr_weight_for_load_balancing = np.array( adata.X.sum( axis = 0 if flag_ramtx_sorted_by_id_feature else 1 ) ).ravel( )[ arr_index_sorting ], return_split_arr_weight = True ) # perform load balancing using the total count for each entry as a weight
    dict_index_sorting_to_int_entry_sorted = dict( ( index_before_sorting, index_after_sorting + 1 ) for index_after_sorting, index_before_sorting in enumerate( arr_index_sorting ) ) # retrieve mapping between index for sorting to int_entry after sorted 
    ''' retrieve mappings of previous int_entry to new int_entry after sorting (1-based) for both barcodes and features '''
    # retrieve sorting indices of barcodes and features
    dict_name_entry_to_dict_int_entry_to_int_entry_new_after_sorting = dict( ( name_entry, dict( ( int_entry_before_sorting, int_entry_after_sorting ) for int_entry_after_sorting, int_entry_before_sorting in enumerate( arr_index_sorting ) ) ) for name_entry, arr_index_sorting in zip( [ 'barcodes', 'features' ], [ arr_index_sorting_obs, arr_index_sorting_var ] ) ) # 0>1 based coordinates. int_entry is in 1-based coordinate format (same as mtx format) # since rank will be used as a new index, it should be 1-based, and 1 will be added to the rank in 0-based coordinates

    # setting for the pipeline
    int_total_weight_for_each_batch = 2500000
    vfunc_map_index_barcode = np.vectorize( MAP.Map( dict_name_entry_to_dict_int_entry_to_int_entry_new_after_sorting[ 'barcodes' ] ).a2b )
    vfunc_map_index_feature = np.vectorize( MAP.Map( dict_name_entry_to_dict_int_entry_to_int_entry_new_after_sorting[ 'features' ] ).a2b )
    def __compress_and_index_a_portion_of_sparse_matrix_as_a_worker__( index_chunk ) :
        ''' # 2022-05-08 13:19:13 
        save a portion of a sparse matrix 
        '''
        # open output files
        file_output = open( f'{path_folder_temp}indexed.{index_chunk}.{str_etx}', 'wb' ) # open file appropriate for the file format
        file_index_output = gzip.open( f'{path_folder_temp}indexed.{index_chunk}.{str_etx}.{str_ext_index}', 'wb' ) # open file appropriate for the file format

        int_num_bytes_written = 0 # track the number of written bytes
        # methods and variables for handling metadata
        int_total_weight_current_batch = 0
        l_index_sorting_current_batch = [ ]
        def __process_batch__( file_output, file_index_output, int_num_bytes_written, l_index_sorting_current_batch ) :
            ''' # 2022-05-08 13:19:07 
            process the current batch and return updated 'int_num_bytes_written' 
            '''
            # retrieve the number of index_entries
            int_num_entries = len( l_index_sorting_current_batch )
            # handle invalid inputs
            if int_num_entries == 0 :
                return 0 # '0' bytes were written

            # retrieve data for the current batch
            arr_int_row, arr_int_col, arr_value = scipy.sparse.find( adata.X[ :, l_index_sorting_current_batch ] if flag_ramtx_sorted_by_id_feature else adata.X[ l_index_sorting_current_batch, : ] )
            # handle invalid batch
            if len( arr_int_row ) == 0 :
                return 0 # '0' bytes were written
            # renumber indices of sorted entries to match that in the sorted matrix
            int_entry_sorted_first_entry_in_a_batch = dict_name_entry_to_dict_int_entry_to_int_entry_new_after_sorting[ 'features' if flag_ramtx_sorted_by_id_feature else 'barcodes' ][ l_index_sorting_current_batch[ 0 ] ] # retrieve the new index for the first record
            if flag_ramtx_sorted_by_id_feature :
                arr_int_col += int_entry_sorted_first_entry_in_a_batch # since the entries were already sorted, indices reset by starting from 0, simply adding the new index of the first entry is sufficient for renumbering
                arr_int_row = vfunc_map_index_barcode( arr_int_row ) # renumber barcodes (new indices are indices after sorting)
            else :
                arr_int_col = vfunc_map_index_feature( arr_int_col ) # renumber features (new indices are indices after sorting)
                arr_int_row += int_entry_sorted_first_entry_in_a_batch # since the entries were already sorted, indices reset by starting from 0, simply adding the new index of the first entry is sufficient for renumbering
                
            # by default, AnnData.X sparsematrix is sorted by feature. therefore, it should be sorted by barcode if 'flag_ramtx_sorted_by_id_feature' is False
            if not flag_ramtx_sorted_by_id_feature :
                argsort_arr_int_row = arr_int_row.argsort( ) # retrieve sorting indices to sort values based on 'arr_int_row'
                # sort by 'arr_int_row'
                arr_int_row = arr_int_row[ argsort_arr_int_row ]
                arr_int_col = arr_int_col[ argsort_arr_int_row ]
                arr_value = arr_value[ argsort_arr_int_row ]

            # retrieve the start of the block, marked by the change of int_entry 
            l_pos_start_block = [ 0 ] + list( np.where( np.diff( arr_int_col if flag_ramtx_sorted_by_id_feature else arr_int_row ) )[ 0 ] + 1 ) + [ len( arr_value ) ] # np.diff decrease the index of entries where change happens, and +1 should be done 
            for index_block in range( len( l_pos_start_block ) - 1 ) : # for each block (each block contains records of a single entry)
                slice_for_the_current_block = slice( l_pos_start_block[ index_block ], l_pos_start_block[ index_block + 1 ] )
                arr_int_row_of_the_current_block, arr_int_col_of_the_current_block, arr_value_of_the_current_block = arr_int_row[ slice_for_the_current_block ], arr_int_col[ slice_for_the_current_block ], arr_value[ slice_for_the_current_block ] # retrieve data for the current block
                int_entry = arr_int_col_of_the_current_block[ 0 ] if flag_ramtx_sorted_by_id_feature else arr_int_row_of_the_current_block[ 0 ] # retrieve int_entry of the current block
                
                bytes_processed = func_arrays_mtx_to_processed_bytes( ( arr_int_col_of_the_current_block, arr_int_row_of_the_current_block, arr_value_of_the_current_block ) ) # convert arrays_mtx to processed bytes
                int_num_bytes_written_for_the_current_entry = len( bytes_processed ) # record the number of bytes of the written data
                # write the processed bytes to the output file
                file_output.write( bytes_processed )

                # write the index
                file_index_output.write( ( '\t'.join( map( str, [ int_entry + 1, int_num_bytes_written, int_num_bytes_written + int_num_bytes_written_for_the_current_entry ] ) ) + '\n' ).encode( ) ) # write an index for the current entry # 0>1 coordinates for 'int_entry'
                int_num_bytes_written += int_num_bytes_written_for_the_current_entry # update the number of bytes written

            return int_num_bytes_written

        for int_entry, float_weight in zip( l_arr_index_sorting_for_each_chunk[ index_chunk ], l_arr_weight_for_each_chunk[ index_chunk ] ) : # retrieve inputs for the current process
            # add current index_sorting to the current batch
            l_index_sorting_current_batch.append( int_entry )
            int_total_weight_current_batch += float_weight
            # if the weight becomes larger than the threshold, process the batch and reset the batch
            if int_total_weight_current_batch > int_total_weight_for_each_batch :
                # process the current batch
                int_num_bytes_written = __process_batch__( file_output, file_index_output, int_num_bytes_written, l_index_sorting_current_batch )
                # initialize the next batch
                l_index_sorting_current_batch = [ ]
                int_total_weight_current_batch = 0
                
        # process the remaining entries
        int_num_bytes_written = __process_batch__( file_output, file_index_output, int_num_bytes_written, l_index_sorting_current_batch )
        
        # close files
        for file in [ file_output, file_index_output ] :
            file.close( )
        
        # if no entries have been written, delete the output files
        if int_num_bytes_written == 0 :
            for path_file in [ f'{path_folder_temp}indexed.{index_chunk}.{str_etx}', f'{path_folder_temp}indexed.{index_chunk}.{str_etx}.{str_ext_index}' ] :
                os.remove( path_file )

    l_worker = list( mp.Process( target = __compress_and_index_a_portion_of_sparse_matrix_as_a_worker__, args = ( index_chunk, ) ) for index_chunk in range( int_num_threads ) )

    ''' start works and wait until all works are completed by workers '''
    for p in l_worker :
        p.start( ) # start workers
    for p in l_worker :
        p.join( )    

    ''' retrieve metadata from AnnData.X '''
    int_num_barcodes, int_num_features = adata.X.shape # AnnData sparse matrix: row  = barcode, col = feature
    int_num_records = adata.X.count_nonzero( ) # retrieve the number of entries

    ''' write the metadata row of the matrix '''
    with gzip.open( f'{path_folder_temp}indexed.-1.{str_etx}', 'wb' ) as file : # the metadata row should be located at the front of the output matrix file
        file.write( f"""%%MatrixMarket matrix coordinate integer general\n%\n{int_num_features} {int_num_barcodes} {int_num_records}\n""".encode( ) ) # 10X matrix: row = feature, col = barcode
    int_num_bytes_for_mtx_header = os.stat( f'{path_folder_temp}indexed.-1.{str_etx}' ).st_size # retrieve the file size of the matrix header

    ''' combine outputs matrix '''
    # combine output matrix files
    for str_filename_glob in [ f'indexed.*.{str_etx}' ] :
        # collect the list of input files in the order of 'index_chunk'
        df = GLOB_Retrive_Strings_in_Wildcards( f'{path_folder_temp}{str_filename_glob}' )
        df[ 'wildcard_0' ] = df.wildcard_0.astype( int )
        df.sort_values( 'wildcard_0', inplace = True ) # sort the list of input files in the order of 'index_chunk'
        # combine input files
        OS_Run( [ 'cat' ] + list( df.path.values ), path_file_stdout = f"{path_folder_temp}{str_filename_glob.replace( '.*.', '.' )}", stdout_binary = True )
        # delete input files
#         for path_file in df.path.values :
#             os.remove( path_file )
    ''' combine output index '''
    # collect the paths of the index files
    df_file = GLOB_Retrive_Strings_in_Wildcards( f'{path_folder_temp}indexed.*.{str_etx}.{str_ext_index}' )
    df_file[ 'wildcard_0' ] = df_file.wildcard_0.astype( int )
    df_file.sort_values( 'wildcard_0', inplace = True ) # sort the list of input files in the order of 'index_chunk'
    # update index information after concatenation of the chunks and combine index files
    l = [ ]
    int_num_bytes_written = int_num_bytes_for_mtx_header # consider the number of bytes for the written header line
    for path_file in df_file.path.values :
        # read index of the current chunk
        df_index = pd.read_csv( path_file, sep = '\t', header = None )
        df_index.columns = [ 'index_entry', 'int_pos_start', 'int_pos_end' ]
        # update index information after concatenation
        int_num_bytes_written_for_the_current_chunk = df_index.int_pos_end.values[ -1 ]
        df_index[ 'int_pos_start' ] = df_index.int_pos_start + int_num_bytes_written
        df_index[ 'int_pos_end' ] = df_index.int_pos_end + int_num_bytes_written
        int_num_bytes_written += int_num_bytes_written_for_the_current_chunk # update the number of bytes written after concatenation
        l.append( df_index )
    pd.concat( l ).to_csv( f'{path_folder_temp}indexed.{str_etx}.{str_ext_index}', sep = '\t', index = False )
    ''' delete temporary files '''
    if not flag_debugging :
        # delete temporary files
        for str_name_file_glob in [ f'indexed.*.{str_etx}.{str_ext_index}' ] :
            for path_file in glob.glob( f'{path_folder_temp}{str_name_file_glob}' ) :
                os.remove( path_file )

    ''' export features and barcodes '''
    # rename output files
    os.rename( f"{path_folder_temp}indexed.{str_etx}", f"{path_folder_ramtx_output}matrix.{str_etx}" )
    os.rename( f"{path_folder_temp}indexed.{str_etx}.{str_ext_index}", f"{path_folder_ramtx_output}matrix.{str_etx}.{str_ext_index}" )

    # delete temporary folder
    if not flag_debugging :
        shutil.rmtree( path_folder_temp ) 

    # export features and barcodes
    df_obs[ [ 'id_cell' ] ].to_csv( f"{path_folder_ramtx_output}barcodes.tsv.gz", sep = '\t', header = None, index = False )
    df_var[ [ 'id_feature', 'name_feature', 'feature_types' ] ].to_csv( f"{path_folder_ramtx_output}features.tsv.gz", sep = '\t', header = None, index = False )

    ''' export settings used for sort, indexing, and exporting '''
    dict_metadata = { 
        'path_folder_mtx_10x_input' : None,
        'flag_ramtx_sorted_by_id_feature' : flag_ramtx_sorted_by_id_feature,
        'str_completed_time' : TIME_GET_timestamp( True ),
        'flag_output_value_is_float' : flag_output_value_is_float,
        'int_num_digits_after_floating_point_for_export' : int_num_digits_after_floating_point_for_export,
        'int_num_features' : int_num_features,
        'int_num_barcodes' : int_num_barcodes,
        'int_num_records' : int_num_records,
        'file_format' : [ file_format ],
    }
    with open( f"{path_folder_ramtx_output}ramtx.metadata.json", 'w' ) as file :
        json.dump( dict_metadata, file )
def Convert_Scanpy_AnnData_to_RamData( adata, path_folder_ramdata_output, name_data = 'from_anndata', file_format = 'mtx_gzipped', int_num_threads = 15, verbose = False, flag_overwrite_existing_files = False, flag_sort_and_index_again = False, flag_debugging = False, flag_covert_dense_matrix = False, inplace = False, int_num_digits_after_floating_point_for_export = 5, flag_output_value_is_float = True, dtype_of_row_and_col_indices = np.uint32, dtype_of_value = None, flag_simultaneous_indexing_of_cell_and_barcode = True ) :
    ''' # 2022-05-14 16:45:56 
    build RAMtx from Scanpy AnnData. 
    It is recommended to load all data on the memory for efficient conversion. It is often not practical to use the disk-backed AnnData to build RAMtx due to slow access time of the object.
    
    inputs:
    ========
    'path_folder_ramdata_output' an output directory of RamData
    'name_data' : a name of the given data
    'file_format' : file_format of the output RAMtx objects. please check 'Convert_MTX_10X_to_RAMtx' function docstring for more details.
    'flag_covert_dense_matrix' : by default, only sparse matrix will be converted to RAMtx. However, dense matrix can be also converted to RAMtx if this flag is set to True
    'inplace' : the original anndata will be modified, and the count data stored at adata.X will be deleted.
    'flag_simultaneous_indexing_of_cell_and_barcode' : if True, create cell-sorted RAMtx and feature-sorted RAMtx simultaneously using two worker processes with the half of given 'int_num_threads'. it is generally recommended to turn this feature on, since the last step of the merge-sort is always single-threaded.
    'dtype_of_row_and_col_indices' (default: np.uint32), 'dtype_of_value' (default: None (auto-detect)) : numpy dtypes for pickle and feather outputs. choosing smaller format will decrease the size of the object on disk
    '''
    # create the RamData output folder
    os.makedirs( path_folder_ramdata_output, exist_ok = True ) 

    # build barcode- and feature-sorted RAMtx objects
    path_folder_data = f"{path_folder_ramdata_output}{name_data}/"
    if flag_simultaneous_indexing_of_cell_and_barcode :
        l_process = list( mp.Process( target = Convert_Scanpy_AnnData_to_RAMtx, args = ( adata, path_folder_ramtx_output, flag_ramtx_sorted_by_id_feature, file_format, int_num_threads_for_the_current_process, verbose, flag_overwrite_existing_files, flag_sort_and_index_again, flag_debugging, flag_covert_dense_matrix, int_num_digits_after_floating_point_for_export, flag_output_value_is_float, dtype_of_row_and_col_indices, dtype_of_value ) ) for path_folder_ramtx_output, flag_ramtx_sorted_by_id_feature, int_num_threads_for_the_current_process in zip( [ f"{path_folder_data}sorted_by_barcode/", f"{path_folder_data}sorted_by_feature/" ], [ False, True ], [ int( np.floor( int_num_threads / 2 ) ), int( np.ceil( int_num_threads / 2 ) ) ] ) )
        for p in l_process : p.start( )
        for p in l_process : p.join( )
    else :
        Convert_Scanpy_AnnData_to_RAMtx( adata, path_folder_ramtx_output = f"{path_folder_data}sorted_by_barcode/", flag_ramtx_sorted_by_id_feature = False, file_format = file_format, int_num_threads = int_num_threads, verbose = verbose, flag_overwrite_existing_files = flag_overwrite_existing_files, flag_sort_and_index_again = flag_sort_and_index_again, flag_debugging = flag_debugging, flag_covert_dense_matrix = flag_covert_dense_matrix, int_num_digits_after_floating_point_for_export = int_num_digits_after_floating_point_for_export, flag_output_value_is_float = flag_output_value_is_float, dtype_of_row_and_col_indices = dtype_of_row_and_col_indices, dtype_of_value = dtype_of_value )
        Convert_Scanpy_AnnData_to_RAMtx( adata, path_folder_ramtx_output = f"{path_folder_data}sorted_by_feature/", flag_ramtx_sorted_by_id_feature = True, file_format = file_format, int_num_threads = int_num_threads, verbose = verbose, flag_overwrite_existing_files = flag_overwrite_existing_files, flag_sort_and_index_again = flag_sort_and_index_again, flag_debugging = flag_debugging, flag_covert_dense_matrix = flag_covert_dense_matrix, int_num_digits_after_floating_point_for_export = int_num_digits_after_floating_point_for_export, flag_output_value_is_float = flag_output_value_is_float, dtype_of_row_and_col_indices = dtype_of_row_and_col_indices, dtype_of_value = dtype_of_value )
    
    ''' export anndata without any count data to the RamData output folder '''
    if not inplace :
        adata = adata.copy( ) # copy adata
    adata.X = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( ( [], ( [], [] ) ), shape = ( len( adata.obs ), len( adata.var ) ) ) ) # store the empty sparse matrix
    # sort barcode and features
    adata.obs = adata.obs.sort_index( )
    adata.var = adata.var.sort_values( 'gene_ids' )
    adata.write( f'{path_folder_ramdata_output}main.h5ad' )
    
    ''' copy features and barcode files of the output RAMtx object to the RamData root directory '''
    for name_file in [ 'features.tsv.gz', 'barcodes.tsv.gz' ] :
        OS_Run( [ 'cp', f"{path_folder_data}sorted_by_barcode/{name_file}", f"{path_folder_ramdata_output}{name_file}" ] )
''' methods for combining/analyzing RAMtx objects '''
def Convert_df_count_to_RAMtx( path_file_df_count, path_folder_ramtx_output, flag_ramtx_sorted_by_id_feature = True, int_num_threads = 15, flag_debugging = False, inplace = False ) :
    """ # 2022-05-12 17:37:15 
    Convert df_count from the scarab output to RAMtx count matrix object (in 'mtx_gzipped' format only) (this function is used internally by scarab).
    This function load an entire 'df_count' in memory, which makes processing of samples with large 'df_count' difficult.
    
    'path_file_df_count' : a df_count dataframe or a file path of the saved 'df_count'
    'path_folder_ramtx_output' : an output folder for exporting RAMtx data
    'flag_ramtx_sorted_by_id_feature' : sort by id_feature if 'flag_ramtx_sorted_by_id_feature' is True. else, sort by id_cell/barcode
    'int_num_threads' : number of threads for compressing and indexing RAMtx data
    'inplace' : create a copy of 'df_count' and does not modify 'df_count' while exporting the count matrix
    """
    ''' handle inputs '''
    if isinstance( path_file_df_count, str ) :
        df_count = pd.read_csv( path_file_df_count, sep = '\t' )
    elif isinstance( path_file_df_count, pd.DataFrame ) : # if dataframe is given, use the dataframe as df_count
        df_count = path_file_df_count
    else :
        OSError( f"invalid input for the argument {path_file_df_count}" )

    ''' create an RAMtx output folder '''
    os.makedirs( path_folder_ramtx_output, exist_ok = True )

    ''' save barcode file '''
    # retrieve list of barcodes
    arr_barcode = LIST_COUNT( df_count.barcode, duplicate_filter = None ).index.values
    pd.DataFrame( arr_barcode ).to_csv( f"{path_folder_ramtx_output}barcodes.tsv.gz", sep = '\t', index = False, header = False ) # does not need to add '-1' since the barcoded bam already has it

    ''' save feature file '''
    # compose a feature dataframe
    df_feature = df_count[ [ 'feature', 'id_feature' ] ].drop_duplicates( ignore_index = True )
    df_feature.sort_values( 'id_feature', inplace = True ) # sort by 'id_feature'
    df_feature[ '10X_type' ] = 'Gene Expression'
    df_feature[ [ 'id_feature', 'feature', '10X_type' ] ].to_csv( f"{path_folder_ramtx_output}features.tsv.gz", sep = '\t', index = False, header = False ) # save as a file
    # retrieve list of features
    arr_id_feature = df_feature.id_feature.values

    ''' prepare matrix file for indexing '''
    # convert feature and barcode to integer indices
    df_count.drop( columns = [ 'feature' ], inplace = True ) # drop unncessary columns
    df_count.id_feature = df_count.id_feature.apply( MAP.Map( dict( ( e, i + 1 ) for i, e in enumerate( arr_id_feature ) ) ).a2b ) # 0>1-based coordinates
    df_count.barcode = df_count.barcode.apply( MAP.Map( dict( ( e, i + 1 ) for i, e in enumerate( arr_barcode ) ) ).a2b ) # 0>1-based coordinates
    # sort df_count based on the entry 
    df_count.sort_values( 'id_feature' if flag_ramtx_sorted_by_id_feature else 'barcode', inplace = True )

    """ save df_count data as RAMtx """
    # use multiple processes
    # create a temporary folder
    path_folder_temp = f'{path_folder_ramtx_output}temp_{UUID( )}/'
    os.makedirs( path_folder_temp, exist_ok = True )

    # prepare multiprocessing
    name_col_entry = 'id_feature' if flag_ramtx_sorted_by_id_feature else 'barcode'
    s = df_count[ [ name_col_entry, 'read_count', ] ].groupby( [ name_col_entry ] ).sum( ).read_count.sort_index( ) # retrieve index_entry and weights for index_entry
    arr_index_entry, arr_index_entry_weight = s.index.values, s.values # use total counts as weight
    l_arr_index_entry_for_each_chunk, l_arr_weight_for_each_chunk = LIST_Split( arr_index_entry, int_num_threads, flag_contiguous_chunk = True, arr_weight_for_load_balancing = arr_index_entry_weight, return_split_arr_weight = True ) # perform load balancing using the total count for each entry as a weight

    # retrieve boundaries of blocks for each entry
    l_pos_start_block = [ 0 ] + list( np.where( np.diff( df_count[ name_col_entry ] ) )[ 0 ] + 1 ) + [ len( df_count ) ] # np.diff decrease the index of entries where change happens, and +1 should be done 

    # retrieve data of df_count
    arr_df = df_count[ [ 'barcode', 'id_feature', 'read_count' ] ].values

    # setting for the pipeline
    def __compress_and_index_a_portion_of_df_count_as_a_worker__( index_chunk ) :
        ''' # 2022-05-08 13:19:13 
        save a portion of a sparse matrix 
        '''
        # open output files
        file_output = open( f'{path_folder_temp}indexed.{index_chunk}.mtx.gz', 'wb' )
        file_index_output = gzip.open( f'{path_folder_temp}indexed.{index_chunk}.mtx.idx.tsv.gz', 'wb' )

        int_num_bytes_written = 0 # track the number of written bytes
        for index_entry, float_weight in zip( l_arr_index_entry_for_each_chunk[ index_chunk ], l_arr_weight_for_each_chunk[ index_chunk ] ) : # retrieve inputs for the current process
            # process the current entry        
            # retrieve data for the current entry
            arr_index_col, arr_index_row, arr_value = arr_df[ l_pos_start_block[ index_entry - 1 ] : l_pos_start_block[ index_entry ] ].T

            # compress the data for the entry 
            gzip_file = io.BytesIO( )

            with gzip.GzipFile( fileobj = gzip_file, mode = 'w' ) as file :
                file.write( ( '\n'.join( list( str( index_row ) + ' ' + str( index_col ) + ' ' + str( value ) for index_row, index_col, value in zip( arr_index_row, arr_index_col, arr_value ) ) ) + '\n' ).encode( ) ) # read the data for the entry from input file # assumes all values of the count matrix is an integer
            int_num_bytes_written_for_the_current_entry = gzip_file.tell( ) # record the number of bytes of the compressed data

            # write the compressed file to the output file
            gzip_file.seek( 0 )
            file_output.write( gzip_file.read( ) )
            gzip_file.close( )

            # write the index
            file_index_output.write( ( '\t'.join( map( str, [ index_entry, int_num_bytes_written, int_num_bytes_written + int_num_bytes_written_for_the_current_entry ] ) ) + '\n' ).encode( ) ) # write an index for the current entry
            int_num_bytes_written += int_num_bytes_written_for_the_current_entry # update the number of bytes written

        # close files
        for file in [ file_output, file_index_output ] :
            file.close( )

    l_worker = list( mp.Process( target = __compress_and_index_a_portion_of_df_count_as_a_worker__, args = ( index_chunk, ) ) for index_chunk in range( int_num_threads ) )

    ''' start works and wait until all works are completed by workers '''
    for p in l_worker :
        p.start( ) # start workers
    for p in l_worker :
        p.join( )  

    ''' retrieve metadata from df_count ('header') '''
    int_num_rows, int_num_columns, int_num_records = len( arr_id_feature ), len( arr_barcode ), len( df_count ) # 10X matrix: row = feature, col = barcode

    ''' write the metadata row of the matrix '''
    with gzip.open( f'{path_folder_temp}indexed.-1.mtx.gz', 'wb' ) as file : # the metadata row should be located at the front of the output matrix file
        file.write( f"""%%MatrixMarket matrix coordinate integer general\n%\n{int_num_rows} {int_num_columns} {int_num_records}\n""".encode( ) )
    int_num_bytes_for_mtx_header = os.stat( f'{path_folder_temp}indexed.-1.mtx.gz' ).st_size # retrieve the file size of the matrix header

    ''' combine outputs matrix '''
    # combine output matrix files
    for str_filename_glob in [ 'indexed.*.mtx.gz' ] :
        # collect the list of input files in the order of 'index_chunk'
        df = GLOB_Retrive_Strings_in_Wildcards( f'{path_folder_temp}{str_filename_glob}' )
        df[ 'wildcard_0' ] = df.wildcard_0.astype( int )
        df.sort_values( 'wildcard_0', inplace = True ) # sort the list of input files in the order of 'index_chunk'
        # combine input files
        OS_Run( [ 'cat' ] + list( df.path.values ), path_file_stdout = f"{path_folder_temp}{str_filename_glob.replace( '.*.', '.' )}", stdout_binary = True )
        # delete input files
        for path_file in df.path.values :
            os.remove( path_file )
    ''' combine output index '''
    # collect the paths of the index files
    df_file = GLOB_Retrive_Strings_in_Wildcards( f'{path_folder_temp}indexed.*.mtx.idx.tsv.gz' )
    df_file[ 'wildcard_0' ] = df_file.wildcard_0.astype( int )
    df_file.sort_values( 'wildcard_0', inplace = True ) # sort the list of input files in the order of 'index_chunk'
    # update index information after concatenation of the chunks and combine index files
    l = [ ]
    int_num_bytes_written = int_num_bytes_for_mtx_header # consider the number of bytes for the written header line
    for path_file in df_file.path.values :
        # read index of the current chunk
        df_index = pd.read_csv( path_file, sep = '\t', header = None )
        df_index.columns = [ 'index_entry', 'int_pos_start', 'int_pos_end' ]
        # update index information after concatenation
        int_num_bytes_written_for_the_current_chunk = df_index.int_pos_end.values[ -1 ]
        df_index[ 'int_pos_start' ] = df_index.int_pos_start + int_num_bytes_written
        df_index[ 'int_pos_end' ] = df_index.int_pos_end + int_num_bytes_written
        int_num_bytes_written += int_num_bytes_written_for_the_current_chunk # update the number of bytes written after concatenation
        l.append( df_index )
    pd.concat( l ).to_csv( f'{path_folder_temp}indexed.mtx.idx.tsv.gz', sep = '\t', index = False )
    ''' delete temporary files '''
    if not flag_debugging :
        # delete temporary files
        for str_name_file_glob in [ 'indexed.*.mtx.idx.tsv.gz' ] :
            for path_file in glob.glob( f'{path_folder_temp}{str_name_file_glob}' ) :
                os.remove( path_file )

    ''' export results to the output folder '''
    # rename output files
    os.rename( f"{path_folder_temp}indexed.mtx.gz", f"{path_folder_ramtx_output}matrix.mtx.gz" )
    os.rename( f"{path_folder_temp}indexed.mtx.idx.tsv.gz", f"{path_folder_ramtx_output}matrix.mtx.gz.idx.tsv.gz" )

    # delete temporary folder
    shutil.rmtree( path_folder_temp ) 

    ''' export settings used for sort and indexing '''
    dict_metadata = { 
        'path_folder_mtx_10x_input' : None,
        'flag_ramtx_sorted_by_id_feature' : flag_ramtx_sorted_by_id_feature,
        'str_completed_time' : TIME_GET_timestamp( True ),
        'int_num_features' : int_num_rows,
        'int_num_barcodes' : int_num_columns,
        'int_num_records' : int_num_records,
        'file_format' : [ 'mtx_gzipped' ],
    }
    with open( f"{path_folder_ramtx_output}ramtx.metadata.json", 'w' ) as file :
        json.dump( dict_metadata, file )
