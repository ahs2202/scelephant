# core
from biobookshelf.main import *
from biobookshelf import *
import biobookshelf as bk
pd.options.mode.chained_assignment = None  # default='warn' # to disable worining
import zarr # SCElephant is currently implemented using Zarr
import numcodecs
import anndata
import scanpy
import shelve # for persistent database (key-value based database)
import sklearn as skl
from pynndescent import NNDescent # for fast approximate kNN search

from numba import jit # for speed up

# dimension reduction / clustering
import umap.parametric_umap as pumap # parametric UMAP
import hdbscan # for clustering

# for fast gzip compression/decompression
import pgzip

# might not be used
import scipy.spatial.distance as dist # for distance metrics calculation
import sklearn.cluster as skc # K-means

# define version
_version_ = '0.0.2'
_scelephant_version_ = _version_
_last_modified_time_ = '2022-07-21 10:35:32'

""" # 2022-07-21 10:35:42  realease note

HTTP hosted RamData subclustering tested and verified.
identified problems: Zarr HTTP Store is not thread-safe (crashes the program on run time) nor process-safe (dead lock... why?).
Therefore, serializing access to Zarr HTTP Store is required 

TODO: do not use fork when store is Zarr HTTP Store
TODO: use Pynndescent for efficient HDBSCAN cluster label assignment, even when program is single-threaded (when Zarr HTTP store is used)
TODO: use Pynndescent for better subsampling of cells before UMAP embedding

if these implementations are in place, subclustering on HTTP-hosted RamData will be efficient and accurate.
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
                    index_row, index_col, int_value = tuple( map( int, line.strip( ).split( ) ) ) # parse each entry of the current matrix 
                    
                    newfile.write( ( ' '.join( tuple( map( str, ( [ dict_id_entry_to_index_entry[ arr_id_entry[ index_row - 1 ] ], index_col + int_total_n_entries_of_previously_written_matrices ] if flag_renumber_feature_index else [ index_row + int_total_n_entries_of_previously_written_matrices, dict_id_entry_to_index_entry[ arr_id_entry[ index_col - 1 ] ] ] ) + [ int_value ] ) ) ) + '\n' ).encode( ) ) # translate indices of the current matrix to that of the combined matrix            
                    line = file.readline( ).decode( ) # read the next line
def MTX_10X_Combine( path_folder_mtx_10x_output, * l_path_folder_mtx_10x_input, int_num_threads = 15, flag_split_mtx = True, flag_split_mtx_again = False, int_max_num_entries_for_chunk = 10000000, flag_low_memory_mode_because_there_is_no_shared_cell_between_mtxs = None, flag_low_memory_mode_because_there_is_no_shared_feature_between_mtxs = None, verbose = False ) :
    '''
    # 2022-05-16 11:39:24 
    Combine 10X count matrix files from the given list of folders and write combined output files to the given output folder 'path_folder_mtx_10x_output'
    If there are no shared cells between matrix files, a low-memory mode will be used. The output files will be simply combined since no count summing operation is needed. Only feature matrix will be loaded and updated in the memory.
    'id_feature' should be unique across all features
    
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
            with gzip.open( f'{path_folder_mtx_10x}matrix.mtx.gz', 'rb' ) as file : # retrieve a list of features
                file.readline( ), file.readline( )
                int_total_n_records += int( file.readline( ).decode( ).strip( ).split( )[ 2 ] ) # update the total number of entries

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
    
    Returns:
    int_num_rows, int_num_columns, int_num_entries
    """
    ''' handle inputs '''
    if path_folder_mtx_10x_input[ -1 ] != '/' :
        path_folder_mtx_10x_input += '/'

    # define input file directories
    path_file_input_mtx = f'{path_folder_mtx_10x_input}matrix.mtx.gz'
    
    # check whether all required files are present
    if sum( list( not os.path.exists( path_folder ) for path_folder in [ path_file_input_mtx ] ) ) :
        if verbose :
            print( f'required file(s) is not present at {path_folder_mtx_10x}' )
    
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
    ''' # 2022-07-07 21:01:59 
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
    """ # 2022-07-20 16:48:27  
    copy zarr object of the soruce chunks by chunks along the primary axis (axis 0)
    
    'path_folder_zarr_source' : source zarr object path
    'path_folder_zarr_sink' : sink zarr object path
    'int_num_chunks_per_batch' : number of chunks along the primary axis (axis 0) to be copied for each loop. for example, when the size of an array is (100, 100), chunk size is (10, 10), and 'int_num_chunks_per_batch' = 1, 10 chunks along the secondary axis (axis = 1) will be saved for each batch.
    """
    za = zarr.open( path_folder_zarr_source )
    za_sink = zarr.open( path_folder_zarr_sink, mode = 'w', shape = za.shape, chunks = za.chunks, dtype = za.dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # open the output zarr
    
    int_total_num_rows = za.shape[ 0 ]
    int_num_rows_in_batch = za.chunks[ 0 ] * int( int_num_chunks_per_batch )
    for index_batch in range( int( np.ceil( int_total_num_rows / int_num_rows_in_batch ) ) ) :
        sl = slice( index_batch * int_num_rows_in_batch, ( index_batch + 1 ) * int_num_rows_in_batch )
        za_sink[ sl ] = za[ sl ] # copy batch by batch

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
    """ # 2022-07-20 23:05:55 
    storage-based persistant DataFrame backed by Zarr persistent arrays.
    each column can be separately loaded, updated, and unloaded.
    a filter can be set, which allows updating and reading ZarrDataFrame as if it only contains the rows indicated by the given filter.
    currently supported dtypes are bool, float, int, strings (will be interpreted as categorical data).
    the one of the functionality of this class is to provide a Zarr-based dataframe object that is compatible with Zarr.js (javascript implementation of Zarr), with a categorical data type (the format used in zarr is currently not supported in zarr.js) compatible with zarr.js.
    
    Of note, secondary indexing (row indexing) is always applied to unfiltered columns, not to a subset of column containing filtered rows.
    '__getitem__' function is thread and process-safe, while '__setitem__' is not thread nor prosess-safe. 
    
    # 2022-07-04 10:40:14 implement handling of categorical series inputs/categorical series output. Therefore, convertion of ZarrDataFrame categorical data to pandas categorical data should occurs only when dataframe was given as input/output is in dataframe format.
    # 2022-07-04 10:40:20 also, implement a flag-based switch for returning series-based outputs
    # 2022-07-20 22:29:41 : mask functionality was added
    
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
    """
    def __init__( self, path_folder_zdf, df = None, int_num_rows = None, int_num_rows_in_a_chunk = 10000, ba_filter = None, flag_enforce_name_col_with_only_valid_characters = False, flag_store_string_as_categorical = True, flag_retrieve_categorical_data_as_integers = False, flag_load_data_after_adding_new_column = True, mode = 'a', path_folder_mask = None, flag_is_read_only = False ) :
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
        self._flag_retrieve_categorical_data_as_integers = flag_retrieve_categorical_data_as_integers
        self._flag_load_data_after_adding_new_column = flag_load_data_after_adding_new_column
        self._ba_filter = None # initialize the '_ba_filter' attribute
        self.filter = ba_filter
        
        # open or initialize zdf and retrieve associated metadata
        if not zarr_exists( path_folder_zdf ) : # if the object does not exist, initialize ZarrDataFrame
            # create the output folder
            os.makedirs( path_folder_zdf, exist_ok = True )
            
            self._root = zarr.open( path_folder_zdf, mode = 'a' )
            self._dict_zdf_metadata = { 'version' : _version_, 'columns' : set( ), 'int_num_rows_in_a_chunk' : int_num_rows_in_a_chunk, 'flag_enforce_name_col_with_only_valid_characters' : flag_enforce_name_col_with_only_valid_characters, 'flag_store_string_as_categorical' : flag_store_string_as_categorical  } # to reduce the number of I/O operations from lookup, a metadata dictionary will be used to retrieve/update all the metadata
            # if 'int_num_rows' has been given, add it to the metadata
            if int_num_rows is not None :
                self._dict_zdf_metadata[ 'int_num_rows' ] = int_num_rows
            self._save_metadata_( ) # save metadata
        else :
            # read existing zdf object
            self._root = zarr.open( path_folder_zdf, mode = 'a' )
                
            # retrieve metadata
            self._dict_zdf_metadata = self._root.attrs[ 'dict_zdf_metadata' ] 
            # convert 'columns' list to set
            if 'columns' in self._dict_zdf_metadata :
                self._dict_zdf_metadata[ 'columns' ] = set( self._dict_zdf_metadata[ 'columns' ] )
        
        # if a mask is given, open the mask zdf
        self._mask = None # initialize 'mask'
        if path_folder_mask is not None : # if a mask is given
            self._mask = ZarrDataFrame( path_folder_mask, df = df, int_num_rows = self.n_rows, int_num_rows_in_a_chunk = self.metadata[ 'int_num_rows_in_a_chunk' ], ba_filter = ba_filter, flag_enforce_name_col_with_only_valid_characters = self.metadata[ 'flag_enforce_name_col_with_only_valid_characters' ], flag_store_string_as_categorical = self.metadata[ 'flag_store_string_as_categorical' ], flag_retrieve_categorical_data_as_integers = flag_retrieve_categorical_data_as_integers, flag_load_data_after_adding_new_column = flag_load_data_after_adding_new_column, mode = 'a', path_folder_mask = None, flag_is_read_only = False ) # the mask ZarrDataFrame shoud not have mask, should be modifiable, and not mode == 'r'.
        
        # handle input arguments
        self._str_invalid_char = '! @#$%^&*()-=+`~:;[]{}\|,<.>/?' + '"' + "'" if self._dict_zdf_metadata[ 'flag_enforce_name_col_with_only_valid_characters' ] else '/' # linux file system does not allow the use of linux'/' character in the folder/file name
        
        # initialize loaded data
        self._loaded_data = dict( )
        
        # initialize temp folder
        self._initialize_temp_folder_( )
        
        if isinstance( df, pd.DataFrame ) : # if a valid pandas.dataframe has been given
            # update zdf with the given dataframe
            self.update( df )
            
        # initialize attribute storing columns as dictionaries
        self.dict = dict( )
    @property
    def metadata( self ) :
        ''' # 2022-07-21 02:38:31 
        '''
        return self._dict_zdf_metadata
    def _initialize_temp_folder_( self ) :
        """ # 2022-07-20 10:50:19 
        empty the temp folder
        """
        if self._flag_is_read_only : # ignore the operation if the current object is read-only
            self._path_folder_temp = None # initialize the temp folder
            return
        # initialize temporary data folder directory
        self._path_folder_temp = f"{self._path_folder_zdf}.__temp__/" # set temporary folder
        if os.path.exists( self._path_folder_temp ) : # if temporary folder already exists
            OS_Run( [ 'rm', '-rf', self._path_folder_temp ] ) # delete temporary folder
        os.makedirs( self._path_folder_temp, exist_ok = True ) # recreate temporary folder
    @property
    def _n_rows_unfiltered( self ) :
        """ # 2022-06-22 23:12:09 
        retrieve the number of rows in unfiltered ZarrDataFrame. return None if unavailable.
        """
        if 'int_num_rows' not in self._dict_zdf_metadata : # if 'int_num_rows' has not been set, return None
            return None
        else : # if 'int_num_rows' has been set
            return self._dict_zdf_metadata[ 'int_num_rows' ]
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
        """ # 2022-07-21 08:58:16 
        change filter, and empty the cache
        """
        if ba_filter is None : # if filter is removed, 
            # if the filter was present before the filter was removed, empty the cache and the temp folder
            if self.filter is not None :
                self._loaded_data = dict( ) # empty the cache
                self._initialize_temp_folder_( ) # empty the temp folder
                self.dict = dict( ) # empty the cache for columns stored as dictionaries
            self._ba_filter = None
            self._n_rows_after_applying_filter = None
        else :
            # check whether the given filter is bitarray
            if isinstance( ba_filter, np.ndarray ) : # convert numpy array to bitarray
                ba_filter = bk.BA.to_bitarray( ba_filter )
            assert isinstance( ba_filter, bitarray ) # make sure that the input value is a bitarray object
            
            # check the length of filter bitarray
            if 'int_num_rows' not in self._dict_zdf_metadata : # if 'int_num_rows' has not been set, set 'int_num_rows' using the length of the filter bitarray
                self._dict_zdf_metadata[ 'int_num_rows' ] = len( ba_filter )
                self._save_metadata_( ) # save metadata
            else :
                # check the length of filter bitarray
                assert len( ba_filter ) == self._dict_zdf_metadata[ 'int_num_rows' ]

            self._loaded_data = dict( ) # empty the cache
            self._initialize_temp_folder_( ) # empty the temp folder
            self.dict = dict( ) # empty the cache for columns stored as dictionaries
            self._n_rows_after_applying_filter = ba_filter.count( ) # retrieve the number of rows after applying the filter

            self._ba_filter = ba_filter # set bitarray filter
        # set filter of mask
        if hasattr( self, '_mask' ) and self._mask is not None : # propagate filter change to the mask ZDF
            self._mask.filter = ba_filter
    def __getitem__( self, args ) :
        ''' # 2022-07-21 09:22:36 
        retrieve data of a column.
        partial read is allowed through indexing (slice/integer index/boolean mask/bitarray is supported)
        when a filter is active, the filtered data will be cached in the temporary directory as a Zarr object and will be retrieved in subsequent accesses
        if mask is set, retrieve data from the mask if the column is available in the mask
        '''
        """
        1) parse arguments
        """
        # initialize indexing
        flag_indexing = False # a boolean flag indicating whether an indexing is active
        flag_coords_in_bool_mask = False
        # parse arguments
        if isinstance( args, tuple ) :
            flag_indexing = True # update the flag
            name_col, coords = args # parse the args
            # detect boolean mask
            flag_coords_in_bool_mask = bk.BA.detect_boolean_mask( coords )
            # convert boolean masks to np.ndarray object
            if flag_coords_in_bool_mask :
                coords = bk.BA.convert_mask_to_array( coords )
        else :
            # when indexing is not active
            name_col, coords = args, slice( None, None, None ) # retrieve all data if only 'name_col' is given

        """
        2) retrieve data
        """
        if self._mask is not None and name_col in self._mask : # if mask is available and the column is available in the mask, return the data in the mask
            return self._mask[ args ]
        elif name_col in self : # if name_col is valid
            if name_col in self._loaded_data and not flag_indexing : # if a loaded data is available and indexing is not active, return the cached data
                return self._loaded_data[ name_col ]
            else : # in other cases, read the data from Zarr object
                if self.filter is None or flag_indexing : # if filter is not active or indexing is active, open a Zarr object containing all data
                    za = zarr.open( f"{self._path_folder_zdf}{name_col}/", mode = 'r' ) # read data from the Zarr object
                else : # if filter is active and indexing is not active
                    flag_is_cache_available = False # initialize a flag
                    if self._path_folder_temp is not None : # when mode is not read-only
                        path_folder_temp_zarr = f"{self._path_folder_temp}{name_col}/" # retrieve path of cached zarr object containing filtered data
                        if zarr_exists( path_folder_temp_zarr ) : # if a cache is available
                            za = zarr.open( path_folder_temp_zarr, mode = 'r' ) # open the cached Zarr object containing filtered data
                            flag_is_cache_available = True # update the flag
                    if not flag_is_cache_available : # if a cache is not available, retrieve filtered data and write a cache to disk
                        za = zarr.open( f"{self._path_folder_zdf}{name_col}/", mode = 'r' ) # read data from the Zarr object
                        za = za.get_mask_selection( bk.BA.to_array( self.filter ) ) # retrieve filtered data as np.ndarray
                        if self._path_folder_temp is not None :
                            za_cached = zarr.open( path_folder_temp_zarr, 'w', shape = ( self._n_rows_after_applying_filter, ), chunks = ( self._dict_zdf_metadata[ 'int_num_rows_in_a_chunk' ], ), dtype = za.dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # open a new Zarr object for caching # overwriting existing data # use the same dtype as the parent dtype
                            za_cached[ : ] = za # save the filtered data for caching
                values = za[ coords ] if not flag_coords_in_bool_mask or isinstance( za, np.ndarray ) else za.get_mask_selection( coords ) # retrieve data using the 'get_mask_selection' method if 'flag_coords_in_bool_mask' is True and 'za' is not np.ndarray # when indexing is not active, coords = slice( None, None, None )
                
                # check whether the current column contains categorical data
                l_value_unique = self.get_categories( name_col )
                if len( l_value_unique ) == 0 or self._flag_retrieve_categorical_data_as_integers : # handle non-categorical data
                    return values
                else : # handle categorical data
                    values_decoded = np.zeros( len( values ), dtype = object ) # initialize decoded values
                    for i in range( len( values ) ) :
                        values_decoded[ i ] = l_value_unique[ values[ i ] ] if values[ i ] >= 0 else np.nan # convert integer representations to its original string values # -1 (negative integers) encodes np.nan
                    return values_decoded
    def __setitem__( self, args, values ) :
        ''' # 2022-07-20 23:23:47 
        save/update a column at indexed positions.
        when a filter is active, only active entries will be saved/updated automatically.
        boolean mask/integer arrays/slice indexing is supported. However, indexing will be applied to the original column with unfiltered rows (i.e., when indexing is active, filter will be ignored)
        if mask is set, save data to the mask
        
        automatically detect dtype of the input array/list, including that of categorical data (all string data will be interpreted as categorical data). when the original dtype and dtype inferred from the updated values are different, an error will occur.
        '''
        """
        1) parse arguments
        """
        if self._mode == 'r' : # if mode == 'r', ignore __setitem__ method calls
            return 
        
        # initialize indexing
        flag_indexing = False # a boolean flag indicating whether an indexing is active
        flag_coords_in_bool_mask = False
        # parse arguments
        if isinstance( args, tuple ) :
            flag_indexing = True # update the flag
            name_col, coords = args # parse the args
            # detect boolean mask
            flag_coords_in_bool_mask = bk.BA.detect_boolean_mask( coords )
            # convert boolean masks to np.ndarray object
            if flag_coords_in_bool_mask :
                coords = bk.BA.convert_mask_to_array( coords )
        else :
            name_col, coords = args, slice( None, None, None ) # retrieve all data if only 'name_col' is given

        # check whether the given name_col contains invalid characters(s)
        for char_invalid in self._str_invalid_char :
            if char_invalid in name_col :
                raise TypeError( f"the character '{char_invalid}' cannot be used in 'name_col'. Also, the 'name_col' cannot contains the following characters: {self._str_invalid_char}" )
        
        """
        2) set data
        """
        # if mask is available, save new data to the mask
        if self._mask is not None : # if mask is available, save new data to the mask
            if name_col in self and name_col not in self._mask : # if the 'name_col' exists in the current ZarrDataFrame and not in mask, copy the column to the mask
                zarr_copy( f"{self._path_folder_zdf}{name_col}/", f"{self._mask._path_folder_zdf}{name_col}/" ) # copy zarr object from the source to the mask
            self._mask[ args ] = values # set values to the mask
            return # exit
    
        if self._flag_is_read_only : # if current store is read-only (and mask is not set), exit
            return # exit
        
        # ❤️❤️❤️ implement 'broadcasting'; when a single value is given, the value will be copied to all rows.
        # retrieve data values from the 'values' 
        if isinstance( values, bitarray ) :
            values = bk.BA.to_array( values ) # retrieve boolean values from the input bitarray
        if isinstance( values, pd.Series ) :
            values = values.values
        # retrieve data type of values
        # if values is numpy.ndarray, use the dtype of the array
        if isinstance( values, np.ndarray ) :
            dtype = values.dtype
        # if values is not numpy.ndarray or the dtype is object datatype, use the type of the data returned by the type( ) python function.
        if not isinstance( values, np.ndarray ) or dtype is np.dtype( 'O' ) : 
            dtype = type( values[ 0 ] )
            # handle the case when the first element is np.nan
            if dtype is float and values[ 0 ] is np.nan :
                # examine the values and set the dtype to string if at least one value is string
                for e in values : 
                    if type( e ) is str :
                        dtype = str
                        break
        # convert values that is not numpy.ndarray to numpy.ndarray object (for the consistency of the loaded_data)
        if not isinstance( values, np.ndarray ) :
            values = np.array( values, dtype = object if dtype is str else dtype ) # use 'object' dtype when converting values to a numpy.ndarray object if dtype is 'str'
        
        # check whether the number of values is valid
        int_num_values = len( values )
        if self.n_rows is not None : # if a valid information about the number of rows is available
            # check the number of rows of the current Zdf (after applying filter, if a filter is available)
#             assert int_num_values == self.n_rows
            pass # currently validality will not be checked # implemented in the future ❤️❤️❤️
        else :
            self._dict_zdf_metadata[ 'int_num_rows' ] = int_num_values # record the number of rows of the dataframe
            self._save_metadata_( ) # save metadata
        
        # define zarr object directory
        path_folder_col = f"{self._path_folder_zdf}{name_col}/" # compose the output folder
        # retrieve/initialize metadata
        flag_col_already_exists = os.path.exists( path_folder_col ) # retrieve a flag indicating that the column already exists
        if flag_col_already_exists :
            za = zarr.open( path_folder_col, 'a' ) # open Zarr object
            dict_col_metadata = za.attrs[ 'dict_col_metadata' ] # load previous written metadata
        else :
            dict_col_metadata = { 'flag_categorical' : False } # set a default value for 'flag_categorical' metadata attribute
            dict_col_metadata[ 'flag_filtered' ] = self.filter is not None # mark the column containing filtered data
        
        # write categorical data
        if dtype is str and self._dict_zdf_metadata[ 'flag_store_string_as_categorical' ] : # storing categorical data            
            # compose metadata of the column
            dict_col_metadata[ 'flag_categorical' ] = True # set metadata for categorical datatype
            
            set_value_unique = set( values ) # retrieve a set of unique values
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
            za = zarr.open( path_folder_col, mode = 'a', synchronizer = zarr.ThreadSynchronizer( ) ) if os.path.exists( path_folder_col ) else zarr.open( path_folder_col, mode = 'w', shape = ( self._n_rows_unfiltered, ), chunks = ( self._dict_zdf_metadata[ 'int_num_rows_in_a_chunk' ], ), dtype = dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # create a new Zarr object if the object does not exist.
            
            # if dtype changed from the previous zarr object, re-write the entire Zarr object with changed dtype. (this will happens very rarely, and will not significantly affect the performance)
            if dtype != za.dtype : # dtype should be larger than za.dtype if they are not equal (due to increased number of bits required to encode categorical data)
                print( f'{za.dtype} will be changed to {dtype}' )
                path_folder_col_new = f"{self._path_folder_zdf}{name_col}_{UUID( )}/" # compose the new output folder
                za_new = zarr.open( path_folder_col_new, mode = 'w', shape = ( self._n_rows_unfiltered, ), chunks = ( self._dict_zdf_metadata[ 'int_num_rows_in_a_chunk' ], ), dtype = dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # create a new Zarr object using the new dtype
                za_new[ : ] = za[ : ] # copy the data 
                shutil.rmtree( path_folder_col ) # delete the previous Zarr object
                os.rename( path_folder_col_new, path_folder_col ) # replace the previous Zarr object with the new object
                za = zarr.open( path_folder_col, mode = 'a', synchronizer = zarr.ThreadSynchronizer( ) ) # open the new Zarr object
            
            # encode data
            dict_encode_category = dict( ( e, i ) for i, e in enumerate( l_value_unique ) ) # retrieve a dictionary encoding value to integer representation of the value
            values_encoded = np.array( list( dict_encode_category[ value ] if value in dict_encode_category else -1 for value in values ), dtype = dtype ) # retrieve encoded values # np.nan will be encoded as -1 values
            if self._flag_retrieve_categorical_data_as_integers : # if 'self._flag_retrieve_categorical_data_as_integers' is True, use integer representation of values for caching
                values = values_encoded
            
            # write data 
            if self.filter is None or flag_indexing : # when filter is not set or indexing is active
                if flag_coords_in_bool_mask : # indexing with mask
                    za.set_mask_selection( coords, values_encoded )  # save partial data 
                else : # indexing with slice or integer coordinates
                    za[ coords ] = values_encoded # write encoded data
            else : # when filter is present and indexing is not active
                za.set_mask_selection( bk.BA.to_array( self.filter ), values_encoded )  # save filtered data 
                
        else : # storing non-categorical data
            za = zarr.open( path_folder_col, mode = 'a', synchronizer = zarr.ThreadSynchronizer( ) ) if os.path.exists( path_folder_col ) else zarr.open( path_folder_col, mode = 'w', shape = ( self._n_rows_unfiltered, ), chunks = ( self._dict_zdf_metadata[ 'int_num_rows_in_a_chunk' ], ), dtype = dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # create a new Zarr object if the object does not exist.
            
            # write data
            if self.filter is None or flag_indexing : # when filter is not set or indexing is active
                if flag_coords_in_bool_mask : # indexing with mask
                    za.set_mask_selection( coords, values )  # write data 
                else : # indexing with slice or integer coordinates
                    za[ coords ] = values # write data
            else : # when filter is present
                za.set_mask_selection( bk.BA.to_array( self.filter ), values ) # save filtered data 
            
        # save column metadata
        za.attrs[ 'dict_col_metadata' ] = dict_col_metadata
        
        # update zdf metadata
        if name_col not in self._dict_zdf_metadata[ 'columns' ] :
            self._dict_zdf_metadata[ 'columns' ].add( name_col )
            self._save_metadata_( )
        
        # if indexing was used to partially update the data, remove the cache, because it can cause in consistency
        if flag_indexing and name_col in self._loaded_data :
            del self._loaded_data[ name_col ]
        # add data to the loaded data dictionary (object cache) if 'self._flag_load_data_after_adding_new_column' is True and indexing was not used
        if self._flag_load_data_after_adding_new_column and not flag_indexing : 
            self._loaded_data[ name_col ] = values
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
            self._dict_zdf_metadata[ 'columns' ].remove( name_col )
            self._save_metadata_( ) # update metadata
            # delete the column from the disk ZarrDataFrame object
            shutil.rmtree( f"{self._path_folder_zdf}{name_col}/" ) #             OS_Run( [ 'rm', '-rf', f"{self._path_folder_zdf}{name_col}/" ] )
    def __repr__( self ) :
        """ # 2022-07-20 23:00:15 
        """
        return f"<ZarrDataFrame object stored at {self._path_folder_zdf}\n\twith the following columns: {sorted( self._dict_zdf_metadata[ 'columns' ] )}>"
    @property
    def columns( self ) :
        ''' # 2022-07-20 23:01:48 
        return available column names as a set
        '''
        if self._mask is not None : # if mask is available :
            return self._dict_zdf_metadata[ 'columns' ].union( self._mask._dict_zdf_metadata[ 'columns' ] ) # return the column names of the current ZDF and the mask ZDF
        else :
            return self._dict_zdf_metadata[ 'columns' ]
    def __contains__( self, name_col ) :
        """ # 2022-07-20 22:56:19 
        check whether a column name exists in the given ZarrDataFrame
        """
        if self._mask is not None : # if mask is available :
            return name_col in self.columns or name_col in self._mask # search columns in the current ZDF and the mask ZDF
        else :
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
    def _save_metadata_( self ) :
        ''' # 2022-07-20 10:31:39 
        save metadata of the current ZarrDataFrame
        '''
        if not self._flag_is_read_only : # save metadata only when it is not in the read-only mode
            # convert 'columns' to list before saving attributes
            temp = self._dict_zdf_metadata[ 'columns' ]
            self._dict_zdf_metadata[ 'columns' ] = list( temp )
            self._root.attrs[ 'dict_zdf_metadata' ] = self._dict_zdf_metadata # update metadata
            self._dict_zdf_metadata[ 'columns' ] = temp # revert 'columns' to set
    def get_categories( self, name_col ) :
        """ # 2022-06-21 00:57:37 
        for columns with categorical data, return categories. if the column contains non-categorical data, return an empty list
        """
        if name_col in self : # if the current column is valid
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
        """ # 2022-07-02 09:22:40 
        get dataframe of a given list of columns, and empty the cache
        """
        set_name_col = set( l_name_col ) # convert 'l_name_col' to set
        self.unload( * list( name_col for name_col in self if name_col not in set_name_col ) ) # drop the columns that do not belonging to 'l_name_col'
        self.load( * l_name_col ) # load the given list of columns
        df = self.df # retrieve dataframe
        self.unload( ) # empty the cache
        return df
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
            zdf[ name_col ] = self[ name_col ]
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
            
''' methods for creating RAMtx objects '''
def __Merge_Sort_MTX_10X_and_Write_and_Index_Zarr__( za_mtx, za_mtx_index, * l_path_file_input, flag_ramtx_sorted_by_id_feature = True, flag_delete_input_file_upon_completion = False, dtype_mtx = np.float64, dtype_mtx_index = np.float64, int_size_buffer_for_mtx_index = 1000 ) :
    """ # 2022-07-02 11:37:05 
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
    ''' # 2022-07-08 01:56:32 
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
        root.attrs[ 'dict_ramtx_metadata' ] = { 
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
    """ # 2022-07-08 02:00:14 
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
    root.attrs[ 'dict_ramdata_metadata' ] = { 
        'path_folder_mtx_10x_input' : path_folder_mtx_10x_input,
        'str_completed_time' : TIME_GET_timestamp( True ),
        'int_num_features' : int_num_features,
        'int_num_barcodes' : int_num_barcodes,
        'int_num_of_records_in_a_chunk_zarr_matrix' : int_num_of_records_in_a_chunk_zarr_matrix,
        'int_num_of_entries_in_a_chunk_zarr_matrix_index' : int_num_of_entries_in_a_chunk_zarr_matrix_index,
        'layers' : [ name_layer ],
        'version' : _version_,
    }
    
''' a class for accessing Zarr-backed count matrix data (RAMtx, Random-access matrix) '''
class RAMtx( ) :
    """ # 2022-07-21 00:03:40 
    This class represent a random-access mtx format for memory-efficient exploration of extremely large single-cell transcriptomics/genomics data.
    This class use a count matrix data stored in a random read-access compatible format, called RAMtx, enabling exploration of a count matrix with hundreds of millions cells with hundreds of millions of features.
    Also, the RAMtx format is supports multi-processing, and provide convenient interface for parallel processing of single-cell data
    Therefore, for exploration of count matrix produced from 'scarab count', which produces dozens of millions of features extracted from both coding and non coding regions, this class provides fast front-end application for exploration of exhaustive data generated from 'scarab count'
    
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
    
    """
    def __init__( self, path_folder_ramtx, ramdata = None, dtype_of_feature_and_barcode_indices = np.int32, dtype_of_values = np.float64, int_num_cpus = 1, verbose = False, flag_debugging = False, mode = 'a', flag_is_read_only = False, path_folder_ramtx_mask = None ) :
        """ # 2022-07-21 00:03:35 
        """
        # read metadata
        self._root = zarr.open( path_folder_ramtx, 'a' )
        dict_ramtx_metadata = self._root.attrs[ 'dict_ramtx_metadata' ]
        self._dict_ramtx_metadata = dict_ramtx_metadata # set the metadata of the sort, index and export settings
        
        # parse the metadata of the RAMtx object
        self._int_num_features, self._int_num_barcodes, self._int_num_records = self._dict_ramtx_metadata[ 'int_num_features' ], self._dict_ramtx_metadata[ 'int_num_barcodes' ], self._dict_ramtx_metadata[ 'int_num_records' ]
        
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
        
        # set filters using RamData
        self.ba_filter_features = ramdata.ft.filter if ramdata is not None else None
        self.ba_filter_barcodes = ramdata.bc.filter if ramdata is not None else None
        
        # open Zarr object containing matrix and matrix indices
        self._za_mtx_index = zarr.open( f'{self._path_folder_ramtx}matrix.index.zarr', 'r' )
        self._za_mtx = zarr.open( f'{self._path_folder_ramtx}matrix.zarr', 'r' )
    @property
    def _path_folder_ramtx_modifiable( self ) :
        """ # 2022-07-21 09:04:28 
        return the path to the modifiable RAMtx object
        """
        return ( None if self._flag_is_read_only else self._path_folder_ramtx ) if self._path_folder_ramtx_mask is None else self._path_folder_ramtx_mask
    @property
    def ba_active_entries( self ) :
        """ # 2022-07-21 09:17:14 
        return a bitarray filter of the indexed axis where all the entries with valid count data is marked '1'
        """
        # internal settings
        int_num_chunks_in_a_batch = 100 # 'int_num_chunks_in_a_batch' : the number of chunks in a batch. increasing this number will increase the memory consumption
        
        try :
            za = zarr.open( f'{self._path_folder_ramtx}matrix.index.active_entries.zarr/', mode = 'r', synchronizer = zarr.ThreadSynchronizer( ) ) # open zarr object of the current RAMtx object
        except : # if the zarr object (cache) is not available
            # if the boolean array of the active entries is not available
            if self._path_folder_ramtx_modifiable is None : # if modifiable RAMtx object does not exist
                za = ( self._za_mtx_index[ :, 1 ] - self._za_mtx_index[ :, 0 ] ) > 0 # retrieve mask without chunking
            else :
                # if modifiable RAMtx object is available, using zarr object, retrieve mask chunk by chunk
                int_size_chunk = self._za_mtx_index.chunks[ 0 ] # retrieve the size of the chunk
                za = zarr.open( f"{self._path_folder_ramtx_modifiable}matrix.index.active_entries.zarr/", mode = 'w', shape = ( self.len_indexed_axis, ), chunks = ( int_size_chunk * int_num_chunks_in_a_batch, ), dtype = bool, synchronizer = zarr.ThreadSynchronizer( ) ) # the size of the chunk will be 100 times of the chunk used for matrix index, since the dtype is boolean
                len_indexed_axis = self.len_indexed_axis
                int_pos_start = 0
                int_num_entries_to_retrieve = int( int_size_chunk * int_num_chunks_in_a_batch )
                while int_pos_start < len_indexed_axis :
                    sl = slice( int_pos_start, int_pos_start + int_num_entries_to_retrieve )
                    za[ sl ] = ( self._za_mtx_index[ sl ][ :, 1 ] - self._za_mtx_index[ sl ][ :, 0 ] ) > 0 # active entry is defined by finding entries with at least one count record
                    int_pos_start += int_num_entries_to_retrieve # update the position
            
        ba = bk.BA.to_bitarray( za[ : ] ) # return the boolean array of active entries as a bitarray object
        self._n_active_entries = ba.count( ) # calculate the number of active entries
        return ba
    @property
    def n_active_entries( self ) :
        ''' # 2022-07-02 15:04:39 
        calculate the number of active entries
        '''
        # calculate the number of active entries if it has not been calculated.
        if not hasattr( self, '_n_active_entries' ) :
            self._n_active_entries = self.ba_active_entries.count( )
        return self._n_active_entries
    def __repr__( self ) :
        return f"<RAMtx object containing {self._int_num_records} records of {self._int_num_features} features X {self._int_num_barcodes} barcodes\n\tRAMtx path: {self._path_folder_ramtx}>"
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
    def flag_ramtx_sorted_by_id_feature( self ) :
        ''' # 2022-06-23 09:06:51 
        retrieve 'flag_ramtx_sorted_by_id_feature' setting from the RAMtx metadata
        '''
        return self._dict_ramtx_metadata[ 'flag_ramtx_sorted_by_id_feature' ]
    @property
    def len_indexed_axis( self ) :
        ''' # 2022-06-23 09:08:44 
        retrieve number of elements of the indexed axis
        '''
        return self._int_num_features if self.flag_ramtx_sorted_by_id_feature else self._int_num_barcodes
    @property
    def indexed_axis( self ) :
        """
        # 2022-06-30 21:45:48 
        return 'Axis' object of the indexed axis
        """
        return None if self._ramdata is None else ( self._ramdata.ft if self.flag_ramtx_sorted_by_id_feature else self._ramdata.bc )
    @property
    def ba_filter_indexed_axis( self ) :
        ''' # 2022-06-23 09:08:44 
        retrieve filter of the indexed axis
        '''
        return self.ba_filter_features if self.flag_ramtx_sorted_by_id_feature else self.ba_filter_barcodes
    def __contains__( self, x ) -> bool :
        ''' # 2022-06-23 09:13:31 
        check whether an integer representation of indexed entry is available in the index. if filter is active, also check whether the entry is active '''
        return ( 0 <= x < self.len_indexed_axis ) and ( self.ba_filter_indexed_axis is None or self.ba_filter_indexed_axis[ x ] ) # x should be in valid range, and if it is, check whether x is an active element in the filter (if filter has been set)
    def __iter__( self ) :
        ''' # 2022-06-23 09:13:46 
        yield each entry in the index upon iteration. if filter is active, ignore inactive elements '''
        if self.ba_filter_indexed_axis is None : # if filter is not set, iterate over all elements
            return iter( range( self.len_indexed_axis ) )
        else : # if filter is active, yield indices of active elements only
            return iter( bk.BA.to_integer_indices( self.ba_filter_indexed_axis ) )
    def __getitem__( self, l_int_entry ) : 
        """ # 2022-07-05 23:13:11 
        Retrieve data from RAMtx as lists of values and arrays, each value and array contains data of a single 'int_entry' of the indexed axis
        '__getitem__' can be used to retrieve minimal data required to build a sparse matrix
        
        Returns:
        l_int_entry_indexed_valid, l_arr_int_entry_not_indexed, l_arr_value 
        """
        # initialize the output data structures
        l_int_entry_indexed_valid, l_arr_int_entry_not_indexed, l_arr_value = [ ], [ ], [ ]
        
        # wrap in a list if a single entry was queried
        if isinstance( l_int_entry, ( int, np.int64, np.int32, np.int16, np.int8 ) ) : # check whether the given entry is an integer
            l_int_entry = [ l_int_entry ]
        ''' retrieve filters '''
        flag_ramtx_sorted_by_id_feature = self.flag_ramtx_sorted_by_id_feature
        ba_filter_indexed_axis, ba_filter_not_indexed_axis = ( self.ba_filter_features, self.ba_filter_barcodes ) if flag_ramtx_sorted_by_id_feature else ( self.ba_filter_barcodes, self.ba_filter_features )
            
        ''' filter 'int_entry', if a filter has been set '''
        if ba_filter_indexed_axis is not None :
            l_int_entry = bk.BA.to_integer_indices( ba_filter_indexed_axis ) if len( l_int_entry ) == 0 else list( int_entry for int_entry in l_int_entry if ba_filter_indexed_axis[ int_entry ] ) # filter 'l_int_entry' or use the entries in the given filter (if no int_entry was given, use all active entries)
                
        # if no valid entries are available, return an empty result
        if len( l_int_entry ) == 0 :
            return l_int_entry_indexed_valid, l_arr_int_entry_not_indexed, l_arr_value
            
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
        flag_change_dtype_mtx_index = self._za_mtx_index.dtype != np.int64
        flag_change_dtype_of_feature_and_barcode_indices = self._za_mtx.dtype != self._dtype_of_feature_and_barcode_indices
        flag_change_dtype_of_values = self._za_mtx.dtype != self._dtype_of_values
        
        # retrieve dictionaries for changing coordinates
        dict_change_int_entry_indexed, dict_change_int_entry_not_indexed = None, None # initialize the dictionaries
        if self._ramdata is not None : # if RAMtx has been attached to RamData, retrieve dictionaries that can be used to change coordinate
            if self.flag_ramtx_sorted_by_id_feature :
                dict_change_int_entry_indexed = self._ramdata.ft.dict_change
                dict_change_int_entry_not_indexed = self._ramdata.bc.dict_change
            else :
                dict_change_int_entry_indexed = self._ramdata.bc.dict_change
                dict_change_int_entry_not_indexed = self._ramdata.ft.dict_change
        # compose a vectorized function for the conversion of int_entries of the non-indexed axis.
        def f( i ) :
            return dict_change_int_entry_not_indexed[ i ]
        vchange_int_entry_not_indexed = np.vectorize( f ) if dict_change_int_entry_not_indexed is not None else None
        
        def __retrieve_data_from_ramtx_as_a_worker__( pipe_from_main_thread = None, pipe_to_main_thread = None, flag_as_a_worker = True ) :
            """ # 2022-07-02 12:35:44 
            retrieve data as a worker in a worker process or in the main processs (in single-process mode)
            """
            # handle inputs
            l_int_entry = pipe_from_main_thread.recv( ) if flag_as_a_worker else pipe_from_main_thread  # receive work if 'flag_as_a_worker' is True or use 'pipe_from_main_thread' as a list of works
            # for each int_entry, retrieve data and collect records
            l_int_entry_indexed_valid, l_arr_int_entry_not_indexed, l_arr_value = [ ], [ ], [ ]
            
            # retrieve mtx_index data and remove invalid entries
            arr_index = self._za_mtx_index.get_orthogonal_selection( l_int_entry ) # retrieve mtx_index data 
            if flag_change_dtype_mtx_index : # convert dtype of retrieved mtx_index data
                arr_index = arr_index.astype( np.int64 )
            # iterate through each 'int_entry'
            for int_entry, index in zip( l_int_entry, arr_index ) : # iterate through each entry
                st, en = index
                if st == en : # if there is no count data for the 'int_entry', continue on to the next 'int_entry' # drop 'int_entry' lacking count data (when start and end index is the same, the 'int_entry' does not contain any data)
                    continue
                arr_int_entry_not_indexed, arr_value = self._za_mtx[ st : en ].T # retrieve count data from the Zarr object
                
                ''' if a filter for not-indexed axis has been set, apply the filter to the retrieved records '''
                if ba_filter_not_indexed_axis is not None :
                    arr_int_entry_not_indexed = arr_int_entry_not_indexed.astype( np.int64 ) # convert to integer type
                    arr_mask = np.zeros( len( arr_int_entry_not_indexed ), dtype = bool ) # initialize the mask for filtering records
                    for i, int_entry_not_indexed in enumerate( arr_int_entry_not_indexed ) : # iterate through each record
                        if ba_filter_not_indexed_axis[ int_entry_not_indexed ] : # check whether the current int_entry is included in the filter
                            arr_mask[ i ] = True # include the record
                    # if no valid data exists (all data were filtered out), continue to the next 'int_entry'
                    if arr_mask.sum( ) == 0 :
                        continue
                        
                    # filter records using the mask
                    arr_int_entry_not_indexed = arr_int_entry_not_indexed[ arr_mask ]
                    arr_value = arr_value[ arr_mask ]
                    
                    # convert int_entry for the non-indexed axis if a mapping has been given
                    if vchange_int_entry_not_indexed is not None :
                        arr_int_entry_not_indexed = vchange_int_entry_not_indexed( arr_int_entry_not_indexed )
                    
                ''' convert dtypes of retrieved data '''
                if flag_change_dtype_of_feature_and_barcode_indices :
                    arr_int_entry_not_indexed = arr_int_entry_not_indexed.astype( self._dtype_of_feature_and_barcode_indices )
                if flag_change_dtype_of_values :
                    arr_value = arr_value.astype( self._dtype_of_values )
                
                ''' append the retrieved data to the output results '''
                l_int_entry_indexed_valid.append( int_entry if dict_change_int_entry_indexed is None else dict_change_int_entry_indexed[ int_entry ] ) # convert int_entry for the indexed axis if a mapping has been given 
                l_arr_int_entry_not_indexed.append( arr_int_entry_not_indexed )
                l_arr_value.append( arr_value )
            
            ''' return the retrieved data '''
            # compose a output value
            output = ( l_int_entry_indexed_valid, l_arr_int_entry_not_indexed, l_arr_value )
            # if 'flag_as_a_worker' is True, send the result or return the result
            if flag_as_a_worker :
                pipe_to_main_thread.send( output ) # send unzipped result back
            else :
                return output
        
        # load data using multiprocessing
        if self.int_num_cpus > 1 and int_num_entries > 1 : # enter multi-processing mode only more than one entry should be retrieved
            # initialize workers
            int_n_workers = min( self.int_num_cpus, int_num_entries ) # one thread for distributing records. Minimum numbers of workers for sorting is 1 # the number of workers should not be larger than the number of entries to retrieve.
            l_pipes_from_main_process_to_worker = list( mp.Pipe( ) for _ in range( self.int_num_cpus ) ) # create pipes for sending records to workers # add process for receivers
            l_pipes_from_worker_to_main_process = list( mp.Pipe( ) for _ in range( self.int_num_cpus ) ) # create pipes for collecting results from workers
            l_processes = list( mp.Process( target = __retrieve_data_from_ramtx_as_a_worker__, args = ( l_pipes_from_main_process_to_worker[ index_worker ][ 1 ], l_pipes_from_worker_to_main_process[ index_worker ][ 0 ] ) ) for index_worker in range( int_n_workers ) ) # add a process for distributing fastq records
            for p in l_processes :
                p.start( )
            # distribute works
            for index_worker, l_int_entry_for_each_worker in enumerate( LIST_Split( l_int_entry, int_n_workers ) ) : # continuous or distributed ? what would be more efficient?
                l_pipes_from_main_process_to_worker[ index_worker ][ 0 ].send( l_int_entry_for_each_worker )
            # wait until all works are completed
            int_num_workers_completed = 0
            while int_num_workers_completed < int_n_workers : # until all works are completed
                for _, pipe in l_pipes_from_worker_to_main_process :
                    if pipe.poll( ) :
                        otuput = pipe.recv( )
                        l_int_entry_indexed_valid.extend( otuput[ 0 ] )
                        l_arr_int_entry_not_indexed.extend( otuput[ 1 ] )
                        l_arr_value.extend( otuput[ 2 ] )
                        del otuput
                        int_num_workers_completed += 1
                time.sleep( 0.1 )
            # dismiss workers once all works are completed
            for p in l_processes :
                p.join( )
        else : # single thread mode
            l_int_entry_indexed_valid, l_arr_int_entry_not_indexed, l_arr_value = __retrieve_data_from_ramtx_as_a_worker__( l_int_entry, flag_as_a_worker = False )
        
        return l_int_entry_indexed_valid, l_arr_int_entry_not_indexed, l_arr_value
    def get_sparse_matrix( self, l_int_entry ) :
        """ # 2022-07-13 09:25:59 
        
        get sparse matrix for the given list of integer representations of the entries.
        """
        l_int_entry_indexed_valid, l_arr_int_entry_not_indexed, l_arr_value = self[ l_int_entry ] # parse retrieved result
        
        # combine the arrays
        arr_int_entry_not_indexed = np.concatenate( l_arr_int_entry_not_indexed )
        arr_value = np.concatenate( l_arr_value )
        del l_arr_value # delete intermediate objects
        
        # compose 'arr_int_entry_indexed'
        arr_int_entry_indexed = np.zeros( len( arr_int_entry_not_indexed ), dtype = self._dtype_of_feature_and_barcode_indices ) # create an empty array
        int_pos = 0
        for int_entry_indexed, a in zip( l_int_entry_indexed_valid, l_arr_int_entry_not_indexed ) :
            n = len( a )
            arr_int_entry_indexed[ int_pos : int_pos + n ] = int_entry_indexed # compose 'arr_int_entry_indexed'
            int_pos += n # update the current position
        del l_int_entry_indexed_valid, l_arr_int_entry_not_indexed # delete intermediate objects
        
        # get 'arr_int_barcode' and 'arr_int_feature' based on 'self.flag_ramtx_sorted_by_id_feature'
        if self.flag_ramtx_sorted_by_id_feature :
            arr_int_barcode = arr_int_entry_not_indexed
            arr_int_feature = arr_int_entry_indexed
        else :
            arr_int_barcode = arr_int_entry_indexed
            arr_int_feature = arr_int_entry_not_indexed
        del arr_int_entry_indexed, arr_int_entry_not_indexed # delete intermediate objects
        
        n_bc, n_ft = ( self._int_num_barcodes, self._int_num_features ) if self._ramdata is None else ( self._ramdata.bc.meta.n_rows, self._ramdata.ft.meta.n_rows ) # detect whether the current RAMtx has been attached to a RamData and retrieve the number of barcodes and features accordingly
        X = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( ( arr_value, ( arr_int_barcode, arr_int_feature ) ), shape = ( n_bc, n_ft ) ) ) # convert count data to a sparse matrix
        return X # return the composed sparse matrix 
    def batch_generator( self, ba = None, int_num_entries_for_each_weight_calculation_batch = 1000, int_total_weight_for_each_batch = 1000000 ) :
        ''' # 2022-07-12 22:05:27 
        generate batches of list of integer indices of the active entries in the given bitarray 'ba'. 
        Each bach has the following characteristics:
            monotonous: active entries in a batch are in an increasing order
            balanced: the total weight of a batch is around (but not exactly) 'int_total_weight_for_each_batch'
        
        'ba' : (default None) if None is given, self.ba_active_entries bitarray will be used.
        '''
        # set defaule arguments
        if ba is None :
            ba = self.ba_filter_indexed_axis # if None is given, ba_filter of the currently indexed axis will be used.
            if ba is None : # if filter is not set or the current RAMtx has not been attached to a RamData object, use the active entries
                ba = self.ba_active_entries # if None is given, self.ba_active_entries bitarray will be used.
        # initialize
        # a namespace that can safely shared between functions
        ns = { 'int_accumulated_weight_current_batch' : 0, 'l_int_entry_current_batch' : [ ], 'l_int_entry_for_weight_calculation_batch' : [ ] }
        
        def find_batch( ) :
            """ # 2022-07-03 22:11:06 
            retrieve indices of the current 'weight_current_batch', calculate weights, and yield a batch
            """
            st, en = self._za_mtx_index.get_orthogonal_selection( ns[ 'l_int_entry_for_weight_calculation_batch' ] ).T # retrieve start and end coordinates of the entries
            arr_weight = en - st # calculate weight for each entry
            del st, en
            for int_entry, weight in zip( ns[ 'l_int_entry_for_weight_calculation_batch' ], arr_weight ) :
                # update the current batch
                ns[ 'l_int_entry_current_batch' ].append( int_entry )
                ns[ 'int_accumulated_weight_current_batch' ] += weight

                # check whether the current batch is full
                if ns[ 'int_accumulated_weight_current_batch' ] >= int_total_weight_for_each_batch : # a current batch is full, yield the batch
                    yield ns[ 'l_int_entry_current_batch' ] # return a batch
                    # initialize the next batch
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
            yield ns[ 'l_int_entry_current_batch' ]
''' a class for representing axis of RamData (barcodes/features) '''
class Axis( ) :
    """ # 2022-07-16 23:56:47 
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
    def _convert_to_bitarray( self, ba_filter ) :
        ''' # 2022-06-28 20:01:25 
        handle non-None filter objects and convert these formats to the bitarray filter object
        '''
        assert self.int_num_entries == len( ba_filter )

        ''' handle non-bitarray input types '''
        # handle when a list type has been given (convert it to np.ndarray)
        if isinstance( ba_filter, list ) :
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
        ''' # 2022-07-02 09:45:27 
        returns the number of entries in the Axis
        '''
        return self.int_num_entries
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
    def load_str( self, int_index_col = None ) : 
        ''' # 2022-06-24 22:38:18 
        load string representation of the entries of the current axis, and retrieve a mapping from string representation to integer representation
        
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
        return f"<Axis'{self._name_axis}' available at {self._path_folder}\n\tavailable metadata columns are {sorted( self.meta.columns )}>"
    def all( self, flag_return_valid_entries_in_the_currently_active_layer = True ) :
        """ # 2022-06-27 21:41:38  
        return bitarray filter with all entries marked 'active'
        
        'flag_return_valid_entries_in_the_currently_active_layer' : return bitarray filter containing only the active entries in the current layer 
        """
        if flag_return_valid_entries_in_the_currently_active_layer and self._ramdata.layer is not None : # if RamData has an active layer and 'flag_return_valid_entries_in_the_currently_active_layer' setting is True, return bitarray where entries with valid count data is marked as '1'
            ba = getattr( self._ramdata._layer, f'_ramtx_{self._name_axis}' ).ba_active_entries
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
    def batch_generator( self, ba = None, int_num_entries_for_batch = 1000, flag_return_the_number_of_previously_returned_entries = False, flag_mix_randomly = False ) :
        ''' # 2022-07-16 22:57:23 
        generate batches of list of integer indices of the active entries in the given bitarray 'ba'. 
        Each bach has the following characteristics:
            monotonous: active entries in a batch are in an increasing order
            same size: except for the last batch, each batch has the same number of active entries 'int_num_entries_for_batch'.
        
        'ba' : (default None) if None is given, self.filter bitarray will be used.
        'flag_return_the_number_of_previously_returned_entries' : yield the number of previously returned entries along side with the list of entries of the current batch.
        'flag_mix_randomly' : generate batches of entries after mixing randomly 
        '''
        # set defaule arguments
        # set default filter
        if ba is None :
            ba = self.ba_active_entries # iterate through an active entries
            
        # initialize
        # a namespace that can safely shared between functions
        ns = { 'l_int_entry_current_batch' : [ ] }
        
        if flag_mix_randomly : # randomly select barcodes across the 
            int_num_active_entries = ba.count( ) # retrieve the total number of active entries 
            float_ratio_batch_size_to_total_size = int_num_entries_for_batch / int_num_active_entries # retrieve approximate number of batches to generate
            # initialize
            int_num_of_previously_returned_entries = 0
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
                        yield ( int_num_of_previously_returned_entries, ns[ 'l_int_entry_current_batch' ] ) if flag_return_the_number_of_previously_returned_entries else ns[ 'l_int_entry_current_batch' ] # return 'int_num_of_previously_returned_entries' if 'flag_return_the_number_of_previously_returned_entries' is True
                        int_num_of_previously_returned_entries += len( ns[ 'l_int_entry_current_batch' ] ) # update the number of returned entries
                        float_prob_selection = int_num_entries_for_batch / max( 1, int_num_active_entries - int_num_of_previously_returned_entries ) # update the probability for selection of an entry
                        ns[ 'l_int_entry_current_batch' ] = [ ] # initialize the next batch
            # return the remaining int_entries as the last batch (if available)
            if len( ns[ 'l_int_entry_current_batch' ] ) > 0 :
                ns[ 'l_int_entry_current_batch' ] = np.sort( ns[ 'l_int_entry_current_batch' ] ) # sort the list of int_entries
                yield ( int_num_of_previously_returned_entries, ns[ 'l_int_entry_current_batch' ] ) if flag_return_the_number_of_previously_returned_entries else ns[ 'l_int_entry_current_batch' ] # return 'int_num_of_previously_returned_entries' if 'flag_return_the_number_of_previously_returned_entries' is True
        else : # return barcodes in a batch sequentially
            int_num_of_previously_returned_entries = 0
            for int_entry in bk.BA.find( ba ) : # iterate through active entries of the given bitarray
                ns[ 'l_int_entry_current_batch' ].append( int_entry ) # collect int_entry for the current batch
                # once the batch is full, yield the batch
                if len( ns[ 'l_int_entry_current_batch' ] ) >= int_num_entries_for_batch :
                    yield ( int_num_of_previously_returned_entries, ns[ 'l_int_entry_current_batch' ] ) if flag_return_the_number_of_previously_returned_entries else ns[ 'l_int_entry_current_batch' ] # return 'int_num_of_previously_returned_entries' if 'flag_return_the_number_of_previously_returned_entries' is True
                    int_num_of_previously_returned_entries += len( ns[ 'l_int_entry_current_batch' ] ) # update the number of returned entries
                    ns[ 'l_int_entry_current_batch' ] = [ ] # initialize the next batch
            # return the remaining int_entries as the last batch (if available)
            if len( ns[ 'l_int_entry_current_batch' ] ) > 0 :
                yield ( int_num_of_previously_returned_entries, ns[ 'l_int_entry_current_batch' ] ) if flag_return_the_number_of_previously_returned_entries else ns[ 'l_int_entry_current_batch' ] # return 'int_num_of_previously_returned_entries' if 'flag_return_the_number_of_previously_returned_entries' is True
    def change_filter( self, name_col_filter ) :
        """ # 2022-07-16 17:17:29 
        change filter using the filter saved in the metadata with 'name_col_filter' column name. if 'name_col_filter' is not available, current filter setting will not be changed.
        
        'name_col_filter' : name of the column of the metadata ZarrDataFrame containing the filter
        """
        if name_col_filter in self.meta : # if a given column name exists in the current metadata ZarrDataFrame
            self.filter = self.meta[ name_col_filter, : ] # retrieve filter from the storage and apply the filter to the axis
    def save_filter( self, name_col_filter ) :
        """ # 2022-07-16 17:17:29 
        save current filter using the filter to the metadata with 'name_col_filter' column name. if a filter is not active, the metadata will not be updated.
        
        'name_col_filter' : name of the column of the metadata ZarrDataFrame that will contain the filter
        """
        if name_col_filter is not None : # if a given filter name is valid
            if self.filter is not None : # if a filter is active in the current 'Axis'
                self.meta[ name_col_filter, : ] = self.filter # save filter to the storage
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
class Layer( ) :
    """ # 2022-06-25 16:32:19 
    A class for interactions with a pair of RAMtx objects of a count matrix. 
    
    'path_folder_ramdata' : location of RamData
    'int_num_cpus' : number of CPUs for RAMtx objects
    'mode' : file mode. 'r' for read-only mode and 'a' for mode allowing modifications
    'flag_is_read_only' : read-only status of RamData
    'path_folder_ramdata_mask' : a local (local file system) path to the mask of the RamData object that allows modifications to be written without modifying the source. if a valid local path to a mask is given, all modifications will be written to the mask
    """
    def __init__( self, path_folder_ramdata, name_layer, ramdata = None, dtype_of_feature_and_barcode_indices = np.int32, dtype_of_values = np.float64, int_num_cpus = 1, verbose = False, mode = 'a', path_folder_ramdata_mask = None, flag_is_read_only = False ) :
        """ # 2022-06-25 16:40:44 
        """
        # set attributes
        self._path_folder_ramdata = path_folder_ramdata
        self._name_layer = name_layer
        self._ramdata = ramdata
        self._mode = mode
        self._path_folder_ramdata_mask = path_folder_ramdata_mask
        self._flag_is_read_only = flag_is_read_only
        
        # retrieve filters from the axes
        ba_filter_features = ramdata.ft.filter if ramdata is not None else None
        ba_filter_barcodes = ramdata.bc.filter if ramdata is not None else None
        
        # open RAMtx objects without filters
        self._ramtx_features = RAMtx( f'{self._path_folder_ramdata}{name_layer}/sorted_by_feature/', ramdata = ramdata, dtype_of_feature_and_barcode_indices = dtype_of_feature_and_barcode_indices, dtype_of_values = dtype_of_values, int_num_cpus = int_num_cpus, verbose = verbose, flag_debugging = False, mode = self._mode, path_folder_ramtx_mask = None if self._path_folder_ramdata_mask is None else f'{self._path_folder_ramdata_mask}{name_layer}/sorted_by_feature/', flag_is_read_only = self._flag_is_read_only )
        self._ramtx_barcodes = RAMtx( f'{self._path_folder_ramdata}{name_layer}/sorted_by_barcode/', ramdata = ramdata, dtype_of_feature_and_barcode_indices = dtype_of_feature_and_barcode_indices, dtype_of_values = dtype_of_values, int_num_cpus = int_num_cpus, verbose = verbose, flag_debugging = False, mode = self._mode, path_folder_ramtx_mask = None if self._path_folder_ramdata_mask is None else f'{self._path_folder_ramdata_mask}{name_layer}/sorted_by_barcode/', flag_is_read_only = self._flag_is_read_only )
        
        # set filters of the current layer
        self.ba_filter_features = ba_filter_features
        self.ba_filter_barcodes = ba_filter_barcodes
    @property
    def int_num_features( self ) :
        """ # 2022-06-28 21:39:20 
        return the number of features
        """
        return self._ramtx_features.len_indexed_axis
    @property
    def int_num_barcodes( self ) :
        """ # 2022-06-28 21:39:20 
        return the number of features
        """
        return self._ramtx_barcodes.len_indexed_axis
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
        self._ramtx_features.ba_filter_features = ba_filter
        self._ramtx_barcodes.ba_filter_features = ba_filter
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
        self._ramtx_features.ba_filter_barcodes = ba_filter
        self._ramtx_barcodes.ba_filter_barcodes = ba_filter
    def __getitem__( self, l_int_bc = slice( None, None, None ), l_int_ft = slice( None, None, None ) ) :
        """ # 2022-06-28 21:22:38 
        return a count data of a given list of integer representation of barcodes and features. slice can also be used.
        """
        sl_all = slice( None, None, None ) # slice selecting all entries
        def get_l_int( l_int, length ) :
            ''' # 2022-06-28 22:29:08 
            get list of integer indices for the iterable with a given length ('length')
            '''
            # when a slice is given, convert slice to the list of integer indices
            if isinstance( l_int, slice ) :
                l_int = list( bk.Slice_to_Range( l_int, length ) )
            return l_int # returns the list of integer indices
        # 
        if l_int_bc == sl_all and l_int_ft == sl_all : # if all barcodes and all features are selected
            if self.int_num_features > self.int_num_barcodes : # if the number of features are greter than the number of barcodes, retrieve data for each barcode (less number of RAMtx accesses)
                return self._ramtx_barcodes[ np.arange( self.int_num_barcodes ) ]
            else :
                return self._ramtx_features[ np.arange( self.int_num_features ) ]
        elif l_int_bc == sl_all : # if all barcodes will be used, use ramtx_features to retrieve count data
            return self._ramtx_features[ get_l_int( l_int_ft, self.int_num_features ) ]
        elif l_int_ft == sl_all : # if all features will be used, use ramtx_barcodes to retrieve count data
            return self._ramtx_barcodes[ get_l_int( l_int_bc, self.int_num_barcodes ) ]
        else : # if count data of a subset of barcodes and a subset of features will be retrieved
            l_int_ft = get_l_int( l_int_ft, self.int_num_features )
            l_int_bc = get_l_int( l_int_bc, self.int_num_barcodes )
            if len( l_int_ft ) > len( l_int_bc ) : # if the number of features are larger than the number of barcodes, use 'ramtx_barcodes'
                return self._ramtx_barcodes[ l_int_bc ]
            else :
                return self._ramtx_features[ l_int_ft ]
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

class RamData( ) :
    """ # 2022-07-21 01:33:35 
    This class provides frameworks for single-cell transcriptomic/genomic data analysis, utilizing RAMtx data structures, which is backed by Zarr persistant arrays.
    Extreme lazy loading strategies used by this class allows efficient parallelization of analysis of single cell data with minimal memory footprint, loading only essential data required for analysis. 
    
    'path_folder_ramdata' : a local folder directory or a remote location (https://, s3://, etc.) containing RamData object
    'int_num_cpus' : number of CPUs (processes) to use to distribute works.
    'int_index_str_rep_for_barcodes', 'int_index_str_rep_for_features' : a integer index for the column for the string representation of 'barcodes'/'features' in the string Zarr object (the object storing strings) of 'barcodes'/'features'
    'dict_kw_zdf' : settings for 'Axis' metadata ZarrDataFrame
    'dict_kw_view' : settings for 'Axis' object for creating a view based on the active filter.
    'mode' : file mode. 'r' for read-only mode and 'a' for mode allowing modifications
    'path_folder_ramdata_mask' : the LOCAL file system path where the modifications of the RamData ('MASK') object will be saved and retrieved. If this attribute has been set, the given RamData in the the given 'path_folder_ramdata' will be used as READ-ONLY. For example, when RamData resides in the HTTP server, data is often read-only (data can be only fetched from the server, and not the other way around). However, by giving a local path through this argument, the read-only RamData object can be analyzed as if the RamData object can be modified. This is possible since all the modifications made on the input RamData will be instead written to the local RamData object 'mask' and data will be fetced from the local copy before checking the availability in the remote RamData object.
    
    ==== AnnDataContainer ====
    'flag_enforce_name_adata_with_only_valid_characters' : enforce valid characters in the name of AnnData
    """
    def __init__( self, path_folder_ramdata, name_layer = 'raw', int_num_cpus = 64, dtype_of_feature_and_barcode_indices = np.int32, dtype_of_values = np.float64, int_index_str_rep_for_barcodes = 0, int_index_str_rep_for_features = 1, mode = 'a', path_folder_ramdata_mask = None, dict_kw_zdf = { 'flag_retrieve_categorical_data_as_integers' : False, 'flag_load_data_after_adding_new_column' : True, 'flag_enforce_name_col_with_only_valid_characters' : True }, dict_kw_view = { 'float_min_proportion_of_active_entries_in_an_axis_for_using_array' : 0.1, 'dtype' : np.int32 }, flag_enforce_name_adata_with_only_valid_characters = True, verbose = True, flag_debugging = False ) :
        """ # 2022-07-21 02:12:46 
        """
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
        self._dtype_of_feature_and_barcode_indices = dtype_of_feature_and_barcode_indices
        self._dtype_of_values = dtype_of_values
        
        ''' check read-only status of the given RamData '''
        try :
            zarr.open( f'{self._path_folder_ramdata}modification.test.zarr/', 'w' )
            self._flag_is_read_only = False
        except :
            # if test zarr data cannot be written to the source, consider the given RamData object as read-only
            self._flag_is_read_only = True
            if self._path_folder_ramdata_mask is None : # if mask is not given, automatically change the mode to 'r'
                self._mode = 'r'
                if self.verbose :
                    print( 'The current RamData object cannot be modified yet no mask location is given. Therefore, the current RamData object will be "read-only"' )
        
        # initialize axis objects
        self.bc = Axis( path_folder_ramdata, 'barcodes', ba_filter = None, ramdata = self, dict_kw_zdf = dict_kw_zdf, dict_kw_view = dict_kw_view, int_index_str_rep = int_index_str_rep_for_barcodes, verbose = verbose, mode = self._mode, path_folder_mask = self._path_folder_ramdata_mask, flag_is_read_only = self._flag_is_read_only )
        self.ft = Axis( path_folder_ramdata, 'features', ba_filter = None, ramdata = self, dict_kw_zdf = dict_kw_zdf, dict_kw_view = dict_kw_view, int_index_str_rep = int_index_str_rep_for_features, verbose = verbose, mode = self._mode, path_folder_mask = self._path_folder_ramdata_mask, flag_is_read_only = self._flag_is_read_only )
        
        # initialize layor object
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
                    root_mask.attrs[ 'dict_ramdata_metadata' ] = self._root.attrs[ 'dict_ramdata_metadata' ] # copy the ramdata attribute of the current RamData to the mask object
                self._root = root_mask # use the mask object zarr group to save/retrieve ramdata metadata
                
            # retrieve metadata 
            self._dict_ramdata_metadata = self._root.attrs[ 'dict_ramdata_metadata' ]
            self._dict_ramdata_metadata[ 'layers' ] = set( self._dict_ramdata_metadata[ 'layers' ] )
        # return metadata
        return self._dict_ramdata_metadata
    def _save_metadata_( self ) :
        ''' # 2022-07-21 00:45:03 
        a semi-private method for saving metadata to the disk 
        '''
        if not self._flag_is_read_only : # update metadata only when the current RamData object is not read-only
            if hasattr( self, '_dict_ramdata_metadata' ) : # if metadata has been loaded
                # convert 'columns' to list before saving attributes
                temp = self._dict_ramdata_metadata[ 'layers' ] # save the set as a temporary variable 
                self._dict_ramdata_metadata[ 'layers' ] = list( temp ) # convert to list
                self._root.attrs[ 'dict_ramdata_metadata' ] = self._dict_ramdata_metadata # update metadata
                self._dict_ramdata_metadata[ 'layers' ] = temp # revert to set
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
    def layer( self ) :
        """ # 2022-06-24 00:16:56 
        retrieve the name of layer from the layer object if it has been loaded.
        """
        return self._layer._name_layer if hasattr( self, '_layer' ) else None # if no layer is set, return None
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

            if name_layer != self.layer : # load new layer
                self._layer = Layer( self._path_folder_ramdata, name_layer, ramdata = self, dtype_of_feature_and_barcode_indices = self._dtype_of_feature_and_barcode_indices, dtype_of_values = self._dtype_of_values, int_num_cpus = 1, verbose = self.verbose, mode = self._mode, path_folder_ramdata_mask = self._path_folder_ramdata_mask, flag_is_read_only = self._flag_is_read_only )

                if self.verbose :
                    print( f"'{name_layer}' layer has been loaded" )
    def __repr__( self ) :
        """ # 2022-07-20 00:38:24 
        display RamData
        """
        return f"<{'' if not self._flag_is_read_only else '(read-only) '}RamData object ({'' if self.bc.filter is None else f'{self.bc.meta.n_rows}/'}{self.metadata[ 'int_num_barcodes' ]} barcodes X {'' if self.ft.filter is None else f'{self.ft.meta.n_rows}/'}{self.metadata[ 'int_num_features' ]} features" + ( '' if self.layer is None else f", {self._layer._ramtx_barcodes._int_num_records} records in the currently active layer '{self.layer}'" ) + f") stored at {self._path_folder_ramdata}{'' if self._path_folder_ramdata_mask is None else f' with local mask available at {self._path_folder_ramdata_mask}'}\n\twith the following layers : {self.layers}\n\t\tcurrent layer is '{self.layer}'>" # show the number of records of the current layer if available.
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
        ''' # 2022-07-16 17:10:01 
        please include 'str' in 'barcode_column' and 'feature_column' in order to use string representations in the output AnnData object
        
        possible usages:
        
        [ name_layer, barcode_index, barcode_column, feature_index, feature_column ]
        [ barcode_index, barcode_column, feature_index, feature_column ]
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
        
        # count the number of valid queried entries
        int_num_entries_queried_bc = ba_entry_bc.count( )
        int_num_entries_queried_ft = ba_entry_ft.count( )
        
        # detect and handle the cases when one of the axes is empty
        if int_num_entries_queried_bc == 0 or int_num_entries_queried_ft == 0 :
            if self.verbose :
                print( f"Error: currently queried view is (barcode x features) {int_num_entries_queried_bc} x {int_num_entries_queried_ft}. please change the filter or queries in order to retrieve a valid view" )
            return None

        # retrieve ramtx
        flag_use_ramtx_indexed_by_features = int_num_entries_queried_bc >= int_num_entries_queried_ft # select which axis to use. if there is more number of barcodes than features, use ramtx indexed with 'feature' axis
        rtx = self._layer._ramtx_features if flag_use_ramtx_indexed_by_features else self._layer._ramtx_barcodes
        
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
        if current ramdata location is read-only, None will be returned.
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
    def summarize( self, name_layer, axis, summarizing_func, int_num_threads = None, flag_overwrite_columns = True, int_num_entries_for_each_weight_calculation_batch = 1000, int_total_weight_for_each_batch = 1000000 ) :
        ''' # 2022-07-20 23:40:02 
        this function summarize entries of the given axis (0 = barcode, 1 = feature) using the given function
        
        example usage: calculate total sum, standard deviation, pathway enrichment score calculation, etc.
        
        =========
        inputs 
        =========
        'name_layer' : name of the data in the given RamData object to summarize
        'axis': int or str. 
               0 or 'barcode' for applying a given summarizing function for barcodes
               1 or 'feature' for applying a given summarizing function for features
        'summarizing_func' : function object. a function that takes a RAMtx output and return a dictionary containing 'name_of_summarized_data' as key and 'value_of_summarized_data' as value. the resulting summarized outputs will be added as metadata of the given Axis (self.bc.meta or self.ft.meta)
        
                    summarizing_func( self, int_entry_indexed, arr_int_entries_not_indexed, arr_value ) -> dictionary containing 'key' as summarized metric name and 'value' as a summarized value for the entry
                    
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
                    
        'int_num_threads' : the number of CPUs to use. by default, the number of CPUs set by the RamData attribute 'int_num_cpus' will be used.
        'flag_overwrite_columns' : (Default: True) overwrite the columns of the output annotation dataframe of RamData.adata if columns with the same colume name exists
        'int_num_entries_for_each_weight_calculation_batch' : number of entries to process for each batch for calculation of weights of each entries (weight = the number of unfiltered matrix records to process)
        'int_total_weight_for_each_batch' : the total weight (minimum weight) for each batch for summary calculation
        
        =========
        outputs 
        =========
        the summarized metrics will be added to appropriate dataframe attribute of the AnnData of the current RamData (self.adata.obs for axis = 0 and self.adata.var for axis = 1).
        the column names will be constructed as the following :
            f"{name_layer}_{key}"
        if the column name already exist in the dataframe, the values of the columns will be overwritten (alternatively, a suffix of current datetime will be added to the column name, by setting 'flag_overwrite_columns' to False)
        '''
        """
        1) Prepare
        """
        # check the validility of the input arguments
        if name_layer not in self.layers :
            if self.verbose :
                print( f"[ERROR] [RamData.summarize] invalid argument 'name_layer' : '{name_layer}' does not exist." )
            return -1 
        if axis not in { 0, 'barcode', 1, 'feature' } :
            if self.verbose :
                print( f"[ERROR] [RamData.summarize] invalid argument 'axis' : '{name_layer}' is invalid. use one of { { 0, 'barcode', 1, 'feature' } }" )
            return -1 
        # set layer
        self.layer = name_layer
        # handle inputs
        flag_summarizing_barcode = axis in { 0, 'barcode' } # retrieve a flag indicating whether the data is summarized for each barcode or not
        
        # retrieve the total number of entries in the axis that was not indexed (if calculating average expression of feature across barcodes, divide expression with # of barcodes, and vice versa.)
        int_total_num_entries_not_indexed = self.ft.meta.n_rows if flag_summarizing_barcode else self.bc.meta.n_rows 

        if int_num_threads is None : # use default number of threads
            int_num_threads = self.int_num_cpus
        if summarizing_func == 'sum' :
            def summarizing_func( self, int_entry_indexed, arr_int_entries_not_indexed, arr_value ) :
                ''' # 2022-07-19 12:19:49 
                calculate sum of the values of the current entry
                
                assumes 'int_num_records' > 0
                '''
                int_num_records = len( arr_value ) # retrieve the number of records of the current entry
                dict_summary = { 'sum' : np.sum( arr_value ) if int_num_records > 30 else sum( arr_value ) } # if an input array has more than 30 elements, use np.sum to calculate the sum
                dict_summary[ 'mean' ] = dict_summary[ 'sum' ] / int_total_num_entries_not_indexed # calculate the mean
                return dict_summary
        elif summarizing_func == 'sum_and_dev' :
            def summarizing_func( self, int_entry_indexed, arr_int_entries_not_indexed, arr_value ) :
                ''' # 2022-07-19 12:19:53 
                calculate sum and deviation of the values of the current entry
                
                assumes 'int_num_records' > 0
                '''
                int_num_records = len( arr_value ) # retrieve the number of records of the current entry
                dict_summary = { 'sum' : np.sum( arr_value ) if int_num_records > 30 else sum( arr_value ) } # if an input array has more than 30 elements, use np.sum to calculate the sum
                dict_summary[ 'mean' ] = dict_summary[ 'sum' ] / int_total_num_entries_not_indexed # calculate the mean
                arr_dev = ( arr_value - dict_summary[ 'mean' ] ) ** 2 # calculate the deviation
                dict_summary[ 'deviation' ] = np.sum( arr_dev ) if int_num_records > 30 else sum( arr_dev )
                dict_summary[ 'variance' ] = dict_summary[ 'deviation' ] / ( int_total_num_entries_not_indexed - 1 ) if int_total_num_entries_not_indexed > 1 else np.nan
                return dict_summary
        elif not hasattr( summarizing_func, '__call__' ) : # if 'summarizing_func' is not a function, report error message and exit
            if self.verbose :
                print( f"given summarizing_func is not a function, exiting" )
            return -1
        # retrieve the list of key values returned by 'summarizing_func' by applying dummy values
        arr_dummy_one, arr_dummy_zero = np.ones( 10, dtype = int ), np.zeros( 10, dtype = int )
        dict_res = summarizing_func( self, 0, arr_dummy_zero, arr_dummy_one )
        l_name_col_summarized = sorted( list( dict_res ) ) # retrieve the list of key values of an dict_res result returned by 'summarizing_func'
        l_name_col_summarized_with_name_layer_prefix = list( f"{name_layer}_{e}" for e in l_name_col_summarized ) # retrieve the name_col containing summarized data with f'{name_layer}_' prefix 
        
        
        # retrieve RAMtx object to summarize
        rtx = self._layer._ramtx_barcodes if flag_summarizing_barcode else self._layer._ramtx_features
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
        def process_batch( l_int_entry_current_batch, pipe_to_main_process ) :
            ''' # 2022-05-08 13:19:07 
            summarize a given list of entries, and write summarized result as a tsv file, and return the path to the output file
            '''
            # retrieve the number of index_entries
            int_num_entries_in_a_batch = len( l_int_entry_current_batch )
            
            if int_num_entries_in_a_batch == 0 :
                print( 'empty batch detected' )
            
            # open an output file
            path_file_output = f"{path_folder_temp}{UUID( )}.tsv.gz" # define path of the output file
            newfile = gzip.open( path_file_output, 'wb' )
            
            # iterate through the data of each entry
            for int_entry_indexed_valid, arr_int_entry_not_indexed, arr_value in zip( * rtx[ l_int_entry_current_batch ] ) : # retrieve data for the current batch
                # retrieve summary for the entry
                dict_res = summarizing_func( self, int_entry_indexed_valid, arr_int_entry_not_indexed, arr_value ) # summarize the data for the entry
                # write the result to an output file
                newfile.write( ( '\t'.join( map( str, [ int_entry_indexed_valid ] + list( dict_res[ name_col ] if name_col in dict_res else np.nan for name_col in l_name_col_summarized ) ) ) + '\n' ).encode( ) ) # write an index for the current entry # 0>1 coordinate conversion for 'int_entry'
            newfile.close( ) # close file
            pipe_to_main_process.send( path_file_output ) # send information about the output file
        def post_process_batch( path_file_result ) :
            """ # 2022-07-06 03:21:49 
            """
            df = pd.read_csv( path_file_result, sep = '\t', index_col = 0, header = None ) # read summarized output file, using the first column as the integer indices of the entries
            df.columns = l_name_col_summarized_with_name_layer_prefix # name columns of the dataframe using 'name_col' with f'{name_layer}_' prefix
            ax.meta.update( df, flag_use_index_as_integer_indices = True ) # update metadata using ZarrDataFrame method
            os.remove( path_file_result ) # remove the output file
        # summarize the RAMtx using multiple processes
        bk.Multiprocessing_Batch( rtx.batch_generator( ax.filter, int_num_entries_for_each_weight_calculation_batch, int_total_weight_for_each_batch ), process_batch, post_process_batch = post_process_batch, int_num_threads = int_num_threads, int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop = 0.2 )
        
        # remove temp folder
        shutil.rmtree( path_folder_temp )
    def apply( self, name_layer, name_layer_new, func = None, path_folder_ramdata_output = None, dtype_of_row_and_col_indices = np.int32, dtype_of_value = np.float64, int_num_threads = None, flag_simultaneous_processing_of_paired_ramtx = True, int_num_entries_for_each_weight_calculation_batch = 1000, int_total_weight_for_each_batch = 1000000 ) :
        ''' # 2022-07-06 10:22:21 
        this function apply a function and/or filters to the records of the given data, and create a new data object with 'name_layer_new' as its name.
        
        example usage: calculate normalized count data, perform log1p transformation, cell filtering, etc.                             
        
        =========
        inputs 
        =========

        'name_layer' : (required) name of the data in the given RamData object to analyze
        'name_layer_new' : (required) name of the new data for the paired RAMtx objects that will contains transformed values (the outputs of the functions applied to previous data values). The disk size of the RAMtx objects can be larger or smaller than the RAMtx objects of 'name_layer'. please make sure that sufficient disk space remains before calling this function.
        'path_folder_ramdata_output' : (Default: store inside the current RamData). The directory of the RamData object that will contain the outputs (paired RAMtx objects). if integer representations of features and barcodes are updated from filtering, the output RAMtx is now incompatible with the current RamData and should be stored as a separate RamData object. The output directory of the new RamData object can be given through this argument.
        'func' : function object or string (Default: identity) a function that takes a tuple of two integers (integer representations of barcode and feature) and another integer or float (value) and returns a modified record. Also, the current RamData object will be given as the first argument (self), and attributes of the current RamData can be used inside the function

                 func( self, int_entry_indexed, arr_int_entries_not_indexed, arr_value ) -> int_entry_indexed, arr_int_entries_not_indexed, arr_value

                 if None is returned, the entry will be discarded in the output RAMtx object. Therefore, the function can be used to both filter or/and transform values
                 
                 a list of pre-defined functions are the followings:
                 'log1p' :
                          X_new = log_10(X_old + 1)
                 'ident' or None :
                          X_new = X_old
                 
        'flag_simultaneous_processing_of_paired_ramtx' : (Default: True) process the paired RAMtx simultaneously using two processes.
        'int_num_threads' : the number of CPUs to use. by default, the number of CPUs set by the RamData attribute 'int_num_cpus' will be used.
        'dtype_of_row_and_col_indices', 'dtype_of_value' : the dtype of the output matrix

        =================
        input attributes 
        =================
        the attributes shown below or any other custom attributes can be used internally as READ-ONLY data objects when executing the given 'func'. 
        
        For example, one can define the following function:
        
        ram = RamData( path_folder_to_ramdata )
        ram.a_variable = 10
        def func( self, int_barcode, int_feature, value ) :
            return int_barcode, int_feature, value * self.a_variable
        
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
        
        # parse 'func' or set default functions, retrieving 'func_bc' and 'func_ft'.
        if isinstance( func, dict ) : # if 'func' is dictionary, parse functions for each axes
            func_bc = func[ 'barcode' ]
            func_ft = func[ 'feature' ]
        elif isinstance( func, tuple ) :
            assert len( func ) == 2 # if 'func' is tuple, the length of 'func' should be 2
            func_bc, func_ft = func
        elif func == 'ident' or func is None  :
            # define identity function if 'func' has not been given
            def func_bc( self, int_entry_indexed, arr_int_entries_not_indexed, arr_value ) :
                return int_entry_indexed, arr_int_entries_not_indexed, arr_value
            func_ft = func_bc # use the same function for the other axis
        elif func == 'log1p' :
            def func_bc( self, int_entry_indexed, arr_int_entries_not_indexed, arr_value ) :
                return int_entry_indexed, arr_int_entries_not_indexed, np.log10( arr_value + 1 )
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
        
        def RAMtx_Apply( self, rtx, ax, func, path_folder_ramtx_output, int_num_threads ) :
            ''' # 2022-07-06 10:22:12 
            
            inputs 
            =========

            'rtx': an input RAMtx object
            'path_folder_ramtx_output' : an output folder for the new RAMtx object
            '''
            # create an ramtx output folder
            os.makedirs( path_folder_ramtx_output, exist_ok = True )
            # create a temporary folder
            path_folder_temp = f'{path_folder_ramtx_output}temp_{UUID( )}/'
            for path_folder in [ path_folder_ramtx_output, path_folder_temp ] :
                os.makedirs( path_folder, exist_ok = True )
                
            # open Zarr matrix and index objects of the output RAMtx
            path_folder_za_mtx_ramtx = f"{path_folder_ramtx_output}matrix.zarr/" # retrieve the folder path of the output RAMtx Zarr matrix object.
            za_mtx_index = zarr.open( f'{path_folder_ramtx_output}matrix.index.zarr', mode = 'w', shape = rtx._za_mtx_index.shape, chunks = rtx._za_mtx_index.chunks, dtype = rtx._za_mtx_index.dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # use the same dtype and chunk size of the current RAMtx
            za_mtx = zarr.open( path_folder_za_mtx_ramtx, mode = 'w', shape = rtx._za_mtx.shape, chunks = rtx._za_mtx.chunks, dtype = dtype_of_value, synchronizer = zarr.ThreadSynchronizer( ) ) # use the same chunk size of the current RAMtx
            ns = dict( ) # create a namespace that can safely shared between different scopes of the functions
            ns[ 'int_num_records_written_to_ramtx' ] = 0 # initlaize the total number of records written to ramtx object
            ns[ 'int_num_chunks_written_to_ramtx' ] = 0 # initialize the number of chunks written to ramtx object
            int_num_records_in_a_chunk_of_mtx = rtx._za_mtx.chunks[ 0 ] # retrieve the number of records in a chunk of a Zarr matrix
            
            """ convert matrix values and save it to the output RAMtx object """
            # define functions for multiprocessing step
            def process_batch( l_int_entry_current_batch, pipe_to_main_process ) :
                ''' # 2022-05-08 13:19:07 
                retrieve data for a given list of entries, transform values, and save to a Zarr object and index the object, and returns the number of written records and the paths of the written objects (index and Zarr matrix)
                '''
                # retrieve the number of index_entries
                int_num_entries = len( l_int_entry_current_batch )

                # open an Zarr object
                path_folder_zarr_output = f"{path_folder_temp}{UUID( )}.zarr/" # define output Zarr object path
                za_output = zarr.open( path_folder_zarr_output, mode = 'w', shape = rtx._za_mtx.shape, chunks = rtx._za_mtx.chunks, dtype = dtype_of_value, synchronizer = zarr.ThreadSynchronizer( ) )
                # open an index file
                path_file_index_output = f"{path_folder_temp}{UUID( )}.index.tsv.gz" # define output index file path
                newfile_index = gzip.open( path_file_index_output, 'wb' )
                int_num_records_written = 0 # initialize the record count

                # iterate through the data of each entry
                for int_entry_indexed_valid, arr_int_entry_not_indexed, arr_value in zip( * rtx[ l_int_entry_current_batch ] ) : # retrieve data for the current batch
                    # transform the values of an entry
                    int_entry_indexed_valid, arr_int_entry_not_indexed, arr_value = func( self, int_entry_indexed_valid, arr_int_entry_not_indexed, arr_value ) 
                    int_num_records = len( arr_value ) # retrieve number of returned records
                    za_output[ int_num_records_written : int_num_records_written + int_num_records ] = np.vstack( ( arr_int_entry_not_indexed, arr_value ) ).T # save transformed data
                    # write the result to the index file
                    newfile_index.write( ( '\t'.join( map( str, [ int_entry_indexed_valid, int_num_records_written, int_num_records_written + int_num_records ] ) ) + '\n' ).encode( ) ) # write an index for the current entry # 0>1 coordinate conversion for 'int_entry'
                    # update the number of records written
                    int_num_records_written += int_num_records
                newfile_index.close( ) # close file
                za_output.resize( int_num_records_written, 2 ) # resize the output Zarr object so that there is no 
                pipe_to_main_process.send( ( int_num_records_written, path_folder_zarr_output, path_file_index_output ) ) # send information about the output files
            def post_process_batch( res ) :
                """ # 2022-07-06 10:22:05 
                """
                # parse result
                int_num_records_written, path_folder_zarr_output, path_file_index_output = res
                
                ns[ 'int_num_records_written_to_ramtx' ] += int_num_records_written # update the number of records written to the output RAMtx
                int_num_chunks_written_for_a_batch = int( np.ceil( int_num_records_written / int_num_records_in_a_chunk_of_mtx ) ) # retrieve the number of chunks that were written for a batch
                int_num_chunks_written_to_ramtx = ns[ 'int_num_chunks_written_to_ramtx' ] # retrieve the number of chunks already present in the output RAMtx zarr matrix object
                
                # check size of Zarr matrix object, and increase the size if needed.
                int_min_num_rows_required = ( int_num_chunks_written_to_ramtx + int_num_chunks_written_for_a_batch ) * int_num_records_in_a_chunk_of_mtx # calculate the minimal number of rows required in the RAMtx Zarr matrix object
                if za_mtx.shape[ 0 ] < int_min_num_rows_required : # check whether the size of Zarr matrix is smaller than the minimum requirement
                    za_mtx.resize( int_min_num_rows_required, 2 ) # resize the Zarr matrix so that data can be safely added
                
                # copy Zarr chunks to RAMtx Zarr matrix object
                os.chdir( path_folder_zarr_output )
                for e in glob.glob( '*.0' ) : # to reduce the size of file paths returned by glob, use relative path to retrieve the list of chunk files of the Zarr matrix of the current batch
                    index_chunk = int( e.split( '.0', 1 )[ 0 ] ) # retrieve the integer index of the chunk
                    os.rename( e, path_folder_za_mtx_ramtx + str( index_chunk + int_num_chunks_written_to_ramtx ) + '.0' ) # simply rename the chunk to transfer stored values
                
                # retrieve index data of the current batch
                arr_index = pd.read_csv( path_file_index_output, header = None, sep = '\t' ).values.astype( int ) # convert to integer dtype
                arr_index[ :, 1 : ] += int_num_chunks_written_to_ramtx * int_num_records_in_a_chunk_of_mtx # match the chunk boundary. if there are empty rows in the chunks currently written to ramtx, these empty rows will be considered as rows containing records, so that Zarr matrix written for a batch can be easily transferred by simply renaming the chunk files
                za_mtx_index.set_orthogonal_selection( arr_index[ :, 0 ], arr_index[ :, 1 : ] ) # update the index of the entries of the current batch
                
                # update the number of chunks written to RAMtx Zarr matrix object
                ns[ 'int_num_chunks_written_to_ramtx' ] += int_num_chunks_written_for_a_batch
                
                # delete temporary files and folders
                shutil.rmtree( path_folder_zarr_output )
                os.remove( path_file_index_output )
                
            # transform the values of the RAMtx using multiple processes
            bk.Multiprocessing_Batch( rtx.batch_generator( ax.filter, int_num_entries_for_each_weight_calculation_batch, int_total_weight_for_each_batch ), process_batch, post_process_batch = post_process_batch, int_num_threads = int_num_threads, int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop = 0.2 )

            # remove temp folder
            shutil.rmtree( path_folder_temp )
            
            ''' export ramtx settings '''
            root = zarr.group( path_folder_ramtx_output )
            root.attrs[ 'dict_ramtx_metadata' ] = { 
                'flag_ramtx_sorted_by_id_feature' : rtx.flag_ramtx_sorted_by_id_feature,
                'str_completed_time' : TIME_GET_timestamp( True ),
                'int_num_features' : rtx._int_num_features,
                'int_num_barcodes' : rtx._int_num_barcodes,
                'int_num_records' : ns[ 'int_num_records_written_to_ramtx' ],
                'int_num_of_records_in_a_chunk_zarr_matrix' : rtx._za_mtx.chunks[ 0 ],
                'int_num_of_entries_in_a_chunk_zarr_matrix_index' : rtx._za_mtx_index.chunks[ 0 ],
                'version' : _version_,
            }
            return 
            
        """
        Create output folders and copy feature and barcode files
        """
        # apply the given function to each RAMtx object
        path_folder_data_new = f"{path_folder_ramdata_output}{name_layer_new}/" # compose the output directory of the paird RAMtx objects inside the output RamData object
                
        if flag_simultaneous_processing_of_paired_ramtx :
            l_process = list( mp.Process( target = RAMtx_Apply, args = ( self, rtx, ax, func, path_folder_ramtx_output, int_num_threads_for_the_current_process ) ) for rtx, ax, func, path_folder_ramtx_output, int_num_threads_for_the_current_process in zip( [ self._layer._ramtx_features, self._layer._ramtx_barcodes ], [ self.ft, self.bc ], [ func_ft, func_bc ], [ f"{path_folder_data_new}sorted_by_feature/", f"{path_folder_data_new}sorted_by_barcode/" ], [ int( np.floor( int_num_threads / 2 ) ), int( np.ceil( int_num_threads / 2 ) ) ] ) )
            for p in l_process : p.start( )
            for p in l_process : p.join( )
        else :
            RAMtx_Apply( self, self._layer._ramtx_features, self.ft, func_ft, f"{path_folder_data_new}sorted_by_feature/", int_num_threads = int_num_threads )
            RAMtx_Apply( self, self._layer._ramtx_barcodes, self.bc, func_bc, f"{path_folder_data_new}sorted_by_barcode/", int_num_threads = int_num_threads )

        if self.verbose :
            print( f'new layer {name_layer_new} has been successfully created' )
            
        # update 'layers' if the layer has been saved in the current RamData object (or the mask of the current RamData object)
        if flag_new_layer_added_to_the_current_ramdata :
            self.layers.add( name_layer_new )
            self._save_metadata_( )
    def subset( self, path_folder_ramdata_output, l_name_layer = [ ], int_num_threads = None, flag_simultaneous_processing_of_paired_ramtx = True, int_num_entries_for_each_weight_calculation_batch = 1000, int_total_weight_for_each_batch = 1000000 ) :
        ''' # 2022-07-05 23:20:02 
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
            self.apply( name_layer, name_layer_new = None, func = 'ident', path_folder_ramdata_output = path_folder_ramdata_output, flag_simultaneous_processing_of_paired_ramtx = flag_simultaneous_processing_of_paired_ramtx, int_num_threads = int_num_threads, int_num_entries_for_each_weight_calculation_batch = int_num_entries_for_each_weight_calculation_batch, int_total_weight_for_each_batch = int_total_weight_for_each_batch ) # flag_dtype_output = None : use the same dtype as the input RAMtx object
        self.unload_dict_change( ) # unload 'dict_change' after the conversion process
        
        # compose metadata
        root = zarr.group( path_folder_ramdata_output )
        root.attrs[ 'dict_ramdata_metadata' ] = { 
            'str_completed_time' : TIME_GET_timestamp( True ),
            'int_num_features' : self.ft.meta.n_rows, # record the number of features/barcodes after filtering
            'int_num_barcodes' : self.bc.meta.n_rows,
            'int_num_of_records_in_a_chunk_zarr_matrix' : self.metadata[ 'int_num_of_records_in_a_chunk_zarr_matrix' ],
            'int_num_of_entries_in_a_chunk_zarr_matrix_index' : self.metadata[ 'int_num_of_entries_in_a_chunk_zarr_matrix_index' ],
            'layers' : list( set_name_layer ),
            'version' : _version_,
        }
    def normalize( self, name_layer = 'raw', name_layer_new = 'normalized', name_col_total_count = 'raw_sum', int_total_count_target = 10000, flag_simultaneous_processing_of_paired_ramtx = True, int_num_threads = None, ** args ) :
        ''' # 2022-07-06 23:58:15 
        this function perform normalization of a given data and will create a new data in the current RamData object.

        =========
        inputs 
        =========

        'name_layer' : name of input data
        'name_layer_new' : name of the output (normalized) data
        'name_col_total_count' : name of column of barcode metadata (ZarrDataFrame) to use as total counts of barcodes
        'int_total_count_target' : the target total count. the count data will be normalized according to the total counts of barcodes so that the total counts of each barcode after normalization becomes 'int_total_count_target'.
        'flag_simultaneous_processing_of_paired_ramtx' : (Default: True) process the paired RAMtx simultaneously using two processes at a time.
        'int_num_threads' : the number of CPUs to use. by default, the number of CPUs set by the RamData attribute 'int_num_cpus' will be used.
        
        ** args : arguments for 'self.apply' method
        '''
        # check validity of inputs
        assert name_col_total_count in self.bc.meta # 'name_col_total_count' column should be available in the metadata
        
        # load total count data
        flag_name_col_total_count_already_loaded = name_col_total_count in self.bc.meta.dict # a flag indicating that the total count data of barcodes has been already loaded
        if not flag_name_col_total_count_already_loaded : # load count data of barcodes in memory
            self.bc.meta.load_as_dict( name_col_total_count )
        dict_count = self.bc.meta.dict[ name_col_total_count ] # retrieve total count data as a dictionary
        
        # load layer
        self.layer = name_layer
        
        # define functions for normalization
        def func_norm_barcode_indexed( self, int_entry_indexed, arr_int_entries_not_indexed, arr_value ) :
            """ # 2022-07-06 23:58:27 
            """
            return int_entry_indexed, arr_int_entries_not_indexed, ( arr_value / dict_count[ int_entry_indexed ] * int_total_count_target ) # normalize count data of a single barcode
        
        def func_norm_feature_indexed( self, int_entry_indexed, arr_int_entries_not_indexed, arr_value ) : # normalize count data of a single feature containing (possibly) multiple barcodes
            """ # 2022-07-06 23:58:38 
            """
            # perform normalization in-place
            for i, e in enumerate( arr_int_entries_not_indexed.astype( int ) ) : # iterate through barcodes
                arr_value[ i ] = arr_value[ i ] / dict_count[ e ] # perform normalization of count data for each barcode
            arr_value *= int_total_count_target
            return int_entry_indexed, arr_int_entries_not_indexed, arr_value
        
        ''' normalize the RAMtx matrices '''
        self.apply( name_layer, name_layer_new, func = ( func_norm_barcode_indexed, func_norm_feature_indexed ), flag_simultaneous_processing_of_paired_ramtx = flag_simultaneous_processing_of_paired_ramtx, int_num_threads = int_num_threads, ** args ) # flag_dtype_output = None : use the same dtype as the input RAMtx object
    
        if not flag_name_col_total_count_already_loaded : # unload count data of barcodes from memory if the count data was not loaded before calling this method
            del self.bc.meta.dict[ name_col_total_count ]
    def identify_highly_variable_features( self, name_layer = 'normalized_log1p', int_num_highly_variable_features = 2000, float_min_mean = 0.01, float_min_variance = 0.01, name_col_filter = 'filter_normalized_log1p_highly_variable', flag_show_graph = True, coords_for_sampling = range( 0, 1000000 ) ) :
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
        self.ft.filter = None
        
        # retrieve a filter of features that satisfy the conditions
        ba = self.ft.AND( self.ft.meta[ 'normalized_log1p_variance' ] > float_min_variance, self.ft.meta[ 'normalized_log1p_mean' ] > float_min_mean )

        # set default threshold for highly_variable_feature_score
        float_min_score_highly_variable = 0
        # select the specified number of features if 'int_num_highly_variable_features' is given
        if int_num_highly_variable_features is None :
            self.ft.filter = ba # set feature filter for retrieving highly_variable_feature_score for only the genes satisfying the thresholds.

            # retrieve highly variable scores and set the threshold based on the 'int_num_highly_variable_features'
            arr = self.ft.meta[ 'normalized_log1p__float_score_highly_variable_feature_from_mean' ]
            arr.sort( )
            float_min_score_highly_variable = arr[ - int_num_highly_variable_features ] if len( arr ) >= int_num_highly_variable_features else arr[ 0 ] # retrieve threshold (if the number of available features are less than 'int_num_highly_variable_features', use all genes.

            # reset feature filter prior to the identification of highly variable features.
            self.ft.filter = None
        
        # identify highly variable features satisfying all conditions    
        ba = self.ft.AND( ba, self.ft.meta[ 'normalized_log1p__float_score_highly_variable_feature_from_mean' ] > float_min_score_highly_variable ) 
        self.ft.meta[ name_col_filter ] = ba # save the feature filter as a metadata
        self.ft.filter = ba

        # discard cells that does not contain data for the identified highly variable features
        self.summarize( 'normalized_log1p', 'barcode', 'sum' ) # calculate the total expression values for the selected highly variable genes for each barcode
        self.bc.filter = None # reset barcode filter prior to the identification of valid barcodes (that contains valid expression data for the selected set of highly variable features).
        ba = self.bc.meta[ 'normalized_log1p_sum' ] > 0 # filter out barcodes with no expression for the selected highly variable genes, since these will not be clustered properly and consumes extra memory and running time.
        self.bc.meta[ name_col_filter, : ] = ba # save the barcode filter as a metadata
        self.bc.filter = ba
    ''' utility functions for filter '''
    def change_filter( self, name_col_filter = None, name_col_filter_bc = None, name_col_filter_ft = None ) :
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
    def save_filter( self, name_col_filter = None, name_col_filter_bc = None, name_col_filter_ft = None  ) :
        """ # 2022-07-16 17:27:54 
        save filters for 'barcode' and 'feature' Axes
        
        'name_col_filter_bc', 'name_col_filter_ft' will override 'name_col_filter' when saving filters
        if filter has not been set, filter containing all active entries (containing valid count data) will be saved instead for consistency
        
        if all name_cols are invalid, no filters will be saved
        """
        # check validity of name_cols for filter
        # bc
        if name_col_filter_bc not in self.bc.meta :
            name_col_filter_bc = name_col_filter if name_col_filter in self.bc.meta else None # use 'name_col_filter' instead if 'name_col_filter_bc' is invalid
        # ft
        if name_col_filter_ft not in self.ft.meta :
            name_col_filter_ft = name_col_filter if name_col_filter in self.ft.meta else None # use 'name_col_filter' instead if 'name_col_filter_ft' is invalid
        
        # save filters
        self.bc.save_filter( name_col_filter_bc ) # bc
        self.ft.save_filter( name_col_filter_ft ) # ft
    ''' memory-efficient demension reduction/clustering functions '''
    def pca( self, name_layer = 'normalized_log1p', prefix_name_col = 'pca_', int_num_components = 50, int_num_barcodes_in_ipca_batch = 1000, name_col_filter = 'filter_normalized_log1p_highly_variable', float_prop_subsampling = 1, name_col_filter_subsampled = 'filter_subsampling_for_pca', flag_ipca_whiten = False, name_model = 'ipca', int_num_threads = 3, flag_show_graph = True ) :
        """ # 2022-07-18 15:09:57 
        Perform incremental PCA in a very memory-efficient manner.
        the resulting incremental PCA model will be saved in the RamData.ns database.
        
        arguments:
        'name_layer' : name of the data source layer (the layer from which gene expression data will be retrieved for the barcodes)
        'prefix_name_col' : a prefix for the 'name_col' of the columns that will be added to Axis.meta ZDF.
        'name_col_filter' : the name of 'feature'/'barcode' Axis metadata column to retrieve selection filter for highly-variable-features. (default: None) if None is given, current feature filter (if it has been set) will be used as-is. if a valid filter is given, filter WILL BE CHANGED.
        'name_col_filter_subsampled' : the name of 'feature'/'barcode' Axis metadata column to retrieve or save mask containing subsampled barcodes. if 'None' is given and 'float_prop_subsampling' is below 1 (i.e. subsampling will be used), the subsampling filter generated for retrieving gene expression data of selected barcodes will not be saved.
        'int_num_components' : number of PCA components.
        'int_num_barcodes_in_ipca_batch' : number of barcodes in an Incremental PCA computation
        'float_prop_subsampling' : proportion of barcodes to used to train representation of single-barcode data using incremental PCA. 1 = all barcodes, 0.1 = 10% of barcodes, etc. subsampling will be performed using a random probability, meaning the actual number of barcodes subsampled will not be same every time.
        'flag_ipca_whiten' : a flag for an incremental PCA computation (Setting this flag to 'True' will reduce the efficiency of model learning, but might make the model more generalizable)
        'name_model' : the trained incremental PCA model will be saved to RamData.ns database with this name. if None is given, the model will not be saved.
        'int_num_threads' : number of threads for parallel data retrieval/iPCA transformation/ZarrDataFrame update. 3~5 would be ideal.
        'flag_show_graph' : show graph
        """
        """
        1) Prepare
        """
        # check the validility of the input arguments
        if name_layer not in self.layers :
            if self.verbose :
                print( f"[ERROR] [RamData.summarize] invalid argument 'name_layer' : '{name_layer}' does not exist." )
            return -1 
        # set layer
        self.layer = name_layer

        # retrieve RAMtx object (sorted by barcodes) to summarize # retrieve 'Barcode' Axis object
        rtx, ax = self._layer._ramtx_barcodes, self.bc

        # set filters for PCA calculation
        if name_col_filter is not None :
            self.change_filter( name_col_filter )
        
        # create view for 'feature' Axis
        self.ft.create_view( )
        
        # retrieve a flag indicating whether a subsampling is active
        flag_is_subsampling_active = ( name_col_filter_subsampled in self.bc.meta ) or ( float_prop_subsampling is not None and float_prop_subsampling < 1 )
        
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
            int_num_of_previously_returned_entries, l_int_entry_current_batch = batch 
            int_num_retrieved_entries = len( l_int_entry_current_batch )

            pipe_to_main_process.send( ( int_num_of_previously_returned_entries, int_num_retrieved_entries, rtx.get_sparse_matrix( l_int_entry_current_batch )[ int_num_of_previously_returned_entries : int_num_of_previously_returned_entries + int_num_retrieved_entries ] ) ) # retrieve and send sparse matrix as an input to the incremental PCA # resize sparse matrix
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
            
            if self.verbose : # report
                print( f'fit completed for {int_num_of_previously_returned_entries + 1}-{int_num_of_previously_returned_entries + int_num_retrieved_entries} barcodes' )
        # fit iPCA using multiple processes
        bk.Multiprocessing_Batch( ax.batch_generator( int_num_entries_for_batch = int_num_barcodes_in_ipca_batch, flag_return_the_number_of_previously_returned_entries = True ), process_batch, post_process_batch = post_process_batch, int_num_threads = max( min( int_num_threads, 5 ), 2 ), int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop = 0.2 ) # number of threads for multi-processing is 2 ~ 5 # generate batch with fixed number of barcodes
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
            self.bc.create_view( ) # recreate view for the 'barcode' Axis

        """
        2) Transform Data
        """
        # initialize metadata columns 
        l_name_col_component = list( f"{prefix_name_col}{i}" for i in range( int_num_components ) ) # retrieve name_col for the transformed PCA data

        # define functions for multiprocessing step
        def process_batch( batch, pipe_to_main_process ) :
            ''' # 2022-07-13 22:18:22 
            retrieve data and retrieve transformed PCA values for the batch
            '''
            # parse the received batch
            int_num_of_previously_returned_entries, l_int_entry_current_batch = batch 
            int_num_retrieved_entries = len( l_int_entry_current_batch )

            pipe_to_main_process.send( ( l_int_entry_current_batch, rtx.get_sparse_matrix( l_int_entry_current_batch )[ int_num_of_previously_returned_entries : int_num_of_previously_returned_entries + int_num_retrieved_entries ] ) ) # retrieve data as a sparse matrix and send the result of PCA transformation # send the integer representations of the barcodes for PCA value update
        def post_process_batch( res ) :
            """ # 2022-07-13 22:18:26 
            perform partial fit for batch
            """
            # parse result 
            l_int_entry_current_batch, X = res
            
            X_transformed = ipca.transform( X ) # perform PCA transformation
            del X

            # iterate through components
            for i, arr_component in enumerate( X_transformed.T ) :
                # update the components for the barcodes of the current batch
                ax.meta[ l_name_col_component[ i ], l_int_entry_current_batch ] = arr_component

        # transform values using iPCA using multiple processes
        bk.Multiprocessing_Batch( ax.batch_generator( int_num_entries_for_batch = int_num_barcodes_in_ipca_batch, flag_return_the_number_of_previously_returned_entries = True ), process_batch, post_process_batch = post_process_batch, int_num_threads = max( min( int_num_threads, 5 ), 2 ), int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop = 0.2 ) # number of threads for multi-processing is 2 ~ 5 # generate batch with fixed number of barcodes

        # destroy the view
        self.destroy_view( )
        
        # save iPCA metrics in the shelve
        if name_model is not None : # if the given 'name_model' is valid 
            self.ns[ name_model ] = ipca # save the trained model to the database
        
        # draw graphs
        if flag_show_graph :
            # draw 'explained variance ratio' graph
            fig, ax = plt.subplots( 1, 1 )
            ax.plot( ipca.explained_variance_ratio_, 'o-' )
            bk.MATPLOTLIB_basic_configuration( x_label = 'principal components', y_label = 'explained variance ratio', title = 'PCA result', show_grid = True )
        
        return ipca # return the model
    def get_array( self, axis = 'barcode', prefix_name_col = 'pca_', int_num_components = 50, name_col_filter = None, l_int_entries = None ) :
        """ # 2022-07-15 19:59:11 
        Retrieve array data for the 'int_num_components' number of components with 'prefix_name_col' prefix for the currently active filters (if filters have been set).
        if 'l_int_entries' is given, the entries specified by 'l_int_entries' will be retrieved
        
        'axis' : axis to retrieve array data { 'barcode', 0 } for 'barcode' axis and { 'feature', 1 } for 'feature' axis
        'prefix_name_col' : name of columns in the 'barcode' Axis metadata containing PCA components. please refer to RamData.pca function
        'name_col_filter' : the name of 'feature'/'barcode' Axis metadata column to retrieve selection filter for highly-variable-features. (default: None) if None is given, current feature filter (if it has been set) will be used as-is. if a valid filter is given, filter WILL BE CHANGED.
        'l_int_entries' : to retrieve data of a specific entries, use this argument to pass the list of integer representations of the entries. filter (if it is active) will not be applied
        """
        # retrieve Axis object based on the 'axis' argument.
        ax = self.bc if axis in { 'barcode', 0 } else self.ft

        # set appropriate filters
        self.change_filter( name_col_filter )
        
        # survey the maximum number of available PCA components
        int_num_available_components = 0
        for i in range( int_num_components ) :
            name_col = f"{prefix_name_col}{i}"
            if name_col in ax.meta :
                int_num_available_components = i + 1

        # correct the 'int_num_components' to use based on the maximum number of available PCA dimensions
        int_num_components = min( int_num_components, int_num_available_components )

        # initialize the array 
        arr_data = np.zeros( ( ax.meta.n_rows if l_int_entries is None else len( l_int_entries ), int_num_components ) )

        # retrieve PCA transformed data
        for i in range( int_num_components ) :
            name_col = f"{prefix_name_col}{i}"
            if self.verbose :
                print( f"retrieving {name_col}" )
            arr_data[ :, i ] = ax.meta[ name_col ] if l_int_entries is None else ax.meta[ name_col, l_int_entries ] # retrieve data of specific entries if 'l_int_entries' is given

        # return retrieved PCA components
        return arr_data
    def get_pca( self, prefix_name_col = 'pca_', int_num_components = 50, name_col_filter = None, l_int_entries = None ) :
        """ # 2022-07-14 10:01:59 
        a convenient wrapper of 'RamData.get_array' function
        Retrieve PCA data for the 'int_num_components' number of components with 'prefix_name_col' prefix for the currently active filters (if filters have been set).
        if 'l_int_entries' is given, the entries specified by 'l_int_entries' will be retrieved
        
        'prefix_name_col' : name of columns in the 'barcode' Axis metadata containing PCA components. please refer to RamData.pca function
        'name_col_filter' : the name of 'feature'/'barcode' Axis metadata column to retrieve selection filter for highly-variable-features. (default: None) if None is given, current feature filter (if it has been set) will be used as-is. if a valid filter is given, filter WILL BE CHANGED.
        'l_int_entries' : to retrieve data of a specific entries, use this argument to pass the list of integer representations of the entries. filter (if it is active) will not be applied
        """
        return self.get_array( axis = 'barcode', prefix_name_col = prefix_name_col, int_num_components = int_num_components, name_col_filter = name_col_filter, l_int_entries = l_int_entries )
    def umap( self, prefix_name_col_pca = 'pca_', int_num_components_pca = 20, prefix_name_col_umap = 'umap_', int_num_components_umap = 2, int_num_barcodes_in_pumap_batch = 20000, name_col_filter = 'filter_normalized_log1p_highly_variable', float_prop_subsampling = 1, name_col_filter_subsampled = 'filter_subsampling_for_umap', name_pumap_model = 'pumap', name_pumap_model_new = 'pumap' ) :
        """ # 2022-07-16 22:23:46 
        Perform Parametric UMAP to embed cells in reduced dimensions for a scalable analysis of single-cell data
        Parametric UMAP has several advantages over non-parametric UMAP (conventional UMAP), which are 
            (1) GPU can be utilized during training of neural network models
            (2) learned embedding can be applied to other cells not used to build the embedding
            (3) learned embedding can be updated by training with additional cells
        Therefore, parametric UMAP is suited for generating embedding of single-cell data with extremely large number of cells

        arguments:
        'name_layer' : name of the data source layer
        'prefix_name_col_pca' : a prefix for the 'name_col' of the columns containing PCA transformed values.
        'int_num_components_pca' : number of PCA components to use as inputs for Parametric UMAP learning
        'prefix_name_col_umap' : a prefix for the 'name_col' of the columns containing UMAP transformed values.
        'int_num_components_umap' : number of output UMAP components. (default: 2)
        'int_num_barcodes_in_pumap_batch' : number of barcodes in a batch for Parametric UMAP model update.
        'name_col_filter' : the name of 'feature'/'barcode' Axis metadata column to retrieve selection filter for highly-variable-features. (default: None) if None is given, current feature filter (if it has been set) will be used as-is. if a valid filter is given, filter WILL BE CHANGED.
        'float_prop_subsampling' : proportion of barcodes to used to train representation of single-barcode data using Parametric UMAP. 1 = all barcodes, 0.1 = 10% of barcodes, etc. subsampling will be performed using a random probability, meaning the actual number of barcodes subsampled will not be same every time.
        'name_col_filter_subsampled' : the name of 'feature'/'barcode' Axis metadata column to retrieve or save mask containing subsampled barcodes. if 'None' is given and 'float_prop_subsampling' is below 1 (i.e. subsampling will be used), the subsampling filter generated for retrieving gene expression data of selected barcodes will not be saved.
        'name_pumap_model' = 'pumap' : the name of the parametric UMAP model. if None is given, the trained model will not be saved to the RamData object. if the model already exists, the model will be loaded and trained again.
        'name_pumap_model_new' = 'pumap' : the name of the new parametric UMAP model after the training. if None is given, the new model will not be saved. if 'name_pumap_model' and 'name_pumap_model_new' are the same, the previously written model will be overwritten.
        """
        """
        1) Prepare
        """
        # # retrieve 'Barcode' Axis object
        ax = self.bc

        # set filters for UMAP calculation
        if name_col_filter is not None :
            self.change_filter( name_col_filter )

        # retrieve a flag indicating whether a subsampling is active
        flag_is_subsampling_active = ( name_col_filter_subsampled in self.bc.meta ) or ( float_prop_subsampling is not None and float_prop_subsampling < 1 )

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
        
        """
        2) Train Parametric UMAP with/without subsampling of barcodes
        """
        # initialize Parametric UMAP object
        if name_pumap_model is not None and self._path_folder_ramdata_local is not None : # if 'name_pumap_model' has been given # if local RamDatat path is available, check the availability of the model and load the model if the saved model is available
            assert '/' not in name_pumap_model # check validity of 'name_pumap_model'
            path_model = f'{self._path_folder_ramdata_local}{name_pumap_model}.pumap' # model should be loaded/saved in local file system (others are not implemented yet)
            pumap_embedder = pumap.load_ParametricUMAP( path_model ) if os.path.exists( path_model ) else pumap.ParametricUMAP( low_memory = True ) # load model if model exists
                
            # fix 'load_ParametricUMAP' error ('decoder' attribute does not exist)
            if os.path.exists( path_model ) and not hasattr( pumap_embedder, 'decoder' ) : 
                pumap_embedder.decoder = None
        else :
            pumap_embedder = pumap.ParametricUMAP( low_memory = True ) # load an empty model if a saved model is not available

        # iterate through batches
        for int_num_of_previously_returned_entries, l_int_entry_current_batch in ax.batch_generator( int_num_entries_for_batch = int_num_barcodes_in_pumap_batch, flag_return_the_number_of_previously_returned_entries = True, flag_mix_randomly = True ) : # mix barcodes randomly for efficient learning for each batch
            int_num_entries_current_batch = len( l_int_entry_current_batch ) # retrieve the number of entries in the current batch
            # if the number of entries in the current batch is below the given size of the batch 'int_num_barcodes_in_pumap_batch', and the model has been already trained, skip training with smaller number of barcodes, since it can lead to reduced accuracy of embedding due to training with smaller number of barcodes
            if int_num_entries_current_batch < int_num_barcodes_in_pumap_batch and int_num_of_previously_returned_entries > 0 :
                continue
            # train parametric UMAP model
            pumap_embedder.fit( self.get_pca( prefix_name_col = prefix_name_col_pca, int_num_components = int_num_components_pca, l_int_entries = l_int_entry_current_batch ) ) 
            if self.verbose : # report
                print( f'training completed for {int_num_of_previously_returned_entries + 1}-{int_num_of_previously_returned_entries + int_num_entries_current_batch} barcodes' )

        # report
        if self.verbose :
            print( 'training completed' )

        # if subsampling has been completed, revert to the original barcode selection filter
        if flag_is_subsampling_active :
            self.bc.filter = ba_filter_bc_before_subsampling
            del ba_filter_bc_before_subsampling

        """
        2) Transform Data
        """
        # initialize metadata columns 
        l_name_col_component = list( f"{prefix_name_col_umap}{i}" for i in range( int_num_components_umap ) ) # retrieve name_col for the transformed components

        # iterate through batches
        for int_num_of_previously_returned_entries, l_int_entry_current_batch in ax.batch_generator( int_num_entries_for_batch = int_num_barcodes_in_pumap_batch, flag_return_the_number_of_previously_returned_entries = True ) :
            # retrieve UMAP embedding of barcodes of the current batch
            X_transformed = pumap_embedder.transform( self.get_pca( prefix_name_col = prefix_name_col_pca, int_num_components = int_num_components_pca, l_int_entries = l_int_entry_current_batch ) ) 

            # iterate through components
            for i, arr_component in enumerate( X_transformed.T ) :
                # update the components for the barcodes of the current batch
                ax.meta[ l_name_col_component[ i ], l_int_entry_current_batch ] = arr_component

        # save Parametric UMAP object after training 
        if name_pumap_model_new is not None and self._mode != 'r' and self._path_folder_ramdata_local is not None : # if 'name_pumap_model_new' has been given, 'mode' is not 'r', and local RamData path is available
            assert '/' not in name_pumap_model_new # check validity of 'name_pumap_model_new'
            path_model = f'{self._path_folder_ramdata_local}{name_pumap_model_new}.pumap'
            if self.verbose :
                if os.path.exists( path_model ) :
                    print( f'existing model {name_pumap_model_new} will be overwritten' )
            pumap_embedder.save( path_model ) # save model

        return pumap_embedder # return the model
    def get_umap( self, prefix_name_col = 'umap_', int_num_components = 2, name_col_filter = None, l_int_entries = None ) :
        """ # 2022-07-14 10:01:59 
        a convenient wrapper of 'RamData.get_array' function
        Retrieve PCA data for the 'int_num_components' number of components with 'prefix_name_col' prefix for the currently active filters (if filters have been set).
        if 'l_int_entries' is given, the entries specified by 'l_int_entries' will be retrieved
        
        'prefix_name_col' : name of columns in the 'barcode' Axis metadata containing PCA components. please refer to RamData.pca function
        'name_col_filter' : the name of 'feature'/'barcode' Axis metadata column to retrieve selection filter for highly-variable-features. (default: None) if None is given, current feature filter (if it has been set) will be used as-is. if a valid filter is given, filter WILL BE CHANGED.
        'l_int_entries' : to retrieve data of a specific entries, use this argument to pass the list of integer representations of the entries. filter (if it is active) will not be applied
        """
        return self.get_array( axis = 'barcode', prefix_name_col = prefix_name_col, int_num_components = int_num_components, name_col_filter = name_col_filter, l_int_entries = l_int_entries )
    def hdbscan( self, name_model = 'hdbscan', min_cluster_size = 30, min_samples = 30, prefix_name_col_umap = 'umap_', int_num_components_umap = 2, name_col_hdbscan = 'hdbscan', flag_reanalysis_of_previous_clustering_result = False, cut_distance = 0.15, flag_use_pynndescent = True, int_num_threads = 10, int_num_barcodes_in_cluster_label_prediction_batch = 10000, name_col_filter = 'filter_normalized_log1p_highly_variable', float_prop_subsampling_for_clustering = 0.2, name_col_filter_subsampled_for_clustering = 'filter_hdbscan_subsampling_for_clustering', float_prop_subsampling_for_cluster_label_prediction = 0.2, flag_draw_graph = True, dict_kw_scatter = { 's' : 10, 'linewidth' : 0, 'alpha' : 0.05 }, flag_no_prediction = True ) :
        """ # 2022-07-21 10:34:43 
        Perform HDBSCAN with subsampling for a scalable analysis of single-cell data


        arguments:
        'name_model' : name of the model saved/will be saved in RamData.ns database. if the model already exists, 'flag_recluster' 'cut_distance', and 'min_cluster_size' arguments will become active.
        'min_cluster_size', 'min_samples' : arguments for HDBSCAN method. please refer to the documentation of HDBSCAN (https://hdbscan.readthedocs.io/)
        'prefix_name_col_umap' : a prefix for the 'name_col' of the columns containing UMAP transformed values.
        'int_num_components_umap' : number of output UMAP components to use for clustering (default: 2)
        'flag_reanalysis_of_previous_clustering_result' : if 'flag_reanalysis_of_previous_clustering_result' is True and 'name_model' exists in the RamData.ns database, use the hdbscan model saved in the database to re-analyze the previous hierarchical DBSCAN clustering result. 'cut_distance' and 'min_cluster_size' arguments can be used to re-analyze the clustering result and retrieve more fine-grained/coarse-grained cluster labels (for more info., please refer to hdbscan.HDBSCAN.single_linkage_tree_.get_clusters docstring). To perform hdbscan from the start, change name_model to a new name or delete the model from RamData.ns database
        'cut_distance' and 'min_cluster_size' : arguments for the re-analysis of the clustering result for retrieving more fine-grained/coarse-grained cluster labels (for more info., please refer to hdbscan.HDBSCAN.single_linkage_tree_.get_clusters docstring). 
        'name_col_hdbscan' : the name of the RamData.ft.meta ZarrDataFrame column to which estimated/predicted cluster labels will be written (if the column already exists, it will be (partially, depending on the filter) overwritten).

        'flag_use_pynndescent' : set this flag to True to use pynndescent for kNN search (k=1) for efficient cluster label identification
        'int_num_threads' : the number of threads for the cluster label prediction jobs
        'int_num_barcodes_in_cluster_label_prediction_batch' : when predicting the cluster labels using the clustering results from subsampled cells, using this number of barcodes for each batch
        'name_col_filter' : the name of 'feature'/'barcode' Axis metadata column to retrieve selection filter for running the current method. if None is given, current barcode/feature filters (if it has been set) will be used as-is.
        'float_prop_subsampling_for_clustering' : perform clustering using smaller number of barcodes than the actual dataset using subsampling. decreasing this value will reduce the memory consumption and comoputation time for HDBSCAN, but the clustering label qualities might decrease due to limited representations of barcodes in the dataset.
        'name_col_filter_subsampled_for_clustering' : the name of 'barcode' Axis metadata column to retrieve or save mask containing subsampled barcodes for clustering. if 'None' is given and 'float_prop_subsampling_for_clustering' is below 1 (i.e. subsampling will be used), the generated subsampling filter will not be saved.
        'float_prop_subsampling_for_cluster_label_prediction' : for cluster label prediction, further subsampling of barcodes that have been used for clustering will increase the prediction efficiency while decreasing the prediction accuracy minimally for most clusters (for clusters with small number/low density of barcodes, prediction accuracy might decrease significantly.)    

        'flag_draw_graph' : visualize clustering results
        'dict_kw_scatter' : arguments for 'matplotlib Axes.scatter' that will be used for plotting
        'flag_no_prediction' : skip prediction step (if subsampling is used) for manual iterative parameter optimization

        returns:
        embedded_for_training, arr_cluster_label, clusterer(hdbscan object)
        """
        """
        1) Prepare
        """
        # # retrieve 'Barcode' Axis object
        ax = self.bc

        # set filters for operation
        if name_col_filter is not None :
            self.change_filter( name_col_filter )

        # retrieve a flag indicating whether a subsampling is active
        float_prop_subsampling, name_col_filter_subsampled = float_prop_subsampling_for_clustering, name_col_filter_subsampled_for_clustering # rename values
        flag_is_subsampling_active = ( name_col_filter_subsampled in self.bc.meta ) or ( float_prop_subsampling is not None and float_prop_subsampling < 1 )

        # if a subsampling is active, retrieve a filter containing subsampled barcodes and apply the filter to the 'barcode' Axis
        if flag_is_subsampling_active :
            # retrieve barcode filter before subsampling (back-up)
            ba_filter_bc_before_subsampling = self.bc.filter

            # set barcode filter after subsampling
            if name_col_filter_subsampled in self.bc.meta : # if 'name_col_filter_subsampled' barcode filter is available, load the filter
                self.bc.change_filter( name_col_filter_subsampled )
            else : # if the 'name_col_filter_subsampled' barcode filter is not available, build a filter containing subsampled entries and save the filter
                self.bc.filter = self.bc.subsample( float_prop_subsampling = float_prop_subsampling ) 
                self.bc.save_filter( name_col_filter_subsampled )

        """
        2) Train Model with/without subsampling of barcodes, and retrieve cluster labels
        """
        # initialize
        embedded_for_training = self.get_umap( prefix_name_col = prefix_name_col_umap, int_num_components = int_num_components_umap ) # retrieve embedded barcodes (with/without subsampling)

        if name_model in self.ns : # if 'name_model' exists in the database, use the previously computed clustering results
            clusterer = self.ns[ name_model ] # retrieve previously saved model
            if flag_reanalysis_of_previous_clustering_result : # if 'flag_reanalysis_of_previous_clustering_result' is True, perform re-analysis of the clustering result
                arr_cluster_label = clusterer.single_linkage_tree_.get_clusters( cut_distance = cut_distance, min_cluster_size = min_cluster_size ) # re-analyze previous clustering result, and retrieve cluster labels
            else :
                arr_cluster_label = clusterer.labels_ # retrieve previously calculated cluster labels
        else :
            clusterer = hdbscan.HDBSCAN( min_cluster_size = min_cluster_size, min_samples = min_samples )
            clusterer.fit( embedded_for_training ) # clustering embedded barcodes
            arr_cluster_label = clusterer.labels_ # retrieve cluster labels
            # save trained model
            if name_model is not None : # check validity of 'name_model' 
                self.ns[ name_model ] = clusterer
        
        # report
        if self.verbose :
            print( 'clustering completed' )

        # if subsampling has been completed, revert to the original barcode selection filter (restore)
        if flag_is_subsampling_active :
            self.bc.filter = ba_filter_bc_before_subsampling
            del ba_filter_bc_before_subsampling

        if flag_draw_graph : # visualize clustering results :
            color_palette = sns.color_palette( 'Paired', len( set( arr_cluster_label ) ) )
            cluster_colors = [ color_palette[ x ] if x >= 0 else ( 0.5, 0.5, 0.5 ) for x in arr_cluster_label ]
            fig, plt_ax = plt.subplots( 1, 1, figsize = ( 7, 7 ) )
            plt_ax.scatter( * embedded_for_training.T, c = cluster_colors, ** dict_kw_scatter )

        if flag_no_prediction : # complete the method without cluster label prediction of remaining embedded barcodes if 'flag_no_prediction' is True
            return embedded_for_training, arr_cluster_label, clusterer # return the trained model and computed cluster labels

        """
        2) Transform Data (prediction of cluster labels)
        """
        if flag_is_subsampling_active : # when subsampling was used
            # subsample the training dataset if 'float_prop_subsampling_for_cluster_label_prediction' < 1
            if float_prop_subsampling_for_cluster_label_prediction < 1 : # if subsampling of traning data is active, subsamples training data for more efficient (but possibly less accurate) prediction of cluster labels
                arr_mask = np.random.random( len( arr_cluster_label ) ) < float_prop_subsampling_for_cluster_label_prediction # retrieve arr_mask for subsampling of data used for training
                embedded_for_training = embedded_for_training[ arr_mask ]
                arr_cluster_label = arr_cluster_label[ arr_mask ]
                del arr_mask

            if flag_use_pynndescent : # use pynndescent for fast approximate kNN search for k=1
                pass
#                                     index = NNDescent( data )
#                     index pynndescent.NNDescent(  )
#                     index = NNDescent(data)
#                     You can then use the index for searching (and can pickle it to disk if you wish). To search a pynndescent index for the 15 nearest neighbors of a test data set query_data you can do something like

#                     index.query(query_data, k=15)
                
            else : # if no other external algorithms are used, fallback to a jit-compiled kNN search algorithm
                # jit compile a function for finding cluster labels of nearest neighbors
                @jit( nopython = True )
                def find_cluster_labels_of_nearest_neighbors( embedded_for_training : np.ndarray, arr_cluster_label : np.ndarray, embedded_for_prediction : np.ndarray, arr_cluster_label_predicted : np.ndarray ) :
                    ''' # 2022-07-17 19:12:56 
                    find cluster labels of nearest neighbors
                    '''
                    for i in range( len( embedded_for_prediction ) ) : # for each embedded point
                        arr_cluster_label_predicted[ i ] = arr_cluster_label[ ( ( embedded_for_training - embedded_for_prediction[ i ] ) ** 2 ).sum( axis = 1 ).argmin( ) ] # identify cluster label of the nearest neighbor of the current embedded point
                    return arr_cluster_label_predicted


            # define functions for multiprocessing step
            def process_batch( batch, pipe_to_main_process ) :
                ''' # 2022-07-13 22:18:22 
                retrieve embedding of the barcodes of the current batch and predict cluster labels based on the inputs and outputs of the (subsampled) training data
                '''
                # parse the received batch
                int_num_of_previously_returned_entries, l_int_entry_current_batch = batch 

                embedded_for_prediction = self.get_umap( prefix_name_col = prefix_name_col_umap, int_num_components = int_num_components_umap, l_int_entries = l_int_entry_current_batch ) # retrieve umap embedding from ax.meta ZDF for the current batch
                arr_cluster_label_predicted = np.zeros( len( embedded_for_prediction ), dtype = int ) # initialize 'arr_cluster_label_predicted'
                arr_cluster_label_predicted = find_cluster_labels_of_nearest_neighbors( embedded_for_training, arr_cluster_label, embedded_for_prediction, arr_cluster_label_predicted ) # find cluster lables of embedded barcodes using nearest neighbors
                del embedded_for_prediction

                pipe_to_main_process.send( ( l_int_entry_current_batch, arr_cluster_label_predicted ) ) # send the integer representations of the barcodes for PCA value update
            def post_process_batch( res ) :
                """ # 2022-07-13 22:18:26 
                retrieve predicted cluster labels and write to the Axis.metadata of the 'barcode' axis
                """
                # parse result 
                l_int_entry_current_batch, arr_cluster_label_predicted = res

                # update cluster labels for the barcodes of the current batch
                ax.meta[ name_col_hdbscan, l_int_entry_current_batch ] = arr_cluster_label_predicted

            # transform values using iPCA using multiple processes
            bk.Multiprocessing_Batch( ax.batch_generator( int_num_entries_for_batch = int_num_barcodes_in_cluster_label_prediction_batch, flag_return_the_number_of_previously_returned_entries = True ), process_batch, post_process_batch = post_process_batch, int_num_threads = max( int_num_threads, 2 ), int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop = 0.2 ) # number of threads for multi-processing should be >1 # generate batch with fixed number of barcodes
        else : # if all barcodes were used for clustering, simply add retrieved cluster labels to the metadata
            ax.meta[ name_col_hdbscan ] = arr_cluster_label

        return embedded_for_training, arr_cluster_label, clusterer # return the trained model and computed cluster labels
    
    ''' satellite methods for analyzing RamData '''
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
    """ # 2022-07-19 13:24:22 
    draw resulting UMAP graph
    """
    arr_cluster_label = self.bc.meta[ 'hdbscan' ]
    embedded_for_training = self.get_umap( )
    
    # adjust graph settings based on the number of cells to plot
    int_num_bc = embedded_for_training.shape[ 0 ]
    size_factor = max( 1, np.log( int_num_bc ) - np.log( 5000 ) )
    dict_kw_scatter = { 's' : 20 / size_factor, 'linewidth' : 0, 'alpha' : 0.5 / size_factor }
    
    color_palette = sns.color_palette( 'Paired', len( set( arr_cluster_label ) ) )
    cluster_colors = [ color_palette[ x ] if x >= 0 else ( 0.5, 0.5, 0.5 ) for x in arr_cluster_label ]
    fig, plt_ax = plt.subplots( 1, 1, figsize = ( 7, 7 ) )
    plt_ax.scatter( * embedded_for_training.T, c = cluster_colors, ** dict_kw_scatter )
    MPL_basic_configuration( x_label = 'UMAP_1', y_label = 'UMAP_1', show_grid = False )
    MPL_SAVE( 'umap.hdbscan', folder = path_folder_graph )

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