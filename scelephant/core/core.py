from biobookshelf.main import *
from biobookshelf import *
import biobookshelf as bk
pd.options.mode.chained_assignment = None  # default='warn' # to disable worining
import zarr # SCElephant is currently implemented using Zarr
import numcodecs
import scanpy
# import shelve # for persistent dictionary (key-value based database)

# define version
_version_ = '0.0.0'
_scelephant_version_ = _version_
_last_modified_time_ = '2022-06-23 00:01:26 ' 

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
                if line[ 0 ] == '%' :
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
                if line[ 0 ] == '%' :
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
            if line[ 0 ] == '%' :
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
        if line[ 0 ] == '%' :
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
            if line[ 0 ] == '%' :
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
            if line[ 0 ] == '%' :
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
                if line[ 0 ] == '%' :
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
    ''' # 2022-02-22 01:39:45  hyunsu-an
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
    OS_FILE_Combine_Files_in_order( df_file.path.values, f"{path_folder_mtx_10x_output}matrix.mtx.gz", delete_input_files = not flag_split_mtx, header = f"%%MatrixMarket matrix coordinate integer general\n%\n{len( dict_id_row_previous_to_id_row_current )} {len( dict_id_column_previous_to_id_column_current )} {int_total_n_entries}\n" ) # combine the output mtx files in the order # does not delete temporary files if 'flag_split_mtx' is True
    
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

''' a class for Zarr-based DataFrame object '''
class ZarrDataFrame( ) :
    """ # 2022-06-22 23:56:01 
    on-demend persistant DataFrame backed by Zarr persistent arrays.
    each column can be separately loaded, updated, and unloaded.
    a filter can be set, which allows updating and reading ZarrDataFrame as if it only contains the rows indicated by the given filter.
    the one of the functionality of this class is to provide a Zarr-based dataframe object that is compatible with Zarr.js (javascript implementation of Zarr), with a categorical data type (the format used in zarr is currently not supported in zarr.js) compatible with zarr.js.
    
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
    """
    def __init__( self, path_folder_zdf, df = None, int_num_rows = None, int_num_rows_in_a_chunk = 10000, ba_filter = None, flag_enforce_name_col_with_only_valid_characters = False, flag_store_string_as_categorical = True, flag_retrieve_categorical_data_as_integers = False, flag_load_data_after_adding_new_column = True ) :
        """ # 2022-06-22 16:31:07 
        """
        # handle path
        path_folder_zdf = os.path.abspath( path_folder_zdf ) # retrieve absolute path
        if path_folder_zdf[ -1 ] != '/' : # add '/' to the end of path to mark that this is a folder directory
            path_folder_zdf += '/'
        self._path_folder_zdf = path_folder_zdf
        self._flag_retrieve_categorical_data_as_integers = flag_retrieve_categorical_data_as_integers
        self._flag_load_data_after_adding_new_column = flag_load_data_after_adding_new_column
        self.filter = ba_filter
            
        # open or initialize zdf and retrieve associated metadata
        if not os.path.exists( path_folder_zdf ) : # if the object does not exist, initialize ZarrDataFrame
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
           
        # handle input arguments
        self._str_invalid_char = '! @#$%^&*()-=+`~:;[]{}\|,<.>/?' + '"' + "'" if self._dict_zdf_metadata[ 'flag_enforce_name_col_with_only_valid_characters' ] else '/' # linux file system does not allow the use of linux'/' character in the folder/file name
        
        # initialize loaded data
        self._loaded_data = dict( )
        
        # initialize temp folder
        self._initialize_temp_folder_( )
        
        if isinstance( df, pd.DataFrame ) : # if a valid pandas.dataframe has been given
            # update zdf with the given dataframe
            self.update( df )
    def _initialize_temp_folder_( self ) :
        """ # 2022-06-22 21:49:38
        empty the temp folder
        """
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
        return self._za_filter
    @filter.setter
    def filter( self, ba_filter ) :
        """ # 2022-06-22 21:52:00 
        change filter, and empty the cache
        """
        if ba_filter is None : # if filter is removed, 
            self._za_filter = None
            self._n_rows_after_applying_filter = None
        else :
            # check whether the given filter is bitarray
            assert isinstance( ba_filter, bitarray )
            
            # check the length of filter bitarray
            if 'int_num_rows' not in self._dict_zdf_metadata : # if 'int_num_rows' has not been set, set 'int_num_rows' using the length of the filter bitarray
                self._dict_zdf_metadata[ 'int_num_rows' ] = len( ba_filter )
                self._save_metadata_( ) # save metadata
            else :
                # check the length of filter bitarray
                assert len( ba_filter ) == self._dict_zdf_metadata[ 'int_num_rows' ]

            self._loaded_data = dict( ) # empty the cache
            self._initialize_temp_folder_( ) # empty the temp folder
            
            # compose a list of integer indices of active rows after applying filter
            l = [ ]
            for st, en in bk.BA.Find_Segment( ba_filter, background = 0 ) : # retrieve active segments from bitarray filter 
                l.extend( np.arange( st, en ) ) # retrieve integer indices of the active rows
            self._n_rows_after_applying_filter = len( l ) # retrieve the number of rows after applying filter (which is equal to the number of integer indices representing the filter)

            path_folder_filter = f"{self._path_folder_temp}filter.zarr/" # define a Zarr object for caching filter data
            za = zarr.open( path_folder_filter, 'w', shape = ( self._n_rows_after_applying_filter, ), chunks = ( self._dict_zdf_metadata[ 'int_num_rows_in_a_chunk' ], ), dtype = np.int64, synchronizer = zarr.ThreadSynchronizer( ) ) # write 'integer filter' (list of integer indices of active rows) as a Zarr object
            za[ : ] = np.array( l, dtype = np.int64 ) # write the integer filter array to the output Zarr object
            self._za_filter = zarr.open( path_folder_filter, mode = 'r' ) # open in read-only mode, and set '_za_filter' attribute
    def __getitem__( self, args ) :
        ''' # 2022-06-22 22:41:25 
        retrieve data of a column.
        partial read is allowed through indexing
        when a filter is active, the filtered data will be cached in the temporary directory as a Zarr object and will be retrieved in subsequent accesses
        '''
        # parse arguments
        if isinstance( args, tuple ) :
            name_col, coords = args
        else :
            name_col, coords = args, slice( None, None, None ) # retrieve all data if only 'name_col' is given
            
        if name_col in self : # if name_col is valid
            if name_col in self._loaded_data : # if a loaded data is available
                return self._loaded_data[ name_col ][ coords ]
            else :
                if self.filter is None : # if filter is not active
                    za = zarr.open( f"{self._path_folder_zdf}{name_col}/", mode = 'r' ) # read data from the Zarr object
                else : # if filter is active
                    path_folder_temp_zarr = f"{self._path_folder_temp}{name_col}/" # retrieve path of cached zarr object containing filtered data
                    if os.path.exists( path_folder_temp_zarr ) : # if a cache is available
                        za = zarr.open( path_folder_temp_zarr, mode = 'r' ) # open the cached Zarr object containing
                    else : # if a cache is not available, retrieve filtered data and write a cache
                        za = zarr.open( f"{self._path_folder_zdf}{name_col}/", mode = 'r' ) # read data from the Zarr object
                        za_cached = zarr.open( path_folder_temp_zarr, 'w', shape = ( self._n_rows_after_applying_filter, ), chunks = ( self._dict_zdf_metadata[ 'int_num_rows_in_a_chunk' ], ), dtype = za.dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # open a new Zarr object for caching
                        za = za[ self.filter[ : ] ] # retrieve filtered data 
                        za_cached[ : ] = za # save the filtered data for caching
                values = za[ coords ] # retrieve data
                
                # check whether the current column contains categorical data
                l_value_unique = self.get_categories( name_col )
                if len( l_value_unique ) == 0 or self._flag_retrieve_categorical_data_as_integers : # handle non-categorical data
                    return values
                else : # handle categorical data
                    values_decoded = np.zeros( len( values ), dtype = object ) # initialize decoded values
                    for i in range( len( values ) ) :
                        values_decoded[ i ] = l_value_unique[ values[ i ] ] if values[ i ] >= 0 else np.nan # convert integer representations to its original string values # -1 (negative integers) encodes np.nan
                    return values_decoded
    def __setitem__( self, name_col, values ) :
        ''' # 2022-06-22 23:55:53 
        save a column. partial update is not allowed, meaning that an entire column should be updated.
        when a filter is active, write filtered data
        '''
        # check whether the given name_col contains invalid characters(s)
        for char_invalid in self._str_invalid_char :
            if char_invalid in name_col :
                raise TypeError( f"the character '{char_invalid}' cannot be used in 'name_col'. Also, the 'name_col' cannot contains the following characters: {self._str_invalid_char}" )
        
        # retrieve data values from the 'values' 
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
            assert int_num_values == self.n_rows
        else :
            self._dict_zdf_metadata[ 'int_num_rows' ] = int_num_values # record the number of rows of the dataframe
            self._save_metadata_( ) # save metadata
        
        # compose metadata of the column
        dict_col_metadata = { 'flag_categorical' : False } # set a default value for 'flag_categorical' metadata attribute
        dict_col_metadata[ 'flag_filtered' ] = self.filter is not None # mark the column containing filtered data
        
        # write data
        if dtype is str and self._dict_zdf_metadata[ 'flag_store_string_as_categorical' ] : # storing categorical data            
            dict_col_metadata[ 'flag_categorical' ] = True # update metadata for categorical datatype
            
            set_value_unique = set( values ) # retrieve a set of unique values
            # handle when np.nan value exist 
            if np.nan in set_value_unique : 
                dict_col_metadata[ 'flag_contains_nan' ] = True # mark that the column contains np.nan values
                set_value_unique.remove( np.nan ) # removes np.nan from the category
            
            l_value_unique = list( set_value_unique ) # retrieve a list of unique values # can contain mixed types (int, float, str)
            dict_col_metadata[ 'l_value_unique' ] = list( str( e ) for e in set_value_unique ) # update metadata # convert entries to string (so that all values with mixed types can be interpreted as strings)
            
            # retrieve appropriate datatype for encoding unique categorical values
            int_min_number_of_bits = int( np.ceil( math.log2( len( l_value_unique ) ) ) ) + 1 # since signed int will be used, an additional bit is required to encode the data
            if int_min_number_of_bits <= 8 :
                dtype = np.int8
            elif int_min_number_of_bits <= 16 :
                dtype = np.int16
            else :
                dtype = np.int32
            
            # open Zarr object representing the current column
            za = zarr.open( f"{self._path_folder_zdf}{name_col}/", mode = 'w', shape = ( self._n_rows_unfiltered, ), chunks = ( self._dict_zdf_metadata[ 'int_num_rows_in_a_chunk' ], ), dtype = dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # overwrite mode (purge previously written data) # saved data will have 'self._n_rows_unfiltered' number of items 
            
            # encode data
            dict_encode_category = dict( ( e, i ) for i, e in enumerate( l_value_unique ) ) # retrieve a dictionary encoding value to integer representation of the value
            values_encoded = np.array( list( dict_encode_category[ value ] if value in dict_encode_category else -1 for value in values ), dtype = dtype ) # retrieve encoded values # np.nan will be encoded as -1 values
            if self._flag_retrieve_categorical_data_as_integers : # if 'self._flag_retrieve_categorical_data_as_integers' is True, use integer representation of values for caching
                values = values_encoded
            
            # write data 
            if self.filter is None : # when filter is not set
                za[ : ] = values_encoded # write encoded data
            else : # when filter is present
                za[ self.filter[ : ] ] = values_encoded # save filtered data 
                
        else : # storing non-categorical data
            za = zarr.open( f"{self._path_folder_zdf}{name_col}/", mode = 'w', shape = ( self._n_rows_unfiltered, ), chunks = ( self._dict_zdf_metadata[ 'int_num_rows_in_a_chunk' ], ), dtype = dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # overwrite mode (purge previously written data) # saved data will have 'self._n_rows_unfiltered' number of items 
            
            # write data
            if self.filter is None : # when filter is not set
                za[ : ] = values # write data
            else : # when filter is present
                za[ self.filter[ : ] ] = values # save filtered data 
            
        # save column metadata
        za.attrs[ 'dict_col_metadata' ] = dict_col_metadata
        
        # update zdf metadata
        if name_col not in self._dict_zdf_metadata[ 'columns' ] :
            self._dict_zdf_metadata[ 'columns' ].add( name_col )
            self._save_metadata_( )
        
        # add data to the loaded data dictionary (object cache) if 'self._flag_load_data_after_adding_new_column' is True
        if self._flag_load_data_after_adding_new_column :
            self._loaded_data[ name_col ] = values
    def __delitem__( self, name_col ) :
        ''' # 2022-06-20 21:57:38 
        remove the column from the memory and the object on disk
        '''
        if name_col in self : # if the given name_col is valid
            # remove column from the memory
            self.unload( name_col ) 
            # remove the column from metadata
            self._dict_zdf_metadata[ 'columns' ].remove( name_col )
            self._save_metadata_( ) # update metadata
            # delete the column from the disk ZarrDataFrame object
            OS_Run( [ 'rm', '-rf', f"{self._path_folder_zdf}{name_col}/" ] )
    def __contains__( self, name_col ) :
        return name_col in self._dict_zdf_metadata[ 'columns' ]
    def __iter__( self ) :
        return iter( self._dict_zdf_metadata[ 'columns' ] )
    def __repr__( self ) :
        return f"<ZarrDataFrame object stored at {self._path_folder_zdf}\n\twith the following metadata: {self._dict_zdf_metadata}>"
    @property
    def columns( self ) :
        ''' # 2022-06-21 15:58:41 
        return available column names
        '''
        return self._dict_zdf_metadata[ 'columns' ]
    @property
    def df( self ) :
        ''' # 2022-06-21 16:00:11 
        return loaded data as a dataframe
        '''
        return pd.DataFrame( self._loaded_data )
    def _save_metadata_( self ) :
        ''' # 2022-06-20 21:44:42 
        save metadata of the current ZarrDataFrame
        '''
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
    def update( self, df ) :
        """ # 2022-06-20 21:36:55 
        update ZarrDataFrame with the given 'df'
        """
        # update each column
        for name_col in df.columns.values :
            self[ name_col ] = df[ name_col ]
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
        remove the column from the memory
        """
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

''' a class for containing disk-backed AnnData objects '''
class AnnDataContainer( ) :
    """ # 2022-06-09 18:35:04 
    AnnDataContainer
    Also contains utility functions for handling multiple AnnData objects on the disk sharing the same list of cells
    
    this object will contain AnnData objects and their file paths on the disk, and provide a convenient interface of accessing the items.
    
    'flag_enforce_name_adata_with_only_valid_characters' : (Default : True). does not allow the use of 'name_adata' containing the following characters { ' ', '/', '-', '"', "'" ";", and other special characters... }
    'path_prefix_default' : a default path of AnnData on disk. f'{path_prefix_default}{name_adata}.h5ad' will be used.
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
    def __init__( self, flag_enforce_name_adata_with_only_valid_characters = True, path_prefix_default = None, ** args ) :
        self.__str_invalid_char = '! @#$%^&*()-=+`~:;[]{}\|,<.>/?' + '"' + "'" if flag_enforce_name_adata_with_only_valid_characters else ''
        self.path_prefix_default = path_prefix_default
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
            
''' methods for creating RAMtx objects '''
def __Merge_Sort_MTX_10X_and_Write_and_Index_Zarr__( za_mtx, za_mtx_index, * l_path_file_input, flag_ramtx_sorted_by_id_feature = True, flag_delete_input_file_upon_completion = False, dtype = np.int64, int_size_buffer_for_mtx_index = 1000 ) :
    """ # 2022-06-21 16:26:00 
    merge sort mtx files into a single mtx uncompressed file and index entries in the combined mtx file while writing the file
    
    
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
        ''' parse and decorate mtx record for sorting '''
        while True :
            line = file.readline( )
            if len( line ) == 0 :
                break
            ''' parse a mtx record '''
            line_decoded = line.decode( ) if flag_input_binary else line
            index_row, index_column, float_value = ( line_decoded ).strip( ).split( ) # parse a record of a matrix-market format file
            index_row, index_column, float_value = int( index_row ) - 1, int( index_column ) - 1, float( float_value ) # 1-based > 0-based coordinates
            yield index_row if flag_ramtx_sorted_by_id_feature else index_column, ( index_row, index_column, float_value ) # return 0-based coordinates

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
            za_mtx_index[ int_num_mtx_index_records_written : int_num_mtx_index_records_written + int_num_newly_added_mtx_index_records ] = np.array( l_mtx_index, dtype = dtype ) # add data to the zarr data sink
            int_num_mtx_index_records_written += int_num_newly_added_mtx_index_records # update 'int_num_mtx_index_records_written'
        return int_num_mtx_index_records_written
    def flush_matrix( int_num_mtx_records_written ) : 
        ''' # 2022-06-21 16:26:09 
        flush a block of matrix data of a single entry (of the axis used for sorting) to Zarr and index the block, and update 'int_num_mtx_records_written' '''
        int_num_newly_added_mtx_records = len( l_mtx_record )
        if int_num_newly_added_mtx_records > 0 : # if there is valid record to be flushed
            # add records to mtx_index
            l_mtx_index.append( [ int_entry_currently_being_written, int_num_mtx_records_written, int_num_mtx_records_written + int_num_newly_added_mtx_records ] ) # collect information required for indexing
            for int_entry in range( int_entry_currently_being_written + 1, int_entry_of_the_current_record ) : # for the int_entry that lack count data and does not have indexing data, put place holder values
                l_mtx_index.append( [ int_entry, -1, -1 ] ) # put place holder values for int_entry lacking count data.
            
            za_mtx[ int_num_mtx_records_written : int_num_mtx_records_written + int_num_newly_added_mtx_records ] = np.array( l_mtx_record, dtype = dtype ) # add data to the zarr data sink
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
def Convert_MTX_10X_to_RAMtx( path_folder_mtx_10x_input, path_folder_ramtx_output, flag_ramtx_sorted_by_id_feature = True, int_num_threads = 15, int_max_num_entries_for_chunk = 10000000, int_max_num_files_for_each_merge_sort = 5, dtype = np.float64, int_num_of_records_in_a_chunk_zarr_matrix = 10000, int_num_of_entries_in_a_chunk_zarr_matrix_index = 1000, verbose = False, flag_debugging = False ) :
    ''' # 2022-06-21 12:44:42 
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
    'dtype' (default: np.float64), dtype of the output zarr array
    'flag_debugging' : if True, does not delete temporary files
    'int_num_of_records_in_a_chunk_zarr_matrix' : chunk size for output zarr objects
    'int_num_of_entries_in_a_chunk_zarr_matrix_index' : chunk size for output zarr objects
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
            ''' # 2022-06-10 21:15:51 
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
                        df.columns = [ 'index_row', 'index_col', 'float_value' ]
                        for name_col, name_file in zip( [ 'index_row', 'index_col' ], [ 'features.tsv.gz', 'barcodes.tsv.gz' ] ) :
                            df[ name_col ] = df[ name_col ].apply( dict_name_file_to_mo_from_index_entry_to_index_entry_new_after_sorting[ name_file ] ) # retrieve ranks of the entries, or new indices after sorting
                        df.sort_values( 'index_row' if flag_ramtx_sorted_by_id_feature else 'index_col', inplace = True ) # sort by row if the matrix is sorted by features and sort by column if the matrix is sorted by barcodes
                        df.to_csv( f"{path_temp}0.{UUID( )}.mtx.gz", sep = ' ', header = None, index = False ) # save the sorted mtx records as a file
                    except :
                        print( byte_content.decode( ).split( '\n', 1 )[ 0 ] )
                        break
                pipe_to_mtx_record_parser.send( 'flush completed' ) # send signal that flushing the received data has been completed, and now ready to export matrix again

        int_n_workers_for_sorting = min( 3, max( int_num_threads - 1, 1 ) ) # one thread for distributing records. Minimum numbers of workers for sorting is 1 # the number of worker for sorting should not exceed 3
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
        za_mtx = zarr.open( f'{path_folder_ramtx_output}matrix.zarr', mode = 'w', shape = ( int_num_records, 3 ), chunks = ( int_num_of_records_in_a_chunk_zarr_matrix, 3 ), dtype = dtype )
        za_mtx_index = zarr.open( f'{path_folder_ramtx_output}matrix.index.zarr', mode = 'w', shape = ( int_num_features if flag_ramtx_sorted_by_id_feature else int_num_barcodes, 3 ), chunks = ( int_num_of_entries_in_a_chunk_zarr_matrix_index, 3 ), dtype = np.float64 ) # max number of matrix index entries is 'int_num_records' of the input matrix. this will be resized # dtype of index will be np.float64, since javascript number primitives are actually float64

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
def Convert_MTX_10X_to_RamData( path_folder_mtx_10x_input, path_folder_ramdata_output, name_layer = 'raw', int_num_threads = 15, int_max_num_entries_for_chunk = 10000000, int_max_num_files_for_each_merge_sort = 5, dtype = np.float64, int_num_of_records_in_a_chunk_zarr_matrix = 10000, int_num_of_entries_in_a_chunk_zarr_matrix_index = 1000, flag_simultaneous_indexing_of_cell_and_barcode = True, verbose = False, flag_debugging = False ) :
    """ # 2022-06-22 00:11:46 
    convert 10X count matrix data to the two RAMtx object, one sorted by features and the other sorted by barcodes, and construct a RamData data object on disk, backed by Zarr persistant arrays

    inputs:
    ========
    'path_folder_mtx_10x_input' : an input directory of 10X matrix data
    'path_folder_ramdata_output' an output directory of RamData
    'name_layer' : a name of the given data layer
    'int_num_threads' : the number of threads for multiprocessing
    'dtype' (default: np.uint32), 'dtype_of_value' (default: None (auto-detect)) : numpy dtypes for pickle and feather outputs. choosing smaller format will decrease the size of the object on disk
    'flag_simultaneous_indexing_of_cell_and_barcode' : if True, create cell-sorted RAMtx and feature-sorted RAMtx simultaneously using two worker processes with the half of given 'int_num_threads'. it is generally recommended to turn this feature on, since the last step of the merge-sort is always single-threaded.
    """
    # build barcode- and feature-sorted RAMtx objects
    path_folder_data = f"{path_folder_ramdata_output}{name_layer}/" # define directory of the output data
    if flag_simultaneous_indexing_of_cell_and_barcode :
        l_process = list( mp.Process( target = Convert_MTX_10X_to_RAMtx, args = ( path_folder_mtx_10x_input, path_folder_ramtx_output, flag_ramtx_sorted_by_id_feature, int_num_threads_for_the_current_process, int_max_num_entries_for_chunk, int_max_num_files_for_each_merge_sort, dtype, int_num_of_records_in_a_chunk_zarr_matrix, int_num_of_entries_in_a_chunk_zarr_matrix_index, verbose, flag_debugging ) ) for path_folder_ramtx_output, flag_ramtx_sorted_by_id_feature, int_num_threads_for_the_current_process in zip( [ f"{path_folder_data}sorted_by_barcode/", f"{path_folder_data}sorted_by_feature/" ], [ False, True ], [ int( np.floor( int_num_threads / 2 ) ), int( np.ceil( int_num_threads / 2 ) ) ] ) )
        for p in l_process : p.start( )
        for p in l_process : p.join( )
    else :
        Convert_MTX_10X_to_RAMtx( path_folder_mtx_10x_input, path_folder_ramtx_output = f"{path_folder_data}sorted_by_barcode/", flag_ramtx_sorted_by_id_feature = False, int_num_threads = int_num_threads, int_max_num_entries_for_chunk = int_max_num_entries_for_chunk, int_max_num_files_for_each_merge_sort = int_max_num_files_for_each_merge_sort, dtype = dtype, int_num_of_records_in_a_chunk_zarr_matrix = int_num_of_records_in_a_chunk_zarr_matrix, int_num_of_entries_in_a_chunk_zarr_matrix_index = int_num_of_entries_in_a_chunk_zarr_matrix_index, verbose = verbose, flag_debugging = flag_debugging )
        Convert_MTX_10X_to_RAMtx( path_folder_mtx_10x_input, path_folder_ramtx_output = f"{path_folder_data}sorted_by_feature/", flag_ramtx_sorted_by_id_feature = True, int_num_threads = int_num_threads, int_max_num_entries_for_chunk = int_max_num_entries_for_chunk, int_max_num_files_for_each_merge_sort = int_max_num_files_for_each_merge_sort, dtype = dtype, int_num_of_records_in_a_chunk_zarr_matrix = int_num_of_records_in_a_chunk_zarr_matrix, int_num_of_entries_in_a_chunk_zarr_matrix_index = int_num_of_entries_in_a_chunk_zarr_matrix_index, verbose = verbose, flag_debugging = flag_debugging )

    # copy features/barcode.tsv.gz random access files for the web (stacked base64 encoded tsv.gz files)
    # copy features/barcode string representation zarr objects
    # copy features/barcode ZarrDataFrame containing number/categorical data
    for name_axis_singular in [ 'feature', 'barcode' ] :
        for str_suffix in [ 's.tsv.gz.base64.concatanated.txt', 's.index.tsv.gz.base64.txt', 's.str.zarr', 's.num_and_cat.zdf' ] :
            OS_Run( [ 'cp', '-r', f"{path_folder_data}sorted_by_{name_axis_singular}/{name_axis_singular}{str_suffix}", f"{path_folder_ramdata_output}{name_axis_singular}{str_suffix}" ] )
            
    # write metadata 
    int_num_features, int_num_barcodes, int_num_records = MTX_10X_Retrieve_number_of_rows_columns_and_records( path_folder_mtx_10x_input ) # retrieve metadata of the input 10X mtx
    root = zarr.group( path_folder_ramdata_output )
    root.attrs[ 'dict_ramdata_metadata' ] = { 
        'path_folder_mtx_10x_input' : path_folder_mtx_10x_input,
        'str_completed_time' : TIME_GET_timestamp( True ),
        'int_num_features' : int_num_features,
        'int_num_barcodes' : int_num_barcodes,
        'int_num_records' : int_num_records,
        'int_num_of_records_in_a_chunk_zarr_matrix' : int_num_of_records_in_a_chunk_zarr_matrix,
        'int_num_of_entries_in_a_chunk_zarr_matrix_index' : int_num_of_entries_in_a_chunk_zarr_matrix_index,
        'layers' : [ name_layer ],
        'version' : _version_,
    }
            
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

''' define classes and associated methods of RamData '''
class RAMtx( ) :
    """ # 2022-06-01 15:35:20 
    This class represent a random read-access mtx format for memory-efficient exploration single-cell transcriptomics/genomics data.
    This class use a count matrix data stored in a random read-access compatible format, called RAMtx, enabling exploration of a count matrix with tens of million cells with tens of millions of features.
    Also, the RAMtx format is optimized for multi-processing, and can be read by multiple processes simultaneously.
    Therefore, for exploration of count matrix produced from scarab short_read analyzer, which produces several millions of features extracted from both coding and non coding regions, this class provides fast front-end application for exploration of exhaustive data generated from scarab 'short_read'
    
    arguments:
    'int_num_threads_for_ramdom_accessing' : the number of threads that can be used for random access of the data
    'int_num_entries_to_randomly_access_during_warm_up_upon_initializing' : default: 1000 entries. randomly access certain number of entries using multiple threads upon initializing. if 0 or negative integer is given, skip the warm-up step.
    'file_format' : one of [ 
                                'mtx_gzipped' : 10X matrix format. (pros) the RAMtx can be read by other program that can read 10X matrix file, small disk size (cons) very slow write speed, slow read speed
                 (Default)  --> 'pickle' : uncompressed python pickle format. (pros) very fast write speed, very fast read speed. (cons) 5~10 times larger disk usage, python-specific data format
                                'pickle_gzipped' : gzipped python pickle format. (pros) fast read speed. disk usage is 20~50% smaller than 10X matrix file. the most efficient storage format. (cons) very slow write speed, python-specific data format
                                'feather' : uncompressed Apache Arrow feather storage format for DataFrames. (pros) very fast write speed, fast read speed, language-agnostic (R, Python, Julia, JS, etc.). (cons) ~2 times larger disk usage
                                'feather_lz4' : LZ4 compressed (a default compression of 'feather') Apache Arrow feather storage format for DataFrames. (pros) very fast write speed, fast read speed, language-agnostic (R, Python, Julia, JS, etc.). (cons) ~2 times larger disk usage.
                            ]
    'flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only' : (Default: False) if True, __getitem__ call will return three arrays containing int_barcode, int_feature, and data values, respectively. if False, an RAMtx object will return an AnnData object with features and barcodes information. This will require a RAMtx dataset to load both lists of features and barcode informations on memory, which will use an additional space of memory. If the lists of features and barcodes are already loaded, setting this flag to 'True' will further reduce the memory footprint of the RAMtx object. Also, if 'False' a string representing entries (either barcodes or features, depending on which entry is used to index the RAMtx object) can be used to access items in the RAMtx object though the __getitem__ function or using a large bracket [ ]. However, this will require loading an entire list of the string representations of the indexed entries, which takes several GB of memorys for very large datasets with dozens of million features/barcodes. Therefore, if the list of string representations of the indexed entries are already loaded or not required to utilize the item returned by a RAMtx object, setting this flag to 'True' allows each item can be access using an integer index of the indexed entry, and further reduce the memory footprint of the RAMtx object.
    'dtype_of_row_and_col_indices' (default: np.uint32), 'dtype_of_value' (default: None (auto-detect)) : numpy dtypes for pickle and feather outputs. choosing smaller format will decrease the size of the object on disk.
    """
    def __init__( self, path_folder_mtx, path_folder_mtx_new = None, flag_ramtx_sorted_by_id_feature = True, file_format = 'feather_lz4', flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only = False, int_num_threads_for_ramdom_accessing = 1, int_num_entries_to_randomly_access_during_warm_up_upon_initializing = 1000, int_num_cpus = 15, int_max_num_entries_for_chunk = 10000000, verbose = False, flag_debugging = False, int_max_num_files_for_each_merge_sort = 5, dtype_of_row_and_col_indices = None, dtype_of_value = None, flag_overwrite_existing_files = False, flag_sort_and_index_again = False ) :
        # handle input
        if path_folder_mtx_new is None :
            path_folder_mtx_new = path_folder_mtx
        # convert the input matrix into RAMtx if it is not the RAMtx object
        dict_metadata = Convert_MTX_10X_to_RAMtx( path_folder_mtx, path_folder_mtx_new, flag_ramtx_sorted_by_id_feature = flag_ramtx_sorted_by_id_feature, int_num_threads = int_num_cpus, int_max_num_entries_for_chunk = int_max_num_entries_for_chunk, verbose = verbose, int_max_num_files_for_each_merge_sort = int_max_num_files_for_each_merge_sort, flag_overwrite_existing_files = flag_overwrite_existing_files, flag_sort_and_index_again = flag_sort_and_index_again ) # sort the input matrix file
        self._dict_metadata = dict_metadata # set the metadata of the sort, index and export settings
        # parse the metadata of the RAMtx object
        self.flag_ramtx_sorted_by_id_feature = self._dict_metadata[ 'flag_ramtx_sorted_by_id_feature' ]
        self._int_num_features, self._int_num_barcodes, self._int_num_records = self._dict_metadata[ 'int_num_features' ], self._dict_metadata[ 'int_num_barcodes' ], self._dict_metadata[ 'int_num_records' ]
        self.file_format_current = file_format
        self._l_file_format = self._dict_metadata[ 'file_format' ] # retrieve the list of available file_format of the current RAMtx object
        if self.file_format_current not in self._l_file_format : # if the given file_format is invalid, raise an error
            if len( self._l_file_format ) > 0 :
                self.file_format_current = self._l_file_format[ 0 ] # use the first file_format 
            else :
                raise FileNotFoundError( f"the given RAMtx format {self.file_format_current} is unavailable in the current RAMtx object" )
        
        ''' set the datatype of stored values based on RAMtx setting for 'mtx_gzipped' format '''
        if self.file_format_current == 'mtx_gzipped' :
            dtype_of_value = np.float64 if 'flag_output_value_is_float' in self._dict_metadata else np.int64
        
        ''' retrieve RAMtx format-specific import settings '''
        self._str_ext, self._str_ext_index, self._func_processed_bytes_to_arrays_mtx = _get_func_processed_bytes_to_arrays_mtx_and_other_settings_based_on_file_format( self.file_format_current, dtype_of_row_and_col_indices, dtype_of_value )

        # set attributes and settings
        self._dtype_of_row_and_col_indices = dtype_of_row_and_col_indices
        self.path_folder_mtx = path_folder_mtx_new
        self.verbose = verbose 
        self.flag_debugging = flag_debugging
        self.int_num_cpus = int_num_cpus
        self.int_num_threads_for_ramdom_accessing = int_num_threads_for_ramdom_accessing
        self.__flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only = flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only # this setting cannot be changed once RAMtx data object has been initialized
        # read index
        self._df_index = pd.read_csv( f'{self.path_folder_mtx}matrix.{self._str_ext}.{self._str_ext_index}', sep = '\t' )
        self._df_index[ 'int_num_bytes' ] = self._df_index.int_pos_end - self._df_index.int_pos_start # calculate the number of bytes of each chunk
        self._arr_data_df_index = self._df_index[ [ 'int_pos_start', 'int_pos_end' ] ].values # retrieve the index data as an array
        self._int_num_entry_indexed = len( self._df_index ) # retrieve the number of indexed entries
        # if 'flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only' is False, read barcodes and features, and construct an AnnData object with an empty count data
        if not flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only :
            # read feature
            df_feature = pd.read_csv( f'{self.path_folder_mtx}features.tsv.gz', sep = '\t', header = None )
            df_feature.columns = [ 'id_feature', 'feature', 'feature_type' ] + list( df_feature.columns.values[ 3 : ] )
            df_feature.set_index( 'feature', inplace = True ) # assumes 'feature' is unique
            # read barcode
            df_barcode = pd.read_csv( f'{self.path_folder_mtx}barcodes.tsv.gz', sep = '\t', header = None )
            df_barcode.columns = [ 'barcode' ] + list( df_barcode.columns.values[ 1 : ] )
            df_barcode.set_index( 'barcode', inplace = True )
            # initialize the anndata that will contain fetched data from RAMtx data structure
            self.adata = anndata.AnnData( obs = df_barcode, var = df_feature )
            
            # set the index of the df_index using the string representation of the indexed entries
            arr_entry = ( df_feature if self.flag_ramtx_sorted_by_id_feature else df_barcode ).index.values
            self._df_index.index = list( arr_entry[ i ] for i in ( self._df_index.index_entry.values - 1 ) ) # map string representation of indexed entries to the integer index of the entry in the list of entries(barcodes/features) # 'index_entry' 1>0 coordinate # handle the case where some entries are missing in the index (the data of the entries does not exist in the matrix)
            
            del df_feature, df_barcode, arr_entry # remove temporary objects
        else :
            self._df_index.index = list( i for i in ( self._df_index.index_entry.values - 1 ) ) # use the integer representation of the indexed entries # 'index_entry' 1>0 coordinate # handle the case where some entries are missing in the index (the data of the entries does not exist in the matrix)
            self.adata = None
        # build mapping between entry and index_entry
        self._dict_entry_to_index_entry = dict( ( e, i ) for i, e in enumerate( self._df_index.index.values ) )

        # if 'flag_warm_up_upon_initializing' is True, warm up upon initialization
        if int_num_entries_to_randomly_access_during_warm_up_upon_initializing > 0 :
            self._warm_up_( int_num_entries_to_randomly_access = int( int_num_entries_to_randomly_access_during_warm_up_upon_initializing ) )
    def _Retrieve_data_from_ramtx_( self, l_entry ) :
        """ # 2022-05-22 10:57:57 
        Retrieve data from RAMtx as bytes
        index should be an a string or an iterable
        
        Returns:
        bytes of the data belongining to 'l_entry'
        """
        # handle the case when only a single entry was submitted
        if isinstance( l_entry, ( str, int, np.int64 ) ) :
            l_entry = [ l_entry ]
        # retrieve index of the given entries (ignore entries that are note present in the mapping)
        l_index_entry = list( self._dict_entry_to_index_entry[ entry ] for entry in l_entry if entry in self._dict_entry_to_index_entry ) # map a string representation/integer index of indexed entries to the integer index of the entry in the array
        # sort indices of entries so that the data access can occur in the same direction across threads
        int_num_entries = len( l_index_entry )
        if int_num_entries > 30 : # use numpy sort function only when there are sufficiently large number of indices of entries to be sorted
            l_index_entry = np.sort( l_index_entry )
        else :
            l_index_entry = sorted( l_index_entry )
        
        """
        retrieve byte content from RAMtx data structure
        """
        def __retrieve_data_from_ramtx_as_a_worker__( pipe_from_main_thread = None, pipe_to_main_thread = None, flag_as_a_worker = True ) :
            with open( f'{self.path_folder_mtx}matrix.{self._str_ext}', 'rb+' ) as file : # open indexed matrix file
                l_index_entry = pipe_from_main_thread.recv( ) if flag_as_a_worker else pipe_from_main_thread  # receive work if 'flag_as_a_worker' is True or use 'pipe_from_main_thread' as a list of works
                # for each index_entry, retrieve data and collect records
                l_arr_int_feature, l_arr_int_barcode, l_arr_value = [ ], [ ], [ ]
                for index_entry in l_index_entry :
                    pos_start_block, pos_end_block = self._arr_data_df_index[ index_entry ]
                    file.seek( pos_start_block )
                    arr_int_feature, arr_int_barcode, arr_value = self._func_processed_bytes_to_arrays_mtx( file.read( pos_end_block - pos_start_block ) )
                    l_arr_int_feature.append( arr_int_feature )
                    l_arr_int_barcode.append( arr_int_barcode )
                    l_arr_value.append( arr_value )
                l_arrays_mtx = ( l_arr_int_feature, l_arr_int_barcode, l_arr_value )
                # if 'flag_as_a_worker' is True, send the result or return the result
                if flag_as_a_worker :
                    pipe_to_main_thread.send( l_arrays_mtx ) # send unzipped result back
                else :
                    return l_arrays_mtx
        
        if self.int_num_threads_for_ramdom_accessing > 1 and int_num_entries > 1 : # enter multithreading mode only more than one entry should be retrieved
            # initialize workers
            int_n_workers = min( self.int_num_threads_for_ramdom_accessing, int_num_entries ) # one thread for distributing records. Minimum numbers of workers for sorting is 1 # the number of workers should not be larger than the number of entries to retrieve.
            l_pipes_from_main_process_to_worker = list( mp.Pipe( ) for _ in range( self.int_num_threads_for_ramdom_accessing ) ) # create pipes for sending records to workers # add process for receivers
            l_pipes_from_worker_to_main_process = list( mp.Pipe( ) for _ in range( self.int_num_threads_for_ramdom_accessing ) ) # create pipes for collecting results from workers
            l_processes = list( mp.Process( target = __retrieve_data_from_ramtx_as_a_worker__, args = ( l_pipes_from_main_process_to_worker[ index_worker ][ 1 ], l_pipes_from_worker_to_main_process[ index_worker ][ 0 ] ) ) for index_worker in range( int_n_workers ) ) # add a process for distributing fastq records
            for p in l_processes :
                p.start( )
            # distribute works
            for index_worker, l_index_entry_for_each_worker in enumerate( LIST_Split( l_index_entry, int_n_workers ) ) : # continuous or distributed ? what would be more efficient?
                l_pipes_from_main_process_to_worker[ index_worker ][ 0 ].send( l_index_entry_for_each_worker )
            # wait until all works are completed
            int_num_workers_completed = 0
            l_arr_int_feature, l_arr_int_barcode, l_arr_value = [ ], [ ], [ ]
            while int_num_workers_completed < int_n_workers : # until all works are completed
                for _, pipe in l_pipes_from_worker_to_main_process :
                    if pipe.poll( ) :
                        l_arrays_mtx = pipe.recv( )
                        l_arr_int_feature.extend( l_arrays_mtx[ 0 ] )
                        l_arr_int_barcode.extend( l_arrays_mtx[ 1 ] )
                        l_arr_value.extend( l_arrays_mtx[ 2 ] )
                        del l_arrays_mtx
                        int_num_workers_completed += 1
                time.sleep( 0.1 )
            # dismiss workers once all works are completed
            for p in l_processes :
                p.join( )
        else : # single thread mode
            l_arr_int_feature, l_arr_int_barcode, l_arr_value = __retrieve_data_from_ramtx_as_a_worker__( l_index_entry, flag_as_a_worker = False )
        
        # parse the retrieved results, and combine the results
        arr_int_feature = np.concatenate( l_arr_int_feature )
        arr_int_barcode = np.concatenate( l_arr_int_barcode )
        arr_value = np.concatenate( l_arr_value )
        return arr_int_feature, arr_int_barcode, arr_value
    def __getitem__( self, l_entry ) : 
        # retrieve data from RAMtx as a dataframe, and parse them as arrays, and convert 1-based indices to 0-based indices
        arr_int_feature, arr_int_barcode, arr_value = self._Retrieve_data_from_ramtx_( l_entry )
        if self.__flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only : # return the sparse matrix directly to increase the efficiency and reduce the memory footprint of the RAMtx object.
            return arr_int_feature, arr_int_barcode, arr_value
        else : 
            # compose a sparse matrix using the arrays
            sp_csr_data = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( ( arr_value, ( arr_int_barcode, arr_int_feature ) ), shape = ( self._int_num_barcodes, self._int_num_features ) ) ) # in anndata.X, row = barcode, column = feature
            self.adata.X = sp_csr_data # update 'adata' with the fetched data
            return self.adata # return the anndata containing fetched data
    def __repr__( self ) :
        return f"RAMtx object containing {self._int_num_records} records of {self._int_num_features} features X {self._int_num_barcodes} barcodes\n\tRAMtx path: {self.path_folder_mtx}\n\tcurrent settings: int_num_threads_for_ramdom_accessing = {self.int_num_threads_for_ramdom_accessing}, flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only = {self.__flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only}"
    def __contains__( self, x ) -> bool :
        ''' check whether an entry is available in the index '''
        return x in self._dict_entry_to_index_entry
    def __iter__( self ) :
        ''' yield each entry in the index upon iteration '''
        return iter( self._dict_entry_to_index_entry )
    def _warm_up_( self, int_num_threads = 10, int_num_entries_to_randomly_access = 1000 ) :
        ''' # 2022-05-12 21:57:07 
        Rapidly warm up and precondition the disk (disk caching) for accessing RAMtx data by randomly accessing 'int_num_entries' number of entries using 'int_num_threads' threads simultaneously.
        '''
        arr_entry = self._df_index.index.values
        int_num_threads_prev = self.int_num_threads_for_ramdom_accessing # retrieve the previous setting
        self.int_num_threads_for_ramdom_accessing = int_num_threads # use 'int_num_threads' for warm-up
        self[ list( arr_entry[ int( np.random.random( ) * self._int_num_entry_indexed ) ] for i in range( int_num_entries_to_randomly_access ) ) ] # warm-up!
        self.int_num_threads_for_ramdom_accessing = int_num_threads_prev # restore the previous setting
    def convert( self, file_format = 'pickle', int_num_digits_after_floating_point_for_export = 5, flag_output_value_is_float = True, dtype_of_row_and_col_indices = np.uint32, dtype_of_value = None, int_num_threads = None, flag_debugging = False ) :
        """ # 2022-05-25 21:43:59 
        convert the file_format of the current RAMtx to that of 'file_format'
        
        'flag_output_value_is_float' : (Default: True) a flag indicating whether the output value is a floating point number. If True, 'int_num_digits_after_floating_point_for_export' argument will be active.
        'file_format' : for more detail, see docstring of the RAMtx class
        'int_num_digits_after_floating_point_for_export' : (Default: 5) the number of digitst after the floating point when exporting the resulting values to the disk.
        """
        # handle input
        if int_num_threads is None :
            int_num_threads = self.int_num_cpus
        # check whether the given file_format already exists
        if file_format in self._l_file_format :
            if self.verbose :
                print( f"the current RAMtx object is already available in the given file_format '{file_format}', exiting" )
                return -1
        
        """ retrieve RAMtx_file_format specific export settings """
        str_format_value = "{:." + str( int_num_digits_after_floating_point_for_export ) + "f}" if flag_output_value_is_float else "{}" # a string for formating value # format for a float or an integer 
        str_etx, str_ext_index, func_arrays_mtx_to_processed_bytes = _get_func_arrays_mtx_to_processed_bytes_and_other_settings_based_on_file_format( file_format, str_format_value, dtype_of_row_and_col_indices, dtype_of_value )

        """ convert matrix values and save it to the output RAMtx object """
        # use multiple processes
        # create a temporary folder
        path_folder_ramtx_output = self.path_folder_mtx # export to the current RAMtx object
        path_folder_temp = f'{path_folder_ramtx_output}temp_{UUID( )}/'
        os.makedirs( path_folder_temp, exist_ok = True )

        # prepare multiprocessing
        arr_entry = self._df_index.index.values # retrieve list of entries from _df_index
        l_arr_entry_for_each_chunk, l_arr_weight_for_each_chunk = LIST_Split( arr_entry, int_num_threads, flag_contiguous_chunk = True, arr_weight_for_load_balancing = self._df_index.loc[ arr_entry ].int_num_bytes.values, return_split_arr_weight = True ) # perform load balancing using the total count for each entry as a weight
        
        # setting for the pipeline
        int_total_weight_for_each_batch = 2500000
        def __compress_and_index_a_portion_of_ramtx_as_a_worker__( index_chunk, q ) :
            ''' # 2022-05-08 13:19:13 
            save a portion of a sparse matrix referenced by 'index_chunk'
            'q' : multiprocessing.Queue object for collecting results
            '''
            # open output files
            file_output = open( f'{path_folder_temp}indexed.{index_chunk}.{str_etx}', 'wb' )
            file_index_output = gzip.open( f'{path_folder_temp}indexed.{index_chunk}.{str_etx}.{str_ext_index}', 'wb' )

            int_num_bytes_written = 0 # track the number of written bytes
            int_num_records_written = 0 # track the number of records written to the output file
            # methods and variables for handling metadata
            int_total_weight_current_batch = 0
            l_entry_current_batch = [ ]
            def __process_batch__( file_output, file_index_output, int_num_bytes_written, int_num_records_written, l_entry_current_batch ) :
                ''' # 2022-05-08 13:19:07 
                process the current batch and return updated 'int_num_bytes_written' and 'int_num_records_written'
                '''
                # retrieve the number of index_entries
                int_num_entries = len( l_entry_current_batch )
                # handle invalid inputs
                if int_num_entries == 0 :
                    return int_num_bytes_written, int_num_records_written

                # retrieve data for the current batch 
                arr_int_feature, arr_int_barcode, arr_value = self._Retrieve_data_from_ramtx_( l_entry_current_batch )
                # renumber indices of sorted entries to match that in the sorted matrix

                # retrieve the start of the block, marked by the change of int_entry 
                l_pos_start_block = [ 0 ] + list( np.where( np.diff( arr_int_feature if self.flag_ramtx_sorted_by_id_feature else arr_int_barcode ) )[ 0 ] + 1 ) + [ len( arr_value ) ] # np.diff decrease the index of entries where change happens, and +1 should be done # 10X matrix data: row = feature, col = barcodes
                # prepare

                for index_block in range( len( l_pos_start_block ) - 1 ) : # for each block (each block contains records of a single entry)
                    slice_for_the_current_block = slice( l_pos_start_block[ index_block ], l_pos_start_block[ index_block + 1 ] )
                    arr_int_feature_of_the_current_block, arr_int_barcode_of_the_current_block, arr_value_of_the_current_block = arr_int_feature[ slice_for_the_current_block ], arr_int_barcode[ slice_for_the_current_block ], arr_value[ slice_for_the_current_block ]
                    int_entry = ( arr_int_feature_of_the_current_block if self.flag_ramtx_sorted_by_id_feature else arr_int_barcode_of_the_current_block )[ 0 ] # retrieve the interger representation of the current entry
                    int_num_records_written_for_the_current_entry = len( arr_value_of_the_current_block ) # retrieve the number of records
                    bytes_processed = func_arrays_mtx_to_processed_bytes( ( arr_int_feature[ slice_for_the_current_block ], arr_int_barcode[ slice_for_the_current_block ], arr_value[ slice_for_the_current_block ] ) ) # retrieve data for the current block
                    int_num_bytes_written_for_the_current_entry = len( bytes_processed ) # record the number of bytes of the written data
                    # write the processed bytes to the output file
                    file_output.write( bytes_processed )

                    # write the index
                    file_index_output.write( ( '\t'.join( map( str, [ int_entry + 1, int_num_bytes_written, int_num_bytes_written + int_num_bytes_written_for_the_current_entry, int_num_records_written_for_the_current_entry ] ) ) + '\n' ).encode( ) ) # write an index for the current entry # 0>1 coordinate conversion for 'int_entry'

                    int_num_bytes_written += int_num_bytes_written_for_the_current_entry # update the number of bytes written
                    int_num_records_written += int_num_records_written_for_the_current_entry # update the number of records written
                return int_num_bytes_written, int_num_records_written

            for entry, float_weight in zip( l_arr_entry_for_each_chunk[ index_chunk ], l_arr_weight_for_each_chunk[ index_chunk ] ) : # retrieve inputs for the current process
                # add current index_sorting to the current batch
                l_entry_current_batch.append( entry )
                int_total_weight_current_batch += float_weight
                # if the weight becomes larger than the threshold, process the batch and reset the batch
                if int_total_weight_current_batch > int_total_weight_for_each_batch :
                    # process the current batch
                    int_num_bytes_written, int_num_records_written = __process_batch__( file_output, file_index_output, int_num_bytes_written, int_num_records_written, l_entry_current_batch )
                    # initialize the next batch
                    l_entry_current_batch = [ ]
                    int_total_weight_current_batch = 0

            # process the remaining entries
            int_num_bytes_written, int_num_records_written = __process_batch__( file_output, file_index_output, int_num_bytes_written, int_num_records_written, l_entry_current_batch )

            # close files
            for file in [ file_output, file_index_output ] :
                file.close( )

            # record the number of records written for the current chunk
            q.put( int_num_records_written ) 

        q = mp.Queue( ) # multiprocessing queue is process-safe
        l_worker = list( mp.Process( target = __compress_and_index_a_portion_of_ramtx_as_a_worker__, args = ( index_chunk, q ) ) for index_chunk in range( int_num_threads ) )

        ''' start works and wait until all works are completed by workers '''
        for p in l_worker :
            p.start( ) # start workers
        for p in l_worker :
            p.join( )  

        ''' summarize output values '''
        # make sure the number of returned results match that of deployed workers
        assert q.qsize( ) == len( l_worker ) 
        int_num_records_written = np.sum( list( q.get( ) for i in range( q.qsize( ) ) ) ) # retrieve the total number of written records

        ''' retrieve metadata '''
        # 10X matrix: row  = barcode, col = feature
        int_num_features = self._int_num_features
        int_num_barcodes = self._int_num_barcodes
        int_num_records = int_num_records_written # retrieve the number of entries

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
            for path_file in df.path.values :
                os.remove( path_file )
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
            df_index.columns = [ 'index_entry', 'int_pos_start', 'int_pos_end', 'int_num_records' ]
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
            for str_name_file_glob in [ 'indexed.*.mtx.idx.tsv.gz' ] :
                for path_file in glob.glob( f'{path_folder_temp}{str_name_file_glob}' ) :
                    os.remove( path_file )

        ''' export features and barcodes '''
        # rename output files
        os.rename( f"{path_folder_temp}indexed.{str_etx}", f"{path_folder_ramtx_output}matrix.{str_etx}" )
        os.rename( f"{path_folder_temp}indexed.{str_etx}.{str_ext_index}", f"{path_folder_ramtx_output}matrix.{str_etx}.{str_ext_index}" )

        # delete temporary folder
        if not flag_debugging :
            shutil.rmtree( path_folder_temp ) 
        
        ''' update the RAMtx metadata once the export has been completed. '''
        self._dict_metadata[ 'file_format' ].append( file_format )
        with open( f"{path_folder_ramtx_output}ramtx.metadata.json", 'w' ) as file :
            json.dump( self._dict_metadata, file )
class RamData( ) :
    """ # 2022-05-31 23:17:38 
    This class contains single-cell transcriptomic/genomic data utilizing RAMtx data structures, allowing efficient parallelization of analysis of single cell data with minimal memory consumption. 
    
    'name_data_current' : if None, automatically load the paired RAMtx objects of the most appropriate 'name_data' of the current RamData object. 
    'int_num_cpus' : number of CPUs (processes) to use to distribute works.
    'file_format' : preferred file format of RAMtx data objects for the current the RamData object.
    
    ==== AnnDataContainer ====
    'flag_enforce_name_adata_with_only_valid_characters' : enforce
    """
    def __init__( self, path_folder_ramdata, name_data_current = None, int_num_cpus = 64, file_format = 'feather_lz4', verbose = False, flag_debugging = False, flag_enforce_name_adata_with_only_valid_characters = True ) :
        self.path_folder_ramdata = path_folder_ramdata
        
        ''' handle 'name_data_current' argument ''' 
        self.set_name_data = set( GLOB_Retrive_Strings_in_Wildcards( f"{self.path_folder_ramdata}*/*/*" ).wildcard_0.values ) # retrieve the set of existing ramtx objects in the current RamData object
        if name_data_current not in self.set_name_data :
            name_data_current = None
        if 'from_anndata' in self.set_name_data :
            name_data_current = 'from_anndata'
        elif 'raw' in self.set_name_data :
            name_data_current = 'raw'
        if name_data_current is None :
            raise KeyError( f"'{name_data_current}' RAMtx object does not exists in the current RamData" )
        self.name_data_current = name_data_current
        self.verbose = verbose
        self.flag_debugging = flag_debugging
        self.int_num_cpus = int_num_cpus
        self.file_format = file_format
        
        if self.verbose :
            print( 'loading RAMtxs ... ', end = '' )
        # load RAMtx objects, sorted by feature and sorted by barcodes # receive only sparse matrix from RAMtx to reduce the memory footprint 
        self.ramtx_for_feature = RAMtx( f'{self.path_folder_ramdata}{self.name_data_current}/sorted_by_feature/', flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only = True, int_num_cpus = int_num_cpus, file_format = file_format )
        self.ramtx_for_barcode = RAMtx( f'{self.path_folder_ramdata}{self.name_data_current}/sorted_by_barcode/', flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only = True, int_num_cpus = int_num_cpus, file_format = file_format )
        if self.verbose :
            print( 'done' )
        
        # load the main anndata object
        if not os.path.exists( f'{self.path_folder_ramdata}main.h5ad' ) :
            raise FileNotFoundError( f'{self.path_folder_ramdata}main.h5ad does not exist.' )
        self.adata = sc.read_h5ad( f'{self.path_folder_ramdata}main.h5ad' )
        self._int_num_barcodes, self._int_num_features, self._int_num_records = len( self.adata.obs ), len( self.adata.var ), self.ramtx_for_feature._int_num_records # retrieve the number of barcodes, features, and entries
        
        # retrieve mapping from string representations of cells and features to integer indices
        self._dict_name_feature_to_int_index = dict( ( e, i ) for i, e in enumerate( self.adata.var.index.values ) )
        self._dict_id_cell_to_int_index = dict( ( e, i ) for i, e in enumerate( self.adata.obs.index.values ) )
        
        # set AnnDataContainer attribute for containing various AnnData objects associated with the current RamData
        self.ad = AnnDataContainer( path_prefix_default = self.path_folder_ramdata, flag_enforce_name_adata_with_only_valid_characters = flag_enforce_name_adata_with_only_valid_characters, main = self.adata, ** PD_Select( GLOB_Retrive_Strings_in_Wildcards( f'{self.path_folder_ramdata}*.h5ad' ), wildcard_0 = 'main', deselect = True ).set_index( 'wildcard_0' ).path.to_dict( ) ) # load the 'main' AnnData and file paths of the other AnnData objects as place holders to the 'AnnDataContainer' object
        
        # set miscellaneous attributes
        self.set_int_barcode = None # 'None' means all attributes are active
        self.set_int_feature = None # 'None' means all attributes are active
    def save( self, * l_name_adata ) :
        ''' wrapper of AnnDataContainer.save '''
        self.ad.update( * l_name_adata )
    def __contains__( self, x ) -> bool :
        ''' check whether an 'name_data' is available in the current RamData '''
        return x in self.set_name_data
    def __iter__( self ) :
        ''' yield each name_data upon iteration '''
        return iter( self.set_name_data )
    def set_data( self, name_data ) :
        """ # 2022-05-22 22:27:55 
        change current data to the data referenced by the given argument 'name_data'
        
        'name_data' : 
        """
        if name_data != self.name_data_current :
            if name_data in self.set_name_data :
                # load RAMtx objects, sorted by feature and sorted by barcodes # receive only sparse matrix from RAMtx to reduce the memory footprint 
                ramtx_for_feature = RAMtx( f'{self.path_folder_ramdata}{name_data}/sorted_by_feature/', flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only = True )
                ramtx_for_barcode = RAMtx( f'{self.path_folder_ramdata}{name_data}/sorted_by_barcode/', flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only = True )
                
                if self.verbose :
                    print( f"RAMtxs of the data '{name_data}' has been loaded" )
                # once RAMtx objects are loaded, set the ramtx objects as attributes of the RamData
                self.ramtx_for_feature = ramtx_for_feature
                self.ramtx_for_barcode = ramtx_for_barcode
                self.name_data_current = name_data
                if self.verbose :
                    print( f"'{name_data}' data has been loaded" )
            else :
                if self.verbose :
                    print( f"'{name_data}' does not exist in the current RamData" )        
                return - 1
    def __repr__( self ) :
        return f"RamData object stored at {self.path_folder_ramdata}\n\twith with the following data : {self.set_name_data}\n\tcurrent RAMtx object '{self.name_data_current}' is stored at '{self.ramtx_for_feature.path_folder_mtx}' (sorted by feature) and '{self.ramtx_for_barcode.path_folder_mtx}' (sorted by barcode)\n\tcurrent AnnData:\n{self.adata}"
    def __map_string_indices_to_valid_int_indices__( self, l_index, dict_mapping ) :
        ''' map string indices 'l_index' to valid integer indices using the given 'dict_mapping' '''
        return list( dict_mapping[ e ] for e in l_index if e in dict_mapping )
    def __getitem__( self, index ) :
        ''' handle multiple axes '''
        if isinstance( index, tuple ) and len( index ) == 2 : # if the object received a tuple of length 2, assumes indices for two axes are given.
            index_barcode, index_feature = index
            if isinstance( index_barcode, slice ) :
                index = index_feature
            elif isinstance( index_feature, slice ) :
                index = index_barcode
        ''' handle single axis '''
        if isinstance( index, str ) : # if only a single string is given, wrap the string in a list
            index = [ index ]
        if index[ 0 ] in self._dict_name_feature_to_int_index :
            row, col, data = self.ramtx_for_feature[ self.__map_string_indices_to_valid_int_indices__( index, self._dict_name_feature_to_int_index ) ]
        else :
            row, col, data = self.ramtx_for_barcode[ self.__map_string_indices_to_valid_int_indices__( index, self._dict_id_cell_to_int_index ) ]
            
        self.adata.X = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( ( data, ( col, row ) ), shape = ( self._int_num_barcodes, self._int_num_features ) ) ) # in anndata.X, row = barcode, column = feature
        return self.adata
    ''' core methods for analyzing RamData '''
    def summarize( self, name_data, axis, summarizing_func, int_num_threads = None, flag_overwrite_columns = True ) :
        ''' 
        this function summarize entries of the given axis (0 = barcode, 1 = feature) using the given function
        
        example usage: calculate normalized count data, perform log1p transformation, cell filtering, etc.
        
        =========
        inputs 
        =========

        'ram': an input RamData object
        'name_data' : name of the data in the given RamData object to summarize
        'axis': int or str. 
               0 or 'barcode' for applying a given summarizing function for barcodes
               1 or 'feature' for applying a given summarizing function for features
        'summarizing_func' : function object. a function that takes a sparse matrix (note: not indexed like a dataframe, and integer representations of barcodes and features should be used to summarize the matrix) and return a dictionary containing 'name_of_summarized_data' as key and 'value_of_summarized_data' as value. the resulting summarized outputs will be added as columns of appriproate dataframe of the AnnData of the given RamData object (self.adata.obs or self.adata.var)
        
                    summarizing_func( self, arr_int_barcode_of_an_entry, arr_int_feature_of_an_entry, arr_value_of_an_entry ) -> dictionary containing 'key' as summarized metric name and 'value' as a summarized value for the entry
                    
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
                            
                    'sum_scarab_feature_category' (axis = 'barcode') :
                            calculate the total sum (and mean) for each feature classification type of the scarab output for each barcode ('cell')
                            useful for initial barcode filtering
                            
                            returns: 'sum', 'mean', and summed values of features for each feature classification type of the scarab output
                    
        'int_num_threads' : the number of CPUs to use. by default, the number of CPUs set by the RamData attribute 'int_num_cpus' will be used.
        'flag_overwrite_columns' : (Default: True) overwrite the columns of the output annotation dataframe of RamData.adata if columns with the same colume name exists
        
        =========
        outputs 
        =========
        the summarized metrics will be added to appropriate dataframe attribute of the AnnData of the current RamData (self.adata.obs for axis = 0 and self.adata.var for axis = 1).
        the column names will be constructed as the following :
            f"{name_data}_{key}"
        if the column name already exist in the dataframe, the values of the columns will be overwritten (alternatively, a suffix of current datetime will be added to the column name, by setting 'flag_overwrite_columns' to True)
        '''
        # check the validility of the input arguments
        if name_data not in self.set_name_data :
            if self.verbose :
                print( f"[ERROR] [RamData.summarize] invalid argument 'name_data' : '{name_data}' does not exist." )
            return -1 
        if axis not in { 0, 'barcode', 1, 'feature' } :
            if self.verbose :
                print( f"[ERROR] [RamData.summarize] invalid argument 'axis' : '{name_data}' is invalid. use one of { { 0, 'barcode', 1, 'feature' } }" )
            return -1 
        # handle inputs
        flag_summarizing_barcode = axis in { 0, 'barcode' } # retrieve a flag indicating whether the data is summarized for each barcode or not
        if int_num_threads is None :
            int_num_threads = self.int_num_cpus
        if summarizing_func == 'sum' :
            def summarizing_func( self, arr_int_barcode_of_an_entry, arr_int_feature_of_an_entry, arr_value_of_an_entry ) :
                ''' # 2022-05-23 12:11:55 
                calculate sum of the values of the current entry
                
                assumes 'int_num_records' > 0
                '''
                int_num_records = len( arr_value_of_an_entry ) # retrieve the number of records of the current entry
                dict_summary = { 'sum' : np.sum( arr_value_of_an_entry ) if int_num_records > 30 else sum( arr_value_of_an_entry ) } # if an input array has more than 30 elements, use np.sum to calculate the sum
                int_num_entries = self._int_num_barcodes if flag_summarizing_barcode else self._int_num_features # retrieve the number of entries based on the axis
                dict_summary[ 'mean' ] = dict_summary[ 'sum' ] / int_num_entries # calculate the mean
                return dict_summary
        elif summarizing_func == 'sum_and_dev' :
            def summarizing_func( self, arr_int_barcode_of_an_entry, arr_int_feature_of_an_entry, arr_value_of_an_entry ) :
                ''' # 2022-05-23 12:11:55 
                calculate sum and deviation of the values of the current entry
                
                assumes 'int_num_records' > 0
                '''
                int_num_records = len( arr_value_of_an_entry ) # retrieve the number of records of the current entry
                dict_summary = { 'sum' : np.sum( arr_value_of_an_entry ) if int_num_records > 30 else sum( arr_value_of_an_entry ) } # if an input array has more than 30 elements, use np.sum to calculate the sum
                int_num_entries = self._int_num_barcodes if flag_summarizing_barcode else self._int_num_features # retrieve the number of entries based on the axis
                dict_summary[ 'mean' ] = dict_summary[ 'sum' ] / int_num_entries # calculate the mean
                arr_dev = ( arr_value_of_an_entry - dict_summary[ 'mean' ] ) ** 2 # calculate the deviation
                dict_summary[ 'deviation' ] = np.sum( arr_dev ) if int_num_records > 30 else sum( arr_dev )
                dict_summary[ 'variance' ] = dict_summary[ 'deviation' ] / ( int_num_entries - 1 ) if int_num_entries > 1 else np.nan
                return dict_summary
        elif summarizing_func == 'sum_scarab_feature_category' and flag_summarizing_barcode :
            self._classify_feature_of_scarab_output_( ) # classify feature with a default setting
            def summarizing_func( self, arr_int_barcode_of_an_entry, arr_int_feature_of_an_entry, arr_value_of_an_entry ) :
                ''' # 2022-05-30 15:23:37 
                calculate sum of each category of the features 

                assumes 'int_num_records' > 0
                '''
                dict_data = self._dict_data_for_feature_classification
                
                dict_summary = dict( ( 'sum___category_detailed___' + e, 0 ) for e in dict_data[ 'l_name_feaure_category_detailed' ] ) # initialize the output dictionary
                dict_res = pd.DataFrame( { 2 : arr_value_of_an_entry, 3 : dict_data[ 'vectorized_get_int_feature_category_detailed' ]( arr_int_feature_of_an_entry ) }, index = np.zeros( len( arr_int_feature_of_an_entry ), dtype = np.uint8 ) ).groupby( [ 3 ] ).sum( )[ 2 ].to_dict( ) # currently the most efficient way of summarizing result (as far as the method tested..)
                dict_summary.update( dict( ( 'sum___category_detailed___' + dict_data[ 'l_name_feaure_category_detailed' ][ int_feature_class ], dict_res[ int_feature_class ] ) for int_feature_class in dict_res ) ) 
                
                int_num_records = len( arr_value_of_an_entry ) # retrieve the number of records of the current entry
                dict_summary[ 'sum' ] = np.sum( arr_value_of_an_entry ) if int_num_records > 30 else sum( arr_value_of_an_entry ) # if an input array has more than 30 elements, use np.sum to calculate the sum
                int_num_entries = self._int_num_barcodes if flag_summarizing_barcode else self._int_num_features # retrieve the number of entries based on the axis
                dict_summary[ 'mean' ] = dict_summary[ 'sum' ] / int_num_entries # calculate the mean
                return dict_summary
        elif not hasattr( summarizing_func, '__call__' ) : # if 'summarizing_func' is not a function, report error message and exit
            if self.verbose :
                print( f"given summarizing_func is not a function, exiting" )
            return -1
        # retrieve the list of key values returned by 'summarizing_func' by applying dummy values
        arr_dummy_one, arr_dummy_zero = np.ones( 10, dtype = int ), np.zeros( 10, dtype = int )
        dict_res = summarizing_func( self, arr_dummy_zero, arr_dummy_zero, arr_dummy_one )
        l_name_col_summarized = sorted( list( dict_res ) ) # retrieve the list of key values of an dict_res result returned by 'summarizing_func'
        
        def RAMtx_Summarize( self, rtx, summarizing_func, l_name_col_summarized, int_num_threads = 8 ) :
            ''' # 2022-05-23 11:38:24 
            Assumes 'flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only' setting for RAmtx object is set to 'True'
            
            =========
            inputs 
            =========

            'self' : an input RamData object
            'rtx': an input RAMtx object
            'summarizing_func' : function object. a function that takes a sparse matrix (note: not indexed like a dataframe, and integer representations of barcodes and features should be used to summarize the matrix) and return a dictionary containing 'name_of_summarized_data' as key and 'value_of_summarized_data' as value. the resulting summarized outputs will be added as columns of appriproate dataframe of the AnnData of the given RamData object (self.adata.obs or self.adata.var)
        
                    summarizing_func( self, arr_int_barcode_of_an_entry, arr_int_feature_of_an_entry, arr_value_of_an_entry ) -> dictionary containing 'key' as summarized metric name and 'value' as a summarized value for the entry
                    
            '''
            """ convert matrix values and save it to the output RAMtx object """
            # use multiple processes
            # create a temporary folder inside the folder containing the input RAMtx object
            path_folder_temp = f'{rtx.path_folder_mtx}temp_{UUID( )}/'
            os.makedirs( path_folder_temp, exist_ok = True )

            # prepare multiprocessing
            arr_int_entry = rtx._df_index.index.values 
            l_arr_int_entry_for_each_chunk, l_arr_weight_for_each_chunk = LIST_Split( arr_int_entry, int_num_threads, flag_contiguous_chunk = True, arr_weight_for_load_balancing = rtx._df_index.loc[ arr_int_entry ].int_num_bytes.values, return_split_arr_weight = True ) # perform load balancing using the total count for each entry as a weight

            # setting for the pipeline
            int_total_weight_for_each_batch = 2500000
            def __summarize_a_portion_of_ramtx_as_a_worker__( index_chunk ) :
                ''' # 2022-05-08 13:19:13 
                summarize a portion of a sparse matrix containined in the input RAMtx object, referenced by 'index_chunk'
                '''
                # open output files
                file_summary_output = gzip.open( f'{path_folder_temp}summarized.{index_chunk}.tsv.gz', 'wb' )

                # methods and variables for handling metadata
                int_total_weight_current_batch = 0
                l_int_entry_current_batch = [ ]
                def __process_batch__( file_summary_output, l_int_entry_current_batch ) :
                    ''' # 2022-05-08 13:19:07 
                    process the current batch of entries
                    '''
                    # retrieve the number of index_entries
                    int_num_entries = len( l_int_entry_current_batch )
                    # handle invalid inputs
                    if int_num_entries == 0 :
                        return -1

                    # retrieve data for the current batch
                    arr_int_feature, arr_int_barcode, arr_value = rtx[ l_int_entry_current_batch ]
                    # renumber indices of sorted entries to match that in the sorted matrix
                    
                    # prepare
                    # retrieve the start of the block, marked by the change of int_entry 
                    l_pos_start_block = [ 0 ] + list( np.where( np.diff( arr_int_feature if rtx.flag_ramtx_sorted_by_id_feature else arr_int_barcode ) )[ 0 ] + 1 ) + [ len( arr_value ) ] # np.diff decrease the index of entries where change happens, and +1 should be done # 10X matrix data: row = feature, col = barcodes
                    for index_block in range( len( l_pos_start_block ) - 1 ) : # for each block (each block contains records of a single entry)
                        slice_for_the_current_block = slice( l_pos_start_block[ index_block ], l_pos_start_block[ index_block + 1 ] )
                        arr_int_feature_of_the_current_block, arr_int_barcode_of_the_current_block, arr_value_of_the_current_block = arr_int_feature[ slice_for_the_current_block ], arr_int_barcode[ slice_for_the_current_block ], arr_value[ slice_for_the_current_block ] # retrieve data for the current block
                        int_entry = ( arr_int_feature_of_the_current_block if rtx.flag_ramtx_sorted_by_id_feature else arr_int_barcode_of_the_current_block )[ 0 ] # retrieve the interger representation of the current entry
                        
                        dict_res = summarizing_func( self, arr_int_barcode_of_the_current_block, arr_int_feature_of_the_current_block, arr_value_of_the_current_block ) # summarize result
                        # write the result
                        file_summary_output.write( ( '\t'.join( map( str, [ int_entry ] + list( dict_res[ name_col ] if name_col in dict_res else np.nan for name_col in l_name_col_summarized ) ) ) + '\n' ).encode( ) ) # write an index for the current entry # 0>1 coordinate conversion for 'int_entry'

                for int_entry, float_weight in zip( l_arr_int_entry_for_each_chunk[ index_chunk ], l_arr_weight_for_each_chunk[ index_chunk ] ) : # retrieve inputs for the current process
                    # add current index_sorting to the current batch
                    l_int_entry_current_batch.append( int_entry )
                    int_total_weight_current_batch += float_weight
                    # if the weight becomes larger than the threshold, process the batch and reset the batch
                    if int_total_weight_current_batch > int_total_weight_for_each_batch :
                        # process the current batch
                        __process_batch__( file_summary_output, l_int_entry_current_batch )
                        # initialize the next batch
                        l_int_entry_current_batch = [ ]
                        int_total_weight_current_batch = 0

                # process the remaining entries
                __process_batch__( file_summary_output, l_int_entry_current_batch )

                # close files
                for file in [ file_summary_output ] :
                    file.close( )
            
            # organize workers
            l_worker = list( mp.Process( target = __summarize_a_portion_of_ramtx_as_a_worker__, args = ( index_chunk, ) ) for index_chunk in range( int_num_threads ) )

            ''' start works and wait until all works are completed by workers '''
            for p in l_worker :
                p.start( ) # start workers
            for p in l_worker :
                p.join( )  

            ''' combine outputs matrix '''
            # combine output matrix files
            for str_filename_glob in [ 'summarized.*.tsv.gz' ] :
                # collect the list of input files in the order of 'index_chunk'
                df = GLOB_Retrive_Strings_in_Wildcards( f'{path_folder_temp}{str_filename_glob}' )
                df[ 'wildcard_0' ] = df.wildcard_0.astype( int )
                df.sort_values( 'wildcard_0', inplace = True ) # sort the list of input files in the order of 'index_chunk'
                # combine input files
                OS_Run( [ 'cat' ] + list( df.path.values ), path_file_stdout = f"{path_folder_temp}{str_filename_glob.replace( '.*.', '.' )}", stdout_binary = True )
                # delete input files
                for path_file in df.path.values :
                    os.remove( path_file )

            ''' read combined output file '''
            df_summarized = pd.read_csv( f"{path_folder_temp}summarized.tsv.gz", sep = '\t', header = None )
            df_summarized.columns = [ 'int_entry' ] + l_name_col_summarized # add appropriate column names
            
            # delete temporary folder
            if not self.flag_debugging :
                shutil.rmtree( path_folder_temp ) 
                
            # return summarized result as a dataframe
            return df_summarized
        
        df_summarized = RAMtx_Summarize( self, ( self.ramtx_for_barcode if flag_summarizing_barcode else self.ramtx_for_feature ), summarizing_func = summarizing_func, l_name_col_summarized = l_name_col_summarized, int_num_threads = int_num_threads ) # summarize a RAMtx object of a given axis
        
        # append the summarized results to the annotation dataframe of the main AnnData attribute of the current RamData
        df_entry = self.adata.obs if flag_summarizing_barcode else self.adata.var
        arr_entry = df_entry.index.values
        df_summarized[ 'str_entry' ] = list( arr_entry[ i ] for i in df_summarized.int_entry.values )
        df_summarized.drop( columns = [ 'int_entry' ], inplace = True )
        df_summarized.set_index( 'str_entry', inplace = True )
        df_summarized.columns = list( f"{name_data}_{s}" for s in df_summarized.columns.values ) # rename column names and add 'name_data' + '_' as a prefix
        if flag_overwrite_columns :
            set_df_summarized_columns = set( df_summarized.columns.values )
            df_entry.drop( columns = list( s for s in df_entry.columns.values if s in set_df_summarized_columns ), inplace = True ) # drop columns of df_entry that exist in 'df_summarized', a data that will be added to 'df_entry'
        df_entry = df_entry.join( df_summarized, how = 'left', rsuffix = TIME_GET_timestamp( ) ) # add time stamp as a suffix
        if flag_summarizing_barcode :
            self.adata.obs = df_entry
        else :
            self.adata.var = df_entry
        return df_entry # return resulting dataframe
    def apply( self, name_data, name_data_new = None, func = None, path_folder_ramdata_output = None, flag_dtype_output = np.float64, flag_output_value_is_float = True, file_format = 'mtx_gzipped', int_num_digits_after_floating_point_for_export = 5, int_num_threads = None, dtype_of_row_and_col_indices = np.uint32, dtype_of_value = None, flag_simultaneous_processing_of_paired_ramtx = True, ba_mask_barcode = None, ba_mask_feature = None, verbose = False ) :
        ''' # 2022-06-04 02:06:56 
        this function apply a function and/or filters to the records of the given data, and create a new data object with 'name_data_new' as its name.
        
        example usage: calculate normalized count data, perform log1p transformation, cell filtering, etc.                             
        
        =========
        inputs 
        =========

        'name_data' : name of the data in the given RamData object to analyze
        'name_data_new' : (Default: 'name_data') name of the new data for the paired RAMtx objects that will contains transformed values (the outputs of the functions applied to previous data values). The disk size of the RAMtx objects can be larger or smaller than the RAMtx objects of 'name_data'. please make sure that sufficient disk space remains before calling this function.
        'path_folder_ramdata_output' : (Default: store inside the current RamData). The directory of the RamData object that will contain the outputs (paired RAMtx objects). if integer representations of features and barcodes are updated from filtering, the output RAMtx is now incompatible with the current RamData and should be stored outside the current RamData object. The output directory of the new RamData object can be given through this argument. The RamData object directory should contains new features.tsv.gz and barcodes.tsv.gz (with updated integer representation of features and barcodes)
        'flag_dtype_output' : a datatype of the output values
        'func' : function object or string (Default: identity) a function that takes a tuple of two integers (integer representations of barcode and feature) and another integer or float (value) and returns a modified record. Also, the current RamData object will be given as the first argument (self), and attributes of the current RamData can be used inside the function

                 func( self, int_barcode, int_feature, value ) -> int_barcode_new, int_feature_new, value_new

                 if None is returned, the entry will be discarded in the output RAMtx object. Therefore, the function can be used to both filter or/and transform values
                 
                 a list of pre-defined functions are the followings:
                 'log1p' :
                          X_new = log_10(X_old + 1)
                 
        'flag_output_value_is_float' : (Default: True) a flag indicating whether the output value is a floating point number. If True, 'int_num_digits_after_floating_point_for_export' argument will be active. if 'flag_output_value_is_float' is None, use the same setting as the input RAMtx object. This argument is only valid for the 'mtx_gzipped' output format
        'int_num_digits_after_floating_point_for_export' : (Default: 5) the number of digitst after the floating point when exporting the resulting values to the disk.
        'flag_simultaneous_processing_of_paired_ramtx' : (Default: True) process the paired RAMtx simultaneously using two processes.
        'ba_mask_barcode', 'ba_mask_feature' : the bitarray masks of integer representations of barcodes and features to be included in the output RAMtx (0 = exclude, 1 = include). the barcodes that are exist in the given RAMtx object but absent in the 'ba_mask_barcode' will be deleted from the list. However, the integer representations will NOT be updated automatically, and the user should implement the function given through the 'func' argument to update the 'int_barcode' and 'int_feature'
        'int_num_threads' : the number of CPUs to use. by default, the number of CPUs set by the RamData attribute 'int_num_cpus' will be used.

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
        if name_data_new is None :
            name_data_new = name_data
        if path_folder_ramdata_output is None :
            path_folder_ramdata_output = self.path_folder_ramdata
        if func is None :
            # define identity function if 'func' has not been given
            def func( self, int_barcode, int_feature, value ) :
                return int_barcode, int_feature, value
        elif func == 'log1p' :
            def func( self, int_barcode, int_feature, value ) :
                try :
                    return int_barcode, int_feature, math.log10( value + 1 )
                except : # if an error occurred, return 'None' indicating the output value is invalid
                    return int_barcode, int_feature, None
        # check the validility of the input arguments
        if not name_data in self.set_name_data :
            if verbose :
                print( f"[ERROR] [RamData.Apply] invalid argument 'name_data' : '{name_data}' does not exist." )
            return -1 
        elif path_folder_ramdata_output is None and name_data_new in self.set_name_data : # if the new RAMtx object will be saved to the current RamData and the name of the RAMtx already exists in the current RamData
            if verbose :
                print( f"[ERROR] [RamData.Apply] invalid argument 'name_data_new' : '{name_data_new}' is already present in the current RamData." )
            return -1 
        
        ''' set 'name_data' as current data of RamData '''
        self.set_data( name_data )
        ''' retrieve the default setting '''
        if flag_output_value_is_float is None : # for mtx_gzipped file format, if the input file is integer format, the output file will be also in the integer format
            flag_output_value_is_float = 'flag_output_value_is_float' in self.ramtx_for_barcode._dict_metadata and self.ramtx_for_barcode._dict_metadata[ 'flag_output_value_is_float' ]
        
        """ retrieve RAMtx_file_format specific export settings """
        str_format_value = "{:." + str( int_num_digits_after_floating_point_for_export ) + "f}" if flag_output_value_is_float else "{}" # a string for formating value # format for a float or an integer 
        str_etx, str_ext_index, func_arrays_mtx_to_processed_bytes = _get_func_arrays_mtx_to_processed_bytes_and_other_settings_based_on_file_format( file_format, str_format_value, dtype_of_row_and_col_indices, dtype_of_value )
        
        def RAMtx_Apply( self, rtx, path_folder_ramtx_output, func, flag_output_value_is_float = True, file_format = 'feather_lz4', int_num_digits_after_floating_point_for_export = 5, flag_dtype_output = np.float64, int_num_threads = 8, ba_mask_barcode = None, ba_mask_feature = None, verbose = False, flag_debugging = False ) :
            ''' 
            Assumes 'flag_return_arrays_of_int_feature_int_barcode_value_from_getitem_calls_and_access_items_using_integer_indices_only' setting for RAmtx object is set to 'True'
            
            inputs 
            =========

            'rtx': an input RAMtx object
            'path_folder_ramtx_output' : an output folder for the new RAMtx object
            'func' : a function that takes a tuple of two integers (integer representations of barcode and feature) and another integer or float and returns an modified record. 

                     func( self, int_barcode, int_feature, value ) -> int_barcode_new, int_feature_new, value_new

                     if None or np.nan is returned, the entry will be discarded in the output RAMtx object. Therefore, the function can be used to filter or/and transform values
            'flag_output_value_is_float' : (Default: True) a flag indicating whether the output value is a floating point number. If True, 'int_num_digits_after_floating_point_for_export' argument will be active.
            'file_format' : for more detail, see docstring of the RAMtx class
            'int_num_digits_after_floating_point_for_export' : (Default: 5) the number of digitst after the floating point when exporting the resulting values to the disk.
            'ba_mask_barcode', 'ba_mask_feature' : the bitarray masks of integer representations of barcodes and features to be included in the output RAMtx (0 = exclude, 1 = include). the barcodes that are exist in the given RAMtx object but absent in the 'ba_mask_barcode' will be deleted from the list. However, the integer representations will NOT be updated automatically, and the user should implement the function given through the 'func' argument to update the 'int_barcode' and 'int_feature'
            '''
            # create an ramtx output folder
            os.makedirs( path_folder_ramtx_output, exist_ok = True )

            """ convert matrix values and save it to the output RAMtx object """
            # use multiple processes
            # create a temporary folder
            path_folder_temp = f'{path_folder_ramtx_output}temp_{UUID( )}/'
            os.makedirs( path_folder_temp, exist_ok = True )

            # prepare multiprocessing
            ''' does not retrieve data of entries not included in 'ba_mask_feature' (if 'flag_ramtx_sorted_by_id_feature' is True) or 'ba_mask_barcode'. '''
            ba_mask_entry = ba_mask_feature if rtx.flag_ramtx_sorted_by_id_feature else ba_mask_barcode # retrieve valid set of 'int_entry'
            if ba_mask_entry is not None : # isinstance( ba, bitarray )
                arr_int_entry = list( i for i in rtx._df_index.index.values if ba_mask_entry[ i ] ) # retrieve valid set of 'int_entry' (given by 'ba_mask') available in the current RAMtx
            else :
                arr_int_entry = rtx._df_index.index.values # if 'ba_mask' is None, use all 'int_entry' available in the current RAMtx
            l_arr_int_entry_for_each_chunk, l_arr_weight_for_each_chunk = LIST_Split( arr_int_entry, int_num_threads, flag_contiguous_chunk = True, arr_weight_for_load_balancing = rtx._df_index.loc[ arr_int_entry ].int_num_bytes.values, return_split_arr_weight = True ) # perform load balancing using the total count for each entry as a weight

            # setting for the pipeline
            int_total_weight_for_each_batch = 2500000
            def __compress_and_index_a_portion_of_ramtx_as_a_worker__( index_chunk, q ) :
                ''' # 2022-05-08 13:19:13 
                save a portion of a sparse matrix referenced by 'index_chunk'
                'q' : multiprocessing.Queue object for collecting results
                '''
                # open output files
                file_output = open( f'{path_folder_temp}indexed.{index_chunk}.{str_etx}', 'wb' )
                file_index_output = gzip.open( f'{path_folder_temp}indexed.{index_chunk}.{str_etx}.{str_ext_index}', 'wb' )

                int_num_bytes_written = 0 # track the number of written bytes
                int_num_records_written = 0 # track the number of records written to the output file
                # methods and variables for handling metadata
                int_total_weight_current_batch = 0
                l_int_entry_current_batch = [ ]
                def __process_batch__( file_output, file_index_output, int_num_bytes_written, int_num_records_written, l_int_entry_current_batch ) :
                    ''' # 2022-05-08 13:19:07 
                    process the current batch and return updated 'int_num_bytes_written' and 'int_num_records_written'
                    '''
                    # retrieve the number of index_entries
                    int_num_entries = len( l_int_entry_current_batch )
                    # handle invalid inputs
                    if int_num_entries == 0 :
                        return int_num_bytes_written, int_num_records_written

                    # retrieve data for the current batch
                    arr_int_feature, arr_int_barcode, arr_value = rtx[ l_int_entry_current_batch ]
                    # renumber indices of sorted entries to match that in the sorted matrix

                    # retrieve the start of the block, marked by the change of int_entry 
                    l_pos_start_block = [ 0 ] + list( np.where( np.diff( arr_int_feature if rtx.flag_ramtx_sorted_by_id_feature else arr_int_barcode ) )[ 0 ] + 1 ) + [ len( arr_value ) ] # np.diff decrease the index of entries where change happens, and +1 should be done # 10X matrix data: row = feature, col = barcodes
                    # prepare
                    
                    for index_block in range( len( l_pos_start_block ) - 1 ) : # for each block (each block contains records of a single entry)
                        slice_for_the_current_block = slice( l_pos_start_block[ index_block ], l_pos_start_block[ index_block + 1 ] )
                        arr_int_feature_of_the_current_block, arr_int_barcode_of_the_current_block, arr_value_of_the_current_block = arr_int_feature[ slice_for_the_current_block ], arr_int_barcode[ slice_for_the_current_block ], arr_value[ slice_for_the_current_block ] # retrieve data for the current block

                        ''' iterate through each record, and apply function to each record '''
                        l_int_feature_of_the_current_block, l_int_barcode_of_the_current_block, l_value_of_the_current_block = [ ], [ ], [ ] # collect the result values
                        int_entry = None # initialize the int_entry
                        for int_feature, int_barcode, value in zip( arr_int_feature_of_the_current_block, arr_int_barcode_of_the_current_block, arr_value_of_the_current_block ) :
                            ''' filter the record based on the given 'ba_mask_feature' and 'ba_mask_barcode' '''
                            if ( ba_mask_feature is not None ) and ( not ba_mask_feature[ int_feature ] ) :
                                continue
                            elif ( ba_mask_barcode is not None ) and ( not ba_mask_barcode[ int_barcode ] ) :
                                continue

                            """ apply function (filter) to each record  """
                            int_barcode_new, int_feature_new, value_new = func( self, int_barcode, int_feature, value )
                            if int_entry is None : # retrieve the int_entry
                                int_entry = int_feature_new if rtx.flag_ramtx_sorted_by_id_feature else int_barcode_new # retrieve int_entry of the current block # row = feature, col = barcode

                            ''' convert output values to a line in the output file '''
                            if value_new is not None :
                                l_int_feature_of_the_current_block.append( int_feature_new )
                                l_int_barcode_of_the_current_block.append( int_barcode_new )
                                l_value_of_the_current_block.append( value_new )
                                
                        int_num_records_written_for_the_current_entry = len( l_value_of_the_current_block )    
                        if int_num_records_written_for_the_current_entry > 0 : # if current entry contains valid record(s)
                            # convert list of values to numpy arrays
                            arr_int_feature_of_the_current_block_new = np.array( l_int_feature_of_the_current_block, dtype = arr_int_feature_of_the_current_block.dtype )
                            arr_int_barcode_of_the_current_block_new = np.array( l_int_barcode_of_the_current_block, dtype = arr_int_barcode_of_the_current_block.dtype )
                            arr_value_of_the_current_block_new = np.array( l_value_of_the_current_block, dtype = arr_value_of_the_current_block.dtype if flag_dtype_output is None else flag_dtype_output ) # by default, use the datatype of the current block

                            bytes_processed = func_arrays_mtx_to_processed_bytes( ( arr_int_feature_of_the_current_block_new, arr_int_barcode_of_the_current_block_new, arr_value_of_the_current_block_new ) ) # convert arrays_mtx_new to processed bytes
                            int_num_bytes_written_for_the_current_entry = len( bytes_processed ) # record the number of bytes of the written data
                            # write the processed bytes to the output file
                            file_output.write( bytes_processed )

                            # write the index
                            file_index_output.write( ( '\t'.join( map( str, [ int_entry + 1, int_num_bytes_written, int_num_bytes_written + int_num_bytes_written_for_the_current_entry, int_num_records_written_for_the_current_entry ] ) ) + '\n' ).encode( ) ) # write an index for the current entry # 0>1 coordinate conversion for 'int_entry'

                            int_num_bytes_written += int_num_bytes_written_for_the_current_entry # update the number of bytes written
                            int_num_records_written += int_num_records_written_for_the_current_entry # update the number of records written

                    return int_num_bytes_written, int_num_records_written

                for int_entry, float_weight in zip( l_arr_int_entry_for_each_chunk[ index_chunk ], l_arr_weight_for_each_chunk[ index_chunk ] ) : # retrieve inputs for the current process
                    # add current index_sorting to the current batch
                    l_int_entry_current_batch.append( int_entry )
                    int_total_weight_current_batch += float_weight
                    # if the weight becomes larger than the threshold, process the batch and reset the batch
                    if int_total_weight_current_batch > int_total_weight_for_each_batch :
                        # process the current batch
                        int_num_bytes_written, int_num_records_written = __process_batch__( file_output, file_index_output, int_num_bytes_written, int_num_records_written, l_int_entry_current_batch )
                        # initialize the next batch
                        l_int_entry_current_batch = [ ]
                        int_total_weight_current_batch = 0

                # process the remaining entries
                int_num_bytes_written, int_num_records_written = __process_batch__( file_output, file_index_output, int_num_bytes_written, int_num_records_written, l_int_entry_current_batch )

                # close files
                for file in [ file_output, file_index_output ] :
                    file.close( )

                # record the number of records written for the current chunk
                q.put( int_num_records_written ) 

            q = mp.Queue( ) # multiprocessing queue is process-safe
            l_worker = list( mp.Process( target = __compress_and_index_a_portion_of_ramtx_as_a_worker__, args = ( index_chunk, q ) ) for index_chunk in range( int_num_threads ) )

            ''' start works and wait until all works are completed by workers '''
            for p in l_worker :
                p.start( ) # start workers
            for p in l_worker :
                p.join( )  

            ''' summarize output values '''
            # make sure the number of returned results match that of deployed workers (make sure that no threads have exited through an unexpected error)
            assert q.qsize( ) == len( l_worker ) 
            int_num_records_written = int( np.sum( list( q.get( ) for i in range( q.qsize( ) ) ) ) ) # retrieve the total number of written records

            ''' retrieve metadata '''
            # 10X matrix: row  = barcode, col = feature
            int_num_features = rtx._int_num_features if ba_mask_feature is None else ba_mask_feature.count( ) # retrieve the total number of features after filtering
            int_num_barcodes = rtx._int_num_barcodes if ba_mask_barcode is None else ba_mask_barcode.count( ) # retrieve the total number of barcodes after filtering
            int_num_records = int_num_records_written # retrieve the number of entries

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
                for path_file in df.path.values :
                    os.remove( path_file )
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
                df_index.columns = [ 'index_entry', 'int_pos_start', 'int_pos_end', 'int_num_records' ]
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
                for str_name_file_glob in [ 'indexed.*.mtx.idx.tsv.gz' ] :
                    for path_file in glob.glob( f'{path_folder_temp}{str_name_file_glob}' ) :
                        os.remove( path_file )

            ''' export features and barcodes '''
            # rename output files
            os.rename( f"{path_folder_temp}indexed.{str_etx}", f"{path_folder_ramtx_output}matrix.{str_etx}" )
            os.rename( f"{path_folder_temp}indexed.{str_etx}.{str_ext_index}", f"{path_folder_ramtx_output}matrix.{str_etx}.{str_ext_index}" )
                      
            # delete temporary folder
            if not flag_debugging :
                shutil.rmtree( path_folder_temp ) 

            ''' export settings used for sort, indexing, and exporting '''
            dict_metadata = { 
                'path_folder_mtx_10x_input' : None,
                'flag_ramtx_sorted_by_id_feature' : rtx.flag_ramtx_sorted_by_id_feature,
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
                
        """
        Create output folders and copy feature and barcode files
        """
        # apply the given function to each RAMtx object
        path_folder_data_new = f"{path_folder_ramdata_output}{name_data_new}/" # compose the output directory of the paird RAMtx objects inside the output RamData object
        for path_folder_ramtx_output in [ f"{path_folder_data_new}sorted_by_feature/", f"{path_folder_data_new}sorted_by_barcode/" ] :
            # create an ramtx output folder
            os.makedirs( path_folder_ramtx_output, exist_ok = True )

            ''' output copy features and barcode files of the input RamData object to the RAMtx output object without modification '''
            for name_file in [ 'features.tsv.gz', 'barcodes.tsv.gz' ] :
                # if the feature and barcode files does not exist in the root directory of the input RamData, copy the files from the input RAMtx object.
                if not os.path.exists( f"{self.path_folder_ramdata}{name_file}" ) :
                    OS_Run( [ 'cp', f"{self.ramtx_for_feature.path_folder_mtx}{name_file}", f"{self.path_folder_ramdata}{name_file}" ] ) # copy the feature and barcode files of tje currently active RAMtx # using shell program to speed up the process 
                OS_Run( [ 'cp', f"{self.path_folder_ramdata}{name_file}", f"{path_folder_ramtx_output}{name_file}" ] )
                
        if flag_simultaneous_processing_of_paired_ramtx :
            l_process = list( mp.Process( target = RAMtx_Apply, args = ( self, rtx, path_folder_ramtx_output, func, flag_output_value_is_float, file_format, int_num_digits_after_floating_point_for_export, flag_dtype_output, int_num_threads_for_the_current_process, ba_mask_barcode, ba_mask_feature ) ) for rtx, path_folder_ramtx_output, int_num_threads_for_the_current_process in zip( [ self.ramtx_for_barcode, self.ramtx_for_feature ], [ f"{path_folder_data_new}sorted_by_barcode/", f"{path_folder_data_new}sorted_by_feature/" ], [ int( np.floor( int_num_threads / 2 ) ), int( np.ceil( int_num_threads / 2 ) ) ] ) )
            for p in l_process : p.start( )
            for p in l_process : p.join( )
        else :
            RAMtx_Apply( self, self.ramtx_for_feature, f"{path_folder_data_new}sorted_by_feature/", func = func, flag_output_value_is_float = flag_output_value_is_float, file_format = file_format, int_num_digits_after_floating_point_for_export = int_num_digits_after_floating_point_for_export, flag_dtype_output = flag_dtype_output, int_num_threads = int_num_threads, ba_mask_barcode = ba_mask_barcode, ba_mask_feature = ba_mask_feature )
            RAMtx_Apply( self, self.ramtx_for_barcode, f"{path_folder_data_new}sorted_by_barcode/", func = func, flag_output_value_is_float = flag_output_value_is_float, file_format = file_format, int_num_digits_after_floating_point_for_export = int_num_digits_after_floating_point_for_export, flag_dtype_output = flag_dtype_output, int_num_threads = int_num_threads, ba_mask_barcode = ba_mask_barcode, ba_mask_feature = ba_mask_feature )

        if self.verbose :
            print( f'new data {name_data_new} has been successfully added' )
        # update 'set_name_data'
        self.set_name_data.add( name_data_new )
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
    def _further_summarize_scarab_output_for_filtering_( self, name_data = 'raw', name_adata = 'main', flag_show_graph = True ) :
        """ # 2022-06-06 01:01:47 
        (1) calculate the total count in gex mode
        (2) calculate_proportion_of_promoter_in_atac_mode
        assumming Scarab output and the output of 'sum_scarab_feature_category', calculate the ratio of counts of promoter features to the total counts in atac mode.

        'name_data' : name of the data from which scarab_feature_category summary was generated. by default, 'raw'
        'name_adata' : name of the AnnData of the current RamData object. by default, 'main'
        """
        df = self.ad[ name_adata ].obs # retrieve data of the given AnnData


        # calculate gex metrics
        df[ f'{name_data}_sum_for_gex_mode' ] = df[ Search_list_of_strings_with_multiple_query( df.columns, f'{name_data}_sum__', '-atac_mode' ) ].sum( axis = 1 ) # calcualte sum for gex mode outputs

        # calcualte atac metrics
        if f'{name_data}_sum___category_detailed___atac_mode' in df.columns.values : # check whether the ATAC mode has been used in the scarab output
            df[ f'{name_data}_sum_for_atac_mode' ] = df[ Search_list_of_strings_with_multiple_query( df.columns, f'{name_data}_sum__', 'atac_mode' ) ].sum( axis = 1 ) # calcualte sum for atac mode outputs
            df[ f'{name_data}_sum_for_promoter_atac_mode' ] = df[ list( Search_list_of_strings_with_multiple_query( df.columns, f'{name_data}_sum___category_detailed___promoter', 'atac_mode' ) ) ].sum( axis = 1 )
            df[ f'{name_data}_proportion_of_promoter_in_atac_mode' ] = df[ f'{name_data}_sum_for_promoter_atac_mode' ] / df[ f'{name_data}_sum_for_atac_mode' ] # calculate the proportion of reads in promoter
            df[ f'{name_data}_proportion_of_promoter_and_gene_body_in_atac_mode' ] = ( df[ f'{name_data}_sum_for_promoter_atac_mode' ] + df[ f'{name_data}_sum___category_detailed___atac_mode' ] ) / df[ f'{name_data}_sum_for_atac_mode' ] # calculate the proportion of reads in promoter

            # show graphs
            if flag_show_graph :
                MPL_Scatter_Align_Two_Series( df[ f'{name_data}_sum_for_atac_mode' ], df[ f'{name_data}_sum_for_gex_mode' ], x_scale = 'log', y_scale = 'log', alpha = 0.005 )
                MPL_Scatter_Align_Two_Series( df[ f'{name_data}_sum_for_atac_mode' ], df[ f'{name_data}_proportion_of_promoter_in_atac_mode' ], x_scale = 'log', alpha = 0.01 )

        self.ad[ name_adata ].obs = df # save the result        
    def subset( self, path_folder_ramdata_output, set_str_barcode = None, set_str_feature = None, int_num_threads = None, flag_simultaneous_processing_of_paired_ramtx = True, ** args ) :
        ''' # 2022-06-01 15:01:30 
        this function will create a new RamData object on disk by creating a subset of the current RamData given by the 'set_str_barcode' and 'set_str_feature'

        example usage: 
            1) in preprocessing step. removing droplets and low-quality cells from RamData before starting the analysis
            2) subclustering of a specific subset of cells for more detailed analysis.

        =========
        inputs 
        =========

        'path_folder_ramdata_output' : (Default: store inside the current RamData). The directory of the RamData object that will contain the outputs (paired RAMtx objects). if integer representations of features and barcodes are updated from filtering (either 'RamData.set_int_barcode' or 'RamData.set_int_feature' is not None), the output RAMtx is now incompatible with the current RamData and should be stored outside the current RamData object. The output directory of the new RamData object can be given through this argument. The RamData object directory should contains new features.tsv.gz and barcodes.tsv.gz (with updated integer representation of features and barcodes)
        'flag_simultaneous_processing_of_paired_ramtx' : (Default: True) process the paired RAMtx simultaneously using two processes at a time.
        'set_str_barcode', 'set_str_feature' : the sets of string representation of barcodes and features to be included in the output RamData. the barcodes that are exist in the given RamData object but absent in the 'set_str_barcode' will be deleted from the list (similar for 'set_str_feature').
        'int_num_threads' : the number of CPUs to use. by default, the number of CPUs set by the RamData attribute 'int_num_cpus' will be used.
        '''

        ''' handle inputs '''
        # check invalid input
        if path_folder_ramdata_output == self.path_folder_ramdata :
            if self.verbose :
                print( f'the output RamData object directory is exactly same that of the current RamData object, exiting' )
        #     return -1
        if set_str_barcode is None and set_str_feature is None :
            if self.verbose :
                print( f"no filtering is needed since 'set_str_barcode' and 'set_str_feature' is None, exiting" )
        #     return -1
        # create the RamData output folder
        os.makedirs( path_folder_ramdata_output, exist_ok = True ) 

        """ 
        Prepare data required for filtering
        """
        dict_name_file_to_dict_data = dict(  )
        for name_file, set_str_entry, int_num_entries in zip( [ 'features.tsv.gz', 'barcodes.tsv.gz' ], [ set_str_feature, set_str_barcode ], [ self._int_num_features, self._int_num_barcodes ] ) :
            ''' if current RamData object lack features or barcode files in its root folder, find  '''
            path_file_input = f"{self.path_folder_ramdata}{name_file}"
            path_file_output = f"{path_folder_ramdata_output}{name_file}"
            if not os.path.exists( path_file_input ) :
                l_path_file = glob.glob( f"{self.path_folder_ramdata}*/*/{name_file}" )
                assert len( l_path_file ) > 0 # there should be at least one available RAMtx object that contains features.tsv.gz or barcodes.tsv.gz
                OS_Run( [ 'cp', l_path_file[ 0 ], path_file_input ], stdout_binary = True )

            ''' retrieve required data for filtering for barcode/feature files '''
            # initialize the data dictionary
            dict_data = dict( )

            flag_is_filter_active = set_str_entry is not None # retrieve a flag indicating whether a filter is active
            ba_mask = bitarray( int_num_entries ) # initialize the bit array
            ba_mask.setall( 1 ) # by default, include all entries # 0 = absent, 1 = inclusion

            # read data
            arr = pd.read_csv( path_file_input, sep = '\t', header = None ).values

            if flag_is_filter_active :
                arr_int_entry_new = np.full( int_num_entries, - 1 ) # by default, all entries are excluded
                int_entry_new_current = 0 # new entry_new starts from 0
                for int_entry_prev, str_entry in enumerate( arr[ :, 0 ] ) :
                    if str_entry not in set_str_entry :
                        ba_mask[ int_entry_prev ] = 0 # modify bitarray to set the current entry will be excluded
                    else :
                        # update tne int_entry_new
                        arr_int_entry_new[ int_entry_prev ] = int_entry_new_current
                        int_entry_new_current += 1
                dict_data[ 'arr_int_entry_new' ] = arr_int_entry_new

                pd.DataFrame( arr[ arr_int_entry_new != -1 ] ).to_csv( path_file_output, sep = '\t', index = False, header = False )
            else :
                OS_Run( [ 'cp', path_file_input, path_file_output ], stdout_binary = True ) # if there is no filtering, use the input file as the output file as-is.

            dict_data[ 'flag_is_filter_active' ] = flag_is_filter_active
            dict_data[ 'ba_mask' ] = ba_mask
            dict_name_file_to_dict_data[ name_file ] = dict_data


        """
        Set a filtering function based on the settings
        """
        flag_is_filter_active_barcode, flag_is_filter_active_feature = dict_name_file_to_dict_data[ 'barcodes.tsv.gz' ][ 'flag_is_filter_active' ], dict_name_file_to_dict_data[ 'features.tsv.gz' ][ 'flag_is_filter_active' ]
        ba_mask_barcode, ba_mask_feature = dict_name_file_to_dict_data[ 'barcodes.tsv.gz' ][ 'ba_mask' ], dict_name_file_to_dict_data[ 'features.tsv.gz' ][ 'ba_mask' ]
        arr_int_barcode_new = dict_name_file_to_dict_data[ 'barcodes.tsv.gz' ][ 'arr_int_entry_new' ] if flag_is_filter_active_barcode else None 
        arr_int_features_new = dict_name_file_to_dict_data[ 'features.tsv.gz' ][ 'arr_int_entry_new' ] if flag_is_filter_active_feature else None 

        # retrieve a subset the main AnnData object based on filtering settings
        if flag_is_filter_active_barcode and flag_is_filter_active_feature :
            adata_subset = self.adata[ arr_int_barcode_new != -1, arr_int_features_new != -1 ].copy( ) 
            def func_subset( self, int_barcode, int_feature, value ) :
                return arr_int_barcode_new[ int_barcode ], arr_int_features_new[ int_feature ], value # renumber both int_barcode and int_feature
        elif flag_is_filter_active_barcode and not flag_is_filter_active_feature :
            adata_subset = self.adata[ arr_int_barcode_new != -1 ].copy( )
            def func_subset( self, int_barcode, int_feature, value ) :
                return arr_int_barcode_new[ int_barcode ], int_feature, value # renumber of int_barcode
        elif not flag_is_filter_active_barcode and flag_is_filter_active_feature :
            adata_subset = self.adata[ :, arr_int_features_new != -1 ].copy( )
            def func_subset( self, int_barcode, int_feature, value ) :
                return int_barcode, arr_int_features_new[ int_feature ], value # renumber int_feature
        else :
            return - 1
        adata_subset.X = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( ( [], ( [], [] ) ), shape = ( len( adata_subset.obs ), len( adata_subset.var ) ) ) ) # store the empty sparse matrix
        adata_subset.write( f'{path_folder_ramdata_output}main.h5ad' ) # save the resulting AnnData object

        ''' filter the RAMtx matrices '''
        for name_data in self.set_name_data :
            self.apply( name_data, name_data_new = None, func = func_subset, path_folder_ramdata_output = path_folder_ramdata_output, flag_dtype_output = None, flag_output_value_is_float = None, flag_simultaneous_processing_of_paired_ramtx = flag_simultaneous_processing_of_paired_ramtx, ba_mask_barcode = ba_mask_barcode, ba_mask_feature = ba_mask_feature, int_num_threads = int_num_threads, ** args ) # flag_dtype_output = None : use the same dtype as the input RAMtx object
    def _filter_cell_scarab_output_( self, path_folder_ramdata_output, name_data = 'raw', name_adata = 'main', int_min_sum_for_atac_mode = 1500, float_min_proportion_of_promoter_in_atac_mode = 0.22, int_min_sum_for_gex_mode = 250 ) :
        ''' # 2022-06-03 15:25:02 
        filter cells from scarab output 

        'path_folder_ramdata_output' : output directory
        '''
        df = self.ad[ name_adata ].obs # retrieve data of the given AnnData

        # retrieve barcodes for filtering
        set_str_barcode = df[ ( df[ f'{name_data}_sum_for_atac_mode' ] >= int_min_sum_for_atac_mode ) & ( df[ f'{name_data}_proportion_of_promoter_in_atac_mode' ] >= float_min_proportion_of_promoter_in_atac_mode ) & ( df[ f'{name_data}_sum_for_gex_mode' ] >= int_min_sum_for_gex_mode ) ].index.values
        # subset the current RamData for valid cells
        self.subset( path_folder_ramdata_output, set_str_barcode = set_str_barcode )
        if self.verbose :
            print( f'cell filtering completed for {len( set_str_barcode )} cells. A filtered RamData was exported at {path_folder_ramdata_output}' )
    def _normalize_scarab_output_( self, name_data = 'raw', name_data_new = 'normalized', name_adata = 'main', name_col_total_count_atac_mode = 'raw_sum_for_atac_mode', name_col_total_count_gex_mode = 'raw_sum_for_gex_mode', int_total_count_target_gex_mode = 10000, int_total_count_target_atac_mode = 30000, flag_dtype_output = np.float64, flag_simultaneous_processing_of_paired_ramtx = True, int_num_threads = None, ** args ) :
        ''' # 2022-06-06 02:36:49 
        this function perform normalization of a given data and will create a new data in the current RamData object.

        =========
        inputs 
        =========

        'name_data' : name of input data
        'name_data_new' : name of the output (normalized) data
        'name_adata' : name of the anndata to retrieve total counts of each cell
        'name_col_total_count_atac_mode' : name of column of the given anndata containing total counts of cells in atac mode
        'name_col_total_count_gex_mode' : name of column of the given anndata containing total counts of cells in gex mode
        'flag_simultaneous_processing_of_paired_ramtx' : (Default: True) process the paired RAMtx simultaneously using two processes at a time.
        'int_num_threads' : the number of CPUs to use. by default, the number of CPUs set by the RamData attribute 'int_num_cpus' will be used.
        'int_total_count_target_gex_mode' = 10000 : 
        'int_total_count_target_atac_mode' = 30000, :
        'flag_dtype_output' = np.float64 : the 
        '''
        
        ''' handle inputs '''
        
        """ 
        Prepare data for normalization
        """
        
        self._classify_feature_of_scarab_output_( ) # classify features using the default settings
        ba_mask_feature_of_atac_mode = self._dict_data_for_feature_classification[ "ba_mask_feature_of_atac_mode" ] # retrieve mask for features of atac mode
        
        # classify inputs
        df_obs = self.ad[ name_adata ].obs
        flag_atac_mode = name_col_total_count_atac_mode in df_obs.columns.values
        flag_gex_mode = name_col_total_count_gex_mode in df_obs.columns.values
        
        # retrieve normalization factors for atac/gex counts for each cell
        if flag_atac_mode :
            arr_normalization_factor_atac_mode = int_total_count_target_atac_mode / self.ad[ name_adata ].obs[ name_col_total_count_atac_mode ].values
        if flag_gex_mode :
            arr_normalization_factor_gex_mode = int_total_count_target_gex_mode / self.ad[ name_adata ].obs[ name_col_total_count_gex_mode ].values

        # retrieve normalization function for each input classification
        if flag_atac_mode and flag_gex_mode :
            def func_norm( self, int_barcode, int_feature, value ) :
                return int_barcode, int_feature, ( arr_normalization_factor_atac_mode[ int_barcode ] * value if ba_mask_feature_of_atac_mode[ int_feature ] else arr_normalization_factor_gex_mode[ int_barcode ] * value )
        elif flag_atac_mode :
            def func_norm( self, int_barcode, int_feature, value ) :
                return int_barcode, int_feature, arr_normalization_factor_atac_mode[ int_barcode ] * value
        elif flag_gex_mode :
            def func_norm( self, int_barcode, int_feature, value ) :
                return int_barcode, int_feature, arr_normalization_factor_gex_mode[ int_barcode ] * value
        else :
            if self.verbose :
                print( 'invalid column names for total atac/gex counts for each cells' )
            return -1 

        ''' normalize the RAMtx matrices '''
        self.apply( name_data, name_data_new = name_data_new, func = func_norm, flag_dtype_output = flag_dtype_output, flag_output_value_is_float = True, flag_simultaneous_processing_of_paired_ramtx = flag_simultaneous_processing_of_paired_ramtx, int_num_threads = int_num_threads, ** args ) # flag_dtype_output = None : use the same dtype as the input RAMtx object
    def _identify_highly_variable_features_scarab_output_( self, name_adata = 'main', name_data = 'normalized_log1p', flag_show_graph = True ) :
        """ # 2022-06-07 22:53:55 
        identify highly variable features for scarab output (multiome)
        learns mean-variable relationship separately for gex and atac results

        'flag_show_graph' : show graphs

        ==========
        returns
        ==========

        f'{name_data}__float_score_highly_variable_feature_from_mean' : the more positive this value is, the feature is likely to be highly variable and contains more information about the cellular heterogeneity of the dataset
        """
        # set the name of the columns that will be used in the current method
        name_col_for_mean, name_col_for_variance = f'{name_data}_mean', f'{name_data}_variance'

        # classify features
        self._classify_feature_of_scarab_output_( )

        # retrieve data from the given 'name_adata'
        df_var = self.ad[ name_adata ].var
        df_data = df_var[ [ name_col_for_mean, name_col_for_variance ] ]

        arr_mask_feature_of_atac_mode = np.array( list( e == 1 for e in self._dict_data_for_feature_classification[ 'ba_mask_feature_of_atac_mode' ] ) ) # retrieve a mask indicating whether a given feature is from the atac mode
        df_var[ 'feature_of_atac_mode' ] = arr_mask_feature_of_atac_mode # add mask to the dataframe
        df_var[ f'{name_data}__float_ratio_of_variance_to_expected_variance_from_mean' ] = np.nan # initialize the data
        df_var[ f'{name_data}__float_diff_of_variance_to_expected_variance_from_mean' ] = np.nan # initialize the data

        for flag_atac_mode in [ True, False ] :
            arr_mask = arr_mask_feature_of_atac_mode if flag_atac_mode else ( ~ arr_mask_feature_of_atac_mode ) # retrieve mask for atac/gex mode

            df = df_data[ arr_mask ].dropna( ) # retrieve data for gex or atac mode separately
            if len( df ) == 0 : # skip 
                continue

            # learn mean-variance relationship for the data
            arr_mean, arr_var = df[ name_col_for_mean ].values, df[ name_col_for_variance ].values

            if flag_show_graph :
                plt.plot( arr_mean[ : : 10 ], arr_var[ : : 10 ], '.', alpha = 0.01 )
                MATPLOTLIB_basic_configuration( x_scale = 'log', y_scale = 'log', x_label = 'mean', y_label = 'variance', title = f"({'ATAC' if flag_atac_mode else 'GEX'} mode) mean-variance relationship\nin '{name_data}' in the '{name_adata}' AnnDAta" )
                plt.show( )
            mean_var_relationship_fit = np.polynomial.polynomial.Polynomial.fit( arr_mean, arr_var, 2 )

            # calculate the deviation from the estimated variance from the mean for the current mode
            arr_data = df_data[ arr_mask ].values
            arr_ratio_of_variance_to_expected_variance_from_mean = np.full( len( arr_data ), np.nan )
            arr_diff_of_variance_to_expected_variance_from_mean = np.full( len( arr_data ), np.nan )

            for i in range( len( arr_data ) ) : # iterate list of means of the features
                mean, var = arr_data[ i ] # retrieve var and mean
                if not np.isnan( var ) : # if current entry is valid
                    var_expected = mean_var_relationship_fit( mean ) # calculate expected variance from the mean
                    if var_expected == 0 : # handle the case when the current expected variance is zero 
                        arr_ratio_of_variance_to_expected_variance_from_mean[ i ] = np.nan
                        arr_diff_of_variance_to_expected_variance_from_mean[ i ] = np.nan
                    else :
                        arr_ratio_of_variance_to_expected_variance_from_mean[ i ] = var / var_expected
                        arr_diff_of_variance_to_expected_variance_from_mean[ i ] = var - var_expected

            # add data to var dataframe
            df_var.loc[ arr_mask, f'{name_data}__float_ratio_of_variance_to_expected_variance_from_mean' ] = arr_ratio_of_variance_to_expected_variance_from_mean
            df_var.loc[ arr_mask, f'{name_data}__float_diff_of_variance_to_expected_variance_from_mean' ] = arr_diff_of_variance_to_expected_variance_from_mean

        # calculate the product of the ratio and difference of variance to expected variance for scoring and sorting highly variable features
        df_var[ f'{name_data}__float_score_highly_variable_feature_from_mean' ] = df_var[ f'{name_data}__float_ratio_of_variance_to_expected_variance_from_mean' ] * df_var[ f'{name_data}__float_diff_of_variance_to_expected_variance_from_mean' ]

        # save result in the given AnnData
        self.ad[ name_adata ].var = df_var
    def umap( self, name_adata, l_str_feature, name_adata_new = None, l_name_col_for_regression = [ ], float_scale_max_value = 10, int_pca_n_comps = 150, int_neighbors_n_neighbors = 10, int_neighbors_n_pcs = 50, dict_kw_umap = dict( ), dict_leiden = dict( ) ) :
        ''' # 2022-06-06 02:32:29 
        using the given AnnData 'name_adata', retrieve all count data of given list of features 'l_str_feature', perform dimension reduction process, and save the new AnnData with umap coordinate and leiden cluster information 'name_adata_new'

        'l_name_col_for_regression' : a list of column names for the regression step. a regression step is often the longest step for demension reduction and clustering process. By simply skipping this step, one can retrieve a brief cluster structures of the cells in a very short time. to skip the regression step, please set this argument to an empty list [ ].
        '''
        # retrieve all count data of 'l_str_feature'
        X_temp = self.ad[ name_adata ].X
        self.ad[ name_adata ].X = self[ l_str_feature ].X # retrieve all data of the features classified as 'highly variable', and start performing clustering
        adata = self.ad[ name_adata ]
        adata = adata[ :, np.array( adata.X.sum( axis = 0 ) ).ravel( ) > 0 ].copy( ) # reduce the number of features in the sparse matrix before converting an entire matrix to dense format
        self.ad[ name_adata ].X = X_temp # restore the 'X' attribute values

        # initialize the new AnnData 'name_adata_new'
        if name_adata_new is None :
            name_adata_new = name_adata
        if name_adata != name_adata_new :
            self.ad[ name_adata_new ] = self.ad[ name_adata ].copy( )

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

        # export results to the new AnnData object ('name_adata_new')
        self.ad.retrieve_attributes( name_adata_new, adata, flag_ignore_var = True )