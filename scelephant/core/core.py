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
        flat_axis_initialized = False # initialize the flag to False 
        for index_chunk, df in enumerate( pd.read_csv( f"{path_folder_mtx_10x_input}{name_axis}.tsv.gz", sep = '\t', header = None, chunksize = int_num_of_entries_in_a_chunk_metadata ) ) : # read chunk by chunk
            if not flat_axis_initialized :
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
                
                flat_axis_initialized = True # set the flag to True
                
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
    """ # 2022-07-30 15:06:08 
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
        'version' : _version_,
    }
    # write layer metadata
    lay = zarr.group( path_folder_ramdata_layer )
    lay.attrs[ 'dict_metadata' ] = { 
        'set_modes' : list( set_modes ) + ( [ 'dense_for_querying_barcodes', 'dense_for_querying_features' ] if 'dense' in set_modes else [ ] ), # dense ramtx can be operated for querying either barcodes/features
        'version' : _version_,
    }
    
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
    za_sink = zarr.open( path_folder_zarr_sink, mode = 'w', shape = za.shape, chunks = za.chunks, dtype = za.dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # open the output zarr
    
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
    """ # 2022-07-22 02:05:09 
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
    'flag_use_mask_for_caching' : use mask for not only storing modifications, but also save retrieved data from (remote) sources for faster access. this behavior can be turned on/off at any time
    """
    def __init__( self, path_folder_zdf, df = None, int_num_rows = None, int_num_rows_in_a_chunk = 10000, ba_filter = None, flag_enforce_name_col_with_only_valid_characters = False, flag_store_string_as_categorical = True, flag_retrieve_categorical_data_as_integers = False, flag_load_data_after_adding_new_column = True, mode = 'a', path_folder_mask = None, flag_is_read_only = False, flag_use_mask_for_caching = True ) :
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
        self.filter = ba_filter
        
        # open or initialize zdf and retrieve associated metadata
        if not zarr_exists( path_folder_zdf ) : # if the object does not exist, initialize ZarrDataFrame
            # create the output folder
            os.makedirs( path_folder_zdf, exist_ok = True )
            
            self._root = zarr.open( path_folder_zdf, mode = 'a' )
            self._dict_metadata = { 'version' : _version_, 'columns' : set( ), 'int_num_rows_in_a_chunk' : int_num_rows_in_a_chunk, 'flag_enforce_name_col_with_only_valid_characters' : flag_enforce_name_col_with_only_valid_characters, 'flag_store_string_as_categorical' : flag_store_string_as_categorical  } # to reduce the number of I/O operations from lookup, a metadata dictionary will be used to retrieve/update all the metadata
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
            self._mask = ZarrDataFrame( path_folder_mask, df = df, int_num_rows = self.n_rows, int_num_rows_in_a_chunk = self.metadata[ 'int_num_rows_in_a_chunk' ], ba_filter = ba_filter, flag_enforce_name_col_with_only_valid_characters = self.metadata[ 'flag_enforce_name_col_with_only_valid_characters' ], flag_store_string_as_categorical = self.metadata[ 'flag_store_string_as_categorical' ], flag_retrieve_categorical_data_as_integers = flag_retrieve_categorical_data_as_integers, flag_load_data_after_adding_new_column = flag_load_data_after_adding_new_column, mode = 'a', path_folder_mask = None, flag_is_read_only = False ) # the mask ZarrDataFrame shoud not have mask, should be modifiable, and not mode == 'r'.
        
        # handle input arguments
        self._str_invalid_char = '! @#$%^&*()-=+`~:;[]{}\|,<.>/?' + '"' + "'" if self._dict_metadata[ 'flag_enforce_name_col_with_only_valid_characters' ] else '/' # linux file system does not allow the use of linux'/' character in the folder/file name
        
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
        return self._dict_metadata
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
            if 'int_num_rows' not in self._dict_metadata : # if 'int_num_rows' has not been set, set 'int_num_rows' using the length of the filter bitarray
                self._dict_metadata[ 'int_num_rows' ] = len( ba_filter )
                self._save_metadata_( ) # save metadata
            else :
                # check the length of filter bitarray
                assert len( ba_filter ) == self._dict_metadata[ 'int_num_rows' ]

            self._loaded_data = dict( ) # empty the cache
            self._initialize_temp_folder_( ) # empty the temp folder
            self.dict = dict( ) # empty the cache for columns stored as dictionaries
            self._n_rows_after_applying_filter = ba_filter.count( ) # retrieve the number of rows after applying the filter

            self._ba_filter = ba_filter # set bitarray filter
        # set filter of mask
        if hasattr( self, '_mask' ) and self._mask is not None : # propagate filter change to the mask ZDF
            self._mask.filter = ba_filter
    def __getitem__( self, args ) :
        ''' # 2022-07-22 02:03:32 
        retrieve data of a column.
        partial read is allowed through indexing (slice/integer index/boolean mask/bitarray is supported)
        when a filter is active, the filtered data will be cached in the temporary directory as a Zarr object and will be retrieved in subsequent accesses
        if mask is set, retrieve data from the mask if the column is available in the mask. 
        also, when the 'flag_use_mask_for_caching' setting is active, use mask for caching data from source data (possibly remote source).
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
        if name_col not in self : # if name_col is not valid (name_col does not exists in current ZDF, including the mask), exit by returning None
            return None
        if self._mask is not None : # if mask is available
            if self.flag_use_mask_for_caching and name_col not in self._mask : # if 'flag_use_mask_for_caching' option is active and the column is not available in the mask, copy the column from the source to the mask
                zarr_copy( f"{self._path_folder_zdf}{name_col}/", f"{self._mask._path_folder_zdf}{name_col}/" ) # copy zarr object from the source to the mask
            if name_col in self._mask : # if 'name_col' is available in the mask, retrieve data from the mask.
                return self._mask[ args ]
        if name_col in self : # if name_col is valid
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
                            za_cached = zarr.open( path_folder_temp_zarr, 'w', shape = ( self._n_rows_after_applying_filter, ), chunks = ( self._dict_metadata[ 'int_num_rows_in_a_chunk' ], ), dtype = za.dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # open a new Zarr object for caching # overwriting existing data # use the same dtype as the parent dtype
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
        # if mask is available, save new data/modify existing data to the mask
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
            self._dict_metadata[ 'int_num_rows' ] = int_num_values # record the number of rows of the dataframe
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
        if dtype is str and self._dict_metadata[ 'flag_store_string_as_categorical' ] : # storing categorical data            
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
            za = zarr.open( path_folder_col, mode = 'a', synchronizer = zarr.ThreadSynchronizer( ) ) if os.path.exists( path_folder_col ) else zarr.open( path_folder_col, mode = 'w', shape = ( self._n_rows_unfiltered, ), chunks = ( self._dict_metadata[ 'int_num_rows_in_a_chunk' ], ), dtype = dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # create a new Zarr object if the object does not exist.
            
            # if dtype changed from the previous zarr object, re-write the entire Zarr object with changed dtype. (this will happens very rarely, and will not significantly affect the performance)
            if dtype != za.dtype : # dtype should be larger than za.dtype if they are not equal (due to increased number of bits required to encode categorical data)
                print( f'{za.dtype} will be changed to {dtype}' )
                path_folder_col_new = f"{self._path_folder_zdf}{name_col}_{UUID( )}/" # compose the new output folder
                za_new = zarr.open( path_folder_col_new, mode = 'w', shape = ( self._n_rows_unfiltered, ), chunks = ( self._dict_metadata[ 'int_num_rows_in_a_chunk' ], ), dtype = dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # create a new Zarr object using the new dtype
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
            za = zarr.open( path_folder_col, mode = 'a', synchronizer = zarr.ThreadSynchronizer( ) ) if os.path.exists( path_folder_col ) else zarr.open( path_folder_col, mode = 'w', shape = ( self._n_rows_unfiltered, ), chunks = ( self._dict_metadata[ 'int_num_rows_in_a_chunk' ], ), dtype = dtype, synchronizer = zarr.ThreadSynchronizer( ) ) # create a new Zarr object if the object does not exist.
            
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
        if name_col not in self._dict_metadata[ 'columns' ] :
            self._dict_metadata[ 'columns' ].add( name_col )
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
            self._dict_metadata[ 'columns' ].remove( name_col )
            self._save_metadata_( ) # update metadata
            # delete the column from the disk ZarrDataFrame object
            shutil.rmtree( f"{self._path_folder_zdf}{name_col}/" ) #             OS_Run( [ 'rm', '-rf', f"{self._path_folder_zdf}{name_col}/" ] )
    def __repr__( self ) :
        """ # 2022-07-20 23:00:15 
        """
        return f"<ZarrDataFrame object stored at {self._path_folder_zdf}\n\twith the following columns: {sorted( self._dict_metadata[ 'columns' ] )}>"
    @property
    def columns( self ) :
        ''' # 2022-07-20 23:01:48 
        return available column names as a set
        '''
        if self._mask is not None : # if mask is available :
            return self._dict_metadata[ 'columns' ].union( self._mask._dict_metadata[ 'columns' ] ) # return the column names of the current ZDF and the mask ZDF
        else :
            return self._dict_metadata[ 'columns' ]
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
            temp = self._dict_metadata[ 'columns' ]
            self._dict_metadata[ 'columns' ] = list( temp )
            self._root.attrs[ 'dict_metadata' ] = self._dict_metadata # update metadata
            self._dict_metadata[ 'columns' ] = temp # revert 'columns' to set
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
            
''' a class for accessing Zarr-backed count matrix data (RAMtx, Random-Access matrix) '''
class RAMtx( ) :
    """ # 2022-07-31 00:50:03 
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
    
    """
    def __init__( self, path_folder_ramtx, ramdata = None, dtype_of_feature_and_barcode_indices = np.uint32, dtype_of_values = np.float64, int_num_cpus = 1, verbose = False, flag_debugging = False, mode = 'a', flag_is_read_only = False, path_folder_ramtx_mask = None, is_for_querying_features = True ) :
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
        
        # set filters using RamData
        self.ba_filter_features = ramdata.ft.filter if ramdata is not None else None
        self.ba_filter_barcodes = ramdata.bc.filter if ramdata is not None else None
        
        self.is_sparse = self.mode != 'dense' # retrieve a flag indicating whether ramtx is dense
        if self.is_sparse :
            self._is_for_querying_features = self._dict_metadata[ 'flag_ramtx_sorted_by_id_feature' ] # for sparse matrix, this attribute is fixed
            # open Zarr object containing matrix and matrix indices
            self._za_mtx_index = zarr.open( f'{self._path_folder_ramtx}matrix.index.zarr', 'r' )
            self._za_mtx = zarr.open( f'{self._path_folder_ramtx}matrix.zarr', 'r' )
        else :
            self.is_for_querying_features = is_for_querying_features # set this attribute
            self._za_mtx = zarr.open( f'{self._path_folder_ramtx}matrix.zarr', 'r' )
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
        """ # 2022-07-31 00:46:44 
        return a bitarray filter of the indexed axis where all the entries with valid count data is marked '1'
        """
        if not self.is_sparse :
            # currently not implemented
            # raise NotImplementedError( 'not supported for dense' )
            # if current ramtx is dense ramtx, assumes all entries are active
            ba = bitarray( self.len_axis_for_querying )
            ba.setall( 1 )
            return ba
        else : # for sparse matrix
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
                    za = zarr.open( f"{self._path_folder_ramtx_modifiable}matrix.index.active_entries.zarr/", mode = 'w', shape = ( self.len_axis_for_querying, ), chunks = ( int_size_chunk * int_num_chunks_in_a_batch, ), dtype = bool, synchronizer = zarr.ThreadSynchronizer( ) ) # the size of the chunk will be 100 times of the chunk used for matrix index, since the dtype is boolean
                    len_axis_for_querying = self.len_axis_for_querying
                    int_pos_start = 0
                    int_num_entries_to_retrieve = int( int_size_chunk * int_num_chunks_in_a_batch )
                    while int_pos_start < len_axis_for_querying :
                        sl = slice( int_pos_start, int_pos_start + int_num_entries_to_retrieve )
                        za[ sl ] = ( self._za_mtx_index[ sl ][ :, 1 ] - self._za_mtx_index[ sl ][ :, 0 ] ) > 0 # active entry is defined by finding entries with at least one count record
                        int_pos_start += int_num_entries_to_retrieve # update the position
                        
                self._n_active_entries = ba.count( ) # calculate the number of active entries

                # update metadata
                self._dict_metadata[ 'n_active_entries' ] = self._n_active_entries 
                self._save_metadata_( )

            ba = bk.BA.to_bitarray( za[ : ] ) # return the boolean array of active entries as a bitarray object
            return ba
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
        """ # 2022-07-31 00:37:59 
        Retrieve data of a given list of entries from RAMtx as lists of values and arrays (i.e. sparse matrix), each value and array contains data of a single 'int_entry' of the indexed axis
        '__getitem__' can be used to retrieve minimal number of values required to build a sparse matrix or dense matrix from it
        
        Returns:
        l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying, l_arr_value :
            'l_int_entry_of_axis_for_querying' only contains int_entry of valid entries
        """
        # initialize the output data structures
        l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying, l_arr_value = [ ], [ ], [ ]
        
        # wrap in a list if a single entry was queried
        if isinstance( l_int_entry, ( int, np.int64, np.int32, np.int16, np.int8 ) ) : # check whether the given entry is an integer
            l_int_entry = [ l_int_entry ]
        ''' retrieve filters '''
        is_for_querying_features = self.is_for_querying_features
        ba_filter_axis_for_querying, ba_filter_not_axis_for_querying = ( self.ba_filter_features, self.ba_filter_barcodes ) if is_for_querying_features else ( self.ba_filter_barcodes, self.ba_filter_features )
            
        ''' filter 'int_entry', if a filter has been set '''
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
            """ # 2022-07-02 12:35:44 
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
                fetch data from sparse ramtx
                """
                arr_index_of_a_batch = np.array( l_index_in_a_batch ) # convert index of the batch to a numpy array
                st_batch, en_batch = arr_index_of_a_batch[ 0, 0 ], arr_index_of_a_batch[ - 1, 1 ] # retrieve start and end positions of the current batch
                arr_int_entry_of_axis_not_for_querying, arr_value = self._za_mtx[ st_batch : en_batch ].T # fetch data from the Zarr object
                
                for int_entry, index in zip( l_int_entry_in_a_batch, arr_index_of_a_batch - st_batch ) : # substract the start position of the batch to retrieve the local index
                    st, en = index
                    sl = slice( st, en )
                    __process_entry( int_entry, arr_int_entry_of_axis_not_for_querying[ sl ], arr_value[ sl ] )
            def __fetch_from_dense_ramtx( l_int_entry_in_a_batch ) :
                """ # 2022-07-30 23:33:37 
                fetch data from dense ramtx
                """
                for int_entry, arr_data in zip( l_int_entry_in_a_batch, self._za_mtx.get_orthogonal_selection( ( slice( None, None ), l_int_entry_in_a_batch ) ).T if is_for_querying_features else self._za_mtx.get_orthogonal_selection( ( l_int_entry_in_a_batch, slice( None, None ) ) ) ) : # fetch data from the Zarr object and iterate through each entry and its data
                    arr_int_entry_of_axis_not_for_querying = np.where( arr_data )[ 0 ]
                    __process_entry( int_entry, arr_int_entry_of_axis_not_for_querying, arr_data[ arr_int_entry_of_axis_not_for_querying ] )
            
            ''' retrieve data '''
            if self.is_sparse : # handle sparse ramtx
                ''' sparse ramtx '''
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
                ''' dense ramtx '''
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
            l_pipes_from_main_process_to_worker = list( mp.Pipe( ) for _ in range( self.int_num_cpus ) ) # create pipes for sending records to workers # add process for receivers
            l_pipes_from_worker_to_main_process = list( mp.Pipe( ) for _ in range( self.int_num_cpus ) ) # create pipes for collecting results from workers
            l_processes = list( mp.Process( target = __retrieve_data, args = ( l_pipes_from_main_process_to_worker[ index_worker ][ 1 ], l_pipes_from_worker_to_main_process[ index_worker ][ 0 ] ) ) for index_worker in range( int_n_workers ) ) # add a process for distributing fastq records
            for p in l_processes :
                p.start( )
            # distribute works
            for index_worker, l_int_entry_for_each_worker in enumerate( LIST_Split( l_int_entry, int_n_workers, flag_contiguous_chunk = True ) ) : # no load balacing for now
                l_pipes_from_main_process_to_worker[ index_worker ][ 0 ].send( l_int_entry_for_each_worker )
            # wait until all works are completed
            int_num_workers_completed = 0
            while int_num_workers_completed < int_n_workers : # until all works are completed
                for _, pipe in l_pipes_from_worker_to_main_process :
                    if pipe.poll( ) :
                        otuput = pipe.recv( )
                        l_int_entry_of_axis_for_querying.extend( otuput[ 0 ] )
                        l_arr_int_entry_of_axis_not_for_querying.extend( otuput[ 1 ] )
                        l_arr_value.extend( otuput[ 2 ] )
                        del otuput
                        int_num_workers_completed += 1
                time.sleep( 0.1 )
            # dismiss workers once all works are completed
            for p in l_processes :
                p.join( )
        else : # single thread mode
            l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying, l_arr_value = __retrieve_data( l_int_entry, flag_as_a_worker = False )
        
        return l_int_entry_of_axis_for_querying, l_arr_int_entry_of_axis_not_for_querying, l_arr_value
    def get_sparse_matrix( self, l_int_entry, flag_return_as_arrays = False ) :
        """ # 2022-08-01 04:47:11 
        
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
        n_bc, n_ft = ( self._int_num_barcodes, self._int_num_features ) if self._ramdata is None else ( self._ramdata.bc.meta.n_rows, self._ramdata.ft.meta.n_rows ) # detect whether the current RAMtx has been attached to a RamData and retrieve the number of barcodes and features accordingly
        X = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( ( arr_value, ( arr_int_barcode, arr_int_feature ) ), shape = ( n_bc, n_ft ) ) ) # convert count data to a sparse matrix
        return X # return the composed sparse matrix 
    def batch_generator( self, ba = None, int_num_entries_for_each_weight_calculation_batch = 1000, int_total_weight_for_each_batch = 1000000, int_min_num_entries_in_a_batch_if_weight_not_available = 2000, int_chunk_size_for_checking_boundary = 2000 ) :
        ''' # 2022-07-12 22:05:27 
        generate batches of list of integer indices of the active entries in the given bitarray 'ba'. 
        Each bach has the following characteristics:
            monotonous: active entries in a batch are in an increasing order
            balanced: the total weight of a batch is around (but not exactly) 'int_total_weight_for_each_batch'
        
        'ba' : (default None) if None is given, self.ba_active_entries bitarray will be used.
        'int_min_num_entries_in_a_batch_if_weight_not_available' : for some reason (e.g. when ramtx is dense and yet the number of entries for each axis has not been calculated), return this number of entries in a batch
        'int_chunk_size_for_checking_boundary' : if this argument is given, each batch will respect the chunk boundary of the given chunk size so that different batches share the same 'chunk'.
        '''
        # set defaule arguments
        if ba is None :
            ba = self.ba_filter_axis_for_querying # if None is given, ba_filter of the currently indexed axis will be used.
            if ba is None : # if filter is not set or the current RAMtx has not been attached to a RamData object, use the active entries
                ba = self.ba_active_entries # if None is given, self.ba_active_entries bitarray will be used.
        # initialize
        # a namespace that can safely shared between functions
        ns = { 'int_accumulated_weight_current_batch' : 0, 'l_int_entry_current_batch' : [ ], 'l_int_entry_for_weight_calculation_batch' : [ ], 'index_chunk_end' : None }
        
        # internal setting
        flag_weight_not_available = not self.is_sparse # currently, weights will be not available for dense matrix
        
        def find_batch( ) :
            """ # 2022-07-03 22:11:06 
            retrieve indices of the current 'weight_current_batch', calculate weights, and yield a batch
            """
            ''' retrieve weights '''
            if flag_weight_not_available : # if weight is not available
                arr_weight = np.full( len( ns[ 'l_int_entry_for_weight_calculation_batch' ] ), int_total_weight_for_each_batch / int_min_num_entries_in_a_batch_if_weight_not_available ) # use a constant weight as a fallback
            else : # if weight is available
                # load weight for the batch
                st, en = self._za_mtx_index.get_orthogonal_selection( ns[ 'l_int_entry_for_weight_calculation_batch' ] ).T # retrieve start and end coordinates of the entries
                arr_weight = en - st # calculate weight for each entry
                del st, en
            
            ''' search for batch '''
            for int_entry, weight in zip( ns[ 'l_int_entry_for_weight_calculation_batch' ], arr_weight ) :
                if ns[ 'index_chunk_end' ] is not None and ns[ 'index_chunk_end' ] != int_entry // int_chunk_size_for_checking_boundary : # if the chunk boundary has been set and the boundary has reached
                    yield ns[ 'l_int_entry_current_batch' ] # return a batch
                    # initialize the next batch
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
class RamDataAxis( ) :
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
            ba = self._ramdata.layer.get_ramtx( flag_is_for_querying_features = self._name_axis == 'features' ).ba_active_entries
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
        self._int_num_cpus = int_num_cpus
        self._path_folder_ramdata_mask = path_folder_ramdata_mask
        if path_folder_ramdata_mask is not None : # set path to the mask of the layer if ramdata mask has been given
            self._path_folder_ramdata_layer_mask = f"{self._path_folder_ramdata_mask}{name_layer}/"
        self._flag_is_read_only = flag_is_read_only
        
        # read metadata
        self._root = zarr.open( self._path_folder_ramdata_layer, 'a' )
        self._dict_metadata = self._root.attrs[ 'dict_metadata' ] # retrieve the metadata 
        
        # retrieve filters from the axes
        ba_filter_features = ramdata.ft.filter if ramdata is not None else None
        ba_filter_barcodes = ramdata.bc.filter if ramdata is not None else None
        
        # load RAMtx objects without filters
        # define arguments for opening RAMtx objects
        dict_kwargs = {
            'ramdata' : ramdata, 
            'dtype_of_feature_and_barcode_indices' : dtype_of_feature_and_barcode_indices, 
            'dtype_of_values' : dtype_of_values, 
            'int_num_cpus' : int_num_cpus, 
            'verbose' : verbose, 
            'flag_debugging' : False, 
            'mode' : self._mode, 
            'path_folder_ramtx_mask' : f'{self._path_folder_ramdata_layer_mask}{mode}/' if self._mask_available else None, 
            'flag_is_read_only' : self._flag_is_read_only
        }
        for mode in self.modes : # iterate through each mode
            if 'dense_for_querying_' in mode :
                rtx = RAMtx( f'{self._path_folder_ramdata_layer}dense/', is_for_querying_features = mode.rsplit( 'dense_for_querying_', 1 )[ 1 ] == 'features', ** dict_kwargs ) # open dense ramtx in querying_features/querying_barcodes modes
            else :
                rtx = RAMtx( f'{self._path_folder_ramdata_layer}{mode}/', ** dict_kwargs )
            setattr( self, f"ramtx_{mode}", rtx ) # set ramtx as an attribute
        
        # set filters of the current layer
        self.ba_filter_features = ba_filter_features
        self.ba_filter_barcodes = ba_filter_barcodes
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
        return iter( list( self[ mode ] for mode in self.modes ) )
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
        return self.get_ramtx( self, flag_is_for_querying_features = flag_use_ramtx_for_querying_feature ) # retrieve ramtx
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
        if len( list( self.modes ) ) > 0 : # return any ramtx object as a fallback
            if self.verbose :
                print( 'returning any ramtx object. it may work or not ... ' )
            return self[ list( self.modes )[ 0 ] ]
        if self.verbose :
            print( f"current layer {self.name} does not contain any valid ramtx objects" )
        return None
    def __getitem__( self, mode ) :
        """ # 2022-07-30 18:44:49 
        """
        if mode in self : # if a given mode is available
            return getattr( self, f"ramtx_{mode}" ) # return the ramtx of the given mode

''' class for storing RamData '''
class RamData( ) :
    """ # 2022-07-21 23:32:59 
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
    
    ==== AnnDataContainer ====
    'flag_enforce_name_adata_with_only_valid_characters' : enforce valid characters in the name of AnnData
    """
    def __init__( self, path_folder_ramdata, name_layer = 'raw', int_num_cpus = 64, int_num_cpus_for_fetching_data = 5, dtype_of_feature_and_barcode_indices = np.int32, dtype_of_values = np.float64, int_index_str_rep_for_barcodes = 0, int_index_str_rep_for_features = 1, mode = 'a', path_folder_ramdata_mask = None, dict_kw_zdf = { 'flag_retrieve_categorical_data_as_integers' : False, 'flag_load_data_after_adding_new_column' : True, 'flag_enforce_name_col_with_only_valid_characters' : True }, dict_kw_view = { 'float_min_proportion_of_active_entries_in_an_axis_for_using_array' : 0.1, 'dtype' : np.int32 }, flag_enforce_name_adata_with_only_valid_characters = True, verbose = True, flag_debugging = False ) :
        """ # 2022-07-21 23:32:54 
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
        self._int_num_cpus_for_fetching_data = int_num_cpus_for_fetching_data
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
        if not self._flag_is_read_only : # update metadata only when the current RamData object is not read-only
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
    def __repr__( self ) :
        """ # 2022-07-20 00:38:24 
        display RamData
        """
        return f"<{'' if not self._flag_is_read_only else '(read-only) '}RamData object ({'' if self.bc.filter is None else f'{self.bc.meta.n_rows}/'}{self.metadata[ 'int_num_barcodes' ]} barcodes X {'' if self.ft.filter is None else f'{self.ft.meta.n_rows}/'}{self.metadata[ 'int_num_features' ]} features" + ( '' if self.layer is None else f", {self.layer.int_num_records} records in the currently active layer '{self.layer.name}'" ) + f") stored at {self._path_folder_ramdata}{'' if self._path_folder_ramdata_mask is None else f' with local mask available at {self._path_folder_ramdata_mask}'}\n\twith the following layers : {self.layers}\n\t\tcurrent layer is '{self.layer.name}'>" # show the number of records of the current layer if available.
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
        rtx = self.layer._ramtx_features if flag_use_ramtx_indexed_by_features else self.layer._ramtx_barcodes
        
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
    def summarize( self, name_layer, axis, summarizing_func, int_num_threads = None, flag_overwrite_columns = True, int_num_entries_for_each_weight_calculation_batch = 2000, int_total_weight_for_each_batch = 2000000 ) :
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
        
                    summarizing_func( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) -> dictionary containing 'key' as summarized metric name and 'value' as a summarized value for the entry
                    
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
        flag_summarizing_barcode = axis in { 0, 'barcode', 'barcodes' } # retrieve a flag indicating whether the data is summarized for each barcode or not
        
        # retrieve the total number of entries in the axis that was not indexed (if calculating average expression of feature across barcodes, divide expression with # of barcodes, and vice versa.)
        int_total_num_entries_not_indexed = self.ft.meta.n_rows if flag_summarizing_barcode else self.bc.meta.n_rows 

        if int_num_threads is None : # use default number of threads
            int_num_threads = self.int_num_cpus
        if summarizing_func == 'sum' :
            def summarizing_func( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) :
                ''' # 2022-07-19 12:19:49 
                calculate sum of the values of the current entry
                
                assumes 'int_num_records' > 0
                '''
                int_num_records = len( arr_value ) # retrieve the number of records of the current entry
                dict_summary = { 'sum' : np.sum( arr_value ) if int_num_records > 30 else sum( arr_value ) } # if an input array has more than 30 elements, use np.sum to calculate the sum
                dict_summary[ 'mean' ] = dict_summary[ 'sum' ] / int_total_num_entries_not_indexed # calculate the mean
                return dict_summary
        elif summarizing_func == 'sum_and_dev' :
            def summarizing_func( self, int_entry_of_axis_for_querying, arr_int_entries_of_axis_not_for_querying, arr_value ) :
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
        # retrieve the list of key values returned by 'summarizing_func' by applying dummy values
        arr_dummy_one, arr_dummy_zero = np.ones( 10, dtype = int ), np.zeros( 10, dtype = int )
        dict_res = summarizing_func( self, 0, arr_dummy_zero, arr_dummy_one )
        l_name_col_summarized = sorted( list( dict_res ) ) # retrieve the list of key values of an dict_res result returned by 'summarizing_func'
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
            for int_entry_of_axis_for_querying, arr_int_entry_of_axis_not_for_querying, arr_value in zip( * rtx[ l_int_entry_current_batch ] ) : # retrieve data for the current batch
                # retrieve summary for the entry
                dict_res = summarizing_func( self, int_entry_of_axis_for_querying, arr_int_entry_of_axis_not_for_querying, arr_value ) # summarize the data for the entry
                # write the result to an output file
                newfile.write( ( '\t'.join( map( str, [ int_entry_of_axis_for_querying ] + list( dict_res[ name_col ] if name_col in dict_res else np.nan for name_col in l_name_col_summarized ) ) ) + '\n' ).encode( ) ) # write an index for the current entry # 0>1 coordinate conversion for 'int_entry'
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
    def apply( self, name_layer, name_layer_new, func = None, mode_instructions = 'sparse_for_querying_features', path_folder_ramdata_output = None, dtype_of_row_and_col_indices = np.int32, dtype_of_value = np.float64, int_num_threads = None, int_num_entries_for_each_weight_calculation_batch = 1000, int_total_weight_for_each_batch = 1000000, int_min_num_entries_in_a_batch_if_weight_not_available = 2000, int_num_of_records_in_a_chunk_zarr_matrix = 20000, int_num_of_entries_in_a_chunk_zarr_matrix_index = 1000, chunks_dense = ( 2000, 1000 ), dtype_dense_mtx = np.float64, dtype_sparse_mtx = np.float64, dtype_sparse_mtx_index = np.float64 ) :
        ''' # 2022-07-31 15:19:07 
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
                --> 'sparse_for_querying_features' ramtx object of the 'name_layer' layer converted to 'sparse_for_querying_features' and 'dense' ramtx object of the 'name_layer_new' layer. however, for this instruction, 'sparse_for_querying_features' will be read twice, which will be redundant.
        
                mode_instructions = [ [ 'sparse_for_querying_features' ],
                                      [ 'dense', [ 'sparse_for_querying_features', 'dense', 'sparse_for_querying_barcodes' ] ] ]
                --> 'sparse_for_querying_features' > 'sparse_for_querying_features'
                    'dense' > 'dense' # by default, dense is set for querying features, but it can be changed so that dense matrix can be constructed by querying barcodes from the source dense ramtx object.
                    'dense_for_querying_barcodes' > 'sparse_for_querying_barcodes' of 'name_layer_new'
                
                mode_instructions = [ [ 'dense_for_querying_features', 'dense_for_querying_barcode' ],
                                      [ 'dense', [ 'sparse_for_querying_features', 'dense', 'sparse_for_querying_barcodes' ] ] ]
                --> 'dense_for_querying_features' > 'dense'
                    'dense_for_querying_features' > 'sparse_for_querying_features'
                    'dense_for_querying_barcodes' > 'sparse_for_querying_barcodes'
                    
                in summary, (1) up to three operations will be performed, to construct three ramtx modes of the resulting layer, (2) the instructions at the front has higher priority, and (3) querying axis of dense can be specified or skipped (default will be used)
        
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
            func_bc = func[ 'barcode' ]
            func_ft = func[ 'feature' ]
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
        
        def RAMtx_Apply( self, rtx, func, flag_dense_ramtx_output, flag_sparse_ramtx_output, int_num_threads ) :
            ''' # 2022-07-31 21:22:18 
            path_folder_ramtx_output
            inputs 
            =========

            'rtx': an input RAMtx object
            '''
            ''' prepare '''
            ax = self.ft if rtx.is_for_querying_features else self.bc # retrieve appropriate axis
            ns = dict( ) # create a namespace that can safely shared between different scopes of the functions
            # create a temporary folder
            path_folder_temp = f'{path_folder_layer_new}temp_{UUID( )}/'
            os.makedirs( path_folder_temp, exist_ok = True )

            ''' initialize output ramtx objects '''
            """ %% DENSE %% """
            if flag_dense_ramtx_output : # if dense output is present
                path_folder_ramtx_dense = f"{path_folder_layer_new}dense/"
                os.makedirs( path_folder_ramtx_dense, exist_ok = True ) # create the output ramtx object folder
                path_folder_ramtx_dense_mtx = f"{path_folder_ramtx_output}matrix.zarr/" # retrieve the folder path of the output RAMtx Zarr matrix object.
                assert not os.path.exists( path_folder_ramtx_dense_mtx ) # output zarr object should NOT exists!
                za_mtx_dense = zarr.open( path_folder_ramtx_dense_mtx, mode = 'w', shape = ( rtx._int_num_barcodes, rtx._int_num_features ), chunks = chunks_dense, dtype = dtype_dense_mtx, synchronizer = zarr.ThreadSynchronizer( ) ) # use the same chunk size of the current RAMtx
            """ %% SPARSE %% """
            if flag_sparse_ramtx_output : # if sparse output is present
                path_folder_ramtx_sparse = f"{path_folder_layer_new}sparse_for_querying_{'features' if rtx.is_for_querying_features else 'barcodes'}/"
                os.makedirs( path_folder_ramtx_sparse, exist_ok = True ) # create the output ramtx object folder
                path_folder_ramtx_sparse_mtx = f"{path_folder_ramtx_sparse}matrix.zarr/" # retrieve the folder path of the output RAMtx Zarr matrix object.
                assert not os.path.exists( path_folder_ramtx_sparse_mtx ) # output zarr object should NOT exists!
                assert not os.path.exists( f'{path_folder_ramtx_sparse}matrix.index.zarr' ) # output zarr object should NOT exists!
                za_mtx_sparse = zarr.open( path_folder_ramtx_sparse_mtx, mode = 'w', shape = ( rtx._int_num_records, 2 ), chunks = ( int_num_of_records_in_a_chunk_zarr_matrix, 2 ), dtype = dtype_sparse_mtx, synchronizer = zarr.ThreadSynchronizer( ) ) # use the same chunk size of the current RAMtx
                za_mtx_sparse_index = zarr.open( f'{path_folder_ramtx_sparse}matrix.index.zarr', mode = 'w', shape = ( rtx.len_axis_for_querying, 2 ), chunks = ( int_num_of_entries_in_a_chunk_zarr_matrix_index, 2 ), dtype = dtype_sparse_mtx_index, synchronizer = zarr.ThreadSynchronizer( ) ) # use the same dtype and chunk size of the current RAMtx
                
                ns[ 'int_num_records_written_to_ramtx' ] = 0 # initlaize the total number of records written to ramtx object
                ns[ 'int_num_chunks_written_to_ramtx' ] = 0 # initialize the number of chunks written to ramtx object
                int_num_records_in_a_chunk_of_mtx_sparse = za_mtx_sparse.chunks[ 0 ] # retrieve the number of records in a chunk of output zarr matrix

            """ convert matrix values and save it to the output RAMtx object """
            # define functions for multiprocessing step
            def process_batch( l_int_entry_current_batch, pipe_to_main_process ) :
                ''' # 2022-05-08 13:19:07 
                retrieve data for a given list of entries, transform values, and save to a Zarr object and index the object, and returns the number of written records and the paths of the written objects (index and Zarr matrix)
                '''
                # retrieve the number of index_entries
                int_num_entries = len( l_int_entry_current_batch )

                """
                prepare writing
                """
                
                """ %% SPARSE %% """
                if flag_sparse_ramtx_output : # if sparse output is present
                    # open an Zarr object
                    path_folder_zarr_output_sparse = f"{path_folder_temp}{UUID( )}.zarr/" # define output Zarr object path
                    za_output_sparse = zarr.open( path_folder_zarr_output_sparse, mode = 'w', shape = ( rtx._int_num_records, 2 ), chunks = za_mtx_sparse.chunks, dtype = dtype_of_value, synchronizer = zarr.ThreadSynchronizer( ) )
                    # open an index file
                    path_file_index_output_sparse = f"{path_folder_temp}{UUID( )}.index.tsv.gz" # define output index file path
                    newfile_index = gzip.open( path_file_index_output_sparse, 'wb' )
                    int_num_records_written_sparse = 0 # initialize the record count
                
                #""" retrieve data from the source """
                #arr_int_barcode, arr_int_feature, arr_value, l_int_num_records = rtx.get_sparse_matrix( l_int_entry_current_batch, flag_return_as_arrays = True )
                
                # iterate through the data of each entry
                for int_entry_of_axis_for_querying, arr_int_entry_of_axis_not_for_querying, arr_value in zip( * rtx[ l_int_entry_current_batch ] ) : # retrieve data for the current batch
                    # transform the values of an entry
                    int_entry_of_axis_for_querying, arr_int_entry_of_axis_not_for_querying, arr_value = func( self, int_entry_of_axis_for_querying, arr_int_entry_of_axis_not_for_querying, arr_value ) 
                    int_num_records = len( arr_value ) # retrieve number of returned records
                    
                    """ %% DENSE %% """
                    if flag_dense_ramtx_output : # if dense output is present
                        arr_int_entry_of_axis_for_querying = np.full_like( arr_int_entry_of_axis_not_for_querying, int_entry_of_axis_for_querying )
                        za_mtx_dense.set_coordinate_selection( ( arr_int_entry_of_axis_not_for_querying, arr_int_entry_of_axis_for_querying ) if rtx.is_for_querying_features else ( arr_int_entry_of_axis_for_querying, arr_int_entry_of_axis_not_for_querying ), arr_value ) # write dense zarr matrix
                        del arr_int_entry_of_axis_for_querying
                    
                    """ %% SPARSE %% """
                    if flag_sparse_ramtx_output : # if sparse output is present
                        za_output_sparse[ int_num_records_written_sparse : int_num_records_written_sparse + int_num_records ] = np.vstack( ( arr_int_entry_of_axis_not_for_querying, arr_value ) ).T # save transformed data
                        # write the result to the index file
                        newfile_index.write( ( '\t'.join( map( str, [ int_entry_of_axis_for_querying, int_num_records_written_sparse, int_num_records_written_sparse + int_num_records ] ) ) + '\n' ).encode( ) ) # write an index for the current entry # 0>1 coordinate conversion for 'int_entry'
                        # update the number of records written
                        int_num_records_written_sparse += int_num_records
                        
                """ %% SPARSE %% """
                if flag_sparse_ramtx_output : # if sparse output is present
                    newfile_index.close( ) # close file
                    za_output_sparse.resize( int_num_records_written_sparse, 2 ) # resize the output Zarr object
                    
                pipe_to_main_process.send( ( int_num_records_written_sparse, path_folder_zarr_output_sparse, path_file_index_output_sparse ) ) # send information about the output files
            def post_process_batch( res ) :
                """ # 2022-07-06 10:22:05 
                """
                
                
                """ %% SPARSE %% """
                if flag_sparse_ramtx_output : # if sparse output is present
                    # parse result
                    int_num_records_written, path_folder_zarr_output, path_file_index_output = res

                    ns[ 'int_num_records_written_to_ramtx' ] += int_num_records_written # update the number of records written to the output RAMtx
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
                
            # transform the values of the RAMtx using multiple processes
            bk.Multiprocessing_Batch( rtx.batch_generator( ax.filter, int_num_entries_for_each_weight_calculation_batch, int_total_weight_for_each_batch, int_min_num_entries_in_a_batch_if_weight_not_available = int_min_num_entries_in_a_batch_if_weight_not_available, int_chunk_size_for_checking_boundary = chunks_dense[ 0 ] if flag_dense_ramtx_output else None ), process_batch, post_process_batch = post_process_batch, int_num_threads = int_num_threads, int_num_seconds_to_wait_before_identifying_completed_processes_for_a_loop = 0.2 ) # create batch considering chunk boundaries

            # remove temp folder
            shutil.rmtree( path_folder_temp )
            
            ''' export ramtx settings '''
            root = zarr.group( path_folder_ramtx_output )
            root.attrs[ 'dict_metadata' ] = { 
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
        for p in l_process : p.start( )
        for p in l_process : p.join( )
        
        """
        update the metadata
        """
        if self.verbose :
            print( f'apply operation {name_layer} > {name_layer_new} has been completed' )
            
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
        root.attrs[ 'dict_metadata' ] = { 
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
        self.apply( name_layer, name_layer_new, func = ( func_norm_barcode_indexed, func_norm_feature_indexed ), flag_simultaneous_processing_of_paired_ramtx = flag_simultaneous_processing_of_paired_ramtx, int_num_threads = int_num_threads, ** args ) # flag_dtype_output = None : use the same dtype as the input RAMtx object
    
        if not flag_name_col_total_count_already_loaded : # unload count data of barcodes from memory if the count data was not loaded before calling this method
            del self.bc.meta.dict[ name_col_total_count ]
    def scale( self, name_layer = 'normalized_log1p', name_layer_new = 'normalized_log1p_scaled', name_col_variance = 'normalized_log1p_variance', name_col_mean = 'normalized_log1p_mean', max_value = 10, flag_simultaneous_processing_of_paired_ramtx = True, int_num_threads = None, ** args ) :
        """ # 2022-07-27 16:32:26 
        current implementation only allows output values to be not zero-centered. the zero-value will remain zero, while Z-scores of the non-zero values will be increased by Z-score of zero values, enabling processing of sparse count data

        'name_layer' : the name of the data source layer
        'name_layer_new' : the name of the data sink layer (new layer)
        'name_col_variance' : name of feature metadata containing variance informatin
        'name_col_mean' : name of feature metadata containing mean informatin
        'max_value' : clip values larger than 'max_value' to 'max_value'
        """
        # check validity of inputs
        # column names should be available in the metadata
        assert name_col_variance in self.ft.meta 
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
        self.apply( name_layer, name_layer_new, func = ( func_barcode_indexed, func_feature_indexed ), flag_simultaneous_processing_of_paired_ramtx = flag_simultaneous_processing_of_paired_ramtx, int_num_threads = int_num_threads, ** args ) # flag_dtype_output = None : use the same dtype as the input RAMtx object

        # unload count data of barcodes from memory if the count data was not loaded before calling this method
    #     if not flag_name_col_mean_already_loaded : 
    #         del self.ft.meta.dict[ name_col_mean ]
        if not flag_name_col_variance_already_loaded : 
            del self.ft.meta.dict[ name_col_variance ]
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
        rtx, ax = self.layer._ramtx_barcodes, self.bc

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
