"""
classes and functions for sharing data across multiple forked processes
"""
from multiprocessing.managers import BaseManager
import numpy as np
from typing import Union, List, Dict
import zarr
import s3fs
import asyncio
import aiofiles
import aiohttp
import nest_asyncio
nest_asyncio.apply( )

''' async utility functions ''' 
def get_or_create_eventloop():
    """ # 2023-09-24 19:41:39 
    reference: https://techoverflow.net/2020/10/01/how-to-fix-python-asyncio-runtimeerror-there-is-no-current-event-loop-in-thread/
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()
        
async def read_local_file_async(file_name, mode):
    """ # 2023-09-24 19:50:02 
    read file using 'aiofiles'
    """
    async with aiofiles.open(file_name, mode=mode) as f:
        return await f.read()
    
async def write_local_file_async(file_name, mode, content ):
    """ # 2023-09-24 19:50:22 
    write file using 'aiofiles'
    """
    async with aiofiles.open(file_name, mode=mode) as f:
        return await f.write( content )
    
async def fetch_http_file_async(session, url):
    """ # 2023-09-24 19:50:22 
    read a http file using 'aiohttp'
    """
    async with session.get( url ) as response:
        if response.status != 200:
            response.raise_for_status( )
        return await response.text( )
    
async def fetch_http_files_async( l_path_file: List[ str ] ) -> List[ str ] :
    """ # 2023-09-24 19:50:22 
    read http files using 'aiohttp'
    """
    async with aiohttp.ClientSession() as session:
        loop = get_or_create_eventloop()
        l_content = loop.run_until_complete( asyncio.gather( * list( fetch_http_file_async( session, path_file ) for path_file in l_path_file ) ) ) # read the contents
    return l_content

# async def start_s3_files_async_session( s3 ):
#     ''' # 2023-09-24 23:17:06 
#     return async s3 session
#     '''
#     return await s3.set_session()

# async def close_s3_files_async_session( session ):
#     ''' # 2023-09-24 23:17:06 
#     return async s3 session
#     '''
#     await session.close( )

async def read_s3_files_async( l_path_file : List[ str ], dict_kwargs_s3 : dict = dict( ) ):
    s3 = s3fs.S3FileSystem( asynchronous = True, ** dict_kwargs_s3 )
    session = await s3.set_session( refresh = True )
    loop = get_or_create_eventloop( )
    l_content = loop.run_until_complete( asyncio.gather( * list( s3._cat_file( path_file ) for path_file in l_path_file ) ) ) # read the contents
    await session.close( )
    return l_content

async def put_s3_files_async( l_path_file_local : List[ str ], l_path_file_remote : List[ str ], dict_kwargs_s3 : dict = dict( ) ):
    s3 = s3fs.S3FileSystem( asynchronous = True, ** dict_kwargs_s3 )
    session = await s3.set_session( refresh = True )
    loop = get_or_create_eventloop( )
    l_content = loop.run_until_complete( asyncio.gather( * list( s3._put_file( path_file_local, path_file_remote ) for path_file_local, path_file_remote in zip( l_path_file_local, l_path_file_remote ) ) ) ) # copy the files
    await session.close( )
    return l_content

''' class for performing file system opertions '''

class FileSystemOperator:
    """# 2023-09-24 14:50:47 
    A class intended for performing asynchronous file system operations in a separate, managed process. By using multiple managers, concurrent, asynchronous operations can be performed in multiple processes. These managers can be used multiple times.
    
    dict_kwargs_s3 : dict = dict( ) # s3 credentials to use
    """

    # constructor
    def __init__(self, dict_kwargs_s3 : dict = dict( )):
        import s3fs
        # save the settings
        self._dict_kwargs_s3 = dict_kwargs_s3
        
        # open async/sync version of s3fs
        self._as3 = s3fs.S3FileSystem( asynchronous = True, **dict_kwargs_s3 )
        self._s3 = s3fs.S3FileSystem( **dict_kwargs_s3 )
        
        # start the async session
#         self._as3_session = asyncio.run( start_s3_files_async_session( self._as3 ) )
    
#     def terminate( self ) :
#         """ # 2023-09-24 23:20:23 
#         terminate the session
#         """
#         # stop the async session
#         asyncio.run( close_s3_files_async_session( self._as3_session ) )

    def exists(self, path_src : str, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        """
        return self._s3.exists(path_src, **kwargs)

    def rm(self, path_src : str, flag_recursive: bool = True, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        """
        return self._s3.rm(path_src, recursive=flag_recursive, **kwargs)  # delete files
    
    def glob(self, path_src : str, flag_recursive: bool = True, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        """
        return list(
            "s3://" + e for e in self._s3.glob(path_src, **kwargs)
        )  # 's3://' prefix should be added

    def mkdir(self, path_src : str, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        """
        # use default 'exist_ok' value
        if "exist_ok" not in kwargs:
            kwargs["exist_ok"] = True
        return self._s3.makedirs(path_src, **kwargs)

    def mv(self, path_src : str, path_dest : str, flag_recursive: bool = True, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        """
        if not self._s3.exists(
            path_dest, **kwargs
        ):  # avoid overwriting of the existing file
            return self._s3.mv(path_src, path_dest, recursive=flag_recursive, **kwargs)
        else:
            return "destionation file already exists, exiting"

    def cp(self, path_src : str, path_dest : str, flag_recursive: bool = True, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        """
        if is_s3_url(path_src) and is_s3_url(path_dest):  # copy from s3 to s3
            return self._s3.copy(path_src, path_dest, recursive=flag_recursive, **kwargs)
        elif is_s3_url(path_src):  # copy from s3 to local
            return self._s3.get(path_src, path_dest, recursive=flag_recursive, **kwargs)
        elif is_s3_url(path_dest):  # copy from local to s3
            return self._s3.put(path_src, path_dest, recursive=flag_recursive, **kwargs)

    def isdir(self, path_src : str, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        """
        return self._s3.isdir(path_src)
    
    def get_zarr_metadata(self, path_src : str, **kwargs):
        """# 2023-01-08 23:05:40
        return the list of keys
        ❤️ test
        """
        return dict( zarr.open( path_src ).attrs )
    
    def read_local_files_async(self, l_path_file: List[ str ], mode = 'rt') -> List[ str ]:
        """ # 2023-09-24 19:42:55 
        read local files asynchronously
        """
        loop = get_or_create_eventloop()
        return loop.run_until_complete( asyncio.gather( * list( read_local_file_async(path_file, mode) for path_file in l_path_file ) ) )
    
    def write_local_files_async(self, dict_path_file_to_content : dict, mode = 'wt') :
        """ # 2023-09-24 19:42:55 
        write local files asynchronously
        """
        loop = get_or_create_eventloop()
        return loop.run_until_complete( asyncio.gather( * list( write_local_file_async(path_file, mode, dict_path_file_to_content[path_file]) for path_file in dict_path_file_to_content ) ) )
    
    def read_http_files_async(self, l_path_file: List[ str ] ) -> List[ str ]:
        """ # 2023-09-24 19:42:55 
        read remote http files asynchronously
        """
        result = asyncio.run( fetch_http_files_async( l_path_file ) )
        return result
    
    def read_s3_files_async( self, l_path_file : List[ str ] ) :
        """ # 2023-09-24 23:13:15 
        """
        result = asyncio.run( read_s3_files_async( l_path_file, self._dict_kwargs_s3 ) )
        return result

    def put_s3_files_async( self, l_path_file_local : List[ str ], l_path_file_remote : List[ str ] ) :
        """ # 2023-09-24 23:15:06 
        """
        result = asyncio.run( put_s3_files_async( l_path_file_local, l_path_file_remote, self._dict_kwargs_s3 ) )
        return result
    
class ZarrObject:
    """# 2023-09-24 17:50:46 
    A class for hosting Zarr object in a spawned, managed process for accessing remote objects in forked processes
    API functions calls mimic those of a zarr object for seamless replacement of a zarr object.

    path_folder_zarr : str # a path to a (remote) zarr object
    mode : str = 'r' # mode

    path_process_synchronizer : Union[ str, None ] = None # path to the process synchronizer. if None is given, does not use any synchronizer
    proxy_object = None, # proxy object of 'ZarrObject'
    """
    def get_object_properties( self ) :
        ''' # 2023-09-24 17:39:41 
        a function for retrieving object properties (for proxy object from which property is not accessible).
        '''
        return self.shape, self.chunks, self.dtype, self.fill_value, self._path_folder_zarr, self._mode, self._path_process_synchronizer
    
    def _sync_object_properties( self ) :
        ''' # 2023-09-24 17:39:41 
        synchronize object properties so that the properties of the proxy object are the same as those of the current object.
        '''
        if self._proxy_object is not None :
            # set properties of the object based on the properties of the proxy object
            self.shape, self.chunks, self.dtype, self.fill_value, self._path_folder_zarr, self._mode, self._path_process_synchronizer = self._proxy_object.get_object_properties( )
            
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
        if self._proxy_object is None :
            # if the zarr object is already opened in the same mode, exit, unless 'reload' flag has been set to True.
            if not reload and path_folder_zarr == self.path_folder and self._mode == mode:
                return

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
            self._path_process_synchronizer = path_process_synchronizer
        else :
            # open zarr object in the proxy object
            self._proxy_object.open(
                path_folder_zarr = path_folder_zarr,
                mode = mode,
                shape = shape,
                chunks = chunks,
                dtype = dtype,
                fill_value = fill_value,
                path_process_synchronizer = path_process_synchronizer,
                reload = reload,
            )
            self._sync_object_properties( ) # synchronize object properties using the proxy object
    
    def __init__(
        self,
        path_folder_zarr : Union[ str, None ] = None,
        mode : Union[ str, None ] = "r",
        shape : tuple =None,
        chunks : tuple =None,
        dtype=np.int32,
        fill_value = 0,
        path_process_synchronizer: Union[str, None] = None,
        proxy_object = None,
    ):
        """# 2023-09-24 14:50:36 """
        # set attributes
        self._proxy_object = proxy_object 
        self._path_folder_zarr = None
        self._mode = None
        
        if self._proxy_object is None : # if proxy object has not been given, open the zarr object
            self.open(
                path_folder_zarr = path_folder_zarr,
                mode = mode,
                shape = shape,
                chunks = chunks,
                dtype = dtype,
                fill_value = fill_value,
                path_process_synchronizer = path_process_synchronizer,
                reload = False,
            )
        else :
            self._sync_object_properties( ) # synchronize object properties using the proxy object

    @property
    def is_proxy_object(self):
        """# 2023-09-24 18:01:20 
        return True if proxy object exists
        """
        return self._proxy_object is not None
            
    @property
    def path_folder(self):
        """# 2023-04-19 17:33:21"""
        return self._path_folder_zarr

    def __repr__(self):
        """# 2023-04-20 01:06:16"""
        return f"<Zarr of {self.path_folder}>"

    @property
    def path_process_synchronizer(self):
        """# 2022-12-07 00:19:29
        return a path of the folder used for process synchronization
        """
        return self._path_process_synchronizer

    def get_attrs(self, *keys):
        """# 2023-04-19 15:00:04
        get an attribute of the currently opened zarr object using the list of key values
        """
        if self._proxy_object is not None :
            return self._proxy_object.get_attrs( *keys )
        else :
            set_keys = set(self._za.attrs)  # retrieve a list of keys
            return dict(
                (key, self._za.attrs[key]) for key in keys if key in set_keys
            )  # return a subset of metadata using the list of key values given as 'args'

    def get_attr(self, key):
        """# 2023-04-20 01:08:59
        a wrapper of 'get_attrs' for a single key value
        """
        if self._proxy_object is not None :
            return self._proxy_object.get_attr( key )
        else :
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
        if self._proxy_object is not None :
            return self._proxy_object.set_attrs( **kwargs )
        else :
            # update the metadata
            for key in kwargs:
                self._za.attrs[key] = kwargs[key]

    def get_coordinate_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'get_coordinate_selection' zarr operation using a spawned process.
        """
        if self._proxy_object is not None :
            return self._proxy_object.get_coordinate_selection(*args, **kwargs)
        else :
            return self._za.get_coordinate_selection(*args, **kwargs)

    def get_basic_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'get_basic_selection' zarr operation using a spawned process.
        """
        if self._proxy_object is not None :
            return self._proxy_object.get_basic_selection(*args, **kwargs)
        else :
            return self._za.get_basic_selection(*args, **kwargs)

    def get_orthogonal_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'get_orthogonal_selection' zarr operation using a spawned process.
        """
        if self._proxy_object is not None :
            return self._proxy_object.get_orthogonal_selection(*args, **kwargs)
        else :
            return self._za.get_orthogonal_selection(*args, **kwargs)

    def get_mask_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'get_mask_selection' zarr operation using a spawned process.
        """
        if self._proxy_object is not None :
            return self._proxy_object.get_mask_selection(*args, **kwargs)
        else :
            return self._za.get_mask_selection(*args, **kwargs)

    def set_coordinate_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'set_coordinate_selection' zarr operation using a spawned process.
        """
        if self._proxy_object is not None :
            return self._proxy_object.set_coordinate_selection(*args, **kwargs)
        else :
            return self._za.set_coordinate_selection(*args, **kwargs)

    def set_basic_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'set_basic_selection' zarr operation using a spawned process.
        """
        if self._proxy_object is not None :
            return self._proxy_object.set_basic_selection(*args, **kwargs)
        else :
            return self._za.set_basic_selection(*args, **kwargs)

    def set_orthogonal_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'set_orthogonal_selection' zarr operation using a spawned process.
        """
        if self._proxy_object is not None :
            return self._proxy_object.set_orthogonal_selection(*args, **kwargs)
        else :
            return self._za.set_orthogonal_selection(*args, **kwargs)

    def set_mask_selection(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'set_mask_selection' zarr operation using a spawned process.
        """
        if self._proxy_object is not None :
            return self._proxy_object.set_mask_selection(*args, **kwargs)
        else :
            return self._za.set_mask_selection(*args, **kwargs)

    def resize(self, *args, **kwargs):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the 'resize' zarr operation using a spawned process.
        """
        if self._proxy_object is not None :
            return self._proxy_object.resize(*args, **kwargs)
        else :
            return self._za.resize(*args, **kwargs)

    def __getitem__(self, args):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the '__getitem__' zarr operation using a spawned process.
        """
        if self._proxy_object is not None :
            return self._proxy_object.getitem(args)
        else :
            return self._za.__getitem__(args)

    def __setitem__(self, args, values):
        """# 2022-12-05 22:55:58
        a (possibly) fork-safe wrapper of the '__setitem__' zarr operation using a spawned process.
        """
        if self._proxy_object is not None :
            return self._proxy_object.setitem(args, values)
        else :
            return self._za.__setitem__(args, values)
        
    def getitem(self, args):
        """# 2023-09-24 18:08:07 
        public wrapper of '__getitem__'
        """
        return self.__getitem__(args)

    def setitem(self, args, values):
        """# 2023-09-24 18:08:07 
        public wrapper of '__setitem__'
        """
        return self.__setitem__(args, values)
    
# configure the manager  
class ManagerFileSystem(BaseManager):
    pass

ManagerFileSystem.register("FileSystemOperator", FileSystemOperator)
ManagerFileSystem.register("ZarrObject", ZarrObject)

class FileSystemPool :
    """# 2023-09-24 14:51:46 
    create a pool of spwaned, managed processes for performing Zarr and FileSystem operations. Alternatively, every operation can be done in the current process without spawning a new process
    
    int_num_processes : Union[ None, int ] = 8 # if 'int_num_processes' is 0 or None, all operations will be performed in the current process without spawning a pool of processes to handle the operations
    dict_kwargs_s3 : dict = dict( ) # arguments for initializing s3fs
    """
    def _get_managed_filesystem( self, dict_kwargs_s3 : dict = dict( ) ) :
        """ # 2023-09-23 23:25:56 
        """
        # %% PROCESS SPAWNING %%
        import multiprocessing as mp
        mpsp = mp.get_context("spawn")
        manager = ManagerFileSystem(ctx = mpsp) # use spawning to create the manager
        manager.start()  # start the manager
        managed_filesystemoperator = getattr(manager, 'FileSystemOperator')(dict_kwargs_s3)
        return {
            "manager": manager,
            "managed_filesystemoperator": managed_filesystemoperator,
        }

    def __init__( 
        self,
        int_num_processes : Union[ None, int ] = 8,
        dict_kwargs_s3 : dict = dict( )
    ):
        """# 2023-09-24 14:50:36 """
        # set attributes
        self._int_num_processes = 0 if int_num_processes is None else max( 0, int( int_num_processes ) )
        self._flag_spawn = self.int_num_processes > 0
        
        # retrieve list of managed filesystems
        if self.flag_spawn :
            self._l_mfs = list( self._get_managed_filesystem( dict_kwargs_s3 ) for _ in range( int_num_processes ) )
        else : # perform all operations in the current process
            self._fs = FileSystemOperator( dict_kwargs_s3 )
        
    def get_operator( self ) :
        """ # 2023-09-24 17:08:07 
        get a filesystemoperator, randomly selected from the pool
        """
        return self._l_mfs[ np.random.randint( self._int_num_processes ) ][ 'managed_filesystemoperator' ] if self.flag_spawn else self._fs
            
    @property
    def int_num_processes( self ) :
        ''' # 2023-09-24 17:02:50 
        indicate whether process spawning has been used.
        '''
        return self._int_num_processes
            
    @property
    def flag_spawn( self ) :
        ''' # 2023-09-24 17:02:50 
        indicate whether process spawning has been used.
        '''
        return self._flag_spawn

    def zarr_open( self, * args, ** kwargs ) :
        """ # 2023-09-24 17:18:19 
        open a Zarr Object
        """
        if self.flag_spawn :
            return ZarrObject( proxy_object = getattr( self._l_mfs[ np.random.randint( self._int_num_processes ) ][ 'manager' ], 'ZarrObject' )( * args, ** kwargs ) ) # open a proxy object of ZarrObject using one of the spawned, managed processes, and wrap the proxy object using ZarrObject
        else :
            return ZarrObject( * args, ** kwargs ) # open ZarrObject
        