"""
classes and functions for sharing data across multiple forked processes
"""
import time
from multiprocessing.managers import BaseManager
import numpy as np
import os
from typing import Union, List, Dict
import uuid
import zarr
import s3fs
import asyncio
import aiofiles
import aiofiles.os
from aioshutil import rmtree as a_rmtree
import aiohttp
import json
import nest_asyncio
nest_asyncio.apply( )

''' file file system operations '''
def UUID():
    """return a 128bit universal unique identifier"""
    return uuid.uuid4().hex

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
    
async def listdir_local_file_async( path_folder ) -> List[str]:
    """ # 2023-09-24 19:50:02 
    list directory file using 'aiofiles'
    """
    return await aiofiles.os.listdir( path_folder )

async def rename_local_file_async(file_name_src, file_name_dst)->None:
    """ # 2023-09-24 19:50:02 
    rename file using 'aiofiles'
    """
    return await aiofiles.os.rename( file_name, file_name_dst )
    
async def remove_local_file_async(path_file, flag_recursive : bool = True)->None:
    """ # 2023-09-24 19:50:02 
    remove file using 'aiofiles' or 'aioshutil'
    """
    return await ( a_rmtree( path_file ) if flag_recursive and path_file[ -1 ] == '/' else aiofiles.os.remove( path_file ) ) # use aioshutil.rmtree if flag_recursive is True and deleting a folder
    
async def check_local_file_async(file_name )->bool:
    """ # 2023-09-24 19:50:22 
    check file using 'aiofiles'
    """
    return await aiofiles.os.path.exists( file_name )
    
async def fetch_http_file_async(session, url : str, mode : str = 'rt' ) -> str :
    """ # 2023-09-24 19:50:22 
    read a http file using 'aiohttp'
    """
    async with session.get( url ) as response:
        if response.status != 200:
            return None # if the file does not exist, return None instead
            # response.raise_for_status( )
        output = await response.read( )
        if mode != 'rb' : # if not reading binary output, decode the output
            output = output.decode( )
        return output
    
async def check_http_file_async(session, url : str ) -> bool:
    """ # 2023-09-24 19:50:22 
    check a http file using 'aiohttp'
    """
    async with session.get( url ) as response:
        return response.status == 200
    
async def fetch_http_files_async( l_path_file: List[ str ], mode : str = 'rt' ) -> List[ str ] :
    """ # 2023-09-24 19:50:22 
    read http files using 'aiohttp'
    """
    async with aiohttp.ClientSession() as session:
        loop = get_or_create_eventloop()
        l_content = loop.run_until_complete( asyncio.gather( * list( fetch_http_file_async( session, path_file, mode = mode ) for path_file in l_path_file ) ) ) # read the contents
    return l_content

async def check_http_files_async( l_path_file: List[ str ] ) -> List[ bool ] :
    """ # 2023-09-24 19:50:22 
    read http files using 'aiohttp'
    """
    async with aiohttp.ClientSession() as session:
        loop = get_or_create_eventloop()
        l_content = loop.run_until_complete( asyncio.gather( * list( check_http_file_async( session, path_file ) for path_file in l_path_file ) ) ) # read the contents
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

async def exists_s3_files_async( l_path_file : List[ str ], dict_kwargs_s3 : dict = dict( ) ):
    s3 = s3fs.S3FileSystem( asynchronous = True, ** dict_kwargs_s3 )
    session = await s3.set_session( refresh = True )
    loop = get_or_create_eventloop( )
    l_content = loop.run_until_complete( asyncio.gather( * list( s3._exists( path_file ) for path_file in l_path_file ) ) ) # read the contents
    await session.close( )
    return l_content

async def rm_s3_files_async( l_path_file : List[ str ], flag_recursive : bool = True, dict_kwargs_s3 : dict = dict( ) ):
    s3 = s3fs.S3FileSystem( asynchronous = True, ** dict_kwargs_s3 )
    session = await s3.set_session( refresh = True )
    loop = get_or_create_eventloop( )
    l_content = loop.run_until_complete( asyncio.gather( * list( s3._rm( path_file, flag_recursive = flag_recursive ) for path_file in l_path_file ) ) ) # read the contents
    await session.close( )
    return l_content

async def read_s3_files_async( l_path_file : List[ str ], dict_kwargs_s3 : dict = dict( ) ):
    s3 = s3fs.S3FileSystem( asynchronous = True, ** dict_kwargs_s3 )
    session = await s3.set_session( refresh = True )
    loop = get_or_create_eventloop( )
    l_content = loop.run_until_complete( asyncio.gather( * list( s3._cat_file( path_file ) for path_file in l_path_file ), return_exceptions = True ) ) # read the contents
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
    path_folder_temp : str = '/tmp/' # whrere a temporary folder will be created
    """
    def local_mkdir( self, path_src : str, ** kwargs ) :
        ''' # 2023-11-07 00:02:04  '''
        # use default 'exist_ok' value
        if "exist_ok" not in kwargs:
            kwargs["exist_ok"] = True
        os.makedirs(path_src, exist_ok=kwargs["exist_ok"])
        
    # constructor
    def __init__(self, dict_kwargs_s3 : dict = dict( ), path_folder_temp : str = '/tmp/' ):
        import s3fs
        # set the settings
        self._dict_kwargs_s3 = dict_kwargs_s3
        self._path_folder_temp = f"{path_folder_temp}{UUID( )}/"
        self.local_mkdir( self._path_folder_temp ) # create the output folder
        
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
    
    @property
    def path_folder_temp( self ) :
        ''' # 2023-11-06 23:51:42  '''
        return self._path_folder_temp
    
    def terminate( self ) :
        ''' # 2023-11-06 23:51:36 remove the temporary files and exit '''
        self.local_rm( self.path_folder_temp )
        
    def local_exists( self, path_src : str, ** kwargs ) :
        ''' # 2023-11-07 00:02:04  '''
        return os.path.exists( path_src )
    
    def local_read_file( self, path_src : str, mode : str = 'rt', ** kwargs ) :
        """ # 2023-11-07 17:58:49  """
        try :
            with open(path_src, mode=mode) as f:
                return f.read( )
        except FileNotFoundError :
            return None # if the file does not exist, return None

    def local_write_file( self, path_src : str, mode : str, content, ** kwargs ) :
        """ # 2023-11-07 17:58:49  """
        with open(path_src, mode=mode) as f:
            f.write( content )
        
    def local_rm( self, path_src : str, flag_recursive : bool = True, ** kwargs ) :
        ''' # 2023-11-07 00:02:04  '''
        if flag_recursive and os.path.isdir(
            path_src
        ):  # when the recursive option is active
            shutil.rmtree(path_src)
        else:
            os.remove(path_src)
            
    def local_glob( self, path_src : str, flag_recursive: bool = False, ** kwargs ) :
        ''' # 2023-11-07 00:02:04  '''
        return glob.glob(path_src, recursive = flag_recursive )
        
    def local_mv( self, path_src : str, path_dest : str, ** kwargs ) :
        ''' # 2023-11-07 00:02:04  '''
        shutil.move(path_src, path_dest)
        
    def local_cp( self, path_src : str, path_dest : str, flag_recursive : bool = True, ** kwargs ) :
        ''' # 2023-11-07 00:02:04  '''
        if flag_recursive and os.path.isdir(
            path_src
        ):  # when the recursive option is active
            shutil.copytree(path_src, path_dest)
        else:
            shutil.copyfile(path_src, path_dest)
            
    def local_isdir( self, path_src : str, ** kwargs ) :
        ''' # 2023-11-07 00:02:04  '''
        return os.path.isdir(path_src)
    
    def local_listdir( self, path_src : str, ** kwargs ) :
        ''' # 2023-11-07 00:02:04  '''
        return os.listdir( path_src )
    
    def http_exists( self, path_src : str, ** kwargs ) :
        return http_response_code(path_src) == 200 # check whether http file (not tested for directory) exists
    
    def http_read_file( self, path_src : str, mode : str = 'rt', ** kwargs ) :
        """ # 2023-11-07 17:58:49  """
        import requests  # download from url
        with requests.get(path_src, stream=True) as r:
            if r.status_code != 200 : # if an error has occured, return None
                return None
            content = r.raw.read( )
        if mode != 'rb' : # decode output if mode is not 'rb'
            content = content.decode( )
        return content

    def s3_exists(self, path_src : str, **kwargs):
        """# 2023-01-08 23:05:40 """
        return self._s3.exists(path_src, **kwargs)
    
    def s3_read_file( self, path_src : str, mode : str = 'rt', ** kwargs ) :
        """ # 2023-11-07 17:58:49  """
        try :
            content = self._s3.cat(path_src)
            if mode != 'rb' : # decode output if mode is not 'rb'
                content = content.decode( )
            return content
        except FileNotFoundError :
            return None # return None if file is not found

    def s3_write_file( self, path_src : str, mode : str, content, ** kwargs ) :
        """ # 2023-11-07 17:58:49  """
        with self._s3.open( path_src, mode ) as f :
            f.write( content )
        
    def s3_rm(self, path_src : str, flag_recursive: bool = True, **kwargs):
        """# 2023-01-08 23:05:40 """
        return self._s3.rm(path_src, recursive=flag_recursive, **kwargs)  # delete files
    
    def s3_glob(self, path_src : str, flag_recursive: bool = False, **kwargs):
        """# 2023-01-08 23:05:40 """
        return list(
            "s3://" + e for e in self._s3.glob(path_src, **kwargs)
        )  # 's3://' prefix should be added

    def s3_mkdir(self, path_src : str, **kwargs):
        """# 2023-01-08 23:05:40 """
        # use default 'exist_ok' value
        if "exist_ok" not in kwargs:
            kwargs["exist_ok"] = True
        return self._s3.makedirs(path_src, **kwargs)

    def s3_mv(self, path_src : str, path_dest : str, flag_recursive: bool = True, flag_overwrite : bool = False, **kwargs):
        """# 2023-01-08 23:05:40 """
        if flag_overwrite or ( not self._s3.exists( path_dest, **kwargs ) ) :  # avoid overwriting of the existing file
            return self._s3.mv(path_src, path_dest, recursive=flag_recursive, **kwargs)
        else:
            raise FileExistsError( "destionation file already exists" )

    def s3_cp(self, path_src : str, path_dest : str, flag_recursive: bool = True, **kwargs):
        """# 2023-01-08 23:05:40 """
        if is_s3_url(path_src) and is_s3_url(path_dest):  # copy from s3 to s3
            return self._s3.copy(path_src, path_dest, recursive=flag_recursive, **kwargs)
        elif is_s3_url(path_src):  # copy from s3 to local
            return self._s3.get(path_src, path_dest, recursive=flag_recursive, **kwargs)
        elif is_s3_url(path_dest):  # copy from local to s3
            return self._s3.put(path_src, path_dest, recursive=flag_recursive, **kwargs)

    def s3_isdir(self, path_src : str, **kwargs):
        """# 2023-01-08 23:05:40 """
        return self._s3.isdir(path_src)
    
    def s3_listdir(self, path_src : str, **kwargs):
        """# 2023-01-08 23:05:40 """
        return self._s3.ls(path_src)
    
    def exists( self, path_src : str, ** kwargs ) :
        ''' # 2023-11-07 16:56:47  '''
        if is_s3_url( path_src ) :
            res = self.s3_exists( path_src, **kwargs )
        elif is_http_url( path_src ) :
            res = self.http_exists( path_src, ** kwargs )
        else :
            res = self.local_exists( path_src, ** kwargs )
        return res
    
    def read_file( self, path_src : str, mode : str = 'rt', ** kwargs ) :
        ''' # 2023-11-07 16:56:47  '''
        if is_s3_url( path_src ) :
            res = self.s3_read_file( path_src, mode, **kwargs )
        elif is_http_url( path_src ) :
            res = self.http_read_file( path_src, mode, ** kwargs )
        else :
            res = self.local_read_file( path_src, mode, ** kwargs )
        return res
    
    def write_file( self, path_src : str, mode : str = 'wt', content = '', ** kwargs ) :
        ''' # 2023-11-07 16:56:47  '''
        if is_s3_url( path_src ) :
            res = self.s3_write_file( path_src, mode, content, **kwargs )
        elif is_http_url( path_src ) :
            raise NotImplementedError( 'http write_file not implemented' )
        else :
            res = self.local_write_file( path_src, mode, content, ** kwargs )
            
    def read_json_file( self, path_src : str ) :
        ''' # 2023-11-07 16:56:47  '''
        return json.loads( self.read_file( path_src, 'rt' ) ) # return the parsed json file
    
    def write_json_file( self, path_src : str, dict_json : dict, ) :
        ''' # 2023-11-07 16:56:47  '''
        self.write_file( path_src, 'wt', json.dumps( dict_json ) ) # write the encoded json file
            
    def mkdir( self, path_src : str, ** kwargs ) :
        """ # 2023-11-07 18:24:34 """
        if is_s3_url( path_src ) :
            res = self.s3_mkdir( path_src, **kwargs )
        elif is_http_url( path_src ) :
            raise NotImplementedError( 'http remove not implemented' )
        else :
            res = self.local_mkdir( path_src, ** kwargs )
    
    def rm( self, path_src : str, flag_recursive : bool = True, ** kwargs ) :
        ''' # 2023-11-07 16:56:47  '''
        if is_s3_url( path_src ) :
            res = self.s3_rm( path_src, flag_recursive, **kwargs )
        elif is_http_url( path_src ) :
            raise NotImplementedError( 'http remove not implemented' )
        else :
            res = self.local_rm( path_src, flag_recursive, ** kwargs )
            
    def glob( self, path_src : str, flag_recursive : bool = False, ** kwargs ) :
        ''' # 2023-11-07 16:56:47  '''
        if is_s3_url( path_src ) :
            res = self.s3_glob( path_src, flag_recursive = flag_recursive, **kwargs )
        elif is_http_url( path_src ) :
            raise NotImplementedError( 'http glob not implemented' )
        else :
            res = self.local_glob( path_src, flag_recursive = flag_recursive, ** kwargs )
        return res
        
    def mv( self, path_src : str, path_dest : str, ** kwargs ) :
        ''' # 2023-11-07 16:56:47  '''
        if is_s3_url( path_src ) and is_s3_url( path_dest ) : # both files should be on s3
            res = self.s3_mv( path_src, path_dest, **kwargs )
        elif is_http_url( path_src ) :
            raise NotImplementedError( 'http mv not implemented' )
        else :
            res = self.s3_mv( path_src, path_dest, ** kwargs )
        
    def cp( self, path_src : str, path_dest : str, flag_recursive : bool = True, ** kwargs ) :
        ''' # 2023-11-07 16:56:47  '''
        if is_s3_url( path_src ) or is_s3_url( path_dest ) : # at least one file is on s3
            res = self.s3_cp( path_src, path_dest, flag_recursive, **kwargs )
        elif is_http_url( path_src ) :
            raise NotImplementedError( 'http cp not implemented' )
        else :
            res = self.local_cp( path_src, path_dest, flag_recursive, ** kwargs )
            
    def isdir( self, path_src : str, ** kwargs ) :
        ''' # 2023-11-07 16:56:47  '''
        if is_s3_url( path_src ) :
            res = self.s3_isdir( path_src, **kwargs )
        elif is_http_url( path_src ) :
            raise NotImplementedError( 'http isdir not implemented' )
        else :
            res = self.local_isdir( path_src, ** kwargs )
        return res
    
    def listdir( self, path_src : str, ** kwargs ) :
        ''' # 2023-11-07 16:56:47  '''
        if is_s3_url( path_src ) :
            res = self.s3_listdir( path_src, **kwargs )
        elif is_http_url( path_src ) :
            raise NotImplementedError( 'http listdir not implemented' )
        else :
            res = self.local_listdir( path_src, ** kwargs )
        return res
    
    def get_zarr_metadata(self, path_src : str, **kwargs):
        """# 2023-01-08 23:05:40 ❤️ test
        """
        return dict( zarr.open( path_src ).attrs )
    
    def local_rm_files_async(self, l_path_file: List[ str ], flag_recursive : bool = True) -> None:
        """ # 2023-09-24 19:42:55 
        read local files asynchronously
        """
        loop = get_or_create_eventloop()
        loop.run_until_complete( asyncio.gather( * list( remove_local_file_async(path_file, flag_recursive) for path_file in l_path_file ) ) )
    
    def local_check_files_async(self, l_path_file: List[ str ]) -> List[ bool ]:
        """ # 2023-09-24 19:42:55 
        read local files asynchronously
        """
        loop = get_or_create_eventloop()
        return loop.run_until_complete( asyncio.gather( * list( check_local_file_async(path_file) for path_file in l_path_file ) ) )
    
    def local_read_files_async(self, l_path_file: List[ str ], mode : str = 'rt') -> List[ str ]:
        """ # 2023-09-24 19:42:55 
        read local files asynchronously
        """
        loop = get_or_create_eventloop()
        result = loop.run_until_complete( asyncio.gather( * list( read_local_file_async(path_file, mode) for path_file in l_path_file ), return_exceptions = True ) )
        result = list( ( None if isinstance( res, FileNotFoundError ) else res ) for res in result ) # if 'FileNotFoundError' is returned, convert it to None
        return result
    
    def local_write_files_async(self, dict_path_file_to_content : dict, mode : str = 'wt') :
        """ # 2023-09-24 19:42:55 
        write local files asynchronously
        """
        loop = get_or_create_eventloop()
        loop.run_until_complete( asyncio.gather( * list( write_local_file_async(path_file, mode, dict_path_file_to_content[path_file]) for path_file in dict_path_file_to_content ) ) )
    
    def http_read_files_async(self, l_path_file: List[ str ], mode : str = 'rt' ) -> List[ str ]:
        """ # 2023-09-24 19:42:55 
        read remote http files asynchronously
        """
        result = asyncio.run( fetch_http_files_async( l_path_file, mode = mode ) )
        return result
    
    def http_check_files_async(self, l_path_file: List[ str ] ) -> List[ bool ]:
        """ # 2023-09-24 19:42:55 
        read remote http files asynchronously
        """
        result = asyncio.run( check_http_files_async( l_path_file ) )
        return result
    
    def s3_check_files_async( self, l_path_file : List[ str ] ) :
        """ # 2023-09-24 23:13:15 
        """
        result = asyncio.run( exists_s3_files_async( l_path_file, self._dict_kwargs_s3 ) )
        return result
    
    def s3_read_files_async( self, l_path_file : List[ str ], mode : str = 'rt' ) :
        """ # 2023-09-24 23:13:15 
        """
        result = asyncio.run( read_s3_files_async( l_path_file, self._dict_kwargs_s3 ) )
        result = list( ( None if isinstance( res, FileNotFoundError ) else res ) for res in result ) # if 'FileNotFoundError' is returned, convert it to None
        if mode != 'rb' : # if not reading binary input files, decode the output files
            result = list( res if res is None else res.decode( ) for res in result )
        return result

    def s3_put_files_async( self, l_path_file_local : List[ str ], l_path_file_remote : List[ str ] ) :
        """ # 2023-09-24 23:15:06 
        """
        result = asyncio.run( put_s3_files_async( l_path_file_local, l_path_file_remote, self._dict_kwargs_s3 ) )
        return result
    
    def s3_rm_files_async( self, l_path_file : List[ str ], flag_recursive : bool = True ) :
        """ # 2023-09-24 23:15:06 
        """
        result = asyncio.run( rm_s3_files_async( l_path_file, flag_recursive = flag_recursive ) )
        return result
    
    def s3_write_files_async( self, dict_path_file_to_content : dict, mode : str = 'wt' ) :
        """ # 2023-09-24 19:42:55 
        write s3 files asynchronously
        """
        ''' write contents to local files temporarily '''
        l_path_file_local, l_path_file_remote, dict_path_file_local_to_content = [ ], [ ], dict( )
        for path_file_remote in dict_path_file_to_content :
            path_file_local = self._path_folder_temp + UUID( )
            l_path_file_local.append( path_file_local )
            l_path_file_remote.append( path_file_remote )
            dict_path_file_local_to_content[ path_file_local ] = dict_path_file_to_content[ path_file_remote ]
        
        self.local_write_files_async( dict_path_file_to_content = dict_path_file_local_to_content, mode = mode )
        ''' upload local files to s3 '''
        self.s3_put_files_async( l_path_file_local = l_path_file_local, l_path_file_remote = l_path_file_remote )
        ''' delete temporary files '''
        for path_file_local in l_path_file_local :
            self.local_rm( path_file_local )
            
    def write_files( self, dict_path_file_to_content : dict, mode : str = 'wt' ) -> None :
        ''' # 2023-11-07 13:21:59 
        write files asynchronously
        '''
        ''' split for each type of resource '''
        dict_type_rsc_to_dict_path_file_to_content = { 
            's3' : dict( ),
            'http' : dict( ),
            'local' : dict( ),
        }
        for path_file in dict_path_file_to_content :
            if is_s3_url( path_file ) :
                dict_type_rsc_to_dict_path_file_to_content[ 's3' ][ path_file ] = dict_path_file_to_content[ path_file ]
            elif is_http_url( path_file ) :
                dict_type_rsc_to_dict_path_file_to_content[ 'http' ][ path_file ] = dict_path_file_to_content[ path_file ]
            else :
                dict_type_rsc_to_dict_path_file_to_content[ 'local' ][ path_file ] = dict_path_file_to_content[ path_file ]
        
        ''' write files for each type of resource '''
        # http not yet implemented
        self.local_write_files_async(
            dict_path_file_to_content = dict_type_rsc_to_dict_path_file_to_content[ 'local' ], 
            mode = mode,
        )
        self.s3_write_files_async(
            dict_path_file_to_content = dict_type_rsc_to_dict_path_file_to_content[ 's3' ], 
            mode = mode,
        )
        
    def _classify_l_path_file( self, l_path_file : List[ str ] ) -> List[ List[ int ] ] :
        """ # 2023-11-07 14:09:43 
        classify_l_path_file, and return l_idx_for_s3, l_idx_for_http, l_idx_for_local
        """
        l_idx_for_s3, l_idx_for_http, l_idx_for_local = [ ], [ ], [ ]
        for idx, path_file in enumerate( l_path_file ) :
            if is_s3_url( path_file ) :
                l_idx_for_s3.append( idx )
            elif is_http_url( path_file ) :
                l_idx_for_http.append( idx )
            else :
                l_idx_for_local.append( idx )
        return l_idx_for_s3, l_idx_for_http, l_idx_for_local
        
    def remove_files( self, l_path_file : List[ str ], flag_recursive : bool = True ) -> None :
        ''' # 2023-11-07 13:21:59 
        remove files asynchronously
        '''
        l_idx_for_s3, l_idx_for_http, l_idx_for_local = self._classify_l_path_file( l_path_file ) # split for each type of resource
        # perform operations for each type of resource
        # http not yet implemented
        self.s3_rm_files_async( list( l_path_file[ i ] for i in l_idx_for_s3 ) )
        self.local_rm_files_async( list( l_path_file[ i ] for i in l_idx_for_local ) )
        
    def read_files( self, l_path_file : List[ str ], mode : str = 'rt' ) -> List[ str ] :
        ''' # 2023-11-07 13:21:59 
        read files asynchronously
        '''
        l_idx_for_s3, l_idx_for_http, l_idx_for_local = self._classify_l_path_file( l_path_file ) # split for each type of resource
        # perform operations for each type of resource
        l_res_s3 = self.s3_read_files_async( list( l_path_file[ i ] for i in l_idx_for_s3 ), mode = mode )
        l_res_http = self.http_read_files_async( list( l_path_file[ i ] for i in l_idx_for_http ), mode = mode )
        l_res_local = self.local_read_files_async( list( l_path_file[ i ] for i in l_idx_for_local ), mode = mode )
        l_res = list( None for _ in range( len( l_path_file ) ) ) # initialize the result
        for l_res_current_rsc, l_idx_current_rsc in zip( 
            [ l_res_s3, l_res_http, l_res_local, ],
            [ l_idx_for_s3, l_idx_for_http, l_idx_for_local, ],
        ) : # combine the results into a single output
            for res, idx in zip( l_res_current_rsc, l_idx_current_rsc ) :
                l_res[ idx ] = res
        return l_res
        
    def check_files( self, l_path_file : List[ str ] ) -> List[ bool ] :
        ''' # 2023-11-07 13:21:59 
        check files asynchronously
        '''
        l_idx_for_s3, l_idx_for_http, l_idx_for_local = self._classify_l_path_file( l_path_file ) # split for each type of resource
        l_res_s3 = self.s3_check_files_async( list( l_path_file[ i ] for i in l_idx_for_s3 ) )
        l_res_http = self.http_check_files_async( list( l_path_file[ i ] for i in l_idx_for_http ) )
        l_res_local = self.local_check_files_async( list( l_path_file[ i ] for i in l_idx_for_local ) )
        l_res = list( None for _ in range( len( l_path_file ) ) ) # initialize the result
        for l_res_current_rsc, l_idx_current_rsc in zip( 
            [ l_res_s3, l_res_http, l_res_local, ],
            [ l_idx_for_s3, l_idx_for_http, l_idx_for_local, ],
        ) : # combine the results into a single output
            for res, idx in zip( l_res_current_rsc, l_idx_current_rsc ) :
                l_res[ idx ] = res
        return l_res
    
    def read_json_files( self, l_path_file : List[ str ] ) -> List[ dict ] :
        ''' # 2023-11-07 21:42:26 '''
        return list( content if content is None else json.loads( content ) for content in self.read_files( l_path_file, 'rt' ) ) # return the parsed json files
    
    def write_json_files( self, dict_path_file_to_dict_json : dict ) -> None :
        ''' # 2023-11-07 21:42:33  '''
        self.write_files( dict( ( path_file, json.dumps( dict_path_file_to_dict_json[ path_file ] ) ) for path_file in dict_path_file_to_dict_json ), 'wt' ) # write the encoded json file
    
class ZarrObject:
    """# 2023-09-24 17:50:46 
    A class for hosting Zarr object in a spawned, managed process for accessing remote objects in forked processes
    API functions calls mimic those of a zarr object for seamless replacement of a zarr object.

    path_folder_zarr : str # a path to a (remote) zarr object
    mode : str = 'r' # mode

    path_process_synchronizer : Union[ str, None ] = None # path to the process synchronizer. if None is given, does not use any synchronizer
    proxy_object = None, # proxy object of 'ZarrObject'
    flag_array : bool = True, # open array
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
            
    def open_array( self, * args, ** kwargs ) :
        ''' #2023-11-06 22:47:29 
        open array
        '''
        self.open( * args, ** kwargs, flag_array = True )
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
        flag_array : bool = True, # open array
    ):
        """# 2023-04-20 02:08:57
        open a new zarr in a ZarrServer object

        reload : bool = False # if True, reload the zarr object even if the 'path_folder' and 'mode' are identical to the currently opened Zarr object. (useful when folder has been updated using the external methods.)
        flag_array : bool = True, # open array
        """
        if self._proxy_object is None :
            # if the zarr object is already opened in the same mode, exit, unless 'reload' flag has been set to True.
            if not reload and path_folder_zarr == self.path_folder and self._mode == mode:
                return

            # open a zarr object
            if mode != "r":  # create a new zarr object
                if (
                    ( shape is None ) or ( chunks is None ) or ( not flag_array )
                ):  # if one of the arguments for opening zarr array is invalid, use open convenience function
                    za = zarr.open(path_folder_zarr, mode)
                else:  # open zarr array
                    za = zarr.open_array(
                        path_folder_zarr,
                        mode,
                        shape=shape,
                        chunks=chunks,
                        dtype=dtype,
                        fill_value=fill_value,
                    )
            else:  # use existing zarr object
                za = zarr.open_array(path_folder_zarr, mode)  if flag_array else zarr.open(path_folder_zarr, mode) 
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
                flag_array = flag_array,
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
        flag_array : bool = True,
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
                reload = False, # for __init__, reload = False
                flag_array = flag_array,
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
    
class SpinLockFileHolder:
    """# 2023-11-06 23:04:15 
    A class for a file system-based locking

    === arguments ===
    flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock : bool = False # if True, does not wait and raise 'RuntimeError' when a modification of a RamData cannot be made due to the resource that need modification is temporarily unavailable, locked by other processes
    float_second_to_wait_before_checking_availability_of_a_spin_lock : float = 0.5 # number of seconds to wait before repeatedly checking the availability of a spin lock if the lock has been acquired by other operations.
    fso = None, # proxy object of 'FileSystemOperator'
    verbose : bool = False # an arugment for debugging purpose
    """

    def __init__(
        self,
        flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock: bool = False,
        float_second_to_wait_before_checking_availability_of_a_spin_lock: float = 0.1,
        fso = None, # proxy object of 'FileSystemOperator'
        dict_kwargs_s3 : dict = dict( ),
        verbose: bool = False,
    ):
        """# 2023-11-06 23:15:34  """
        # set attributes
        self._fso = FileSystemOperator( dict_kwargs_s3 = dict_kwargs_s3 ) if fso is None else fso # set 'FileSystemOperator'
        self._str_uuid_lock = UUID( )  # a unique id of the current SpinLockFileHolder object. This id will be used to acquire and release locks so that lock can only be released by the object that acquired the lock
        self.verbose = verbose

        # set attributes that can be changed anytime during the lifetime of the object
        self.flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock = flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock
        self.float_second_to_wait_before_checking_availability_of_a_spin_lock = float_second_to_wait_before_checking_availability_of_a_spin_lock

        # initialize a set for saving the list of lock objects current SpinLockFileHolder has acquired in order to ignore additional attempts to acquire the lock that has been already acquired
        self._set_path_file_lock = set( )

    @property
    def str_uuid_lock(self):
        """# 2022-12-11 14:04:21
        return a unique id of the current SpinLockFileHolder object
        """
        return self._str_uuid_lock

    @property
    def currently_held_locks(self):
        """# 2022-12-11 16:56:33
        return a copy of a set containing path_file_lock of all the lock objects current SpinLockFileHolder has acquired.
        """
        return set(self._set_path_file_lock)

    def check_lock(self, path_file_lock: str):
        """# 2022-12-10 21:32:38
        check whether the lock currently exists, based on the file system where the current lock object resides.

        path_file_lock : str # an absolute (full-length) path to the lock (an absolute path to the zarr object, representing a spin lock)
        """
        # if lock has been already acquired, return True
        if path_file_lock in self.currently_held_locks :
            return True
        # return the flag indicating whether the lock exists
        return self._fso.exists( path_file_lock )

    def wait_lock(self, path_file_lock: str):
        """# 2022-12-10 21:32:38
        wait for the lock, based on the file system where the current lock object resides.

        path_file_lock : str # an absolute (full-length) path to the lock (an absolute path to the zarr object, representing a spin lock)
        """
        if self.verbose:
            logger.info(
                f"the current SpinLockFileHolder ({self.str_uuid_lock}) is trying to wait for the lock '{path_file_lock}', with currently_held_locks '{self.currently_held_locks}'"
            )

        # if a lock for 'path_file_lock' has been already acquired, does not wait for the lock
        if path_file_lock in self.currently_held_locks:
            return

        # if lock is available and 'flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock' is True, raise a RuntimeError
        if (
            self.check_lock(path_file_lock)
            and self.flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock
        ):
            raise RuntimeError(f"a lock is present at ({path_file_lock}), exiting")
        # implement a spin lock using the sleep function
        while self.check_lock(path_file_lock):  # until a lock is released
            time.sleep(
                self.float_second_to_wait_before_checking_availability_of_a_spin_lock
            )  # wait for 'float_second_to_wait_before_checking_availability_of_a_spin_lock' second

    def acquire_lock(self, path_file_lock: str):
        """# 2023-11-07 17:45:05  
        acquire the lock, based on the file system where the current lock object resides.

        === arguments ===
        path_file_lock : str # an absolute (full-length) path to the lock (an absolute path to the zarr object, representing a spin lock)

        === returns ===
        return True if a lock has been acquired (a lock object was created).
        """
        if self.verbose:
            logger.info(
                f"the current SpinLockFileHolder ({self.str_uuid_lock}) is trying to acquire the lock '{path_file_lock}', with currently_held_locks '{self.currently_held_locks}'"
            )
        if (
            path_file_lock not in self.currently_held_locks
        ):  # if the lock object has not been previously acquired by the current object
            # create the lock zarr object
            while True:
                # attempts to acquire a lock
                # check whether the lock exists
                flag_lock_exists = self.check_lock( path_file_lock )
                
                if not flag_lock_exists : # if the lock object already exists, acquiring lock would fail
                    # attempts to acquire the lock
                    self._fso.write_json_file( path_file_lock, {"str_uuid_lock": self.str_uuid_lock, "time": int(time.time())} )

                    # wait 'sufficient' time to ensure the acquired lock became visible to all other processes that have attempted to acquire the same lock when the lock has not been acquired by any other objects. (waiting the outcome)
                    # check again that the written lock belongs to the current object
                    # when two processes attempts to acquire the same lock object within a time window of 1 ms, two processes will write the same lock object, but one will be overwritten by another.
                    # therefore, the content of the lock should be checked again in order to ensure then the lock has been actually acquired.
                    time.sleep(
                        self.float_second_to_wait_before_checking_availability_of_a_spin_lock * ( 1 + np.random.random( ) )
                    )  # wait for a random interval of time, roughly 1.5 * 'float_second_to_wait_before_checking_availability_of_a_spin_lock' second
                    lock_metadata = self._fso.read_json_file( path_file_lock )  # read the content of the written lock
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
                        f"a lock is present at ({path_file_lock}), exiting"
                    )
                    return False
                # implement a spin lock using the sleep function
                time.sleep(
                    self.float_second_to_wait_before_checking_availability_of_a_spin_lock
                )  # wait for 'float_second_to_wait_before_checking_availability_of_a_spin_lock' second

            # record the 'path_file_lock' of the acquired lock object
            self._set_path_file_lock.add(path_file_lock)
            if self.verbose:
                logger.info(
                    f"the current SpinLockFileHolder ({self.str_uuid_lock}) acquired the lock '{path_file_lock}', with currently_held_locks '{self.currently_held_locks}'"
                )
            return True  # return True if a lock has been acquired
        else:
            return False  # return False if a lock has been already acquired prior to this function call

    def release_lock(self, path_file_lock: str):
        """# 2023-11-07 22:35:52 
        release the lock, based on the file system where the current lock object resides

        path_file_lock : str # an absolute (full-length) path to the lock (an absolute path to the zarr object, representing a spin lock)
        """
        if self.verbose:
            logger.info(
                f"the current SpinLockFileHolder ({self.str_uuid_lock}) is trying to release_lock the lock '{path_file_lock}', with currently_held_locks '{self.currently_held_locks}'"
            )
        if (
            path_file_lock in self.currently_held_locks
        ):  # if the lock object has been previously acquired by the current object
            lock_metadata = self._fso.read_json_file( path_file_lock )  # retrieve the lock metadata
            if lock_metadata is not None and "str_uuid_lock" in lock_metadata:
                if (
                    lock_metadata["str_uuid_lock"] == self.str_uuid_lock
                ):  # if the lock has been acquired by the current object
                    self._fso.rm( path_file_lock )  # release the lock
                    if self.verbose:
                        logger.info(
                            f"the current SpinLockFileHolder ({self.str_uuid_lock}) released the lock '{path_file_lock}'"
                        )
                else:
                    logger.error(
                        f"the current SpinLockFileHolder ({self.str_uuid_lock}) have acquired the lock {path_file_lock} but it appears the lock belongs to another SpinLockFileHolder ({lock_metadata[ 'str_uuid_lock' ]})."
                    )
                    # raise KeyError( f"the current SpinLockFileHolder ({self.str_uuid_lock}) have acquired the lock {path_file_lock} but it appears the lock belongs to another SpinLockFileHolder ({lock_metadata[ 'str_uuid_lock' ]})." )
            else:
                logger.error(
                    f"the current SpinLockFileHolder ({self.str_uuid_lock}) have acquired the lock {path_file_lock} but the lock object does not exist."
                )
                # raise FileNotFoundError( f"the current SpinLockFileHolder ({self.str_uuid_lock}) have acquired the lock {path_file_lock} but the lock object does not exist." )
            self._set_path_file_lock.remove(
                path_file_lock
            )  # remove the released lock's 'path_file_lock' from the list of the acquired lock objects
    
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
    
    def create_spinlockfileholder( 
        self,
        flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock: bool = False,
        float_second_to_wait_before_checking_availability_of_a_spin_lock: float = 0.1,
        verbose: bool = False,
    ) :
        """ # 2023-11-06 23:35:53 
        return SpinLockFileHolder object
        
        flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock: bool = False,
        float_second_to_wait_before_checking_availability_of_a_spin_lock: float = 0.1,
        verbose: bool = False,
        """
        return SpinLockFileHolder( fso = self.get_operator( ), flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock = flag_does_not_wait_and_raise_error_when_modification_is_not_possible_due_to_lock, float_second_to_wait_before_checking_availability_of_a_spin_lock = float_second_to_wait_before_checking_availability_of_a_spin_lock, verbose = verbose )
            
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
        
    def zarr_open_array( self, * args, ** kwargs ) :
        """ # 2023-11-06 22:26:14 
        open a Zarr Array Object
        """
        if self.flag_spawn :
            return ZarrObject( proxy_object = getattr( self._l_mfs[ np.random.randint( self._int_num_processes ) ][ 'manager' ], 'ZarrObject' )( * args, flag_array = True, ** kwargs ) ) # open a proxy object of ZarrObject using one of the spawned, managed processes, and wrap the proxy object using ZarrObject
        else :
            return ZarrObject( * args, flag_array = True, ** kwargs ) # open ZarrObject
        
