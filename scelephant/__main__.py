#!/usr/bin/env python
# the date of last modification
_version_ = '0.0.0'
_last_modified_time_ = '2022-06-09 21:14:47'
from biobookshelf.main import *
from biobookshelf import *
import biobookshelf as bk

pd.options.mode.chained_assignment = None  # default='warn' # to disable worining
import warnings
warnings.filterwarnings( action = 'ignore' )

# retrieve a function for logging
Info = multiprocessing.get_logger( ).info

    
if __name__ == "__main__" :
    pass 