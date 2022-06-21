#!/usr/bin/env python
# the date of last modification
from biobookshelf.main import *
from biobookshelf import *
import biobookshelf as bk

from scelephant.core import *

pd.options.mode.chained_assignment = None  # default='warn' # to disable worining
import warnings
warnings.filterwarnings( action = 'ignore' )

# retrieve a function for logging
Info = multiprocessing.get_logger( ).info

if __name__ == "__main__" :
    pass 