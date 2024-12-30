import sys
import time
import pdb

def my_print(atuple, sep=' ', end='\n', file=None, use_pdb=False):
    fline = sys._getframe().f_back.f_lineno
    fname = sys._getframe(1).f_code.co_filename
    content = "\033[35m **my_print: {} #{}:\033[0m".format(fname, fline)
    if use_pdb:
        pdb.set_trace()
    print(content, end=" ")
    print(atuple)


