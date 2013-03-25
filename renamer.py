#!/usr/bin/env python
from __future__ import division
__author__ = 'Horea Christian'

# This script removes specified rows of a (questionnaire) results matrix - commonly for the protection of privacy
# in cases where questionnaire websites automatically and invariably annotate information which could endanger complete anonymity.
#
# USAGE:
# From the command line: Call the script (add execute permission with "chmod +x sg_anon.py" if needed). Within the CLI yoou should 
# also:
#     1) Specify the absolute path to the directory containing your results (and/or) filter files (sub-directories excluding 
#     ".backup" will also be searched.  
#     2) specify the numbers of the column you want deleted (number zero being the first row), separated by a space
# 
# EXAMPLE:
#    ./sg_anon.py ~/my/data/for/this/experiment/ 0 1 3 4 5 6 7 8 9 10 11 12 13 14 15 17 18 19

import binascii
from os import walk, path, listdir, urandom, renames

globalpath = '~/Data/face-pictures/'
folder_in = 'sets/'
folder_out = 'fin/'
instances = ['q','w','e','r','t','z','u','i','o','p','a','s','d']

globalpath = path.expanduser(globalpath)
folder = globalpath+folder_in
names=[]
if path.isdir(folder):
    for leroot, dirs, files in walk(folder, topdown=False):
        for dire in dirs:
            idtf = binascii.b2a_hex(urandom(4))
            raw_files = [s for s in listdir(leroot+'/'+dire) if s.rpartition('.')[2] in ('NEF','nef')]
            if path.basename(leroot) == 'male':
                gender = 'm'
            elif path.basename(leroot) == 'female':
                gender = 'f'
            for ix, filename in enumerate(sorted(raw_files)):
                oldname = leroot+'/'+dire+'/'+filename
                newname = globalpath+folder_out+path.basename(path.dirname(leroot))+'/'+gender+idtf+'_'+instances[ix]+'.NEF'
                renames(oldname, newname)
elif path.isfile(folder):
    names = [folder]
else:
    exit('The location specified does not exist. Please ensure you typed the path to your files in correctly.')