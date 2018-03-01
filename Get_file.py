import os
def Get_file(path):
    dirpath = []
    dirnames = []
    filenames = []
    for p, dirname, flname in os.walk(path):
        if p == []:
            pass
        else:
            dirpath.append(path)
        if dirname == []:
            pass
        else:
            dirnames = dirname
        if flname == []:
            pass
        else:
            filenames = flname
    return dirpath, dirnames, filenames