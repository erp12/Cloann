import time, glob

outfilename = 'all_' + str((int(time.time()))) + "error_log.txt"
filenames = glob.glob('*error_log.txt')

with open(outfilename, 'w') as outfile:
    for fname in filenames:
        with open(fname, 'r') as readfile:
            print "Merging: ", fname
            outfile.write(readfile.read() + "\n\n")
