import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import argparse
import pdb
import datetime

def find_glitches(data):
    grad = np.gradient(data)
    print grad
    med = np.median(grad)
    print "median = %f"%med
    return np.where (grad != med)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Set up the beamformer debug visualiser. \
        This code takes in a foldername of a beamformer observation.\
        You can als enter a start time and end time to display debug info for a partucular range of\
        times after the observation started.')
    parser.add_argument('-d', '--directory', default = None, help = 'Folder name of observation', dest='directory')
    parser.add_argument('-s', '--start', default = None, help='Start time of visualisation in seconds from the beginning of the observation', dest='start')
    parser.add_argument('-e', '--end', default = None, help = 'End time of visualisation in seconds from the beginning of the observation', dest='end')
    parser.add_argument('-f', '--find-glitches', default = None, help = 'Find timestamp discrepencies', dest='find_glitches')
    args = parser.parse_args()

    print "args = %s"%str(args)

    files = None
    if args.directory:
        files = glob.glob("%s/*.dat"%args.directory)
        files = [f for f in files if "obs_info" not in f]
    else:
        print "No files specified"
        sys.exit()

    for f in files:
        print f

    sTimestamp = None

    if args.start:
        sTimestamp = float(args.start)
        print "sTimestamp = %i"%sTimestamp

    eTimestamp = None

    if args.end:
        eTimestamp = float(args.end)
        print "eTimestamp = %i"%eTimestamp

    ts = int(files[0].split('/')[-1].split('.')[0])

    first_count = ts
    print ts
    if (args.start):
        st = ts + sTimestamp * 12207.031250
    else:
        st = 0
    if (args.end):
        et = ts + eTimestamp * 12207.031250
    else:
        et = sys.float_info.max

    # print ("ts = %d and st = %f and et = %f"%(ts, st,et))
    start = False

    filecount = 1;

        
    for f in files:

        print "-------------File %d----------------------"%filecount
        ts = int(f.split('/')[-1].split('.')[0])
        te = ts + 10 * 12207.031250

        # print ("ts = %d and st = %f and et = %f"%(ts, st,et))

        if (args.start == None or (te >= st and et > ts) ) :
            # println ('oo')

            ts_data = np.fromfile("%s.ts"%f,dtype=np.uint64)
            #print "first timestamp = %i and last timestamp = %i a difference of %i"%(ts_data[0],ts)
            ofs_data = np.fromfile("%s.ofs"%f,dtype=np.uint64)
            indices = np.where((ts_data >= st) & (ts_data < et))

            print ("%i samples"%indices[0].shape)
            print indices

            markers = find_glitches(ofs_data[indices])
            print "MARKERS"
            print markers[0]
            print ts_data[indices][markers]

            if (markers[0].shape[0] < 100):

                for m in markers[0]:
                    print "Glitch at %f s into obs"%((ts_data[indices][m] - first_count)/12207.031250)


            if markers[0].shape[0] > 0:

                # ts_datimes = [datetime.fromtimestamp(t) for t in ts_data]
                x_range = np.arange(0,ts_data.shape[0])
                plt.figure(1)
                plt.suptitle("debug info from file %s starting at %f"%(f,(ts_data[indices][0] - first_count)/12207.031250))
                plt.subplot(121)
                plt.plot(x_range[indices], ts_data[indices], 'g-')
                plt.plot(x_range[indices][markers], ts_data[indices][markers], 'rD')
                plt.title("Timestamps")
                plt.xlabel("Positon in file")
                plt.ylabel("Raw timestamp")
                plt.subplot(122)
                plt.plot(x_range[indices], ofs_data[indices], 'g-')
                plt.plot(x_range[indices][markers], ofs_data[indices][markers], 'rD')
                plt.title("Offsets")
                plt.xlabel("Positon in file")
                plt.ylabel("Raw offset")
                plt.show()
        filecount+=1
        # else:
            # print "%i not in range"%ts





