#!/usr/bin/env python
#
# Paul's Extreme Sound Stretch (Paulstretch) - Python version
#
# by Nasca Octavian PAUL, Targu Mures, Romania
# http://www.paulnasca.com/
#
# http://hypermammut.sourceforge.net/paulstretch/
#
# this file is released under Public Domain
#
#


import sys
from numpy import *
import soundfile as sf
from optparse import OptionParser, OptionGroup
from datetime import datetime

def print_inputfile_stats(filename):
    try:
        wavedata, samplerate = sf.read(filename)
        print(sf.info(filename), True)
        print('frames: {}'.format(len(wavedata)))
    except:
        print('Error loading file: {!s} {!r}'.format(filename, sys.exc_info()[0]))
        return None

def braid_frames(frames, braid_on):
    print('Braid frames, change on every {} frame'.format(braid_on))
    ar = [[], []]
    i = 0
    b_c_0 = 0
    b_c_1 = 1
    for a in frames:
        ar[b_c_0].append(a[0])
        ar[b_c_1].append(a[1])
        i = i + 1
        if i == braid_on:
            if b_c_0 == 0:
                b_c_0 = 1
                b_c_1 = 0
            else:
                b_c_0 = 0
                b_c_1 = 1
            i = 0
    
    return ar

def reverse_on_frame(frames, reverse_on):
    print('Reverse frames, change on every {} frame'.format(reverse_on))
    r_buf = [[], []]
    r_o = 0
    i = 0
    do_revers = False
    l = []
    r = []
    while i < len(frames[0]):
        l.append(frames[0][i])
        r.append(frames[1][i])
        i = i + 1
        r_o = r_o + 1
        if (i == len(frames[0])) or r_o == reverse_on:
            if do_revers:
                l = l[::-1]
                r = r[::-1]
                do_revers = False
            else:
                do_revers = True
            
            r_buf[0].extend(l)
            r_buf[1].extend(r)
            l = []
            r = []
            r_o = 0

    return r_buf

def reverse_frames(frames):
    print('Reverse input file')
    frames[0] = frames[0][::-1]
    frames[1] = frames[1][::-1]
    return frames

def reverse_frames_left_channel(frames):
    print('Reverse left channel on input file')
    frames[0] = frames[0][::-1]
    return frames

def reverse_frames_right_channel(frames):
    print('Reverse right channel on input file')
    frames[1] = frames[1][::-1]
    return frames

def load_wav(filename, start_frame, end_frame, reverse_input, reverse_input_left, reverse_input_right, braid, braid_on, reverse_on):
    try:
        print('\n#Input\n')
        print(sf.info(filename), True)
        
        wavedata, samplerate = sf.read(filename, start=start_frame, stop=end_frame)
    except:
        print('Error loading file: {!s} {!r}'.format(filename, sys.exc_info()[0]))
        return None
    else:
        print('frames read: {}'.format(len(wavedata)))
        print('start_frame: {}'.format(start_frame))
        print('end_frame: {}'.format(end_frame or len(wavedata)))
        
        output_decorations = ''

        print('Correcting input data layout: {} frames'.format(len(wavedata)))

        ar = [[], []]

        if braid:
            output_decorations = output_decorations + '_b{}'.format(braid_on)
            ar = braid_frames(wavedata, braid_on)
        else:
            for a in wavedata:
                ar[0].append(a[0])
                ar[1].append(a[1])
        
        if reverse_on > 0:
            output_decorations = output_decorations + '_ro{}'.format(reverse_on)
            ar = reverse_on_frame(ar, reverse_on)

        if reverse_input:
            output_decorations = output_decorations + '_r'
            ar = reverse_frames(ar)
       
        if reverse_input_left:
            output_decorations = output_decorations + '_rl'
            ar = reverse_frames_left_channel(ar)

        if reverse_input_right:
            output_decorations = output_decorations + '_rr'
            ar = reverse_frames_right_channel(ar)

        wavedata = array(ar)

        print('{} frames corrected'.format(len(wavedata[0])))

        return (samplerate, wavedata, output_decorations)

def optimize_windowsize(n):
    orig_n = n
    while True:
        n = orig_n
        while (n%2) == 0:
            n /= 2
        while (n%3) == 0:
            n /= 3
        while (n%5) == 0:
            n /= 5

        if n < 2:
            break
        orig_n += 1
    return orig_n

def paulstretch(samplerate, smp, stretch, windowsize_seconds, outfilename):
    nchannels = smp.shape[0]

    print('\n#Streching\n')

    print('output file: {!s}\nsamplerate: {} Hz\nchannels: {}\n'.format(outfilename, samplerate, nchannels))
    outfile = sf.SoundFile(outfilename, 'w', samplerate, nchannels)

    #make sure that windowsize is even and larger than 16
    windowsize = int(windowsize_seconds*samplerate)
    if windowsize < 16:
        windowsize = 16
    windowsize = optimize_windowsize(windowsize)
    windowsize = int(windowsize/2) * 2
    half_windowsize = int(windowsize/2)

    #correct the end of the smp
    nsamples = smp.shape[1]

    end_size = int(samplerate*0.05)
    if end_size < 16:
        end_size = 16

    smp[:,nsamples-end_size:nsamples] *= linspace(1, 0, end_size)

    
    #compute the displacement inside the input file
    start_pos = 0.0
    displace_pos = (windowsize*0.5) / stretch

    #create Window window
#    window=0.5-cos(arange(windowsize,dtype='float')*2.0*pi/(windowsize-1))*0.5

    window = pow(1.0-pow(linspace(-1.0, 1.0, windowsize), 2.0), 1.25)

    old_windowed_buf = zeros((2, windowsize))
#    hinv_sqrt2=(1+sqrt(0.5))*0.5
#    hinv_buf=2.0*(hinv_sqrt2-(1.0-hinv_sqrt2)*cos(arange(half_windowsize,dtype='float')*2.0*pi/half_windowsize))/hinv_sqrt2
    
    start_time = datetime.now()
    
    while True:
        #get the windowed buffer
        istart_pos = int(floor(start_pos))
        buf = smp[:,istart_pos:istart_pos+windowsize]
        if buf.shape[1] < windowsize:
            buf=append(buf, zeros((2, windowsize-buf.shape[1])), 1)
        buf = buf * window
    
        #get the amplitudes of the frequency components and discard the phases
        freqs = abs(fft.rfft(buf))

        #randomize the phases by multiplication with a random complex number with modulus=1
        ph = random.uniform(0,2*pi, (nchannels,freqs.shape[1])) * 1j
        freqs = freqs * exp(ph)

        #do the inverse FFT 
        buf = fft.irfft(freqs)

        #window again the output buffer
        buf *= window

        #overlap-add the output
        output = buf[:,0:half_windowsize] + old_windowed_buf[:,half_windowsize:windowsize]
        old_windowed_buf = buf

        #remove the resulted amplitude modulation
        #update: there is no need to the new windowing function
        #output*=hinv_buf
        
        #clamp the values to -1..1 
        output[output>1.0] = 1.0
        output[output<-1.0] = -1.0

        
        #write the output to wav file
        d = int16(output.ravel(1)*32767.0)
        d = array_split(d, len(d)/2)

        outfile.write(d)

        start_pos += displace_pos
        if start_pos >= nsamples:
            print("100 %")
            break
        
        sys.stdout.write('{} % \r'.format(int(100.0*start_pos/nsamples)))
        sys.stdout.flush()

    outfile.close()
    print('Streched in: {}'.format(datetime.now() - start_time))
    print('\n#Output\n')
    print(sf.info(outfilename), True)

########################################

if __name__ == "__main__":
    print("Paul's Extreme Sound Stretch (Paulstretch) - Python version 20141220")
    print("by Nasca Octavian PAUL, Targu Mures, Romania\n")
    parser = OptionParser(usage="usage: %prog [options] input_file output_file(optional)")

    paulstrech_options = OptionGroup(parser, 'Paulstrech Options')
    paulstrech_options.add_option("-s", "--stretch", dest="stretch", help="stretch amount (1.0 = no stretch), above 0.0", type="float", default=8.0)
    paulstrech_options.add_option("-w", "--window_size", dest="window_size", help="window size (seconds), above 0.001", type="float", default=0.25)
    parser.add_option_group(paulstrech_options)
    
    input_mani_options = OptionGroup(parser, 'Input File Manipulation Options')
    input_mani_options.add_option("-t", "--start_frame", dest="start_frame", help="Start read on frame", type="int", default=0)
    input_mani_options.add_option("-e", "--end_frame", dest="end_frame", help="End read on frame", type="int", default=None)
    input_mani_options.add_option("-r", "--reverse", action="store_true", dest="reverse_input", help="Reverse input file", default=False)
    input_mani_options.add_option("--reverse_left", action="store_true", dest="reverse_input_left", help="Reverse left channel on input file", default=False)
    input_mani_options.add_option("--reverse_right", action="store_true", dest="reverse_input_right", help="Reverse right channel on input file", default=False)
    input_mani_options.add_option("-b","--braid", action="store_true", dest="braid", help="Braid right and left channels by frame", default=False)
    input_mani_options.add_option("--braid_on", dest="braid_on", help="Braid on frame (default=1), must be used with braid option", type="int", default=1)
    input_mani_options.add_option("--reverse_on", dest="reverse_on", help="Reverse on frame (default=1)", type="int", default=0)
    parser.add_option_group(input_mani_options)

    parser.add_option("-d","--dont_stretch", action="store_true", dest="dont_stretch", help="Dont stretch file, just store output and exit", default=False)
    parser.add_option("-i", "--input_file_stat", action="store_true", dest="input_file_stat", help="Print inputfile stat and then exit", default=False)
    parser.add_option("-l", "--list_supported_types", action="store_true", dest="list_supported_types", help="List all supported input file types and then exit", default=False)
    parser.add_option("--output_name", dest="output_name", help="Output name added to decorations if not on command line", type="string", default="")

    (options, args) = parser.parse_args()

    if options.list_supported_types:
        print('Supported file types:\n')
        for file_type, desc in sf.available_formats().items():
            print('{} - {}'.format(file_type, desc))
        sys.exit(0)

    if len(args) > 0 and options.input_file_stat:
        print_inputfile_stats(args[0])
        sys.exit(0) 

    if len(args) < 1:
        print("Error in command line parameters. Run this program with --help for help.")
        sys.exit(1)
    elif len(args) == 1:
        outputfile = ""
    else:
        outputfile = args[1]

    (samplerate, smp, output_decorations) = load_wav(
        args[0], 
        options.start_frame, 
        options.end_frame, 
        options.reverse_input, 
        options.reverse_input_left, 
        options.reverse_input_right, 
        options.braid,
        options.braid_on,
        options.reverse_on
        )

    if options.dont_stretch:
        output_decorations = output_decorations + '_d'
        frames = []
        for i in range(len(smp[0])):
            frames.append([smp[0][i], smp[0][i]])

        if outputfile == "":
            infile_split = args[0].split(".")
            if options.output_name == "":
                outputfile = infile_split[0]
            outputfile = outputfile + options.output_name + output_decorations  + '.' + infile_split[len(infile_split)-1]

        sf.write(outputfile, frames, samplerate, 'PCM_16')
        print('\n#Output\n')
        print(sf.info(outputfile), True)
    elif (options.stretch > 0.0) and (options.window_size > 0.001):
        output_decorations = output_decorations + '_s{}_w{}'.format(options.stretch, options.window_size)
        print('stretch amount = {}'.format(options.stretch))
        print('window size = {} seconds'.format(options.window_size))

        if outputfile == "":
            infile_split = args[0].split(".")
            if options.output_name == "":
                outputfile = infile_split[0]
            outputfile = outputfile + options.output_name + output_decorations + '.' + infile_split[len(infile_split)-1]

        paulstretch(samplerate, smp, options.stretch, options.window_size, outputfile)
    else:
        print("Error in command line parameters. Run this program with --help for help.")
        sys.exit(1)
