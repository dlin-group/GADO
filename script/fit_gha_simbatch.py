import os
from optparse import OptionParser
from multiprocessing import Pool
import time

executable  = ''
args        = ''
outfilebase = ''
logfilebase = ''

def task(i):
    global executable, args, outfilebase, logfilebase
    cmd  = executable + ' ' + args
    cmd += ' -r ' + outfilebase + '_' + str(i) + '.Rdata > '
    cmd += logfilebase + '_' + str(i) + '.log'
    print(cmd)
    os.system(cmd)

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-j', '--nthreads', dest='nthreads', type='int',
                      help='number of threads', default=0)
    parser.add_option('-n', '--ntasks', dest='ntasks', type='int',
                      help='number of tasks', default=0)
    parser.add_option('-e', '--executable', dest='executable', type='string',
                      help='path to executable', default='')
    parser.add_option('-o', '--outfilebase', dest='outfilebase',
                      help='base of output filename', default='')
    parser.add_option('-l', '--logfilebase', dest='logfilebase', type='string',
                      help='base of log filename', default='')
    parser.add_option('-a', '--args', dest='args', type='string',
                      help='arguments to executable', default='')
    (options, args) = parser.parse_args()
    if options.nthreads <= 0:
        print('must specify an integer number of threads with -j')
        exit()
    if options.ntasks <= 0:
        print('must specify an integer number of tasks with -n')
        exit()
    if options.executable == '':
        print('must specify an executable with -e')
        exit()
    elif not os.path.exists(options.executable):
        print('executable,', e, 'does not exist')
        exit()
    if options.outfilebase == '':
        print('must specify a base output filename with -o')
        exit()
    elif options.logfilebase == '':
        options.logfilebase = options.outfilebase

    executable  = options.executable
    args        = options.args
    outfilebase = options.outfilebase
    logfilebase = options.logfilebase

    pool = Pool(processes=options.nthreads)
    pool.map(task, range(options.ntasks))
    pool.close()
    pool.join()
