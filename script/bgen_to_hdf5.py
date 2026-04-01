import os
from optparse import OptionParser
from multiprocessing import Pool
import time

executable  = ''
args        = ''
infilebase  = ''

def task(i):
    global executable, args, infilebase
    cmd  = executable + ' ' + args
    cmd += ' -i ' + infilebase + str(i) + '.bgen > '
    cmd += infilebase + str(i) + '.log'
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
    parser.add_option('-i', '--infilebase', dest='infilebase',
                      help='base of input filename', default='')
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
    if options.infilebase == '':
        print('must specify a base input filename with -i')
        exit()

    executable = options.executable
    args       = options.args
    infilebase = options.infilebase

    pool = Pool(processes=options.nthreads)
    pool.map(task, range(1, options.ntasks+1))
    pool.close()
    pool.join()
