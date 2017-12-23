

import timeit
import logging
LOG_FILENAME = 'example.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

logging.debug('This message should go to the log file')


set1 = '''
import okada
import inversion_utilities
inv = inversion_utilities.build_inversion('resample_test')
s = inv.station[0]
'''

set2 = '''
import okada_py
import inversion_utilities
inv = inversion_utilities.build_inversion('resample_test')
s = inv.station[0]
'''
num = 1000
a = timeit.timeit('okada.compute_station_disp(inv, s)', setup=set1, number=num)
b = timeit.timeit('okada_py.compute_station_disp(inv, s)', setup=set2, number=num)

print '{} {} {} {}'.format(a, a/num, b, b/num)
print 'cython is {} faster'.format(b/a)

set= '''
import numpy
import numpy
a = numpy.random.rand(15000,100)
'''

a = timeit.timeit('numpy.linalg.svd(a, compute_uv=False)', setup=set, number=num)
b = a/num
print '{} {}'.format(a, b)
# print 'cython is {} faster'.format(b/a)
