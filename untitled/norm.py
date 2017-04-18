import pylab
from numpy import array, linalg, random, sqrt, inf


def plotUnitCircle(p):
    for i in range(50000):
        x = array([random.rand() * 2 - 1, random.rand() * 2 - 1])
        if linalg.norm(x, p) < 1:
            pylab.plot(x[0], x[1], 'gD')
    pylab.axis([-1.5, 1.5, -1.5, 1.5])


pylab.figure(0)
plotUnitCircle(inf)
pylab.savefig("/home/yohai/Documents/IML/normInf.jpg")
pylab.figure(1)
plotUnitCircle(1)
pylab.savefig("/home/yohai/Documents/IML/norm1.jpg")
