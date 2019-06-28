from apricot import FacilityLocationSelection
import numpy
import time

numpy.random.seed(8)
for n in [500, 1000, 5000]:
    X = numpy.concatenate([numpy.random.normal(7.5, 1, size=(n, 2)),
                           numpy.random.normal(2, 1, size=(n, 2)),
                           numpy.random.normal(15, 1, size=(n, 2))])
    start_time = time.time()
    fl = FacilityLocationSelection(100, 'euclidean')
    fl.fit(X)
    end_time = time.time()
    print("n = ", str(n)," time: " + str(end_time - start_time))
