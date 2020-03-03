import numpy as _np

def _iter_array(a):
    """Returns an iterable which yields tuples of indexes into the array `a`."""
    return _itertools.product(*[list(range(i)) for i in a.shape])

def _make_cells(region, grid_size, events, times):
    xsize, ysize = region.grid_size(grid_size)
    cells = _np.empty((ysize, xsize), dtype=_np.object)
    for x in range(xsize):
        for y in range(ysize):
            cells[y,x] = []
    xcs = _np.floor((events.xcoords - region.xmin) / grid_size)
    ycs = _np.floor((events.ycoords - region.ymin) / grid_size)
    xcs = xcs.astype(_np.int)
    ycs = ycs.astype(_np.int)
    for i, time in enumerate(times):
        cells[ycs[i], xcs[i]].append(time)
    for x in range(xsize):
        for y in range(ysize):
            cells[y,x] = _np.asarray(cells[y,x])
    return cells

#def predict(self, predict_time, cutoff_time=None):
    """Make a prediction at a time, using the data held by this instance.
    That is, evaluate the background rate plus the trigger kernel at
    events before the prediction time.  Optionally you can limit the data
    used, though this is against the underlying statistical model.

    :param predict_time: Time point to make a prediction at.
    :param cutoff_time: Optionally, limit the input data to only be from
      before this time.

    :return: Instance of :class:`open_cp.predictors.GridPredictionArray`
    """
events = data.events_before(None)
nummins = 30000; # a week
start_date = datetime.datetime(2011,1,2)
predict_times = [start_date + datetime.timedelta(minutes=x) for x in range(nummins)]
arr_lam=[]
arr_mu=[]
for predict_time in predict_times:
    times = (_np.datetime64(predict_time) - events.timestamps) / _np.timedelta64(1, "m")
    cells = _make_cells(region, grid_size, events, times)
    

    dt = cells[36,38]
    dt = dt[dt>0]
    lam = _np.sum(theta * omega * _np.exp(-omega * dt))
    lam += mu[36,38]
    
    arr_lam.append(lam)
    arr_mu.append(mu[36,38])

jumps_1 = nummins - dt

fig, axes = plt.subplots(nrows=1, figsize=(18,8))
#axes.set(xlabel="minutes", ylabel="crime intensity(predicted)")
#axes.set(title="For grid cell [36,38]")
#axes.set(xlim=[0,nummins])
axes.plot(predict_times,arr_lam)
axes.plot(predict_times,arr_mu,linestyle='dashed',c = 'r')
#axes.plot(jumps_1, mu[36,38] * np.ones(len(jumps_1)), linestyle = 'None', marker = '^', c = 'black')
#line2, = axes.plot(predict_times,arr_mu,'--', linewidth=1)
#line3, = axes.plot(jumps_1, mu[36,38] * np.ones(len(jumps_1)), linestyle = 'None', marker = '^', c = 'r')
plt.show()