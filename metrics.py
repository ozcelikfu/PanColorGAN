import numpy as np
from scipy.ndimage import sobel
from numpy.linalg import norm

def sam(ms, ps):

    assert ms.ndim == 3 and ms.shape == ps.shape
    dot_sum = np.sum(ms * ps, axis=2)
    norm_true = norm(ms, axis=2)
    norm_pred = norm(ps, axis=2)

    res = np.arccos(dot_sum/norm_pred/norm_true)
    is_nan = np.nonzero(np.isnan(res))
    for (x,y) in zip(is_nan[0], is_nan[1]):
        res[x,y]=0

    sam = np.mean(res)
    return sam

def sam2(ms,ps,degs = True):
    result = np.double(ps)
    target = np.double(ms)
    if result.shape != target.shape:
        raise ValueError('Result and target arrays must have the same shape!')

    bands = target.shape[2]
    rnorm = np.sqrt((result ** 2).sum(axis=2))
    tnorm = np.sqrt((target ** 2).sum(axis=2))
    dotprod = (result * target).sum(axis=2)
    cosines = (dotprod / (rnorm * tnorm))
    sam2d = np.arccos(cosines)
    sam2d[np.invert(np.isfinite(sam2d))] = 0.  # arccos(1.) -> NaN
    if degs:
        sam2d = np.rad2deg(sam2d)
    return sam2d[np.isfinite(sam2d)].mean()


def RMSE(result, target):
    result = np.double(result)
    target = np.double(target)
    if result.shape != target.shape:
        raise ValueError('result and target arrays must have the same shape!')
    return ((result - target) ** 2).mean() ** 0.5


def ERGAS(result, target, pixratio=0.5):
    result = np.double(result)
    target = np.double(target)
    if result.shape != target.shape:
        raise ValueError('result and target arrays must have the same shape!')

    bands = target.shape[2]
    addends = np.zeros(bands)
    for band in range(bands):
        addends[band] = ((RMSE(result[:, :, band], target[:, :, band])) / (target[:, :, band].mean())) ** 2
    ergas = 100 * pixratio * ((1.0 / bands) * addends.sum()) ** 0.5

    return ergas


def QAVE(result, target):
    result = np.double(result)
    target = np.double(target)
    if result.shape != target.shape:
        raise ValueError('result and target arrays must have the same shape!')

    rmean = result.mean(axis=2)
    tmean = target.mean(axis=2)

    rmean1 = result[:, :, 0] - rmean
    rmean2 = result[:, :, 1] - rmean
    rmean3 = result[:, :, 2] - rmean
    rmean4 = result[:, :, 3] - rmean

    tmean1 = target[:, :, 0] - tmean
    tmean2 = target[:, :, 1] - tmean
    tmean3 = target[:, :, 2] - tmean
    tmean4 = target[:, :, 3] - tmean

    QR = (1 / result.shape[2] - 1) * (rmean1 ** 2 + rmean2 ** 2 + rmean3 ** 2 + rmean4 ** 2)
    QT = (1 / result.shape[2] - 1) * (tmean1 ** 2 + tmean2 ** 2 + tmean3 ** 2 + tmean4 ** 2)
    QRT = (1 / result.shape[2] - 1) * (rmean1 * tmean1 + rmean2 * tmean2 + rmean3 * tmean3 + rmean4 * tmean4)

    QAVE = result.shape[2] * ((QRT * rmean) * tmean) / ((QR + QT) * ((rmean ** 2) + (tmean ** 2)))
    m, n = QAVE.shape
    Q = (1 / (m * n)) * np.sum(np.sum(QAVE))

    return Q

def sCC(ms, ps):
    ps_sobel = sobel(ps, mode='constant')
    ms_sobel = sobel(ms, mode='constant')


    return  (np.sum(ps_sobel*ms_sobel)/np.sqrt(np.sum(ps_sobel*ps_sobel))/np.sqrt(np.sum(ms_sobel*ms_sobel)))

def UIQI(ms,ps):
    pass

def Q4(ms,ps):
    pass


def CC(ms,ps):
    pass

def rase(ms,ps):
    pass
