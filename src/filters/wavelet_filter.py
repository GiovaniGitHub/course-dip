import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pywt

def bayes_thresh(details, var):
    dvar = np.mean(details*details)
    eps = np.finfo(details.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    return thresh


def sigma_estimate(detail_coeffs):

    detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]
    denom = stats.norm.ppf(0.75)
    sigma = np.median(np.abs(detail_coeffs)) / denom

    return sigma


def wavelet_threshold(image, wavelet = 'db4', sigma=None, wavelet_levels=None):
    wavelet = pywt.Wavelet(wavelet)

    original_extent = [slice(s) for s in image.shape]

    if wavelet_levels is None:
        wavelet_levels = np.min(
            [pywt.dwt_max_level(s, wavelet.dec_len) for s in image.shape])

        wavelet_levels = max(wavelet_levels - 3, 1)

    coeffs = pywt.wavedecn(image, wavelet=wavelet, level=wavelet_levels)
    dcoeffs = coeffs[1:]

    if sigma is None:
        print(dcoeffs[-1].keys())
        detail_coeffs = dcoeffs[-1]['d' * image.ndim]
        sigma = sigma_estimate(detail_coeffs)

    threshold = [
        { key: bayes_thresh(level[key], sigma**2) for key in level}
         for level in dcoeffs]

    if np.isscalar(threshold):
        denoised_detail = [{key: pywt.threshold(level[key],
                                                value=threshold,
                                                mode='soft') for key in level}
                           for level in dcoeffs]
    else:
        denoised_detail = [{key: pywt.threshold(level[key],
                                                value=thresh[key],
                                                mode='soft') for key in level}
                           for thresh, level in zip(threshold, dcoeffs)]

    denoised_coeffs = [coeffs[0]] + denoised_detail

    return pywt.waverecn(denoised_coeffs, wavelet)[tuple(original_extent)]


