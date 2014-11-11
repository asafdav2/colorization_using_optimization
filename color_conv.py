import numpy as np


def rgb2yiq(rgb):
    rgb = rgb / 255.0
    y = np.clip(np.dot(rgb, np.array([0.299, 0.587, 0.144])),             0,   1)
    i = np.clip(np.dot(rgb, np.array([0.595716, -0.274453, -0.321263])), -0.5957, 0.5957)
    q = np.clip(np.dot(rgb, np.array([0.211456, -0.522591, 0.311135])),  -0.5226, 0.5226)
    yiq = rgb
    yiq[..., 0] = y
    yiq[..., 1] = i
    yiq[..., 2] = q
    return yiq


def yiq2rgb(yiq):
    r = np.dot(yiq, np.array([1.0,  0.956295719758948,  0.621024416465261]))
    g = np.dot(yiq, np.array([1.0, -0.272122099318510, -0.647380596825695]))
    b = np.dot(yiq, np.array([1.0, -1.106989016736491,  1.704614998364648]))
    rgb = yiq
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return np.clip(rgb, 0.0, 1.0) * 255.0
