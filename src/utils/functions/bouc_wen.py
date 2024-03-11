import numpy as np


def identity(x):
    return x


def xidentity(z, x):
    return x


def z_xabs(z, x):
    return z * np.abs(x)


def zabs_x(z, x):
    return np.abs(z) * x


def zabs_xabs(z, x):
    return np.abs(z) * np.abs(x)


def identity_name(x):
    return x


def xidentity_name(z, x):
    return x


def z_xabs_name(z, x):
    return f'{z} * |{x}|'


def zabs_x_name(z, x):
    return f'|{z}| * {x}'


def zabs_xabs_name(z, x):
    return f'|{z}| * |{x}|'
