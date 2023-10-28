import numpy as np
import pysindy as ps

from src.utils.functions import threshold


def get_model(
        model_type: str,
        order: int = 1,
        z: bool = False,
        u: bool = True,
        **model_kwargs
):
    if order == 1:
        feature_names = ['x']
    elif order == 2:
        feature_names = ['xd', 'x']
    else:
        raise ValueError(f'Possible valiues for "order" are 1 and 2!')

    if z:
        feature_names.append('z')
    if u:
        feature_names.append('u')

    optimizer = ps.STLSQ(**model_kwargs)

    if model_type == 'naive':
        model = ps.SINDy(
            feature_library=ps.PolynomialLibrary(include_bias=False),
            feature_names=feature_names,
            optimizer=optimizer
        )
    elif model_type == 'sqrt' or model_type == 'sqrt_poly':
        sqrt_library = ps.CustomLibrary(
            library_functions=[lambda x: np.sqrt(threshold(x))],
            function_names=[lambda x: f'sqrt({x})']
        )
        if model_type == 'sqrt':
            linear_library = ps.PolynomialLibrary(include_bias=False, degree=1)
            sindy_sqrt_library = ps.GeneralizedLibrary(libraries=[linear_library, sqrt_library])
        else:
            poly_library = ps.PolynomialLibrary(include_bias=False)
            sindy_sqrt_library = ps.GeneralizedLibrary(libraries=[poly_library, sqrt_library])
        model = ps.SINDy(
            feature_names=feature_names,
            feature_library=sindy_sqrt_library,
            optimizer=optimizer
        )
    else:
        raise NotImplementedError(f'Unknown model type: "{model_type}"!')

    return model
