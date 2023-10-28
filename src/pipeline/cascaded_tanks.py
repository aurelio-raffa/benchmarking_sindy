"""Code based on https://pysindy.readthedocs.io/en/latest/examples/9_sindypi_with_sympy/example.html#Find-complex-PDE-with-SINDy-PI-with-PDE-functionality
"""
import contextlib
import os
import warnings

import hyperopt
import matplotlib.pyplot as plt
import numpy as np
from fire import Fire
from hyperopt import hp, fmin, tpe

from src.utils.etl import prepare_data
from src.utils.etl.cascaded_tanks import load_data
from src.utils.model_selection.cascaded_tanks import train_validate_model, train_validate_hidden_model


def cascaded_tanks(
        rescale_factor: float = 10.0,
        tvr_gamma: float = 1.0,
        state_space_models_trials: int = 100,
        hidden_models_trials: int = 1000,
        k1_mean: float = 16.0,
        k1_log_std: float = 0.5,
        k3_mean: float = 45.0,
        k3_log_std: float = 0.5,
        z0_lb: float = 0.01,
        z0_ub: float = 0.99,
        alpha_lb: float = 1e-3,
        alpha_ub: float = 1e3,
        threshold_lb: float = 1e-6,
        threshold_ub: float = 1e2,
        seed: int = 42
):
    # 01 - Loading the data
    train_data, validation_data, test_data, dt, t_train, t_val, t_test = load_data()

    # TODO: rescale back when saving the results
    # 01b - Rescaling the data (for numerical stability)
    train_data /= rescale_factor
    validation_data /= rescale_factor
    test_data /= rescale_factor

    # 02 - Computing the derivatives (using TV regularization) and preparing the dataset
    x_train, u_train, x_dot_train, x_ddot_train = prepare_data(
        train_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2
    )
    x_val, u_val, x_dot_val, x_ddot_val = prepare_data(
        validation_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2
    )
    x_test, u_test, x_dot_test, x_ddot_test = prepare_data(
        test_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2
    )

    # 03a - Training the non-informed models
    plt.figure()
    plt.plot(t_test, x_test, label='True')
    for readable_name, model_type, order, integrator_kws in zip(
            ['Naive SINDy', 'SINDy SQRT', 'SINDy 2nd order', 'SINDy 2nd order, SQRT'],
            ['naive', 'sqrt_poly', 'naive', 'sqrt_poly'],
            [1, 1, 2, 2],
            [{}, {}, {'method': 'Radau'}, {'method': 'Radau'}]
    ):
        print(f' {readable_name} '.center(120, '='))

        def validation_rmse(params: dict):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with contextlib.redirect_stderr(open(os.devnull, 'w')):

                    val_rmse = train_validate_model(
                        t_test=t_val,
                        x_train=x_train,
                        u_train=u_train,
                        x_dot_train=x_dot_train,
                        x_test=x_val,
                        u_test=u_val,
                        x_ddot_train=x_ddot_train,
                        model_type=model_type,
                        order=order,
                        integrator_kws=integrator_kws,
                        **params
                    )[-1]

            return val_rmse

        # Hyperparameter tuning
        rstate = np.random.Generator(np.random.PCG64(seed=seed))
        search_space = {
            'alpha': hp.loguniform('alpha', np.log(alpha_lb), np.log(alpha_ub)),
            'threshold': hp.loguniform('threshold', np.log(threshold_lb), np.log(threshold_ub))
        }

        best_parameters = hyperopt.space_eval(
            search_space,
            fmin(
                fn=validation_rmse,
                space=search_space,
                algo=tpe.suggest,
                max_evals=state_space_models_trials,
                rstate=rstate
            )
        )

        # Fitting
        model, simulation, r2, rmse = train_validate_model(
            t_test=t_test,
            x_train=x_train,
            u_train=u_train,
            x_dot_train=x_dot_train,
            x_test=x_test,
            u_test=u_test,
            x_ddot_train=x_ddot_train,
            model_type=model_type,
            order=order,
            integrator_kws=integrator_kws,
            **best_parameters
        )
        model.print()
        print(' Scores: '.center(120, '-'))
        print(f'-> R2: {100 * r2:.2f}%')
        print(f'-> RMSE: {rmse:.4f}')

        plt.plot(t_test, simulation[:, -1], label=readable_name)

    # 03b - Training the models based on hidden state reconstruction
    for readable_name, model_type, integrator_kws in zip(
            ['Hidden SINDy, SQRT', 'Hidden SINDy'],
            ['sqrt', 'naive'],
            [{'method': 'Radau'}, {'method': 'Radau'}]
    ):
        print(f' {readable_name} '.center(120, '='))

        def validation_rmse(params: dict):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    val_rmse = train_validate_hidden_model(
                        t_train=t_train,
                        t_test=t_val,
                        x_train=x_train,
                        u_train=u_train,
                        x_dot_train=x_dot_train,
                        x_test=x_val,
                        u_test=u_val,
                        x_dot_test=x_dot_val,
                        model_type=model_type,
                        integrator_kws=integrator_kws,
                        **params
                    )[-1]

            return val_rmse

        # Hyperparameter tuning
        rstate = np.random.Generator(np.random.PCG64(seed=seed))
        search_space = {
            'alpha': hp.loguniform('alpha', np.log(alpha_lb), np.log(alpha_ub)),
            'threshold': hp.loguniform('threshold', np.log(threshold_lb), np.log(threshold_ub)),
            'k1': hp.lognormal('k1', np.log(k1_mean), k1_log_std),
            'k3': hp.lognormal('k3', np.log(k3_mean), k3_log_std),
            'z0': hp.uniform('z0', z0_lb, z0_ub),
        }

        best_parameters = hyperopt.space_eval(
            search_space,
            fmin(
                fn=validation_rmse,
                space=search_space,
                algo=tpe.suggest,
                max_evals=hidden_models_trials,
                rstate=rstate
            )
        )

        # TODO: save outputs
        # Fitting
        model, simulation, r2, rmse = train_validate_hidden_model(
            t_train=t_train,
            t_test=t_test,
            x_train=x_train,
            u_train=u_train,
            x_dot_train=x_dot_train,
            x_test=x_test,
            u_test=u_test,
            x_dot_test=x_dot_test,
            model_type=model_type,
            integrator_kws=integrator_kws,
            **best_parameters
        )
        model.print()
        print(' Scores: '.center(120, '-'))
        print(f'-> R2: {100 * r2:.2f}%')
        print(f'-> RMSE: {rmse:.4f}')

        plt.plot(t_test, simulation[:, -1], label=readable_name)

    # TODO: plot with right routine
    plt.legend()
    plt.show()


if __name__ == '__main__':
    Fire(cascaded_tanks)
