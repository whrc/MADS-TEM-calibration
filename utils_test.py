import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from unittest import mock

from utils import spaghetti_match_plot, spaghetti_match_plot_r2, plot_r2_rmse, get_params_r2_rmse, find_important_features

def test_spaghetti_match_plot():
    df_x = pd.DataFrame(np.random.rand(10, 5))
    df_y = pd.DataFrame(np.random.rand(11, 5))
    with mock.patch.object(plt, 'show') as mock_show:
        spaghetti_match_plot(df_x, df_y)
        mock_show.assert_called_once()

def test_spaghetti_match_plot_r2():
    df_x = pd.DataFrame(np.random.rand(10, 5))
    df_y = pd.DataFrame(np.random.rand(11, 5))
    with mock.patch.object(plt, 'show') as mock_show:
        spaghetti_match_plot_r2(df_x, df_y)
        mock_show.assert_called_once()

def test_plot_r2_rmse():
    df_y = pd.DataFrame(np.random.rand(11, 5))
    with mock.patch.object(plt, 'show') as mock_show:
        plot_r2_rmse(df_y)
        mock_show.assert_called_once()

def test_find_important_features():
    X = pd.DataFrame(np.random.rand(100, 5), columns=['x1', 'x2', 'x3', 'x4', 'x5'])
    y = pd.Series(np.random.rand(100))
    find_important_features(X, y, ylabel='Test')

def test_get_params_r2_rmse():
    # Create sample data
    x = pd.DataFrame({'param1': [1, 2, 3], 'param2': [4, 5, 6]})
    y = pd.DataFrame({'output1': [0.1, 0.2, 0.3], 'output2': [0.4, 0.5, 0.6], 'output3': [0.7, 0.8, 0.9]})
    # Call the function
    xparams, ymodel = get_params_r2_rmse(x, y)
    print(ymodel.shape)

    # Check that the output is a tuple of dataframes
    assert isinstance(xparams, pd.DataFrame)
    assert isinstance(ymodel, pd.DataFrame)
    # Check that the output dataframes have the correct number of rows and columns
    assert xparams.shape == (2, 5)
    assert ymodel.shape == (2, 3)
    # Check that the output dataframes contain the expected columns
    assert '$R^2$' in xparams.columns
    assert 'RMSE' in xparams.columns
    assert 'MAPE' in xparams.columns
    # Check that the output dataframes contain the expected index
    assert all(xparams.index == [1, 2])
    assert all(ymodel.index == [0, 1])
    # Check that the function works as expected when given a non-default r2lim value
    xparams, ymodel = get_params_r2_rmse(x, y, r2lim=0.1)
    assert xparams.empty
    assert ymodel.empty


test_get_params_r2_rmse()
