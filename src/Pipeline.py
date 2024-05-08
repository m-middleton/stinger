
import time
import os
import json
import joblib

import numpy as np

from sklearn.model_selection import train_test_split

from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import mutual_info_regression, f_regression, r_regression
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor



def peak_signal_to_noise_ratio_score(estimator, x, y):
    """Compute the peak signal to noise ratio (PSNR) for a signal.
    Parameters
    ----------
    true : ndarray
        Ground-truth image.
    pred : ndarray
        Reconstructed image.
    Returns
    -------
    psnr : float
        The PSNR metric.
    """
    pred = estimator.predict(x)

    y = np.array(y, dtype=np.float64)
    pred = np.array(pred, dtype=np.float64)

    mse = np.mean((y - pred) ** 2)
    if mse == 0:
        return np.inf

    return 10 * np.log10(np.max(y) ** 2 / mse)

def start_pipeline_signal_prediction( 
                X,
                y,
                params_model,
                column_names,
                dt,
                train_size=5000,
                test_size=1000,
                features_to_use=20,
                n_search=20,
                save_model = False,
                save_model_name = '',
                model_path = './output/',
                model_stats={}
):
    # params_models - create one dictionary per classification algorithm containing the hyperparameters to try within the randomized search
    # n_folds_outer - number of outer folds: percentage of samples reserved for test set will be 100/n_folds_inner
    # n_folds_inner - number of inner folds: percentage of samples reserved for test set will be 100/n_folds_inner
    # n_search - number of combinations of scaling/feature selection/balancing/models to evaluate to find best combination

    # make list of named pipeline steps in the order in which they should be applied to the feature matrix
    if features_to_use == -1:
        features_to_use = 'all'
    steps = [
      ('scaler', None),
      ('mutal_info', SelectKBest(mutual_info_regression, k=features_to_use)),
      ('model', RandomForestRegressor(random_state=1, n_jobs=-1)),
    ]

    # create a dictionary specifying different types of scaling, feature selection, and class balancing that you want to try within the randomized search
    params_preprocessing = {
                            'scaler': [StandardScaler(),]#MinMaxScaler()],
                           }

    # Because continous data, dont do this
    #cv_outer = StratifiedKFold(n_splits=n_folds_outer, shuffle=True, random_state=42)

    # for each classification algorithm, merge the dictionary for preprocessing with the dictionary for that model
    params = [params_preprocessing | param for param in params_model] # list comprehension

    train_index = train_size+1
    X_train, y_train = X[:train_index], y[:train_index]
    X_test, y_test = X[train_index:train_index+test_size], y[train_index:train_index+test_size]

    print(y_train.shape)
    print(y_test.shape)

    X_train, _, y_train, _  = train_test_split(
        X_train, 
        y_train, 
        train_size=train_size,
        test_size=1,
        shuffle=False)

    print(f'train: x:{X_train.shape} y:{y_train.shape}')
    print(f'test: x:{X_test.shape} y:{y_test.shape}')

    pipeline = Pipeline(steps)
    if n_search == -1:
        search = GridSearchCV(pipeline,
                                params,
                                scoring='r2',#peak_signal_to_noise_ratio_score,
                                refit=True,
                                n_jobs=-1)
    else:
        search = RandomizedSearchCV(pipeline,
                                    params,
                                    n_iter=n_search,
                                    scoring='r2',#peak_signal_to_noise_ratio_score,
                                    refit=True,
                                    n_jobs=-1,
                                    random_state=0)

    # search = GridSearchCV(pipeline,
    #                         params,
    #                         scoring=peak_signal_to_noise_ratio_score,
    #                         refit=True,
    #                         n_jobs=3,
    #                         cv=3)

    search.fit(X_train, y_train)
    model = search.best_estimator_

    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    print(f"train:{score_train}, test:{score_test}")

    psnr_train = peak_signal_to_noise_ratio_score(model, X_train, y_train)
    psnr_test = peak_signal_to_noise_ratio_score(model, X_test, y_test)
    print(f"psnr_train:{psnr_train}, psnr_test:{psnr_test}")

    best_params = search.best_params_
    best_model = model

    # Feature selection
    best_features_translated = ['nan']
    if features_to_use != 'all':
        best_features_fold = model['mutal_info'].get_support()
        best_features_fold = np.array(best_features_fold)
        best_features_fold = best_features_fold.reshape((features_to_use,dt*2))

        best_features_translated = []
        for ind in range(best_features_fold.shape[0]):
            if np.any(best_features_fold[ind]):
                best_features_translated.append(column_names[ind])
                print(column_names[ind])

    best_params['r2_train'] = score_train
    best_params['r2_test'] = score_test
    best_params['psnr_train'] = psnr_train
    best_params['psnr_test'] = psnr_test
    best_params['train_size'] = train_size
    best_params['test_size'] = test_size
    best_params['selected_features'] = best_features_translated

    best_params = best_params | model_stats
    best_params = {k:str(v) for k,v in best_params.items()}

    if save_model:
        save_model_name = f'train_{score_train:.2f}_test_{score_test:.2f}_{save_model_name}'
        file_path = os.path.join(model_path, save_model_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        joblib.dump(model, os.path.join(file_path, f'{save_model_name}.pkl'))
        with open(os.path.join(file_path, f'{save_model_name}_stats.json'), 'w') as fp:
            json.dump(dict(best_params), fp)
    
    return best_model, best_params
