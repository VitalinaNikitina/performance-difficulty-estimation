import numpy as np
import pymc as pm
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as st
from typing import Tuple, Optional

def map_estimator(marks: np.ndarray, aprior_beta_params: list = [3, 3],
                   retakes: bool = False, verbose: bool = False) -> Tuple[pd.Series, pd.Series]:
    """
    Функция оценивает максимум апостериорного распределения (MAP) и возвращают Series
    со значениями параметров performance и difficulty, при которых этот максимум достигается

    Parameters
    ----------
    marks : np.ndarray
        Таблица оценок студентов размера (students_count, units_count)
        - При retakes=False: содержит значения 0 и 1. Допускается NaN
        - При retakes=True: содержит количество попыток до успеха (≥1 для успешных попыток). Допускается NaN
    aprior_beta_params : list, default = [3, 3]
        Содержит два параметра для априорного Бета-распределения. По-умолчанию априорное распределение Beta(3, 3)
    retakes : bool, optional, default = False
        Если True, используется модель с учетом пересдач, где наблюдаемые данные моделируются как смесь
        распределений Бернулли и геометрического. Если False, используется стандартная модель Бернулли
    verbose : bool, optional, default = False
        Если True, будет показываться прогресс оптимизации

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Кортеж из двух Series:
        - performance_table: Series с MAP-оценками параметров performance для каждого студента
        - difficulty_table: Series с MAP-оценками параметров difficulty для каждого юнита

    Notes
    -----
    - Вероятностная модель: p = 1 - difficulty * (1 - performance)
    - При retakes=True используется смешанная модель:
        * Компонента 1: Бернулли с очень малой вероятностью (для неудачных попыток)
        * Компонента 2: Геометрическое распределение (моделирует количество попыток до успеха)
    - Используется MAP (maximum a posteriori) оценка - режим апостериорного распределения
    - Автоматически обрабатывает пропущенные значения (NaN) в данных
    - По умолчанию используется метод L-BFGS-B для оптимизации
    """
    marks = np.array(marks, dtype=float)
    students_count, units_count = marks.shape
    alpha = aprior_beta_params[0]
    beta= aprior_beta_params[1]

    with pm.Model() as model:
        performances = pm.Beta("performance", alpha=alpha, beta=beta, shape=students_count)
        difficulties = pm.Beta("difficulty", alpha=alpha, beta=beta, shape=units_count)

        p = 1 - difficulties[None, :] * (1 - performances[:, None])

        # Игнорирование NaN
        valid_idx = ~np.isnan(marks)
        
        if retakes:
            obs = pm.Mixture(
                "obs",
                w=pm.math.stack((1 - p[valid_idx], p[valid_idx]), axis=1),
                comp_dists=[
                    pm.Bernoulli.dist(p=np.full(np.sum(valid_idx), 0.00000001)),
                    pm.Geometric.dist(p=p[valid_idx])
                ],
                observed=marks[valid_idx],
            )
        else:
            obs = pm.Bernoulli("obs", p[valid_idx], observed=marks[valid_idx])

        map_estimate = pm.find_MAP(progressbar=verbose)

    performance_table = pd.Series(map_estimate['performance'], name='map_perfomance')
    difficulty_table = pd.Series(map_estimate['difficulty'], name='map_difficulty')

    return performance_table, difficulty_table

def mcmc_estimator(marks: np.ndarray, aprior_beta_params: list = [3, 3],
                   retakes: bool = False, n_samples: int = 20,
                   verbose: bool = False) -> Tuple[pd.Series, pd.Series]:
    '''
    Функция возвращает Series с параметрами performance и difficulty, которые являются выборочными средними
    выборки из апостериорного распределения
    
    Parameters
    ----------
    marks : np.ndarray
        Матрица наблюдений (оценок) размера (students_count, units_count)
        - При retakes=False: содержит значения 0 и 1. Допускается NaN
        - При retakes=True: содержит количество попыток до успеха (≥1 для успешных попыток). Допускается NaN
    aprior_beta_params : list, default = [3, 3]
        Содержит два параметра для априорного Бета-распределения. По-умолчанию априорное распределение Beta(3, 3).
    retakes : bool, optional, default = False
        Если True, используется модель с учетом пересдач, где наблюдаемые данные моделируются как смесь
        распределений Бернулли и геометрического. Если False, используется стандартная модель Бернулли
    n_samples : int, optional, default = 20
        Количество семплов для MCMC семплирования. Большее количество семплов может улучшить точность оценки,
        но увеличит время выполнения
    verbose : bool, optional, default = False
        Если True, будет показываться прогресс семплирования
    
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Кортеж из двух Series:
        - performance_table: Series с выборочными средними параметров performance
        - difficulty_table: Series с выборочными средними параметров difficulty
    
    Notes
    -----
    Использует Bayesian подход с MCMC семплированием:
    - Вероятностная модель: p = 1 - difficulty * (1 - performance)
    - При retakes=True используется смешанная модель:
        * Компонента 1: Бернулли с очень малой вероятностью (для неудачных попыток)
        * Компонента 2: Геометрическое распределение (моделирует количество попыток до успеха)
    - Используется указанное количество семплов для MCMC (по умолчанию 20)
    - Автоматически обрабатывает пропущенные значения (NaN) в данных
    '''
    marks = np.array(marks, dtype=float)
    students_count, units_count = marks.shape
    alpha = aprior_beta_params[0]
    beta= aprior_beta_params[1]

    with pm.Model() as model:
        performances = pm.Beta("performance", alpha=alpha, beta=beta, shape=students_count)
        difficulties = pm.Beta("difficulty", alpha=alpha, beta=beta, shape=units_count)

        p = 1 - difficulties[None, :] * (1 - performances[:, None])

        valid_idx = ~np.isnan(marks)

        if retakes:
            obs = pm.Mixture(
                "obs",
                w=pm.math.stack((1 - p[valid_idx], p[valid_idx]), axis=1),
                comp_dists=[
                    pm.Bernoulli.dist(p=np.full(np.sum(valid_idx), 0.00000001)),
                    pm.Geometric.dist(p=p[valid_idx])
                ],
                observed=marks[valid_idx],
            )
        else:
            obs = pm.Bernoulli("obs", p[valid_idx], observed=marks[valid_idx])

        trace = pm.sample(n_samples, return_inferencedata=False, progressbar=verbose)

    performance = trace['performance'].mean(axis=0)
    difficulty = trace['difficulty'].mean(axis=0)

    performance_table = pd.Series(performance, name='mcmc_performance')
    difficulty_table = pd.Series(difficulty, name='mcmc_difficulty')

    return performance_table, difficulty_table

def _log_likelihood(params: np.ndarray, M: np.ndarray, retakes: bool = False, aprior_beta_params: list = [3, 3],) -> float:
    '''
    Функция принимает на вход параметры performance и difficulty
    и возвращает минус логарифмическую функцию правдоподобия
    
    Parameters
    ----------
    params : np.ndarray
        Вектор параметров, содержащий сначала параметры perf, затем параметры diff
    M : np.ndarray
        Матрица наблюдений размера (students_count, units_count)
        - При retakes=False: содержит значения 0 и 1. Допускается NaN
        - При retakes=True: содержит количество попыток до успеха (≥1 для успешных попыток). Допускается NaN
    retakes : bool, optional, default = False
        Если True, используется модель с учетом пересдач со смесью распределений
    aprior_beta_params : list, default = [3, 3]
        Содержит два параметра для априорного Бета-распределения. По-умолчанию априорное распределение Beta(3, 3)
    
    Returns
    -------
    float
        Минус логарифмическая функция правдоподобия (negative log-likelihood)
    
    Notes
    -----
    - Вероятностная модель: p = 1 - difficulty * (1 - performance)
    - При retakes=True используется смесь:
        * Бернулли с p=1e-8 (неудачные попытки)
        * Геометрического (количество попыток до успеха)
    '''
    alpha = aprior_beta_params[0]
    beta= aprior_beta_params[1]
    
    students_count, units_count = M.shape
    performance = params[:students_count]
    difficulty = params[students_count:students_count + units_count]

    p = 1 - difficulty * (1 - performance[:, np.newaxis])

    mask = ~np.isnan(M)
    
    if retakes:
        log_likelihood_val = 0.0
        
        for i in range(students_count):
            for j in range(units_count):
                if not np.isnan(M[i, j]):
                    p_val = p[i, j]
                    observation = M[i, j]
                    
                    # Веса компонент смеси
                    w1 = 1 - p_val  # вес Бернулли (неудача)
                    w2 = p_val      # вес Геометрического (успех после k попыток)
                    
                    # Логарифмы плотностей компонент
                    log_bern = st.bernoulli.logpmf(0, 1e-8) if observation == 0 else -np.inf
                    
                    log_geom = st.geom.logpmf(observation, p_val) if observation >= 1 else -np.inf
                    
                    # Логарифм смеси: log(w1 * exp(log_bern) + w2 * exp(log_geom))
                    # logsumexp для численной стабильности
                    log_mixture = np.logaddexp(
                        np.log(w1) + log_bern,
                        np.log(w2) + log_geom
                    )
                    
                    log_likelihood_val += log_mixture
    else:
        mask_binary = (M == 0.0) | (M == 1.0)
        log_likelihood_val = np.sum(st.bernoulli.logpmf(M[mask_binary], p[mask_binary]))

    log_likelihood_val += np.sum(st.beta.logpdf(difficulty, alpha, beta))
    log_likelihood_val += np.sum(st.beta.logpdf(performance, alpha, beta))
    
    return -log_likelihood_val

def mle_estimator(marks: np.ndarray, aprior_beta_params: list = [3, 3],
                   retakes: bool = False, initial_params: Optional[np.ndarray] = None
                  ) -> Tuple[pd.Series, pd.Series]:
    '''
    Функция возвращает Series с значениями параметров performance и difficulty,
    при которых функция правдоподобия максимальна
    
    Parameters
    ----------
    marks : np.ndarray
        Матрица наблюдений (оценок) размера (students_count, units_count)
        - При retakes=False: содержит значения 0 и 1. Допускается NaN
        - При retakes=True: содержит количество попыток до успеха (≥1 для успешных попыток). Допускается NaN
    aprior_beta_params : list, default = [3, 3]
        Содержит два параметра для априорного Бета-распределения. По-умолчанию априорное распределение Beta(3, 3).
    retakes : bool, optional, default = False
        Если True, используется модель с учетом пересдач, где наблюдаемые данные 
        интерпретируются как количество попыток до успешной сдачи
    initial_params : Optional[np.ndarray], default=None
        Начальные значения параметров для оптимизации. Должны иметь длину (students_count + units_count),
        где первые students_count элементов - параметры performance (успеваемость),
          остальные - параметры difficulty (сложность)
        Если None, используется вектор из 0.5
    
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Кортеж из двух Series:
        - performance_table: Series с оптимизированными параметрами performance
        - difficulty_table: Series с оптимизированными параметрами difficulty
    
    Notes
    -----
    Использует метод Powell для минимизации отрицательного логарифмического правдоподобия.
    Все параметры ограничены диапазоном [0, 1].
    - При retakes=False: используется модель Бернулли для бинарных исходов
    - При retakes=True: используется модель, учитывающая количество попыток до успеха
    - Вероятностная модель: p = 1 - difficulty * (1 - performance)
    '''
    marks = np.array(marks, dtype=float)
    students_count, units_count = marks.shape

    # Установка границ оптимизации
    bounds = [(0, 1)] * (students_count + units_count)

    # Установка параметров по умолчанию
    if initial_params is None:
        initial_params = np.full(students_count + units_count, 0.5)

    # Оптимизация
    result = minimize(_log_likelihood, initial_params, args=(marks, retakes, aprior_beta_params), method='Powell', bounds=bounds)

    optimized_params = result.x

    performance_table = pd.Series(optimized_params[:students_count], name='mle_performance')
    difficulty_table = pd.Series(optimized_params[students_count:], name='mle_difficulty')

    return performance_table, difficulty_table
