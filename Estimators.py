import numpy as np
import pymc as pm
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as st
from typing import Tuple, Optional

def map_estimator(marks: np.ndarray) -> Tuple[pd.Series, pd.Series]:
    """
    Функция генерирует значения параметров performance и difficulty из априорного распределения Beta(3, 3),
    далее задается распределение Bernoulli(p) в качестве правдоподобия.
    После оценивается максимум апостериорного распределения (MAP) с помощью оптимизации и возвращаются Series
    со значениями параметров performance и difficulty, при которых этот максимум достигается.

    Parameters
    ----------
    marks : np.ndarray
        Таблица оценок студентов размера (students_count, units_count), содержащая значения 0, 1 и, возможно, NaN.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Кортеж из двух Series:
        - performance_table: Series с MAP-оценками параметров performance для каждого студента
        - difficulty_table: Series с MAP-оценками параметров difficulty для каждого юнита

    Notes
    -----
    - Априорные распределения: Beta(3, 3) для всех параметров
    - Вероятностная модель: p = 1 - difficulty * (1 - performance)
    - Используется MAP (maximum a posteriori) оценка - режим апостериорного распределения
    - Автоматически обрабатывает пропущенные значения (NaN) в данных
    - По умолчанию используется метод L-BFGS-B для оптимизации
    """
    marks = np.array(marks, dtype=float)
    students_count, units_count = marks.shape
    param_loc = 3.0

    with pm.Model() as model:
        performances = pm.Beta("performance", alpha=param_loc, beta=param_loc, shape=students_count)
        difficulties = pm.Beta("difficulty", alpha=param_loc, beta=param_loc, shape=units_count)

        probs = 1 - difficulties[None, :] * (1 - performances[:, None])

        # Игнорирование NaN
        valid_idx = ~np.isnan(marks)
        
        obs = pm.Bernoulli("obs", probs[valid_idx], observed=marks[valid_idx])
        map_estimate = pm.find_MAP(progressbar=True)

    performance_table = pd.Series(map_estimate['performance'], name='map_perfomance')
    difficulty_table = pd.Series(map_estimate['difficulty'], name='map_difficulty')

    return performance_table, difficulty_table

def mcmc_estimator(marks: np.ndarray) -> Tuple[pd.Series, pd.Series]:
    '''
    Функция генерирует значения параметров performance и difficulty из априорного Бета-распределения,
    далее генерирует случайные величины из апостериорного распределения Бернулли.
    Функция возвращает Series с параметрами performance и difficulty, которые являются выборочными средними
    выборки из апостериорного распределения
    
    Parameters
    ----------
    marks : np.ndarray
        Матрица наблюдений (оценок) размера (students_count, units_count), 
        содержащая значения 0, 1 и, возможно, NaN для пропущенных данных
    
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Кортеж из двух Series:
        - performance_table: Series с выборочными средними параметров performance
        - difficulty_table: Series с выборочными средними параметров difficulty
    
    Notes
    -----
    Использует Bayesian подход с MCMC семплированием:
    - Априорные распределения: Beta(3, 3) для всех параметров
    - Вероятностная модель: p = 1 - difficulty * (1 - performance)
    - Используется 10 семплов для MCMC (может потребоваться увеличение для реальных задач)
    - Автоматически обрабатывает пропущенные значения (NaN) в данных
    '''
    marks = np.array(marks, dtype=float)
    students_count, units_count = marks.shape
    param_loc = 3.0

    with pm.Model() as model:
        performances = pm.Beta("performance", alpha=param_loc, beta=param_loc, shape=students_count)
        difficulties = pm.Beta("difficulty", alpha=param_loc, beta=param_loc, shape=units_count)

        p = 1 - difficulties[None, :] * (1 - performances[:, None])

        valid_idx = ~np.isnan(marks)

        obs = pm.Bernoulli("obs", p=p[valid_idx], observed=marks[valid_idx])

        trace = pm.sample(20, return_inferencedata=False)

    performance = trace['performance'].mean(axis=0)
    difficulty = trace['difficulty'].mean(axis=0)

    performance_table = pd.Series(performance, name='mcmc_performance')
    difficulty_table = pd.Series(difficulty, name='mcmc_difficulty')

    return performance_table, difficulty_table

def _log_likelihood(params: np.ndarray, M: np.ndarray) -> float:
    '''
    Функция принимает на вход параметры performance и difficulty, строит по ним параметр p распределения Бернулли
    и возвращает минус логарифмическую функцию правдоподобия.
    
    Parameters
    ----------
    params : np.ndarray
        Вектор параметров, содержащий сначала параметры perf, затем параметры diff
    M : np.ndarray
        Матрица наблюдений (0 и 1) размера (students_count, units_count)
    
    Returns
    -------
    float
        Минус логарифмическая функция правдоподобия (negative log-likelihood)
    
    Notes
    -----
    - Функция также включает априорные распределения Beta(3,3) для параметров perf и diff
    - Вероятностная модель: p = 1 - difficulty * (1 - performance)
    '''
    a = 3
    b = 3
    
    students_count, _ = M.shape
    performance = params[:students_count]
    difficulty = params[students_count:]

    p = 1 - difficulty * (1 - performance[:, np.newaxis])

    mask = (M == 0.0) | (M == 1.0)
    log_likelihood_val = np.sum(st.bernoulli.logpmf(M[mask], p[mask]))

    log_likelihood_val += np.sum(st.beta.logpdf(difficulty, a, b))
    log_likelihood_val += np.sum(st.beta.logpdf(performance, a, b))
    
    return -log_likelihood_val

def mle_estimator(marks: np.ndarray, initial_params: Optional[np.ndarray] = None) -> Tuple[pd.Series, pd.Series]:
    '''
    Функция возвращает Series с значениями параметров performance и difficulty,
    при которых функция правдоподобия максимальна
    
    Parameters
    ----------
    marks : np.ndarray
        Матрица наблюдений (оценок) размера (students_count, units_count), содержащая значения 0 и 1
    initial_params : Optional[np.ndarray], default=None
        Начальные значения параметров для оптимизации. Должны иметь длину (students_count + units_count),
        где первые students_count элементов - параметры performance (успеваемость),
          остальные - параметры difficulty (сложность).
        Если None, используется вектор из 0.5.
    
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
    '''
    marks = np.array(marks, dtype=float)
    students_count, units_count = marks.shape

    # Установка границ оптимизации
    bounds = [(0, 1)] * (students_count + units_count)

    # Установка параметров по умолчанию
    if initial_params is None:
        initial_params = np.full(students_count + units_count, 0.5)

    # Оптимизация
    result = minimize(_log_likelihood, initial_params, args=(marks, ), method='Powell', bounds=bounds)

    optimized_params = result.x

    performance_table = pd.Series(optimized_params[:students_count], name='mle_performance')
    difficulty_table = pd.Series(optimized_params[students_count:], name='mle_difficulty')

    return performance_table, difficulty_table