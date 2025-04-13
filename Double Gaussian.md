Хорошо, давайте всё соберём в единый, связный документ, включающий описание модели, её кода и краткие комментарии. 

---

## Описание задачи

У нас есть два датчика, которые измеряют температуру:
1. На **блюмсе**: фиксирует только внешнюю температуру \( T_{\text{внешняя}}(t) \), порождая сигнал \( T_{\text{блюмс}}(t) \).
2. На **борту**: фиксирует как внешнюю температуру \( T_{\text{внешняя}}(t) \), так и влияние температуры внутреннего источника \( T_{\text{внутренний}}(t) \), порождая сигнал \( T_{\text{борт}}(t) \).

Цель: используя данные с обоих датчиков, выделить два компонента:
1. **Внешнюю температуру** \( T_{\text{внешняя}}(t) \),
2. **Внутреннюю температуру** \( T_{\text{внутренний}}(t) \), исключив влияние внешних факторов.

## Модельные предположения

Мы предполагаем:
1. Внешняя температура \( T_{\text{внешняя}}(t) \) изменяется во времени плавно и может быть аппроксимирована с помощью гауссовского случайного блуждания (**Gaussian Random Walk**).
2. Внутренний источник \( T_{\text{внутренний}}(t) \) также изменяется плавным образом, но с меньшей амплитудой изменений.
3. Шумы датчиков являются независимыми и распределены нормально с нулевым средним.

Математически это записывается как:
\[
T_{\text{блюмс}}(t) = T_{\text{внешняя}}(t) + \varepsilon_{\text{блюмс}}(t),
\]
\[
T_{\text{борт}}(t) = T_{\text{внешняя}}(t) + T_{\text{внутренний}}(t) + \varepsilon_{\text{борт}}(t),
\]
где \( \varepsilon_{\text{блюмс}}(t) \sim \mathcal{N}(0, \sigma_{\text{блюмс}}^2) \) и \( \varepsilon_{\text{борт}}(t) \sim \mathcal{N}(0, \sigma_{\text{борт}}^2) \).

---

## Реализация модели с использованием NumPyro

Ниже приведён полный код модели и её использования для анализа данных.

### Код модели

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Байесовская модель
def model(T_blums, T_bort, time_points):
    """
    Байесовская модель разложения температуры на борту электролизера 
    на вклад внутреннего источника тепла и внешней среды.
    """
    # Количество временных точек
    n_time_points = len(time_points)
    
    ## Модель внешней температуры (Gaussian Random Walk)
    T_external_mean = numpyro.sample("T_external_mean", dist.Normal(jnp.mean(T_blums), 1.0))
    T_external = numpyro.sample(
        "T_external", dist.GaussianRandomWalk(scale=1.0).expand((n_time_points,))
    )
    T_external = T_external + T_external_mean  # Учитываем априорное среднее
    
    ## Модель внутренней температуры (Gaussian Random Walk)
    T_internal_mean = numpyro.sample("T_internal_mean", dist.Normal(0.0, 1.0))
    T_internal = numpyro.sample(
        "T_internal", dist.GaussianRandomWalk(scale=0.5).expand((n_time_points,))
    )
    T_internal = T_internal + T_internal_mean

    # Шум измерений
    sigma_blums = numpyro.sample("sigma_blums", dist.Exponential(1.0))
    sigma_bort = numpyro.sample("sigma_bort", dist.Exponential(1.0))
    
    # Наблюдения с датчиков
    # Датчик на блюмсе отражает только внешнюю температуру
    with numpyro.plate("data_blums", n_time_points):
        numpyro.sample(
            "obs_blums", dist.Normal(T_external, sigma_blums), obs=T_blums
        )
    
    # Датчик на борту — сумма внутреннего и внешнего источников
    with numpyro.plate("data_bort", n_time_points):
        numpyro.sample(
            "obs_bort", dist.Normal(T_external + T_internal, sigma_bort), obs=T_bort
        )
```

---

### Генерация данных (пример)

Для тестирования кода сгенерируем искусственные данные:

```python
import numpy as np

# Генерация временных точек
n_time_points = 100
time_points = np.arange(n_time_points)

# Искуственные внешние и внутренние температуры
T_external_true = np.sin(time_points / 10) + 0.5 * np.random.randn(n_time_points)
T_internal_true = 1.0 + 0.1 * np.cumsum(np.random.randn(n_time_points))

# Данные датчиков (с добавлением шума)
T_blums = T_external_true + 0.1 * np.random.randn(n_time_points)  # Данные блюмса
T_bort = T_external_true + T_internal_true + 0.2 * np.random.randn(n_time_points)  # Данные борта
```

---

### Вычисление апостериорных распределений

Выполним обучение модели на вышеуказанных данных с использованием MCMC:

```python
# Инициализация ядра NUTS и MCMC
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000, num_chains=2)

# Запуск MCMC
mcmc.run(
    rng_key=numpyro.prng_key(),
    T_blums=T_blums,
    T_bort=T_bort,
    time_points=time_points,
)

# Получение выборок из апостериорных распределений
posterior_samples = mcmc.get_samples()
```

---

### Интерпретация результатов

После выполнения моделирования мы можем получить предсказания для внешней и внутренней температуры на основе апостериорных средних значений.

```python
import matplotlib.pyplot as plt

# Достаем апостериорные средние
T_external_posterior = posterior_samples["T_external"].mean(axis=0)
T_internal_posterior = posterior_samples["T_internal"].mean(axis=0)

# Построение графиков
plt.figure(figsize=(12, 6))
plt.plot(time_points, T_blums, label="Blums (Input)", alpha=0.5)
plt.plot(time_points, T_bort, label="Bort (Input)", alpha=0.5)
plt.plot(time_points, T_external_posterior, label="External Temperature (Posterior)", linestyle="--", color="blue")
plt.plot(time_points, T_internal_posterior, label="Internal Temperature (Posterior)", linestyle="--", color="orange")
plt.legend()
plt.title("Decomposition of Bort Signal into External and Internal Components")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.show()
```

---

## Итоги

1. Мы построили байесовскую модель для разделения сигналов температурных датчиков на внешнюю и внутреннюю компоненты.
2. Модель предполагает, что внешняя температура изменяется плавно и может быть смоделирована случайным блужданием, как и внутренняя температура.
3. Использование MCMC (`NumPyro`) позволило вычислить апостериорные распределения и их характеристики.

Этот подход легко адаптируется для других временных рядов и может быть расширен, например, добавлением сезонных или нелинейных компонентов.