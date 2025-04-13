### Архитектура решения с использованием байесовских методов и библиотеки NumPyro

Для решения задачи очистки температуры датчика на борту электролизера от влияния внешней среды предлагается следующая байесовская модель:

#### 1. Формулировка модели

Мы можем представить измерения датчиков как сумму нескольких компонентов:

- Для датчика на блюмсе (BL):
  \[ T_{BL}(t) = T_{BL}^{internal} + T_{env}(t) + \epsilon_{BL}(t) \]
  где:
  - \( T_{BL}^{internal} \) - постоянная внутренняя температура (априорное знание)
  - \( T_{env}(t) \) - температура внешней среды (скрытая переменная)
  - \( \epsilon_{BL}(t) \) - шум измерений

- Для датчика на борту (BR):
  \[ T_{BR}(t) = T_{BR}^{internal}(t) + \alpha \cdot T_{env}(t) + \epsilon_{BR}(t) \]
  где:
  - \( T_{BR}^{internal}(t) \) - изменяющаяся внутренняя температура (интересующий нас сигнал)
  - \( \alpha \) - коэффициент влияния внешней среды (может быть 1 или другим)
  - \( \epsilon_{BR}(t) \) - шум измерений

#### 2. Априорные распределения

- \( T_{BL}^{internal} \): Нормальное распределение с известным средним (из априорных знаний)
- \( T_{env}(t) \): Гауссовский процесс или авторегрессионная модель для временного ряда
- \( T_{BR}^{internal}(t) \): Гауссовский процесс с подходящим ядром
- \( \alpha \): Нормальное распределение вокруг 1
- Шумы \( \epsilon \): Нормальное распределение с небольшим стандартным отклонением

#### 3. Реализация в NumPyro

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp

def model(T_BL_obs, T_BR_obs):
    # Количество временных точек
    n = len(T_BL_obs)
    
    # Априорные параметры
    # Внутренняя температура блюмса (из априорных знаний)
    T_BL_internal = numpyro.sample("T_BL_internal", dist.Normal(known_mean, known_std))
    
    # Параметры внешней среды
    # Авторегрессионная модель для внешней температуры
    phi = numpyro.sample("phi", dist.Normal(0, 1))  # AR(1) коэффициент
    sigma_env = numpyro.sample("sigma_env", dist.HalfNormal(1.0))
    
    # Моделируем T_env как AR(1) процесс
    T_env = numpyro.sample("T_env", 
                          dist.GaussianRandomWalk(scale=sigma_env, num_steps=n-1))
    
    # Коэффициент влияния внешней среды на датчик борта
    alpha = numpyro.sample("alpha", dist.Normal(1.0, 0.1))
    
    # Внутренняя температура борта - гауссовский процесс
    # Используем экспоненциальное ядро для плавных изменений
    length_scale = numpyro.sample("length_scale", dist.Gamma(3.0, 1.0))
    sigma_br = numpyro.sample("sigma_br", dist.HalfNormal(2.0))
    kernel = sigma_br**2 * dist.gaussian_process.ExpQuad(length_scale)
    T_BR_internal = numpyro.sample("T_BR_internal", 
                                 dist.GaussianProcess(kernel, jnp.arange(n)))
    
    # Шумы измерений
    sigma_noise_BL = numpyro.sample("sigma_noise_BL", dist.HalfNormal(0.5))
    sigma_noise_BR = numpyro.sample("sigma_noise_BR", dist.HalfNormal(0.5))
    
    # Наблюдения
    numpyro.sample("obs_BL", 
                  dist.Normal(T_BL_internal + T_env, sigma_noise_BL),
                  obs=T_BL_obs)
    numpyro.sample("obs_BR", 
                  dist.Normal(T_BR_internal + alpha*T_env, sigma_noise_BR),
                  obs=T_BR_obs)

# Запуск MCMC
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000)
mcmc.run(jax.random.PRNGKey(0), T_BL_obs=jnp.array(T_BL_data), T_BR_obs=jnp.array(T_BR_data))

# Получение результатов
samples = mcmc.get_samples()
T_BR_internal_clean = samples["T_BR_internal"].mean(axis=0)  # Очищенный сигнал
```

#### 4. Анализ и интерпретация результатов

После выполнения MCMC мы получаем:
1. Оценку скрытой температуры внешней среды \( T_{env}(t) \)
2. Очищенный сигнал \( T_{BR}^{internal}(t) \), который отражает только изменения внутреннего источника тепла
3. Неопределенности оценок через доверительные интервалы

#### 5. Дополнительные улучшения

1. **Выбор ядра для GP**: Можно экспериментировать с разными ядрами (Matern, RBF) для \( T_{BR}^{internal}(t) \)
2. **Учет физики процесса**: Если известны уравнения теплопередачи, можно их интегрировать в модель
3. **Иерархическая модель**: Если есть данные с нескольких электролизеров, можно сделать иерархическую модель
4. **Онлайн-версия**: Для реального мониторинга можно реализовать последовательный байесовский вывод

#### 6. Валидация модели

Для проверки модели можно:
1. Сравнить апостериорные предсказания с реальными данными
2. Проверить сходимость MCMC (R-hat, трассировки)
3. Провести проверку на симулированных данных
4. Оценить предсказательную точность через кросс-валидацию

Эта модель позволяет разделить влияние внешней среды и внутренних источников тепла, используя априорные знания о постоянстве внутренней температуры датчика на блюмсе. Байесовский подход дает не только точечные оценки, но и полное распределение неопределенности.