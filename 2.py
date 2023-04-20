# -*- coding: utf-8 -*-
"""Copy of Lab2.py

# Лабораторная работа 1
Численное дифференцирование и интегрирование
Постановка задачи:
> Реализовать численные методы нахождения производной при фиксированном значении шага, а также методы численого интегрирования. Пронаблюдать отклонения численных значений от истинных, вычисленных аналитически, а также проследить зависимость отклонения от величины параметра (шага дифференцирования/интегрирования). Проанализировать результаты, преимущества и ограничения методов.
"""
"""# Часть I. Численное дифференцирование

# Правая разностная производная
"""

def forward_difference(f, x, h):
  return (f(x + h) - f(x)) / h

"""# Левая разностная производная"""

def backward_difference(f, x, h):
  return (f(x) - f(x - h)) / h

"""# Центральная разностная производная"""

def central_difference(f, x, h):
  return (f(x + h) - f(x - h)) / (2.0 * h)

def right_edge_central_difference(f, x, h):
  return (f(x - 2.0 * h) - 4 * f(x - h) + 3 * f(x)) / (2.0 * h)

def left_edge_central_difference(f, x, h):
  return (-3 * f(x) + 4 * f(x + h) - f(x + 2.0 * h)) / (2.0 * h)

"""# Исходные данные

Возьмем произвольные функции 

$f_{1}$ = $x^{3} + 5x^{2} + 28$

$f_{2}$ = $e^{3x} + cos(2x)$



Найдем производные аналитически: 

$f_{1}'$ = $3x^{2} + 10x$

$f_{2}'$ = $3e^{3x} - 2sin(2x)$

И построим график функций с их производными:
"""

import numpy as np
import matplotlib.pyplot as plt

def first_function(x):
  return x ** 3 + 5 * x ** 2 + 28
def second_function(x):
  return np.e ** (3*x) + np.cos(2*x)

h = 2
a1 = -100
b1 = 100
a2 = -20
b2 = 0
x = np.linspace(a1, b1, 1000)
y = first_function(x)
fig, ax = plt.subplots()
ax.plot(x, y, color='cyan')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim(20, 60)
ax.set_xlim(-10, 10)
ax.grid()
ax.set_title("First function")
x = np.linspace(a2, b2, 1000)
y = second_function(x)
fig, ax2 = plt.subplots()
ax2.plot(x, y, color='cyan')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_ylim(-3, 3)
ax2.set_xlim(-10, 10)
ax2.grid()
ax2.set_title("Second function")

def first_derivative(x):
  return 3 * x ** 2 + 10 * x
def second_derivative(x):
  return 3 * np.e ** (3*x) - 2 * np.sin(2*x)

x1 = np.linspace(a1, b1, int((b1 - a1)/(h)))
y = first_derivative(x1)
fig, ax = plt.subplots()
ax.plot(x1, y, 'o-', color='cyan')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim(-10, 60)
ax.set_xlim(-10, 10)
ax.grid()
ax.set_title("First derivative")
x2 = np.linspace(a2, b2, int((b2 - a2)/(h)))
y = second_derivative(x2)
fig, ax2 = plt.subplots()
ax2.plot(x2, y, 'o-', color='cyan')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_ylim(-3, 3)
ax2.set_xlim(-10, 10)
ax2.grid()
ax2.set_title("Second derivative")

"""# Нахождение производной численными методами

Найдем значения производной методами правой разностной производной, левой разностной производной и центральной разностной производной и сравним их на графиках
"""

first_forward_difference = forward_difference(first_function, x1, h)
second_forward_difference = forward_difference(second_function, x2, h)
first_backward_difference = backward_difference(first_function, x1, h)
second_backward_difference = backward_difference(second_function, x2, h)
first_central_difference = central_difference(first_function, x1, h)
second_central_difference = central_difference(second_function, x2, h)
first_central_difference[0] = left_edge_central_difference(first_function, x1[0], h)
first_central_difference[-1] = right_edge_central_difference(first_function, x1[-1], h)
second_central_difference[0] = left_edge_central_difference(second_function, x2[0], h)
second_central_difference[-1] = right_edge_central_difference(second_function, x2[-1], h)

def set_first_graph(ax):
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_ylim(-10, 60)
  ax.set_xlim(-10, 10)
  ax.grid()
def set_second_graph(ax):
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_ylim(-3, 3)
  ax.set_xlim(-10, 10)
  ax.grid()

fig, ax1 = plt.subplots()
ax1.plot(x1, first_forward_difference, 'o-', color='cyan', label='forward')
ax1.plot(x1, first_backward_difference, 'o-', color='red', label='backward')
ax1.plot(x1, first_central_difference, 'o-', color='black', label='central')
set_first_graph(ax1)
ax1.set_title("First derivative by difference methods")
plt.legend()

fig, ax2 = plt.subplots()
ax2.plot(x2, second_forward_difference, 'o-', color='cyan', label='forward')
ax2.plot(x2, second_backward_difference, 'o-', color='red', label='backward')
ax2.plot(x2, second_central_difference, 'o-', color='black', label='central')
set_second_graph(ax2)
ax2.set_title("Second derivative by difference methods")
plt.legend()

"""По первому графику можно предположить, что точнее всего для первой функции будет работать метод левой разностной производной, поскольку его график представляет исходный с отображением по оси OY. По второму графику можно предположить, что значения СКО будут достаточно велики

# Расчёт СКО
"""

def calculate_MSE(a, b, h, func, array):
  sum = 0
  for i in range (0, int((b - a) / h)):
    sum += (func(a + i * h) - array[i]) ** 2
  return np.sqrt(sum / ((b - a) /h))

first_forward_difference_MSE = calculate_MSE(a1, b1, h, first_derivative, first_forward_difference)
first_backward_difference_MSE = calculate_MSE(a1, b1, h, first_derivative, first_backward_difference)
first_central_difference_MSE = calculate_MSE(a1, b1, h, first_derivative, first_central_difference)
print(first_forward_difference_MSE, first_backward_difference_MSE, first_central_difference_MSE)

second_forward_difference_MSE = calculate_MSE(a2, b2, h, second_derivative, second_forward_difference)
second_backward_difference_MSE = calculate_MSE(a2, b2, h, second_derivative, second_backward_difference)
second_central_difference_MSE = calculate_MSE(a2, b2, h, second_derivative, second_central_difference)
print(second_forward_difference_MSE, second_backward_difference_MSE, second_central_difference_MSE)

"""# Расчёт производной и СКО для различных шагов"""

x1_with_step = lambda h: np.linspace(a1, b1, int((b1 - a1)/(h)))
x2_with_step = lambda h: np.linspace(a2, b2, int((b2 - a2)/(h)))
first_forward_difference_with_step = lambda h: forward_difference(first_function, x1_with_step(h), h)
second_forward_difference_with_step = lambda h: forward_difference(second_function, x2_with_step(h), h)
first_backward_difference_with_step = lambda h: backward_difference(first_function, x1_with_step(h), h)
second_backward_difference_with_step = lambda h: backward_difference(second_function, x2_with_step(h), h)

def first_central_difference_with_step(h):
  res = central_difference(first_function, x1_with_step(h), h)
  res[0] = left_edge_central_difference(first_function, x1_with_step(h)[0], h)
  res[-1] = right_edge_central_difference(first_function, x1_with_step(h)[-1], h)
  return res

def second_central_difference_with_step(h):
  res = central_difference(second_function, x2_with_step(h), h)
  res[0] = left_edge_central_difference(second_function, x2_with_step(h)[0], h)
  res[-1] = right_edge_central_difference(second_function, x2_with_step(h)[-1], h)
  return res

def calculate_MSE_for_different_steps(a, b, step_array, func, array):
  sum = 0
  mse_array = []
  for h in step_array:
    method_values = array(h)
    sum = 0
    for i in range (0, int((b - a) / h)):
      sum += (func(a + i * h) - method_values[i]) ** 2
    mse_array.append(np.sqrt(sum / ((b - a) /h)))
  return mse_array

first_step = h
last_step = h / 16
num_of_steps = 5
derivative_step_array = np.geomspace(first_step, last_step, num_of_steps, endpoint=True)
print(derivative_step_array)
first_function_parameter_pack = (a1, b1, derivative_step_array, first_derivative)
second_function_parameter_pack = (a2, b2, derivative_step_array, second_derivative)

first_forward_difference_MSE_for_steps = calculate_MSE_for_different_steps(*first_function_parameter_pack, first_forward_difference_with_step)
first_backward_difference_MSE_for_steps = calculate_MSE_for_different_steps(*first_function_parameter_pack, first_backward_difference_with_step)
first_central_difference_MSE_for_steps = calculate_MSE_for_different_steps(*first_function_parameter_pack, first_central_difference_with_step)
print(first_forward_difference_MSE_for_steps, first_backward_difference_MSE_for_steps, first_central_difference_MSE_for_steps)

second_forward_difference_MSE_for_steps = calculate_MSE_for_different_steps(*second_function_parameter_pack, second_forward_difference_with_step)
second_backward_difference_MSE_for_steps = calculate_MSE_for_different_steps(*second_function_parameter_pack, second_backward_difference_with_step)
second_central_difference_MSE_for_steps = calculate_MSE_for_different_steps(*second_function_parameter_pack, second_central_difference_with_step)
print(second_forward_difference_MSE_for_steps, second_backward_difference_MSE_for_steps, second_central_difference_MSE_for_steps)

def set_first_mse_to_steps_graph(ax):
  ax.set_xlabel('h')
  ax.set_ylabel('mse')
  ax.set_ylim(0, 800)
  ax.set_xlim(0, 2)
  ax.grid()
def set_second_mse_to_steps_graph(ax):
  ax.set_xlabel('h')
  ax.set_ylabel('mse')
  ax.set_ylim(0, 10)
  ax.set_xlim(0, 2)
  ax.grid()

fig, ax1 = plt.subplots()
ax1.plot(derivative_step_array, first_forward_difference_MSE_for_steps, 'o-', color='cyan', label='forward')
ax1.plot(derivative_step_array, first_backward_difference_MSE_for_steps, 'o-', color='red', label='backward')
ax1.plot(derivative_step_array, first_central_difference_MSE_for_steps, 'o-', color='black', label='central')
set_first_mse_to_steps_graph(ax1)
ax1.set_title("MSE to step length (h) for the first function")
plt.legend()

fig, ax2 = plt.subplots()
ax2.plot(derivative_step_array, second_forward_difference_MSE_for_steps, 'o-', color='cyan', label='forward')
ax2.plot(derivative_step_array, second_backward_difference_MSE_for_steps, 'o-', color='red', label='backward')
ax2.plot(derivative_step_array, second_central_difference_MSE_for_steps, 'o-', color='black', label='central')
set_second_mse_to_steps_graph(ax2)
ax2.set_title("MSE to step length (h) for the second function")
plt.legend()

"""По графикам отлично видно, что при увеличении шага СКО растет. Кроме этого, самым устойчивым и точным методом для обеих функций оказался метод левой разностной производной

# Часть II. Численное интегрирование

# Исходные данные

$I_1 = \int_{1}^{21} \left(\cos\left(\frac{\ln x}{2}\right) + 3\right)dx$

$I_2 =\int_{-1}^{1}(1+x+x^2+x^3+x^4) dx$

Desmos с визуализацией для второго интеграла: https://www.desmos.com/calculator/qs5359oewf
"""

import numpy as np
import matplotlib.pyplot as plt

f = lambda x: np.cos(np.log(x) / 2.0) + 3
a1, b1 = 1, 21

g = lambda x: 1 + x + x**2 + x**3 + x ** 4
a2, b2 = -1, 1

"""Изобразим криволинейные трапеции, ограниченные графиками выбранных функций - $f\left(x\right)=\left(\cos\left(\frac{\ln x}{2}\right) + 3\right)$, $g\left(x\right)=(0.2x^2 \sin{x} + 120)$ и прямыми $y=0$, $x=a$, $x=b$, где $a$ и $b$ - соответствующие верхние и нижние пределы интегрирования

"""

x = np.linspace(a1, b1, 1000)
y = f(x)
fig, ax = plt.subplots()

ax.plot(x, y, color='blue', linewidth=2)
ax.plot([a1, a1], [a1, f(a1)], 'k--', linewidth=0.5)
ax.plot([b1, b1], [b1, f(b1)], 'k--', linewidth=0.5)
ax.plot([a1,b1], [0,0], 'k--', linewidth=0.5)
ax.fill_between(x, y, where=(x>a1) & (x<b1), alpha=0.2, color='lightblue')
ax.set_ylim(top=5)

ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title('Curved trapezoid 1', fontsize=16)
ax.grid(True)

x = np.linspace(a2, b2, 1000)
y = g(x)
fig, ax = plt.subplots()

ax.plot(x, y, color='blue', linewidth=2)
ax.plot([0,0], [a2, g(a2)], 'k--', linewidth=0.5)
ax.plot([b2, b2], [b2, g(b2)], 'k--', linewidth=0.5)
ax.plot([a2,b2], [0,0], 'k--', linewidth=0.5)
ax.fill_between(x, y, where=(x>a2) & (x<b2), alpha=0.2, color='lightblue')

ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title('Curved trapezoid 2', fontsize=16)
ax.grid(True)

"""# Метод прямоугольников

Рассмотрим и сравним методы средних, правых и левых прямоугольников, суть которых в замене подынтегрального выражения на интерполяционный многочлен Лагранжа нулевой степени (геометрически – покрываем область прямоугольником, высота которого совпадает со значением функции в выбранной точке, а длина основания равна шагу разбиения h)

Метод левых прямоугольников: $\int_{a}^{b}f\left(x\right)dx\approx\sum_{i=1}^{N}h\cdot f\left(x_{i-1}\right)$

Метод правых прямоугольников: $\int_{a}^{b}f\left(x\right)dx\approx\sum_{i=1}^{N}h\cdot f\left(x_i\right)$

Метод средних прямоугольников: $\int_{a}^{b}f\left(x\right)dx\approx\sum_{i=1}^{N}h\cdot f\left(\frac{x_{i-1}+x_i}{2}\right)$

# Вычисления методом прямоугольников
"""

def rectangle_method(expression, a, b, h, shift):
    x_start = a + shift * h
    sum = 0.0
    for i in range(int((b - a) / h)):
        sum += h * expression(x_start + i * h)
    return sum

"""Метод левых прямоугольников"""

def left_rectangle_method(expression, a, b, h):
    return rectangle_method(expression, a, b, h, 0.0)

"""Метод правых прямоугольников"""

def right_rectangle_method(expression, a, b, h):
    return rectangle_method(expression, a, b, h, 1.0)

"""Метод средних прямоугольников"""

def mid_rectangle_method(expression, a, b, h):
    return rectangle_method(expression, a, b, h, 0.5)

"""Для начала положим некоторые фиксированные шаги $h_1$ и $h_2$ для численного интегрирования при делении интервала на $2$ шага"""

n = 2
h1, h2 = (b1 - a1) / n, (b2 - a2) / n

I1_left = left_rectangle_method(f, a1, b1, h1)
I1_right = right_rectangle_method(f, a1, b1, h1)
I1_mid = mid_rectangle_method(f, a1, b1, h1)

I2_left = left_rectangle_method(g, a2, b2, h2)
I2_right = right_rectangle_method(g, a2, b2, h2)
I2_mid = mid_rectangle_method(g, a2, b2, h2)

I1_left

I1_right

I1_mid

I2_left

I2_right

I2_mid

"""# Метод трапеций

Данный метод заключается в замене подынтегрального выражения многочленом Лагранжа первой степени (геометрически – покрываем область трапециями, основания которых лежат на прямых, параллельных оси ординат, одна из боковых сторон в нашем совпадает с осью абсцисс, а вторая образована секущей, проходящей через точки, соответствующие заданной функции на концах отрезков разбиения)

$\int_{a}^{b}f\left(x\right)dx\approx\sum_{i=1}^{N}\frac{f\left(x_i\right)+f\left(x_{i-1}\right)}{2}h=h\left[\frac{1}{2}\left(f_1+f_N\right)+f_2+\ldots+f_{N-1}\right]$

$=\frac{1}{2}h\left[f\left(a\right)+2\sum_{i=1}^{N-1}f\left(a+i\cdot h\right)+f\left(b\right)\right]$

# Вычисления методом трапеций
"""

def trapezoid_rule(f, a, b, h):
  x_start = a
  sum = 0.0
  for i in range(1, int((b - a) / h) + 1):
    sum += h * ((f(x_start + (i - 1) * h)) + f(x_start + i * h)) / 2.0
  return sum

I1_trapezoid = trapezoid_rule(f, a1, b1, h1)
I2_trapezoid = trapezoid_rule(g, a2, b2, h2)

I1_trapezoid

I2_trapezoid

"""# Метод Симпсона

В данном методе подынтегральная функция заменяется многочленом Лагранжа второй степени (геометрически – аппроксимация параболой, проходящей через точки $x_{i-1}$, $x_i$, $x_{i+1}$)

Если отрезок интегрирования разбивается на чётное количество равных частей ($2N$ с шагом $h=\frac{b-a}{2N}$), то тогда рассматриваются сдвоенные отрезки, на которых $\int_{x_{i-1}}^{x_{i+1}}f\left(x\right)dx\approx\frac{h}{3}\left(f_{i-1}+4f_i+f_{i+1}\right)$

И тогда $\int_{a}^{b}f\left(x\right)dx\approx\frac{h}{3}\left[f_0+f_{2N}+2\sum_{i=2}^{2N-2}f_i+4\sum_{i=1}^{2N-1}f_i\right]$

В случае же разбиения на нечётное количество участков в качестве точек между двумя крайними рассматриваются середины отрезков и тогда $\int_{a}^{b}f\left(x\right)dx\approx\frac{h}{6}\left[f_0+f_{N}+2\sum_{i=1}^{N-1}f_i+4\sum_{i=0.5}^{N-0.5}f_i\right]$

Коэффициенты уравнения параболы $y=ax^2+bx+c$ на сдвоенном отрезке $\left[x_{i-1},x_{i+1}\right]$
однозначно определяются системой

\begin{equation}
    \begin{cases}
      ax_{i-1}^2+bx_{i-1}+c=f_{i-1}\\
      ax_i^2+bx_i+c=f_i\\
      ax_{i+1}^2+bx_{i+1}+c=f_{i+1}
    \end{cases}\,
\end{equation}

Например, для наглядности можем рассчитать соответствующие коэффициенты при выбранном шаге. Так как изначально взято 2 интервала, аппроксимировать можем лишь одной параболой, а значит, и решать нужно только одну систему для каждой из двух кривых

Для функции $f(x)$:

\begin{equation}
    \begin{cases}
      a+b+c=f(1)\\
      100a+10b+c=f(10)\\
      441a+21b+c=f(21)
    \end{cases}\, ⇒ 
\end{equation}

\begin{equation}
    \begin{cases}
      a = \frac{11-20\cos(\frac{\ln{10}}{2}) + 9\cos(\frac{\ln{21}}{2})}{1980}\\
      b = \frac{-31+40\cos(\frac{\ln{10}}{2}) - 9\cos(\frac{\ln{21}}{2})}{180}\\
      c = \frac{275-14\cos(\frac{\ln{10}}{2}) + 3\cos(\frac{\ln{21}}{2})}{66}
    \end{cases}\,
\end{equation}

Для функции $g(x)$:

\begin{equation}
    \begin{cases}
      a\left(-1\right)^2+b\left(-1\right)+c=1\\
      a\left(0\right)^2+0b+c=1\\
      a\left(1\right)^2+1b+c=5
    \end{cases}\, ⇒ 
\end{equation}

\begin{equation}
    \begin{cases}
      a = 2\\
      b = 2\\
      c = 1
    \end{cases}\,
\end{equation}

Изобразим на графике эти параболы и площади, которую ограничивают соответствующие графики:
"""

a = (11-20*np.cos(np.log(10)/2) + 9*np.cos(np.log(21)/2))/1980
b = (-31+40*np.cos(np.log(10)/2) - 9*np.cos(np.log(21)/2))/180
c = (275-14*np.cos(np.log(10)/2) + 3*np.cos(np.log(21)/2))/66

x = np.linspace(a1, b1, 1000)
y1 = a*x**2 + b*x + c
y2 = f(x)

fig, ax = plt.subplots()

ax.plot(x, y1, color='blue', linewidth=2)
ax.fill_between(x, y1, where=((x>=a1) & (x<=b1) & (y1>=0)), color='lightblue')

ax.plot(x, y2, color='green', linewidth=2)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Simpson method for the function f(x)')
ax.grid(True)
plt.show()

a = 2
b = 2
c = 1

x = np.linspace(a2, b2, 1000)
y1 = a*x**2 + b*x + c
y2 = g(x)

fig, ax = plt.subplots()

ax.plot(x, y1, color='blue', linewidth=2)
ax.fill_between(x, y1, where=((x>=a2) & (x<=b2) & (y1>=0)), color='lightblue')

ax.plot(x, y2, color='green', linewidth=2)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Simpson method for the function g(x)')
ax.grid(True)
plt.show()

"""# Вычисления методом Симпсона"""

def Simpson_rule(f, a, b, h):
    n = int((b - a) / h)
    if n & 1:
      n *= 2
      h /= 2
      
    sum = (f(a) + 4 * f(a + h) + f(b))
    for i in range(1, int(n / 2)):
       sum += 2 * f(a + (2 * i) * h) + 4 * f(a + (2 * i + 1) * h)

    return sum * h / 3

I1_Simpson = Simpson_rule(f, a1, b1, h1)
I2_Simpson = Simpson_rule(g, a2, b2, h2)

I1_Simpson

I2_Simpson

"""# Анализ методов численного интегрирования и выводы

Сравним полученные значения со значением, полученным аналитически

$I_{1} = (\frac{1}{5}x(2\sin(\frac{\ln{x}}{2}) + 4\cos(\frac{\ln{x}}{2}) + 15)) \rvert^{21}_{1} = \frac{1}{5}(296 + 42\sin(\frac{\ln{21}}{2}) + 84\cos(\frac{\ln{21}}{2})) \approx 68,405$
"""

indefinite_integral_f = lambda x: x / 5.0 * (2 * np.sin(np.log(x) / 2.0) + 4 * np.cos(np.log(x) / 2.0) + 15)
I1 = indefinite_integral_f(b1) - indefinite_integral_f(a1) 
I1

"""$I_{2} = (x+\frac{x^2}{2}+\frac{x^3}{3}+\frac{x^4}{4}+\frac{x^5}{5}) \rvert^{1}_{-1} = (2 + \frac{0}{2} + \frac{2}{3}+\frac{0}{4}+\frac{2}{5}) = 3,066...$"""

indefinite_integral_g = lambda x: x + x**2 / 2.0 + x**3 / 3.0 + x**4 / 4.0 + x**5 / 5.0
I2 = indefinite_integral_g(b2) - indefinite_integral_g(a2) 
I2

"""Замечаем, что точнее всего оказался метод Симпсона, его ошибка составила

"""

abs(I1 - I1_Simpson)

abs(I2 - I2_Simpson)

"""Среди методов прямоугольников лучше всех показал себя метод средних прямоугольников с ошибкой"""

abs(I1 - I1_mid)

abs(I2 - I2_mid)

"""В то время как метод левых и правых прямоугольников имеют ошибки """

abs(I1 - I1_left)

abs(I2 - I2_left)

"""и"""

abs(I1 - I1_right)

abs(I2 - I2_right)

"""соответственно

Что касается метода трапеций, то его ошибка тоже достаточно велика - вышло, что она превышает ошибку метода средних прямоугольников
"""

abs(I1 - I1_trapezoid)

abs(I2 - I2_trapezoid)

"""Также можно проверить данные на лекциях по математическому анализу ограничения на погрешности методов

Утверждается, что

 $|I - I_{MidRectangles}| \leq \frac{M_{2}}{24}(b-a)h^2$

 $|I - I_{Trapezoid}| \leq \frac{M_{2}}{12}(b-a)h^2$

  $|I - I_{Simpson}| \leq \frac{M_{4}}{180}(b-a)h^4$, 

  где $M_{i_{f}} = \max_{[a,b]}|f^{(i)}(x)|$

Тогда $f''(x) = \frac{-\cos(\ln{x}) + \sin(\ln{x}))}{x^2}$,   $f^{(4)}(x) = -\frac{10\cos(\ln{x})}{x^4}$


$g''(x) = 12x^2 + 6x + 2$,   $g^{(4)}(x) = 24$
"""

f_second_derivative = lambda x: (-np.cos(np.log(x)) + np.sin(np.log(x))) / x**2
f_fourth_derivative = lambda x: (-10 * np.cos(np.log(x))) / x**4
g_second_derivative = lambda x: 12 *  x**2 + 6 * x + 2
g_fourth_derivative = lambda x: 24

"""Найдём теперь соответствующие максимумы"""

import scipy

M4f = abs(scipy.optimize.minimize_scalar(lambda x: -f_fourth_derivative(x), bounds=[a1,b1], method='bounded')['fun'])
M2f = abs(scipy.optimize.minimize_scalar(lambda x: -f_second_derivative(x), bounds=[a1,b1], method='bounded')['fun'])
M4g = abs(scipy.optimize.minimize_scalar(lambda x: -g_fourth_derivative(x), bounds=[a2,b2], method='bounded')['fun'])
M2g = abs(scipy.optimize.minimize_scalar(lambda x: -g_second_derivative(x), bounds=[a2,b2], method='bounded')['fun'])

M4f

M2f

M4g

M2g

"""Проверим ограничения"""

abs(I1 - I1_mid) <= M2f / 24 * (b1 - a1) * h1**2

abs(I2 - I2_mid) <= M2g / 24 * (b2 - a2) * h2**2

abs(I1 - I1_trapezoid) <= M2f / 12 * (b1 - a1) * h1**2

abs(I2 - I2_trapezoid) <= M2g / 12 * (b2 - a2) * h2**2

abs(I1 - I1_Simpson) <= M4f / 180 * (b1 - a1) * h1**4

abs(I2 - I2_Simpson) <= M4g / 180 * (b2 - a2) * h2**4

abs(I2 - I2_Simpson) - M4g / 180 * (b2 - a2) * h2**4

"""Последнее ограничение не выполнено, но порядок разности довольно мал и почти сопоставим с машинной точностью

# Расчёт отклонения в зависимости от шага
"""

def get_deviations_for_method_with_steps(method, function, a, b, analytic_value, steps):
  res = []
  for step in steps:
    res.append(abs(method(function, a, b, step) - analytic_value))
  return res

num_of_steps = 5
integral_step_array_for_f = np.geomspace(h1 / 16, h1, num_of_steps, endpoint=True)
integral_step_array_for_g = np.geomspace(h2 / 16, h2, num_of_steps, endpoint=True)
print(integral_step_array_for_f, integral_step_array_for_g)

f_left_rectangle_method_deviations = get_deviations_for_method_with_steps(left_rectangle_method, f, a1, b1, I1, integral_step_array_for_f)
g_left_rectangle_method_deviations = get_deviations_for_method_with_steps(left_rectangle_method, g, a2, b2, I2, integral_step_array_for_g)
f_mid_rectangle_method_deviations = get_deviations_for_method_with_steps(mid_rectangle_method, f, a1, b1, I1, integral_step_array_for_f)
g_mid_rectangle_method_deviations = get_deviations_for_method_with_steps(mid_rectangle_method, g, a2, b2, I2, integral_step_array_for_g)
f_right_rectangle_method_deviations = get_deviations_for_method_with_steps(right_rectangle_method, f, a1, b1, I1, integral_step_array_for_f)
g_right_rectangle_method_deviations = get_deviations_for_method_with_steps(right_rectangle_method, g, a2, b2, I2, integral_step_array_for_g)
f_trapeziod_method_deviations = get_deviations_for_method_with_steps(trapezoid_rule, f, a1, b1, I1, integral_step_array_for_f)
g_trapeziod_method_deviations = get_deviations_for_method_with_steps(trapezoid_rule, g, a2, b2, I2, integral_step_array_for_g)
f_Simpson_method_deviation = get_deviations_for_method_with_steps(Simpson_rule, f, a1, b1, I1, integral_step_array_for_f)
g_Simpson_method_deviation = get_deviations_for_method_with_steps(Simpson_rule, g, a2, b2, I2, integral_step_array_for_g)

def set_graph_integral(ax, xlim, ylim):
  ax.set_xlabel('h')
  ax.set_ylabel('deviation')
  ax.set_ylim(0, ylim)
  ax.set_xlim(0, xlim)
  ax.grid()

fig, ax = plt.subplots()
ax.plot(integral_step_array_for_f, f_left_rectangle_method_deviations, 'o-', color='cyan', label='left rectangle')
ax.plot(integral_step_array_for_f, f_mid_rectangle_method_deviations, 'o-', color='red', label='mid rectangle')
ax.plot(integral_step_array_for_f, f_right_rectangle_method_deviations, 'o-', color='black', label='right rectangle')
ax.plot(integral_step_array_for_f, f_trapeziod_method_deviations, 'o-', label='trapezoid')
ax.plot(integral_step_array_for_f, f_mid_rectangle_method_deviations, 'o-', label='Simpson')
ax.set_title("deviation of computational result to step length (h) for the first function")
set_graph_integral(ax, 12, 10)
plt.legend()

fig, ax = plt.subplots()
ax.plot(integral_step_array_for_g, g_left_rectangle_method_deviations, 'o-', color='cyan', label='left rectangle')
ax.plot(integral_step_array_for_g, g_mid_rectangle_method_deviations, 'o-', color='red', label='mid rectangle')
ax.plot(integral_step_array_for_g, g_right_rectangle_method_deviations, 'o-', color='black', label='right rectangle')
ax.plot(integral_step_array_for_g, g_trapeziod_method_deviations, 'o-', label='trapezoid')
ax.plot(integral_step_array_for_g, g_Simpson_method_deviation, 'o-', label='Simpson')
ax.set_title("deviation of computational result to step length (h) for the second function")
set_graph_integral(ax, 1, 3)
plt.legend()

"""# Выводы, анализ результатов, преимуществ и ограничений методов

*Численное дифференцирование*

В первой части работы были реализованы методы численного нахождения производной функции. Все эти методы сводятся к построению сетки и нахождению производной как скорости изменения значения функции между узлами сетки. Отличия в методах заключаются в правиле выбора узлов, а точность метода напрямую зависит от числа узлов сетки.

При изначальном фиксированном шаге сетки были реализованы методы левой, правой и центральной разностной производной. Притом метод центральной разностной производной является методом второго порядка точности (следовательно, данный метод даёт точный результат для полиномов максимальной степени два), в отличие от методов левой и правой разностной производной, которые являются методами первого порядка точности.

Были выбраны две произвольные функции и вычислены аналитически производные этих функций. Далее были построены графики их производных, вычисленных в узлах взятой сетки численными методами. Уже на этом этапе, не анализируя зависимость точности метода от количества узлов сетки, можно заметить различия в графиках для трёх представленных методов.

Далее построили зависимость СКО от длины шага сетки. Анализируя графики, можно понять, что с увеличением длины шага увеличивается и СКО ввиду уменьшения количества узлов для вычисления численной производной. Можно также отметить, что для данных функций наиболее точным при любой длине шага оказался метод левой разностной производной.

*Численное интегрирование*

Численное интегрирование можно проводить, используя квадратурные формулы. Введя сетку, можно считать, что исходный интеграл раскладывается в сумму элементарных интегралов, каждый из которых может быть рассчитан как площадь соответствующей криволинейной трапеции, взятой между соседними узлами сетки. Численные методы же отличаются способом нахождения этих элементарных площадей.

Были взяты две произвольные функции, построены графики криволинейных трапеций, ограниченных соответствующими кривыми, и рассчитаны аналитически заданные интегралы.

Метод прямоугольников сводится к приближению площади элементарной криволинейной трапеции прямоугольниками. В зависимости от выбора точки разбиения получаются методы левых, правых и центральных прямоугольников. Среди этих трёх методов наиболее точным оказался метод средних прямоугольников, в то время как методы левых и правых прямоугольников давали в сравнении большие отклонения от аналитического значения.

Метод трапеций заключается в приближении площадей криволинейных трапеций площадями элементарных трапеций. Данный метод показал более точные результаты, чем методы левых и правых прямоугольников, но имел большую ошибку, чем метод средних прямоугольников при тех же значениях длин шага.

Суть метода Симпсона в приближении площадей трапеций площадями парабол, проходящих через три узловые точки. Этот метод оказался точнее всего и дал ошибку значительно меньшую, чем у методов, расмотренных ранее.

Далее проверили выполнение ограничений на погрешности методов с точностью до погрешности компьютерных вычислений с плавающей точкой.

Построили зависимость отклонения численного значения от длины шага разбиения сетки. Из графиков для обеих функций можем сделать вывод, что уменьшение длины шага ведёт к уменьшению отклонения численного значения от аналитического. Наиболее точным при любых значениях длины шага оказался метод Симпсона.
"""