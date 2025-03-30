import operator
import random
import math
from deap import base, creator, gp, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt

HALL_OF_FAME_SIZE = 30
P_CROSSOVER = 0.35
P_MUTATION = 0.25
MAX_GENERATIONS = 400
POPULATION_SIZE = 300
random.seed(42)
def nth_derivative(func, x, n, h=1e-6):
    """
    Численное вычисление n-й производной.

    Args:
        func: Функция для вычисления производной.
        x: Точка вычисления.
        n: Порядок производной.
        h: Шаг для численного дифференцирования.

    Returns:
        Значение n-й производной в точке x.
    """
    if n == 0:
        return func(x)
    elif n == 1:
        return derivative(func, x, h)
    else:
        return (nth_derivative(func, x + h, n - 1, h) - nth_derivative(func, x - h, n - 1, h)) / (2 * h)

def transform_expression(expression):
    """
    Преобразует строковое представление дерева функций в математическое выражение.

    :param expression: строка, представляющая дерево функций.
    :return: строка с преобразованным выражением.
    """
    # Словарь преобразований для операторов
    operators = {
        "mul": "*",
        "add": "+",
        "sub": "-",
        "safe_div": "/"
    }

    # Рекурсивная функция обработки
    def parse(expr):
        # Удаляем внешние пробелы
        expr = expr.strip()
        # Базовый случай: если выражение не содержит операторов, возвращаем его
        if "(" not in expr:
            return expr

        # Извлекаем имя функции или оператора
        func_name_end = expr.index("(")
        func_name = expr[:func_name_end]

        # Разбиваем аргументы, удаляя внешние скобки
        args = expr[func_name_end + 1:-1]

        # Разбираем аргументы на части (учитывая вложенные выражения)
        arg_list = []
        balance = 0
        current_arg = ""

        for char in args:
            if char == "," and balance == 0:
                arg_list.append(current_arg)
                current_arg = ""
            else:
                current_arg += char
                if char == "(":
                    balance += 1
                elif char == ")":
                    balance -= 1

        # Добавляем последний аргумент
        if current_arg:
            arg_list.append(current_arg)

        # Преобразуем аргументы рекурсивно
        parsed_args = [parse(arg) for arg in arg_list]

        # Преобразуем функцию или оператор
        if func_name in operators:
            # Оператор (например, mul, add)
            return f"({operators[func_name].join(parsed_args)})"
        elif func_name in {"sin", "cos", "safe_log", "exp"}:
            # Унарная функция (например, sin, cos)
            return f"{func_name}({parsed_args[0]})"
        else:
            raise ValueError(f"Неизвестная функция: {func_name}")

    # Обрабатываем исходное выражение
    return parse(expression)

def safe_log(x):
    if x <= 0:
        raise ValueError(f"Логарифм неопределён для x = {x}")
    return math.log(abs(x))
def safe_div(a, b):
    if b==0:
        raise ValueError(f'Произошло деление на ноль: {a} / {b}')
    return a/b

def derivative(func, x, h=1e-6):
    return (func(x + h) - func(x)) / h


# Определяем терминалы и функции для генетического дерева
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safe_div, 2)
pset.addPrimitive(safe_log, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.exp, 1)
pset.addTerminal(1)
pset.addTerminal(0)
pset.renameArguments(ARG0='x')


# Определяем основное окружение и настройки для генетического программирования
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
"""base.Fitness - стандартный базовый класс, который отвечает за значения приспособленности
 определенного вида. weight - одно число для оценки приспособленности ("+" т.е. максимизируем)"""

creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
"""class Individual со свойством Fin=tnessMin, наследованный от gp.PrimitiveTree """


# Инициализация алгоритма
toolbox = base.Toolbox()
"""genhalfandhalf - генерирует деревья в формате списка на основе процедуры ramped"""
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
"""initRepeat - создает списки. Аргументы:
1) Контейнер для хранения генов
2) Функция генерации значения гена
3) Число генов в хромосоме (кол-во функций или терминов в списке) - не задано
Каждое значение формируется на основе функции individual
"""
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)



# Задаем начальные условия и функцию ОДУ
K1, K2 = 0, 1
x0 = 0
y0_list = [0, 1]
x_points = np.linspace(-2,1, 100)
def eval_individual(individual, x_points, x0, y0_list, K1, K2):
    func_expr = toolbox.compile(expr=individual)  # Символьная функция
    N = len(x_points)
    deviation_sum = 0
    try:
        for x_val in x_points:
            y_val = func_expr(x_val)  # y(x)
            y_prime_val = derivative(func_expr, x_val)  # y'(x)
            y_double = nth_derivative(func_expr, x_val, 2)
            F = y_double-6*y_prime_val+9*y_val-math.exp(3*x_val) #Уравнение, которое нужно решить
            # F = y_double-12*y_prime_val+36*y_val-32*math.cos(2*x_val)-24*math.sin(2*x_val)
            deviation_sum += F**2
    except (ValueError, OverflowError, ZeroDivisionError, RuntimeWarning):
    # Плохая пригодность при ошибке,
            return 0,
    E_pk = (1 / N) * (deviation_sum)**0.5
    initial_conditions_penalty = 0
    for j, y0 in enumerate(y0_list):
        try:
            if j == 0:
                # Значение функции в x0
                y_val = func_expr(x0)
            else:
                # j-я производная в x0
                y_val = derivative(func_expr, x0) if j == 1 else nth_derivative(func_expr, x0, j)
            initial_conditions_penalty += abs(y_val - y0)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0,  # Если ошибка, вернуть плохую пригодность

    E_pk += K2 * initial_conditions_penalty
    D_pk = len(individual)
    penalty = K1 * D_pk
    fitness = 1 / (1 + E_pk + penalty)
    return fitness.real,

# Операторы селекции, скрещивания и мутации
#eval_individual - кортеж значений приспособленных индивида
toolbox.register("evaluate", eval_individual,
                 x_points=x_points, x0=x0,
                 y0_list=y0_list, K1=K1, K2=K2)
toolbox.register("select", tools.selTournament, tournsize=3) #Турнирый отбор и число особей (3)
toolbox.register("mate", gp.cxOnePoint) #Одноточечное скрещивание
toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)#глубина дерева в процессе мутации
#равномерная мутация вероятности, которая может добавить новое полное поддерево к узлу
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Ограничиваем максимальную высоту деревьев и настраиваем вероятности операций
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))


# Запуск алгоритма
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)
def main():
    population = toolbox.population(n=POPULATION_SIZE)  # Указываем размер популяции
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE) #зал славы
    #population - конечная популяция
    #logbook - собранная статистика
    population, logbook = algorithms.eaSimple(population, toolbox,
                        cxpb=P_CROSSOVER,  # Вероятность скрещивания
                        mutpb=P_MUTATION,  # Вероятность мутации
                        ngen=MAX_GENERATIONS,  # Макс. кол-во итераций
                        stats=stats,halloffame=hof, verbose=True)  # Отображение служебной информации в консоль
    # Выводим лучшего индивида из Hall of Fame
    best_individual = hof[0]
    tick=0
    for ind in hof:
        info = ind.fitness.values[0]
        s=str(ind)
        print(f"Пригодность: {info} из 1 ", transform_expression(s))
        tick+=1
        if tick==5:
            break
    maxFitnessValues, avgFitnessValues = logbook.select("max"), logbook.select('avg')
    plt.figure(2)
    plt.plot(maxFitnessValues, color='red')
    plt.plot(avgFitnessValues, color='green')
    plt.xlabel('Поколение')
    plt.ylabel('Максимальная/Средняя приспособленность')
    plt.title('Зависимость максимальной и средней приспособленности от поколения')
    plt.show()
    return best_individual
best_individual = main()
