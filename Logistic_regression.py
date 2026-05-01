import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Загрузка наборов данных
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# Пример изображения
index = 207
# plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", это класс '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' изображений.")
# plt.show()
m_test = len(test_set_x_orig)
m_train, num_py, num_px = np.shape(train_set_x_orig)[:-1]

print ("Количество обучающих примеров: m_train = " + str(m_train))
print ("Количество тестовых примеров: m_test = " + str(m_test))
print ("Ширина изображения: num_px = " + str(num_px))
print ("Высота изображения: num_py = " + str(num_py))
print ("Форма массива изображения: (" + str(num_py) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


assert np.array_equal(train_set_x_flatten[0:5,0],np.array([17, 31, 56, 22, 33]))
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255
"""После преобразований train_set_x и test_set_x - тензоры размером (12288, 209) и (12288, 50) соответственно.
Это значит, что в каждом из них по 209 и 50 изображений с количество пикселей = 12288 (64*64*3). После нормализации каждое значение 
пикселя лежит в диапозоне от 0 до 1.

train_set_y и test_set_y это тензоры, содержащие метки классов (0 и 1) в количестве 209 и 50 соответственно"""


# Функция активации: sigmoid

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

assert sigmoid(np.array([0,2]))[0] == 0.5
assert round(sigmoid(np.array([0,2]))[1], 6) == 0.880797

# Инициализация нулями: initialize_with_zeros

def initialize_with_zeros(dim):
    b=0
    w=np.zeros((dim, 1))
    # assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b

dim = 12288
w, b = initialize_with_zeros(dim)
# assert np.array_equal(w,np.array([[0.],[0.]]))
# assert np.array_equal(b,0)

# Прямое и обратное распространение ошибки: propagate

def propagate(w, b, X, Y):
    m = X.shape[1]
    summa=0
    # FORWARD PROPAGATION (FROM X TO COST)
    A=sigmoid(np.dot(w.T, X)+b)
    cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    # raise NotImplementedError()

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1/m*np.dot(X, (A-Y).T)
    db = 1/m * np.sum(A - Y)
    # raise NotImplementedError()

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost

w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagate(w, b, X, Y)

assert np.round(grads["dw"])[0] == 1
assert np.round(grads["dw"])[1] == 2
assert np.round(grads["db"],1) == 0.5
assert np.round(cost) == 6

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []

    for i in range(num_iterations):
        # Вычисление градиента и функции стоимости
        grads, cost = propagate(w, b, X, Y)
        # raise NotImplementedError()

        dw = grads["dw"]
        db = grads["db"]

        # обновление весов
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # raise NotImplementedError()

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

# params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
#
# pw = params["w"]
# pb = params["b"]
# assert np.round(pw,5)[0] == 0.11246
# assert np.round(pw,5)[1] == 0.23107
# assert np.round(pb,5) == 1.55930
#
# assert np.round(grads["dw"],5)[0] == 0.90158
# assert np.round(grads["dw"],5)[1] == 1.76251
# assert np.round(grads["db"],5) == 0.43046



# Функция классификации: predict

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    # Вычислите вектор "A"
    A=sigmoid(np.dot(w.T, X)+b)
    # raise NotImplementedError()

    for i in range(A.shape[1]):
        # Произведите классификацию с помощью порогового значения 0.5
        if A[0][i]<=0.5:
            Y_prediction[0][i]=0
        else:
            Y_prediction[0][i]=1
        # raise NotImplementedError()

    assert(Y_prediction.shape == (1, m))

    return Y_prediction

assert np.array_equal(predict(w, b, X),np.array([[1.,1.]]))


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    # инициализация параметов (w, b)
    w, b = initialize_with_zeros(dim)
    # raise NotImplementedError()

    # Градиентный спуск
    grads, cost = propagate(w, b, X_train, Y_train)
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    # raise NotImplementedError()

    w = params["w"]
    b = params["b"]

    # Предсказание Y_prediction_test, Y_prediction_train
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    # raise NotImplementedError()

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.01, print_cost = True)
index = 1
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
plt.show()
print ("y = " + str(test_set_y[0,index]) + ", предсказание нейросети: \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" на изображении.")

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('ошибка')
plt.xlabel('число итераций обучения')
plt.title("Скорость обучения =" + str(d["learning_rate"]))
plt.show()

learning_rates = [0.01, 0.015, 0.008]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('ошибка')
plt.xlabel('число итераций обучения')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


import pickle
def save_model(w, b):
    with open('model_params.pkl', 'wb') as f:
        pickle.dump((w, b), f)
def load_model():
    with open('model_params.pkl', 'rb') as f:
        w, b = pickle.load(f)
    return w, b

# save_model(d["w"], d["b"])
# w, b = load_model()


# def predict_on_new_image(index):
#     # Получить изображение по индексу
#     image = test_set_x[:, index].reshape((num_px, num_px, 3))
#
#     # Сначала сделаем предсказание
#     Y_prediction = predict(w, b, test_set_x[:, index].reshape(-1, 1))
#
#     # Показать картинку и результат предсказания
#     plt.imshow(image)
#     plt.show()
#     print(f"y = {test_set_y[0, index]}, предсказание нейросети: '{classes[int(Y_prediction[0, 0])].decode('utf-8')}'")
#
#
# predict_on_new_image(36)

from PIL import Image


def load_and_preprocess_image(image_path, new_size=(64, 64)):
    image = Image.open(image_path)
    image = image.resize(new_size)
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_flattened = image_array.reshape((-1, 1))
    return image_flattened


def predict_on_image(image_path):
    # Загрузить и подготовить изображение
    image_flattened = load_and_preprocess_image(image_path)
    w, b = load_model()
    Y_prediction = predict(w, b, image_flattened)
    image = Image.open(image_path)
    image.show()
    print(f"Предсказание нейросети: '{classes[int(Y_prediction[0, 0])].decode('utf-8')}'")

image = "кот для теста.jpg"
predict_on_image(image)
