#%%
def step_function(x):
  if x > 0:
    return 1
  else:
    return 0
# %%
def step_function(x):
  y = x > 0
  return y.astype(np.int)
# %%
import numpy as np
x = np.array([-1.0, 1.0, 2.0])
x
y = x > 0
y.astype(np.int)
# %%
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
  return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# %%
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# %%
x = np.array([-1.0, 1.0, 2.0])
sigmoid(x)
# %%
t = np.array([1.0, 2.0, 3.0])
1.0 + t
1.0 / t
# %%
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# %%
def relu(x):
  return np.maximum(0, x)
# %%
import numpy as np
A = np.array([1,2,3,4])
#print(A)
np.ndim(A)
A.shape

B = np.array([[3,4], [3,4], [5,6]])
print(B)
np.ndim(B)
B.shape
# %%
import numpy as np
A = np.array([[1,2], [3,4]])
#A.shape
B = np.array([[5,6], [7,8]])
B.shape
np.dot(A, B)
# %%
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[1,2,],[3,4],[5,6]])
np.dot(A,B)

# %%
C = np.array([[1,2],[3,4]])
#np.dot(A, C)

# %%
x = np.array([1,2])
#x.shape
w = np.array([[1,3,5],[2,4,6]])
print(w)
w.shape
y = np.dot(x, w)
print(y)
# %%
x = np.array([1.0, 0.5])
w1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
b1 = np.array([0.1,0.2,0.3])

print(w1.shape)
print(x.shape)
print(b1.shape)
a1 = np.dot(x, w1) + b1
print(a1)
z1 = sigmoid(a1)
print(z1)
# %%
w2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
b2 = np.array([0.1, 0.2])

print(z1.shape)
print(w2.shape)
print(b2.shape)
a2 = np.dot(z1,w2) + b2
z2 = sigmoid(a2)
# %%
def identify_function(x):
  return x

w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
b3 = np.array([0.1, 0.2])

a3 = np.dot(z2, w3) + b3
y = identify_function(a3)

# %%
def init_network():
  network = {}
  network['w1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['w2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
  network['b2'] = np.array([0.1,0.2])
  network['w3'] = np.array([[0.1,0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])

  return network

def forward(network, x):
  w1, w2, w3 = network['w1'],network['w2'], network['w3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, w1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1,w2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, w3) + b3
  y = identify_function(a3)

  return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
# %%
a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a) # 지수함수
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)
# %%
def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y
# %%
a = np.array([1010, 1000, 990])
np.exp(a) / np.sum(np.exp(a))

c = np.max(a)
a - c

np.exp(a - c) / np.sum(np.exp(a - c))
# %%
def softmax(a):
  c = np.max(a)
  exp_a = np.exp(a - c)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y

# %%
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)

np.sum(y)
# %%
import sys, os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
# %%
import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from PIL import Image

def img_show(img):
  pil_img = Image.fromarray(np.uint8(img))
  pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28,28)
print(img.shape)

img_show(img)

# %%
#import pikcle
import pickle

def get_data():
  (x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
  return x_test, t_test

def init_network():
  with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)

  return network

def predict(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = softmax(a3)

  return y
# %%
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
  y = predict(network, x[i])
  p = np.argmax(y) #확률이 가장 높은 원소의 인덱스를 얻는다.
  if p == t[i]:
    accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


# %%
x, _ = get_data()
network = init_network()
w1,w2,w3 = network['W1'], network['W2'], network['W3']

x.shape

x[0].shape

w1.shape
w2.shape
w3.shape
# %%
x, t = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
  x_batch = x[i:i+batch_size]
  y_batch = predict(network, x_batch)
  p = np.argmax(y_batch, axis=1)
  accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
# %%
