#%%
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
t = [0,0,1,0,0,0,0,0,0,0]

def mean_squared_error(y, t):
  return 0.5*np.sum((y-t)**2)
# %%
import numpy as np

mean_squared_error(np.array(y), np.array(t))
# %%

# %%
y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
mean_squared_error(np.array(y), np.array(t))
# %%
def cross_entropy_error(y, t):
  delta = 1e-7
  return -np.sum(t*np.log(y+delta))

t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
cross_entropy_error(np.array(y), np.array(t))

y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
cross_entropy_error(np.array(y),np.array(t))
# %%
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)


# %%
import numpy as np

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
# %%
#* 미니배치 교차 엔트로피 오차 구현
def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batch_size = y.shape[0]
  return -np.sum(t * np.log(y + 1e-7)) / batch_size

# %%
#* not원핫인코딩, 숫자레이블인경우 교차 엔트로피 오차 구현
def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
    
  batch_size = y.shape[0]
  return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
# %%
# 나쁜 구현의 예
def numerical_diff(f, x):
  h = 1e-4
  return (f(x+h) - f(x-h)) /(2*h)

#np.float32(1e-50)
# %%
def function_1(x):
  return 0.01*x**2 + 0.1*x

import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

# %%
numerical_diff(function_1,5)
numerical_diff(function_1, 10)
# %%
def function_2(x):
  return x[0]**2 + x[1]**2
  # 또는 return np.sum(x**2)
# %%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np

def f(x,y):
  r = 2*x**2 + y**2
  ans = r*np.exp(-r)
  return ans

xn = 10
x = np.linspace(-3,3,xn)
y = np.linspace(-3,3,xn)
temp = np.zeros((len(x),len(y)))
for i in range(xn):
  for j in range(xn):
    temp[j,i] = f(x[i],y[j])

xx,yy = np.meshgrid(x,y)

plt.figure(figsize=(9,9))
ax = plt.subplot(111,projection='3d')
ax.plot_surface(xx,yy,temp)
plt.show()
# %%
fig = plt.figure(figsize = (9,6))
ax = fig.add_subplot(111, projection='3d')

def f(x,y):
  return x**2 + y**2

xn= 100
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
z = x**2 + y**2
ax.plot(x,y,z)
print(z)
# %%
#! 편미분, 목표변수를 제외한 변수는 상수로 취급!
def function_tmp1(x0):
  return x0*x0 + 4.0**2.0

numerical_diff(function_tmp1, 3.0)

# %%
