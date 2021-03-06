#%%
def AND(x1, x2):
  w1, w2, theta = 0.5,0.5,0.7
  tmp = x1*w1 + x2*w2
  if tmp <= theta:
    return 0
  elif tmp > theta:
    return 1
#%%
print(AND(0,0))
print(AND(1,0))
print(AND(0,1))
print(AND(1,1))

# %%
import numpy as np
x = np.array([0,1])
w = np.array([0.5, 0.5])
b = -0.7
w*x
np.sum(w*x)
np.sum(w*x) + b
# %%
#* 가중치와 편향 구현
def AND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.7
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1
    

# %%
#*NAND게이트 OR게이트 구현
def NAND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([-0.5, -0.5])
  b = 0.7
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

def OR(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.2
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1
# %%
def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y
# %%
XOR(1,1)
# %%
