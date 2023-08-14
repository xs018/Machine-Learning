import numpy as np
import matplotlib.pyplot as plt

N = 3
GPU1 = (0, 0,0)
GPU2 = (4.67, 3.67,5.56)
GPU4 = (0,4.78,8)

ind = np.arange(N) 
width = 0.35       
plt.bar(ind, GPU1, width, label='batch size: 256')
plt.bar(ind+ width, GPU2, width, label='batch size: 512')
plt.bar(ind+ width +width, GPU4 , width,
    label='atch size: 1024')

plt.ylabel('training speed (epochs/hrs)')

plt.xticks(ind + width +width/ 2, ('GPU1', 'GPU2', 'GPU4'))
plt.legend(loc='best')
plt.show()