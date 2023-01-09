import numpy as np 
import math
np.random.seed(100)

threshold = 1e-4


# Testing the shift based Hessenberg QR iteration

print("Eigen vector and vaues for custom matrix")
C=[]
with open('../build/mat_A.txt', 'r') as infile:
    mat_A = infile.read()
    C = [float(i) for i in mat_A.split(',') if mat_A.strip()]

N = int(math.sqrt(len(C)))
print('Python: size of the array is: ' + str(N))

C =np.array(C).reshape(N,N)
C = C.astype('float64')

w,v = np.linalg.eig(C)
w_abs =np.array([abs(w[i]) for i in range(w.shape[0])])
w_sort_index = w_abs.argsort()[::-1]
w = w[w_sort_index]
v = np.transpose(v)
v = v[w_sort_index]

# print("\nnumpy eigen values ")
# print(w)

# print("\nnumpy eigen vectors")
# print(v)



v_list =  list(v.reshape(N*N))
w_list = list(w)

w_str = ' '.join(map(str, w_list))
v_str = ' '.join(map(str, v_list))

with open('../build/mat_W.txt', 'w') as Wfile:
    Wfile.write(w_str)

with open('../build/mat_V.txt', 'w') as Vfile:
    Vfile.write(v_str)
