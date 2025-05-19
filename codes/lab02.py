import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Hei'] 
plt.rcParams['axes.unicode_minus'] = False

def f(x):
    return x**3 - 4*x - 1

def golden_section_search(func, a, b, epsilon):

    tau = (math.sqrt(5) - 1) / 2 

    x1 = b - tau * (b - a)
    x2 = a + tau * (b - a)
    f_x1 = func(x1)
    f_x2 = func(x2)
    
    iterations = 0
    print(f"{'迭代次数':<8} | {'a':<12} | {'b':<12} | {'x1':<12} | {'f(x1)':<12} | {'x2':<12} | {'f(x2)':<12} | {'b-a':<12}")
    print("-" * 120)
    
    while (b - a) > epsilon:
        iterations += 1
        # 在这里打印每次迭代的数据
        print(f"{iterations:<12} | {a:<12.6f} | {b:<12.6f} | {x1:<12.6f} | {f_x1:<12.6f} | {x2:<12.6f} | {f_x2:<12.6f} | {(b-a):<12.6f}")
        if f_x1 < f_x2:
            b = x2         
            x2 = x1        
            f_x2 = f_x1    
            x1 = b - tau * (b - a) 
            f_x1 = func(x1)  
        else:
            a = x1         
            x1 = x2        
            f_x1 = f_x2    
            x2 = a + tau * (b - a) 
            f_x2 = func(x2) 
            
    min_x = (a + b) / 2
    min_val = func(min_x)
    
    print("-" * 120) # 在循环结束后打印分隔线
    return min_x, min_val, iterations

a_start = 0.0  
b_start = 3.0  
epsilon_precision = 0.001  

print(f"实验二：黄金分割法求解 f(x) = x^3 - 4x - 1")
print(f"给定区间: ({a_start}, {b_start})")
print(f"收敛精度: {epsilon_precision}")
print("-" * 120)

min_x_found, min_val_found, num_iterations = \
    golden_section_search(f, a_start, b_start, epsilon_precision)

print(f"迭代次数: {num_iterations}")
print(f"找到的极小值点 x_min ≈ {min_x_found:.4f}") 
print(f"对应的极小值 f(x_min) ≈ {min_val_found:.4f}") 
print()

x_vals = np.linspace(a_start - 0.5, b_start + 0.5, 400) 
y_vals = f(x_vals)

plt.figure(figsize=(10, 7))
plt.plot(x_vals, y_vals, label='$f(x) = x^3 - 4x - 1$')

plt.plot(min_x_found, min_val_found, 'ro', markersize=10, label=f'极小值点 ({min_x_found:.4f}, {min_val_found:.4f})')

plt.title('黄金分割法求解函数极小值', fontsize=16)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$f(x)$', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.ylim(-6, 14)
plt.show()
