import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['Hei']  
plt.rcParams['axes.unicode_minus'] = False    

def rosenbrock(x):
    x1, x2 = x
    return (x1**2 - x2)**2 + (1 - x1)**2

def rosenbrock_grad(x):
    x1, x2 = x
    df_dx1 = 4 * x1 * (x1**2 - x2) - 2 * (1 - x1)
    df_dx2 = -2 * (x1**2 - x2)
    return np.array([df_dx1, df_dx2])

def backtracking_line_search(f, grad_f_xk, xk, pk, alpha_init=1.0, rho=0.5, c=1e-4):
    alpha = alpha_init
    fxk = f(xk)
    # Armijo 条件: f(xk + alpha*pk) <= f(xk) + c * alpha * grad_f_xk.T @ pk
    # 注意: pk 是搜索方向。对于最速下降法, pk = -grad_f_xk
    # 所以 grad_f_xk.T @ pk = - ||grad_f_xk||^2
    dot_grad_pk = np.dot(grad_f_xk, pk)

    max_ls_iters = 100 
    count_ls_iters = 0

    while f(xk + alpha * pk) > fxk + c * alpha * dot_grad_pk:
        alpha = rho * alpha
        count_ls_iters += 1
        if alpha < 1e-12 or count_ls_iters > max_ls_iters : # 防止 alpha 过小或迭代过多
            # print(f"Line search alpha too small ({alpha}) or max_iters ({count_ls_iters})")
            if alpha < 1e-12 and f(xk + alpha * pk) > fxk : # 如果alpha太小且没有改善，返回一个极小的alpha或0
                 return 1e-12 # 或引发错误/警告
            break # 退出循环，使用当前的alpha
    return alpha

def steepest_descent(f, grad_f, x0, max_iter=10000, tol=1e-6, line_search_alpha_init=1.0):
    xk = np.array(x0, dtype=float) 
    path = [xk.copy()]  
    
    iteration_summary = [] 

    for k in range(max_iter):
        grad_xk = grad_f(xk)
        grad_norm = np.linalg.norm(grad_xk)
        
        fxk = f(xk)
        iteration_summary.append((k, xk.copy(), fxk, grad_norm))

        if grad_norm < tol:
            # print(f"梯度范数 {grad_norm:.2e} 小于容差 {tol}, 算法收敛。")
            break
        
        pk = -grad_xk  
        
        alpha = backtracking_line_search(f, grad_xk, xk, pk, alpha_init=line_search_alpha_init)
        
        if alpha < 1e-12 : 
            print(f"迭代 {k+1}: 步长 alpha ({alpha:.2e}) 过小，可能无法取得进展。")
            # break # 可以选择在这里停止，或者继续尝试
        
        xk = xk + alpha * pk
        path.append(xk.copy())
        
        if (k + 1) % 500 == 0: # 每500次迭代打印一次信息
            print(f"迭代 {k+1}: x = {xk}, f(x) = {f(xk):.4e}, ||grad f(x)|| = {grad_norm:.4e}, alpha = {alpha:.2e}")

    else: 
        print(f"达到最大迭代次数 {max_iter}。")

    if not iteration_summary or iteration_summary[-1][0] != k: 
         grad_xk = grad_f(xk)
         grad_norm = np.linalg.norm(grad_xk)
         fxk = f(xk)
         iteration_summary.append((k if k < max_iter else max_iter, xk.copy(), fxk, grad_norm))


    return xk, np.array(path), iteration_summary

def plot_rosenbrock_optimization(path, initial_point_str, ax):
    x1_range = np.linspace(min(path[:,0].min(), -2)-0.5, max(path[:,0].max(), 2)+0.5, 300)
    x2_range = np.linspace(min(path[:,1].min(), -1)-0.5, max(path[:,1].max(), 3)+0.5, 300)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = rosenbrock([X1, X2]) 

    ax.contour(X1, X2, Z, levels=np.logspace(-0.5, 3.5, 20, base=10), cmap='viridis')
    ax.plot(path[:, 0], path[:, 1], 'r.-', markersize=3, linewidth=1, label='优化路径')
    ax.plot(path[0, 0], path[0, 1], 'bo', markersize=6, label='初始点')
    ax.plot(1, 1, 'g*', markersize=10, label='全局最小值 (1,1)') 
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Rosenbrock函数最速下降法 (初始点: {initial_point_str})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    padding_x = (path[:,0].max() - path[:,0].min()) * 0.1 + 0.5
    padding_y = (path[:,1].max() - path[:,1].min()) * 0.1 + 0.5
    ax.set_xlim(path[:,0].min() - padding_x, path[:,0].max() + padding_x)
    ax.set_ylim(path[:,1].min() - padding_y, path[:,1].max() + padding_y)


if __name__ == "__main__":
    initial_points = [
        np.array([-1.2, 1.0]),
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0]),
        np.array([-0.5, 2.5]) 
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten() 

    for i, x0_val in enumerate(initial_points):
        # print(f"\n===== 正在从初始点 {x0_val} 开始优化 =====")
        
        x_star, path_history, iter_data = steepest_descent(
            rosenbrock, 
            rosenbrock_grad, 
            x0_val, 
            max_iter=20000, 
            tol=1e-5,
            line_search_alpha_init=1.0 
        )
        
        final_k, final_x, final_fx, final_grad_norm = iter_data[-1]
        print(f"\n--- 优化结果 (初始点: {x0_val}) ---")
        print(f"迭代次数: {final_k}")
        print(f"最优解 x*: {final_x}")
        print(f"最优函数值 f(x*): {final_fx:.6e}")
        print(f"最终梯度范数 ||grad f(x*)||: {final_grad_norm:.6e}")
        print()
        
        current_ax = axes[i] 
        plot_rosenbrock_optimization(path_history, str(x0_val), current_ax)

    plt.tight_layout() 
    plt.show()

    # 打印部分迭代详情 (示例：前5次和后5次)
    # print("\n--- 部分迭代详情 (示例) ---")
    # if iter_data:
    #     print(f"{'k':<5} | {'x1':<12} | {'x2':<12} | {'f(x)':<15} | {'||grad||':<15}")
    #     print("-" * 65)
    #     display_count = 5
    #     for k_val, x_val, fx_val, grad_n_val in iter_data[:display_count]:
    #         print(f"{k_val:<5} | {x_val[0]:<12.6f} | {x_val[1]:<12.6f} | {fx_val:<15.4e} | {grad_n_val:<15.4e}")
    #     if len(iter_data) > 2 * display_count:
    #         print("...")
    #     start_idx = max(display_count, len(iter_data) - display_count)
    #     for k_val, x_val, fx_val, grad_n_val in iter_data[start_idx:]:
    #          print(f"{k_val:<5} | {x_val[0]:<12.6f} | {x_val[1]:<12.6f} | {fx_val:<15.4e} | {grad_n_val:<15.4e}")

