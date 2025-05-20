import numpy as np
import matplotlib.pyplot as plt

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
    """
    回溯线性搜索寻找合适的步长 alpha。
    参数:
    f (function): 目标函数。
    grad_f_xk (np.ndarray): 当前点 xk 的梯度。
    xk (np.ndarray): 当前点。
    pk (np.ndarray): 搜索方向。
    alpha_init (float): 初始步长。
    rho (float): 步长缩减因子 (0 < rho < 1)。
    c (float): Armijo 条件的控制参数 (0 < c < 1)。
    返回:
    float: 计算得到的步长 alpha。
    """
    alpha = alpha_init
    fxk = f(xk)
    dot_grad_pk = np.dot(grad_f_xk, pk) # grad_f_xk.T @ pk

    max_ls_iters = 100 # 避免线性搜索无限循环
    count_ls_iters = 0

    # Armijo 条件: f(xk + alpha*pk) <= f(xk) + c * alpha * grad_f_xk.T @ pk
    while f(xk + alpha * pk) > fxk + c * alpha * dot_grad_pk:
        alpha = rho * alpha
        count_ls_iters += 1
        if alpha < 1e-12 or count_ls_iters > max_ls_iters: # 防止 alpha 过小或迭代过多
            if alpha < 1e-12 and f(xk + alpha * pk) > fxk :
                 return 1e-12 # 返回一个极小的alpha
            break 
    return alpha

# 4. 实现 DFP 拟牛顿法
def dfp_method(f, grad_f, x0, max_iter=500, tol=1e-5, line_search_alpha_init=1.0):
    """
    使用 DFP 拟牛顿法最小化函数 f。
    参数:
    f (function): 目标函数。
    grad_f (function): 目标函数的梯度函数。
    x0 (np.ndarray): 初始点。
    max_iter (int): 最大迭代次数。
    tol (float): 梯度范数的收敛容差。
    line_search_alpha_init (float): 回溯线性搜索的初始alpha值。
    返回:
    tuple: (找到的最优解 x_star, 历史路径 path, 迭代摘要)
    """
    xk = np.array(x0, dtype=float)
    n = len(xk)
    Hk = np.eye(n)  # 初始化 H0 为单位矩阵

    path = [xk.copy()]
    iteration_summary = [] # 用于存储 (k, xk, f(xk), ||grad_f(xk)||)
    
    grad_xk = grad_f(xk) # 初始梯度

    for k in range(max_iter):
        grad_norm = np.linalg.norm(grad_xk)
        fxk = f(xk)
        iteration_summary.append((k, xk.copy(), fxk, grad_norm))

        if grad_norm < tol:
            print(f"迭代 {k}: 梯度范数 {grad_norm:.2e} 小于容差 {tol}, 算法收敛。")
            break
        
        # 1. 计算搜索方向
        pk = -np.dot(Hk, grad_xk)
        
        # 2. 线搜索确定步长 alpha_k
        #    在线搜索中，传入的是当前梯度 grad_xk，因为 pk 是基于它计算的
        alpha_k = backtracking_line_search(f, grad_xk, xk, pk, alpha_init=line_search_alpha_init)

        if alpha_k < 1e-12:
            print(f"迭代 {k+1}: 步长 alpha ({alpha_k:.2e}) 过小，可能无法取得进展或Hk不正定。")
            # 可以考虑重置Hk为单位矩阵或提前终止
            # Hk = np.eye(n) # 重置Hk
            # continue # 尝试用重置的Hk进行下一次迭代
            break


        # 3. 更新点
        x_next = xk + alpha_k * pk
        
        # 4. 计算 s_k 和 y_k
        sk = x_next - xk
        
        grad_x_next = grad_f(x_next)
        yk = grad_x_next - grad_xk
        
        # 5. 更新 H_k (DFP公式)
        # 确保分母不为零，且 s_k^T y_k > 0 (曲率条件)
        skyk_dot = np.dot(sk, yk)
        
        if skyk_dot <= 1e-8: # 如果曲率条件不满足或接近于0
            print(f"迭代 {k+1}: 曲率条件 s_k^T y_k = {skyk_dot:.2e} 不满足或过小，跳过Hk更新或重置。")
            # 可以选择跳过更新，或者重置 Hk 为单位矩阵
            # Hk = np.eye(n) # 重置 Hk
        else:
            term1_num = np.outer(sk, sk) # sk * sk.T
            term1_den = skyk_dot
            term1 = term1_num / term1_den
            
            Hkyk = np.dot(Hk, yk)
            term2_num = np.outer(Hkyk, Hkyk) # (Hk*yk) * (Hk*yk).T
            term2_den = np.dot(yk, Hkyk)    # yk.T * Hk * yk
            
            if abs(term2_den) < 1e-8: # 避免除以零
                 print(f"迭代 {k+1}: DFP更新中分母 yk.T*Hk*yk = {term2_den:.2e} 过小，跳过Hk更新或重置。")
                 # Hk = np.eye(n) # 重置 Hk
            else:
                term2 = term2_num / term2_den
                Hk = Hk + term1 - term2
        
        # 更新迭代变量
        xk = x_next
        grad_xk = grad_x_next # 更新梯度为新点的梯度
        path.append(xk.copy())
        
        if (k + 1) % 50 == 0: # 每50次迭代打印一次信息
            print(f"迭代 {k+1}: x = {xk}, f(x) = {f(xk):.4e}, ||grad f(x)|| = {grad_norm:.4e}, alpha = {alpha_k:.2e}")

    else: # for循环正常结束 (未被break中断)
        print(f"达到最大迭代次数 {max_iter}。")

    # 确保最后一次迭代信息被记录
    if not iteration_summary or iteration_summary[-1][0] != k :
         grad_norm = np.linalg.norm(grad_xk)
         fxk = f(xk)
         iteration_summary.append((k if k < max_iter else max_iter, xk.copy(), fxk, grad_norm))
         
    return xk, np.array(path), iteration_summary

def plot_rosenbrock_optimization_dfp(path, initial_point_str, ax):
    """
    绘制 Rosenbrock 函数的等高线图和DFP优化路径。
    """
    # 动态调整绘图范围以更好地显示路径
    x_min_plot, x_max_plot = path[:,0].min() - 0.5, path[:,0].max() + 0.5
    y_min_plot, y_max_plot = path[:,1].min() - 0.5, path[:,1].max() + 0.5
    
    # 确保 (1,1) 点在视图内
    x_min_plot = min(x_min_plot, 0.5)
    x_max_plot = max(x_max_plot, 1.5)
    y_min_plot = min(y_min_plot, 0.5)
    y_max_plot = max(y_max_plot, 1.5)

    # 限制范围，避免过大
    x_min_plot = max(x_min_plot, -2.5)
    x_max_plot = min(x_max_plot, 2.5)
    y_min_plot = max(y_min_plot, -1.5)
    y_max_plot = min(y_max_plot, 3.5)


    x1_range = np.linspace(x_min_plot, x_max_plot, 300)
    x2_range = np.linspace(y_min_plot, y_max_plot, 300)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = rosenbrock([X1, X2])

    ax.contour(X1, X2, Z, levels=np.logspace(-0.5, 3.5, 20, base=10), cmap='viridis')
    ax.plot(path[:, 0], path[:, 1], 'r.-', markersize=3, linewidth=1, label='优化路径 (DFP)')
    ax.plot(path[0, 0], path[0, 1], 'bo', markersize=6, label='初始点')
    ax.plot(1, 1, 'g*', markersize=10, label='全局最小值 (1,1)')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Rosenbrock函数DFP法 (初始点: {initial_point_str})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(x1_range.min(), x1_range.max())
    ax.set_ylim(x2_range.min(), x2_range.max())


if __name__ == "__main__":
    initial_points = [
        np.array([-1.2, 1.0]),
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0]),
        np.array([-0.5, 2.5]) 
    ]

    num_points = len(initial_points)
    if num_points == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        axes = [ax]
    else:
        ncols = 2
        nrows = (num_points + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 6))
        axes = axes.flatten()

    for i, x0_val in enumerate(initial_points):
        print(f"\n===== 正在从初始点 {x0_val} 开始使用DFP方法优化 =====")
        
        x_star, path_history, iter_data = dfp_method(
            rosenbrock, 
            rosenbrock_grad, 
            x0_val, 
            max_iter=100, # DFP通常比最速下降快，迭代次数可以少些
            tol=1e-5,
            line_search_alpha_init=1.0 
        )
        
        final_k, final_x, final_fx, final_grad_norm = iter_data[-1]
        print(f"\n--- DFP优化结果 (初始点: {x0_val}) ---")
        print(f"迭代次数: {final_k}")
        print(f"最优解 x*: {final_x}")
        print(f"最优函数值 f(x*): {final_fx:.6e}")
        print(f"最终梯度范数 ||grad f(x*)||: {final_grad_norm:.6e}")
        
        current_ax = axes[i] if num_points > 1 else axes[0]
        plot_rosenbrock_optimization_dfp(path_history, str(x0_val), current_ax)

    if num_points > 1:
        for j in range(num_points, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
