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
    alpha = alpha_init
    fxk = f(xk)
    dot_grad_pk = np.dot(grad_f_xk, pk)
    max_ls_iters = 100 
    count_ls_iters = 0
    while f(xk + alpha * pk) > fxk + c * alpha * dot_grad_pk:
        alpha = rho * alpha
        count_ls_iters += 1
        if alpha < 1e-12 or count_ls_iters > max_ls_iters : 
            if alpha < 1e-12 and f(xk + alpha * pk) > fxk : 
                 return 1e-12 
            break 
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
            break
        
        pk = -grad_xk  
        alpha = backtracking_line_search(f, grad_xk, xk, pk, alpha_init=line_search_alpha_init)
        
        if alpha < 1e-12 : 
            print(f"迭代 {k+1}: 步长 alpha ({alpha:.2e}) 过小，可能无法取得进展。")
        
        xk = xk + alpha * pk
        path.append(xk.copy())
        
        if (k + 1) % 1000 == 0: 
            print(f"迭代 {k+1}: x = {xk}, f(x) = {f(xk):.4e}, ||grad f(x)|| = {grad_norm:.4e}, alpha = {alpha:.2e}")
    else: 
        print(f"达到最大迭代次数 {max_iter}。")

    if not iteration_summary or iteration_summary[-1][0] != k: 
         grad_xk = grad_f(xk)
         grad_norm = np.linalg.norm(grad_xk)
         fxk = f(xk)
         iteration_summary.append((k if k < max_iter else max_iter, xk.copy(), fxk, grad_norm))
    return xk, np.array(path), iteration_summary

def plot_rosenbrock_optimization_2d(path, initial_point_str, ax):
    x_min_plot = min(path[:,0].min(), 1, -2) - 0.5
    x_max_plot = max(path[:,0].max(), 1, 2) + 0.5
    y_min_plot = min(path[:,1].min(), 1, -1) - 0.5
    y_max_plot = max(path[:,1].max(), 1, 3) + 0.5
    
    x1_range = np.linspace(x_min_plot, x_max_plot, 200)
    x2_range = np.linspace(y_min_plot, y_max_plot, 200)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = rosenbrock([X1, X2]) 

    ax.contour(X1, X2, Z, levels=np.logspace(-0.5, 3.5, 25, base=10), cmap='viridis') 
    ax.plot(path[:, 0], path[:, 1], 'r.-', markersize=2, linewidth=0.8, label='优化路径') 
    ax.plot(path[0, 0], path[0, 1], 'bo', markersize=5, label='初始点')
    ax.plot(1, 1, 'g*', markersize=8, label='全局最小值 (1,1)') 
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'2D等高线图 (初始点: {initial_point_str})')
    ax.legend(fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(x_min_plot, x_max_plot)
    ax.set_ylim(y_min_plot, y_max_plot)

def plot_rosenbrock_optimization_3d(path, initial_point_str, ax):
    x_min_plot = min(path[:,0].min(), 1, -2) - 0.5
    x_max_plot = max(path[:,0].max(), 1, 2) + 0.5
    y_min_plot = min(path[:,1].min(), 1, -1) - 0.5
    y_max_plot = max(path[:,1].max(), 1, 3) + 0.5

    x1_surf = np.linspace(x_min_plot, x_max_plot, 100) 
    x2_surf = np.linspace(y_min_plot, y_max_plot, 100)
    X1_surf, X2_surf = np.meshgrid(x1_surf, x2_surf)
    Z_surf = rosenbrock([X1_surf, X2_surf])

    ax.plot_surface(X1_surf, X2_surf, Z_surf, cmap='viridis', alpha=0.6, edgecolor='none', rstride=5, cstride=5)

    path_z = np.array([rosenbrock(p) for p in path])
    ax.plot(path[:, 0], path[:, 1], path_z, 'r.-', markersize=2, linewidth=1, label='优化路径')
    ax.scatter(path[0, 0], path[0, 1], rosenbrock(path[0,:]), color='blue', s=50, label='初始点', depthshade=True)
    ax.scatter(1, 1, rosenbrock(np.array([1,1])), color='green', marker='*', s=100, label='全局最小值 (1,1)', depthshade=True)
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1, x_2)$')
    ax.set_title(f'3D曲面图 (初始点: {initial_point_str})')
    ax.legend(fontsize='small')
    ax.view_init(elev=25, azim=-130) 

if __name__ == "__main__":
    initial_points = [
        np.array([-1.2, 1.0]),
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0]),
        np.array([-0.5, 2.5]) 
    ]
    num_initial_points = len(initial_points)

    fig = plt.figure(figsize=(24, 12)) 

    for i, x0_val in enumerate(initial_points):
        x_star, path_history, iter_data = steepest_descent(
            rosenbrock, 
            rosenbrock_grad, 
            x0_val, 
            max_iter=20000, 
            tol=1e-5,
            line_search_alpha_init=1.0 
        )
        
        final_k, final_x, final_fx, final_grad_norm = iter_data[-1]
        print(f"--- 优化结果 (初始点: {x0_val}) ---")
        print(f"迭代次数: {final_k}")
        print(f"最优解 x*: {final_x}")
        print(f"最优函数值 f(x*): {final_fx:.6e}")
        print(f"最终梯度范数 ||grad f(x*)||: {final_grad_norm:.6e}\n")
        
        ax_2d = fig.add_subplot(2, num_initial_points, i + 1)
        plot_rosenbrock_optimization_2d(path_history, str(x0_val), ax_2d)
        
        ax_3d = fig.add_subplot(2, num_initial_points, i + 1 + num_initial_points, projection='3d')
        plot_rosenbrock_optimization_3d(path_history, str(x0_val), ax_3d)

    plt.tight_layout(pad=3.0, h_pad=4.0) 
    plt.suptitle("最速下降法优化Rosenbrock函数 - 不同初始点对比", fontsize=16, y=0.99)
    fig.subplots_adjust(top=0.92) 
    plt.show()
