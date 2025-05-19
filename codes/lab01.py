import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Hei'

x1 = np.linspace(0, 10, 400) 

plt.figure(figsize=(10, 8))

# 约束1: x2 >= (12 - 2x1) / 5
y1 = (12 - 2 * x1) / 5
plt.plot(x1, y1, label=r'$2x_1 + 5x_2 \geq 12$')

# 约束2: x2 <= (8 - x1) / 2
y2 = (8 - x1) / 2
plt.plot(x1, y2, label=r'$x_1 + 2x_2 \leq 8$')

# 约束3: 0 <= x1 <= 4
plt.axvline(x=4, color='gray', linestyle='--', label=r'$x_1 \leq 4$')
plt.axvline(x=0, color='gray', linestyle=':', label=r'$x_1 \geq 0$')

# 约束4: 0 <= x2 <= 3
plt.axhline(y=3, color='purple', linestyle='--', label=r'$x_2 \leq 3$')
plt.axhline(y=0, color='purple', linestyle=':', label=r'$x_2 \geq 0$')

plt.xlim(0, 6) 
plt.ylim(0, 5) 

d = np.linspace(-2, 8, 300)
x1_fill, x2_fill = np.meshgrid(d, d)

feasible_region = (
    (2 * x1_fill + 5 * x2_fill >= 12) &
    (x1_fill + 2 * x2_fill <= 8) &
    (x1_fill >= 0) & (x1_fill <= 4) &
    (x2_fill >= 0) & (x2_fill <= 3)
)
plt.imshow(
    feasible_region.astype(int),
    extent=(x1_fill.min(), x1_fill.max(), x2_fill.min(), x2_fill.max()),
    origin="lower",
    cmap="Greens",
    alpha=0.3,
)

# 找到可行域的顶点
p1 = (2, 3)
p2 = (4, 2) 
p3 = (4, 0.8)

vertices = []
x1_v = 8 - 2*3
if 0 <= x1_v <= 4:
    vertices.append((x1_v, 3)) 

x2_v = (8 - 4) / 2
if 0 <= x2_v <= 3: 
    vertices.append((4, x2_v)) 

x2_v = (12 - 2*4) / 5
if 0 <= x2_v <= 3: 
    vertices.append((4, x2_v)) 

x2_v = 12 / 5
if 0 <= x2_v <= 3: 
    vertices.append((0, x2_v)) 

if (0 + 2*3 <= 8) and (2*0 + 5*3 >=12): 
    vertices.append((0,3))

points_to_check = [
    (0, 2.4), (0, 3), (2, 3), (4, 2), (4, 0.8)
]
valid_vertices = []
for p_x, p_y in points_to_check:
    if (2*p_x + 5*p_y >= 12 - 1e-9) and \
       (p_x + 2*p_y <= 8 + 1e-9) and \
       (0 - 1e-9 <= p_x <= 4 + 1e-9) and \
       (0 - 1e-9 <= p_y <= 3 + 1e-9):
        valid_vertices.append((p_x, p_y))

vx, vy = zip(*valid_vertices)
plt.plot(vx, vy, 'ro', label='可行域顶点') 

max_z = -np.inf
optimal_point = None

print("可行域顶点及其目标函数值:")
for p_x, p_y in valid_vertices:
    z_value = 2 * p_x + p_y
    print(f"点 ({p_x:.2f}, {p_y:.2f}): z = {z_value:.2f}")
    if z_value > max_z:
        max_z = z_value
        optimal_point = (p_x, p_y)

print(f"\n最优解:")
if optimal_point:
    print(f"在点 ({optimal_point[0]:.2f}, {optimal_point[1]:.2f}) 处取得最大值")
    print(f"最大值 z = {max_z:.2f}\n")

    x_opt_line = np.linspace(0, 5, 100)
    y_opt_line = max_z - 2 * x_opt_line
    plt.plot(x_opt_line, y_opt_line, 'k--', label=f'最优目标函数 $z={max_z:.2f}$')
    plt.plot(optimal_point[0], optimal_point[1], 'g*', markersize=15, label=f'最优点 ({optimal_point[0]:.2f}, {optimal_point[1]:.2f})')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('线性规划图解法')
plt.legend(loc='upper right')
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()
