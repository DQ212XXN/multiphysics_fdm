import sys
# 把库所在的绝对路径添加进去
sys.path.append('/path/to/your/library')

import numpy as np
import matplotlib.pyplot as plt
from multiphysics_fdm.mesh import Mesh
from multiphysics_fdm.coupled_solver import CoupledSolver
from multiphysics_fdm.postprocessor import Postprocessor

# 创建网格
mesh = Mesh(
    x_min=0.0, x_max=1.0, nx=50,
    y_min=0.0, y_max=1.0, ny=50,
    z_min=0.0, z_max=0.1, nz=5  # 简化为2D问题
)

# 创建耦合求解器
solver = CoupledSolver(mesh)

# 设置电磁场求解器
solver.set_em_solver(
    epsilon=1.0,
    mu=1.0,
    sigma=1.0  # 电导率
)

# 设置热传导求解器
solver.set_thermal_solver(
    k=0.5,  # 热导率
    rho=1.0,  # 密度
    cp=1.0  # 比热容
)

# 设置流体求解器
solver.set_fluid_solver(
    rho=1.0,  # 流体密度
    mu=0.1,  # 流体粘度
    gravity=(0.0, -9.81, 0.0)  # 重力加速度
)

# 启用物理场耦合
solver.enable_em_thermal_coupling(
    joule_heating_factor=1.0,  # 焦耳热因子
    seebeck_coefficient=0.0  # 塞贝克系数
)

solver.enable_thermal_fluid_coupling(
    thermal_expansion_coefficient=0.001  # 热膨胀系数
)

# 初始化电磁场
def E_initial(x, y, z):
    # 在中心位置设置一个点电荷产生的电场
    center_x, center_y, center_z = 0.5, 0.5, 0.05
    r = np.sqrt((x-center_x)**2 + (y-center_y)**2 + (z-center_z)**2)
    r = max(r, 0.1)  # 避免除以零
    
    Ex = 10.0 * (x - center_x) / r**3
    Ey = 10.0 * (y - center_y) / r**3
    Ez = 10.0 * (z - center_z) / r**3
    
    return Ex, Ey, Ez

solver.em_solver.initialize_fields(
    E_func=E_initial,
    B_func=lambda x, y, z: (0.0, 0.0, 0.0)  # 初始磁场为零
)

# 初始化温度场
def t_initial(x, y, z):
    # 初始温度为300K，中心位置有一个热点
    center_x, center_y, center_z = 0.5, 0.5, 0.05
    r = np.sqrt((x-center_x)**2 + (y-center_y)**2 + (z-center_z)**2)
    
    return 300.0 + 50.0 * np.exp(-r**2 / 0.1**2)

solver.thermal_solver.initialize_temperature(
    T_func=t_initial
)

# 初始化流场
solver.fluid_solver.initialize_fields(
    u_func=lambda x, y, z: (0.0, 0.0, 0.0),  # 初始速度为零
    p_func=lambda x, y, z: 101325.0  # 初始压力为大气压
)

# 设置时间步长
dt = solver.get_max_stable_time_step() * 0.1  # 使用安全系数
solver.set_time_step(dt)

# 创建后处理器
postprocessor = Postprocessor(mesh)

# 求解并可视化结果
total_time = 1.0
num_steps = int(total_time / dt)

for step in range(num_steps):
    solver.solve_step()
    
    # 每10步输出一次进度
    if step % 10 == 0:
        print(f"Step {step}/{num_steps}, Time: {solver.time:.4f}s")
        
        # 可视化温度场
        postprocessor.plot_scalar_field(
            solver.thermal_solver.T.data,
            title=f"Temperature at t={solver.time:.2f}s",
            save_path=f"temperature_step_{step}.png"
        )
        
        # 可视化速度场
        postprocessor.plot_vector_field(
            solver.fluid_solver.u,
            title=f"Velocity at t={solver.time:.2f}s",
            save_path=f"velocity_step_{step}.png"
        )

# 导出最终结果到VTK格式
postprocessor.export_to_vtk({
    'temperature': solver.thermal_solver.T.data,
    'velocity': solver.fluid_solver.u,
    'electric_field': solver.em_solver.E,
    'magnetic_field': solver.em_solver.B
}, "final_result.vtk")    