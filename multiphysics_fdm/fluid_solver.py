import numpy as np
from .field import VectorField, Field
from .operators import gradient, divergence, curl

class FluidSolver:
    """流体求解器 - 基于纳维-斯托克斯方程"""
    
    def __init__(self, mesh, rho=1.0, mu=1.0, gravity=(0.0, 0.0, 0.0)):
        """
        初始化流体求解器
        
        参数:
            mesh: 网格对象
            rho: 流体密度，可以是标量或Field对象
            mu: 流体动力粘度，可以是标量或Field对象
            gravity: 重力加速度矢量
        """
        self.mesh = mesh
        self.rho = rho  # 密度
        self.mu = mu    # 动力粘度
        
        # 流体场
        self.u = VectorField(mesh, "Velocity")  # 速度场
        self.p = Field(mesh, "Pressure")        # 压力场
        self.f = VectorField(mesh, "BodyForce") # 体积力场
        
        # 重力加速度
        self.gravity = gravity
        
        # 时间步长和当前时间
        self.dt = 0.0
        self.time = 0.0
    
    def set_time_step(self, dt):
        """设置时间步长"""
        self.dt = dt
    
    def initialize_fields(self, u_func=None, p_func=None):
        """
        初始化流场
        
        参数:
            u_func: 速度场初始化函数，接受x, y, z返回速度场的三个分量
            p_func: 压力场初始化函数，接受x, y, z返回压力值
        """
        if u_func:
            self.u.set_from_function(u_func)
        else:
            self.u.set_value(0.0, 0.0, 0.0)
        
        if p_func:
            self.p.set_from_function(p_func)
        else:
            self.p.set_value(0.0)
        
        # 设置重力体积力
        self.f.x.set_value(self.gravity[0] * self.rho if not isinstance(self.rho, Field) else 0.0)
        self.f.y.set_value(self.gravity[1] * self.rho if not isinstance(self.rho, Field) else 0.0)
        self.f.z.set_value(self.gravity[2] * self.rho if not isinstance(self.rho, Field) else 0.0)
        
        if isinstance(self.rho, Field):
            for i in range(self.mesh.nx):
                for j in range(self.mesh.ny):
                    for k in range(self.mesh.nz):
                        self.f.x.data[i, j, k] = self.gravity[0] * self.rho.data[i, j, k]
                        self.f.y.data[i, j, k] = self.gravity[1] * self.rho.data[i, j, k]
                        self.f.z.data[i, j, k] = self.gravity[2] * self.rho.data[i, j, k]
    
    def add_body_force(self, f_func):
        """
        添加体积力
        
        参数:
            f_func: 体积力函数，接受x, y, z, u, v, w返回体积力的三个分量
        """
        for i in range(self.mesh.nx):
            for j in range(self.mesh.ny):
                for k in range(self.mesh.nz):
                    x = self.mesh.x[i]
                    y = self.mesh.y[j]
                    z = self.mesh.z[k]
                    u = self.u.x.data[i, j, k]
                    v = self.u.y.data[i, j, k]
                    w = self.u.z.data[i, j, k]
                    fx, fy, fz = f_func(x, y, z, u, v, w)
                    self.f.x.data[i, j, k] += fx
                    self.f.y.data[i, j, k] += fy
                    self.f.z.data[i, j, k] += fz
    
    def solve_step(self):
        """执行一个时间步的求解 (使用投影方法)"""
        if self.dt == 0.0:
            raise ValueError("时间步长未设置，请先调用set_time_step方法")
        
        # 存储旧速度场用于计算
        u_old = self.u.x.data.copy()
        v_old = self.u.y.data.copy()
        w_old = self.u.z.data.copy()
        
        # 第一步: 计算临时速度场 (忽略压力梯度项)
        # ∂u/∂t + (u·∇)u = (μ/ρ)∇²u + f/ρ
        # 使用半隐式方法处理扩散项，显式处理对流项
        
        # 计算速度梯度 (对流项)
        grad_u_x, grad_u_y, grad_u_z = gradient(u_old, self.mesh)
        grad_v_x, grad_v_y, grad_v_z = gradient(v_old, self.mesh)
        grad_w_x, grad_w_y, grad_w_z = gradient(w_old, self.mesh)
        
        # 计算对流项 (u·∇)u
        convection_x = u_old * grad_u_x + v_old * grad_u_y + w_old * grad_u_z
        convection_y = u_old * grad_v_x + v_old * grad_v_y + w_old * grad_v_z
        convection_z = u_old * grad_w_x + v_old * grad_w_y + w_old * grad_w_z
        
        # 计算拉普拉斯算子 (扩散项)
        lap_u = laplacian(u_old, self.mesh)
        lap_v = laplacian(v_old, self.mesh)
        lap_w = laplacian(w_old, self.mesh)
        
        # 计算扩散系数 μ/ρ
        if isinstance(self.mu, Field) or isinstance(self.rho, Field):
            if isinstance(self.mu, Field):
                mu = self.mu.data
            else:
                mu = np.full_like(u_old, self.mu)
            
            if isinstance(self.rho, Field):
                rho = self.rho.data
            else:
                rho = np.full_like(u_old, self.rho)
            
            diffusion_coeff = mu / rho
        else:
            diffusion_coeff = self.mu / self.rho
        
        # 计算临时速度场
        u_temporary = u_old + self.dt * (-convection_x + diffusion_coeff * lap_u + self.f.x.data / self.rho)
        v_temporary = v_old + self.dt * (-convection_y + diffusion_coeff * lap_v + self.f.y.data / self.rho)
        w_temporary = w_old + self.dt * (-convection_z + diffusion_coeff * lap_w + self.f.z.data / self.rho)
        
        # 第二步: 求解压力泊松方程 ∇²p = ρ/Δt ∇·u*
        # 计算临时速度场的散度
        div_u_temporary = divergence(VectorField(self.mesh, "TempVelocity", u_temporary, v_temporary, w_temporary), self.mesh)
        
        # 求解压力泊松方程 (使用简单的迭代方法)
        p_old = self.p.data.copy()
        max_iter = 100
        tolerance = 1e-6
        
        for iter in range(max_iter):
            p_new = p_old.copy()
            
            # 内部点的压力更新
            p_new[1:-1, 1:-1, 1:-1] = 1/6 * (p_old[2:, 1:-1, 1:-1] + p_old[:-2, 1:-1, 1:-1] +
                                           p_old[1:-1, 2:, 1:-1] + p_old[1:-1, :-2, 1:-1] +
                                           p_old[1:-1, 1:-1, 2:] + p_old[1:-1, 1:-1, :-2] -
                                           self.mesh.dx**2 * self.rho * div_u_temporary[1:-1, 1:-1, 1:-1] / self.dt)
            
            # 边界条件 (零梯度边界)
            p_new[0, :, :] = p_new[1, :, :]
            p_new[-1, :, :] = p_new[-2, :, :]
            p_new[:, 0, :] = p_new[:, 1, :]
            p_new[:, -1, :] = p_new[:, -2, :]
            p_new[:, :, 0] = p_new[:, :, 1]
            p_new[:, :, -1] = p_new[:, :, -2]
            
            # 检查收敛性
            error = np.max(np.abs(p_new - p_old))
            if error < tolerance:
                break
            
            p_old = p_new.copy()
        
        self.p.data = p_new
        
        # 第三步: 更新速度场以满足连续性方程 ∇·u = 0
        # u = u* - (Δt/ρ)∇p
        grad_p_x, grad_p_y, grad_p_z = gradient(self.p.data, self.mesh)
        
        self.u.x.data = u_temporary - (self.dt / self.rho) * grad_p_x
        self.u.y.data = v_temporary - (self.dt / self.rho) * grad_p_y
        self.u.z.data = w_temporary - (self.dt / self.rho) * grad_p_z
        
        # 更新时间
        self.time += self.dt
    
    def get_max_stable_time_step(self):
        """
        计算最大稳定时间步长 (CFL条件)
        
        返回:
            float: 最大稳定时间步长
        """
        dx = self.mesh.dx
        dy = self.mesh.dy
        dz = self.mesh.dz
        
        # 计算最大速度
        max_u = np.max(np.abs(self.u.x.data))
        max_v = np.max(np.abs(self.u.y.data))
        max_w = np.max(np.abs(self.u.z.data))
        max_velocity = max(max_u, max_v, max_w)
        
        # 计算最小粘度
        if isinstance(self.mu, Field):
            min_mu = np.min(self.mu.data)
        else:
            min_mu = self.mu
        
        if isinstance(self.rho, Field):
            max_rho = np.max(self.rho.data)
        else:
            max_rho = self.rho
        
        min_viscosity = min_mu / max_rho
        
        # CFL条件: dt <= min(dx/|u|, dy/|v|, dz/|w|, dx^2/(4*ν), dy^2/(4*ν), dz^2/(4*ν))
        dt_cfl = min(dx/(max_velocity + 1e-10), dy/(max_velocity + 1e-10), dz/(max_velocity + 1e-10))
        dt_visc = 0.25 * min(dx**2, dy**2, dz**2) / (min_viscosity + 1e-10)
        
        return min(dt_cfl, dt_visc)    