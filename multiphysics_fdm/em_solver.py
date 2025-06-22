import numpy as np
from .field import VectorField, Field
from .operators import curl, divergence

class EMSolver:
    """电磁场求解器 - 基于麦克斯韦方程组"""
    
    def __init__(self, mesh, epsilon=1.0, mu=1.0, sigma=0.0):
        """
        初始化电磁场求解器
        
        参数:
            mesh: 网格对象
            epsilon: 介电常数，可以是标量或Field对象
            mu: 磁导率，可以是标量或Field对象
            sigma: 电导率，可以是标量或Field对象
        """
        self.mesh = mesh
        self.epsilon = epsilon
        self.mu = mu
        self.sigma = sigma
        
        # 电磁场
        self.E = VectorField(mesh, "ElectricField")  # 电场
        self.B = VectorField(mesh, "MagneticField")  # 磁场
        self.D = VectorField(mesh, "ElectricDisplacement")  # 电位移场
        self.H = VectorField(mesh, "MagneticFieldStrength")  # 磁场强度
        self.J = VectorField(mesh, "CurrentDensity")  # 电流密度
        
        # 电荷密度
        self.rho = Field(mesh, "ChargeDensity")  # 电荷密度
        
        # 时间步长和当前时间
        self.dt = 0.0
        self.time = 0.0
    
    def set_time_step(self, dt):
        """设置时间步长"""
        self.dt = dt
    
    def initialize_fields(self, E_func=None, B_func=None, rho_func=None):
        """
        初始化电磁场
        
        参数:
            E_func: 电场初始化函数，接受x, y, z返回E场的三个分量
            B_func: 磁场初始化函数，接受x, y, z返回B场的三个分量
            rho_func: 电荷密度初始化函数，接受x, y, z返回电荷密度
        """
        if E_func:
            self.E.set_from_function(E_func)
        
        if B_func:
            self.B.set_from_function(B_func)
        
        if rho_func:
            self.rho.set_from_function(rho_func)
        
        # 更新辅助场
        self._update_auxiliary_fields()
    
    def _update_auxiliary_fields(self):
        """更新辅助场"""
        # D = εE
        if isinstance(self.epsilon, Field):
            self.D.x.data = self.epsilon.data * self.E.x.data
            self.D.y.data = self.epsilon.data * self.E.y.data
            self.D.z.data = self.epsilon.data * self.E.z.data
        else:
            self.D.x.data = self.epsilon * self.E.x.data
            self.D.y.data = self.epsilon * self.E.y.data
            self.D.z.data = self.epsilon * self.E.z.data
        
        # H = B/μ
        if isinstance(self.mu, Field):
            self.H.x.data = self.B.x.data / self.mu.data
            self.H.y.data = self.B.y.data / self.mu.data
            self.H.z.data = self.B.z.data / self.mu.data
        else:
            self.H.x.data = self.B.x.data / self.mu
            self.H.y.data = self.B.y.data / self.mu
            self.H.z.data = self.B.z.data / self.mu
        
        # J = σE
        if isinstance(self.sigma, Field):
            self.J.x.data = self.sigma.data * self.E.x.data
            self.J.y.data = self.sigma.data * self.E.y.data
            self.J.z.data = self.sigma.data * self.E.z.data
        else:
            self.J.x.data = self.sigma * self.E.x.data
            self.J.y.data = self.sigma * self.E.y.data
            self.J.z.data = self.sigma * self.E.z.data
    
    def solve_step(self):
        """执行一个时间步的求解"""
        if self.dt == 0.0:
            raise ValueError("时间步长未设置，请先调用set_time_step方法")
        
        # 更新电场 (法拉第电磁感应定律)
        curl_H_x, curl_H_y, curl_H_z = curl(self.H, self.mesh)
        
        # ∂E/∂t = (∇×H - J) / ε
        if isinstance(self.epsilon, Field):
            self.E.x.data += self.dt * (curl_H_x - self.J.x.data) / self.epsilon.data
            self.E.y.data += self.dt * (curl_H_y - self.J.y.data) / self.epsilon.data
            self.E.z.data += self.dt * (curl_H_z - self.J.z.data) / self.epsilon.data
        else:
            self.E.x.data += self.dt * (curl_H_x - self.J.x.data) / self.epsilon
            self.E.y.data += self.dt * (curl_H_y - self.J.y.data) / self.epsilon
            self.E.z.data += self.dt * (curl_H_z - self.J.z.data) / self.epsilon
        
        # 更新磁场 (安培-麦克斯韦定律)
        curl_E_x, curl_E_y, curl_E_z = curl(self.E, self.mesh)
        
        # ∂B/∂t = -∇×E
        self.B.x.data -= self.dt * curl_E_x
        self.B.y.data -= self.dt * curl_E_y
        self.B.z.data -= self.dt * curl_E_z
        
        # 更新电荷密度 (连续性方程)
        div_J = divergence(self.J, self.mesh)
        self.rho.data -= self.dt * div_J
        
        # 更新辅助场
        self._update_auxiliary_fields()
        
        # 更新时间
        self.time += self.dt
    
    def get_max_stable_time_step(self, c=1.0):
        """
        计算最大稳定时间步长 (CFL条件)
        
        参数:
            c: 波传播速度
        
        返回:
            float: 最大稳定时间步长
        """
        dx = self.mesh.dx
        dy = self.mesh.dy
        dz = self.mesh.dz
        
        # 三维空间中的CFL条件
        dt_max = 1.0 / (c * np.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2))
        
        return dt_max    