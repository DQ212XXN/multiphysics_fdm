import numpy as np
from .field import Field
from .operators import laplacian

class ThermalSolver:
    """热传导求解器 - 基于热传导方程"""
    
    def __init__(self, mesh, k=1.0, rho=1.0, cp=1.0):
        """
        初始化热传导求解器
        
        参数:
            mesh: 网格对象
            k: 热导率，可以是标量或Field对象
            rho: 密度，可以是标量或Field对象
            cp: 比热容，可以是标量或Field对象
        """
        self.mesh = mesh
        self.k = k  # 热导率
        self.rho = rho  # 密度
        self.cp = cp  # 比热容
        
        # 温度场
        self.T = Field(mesh, "Temperature")  # 温度场
        self.Q = Field(mesh, "HeatSource")  # 热源项
        
        # 时间步长和当前时间
        self.dt = 0.0
        self.time = 0.0
    
    def set_time_step(self, dt):
        """设置时间步长"""
        self.dt = dt
    
    def initialize_temperature(self, T_func=None):
        """
        初始化温度场
        
        参数:
            T_func: 温度初始化函数，接受x, y, z返回温度值
        """
        if T_func:
            self.T.set_from_function(T_func)
        else:
            self.T.set_value(0.0)
        
        # 初始化热源为零
        self.Q.set_value(0.0)
    
    def add_heat_source(self, Q_func):
        """
        添加热源项
        
        参数:
            Q_func: 热源函数，接受x, y, z, T返回热源强度
        """
        for i in range(self.mesh.nx):
            for j in range(self.mesh.ny):
                for k in range(self.mesh.nz):
                    x = self.mesh.x[i]
                    y = self.mesh.y[j]
                    z = self.mesh.z[k]
                    T = self.T.data[i, j, k]
                    self.Q.data[i, j, k] = Q_func(x, y, z, T)
    
    def solve_step(self):
        """执行一个时间步的求解"""
        if self.dt == 0.0:
            raise ValueError("时间步长未设置，请先调用set_time_step方法")
        
        # 计算拉普拉斯算子 ∇²T
        lap_T = laplacian(self.T.data, self.mesh)
        
        # 计算热扩散系数 α = k/(ρ*cp)
        if isinstance(self.k, Field) or isinstance(self.rho, Field) or isinstance(self.cp, Field):
            if isinstance(self.k, Field):
                k = self.k.data
            else:
                k = np.full_like(self.T.data, self.k)
            
            if isinstance(self.rho, Field):
                rho = self.rho.data
            else:
                rho = np.full_like(self.T.data, self.rho)
            
            if isinstance(self.cp, Field):
                cp = self.cp.data
            else:
                cp = np.full_like(self.T.data, self.cp)
            
            alpha = k / (rho * cp)
        else:
            alpha = self.k / (self.rho * self.cp)
        
        # 热传导方程: ∂T/∂t = α∇²T + Q/(ρ*cp)
        if isinstance(self.rho, Field) or isinstance(self.cp, Field):
            if isinstance(self.rho, Field):
                rho = self.rho.data
            else:
                rho = np.full_like(self.T.data, self.rho)
            
            if isinstance(self.cp, Field):
                cp = self.cp.data
            else:
                cp = np.full_like(self.T.data, self.cp)
            
            self.T.data += self.dt * (alpha * lap_T + self.Q.data / (rho * cp))
        else:
            self.T.data += self.dt * (alpha * lap_T + self.Q.data / (self.rho * self.cp))
        
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
        
        # 计算最小热扩散系数
        if isinstance(self.k, Field) or isinstance(self.rho, Field) or isinstance(self.cp, Field):
            if isinstance(self.k, Field):
                k = self.k.data
            else:
                k = np.full_like(self.T.data, self.k)
            
            if isinstance(self.rho, Field):
                rho = self.rho.data
            else:
                rho = np.full_like(self.T.data, self.rho)
            
            if isinstance(self.cp, Field):
                cp = self.cp.data
            else:
                cp = np.full_like(self.T.data, self.cp)
            
            alpha_min = np.min(k / (rho * cp))
        else:
            alpha_min = self.k / (self.rho * self.cp)
        
        # 三维热传导方程的CFL条件
        dt_max = 0.5 * min(dx**2, dy**2, dz**2) / alpha_min
        
        return dt_max    