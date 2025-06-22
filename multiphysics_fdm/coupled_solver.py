import numpy as np
from .em_solver import EMSolver
from .thermal_solver import ThermalSolver
from .fluid_solver import FluidSolver

class CoupledSolver:
    """多物理场耦合求解器"""
    
    def __init__(self, mesh):
        """
        初始化多物理场耦合求解器
        
        参数:
            mesh: 网格对象
        """
        self.mesh = mesh
        
        # 物理场求解器
        self.em_solver = None  # 电磁场求解器
        self.thermal_solver = None  # 热传导求解器
        self.fluid_solver = None  # 流体求解器
        
        # 时间步长和当前时间
        self.dt = 0.0
        self.time = 0.0
        
        # 耦合系数
        self.em_thermal_coupling = False  # 电磁场-热传导耦合
        self.thermal_fluid_coupling = False  # 热传导-流体耦合
        self.em_fluid_coupling = False  # 电磁场-流体耦合
        
        # 材料属性
        self.thermal_conductivity = 1.0  # 热导率
        self.electrical_conductivity = 1.0  # 电导率
        self.seebeck_coefficient = 0.0  # 塞贝克系数 (热电效应)
        self.joule_heating_factor = 1.0  # 焦耳热因子
        self.thermal_expansion_coefficient = 0.0  # 热膨胀系数
    
    def set_em_solver(self, epsilon=1.0, mu=1.0, sigma=1.0):
        """
        设置电磁场求解器
        
        参数:
            epsilon: 介电常数
            mu: 磁导率
            sigma: 电导率
        """
        self.em_solver = EMSolver(self.mesh, epsilon, mu, sigma)
        self.electrical_conductivity = sigma
    
    def set_thermal_solver(self, k=1.0, rho=1.0, cp=1.0):
        """
        设置热传导求解器
        
        参数:
            k: 热导率
            rho: 密度
            cp: 比热容
        """
        self.thermal_solver = ThermalSolver(self.mesh, k, rho, cp)
        self.thermal_conductivity = k
    
    def set_fluid_solver(self, rho=1.0, mu=1.0, gravity=(0.0, 0.0, 0.0)):
        """
        设置流体求解器
        
        参数:
            rho: 流体密度
            mu: 流体动力粘度
            gravity: 重力加速度
        """
        self.fluid_solver = FluidSolver(self.mesh, rho, mu, gravity)
    
    def enable_em_thermal_coupling(self, joule_heating_factor=1.0, seebeck_coefficient=0.0):
        """
        启用电磁场-热传导耦合
        
        参数:
            joule_heating_factor: 焦耳热因子
            seebeck_coefficient: 塞贝克系数 (热电效应)
        """
        self.em_thermal_coupling = True
        self.joule_heating_factor = joule_heating_factor
        self.seebeck_coefficient = seebeck_coefficient
    
    def enable_thermal_fluid_coupling(self, thermal_expansion_coefficient=0.0):
        """
        启用热传导-流体耦合
        
        参数:
            thermal_expansion_coefficient: 热膨胀系数
        """
        self.thermal_fluid_coupling = True
        self.thermal_expansion_coefficient = thermal_expansion_coefficient
    
    def enable_em_fluid_coupling(self):
        """启用电磁场-流体耦合"""
        self.em_fluid_coupling = True
    
    def set_time_step(self, dt):
        """设置时间步长"""
        self.dt = dt
        if self.em_solver:
            self.em_solver.set_time_step(dt)
        if self.thermal_solver:
            self.thermal_solver.set_time_step(dt)
        if self.fluid_solver:
            self.fluid_solver.set_time_step(dt)
    
    def solve_step(self):
        """执行一个时间步的耦合求解"""
        if self.dt == 0.0:
            raise ValueError("时间步长未设置，请先调用set_time_step方法")
        
        # 1. 电磁场求解
        if self.em_solver:
            self.em_solver.solve_step()
        
        # 2. 热传导求解 (考虑电磁场的影响)
        if self.thermal_solver:
            # 添加焦耳热作为热源
            if self.em_solver and self.em_thermal_coupling:
                def joule_heating(x, y, z, T):
                    # 找到当前位置对应的网格索引
                    i = np.abs(self.mesh.x - x).argmin()
                    j = np.abs(self.mesh.y - y).argmin()
                    k = np.abs(self.mesh.z - z).argmin()
                    
                    # 计算焦耳热: Q = σ|E|²
                    E_x = self.em_solver.E.x.data[i, j, k]
                    E_y = self.em_solver.E.y.data[i, j, k]
                    E_z = self.em_solver.E.z.data[i, j, k]
                    
                    if isinstance(self.electrical_conductivity, Field):
                        sigma = self.electrical_conductivity.data[i, j, k]
                    else:
                        sigma = self.electrical_conductivity
                    
                    return self.joule_heating_factor * sigma * (E_x**2 + E_y**2 + E_z**2)
                
                self.thermal_solver.add_heat_source(joule_heating)
            
            self.thermal_solver.solve_step()
        
        # 3. 流体求解 (考虑热传导和电磁场的影响)
        if self.fluid_solver:
            # 添加热浮力
            if self.thermal_solver and self.thermal_fluid_coupling:
                def thermal_buoyancy(x, y, z, u, v, w):
                    # 找到当前位置对应的网格索引
                    i = np.abs(self.mesh.x - x).argmin()
                    j = np.abs(self.mesh.y - y).argmin()
                    k = np.abs(self.mesh.z - z).argmin()
                    
                    # Boussinesq近似: F = -ρ₀β(T-T₀)g
                    T = self.thermal_solver.T.data[i, j, k]
                    T_ref = 300.0  # 参考温度
                    
                    if isinstance(self.fluid_solver.rho, Field):
                        rho = self.fluid_solver.rho.data[i, j, k]
                    else:
                        rho = self.fluid_solver.rho
                    
                    g_x, g_y, g_z = self.fluid_solver.gravity
                    
                    return (
                        -rho * self.thermal_expansion_coefficient * (T - T_ref) * g_x,
                        -rho * self.thermal_expansion_coefficient * (T - T_ref) * g_y,
                        -rho * self.thermal_expansion_coefficient * (T - T_ref) * g_z
                    )
                
                self.fluid_solver.add_body_force(thermal_buoyancy)
            
            # 添加洛伦兹力
            if self.em_solver and self.em_fluid_coupling:
                def lorentz_force(x, y, z, u, v, w):
                    # 找到当前位置对应的网格索引
                    i = np.abs(self.mesh.x - x).argmin()
                    j = np.abs(self.mesh.y - y).argmin()
                    k = np.abs(self.mesh.z - z).argmin()
                    
                    # 洛伦兹力: F = J × B
                    J_x = self.em_solver.J.x.data[i, j, k]
                    J_y = self.em_solver.J.y.data[i, j, k]
                    J_z = self.em_solver.J.z.data[i, j, k]
                    
                    B_x = self.em_solver.B.x.data[i, j, k]
                    B_y = self.em_solver.B.y.data[i, j, k]
                    B_z = self.em_solver.B.z.data[i, j, k]
                    
                    return (
                        J_y * B_z - J_z * B_y,
                        J_z * B_x - J_x * B_z,
                        J_x * B_y - J_y * B_x
                    )
                
                self.fluid_solver.add_body_force(lorentz_force)
            
            self.fluid_solver.solve_step()
        
        # 更新时间
        self.time += self.dt
    
    def get_max_stable_time_step(self):
        """
        计算最大稳定时间步长 (基于所有物理场的CFL条件)
        
        返回:
            float: 最大稳定时间步长
        """
        dt_max = float('inf')
        
        if self.em_solver:
            dt_em = self.em_solver.get_max_stable_time_step()
            dt_max = min(dt_max, dt_em)
        
        if self.thermal_solver:
            dt_thermal = self.thermal_solver.get_max_stable_time_step()
            dt_max = min(dt_max, dt_thermal)
        
        if self.fluid_solver:
            dt_fluid = self.fluid_solver.get_max_stable_time_step()
            dt_max = min(dt_max, dt_fluid)
        
        return dt_max    