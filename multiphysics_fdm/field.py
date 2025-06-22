import numpy as np

class Field:
    """三维标量场类"""
    
    def __init__(self, mesh, name="field"):
        """
        初始化标量场
        
        参数:
            mesh: 网格对象
            name: 场的名称
        """
        self.mesh = mesh
        self.name = name
        self.data = np.zeros((mesh.nx, mesh.ny, mesh.nz))
    
    def set_value(self, value):
        """将整个场设置为指定值"""
        self.data[:] = value
    
    def set_from_function(self, func):
        """根据函数设置场值
        
        参数:
            func: 函数，接受x, y, z坐标作为参数，返回标量值
        """
        for i in range(self.mesh.nx):
            for j in range(self.mesh.ny):
                for k in range(self.mesh.nz):
                    x = self.mesh.x[i]
                    y = self.mesh.y[j]
                    z = self.mesh.z[k]
                    self.data[i, j, k] = func(x, y, z)


class VectorField:
    """三维矢量场类"""
    
    def __init__(self, mesh, name="vector_field"):
        """
        初始化矢量场
        
        参数:
            mesh: 网格对象
            name: 场的名称
        """
        self.mesh = mesh
        self.name = name
        # 三个分量分别表示x, y, z方向的场
        self.x = Field(mesh, f"{name}_x")
        self.y = Field(mesh, f"{name}_y")
        self.z = Field(mesh, f"{name}_z")
    
    def set_value(self, value_x, value_y, value_z):
        """将整个矢量场设置为指定值"""
        self.x.set_value(value_x)
        self.y.set_value(value_y)
        self.z.set_value(value_z)
    
    def set_from_function(self, func):
        """根据函数设置矢量场值
        
        参数:
            func: 函数，接受x, y, z坐标作为参数，返回(x, y, z)分量的元组
        """
        for i in range(self.mesh.nx):
            for j in range(self.mesh.ny):
                for k in range(self.mesh.nz):
                    x = self.mesh.x[i]
                    y = self.mesh.y[j]
                    z = self.mesh.z[k]
                    vx, vy, vz = func(x, y, z)
                    self.x.data[i, j, k] = vx
                    self.y.data[i, j, k] = vy
                    self.z.data[i, j, k] = vz    