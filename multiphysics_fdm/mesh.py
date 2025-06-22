import numpy as np

class Mesh:
    """三维笛卡尔网格类"""
    
    def __init__(self, x_min, x_max, nx, y_min, y_max, ny, z_min, z_max, nz):
        """
        初始化网格
        
        参数:
            x_min, x_max: x方向的最小和最大坐标
            nx: x方向的网格点数
            y_min, y_max: y方向的最小和最大坐标
            ny: y方向的网格点数
            z_min, z_max: z方向的最小和最大坐标
            nz: z方向的网格点数
        """
        self.x_min = x_min
        self.x_max = x_max
        self.nx = nx
        self.y_min = y_min
        self.y_max = y_max
        self.ny = ny
        self.z_min = z_min
        self.z_max = z_max
        self.nz = nz
        
        # 计算网格间距
        self.dx = (x_max - x_min) / (nx - 1)
        self.dy = (y_max - y_min) / (ny - 1)
        self.dz = (z_max - z_min) / (nz - 1)
        
        # 生成网格坐标
        self.x = np.linspace(x_min, x_max, nx)
        self.y = np.linspace(y_min, y_max, ny)
        self.z = np.linspace(z_min, z_max, nz)
        
        # 创建网格点的三维数组
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
    
    def get_cell_volume(self):
        """返回单个网格单元的体积"""
        return self.dx * self.dy * self.dz
    
    def get_total_cells(self):
        """返回网格单元总数"""
        return (self.nx - 1) * (self.ny - 1) * (self.nz - 1)    