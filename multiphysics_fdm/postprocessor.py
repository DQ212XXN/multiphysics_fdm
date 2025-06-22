import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Postprocessor:
    """多物理场后处理器"""
    
    def __init__(self, mesh):
        """
        初始化后处理器
        
        参数:
            mesh: 网格对象
        """
        self.mesh = mesh
    
    def plot_scalar_field(self, field, title="Scalar Field", save_path=None):
        """
        绘制标量场
        
        参数:
            field: 标量场数据
            title: 图表标题
            save_path: 保存路径，若为None则显示图表
        """
        # 提取网格坐标
        X, Y, Z = self.mesh.X, self.mesh.Y, self.mesh.Z
        
        # 绘制2D切片 (z方向的中间平面)
        k_mid = self.mesh.nz // 2
        
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X[:, :, k_mid], Y[:, :, k_mid], field[:, :, k_mid], 50, cmap='jet')
        plt.colorbar(label='Field Value')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_vector_field(self, vector_field, title="Vector Field", save_path=None, scale=100, width=0.002):
        """
        绘制矢量场
        
        参数:
            vector_field: 矢量场对象
            title: 图表标题
            save_path: 保存路径，若为None则显示图表
            scale: 箭头缩放比例
            width: 箭头宽度
        """
        # 提取网格坐标
        X, Y, Z = self.mesh.X, self.mesh.Y, self.mesh.Z
        
        # 提取矢量场分量
        u = vector_field.x.data
        v = vector_field.y.data
        w = vector_field.z.data
        
        # 绘制2D切片 (z方向的中间平面)
        k_mid = self.mesh.nz // 2
        
        plt.figure(figsize=(10, 8))
        plt.quiver(X[:, :, k_mid], Y[:, :, k_mid], u[:, :, k_mid], v[:, :, k_mid], 
                  scale=scale, width=width, color='blue')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def calculate_field_statistics(self, field, name="Field"):
        """
        计算场的统计数据
        
        参数:
            field: 场数据
            name: 场名称
        
        返回:
            dict: 包含统计数据的字典
        """
        stats = {
            'name': name,
            'min': np.min(field),
            'max': np.max(field),
            'mean': np.mean(field),
            'std': np.std(field)
        }
        
        return stats
    
    def export_to_vtk(self, fields, filename="output.vtk"):
        """
        将场数据导出到VTK格式文件
        
        参数:
            fields: 字典，包含要导出的场 {'field_name': field_data}
            filename: 输出文件名
        """
        with open(filename, 'w') as f:
            # 写入VTK文件头
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Multiphysics Simulation Results\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_GRID\n")
            
            # 写入网格信息
            f.write(f"DIMENSIONS {self.mesh.nx} {self.mesh.ny} {self.mesh.nz}\n")
            
            # 写入点坐标
            f.write(f"POINTS {self.mesh.nx*self.mesh.ny*self.mesh.nz} float\n")
            for k in range(self.mesh.nz):
                for j in range(self.mesh.ny):
                    for i in range(self.mesh.nx):
                        x = self.mesh.x[i]
                        y = self.mesh.y[j]
                        z = self.mesh.z[k]
                        f.write(f"{x} {y} {z}\n")
            
            # 写入点数据
            f.write(f"POINT_DATA {self.mesh.nx*self.mesh.ny*self.mesh.nz}\n")
            
            # 写入每个场
            for field_name, field_data in fields.items():
                if isinstance(field_data, np.ndarray) and field_data.ndim == 3:
                    # 标量场
                    f.write(f"SCALARS {field_name} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for k in range(self.mesh.nz):
                        for j in range(self.mesh.ny):
                            for i in range(self.mesh.nx):
                                f.write(f"{field_data[i, j, k]}\n")
                elif hasattr(field_data, 'x') and hasattr(field_data, 'y') and hasattr(field_data, 'z'):
                    # 矢量场
                    f.write(f"VECTORS {field_name} float\n")
                    for k in range(self.mesh.nz):
                        for j in range(self.mesh.ny):
                            for i in range(self.mesh.nx):
                                vx = field_data.x.data[i, j, k]
                                vy = field_data.y.data[i, j, k]
                                vz = field_data.z.data[i, j, k]
                                f.write(f"{vx} {vy} {vz}\n")    