import numpy as np

def gradient(scalar_field, mesh):
    """计算标量场的梯度
    
    参数:
        scalar_field: 标量场数据
        mesh: 网格对象
    
    返回:
        VectorField: 梯度矢量场
    """
    grad_x = np.zeros_like(scalar_field)
    grad_y = np.zeros_like(scalar_field)
    grad_z = np.zeros_like(scalar_field)
    
    # 内部点使用中心差分
    grad_x[1:-1, :, :] = (scalar_field[2:, :, :] - scalar_field[:-2, :, :]) / (2 * mesh.dx)
    grad_y[:, 1:-1, :] = (scalar_field[:, 2:, :] - scalar_field[:, :-2, :]) / (2 * mesh.dy)
    grad_z[:, :, 1:-1] = (scalar_field[:, :, 2:] - scalar_field[:, :, :-2]) / (2 * mesh.dz)
    
    # 边界点使用向前或向后差分
    grad_x[0, :, :] = (scalar_field[1, :, :] - scalar_field[0, :, :]) / mesh.dx
    grad_x[-1, :, :] = (scalar_field[-1, :, :] - scalar_field[-2, :, :]) / mesh.dx
    grad_y[:, 0, :] = (scalar_field[:, 1, :] - scalar_field[:, 0, :]) / mesh.dy
    grad_y[:, -1, :] = (scalar_field[:, -1, :] - scalar_field[:, -2, :]) / mesh.dy
    grad_z[:, :, 0] = (scalar_field[:, :, 1] - scalar_field[:, :, 0]) / mesh.dz
    grad_z[:, :, -1] = (scalar_field[:, :, -1] - scalar_field[:, :, -2]) / mesh.dz
    
    return grad_x, grad_y, grad_z

def divergence(vector_field, mesh):
    """计算矢量场的散度
    
    参数:
        vector_field: 矢量场对象
        mesh: 网格对象
    
    返回:
        numpy.ndarray: 散度标量场
    """
    vx = vector_field.x.data
    vy = vector_field.y.data
    vz = vector_field.z.data
    
    div = np.zeros_like(vx)
    
    # 内部点使用中心差分
    div[1:-1, 1:-1, 1:-1] = (vx[2:, 1:-1, 1:-1] - vx[:-2, 1:-1, 1:-1]) / (2 * mesh.dx) + \
                          (vy[1:-1, 2:, 1:-1] - vy[1:-1, :-2, 1:-1]) / (2 * mesh.dy) + \
                          (vz[1:-1, 1:-1, 2:] - vz[1:-1, 1:-1, :-2]) / (2 * mesh.dz)
    
    # 边界点使用向前或向后差分
    # x方向边界
    div[0, 1:-1, 1:-1] = (vx[1, 1:-1, 1:-1] - vx[0, 1:-1, 1:-1]) / mesh.dx + \
                       (vy[0, 2:, 1:-1] - vy[0, :-2, 1:-1]) / (2 * mesh.dy) + \
                       (vz[0, 1:-1, 2:] - vz[0, 1:-1, :-2]) / (2 * mesh.dz)
    div[-1, 1:-1, 1:-1] = (vx[-1, 1:-1, 1:-1] - vx[-2, 1:-1, 1:-1]) / mesh.dx + \
                        (vy[-1, 2:, 1:-1] - vy[-1, :-2, 1:-1]) / (2 * mesh.dy) + \
                        (vz[-1, 1:-1, 2:] - vz[-1, 1:-1, :-2]) / (2 * mesh.dz)
    
    # y方向边界
    div[1:-1, 0, 1:-1] = (vx[2:, 0, 1:-1] - vx[:-2, 0, 1:-1]) / (2 * mesh.dx) + \
                       (vy[1:-1, 1, 1:-1] - vy[1:-1, 0, 1:-1]) / mesh.dy + \
                       (vz[1:-1, 0, 2:] - vz[1:-1, 0, :-2]) / (2 * mesh.dz)
    div[1:-1, -1, 1:-1] = (vx[2:, -1, 1:-1] - vx[:-2, -1, 1:-1]) / (2 * mesh.dx) + \
                        (vy[1:-1, -1, 1:-1] - vy[1:-1, -2, 1:-1]) / mesh.dy + \
                        (vz[1:-1, -1, 2:] - vz[1:-1, -1, :-2]) / (2 * mesh.dz)
    
    # z方向边界
    div[1:-1, 1:-1, 0] = (vx[2:, 1:-1, 0] - vx[:-2, 1:-1, 0]) / (2 * mesh.dx) + \
                       (vy[1:-1, 2:, 0] - vy[1:-1, :-2, 0]) / (2 * mesh.dy) + \
                       (vz[1:-1, 1:-1, 1] - vz[1:-1, 1:-1, 0]) / mesh.dz
    div[1:-1, 1:-1, -1] = (vx[2:, 1:-1, -1] - vx[:-2, 1:-1, -1]) / (2 * mesh.dx) + \
                        (vy[1:-1, 2:, -1] - vy[1:-1, :-2, -1]) / (2 * mesh.dy) + \
                        (vz[1:-1, 1:-1, -1] - vz[1:-1, 1:-1, -2]) / mesh.dz
    
    # 角点处理
    # 这里简化处理，使用一阶近似
    div[0, 0, 0] = (vx[1, 0, 0] - vx[0, 0, 0]) / mesh.dx + \
                 (vy[0, 1, 0] - vy[0, 0, 0]) / mesh.dy + \
                 (vz[0, 0, 1] - vz[0, 0, 0]) / mesh.dz
    
    div[-1, 0, 0] = (vx[-1, 0, 0] - vx[-2, 0, 0]) / mesh.dx + \
                  (vy[-1, 1, 0] - vy[-1, 0, 0]) / mesh.dy + \
                  (vz[-1, 0, 1] - vz[-1, 0, 0]) / mesh.dz
    
    div[0, -1, 0] = (vx[1, -1, 0] - vx[0, -1, 0]) / mesh.dx + \
                  (vy[0, -1, 0] - vy[0, -2, 0]) / mesh.dy + \
                  (vz[0, -1, 1] - vz[0, -1, 0]) / mesh.dz
    
    div[0, 0, -1] = (vx[1, 0, -1] - vx[0, 0, -1]) / mesh.dx + \
                  (vy[0, 1, -1] - vy[0, 0, -1]) / mesh.dy + \
                  (vz[0, 0, -1] - vz[0, 0, -2]) / mesh.dz
    
    div[-1, -1, 0] = (vx[-1, -1, 0] - vx[-2, -1, 0]) / mesh.dx + \
                   (vy[-1, -1, 0] - vy[-1, -2, 0]) / mesh.dy + \
                   (vz[-1, -1, 1] - vz[-1, -1, 0]) / mesh.dz
    
    div[-1, 0, -1] = (vx[-1, 0, -1] - vx[-2, 0, -1]) / mesh.dx + \
                   (vy[-1, 1, -1] - vy[-1, 0, -1]) / mesh.dy + \
                   (vz[-1, 0, -1] - vz[-1, 0, -2]) / mesh.dz
    
    div[0, -1, -1] = (vx[1, -1, -1] - vx[0, -1, -1]) / mesh.dx + \
                   (vy[0, -1, -1] - vy[0, -2, -1]) / mesh.dy + \
                   (vz[0, -1, -1] - vz[0, -1, -2]) / mesh.dz
    
    div[-1, -1, -1] = (vx[-1, -1, -1] - vx[-2, -1, -1]) / mesh.dx + \
                    (vy[-1, -1, -1] - vy[-1, -2, -1]) / mesh.dy + \
                    (vz[-1, -1, -1] - vz[-1, -1, -2]) / mesh.dz
    
    return div

def curl(vector_field, mesh):
    """计算矢量场的旋度
    
    参数:
        vector_field: 矢量场对象
        mesh: 网格对象
    
    返回:
        VectorField: 旋度矢量场
    """
    vx = vector_field.x.data
    vy = vector_field.y.data
    vz = vector_field.z.data
    
    curl_x = np.zeros_like(vx)
    curl_y = np.zeros_like(vy)
    curl_z = np.zeros_like(vz)
    
    # 内部点使用中心差分
    curl_x[1:-1, 1:-1, 1:-1] = (vz[1:-1, 2:, 1:-1] - vz[1:-1, :-2, 1:-1]) / (2 * mesh.dy) - \
                             (vy[1:-1, 1:-1, 2:] - vy[1:-1, 1:-1, :-2]) / (2 * mesh.dz)
    
    curl_y[1:-1, 1:-1, 1:-1] = (vx[1:-1, 1:-1, 2:] - vx[1:-1, 1:-1, :-2]) / (2 * mesh.dz) - \
                             (vz[2:, 1:-1, 1:-1] - vz[:-2, 1:-1, 1:-1]) / (2 * mesh.dx)
    
    curl_z[1:-1, 1:-1, 1:-1] = (vy[2:, 1:-1, 1:-1] - vy[:-2, 1:-1, 1:-1]) / (2 * mesh.dx) - \
                             (vx[1:-1, 2:, 1:-1] - vx[1:-1, :-2, 1:-1]) / (2 * mesh.dy)
    
    # 边界点处理（简化处理，使用一阶近似）
    # 实际应用中可能需要更复杂的边界处理
    
    return curl_x, curl_y, curl_z

def laplacian(scalar_field, mesh):
    """计算标量场的拉普拉斯算子
    
    参数:
        scalar_field: 标量场数据
        mesh: 网格对象
    
    返回:
        numpy.ndarray: 拉普拉斯算子结果
    """
    lap = np.zeros_like(scalar_field)
    
    # 内部点使用中心差分
    lap[1:-1, 1:-1, 1:-1] = (scalar_field[2:, 1:-1, 1:-1] - 2 * scalar_field[1:-1, 1:-1, 1:-1] + scalar_field[:-2, 1:-1, 1:-1]) / (mesh.dx**2) + \
                           (scalar_field[1:-1, 2:, 1:-1] - 2 * scalar_field[1:-1, 1:-1, 1:-1] + scalar_field[1:-1, :-2, 1:-1]) / (mesh.dy**2) + \
                           (scalar_field[1:-1, 1:-1, 2:] - 2 * scalar_field[1:-1, 1:-1, 1:-1] + scalar_field[1:-1, 1:-1, :-2]) / (mesh.dz**2)
    
    # 边界点处理（简化处理，使用二阶近似）
    # x方向边界
    lap[0, 1:-1, 1:-1] = (2 * scalar_field[1, 1:-1, 1:-1] - 2 * scalar_field[0, 1:-1, 1:-1]) / (mesh.dx**2) + \
                       (scalar_field[0, 2:, 1:-1] - 2 * scalar_field[0, 1:-1, 1:-1] + scalar_field[0, :-2, 1:-1]) / (mesh.dy**2) + \
                       (scalar_field[0, 1:-1, 2:] - 2 * scalar_field[0, 1:-1, 1:-1] + scalar_field[0, 1:-1, :-2]) / (mesh.dz**2)
    
    lap[-1, 1:-1, 1:-1] = (2 * scalar_field[-2, 1:-1, 1:-1] - 2 * scalar_field[-1, 1:-1, 1:-1]) / (mesh.dx**2) + \
                        (scalar_field[-1, 2:, 1:-1] - 2 * scalar_field[-1, 1:-1, 1:-1] + scalar_field[-1, :-2, 1:-1]) / (mesh.dy**2) + \
                        (scalar_field[-1, 1:-1, 2:] - 2 * scalar_field[-1, 1:-1, 1:-1] + scalar_field[-1, 1:-1, :-2]) / (mesh.dz**2)
    
    # y方向边界
    lap[1:-1, 0, 1:-1] = (scalar_field[2:, 0, 1:-1] - 2 * scalar_field[1:-1, 0, 1:-1] + scalar_field[:-2, 0, 1:-1]) / (mesh.dx**2) + \
                       (2 * scalar_field[1:-1, 1, 1:-1] - 2 * scalar_field[1:-1, 0, 1:-1]) / (mesh.dy**2) + \
                       (scalar_field[1:-1, 0, 2:] - 2 * scalar_field[1:-1, 0, 1:-1] + scalar_field[1:-1, 0, :-2]) / (mesh.dz**2)
    
    lap[1:-1, -1, 1:-1] = (scalar_field[2:, -1, 1:-1] - 2 * scalar_field[1:-1, -1, 1:-1] + scalar_field[:-2, -1, 1:-1]) / (mesh.dx**2) + \
                        (2 * scalar_field[1:-1, -2, 1:-1] - 2 * scalar_field[1:-1, -1, 1:-1]) / (mesh.dy**2) + \
                        (scalar_field[1:-1, -1, 2:] - 2 * scalar_field[1:-1, -1, 1:-1] + scalar_field[1:-1, -1, :-2]) / (mesh.dz**2)
    
    # z方向边界
    lap[1:-1, 1:-1, 0] = (scalar_field[2:, 1:-1, 0] - 2 * scalar_field[1:-1, 1:-1, 0] + scalar_field[:-2, 1:-1, 0]) / (mesh.dx**2) + \
                       (scalar_field[1:-1, 2:, 0] - 2 * scalar_field[1:-1, 1:-1, 0] + scalar_field[1:-1, :-2, 0]) / (mesh.dy**2) + \
                       (2 * scalar_field[1:-1, 1:-1, 1] - 2 * scalar_field[1:-1, 1:-1, 0]) / (mesh.dz**2)
    
    lap[1:-1, 1:-1, -1] = (scalar_field[2:, 1:-1, -1] - 2 * scalar_field[1:-1, 1:-1, -1] + scalar_field[:-2, 1:-1, -1]) / (mesh.dx**2) + \
                        (scalar_field[1:-1, 2:, -1] - 2 * scalar_field[1:-1, 1:-1, -1] + scalar_field[1:-1, :-2, -1]) / (mesh.dy**2) + \
                        (2 * scalar_field[1:-1, 1:-1, -2] - 2 * scalar_field[1:-1, 1:-1, -1]) / (mesh.dz**2)
    
    # 角点处理（简化处理）
    lap[0, 0, 0] = (2 * scalar_field[1, 0, 0] - 2 * scalar_field[0, 0, 0]) / (mesh.dx**2) + \
                 (2 * scalar_field[0, 1, 0] - 2 * scalar_field[0, 0, 0]) / (mesh.dy**2) + \
                 (2 * scalar_field[0, 0, 1] - 2 * scalar_field[0, 0, 0]) / (mesh.dz**2)
    
    # 其他角点类似处理...
    
    return lap    