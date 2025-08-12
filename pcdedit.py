import open3d
import numpy as np

# 創建解析器並添加參數
# 加載PCD文件
pcd = open3d.io.read_point_cloud("MAP1FV.pcd")

# 顯示PCD
open3d.visualization.draw_geometries([pcd])
#計算Z的平均值
#地板設為0

print(len(pcd.points))
print(type(pcd.points))
# 將點雲資料轉換為NumPy數組
points_np = np.asarray(pcd.points)
points_np = points_np[::3]
print(points_np)
print("Z平均值:", np.mean(points_np[:, 2]))

# 將地圖區分成每0.1*0.1的小區域
grid_size = 0.5

x_min, x_max = np.min(points_np[:, 0]), np.max(points_np[:, 0])
y_min, y_max = np.min(points_np[:, 1]), np.max(points_np[:, 1])
x_grid = np.arange(x_min, x_max, grid_size)
y_grid = np.arange(y_min, y_max, grid_size)

print("x_grid:", x_grid)
print("y_grid:", y_grid)    
print("x_grid size:", len(x_grid))
print("y_grid size:", len(y_grid))  

# 計算每個小區域的最小Z值並將該區域的Z值都減去最小Z值
for i in range(len(x_grid) - 1):
    for j in range(len(y_grid) - 1):
        x_start, x_end = x_grid[i], x_grid[i + 1]
        y_start, y_end = y_grid[j], y_grid[j + 1]
        indices = np.where((points_np[:, 0] >= x_start) & (points_np[:, 0] < x_end) &
                          (points_np[:, 1] >= y_start) & (points_np[:, 1] < y_end))
        if len(indices[0]) > 0:
            
            min_z = np.min(points_np[indices, 2])
            print(len(x_grid) ,":",i,",",j,",MinZ=",min_z,len(indices[0]))
            points_np[indices, 2] -= min_z
            # 保存修改後的PCD文件
            # 將修改後的NumPy數組轉換回點雲資料
points_np = points_np[points_np[:, 2] <= 3]
points_np = points_np[points_np[:, 2] >= 0]

x_min, x_max = np.min(points_np[:, 0]), np.max(points_np[:, 0])
y_min, y_max = np.min(points_np[:, 1]), np.max(points_np[:, 1])
z_min, z_max = np.min(points_np[:, 2]), np.max(points_np[:, 2])

print("X最小值:", x_min, "X最大值:", x_max)
print("Y最小值:", y_min, "Y最大值:", y_max)
print("Z最小值:", z_min, "Z最大值:", z_max)

pcd.points = open3d.utility.Vector3dVector(points_np)
open3d.io.write_point_cloud("modified_pcd_T.pcd", pcd)
            # 顯示修改後的PCD
open3d.visualization.draw_geometries([pcd])
