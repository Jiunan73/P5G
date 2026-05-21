
#!/usr/bin/env python3
"""
簡單的 PCD 讀取並顯示程式（使用 open3d）

用法：
	python pcd_cut.py path/to/file.pcd

需求：
	pip install open3d
"""
import sys
import argparse
import numpy as np

def main():
	parser = argparse.ArgumentParser(description="Read, cut by fixed Z range and display a PCD file using Open3D")
	parser.add_argument('pcd', help='Path to the input PCD file')
	#args = parser.parse_args()

	try:
		import open3d as o3d
	except Exception:
		print('需要安裝 open3d: pip install open3d', file=sys.stderr)
		sys.exit(1)

	pcd_path = 'new.pcd'
	pcd = o3d.io.read_point_cloud(pcd_path)
	if pcd.is_empty():
		print(f'讀取失敗或點雲為空: {pcd_path}', file=sys.stderr)
		sys.exit(1)

	pts = np.asarray(pcd.points)
	if pts.size == 0:
		print('點陣列為空', file=sys.stderr)
		sys.exit(1)

	z = pts[:, 2]
	# Fixed cut: remove points with z < 0 or z > 1.5
	zmin = -0.6
	zmax = 1.5
	mask = (z >= zmin) & (z <= zmax)
	kept = int(np.count_nonzero(mask))
	total = int(pts.shape[0])

	new_pcd = o3d.geometry.PointCloud()
	new_pcd.points = o3d.utility.Vector3dVector(pts[mask])

	if pcd.has_colors():
		cols = np.asarray(pcd.colors)
		new_pcd.colors = o3d.utility.Vector3dVector(cols[mask])

	if pcd.has_normals():
		nml = np.asarray(pcd.normals)
		new_pcd.normals = o3d.utility.Vector3dVector(nml[mask])

	out_path = 'new_cut.pcd'
	ok = o3d.io.write_point_cloud(out_path, new_pcd)
	if not ok:
		print(f'寫入失敗: {out_path}', file=sys.stderr)
		sys.exit(1)
	print(f'輸入: {pcd_path}  總點數: {total}')
	print(f'輸出: {out_path}  保留點數: {kept}  範圍: [{zmin}, {zmax}]')

	o3d.visualization.draw_geometries([new_pcd], window_name=f'PCD: {out_path}')

if __name__ == '__main__':
	main()
