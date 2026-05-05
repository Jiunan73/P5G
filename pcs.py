import sys
import os
import argparse
import numpy as np
import open3d as o3d

# pcs.py
# GitHub Copilot
# Usage: python pcs.py input.pcd [output.pcd]
# 讀取 PCD，顯示原始點雲，縮減為 1/5（保留 20% 點），儲存新檔並顯示


def load_pcd(path):
    if not os.path.isfile(path):
        print("找不到檔案:", path)
        sys.exit(1)
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        print("載入的點雲為空:", path)
        sys.exit(1)
    return pcd

def random_downsample(pcd, keep_fraction=0.2, seed=0):
    pts = np.asarray(pcd.points)
    n = pts.shape[0]
    k = max(1, int(round(n * keep_fraction)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    down = o3d.geometry.PointCloud()
    down.points = o3d.utility.Vector3dVector(pts[idx])
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)[idx]
        down.colors = o3d.utility.Vector3dVector(colors)
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)[idx]
        down.normals = o3d.utility.Vector3dVector(normals)
    return down


if __name__ == "__main__":
    if __name__ == "__main__":
        # 固定輸入與輸出檔名
        inp = "New.pcd"
        out = "New1.pcd"

        pcd = load_pcd(inp)
        print(f"原始點數: {len(pcd.points)}")
        o3d.visualization.draw_geometries([pcd], window_name="原始 PCD")

        down = random_downsample(pcd, keep_fraction=0.2, seed=42)
        print(f"縮減後點數: {len(down.points)}")
        ok = o3d.io.write_point_cloud(out, down)
        if not ok:
            print("儲存失敗:", out)
            sys.exit(1)
        print("已儲存:", out)
        o3d.visualization.draw_geometries([down], window_name="縮減後 PCD")