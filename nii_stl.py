import os
import nibabel as nib
import numpy as np
from stl import mesh
from skimage import measure
from scipy import ndimage

def convert_ras_to_lps(input_file):
    """转换到LPS坐标系"""
    nifti_image = nib.load(input_file)
    affine = nifti_image.affine

    # 转换 affine 矩阵：反转 X 和 Y 轴
    affine_lps = affine.copy()
    affine_lps[0, :] = -affine_lps[0, :]
    affine_lps[1, :] = -affine_lps[1, :]

    # print(f"转换后的仿射矩阵:\n{affine_lps}")
    return nifti_image.get_fdata(dtype=np.float64), affine_lps


def preprocess_segmentation(array, class_value, morphology_operations=True):
    """
    预处理分割数据，改善模型封闭性
    """
    # 创建二值掩码
    binary_mask = (np.isclose(array, class_value)).astype(np.uint8)

    if not morphology_operations:
        return binary_mask

    print(f"预处理前体素数: {np.sum(binary_mask)}")

    # 1. 先进行闭操作填充小孔洞
    structure = ndimage.generate_binary_structure(3, 2)  # 3D结构元素
    closed_mask = ndimage.binary_closing(binary_mask, structure=structure, iterations=1)

    # 2. 填充内部孔洞
    filled_mask = ndimage.binary_fill_holes(closed_mask)

    # 3. 去除小的孤立区域
    labeled_mask, num_features = ndimage.label(filled_mask)
    sizes = ndimage.sum(filled_mask, labeled_mask, range(num_features + 1))

    # 只保留最大的连通区域
    if num_features > 1:
        largest_component = np.argmax(sizes[1:]) + 1
        filled_mask = (labeled_mask == largest_component)

    processed_mask = filled_mask.astype(np.uint8)
    print(f"预处理后体素数: {np.sum(processed_mask)}")

    return processed_mask

def nii_to_stl(input_file, output_file_path, name_dict):
    """改进的NIfTI到STL转换函数"""

    # 获取数据和转换后的affine
    array, affine_lps = convert_ras_to_lps(input_file)

    # 获取体素间距
    nifti_image = nib.load(input_file)
    spacing = nifti_image.header.get_zooms()[:3]
    print(f"体素间距: {spacing}")

    # 获取数据中的类别
    input_file_classes = list(np.unique(array))
    if 0 in input_file_classes:
        input_file_classes.remove(0)  # 移除背景

    successful_conversions = 0

    for input_file_class in input_file_classes:
        # 获取类别的名字
        input_file_class_name = name_dict.get(str(int(input_file_class)), "Unknown")
        print(f"\n处理类别: {input_file_class_name} (标签值: {input_file_class})")

        # 预处理分割数据
        array_copy = preprocess_segmentation(array, input_file_class, morphology_operations=True)

        voxel_count = np.sum(array_copy)
        print(f"  有效体素数: {voxel_count}")

        if voxel_count < 50:
            print(f"体素数量不足，跳过 {input_file_class_name}")
            continue

        try:
            verts, faces, _, _ = measure.marching_cubes(array_copy, level=0.5, method='lewiner', allow_degenerate=False)

            # 确保面为三角形
            if faces.shape[1] != 3:
                faces = faces[:, :3]

            # 将顶点从体素坐标转换为世界坐标
            verts_world = nib.affines.apply_affine(affine_lps, verts)

            # 创建 STL 对象
            obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                obj_3d.vectors[i] = verts_world[f]

            # 保存 STL 文件
            output_file_class_path = os.path.join(output_file_path, f"{input_file_class_name}.stl")
            obj_3d.save(output_file_class_path)
            print(f"  ✓ 保存 {input_file_class_name}.stl: {len(verts)}顶点, {len(faces)}面片")

            successful_conversions += 1

        except Exception as e:
            print(f"  ✗ 处理 {input_file_class_name} 时出错: {e}")
            continue

    print(f"\n成功转换 {successful_conversions}/{len(input_file_classes)} 个类别")
    return successful_conversions > 0

def main():
    input_files_path = r"E:\code\nii_to_stl\data\nii"  # 输入 NIfTI 文件的路径
    output_files_path = r"E:\code\nii_to_stl\data\target\stl"  # 输出 STL 文件的路径

    # 定义标签对应的名称字典
    name_dict = {
        "1": "C1", "2": "C2", "3": "C3", "4": "C4", "5": "C5",
        "6": "C6", "7": "C7", "8": "T1", "9": "T2", "10": "T3",
        "11": "T4", "12": "T5", "13": "T6", "14": "T7", "15": "T8",
        "16": "T9", "17": "T10", "18": "T11", "19": "T12", "20": "L1",
        "21": "L2", "22": "L3", "23": "L4", "24": "L5", "25": "L6"
    }

    # 获取所有输入的文件
    input_files = [f for f in os.listdir(input_files_path) if f.endswith(('.nii', '.nii.gz'))]

    if not input_files:
        print("在输入文件夹中没有找到NIfTI文件")
        return

    total_successful_files = 0

    for i, input_file in enumerate(input_files):
        input_file_path = os.path.join(input_files_path, input_file)
        print(f"\n{'=' * 60}")
        print(f"处理 {i + 1}/{len(input_files)}: {input_file}")
        print(f"{'=' * 60}")

        # 创建输出文件夹
        output_file_class_path = os.path.join(output_files_path, input_file.split('.')[0])
        if not os.path.exists(output_file_class_path):
            os.makedirs(output_file_class_path)

        # 转换并保存为 STL
        success = nii_to_stl(input_file_path, output_file_class_path, name_dict)

        if success:
            total_successful_files += 1

    print(f"\n{'=' * 60}")
    print(f"处理完成! 成功处理 {total_successful_files}/{len(input_files)} 个文件")
    print(f"STL文件保存在: {output_files_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()