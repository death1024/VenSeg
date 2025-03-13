import os

def rename_nii_files(directory):
    for filename in os.listdir(directory):
        # 处理后缀是 .nii.gz 的文件
        if filename.endswith('.nii.gz'):
            new_filename = filename.replace('.nii.gz', '_mask_crop.nii.gz')
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f'已重命名: {filename} -> {new_filename}')

if __name__ == '__main__':
    target_directory = input("路径：").strip() or "."
    rename_nii_files(target_directory)
