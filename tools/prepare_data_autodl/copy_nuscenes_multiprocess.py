import os
import shutil
import tarfile
from multiprocessing import Pool
"""利用进程池实现多进程
        代码有问题，别用"""
def prepare_file(source_file):
    target_dir = '/root/autodl-tmp/nuscenes'  # 替换为您的目标目录路径
    
    # 检查是否是文件，避免目录
    # if os.path.isfile(source_file):
    #     # 复制文件到目标目录
    #     shutil.copy(source_file, target_dir)
    #     print(f'复制文件: {source_file} 到 {target_dir}')
    # src_tgz=os.path.join(target_dir,source_file.split('/')[-1])
    src_tgz=source_file
    try:
        with tarfile.open(src_tgz, 'r:gz') as tar:
            tar.extractall(path=target_dir)  # 默认解压到当前工作目录
            print(f"成功解压 {src_tgz}")
    except Exception as e:
        print(f"解压 {src_tgz} 时出错: {e}")
        return
    
    # 删除源文件
    # try:
    #     os.remove(src_tgz)
    #     print(f"成功删除源文件 {src_tgz}")
    # except Exception as e:
    #     print(f"删除文件 {src_tgz} 时出错: {e}")


if __name__=="__main__":
    # 示例使用
    source_directory = '/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval'  # 替换为您的源目录路径
    target_directory = '/root/autodl-tmp/nuscenes'  # 替换为您的目标目录路径
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    tar_files=os.listdir(source_directory)
    with Pool(processes=6) as pool: 
        pool.map(prepare_file,tar_files)    
