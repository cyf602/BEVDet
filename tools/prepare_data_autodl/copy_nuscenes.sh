#!/bin/bash
source_path=/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval
target_path=/root/autodl-tmp/nuscenes
###用于autodl上nuscenes trainval数据集在数据盘的复制及解压
if [[ ! -d "$target_path" ]]; then
    mkdir -p "$target_path"
fi
# 准备一个函数用于解压缩
unzip_file() {
    local file=$1
    # local output_dir="${file%.tgz}"  # 去掉 .tgz 后缀作为目录名
    local output_dir=$2  # 去掉 .tgz 后缀作为目录名
     
    # 复制过去
    # rsync -avP "$file" 

    # 解压缩文件
    tar -xzvf "$file" -C "$output_dir"

    echo "$file 解压缩完成到 $output_dir"
}

# 查找当前目录下的所有 .tgz 文件并进行解压
for file in $source_path/*.tgz; do
    echo "$file 处理中"
    if [[ -f "$file" ]]; then  # 确保是文件
        unzip_file "$file" "$target_path" &  # 后台执行解压函数
    fi
done

# 等待所有后台进程完成
wait
for file in $source_path/*.tgz; do
    echo "$file"
done
echo "所有文件解压完成！"
