import os

# 修改为你保存 splits 的路径
base_path = "E:/balltracker/ball_tracker/monodepth2/monodepth2/splits/splits/custom"

def dedup_file(filename):
    infile = os.path.join(base_path, filename)
    outfile = os.path.join(base_path, filename)
    seen = set()
    lines_out = []
    with open(infile, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line not in seen:  # 保留第一次出现的
                seen.add(line)
                lines_out.append(line)
    with open(outfile, "w") as f:
        for l in lines_out:
            f.write(l + "\n")
    print(f"[Done] {filename}: 去重前 {len(lines_out)+len(seen)-len(seen)} 条，去重后 {len(lines_out)} 条")

for split_file in ["train_files.txt", "val_files.txt", "test_files.txt"]:
    path = os.path.join(base_path, split_file)
    if os.path.exists(path):
        dedup_file(split_file)
    else:
        print(f"[Skip] {split_file} 不存在")
