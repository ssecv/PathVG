import json
import torch
import argparse
import os
from typing import List, Tuple

def pad_strings_to_length(lst: List[str], target_length: int = 3) -> List[str]:
    """
    将列表填充到指定长度，不足部分用空字符串补充
    Pad the list to the specified length, fill the insufficient part with empty strings
    Args:
        lst: 待填充的字符串列表 | List of strings to be padded
        target_length: 目标长度，默认3 | Target length, default is 3
    Returns:
        填充后的列表 | Padded list
    """
    padded_lst = lst.copy()
    while len(padded_lst) < target_length:
        padded_lst.append("")
    return padded_lst

def process_jsonl(input_jsonl_path: str, output_pth_path: str, image_base_path: str = ""):
    """
    将JSONL文件转换为训练用的.pth文件
    Convert JSONL file to .pth file for training
    Args:
        input_jsonl_path: 输入的JSONL文件路径 | Path to input JSONL file
        output_pth_path: 输出的.pth文件路径 | Path to output .pth file
        image_base_path: 图片文件的基础路径（用于拼接完整图片路径，可选）| Base path of image files (used to splice full image path, optional)
    """
    # 校验输入文件是否存在 | Check if the input file exists
    if not os.path.exists(input_jsonl_path):
        raise FileNotFoundError(f"输入的JSONL文件不存在: {input_jsonl_path} | Input JSONL file does not exist: {input_jsonl_path}")
    
    dataset_train = []
    # 读取JSONL文件（逐行解析）| Read JSONL file (parse line by line)
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行 | Skip empty lines
                continue
            try:
                # 解析单行JSON | Parse single line JSON
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON格式错误，已跳过 | 错误信息: {e} | Warning: Line {line_num} has invalid JSON format, skipped | Error: {e}")
                continue

            # 提取字段（适配指定的JSONL行结构）| Extract fields (adapt to the specified JSONL line structure)
            image_id = entry.get('image_id')  # 图片文件名，如10058.jpg | Image file name, e.g., 10058.jpg
            image_rel_path = entry.get('image')  # 图片相对路径，如refpath_image/10058.jpg | Relative image path, e.g., refpath_image/10058.jpg
            bbox = entry.get('bbox')  # 边界框坐标 | Bounding box coordinates
            expression = entry.get('expression')  # 描述表达式 | Description expression
            bbox_id = entry.get('bbox_id')  # 可选字段，保留备用 | Optional field, reserved for later use
            height = entry.get('height')    # 可选字段，保留备用 | Optional field, reserved for later use
            width = entry.get('width')      # 可选字段，保留备用 | Optional field, reserved for later use

            # 字段非空校验 | Check for non-empty fields
            if any(v is None for v in [image_rel_path, bbox, expression]):
                print(f"警告: 第{line_num}行存在缺失字段（image/bbox/expression），已跳过 | Warning: Line {line_num} has missing fields (image/bbox/expression), skipped")
                continue

            # 拼接完整图片路径（优先级：image_base_path + image_rel_path > 直接用image_rel_path）
            # Splice full image path (Priority: image_base_path + image_rel_path > use image_rel_path directly)
            if image_base_path:
                image_filename = os.path.join(image_base_path, os.path.basename(image_rel_path))
            else:
                image_filename = image_rel_path

            # 处理expression（取第一个元素并去除首尾括号）
            # Process expression (take the first element and remove leading/trailing brackets)
            expr = expression[0] if isinstance(expression, list) and len(expression) > 0 else ""
            if expr.startswith('[') and expr.endswith(']'):
                expr = expr[1:-1].strip()

            # 处理bbox（转换为x1,y1,width,height并取整）
            # Process bbox (convert to x1,y1,width,height and round to integer)
            try:
                x1, y1, x2, y2 = map(float, bbox)  # 先转浮点防止整数/浮点混合 | Convert to float to avoid mixing int/float
                width_bbox = x2 - x1
                height_bbox = y2 - y1
                bbox_processed = [int(x1), int(y1), int(width_bbox), int(height_bbox)]
            except (ValueError, TypeError) as e:
                print(f"警告: 第{line_num}行bbox格式错误（需为[x1,y1,x2,y2]），已跳过 | 错误信息: {e} | Warning: Line {line_num} has invalid bbox format (requires [x1,y1,x2,y2]), skipped | Error: {e}")
                continue

            # 处理reasonings（原JSONL中无此字段，设为空列表后填充）
            # Process reasonings (this field does not exist in the original JSONL, set to empty list and pad)
            reasons = pad_strings_to_length([])  # 无reasonings则填充空字符串 | Fill with empty strings if no reasonings

            # 构造数据元组（保持与原代码一致的结构）
            # Construct data tuple (keep the same structure as the original code)
            fixed_pth = 'UNKNOWN.pth'  # 固定路径占位符 | Fixed path placeholder
            data_tuple = (image_filename, fixed_pth, bbox_processed, expr, [], reasons)
            dataset_train.append(data_tuple)

    # 确保输出目录存在 | Ensure the output directory exists
    output_dir = os.path.dirname(output_pth_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存为.pth文件 | Save as .pth file
    torch.save(dataset_train, output_pth_path)
    print(f"转换完成！共处理 {len(dataset_train)} 条有效数据 | Conversion completed! Processed {len(dataset_train)} valid data entries")
    print(f"数据已保存到: {output_pth_path} | Data saved to: {output_pth_path}")

def main():
    # 设置命令行参数解析 | Set up command line argument parsing
    parser = argparse.ArgumentParser(description='将JSONL文件转换为训练用的.pth文件（适配RefPath数据格式）| Convert JSONL file to .pth file for training (adapt to RefPath data format)')
    parser.add_argument('--input_jsonl', required=True, help='输入的JSONL文件路径（必填），例如: ./data/testB.jsonl | Path to input JSONL file (required), e.g., ./data/testB.jsonl')
    parser.add_argument('--output_pth', required=True, help='输出的.pth文件路径（必填），例如: ./data/refpath_testB.pth | Path to output .pth file (required), e.g., ./data/refpath_testB.pth')
    parser.add_argument('--image_base', default="", help='图片基础路径（可选），用于拼接完整图片路径，例如: /mnt/data/test | Base path of images (optional), used to splice full image path, e.g., /mnt/data/test')
    args = parser.parse_args()

    # 执行转换 | Execute conversion
    try:
        process_jsonl(args.input_jsonl, args.output_pth, args.image_base)
    except Exception as e:
        print(f"转换失败: {e} | Conversion failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()