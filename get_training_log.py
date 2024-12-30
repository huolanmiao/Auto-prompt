import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def extract_avg_acc(file_path):
    # 打开文件并读取内容
    with open(file_path, 'r') as file:
        content = file.read()

    # 使用正则表达式提取 avg_acc 后面的数字
    pattern = r'avg_acc:\s*([\d\.]+)'  # 匹配 avg_acc: 后的数字，可以是整数或浮动小数
    avg_acc_values = re.findall(pattern, content)

    # 将提取的数字转为浮动类型并返回
    return [float(value) for value in avg_acc_values]

def extract_and_average_acc(file_path):
    # 打开文件并读取内容
    with open(file_path, 'r') as file:
        content = file.read()

    # 正则表达式匹配 `acc:` 后面紧跟着的列表（假设列表是以 [ 开始， ] 结束的）
    pattern = r'acc:\s*\[([0-9.,\s-]+)\]'  # 匹配 acc: 后面的大括号中的数字
    acc_lists = re.findall(pattern, content)

    # 创建一个列表用于存放每个列表的平均值
    avg_acc_list = []

    for acc_str in acc_lists:
        # 将字符串中的数字转换为浮动类型的列表
        acc_values = list(map(float, acc_str.split(',')))
        
        # 计算该列表的平均值
        avg_acc = np.mean(acc_values)
        
        # 将计算的平均值添加到结果列表
        avg_acc_list.append(avg_acc)
    
    return avg_acc_list

def extract_and_average_validation_acc(file_path, batch_size=10):
    # 打开文件并读取内容
    with open(file_path, 'r') as file:
        content = file.read()

    # 使用正则表达式匹配 "Validation Acc" 后面的数字
    pattern = r'Validation Acc\s*[:=]?\s*([0-9.]+)'  # 匹配 "Validation Acc" 后的数字
    acc_values = re.findall(pattern, content)

    # 将所有提取出的字符串转为浮动类型的数字
    acc_values = list(map(float, acc_values))

    # 每10个数据求一个平均值
    avg_acc_list = []
    for i in range(0, len(acc_values), batch_size):
        batch = acc_values[i:i + batch_size]
        avg_acc = np.mean(batch)  # 计算当前批次的平均值
        avg_acc_list.append(avg_acc)

    return avg_acc_list

file_path = 'log_10poch_lr_1e-2.txt'  
avg_acc_list = extract_avg_acc(file_path)
avg_val_list = extract_and_average_validation_acc(file_path)
# file_path = 'log_default_5epoch_lr_1e-3.txt'
# avg_acc_list = extract_and_average_acc(file_path)
avg_val_list = extract_and_average_validation_acc(file_path)
print(avg_val_list)



def plot_interp_avg_acc(avg_acc_list):
    # 创建x轴坐标，表示数据点的索引
    x = np.arange(len(avg_acc_list))

    # 使用spline插值方法平滑数据
    spline = make_interp_spline(x, avg_acc_list, k=3)  # k=3表示三次样条插值
    x_smooth = np.linspace(x.min(), x.max(), 500)  # 生成平滑的x值
    y_smooth = spline(x_smooth)  # 通过插值计算平滑后的y值

    # 绘制原始数据点和插值平滑曲线
    plt.plot(x_smooth, y_smooth, label="Smoothed avg_acc", color='b')
    plt.scatter(x, avg_acc_list, color='r', label="Original data", alpha=0.5)  # 显示原始数据点

    # 添加标题和标签
    plt.title("Smoothed avg_acc over time")
    plt.xlabel("Index")
    plt.ylabel("avg_acc")
    plt.legend()
    
    # 显示图形
    # plt.savefig("acc_avg.png")
    # plt.savefig("acc_avg_interp_default.png")
    plt.savefig("acc_avg_interp_1e-2.png")

def moving_average(data, window_size):
    """
    计算移动平均。
    :param data: 输入数据列表
    :param window_size: 窗口大小
    :return: 平滑后的数据列表
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_moving_avg_acc(avg_acc_list, window_size=30):
    # 计算移动平均平滑数据
    smoothed_data = moving_average(avg_acc_list, window_size)
    
    # 创建x轴坐标，表示数据点的索引
    x_smooth = np.arange(len(smoothed_data))

    # 绘制平滑曲线
    plt.plot(x_smooth, smoothed_data, label=f"Smoothed avg_acc (window={window_size})", color='b')

    # 添加标题和标签
    plt.title("Smoothed avg_acc over time")
    plt.xlabel("Index")
    plt.ylabel("avg_acc")
    plt.legend()

    # 显示图形
    # plt.savefig("acc_avg_moving_default.png")
    # plt.savefig("acc_avg_moving_1e-2.png")
    

def plot_avg_acc(avg_acc_list):
    # 创建x轴坐标，表示批次的索引
    x = range(1, len(avg_acc_list) + 1)

    # 绘制图形
    plt.plot(x, avg_acc_list, label="Average Validation Acc", color='b', marker='o')

    # 添加标题和标签
    plt.title("Smoothed Validation Accuracy (Averaged over 10 epochs)")
    plt.xlabel("Batch Index")
    plt.ylabel("Average Validation Accuracy")
    plt.legend()

    plt.savefig("val_acc_avg_moving_1e-2.png")
    
# plot_interp_avg_acc(avg_acc_list)
plot_avg_acc(avg_val_list)