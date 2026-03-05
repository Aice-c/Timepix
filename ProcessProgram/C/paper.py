import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy import ndimage
#添加了中心点像素坐标
#统计图显示粒子编号
#输出对应单个粒子数据文件
#显示粒子中心点位置
#输出图片文件中，每个像素点上显示原始数据的值


numb = 0
def center_particle_in_box(particle, original_data, box_size=7):
    # 找到粒子的边界
    rows = np.any(particle, axis=1)
    cols = np.any(particle, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # 裁剪粒子
    cropped = original_data[ymin:ymax+1, xmin:xmax+1]

    # 计算新图像的中心
    centered_image = np.zeros((box_size, box_size), dtype=original_data.dtype)
    y_center = (box_size - (ymax - ymin + 1)) // 2
    x_center = (box_size - (xmax - xmin + 1)) // 2

    # 将粒子置于中心
    centered_image[y_center:y_center+ymax-ymin+1, x_center:x_center+xmax-xmin+1] = cropped
    return centered_image

def process_file(file_path, energy_list, time_list, toa_folder):
    global numb  # 声明numb为全局变量
    # 读取数据
    data = np.loadtxt(file_path)

    # 检查数据尺寸
    if data.shape != (2048, 2048):
        raise ValueError("数据尺寸不正确")

    # 设置阈值
    threshold = 0

    # 找到所有像素值大于阈值的区域
    high_energy_pixels = data > threshold

    # 计算连通区域
    labeled, num_features = ndimage.label(high_energy_pixels)

    # 处理每个区域
    particle_count = 0
    for feature in range(1, num_features + 1):
        # 提取单个粒子轨迹
        single_particle = labeled == feature

        # 检查该区域的总能量是否大于2000
        if np.sum(data[single_particle]) > 10 and np.sum(single_particle) > 4: # 连通区域大于10个像素点
            particle_count += 1
            # 将粒子轨迹居中，使用原始数据中的灰度值
            centered_particle = center_particle_in_box(single_particle, data, box_size=7)

            if centered_particle is not None:
                # 计算粒子连通区域面积
                area = np.sum(single_particle)
                numb += 1
                # 计算粒子到达时间的最小值
                toa_file_path = os.path.join(toa_folder, os.path.basename(file_path).replace('ToT.txt', 'ToA.txt'))
                toa_data = np.loadtxt(toa_file_path)
                min_toa_value = np.max(toa_data[single_particle])-np.min(toa_data[single_particle])

                # 获取粒子中心点在原始数据中的位置
                rows = np.any(single_particle, axis=1)
                cols = np.any(single_particle, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2

                # 将值记录到列表中
                energy_list.append(np.sum(centered_particle))
                time_list.append(min_toa_value)
                # 创建新的图像
                plt.figure()

                # 替换为0的数据为白色
                data_for_plot = np.where(centered_particle == 0, np.nan, centered_particle)

                # 可视化数据
                plt.imshow(data_for_plot, cmap='turbo', interpolation='nearest', vmin=0, vmax=255, extent=[0, 7, 0, 7])

                ax = plt.gca()
                ax.set_xlim(0, 7)
                ax.set_ylim(0, 7)
                ax.set_aspect('equal')
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)

                for spine in ax.spines.values():
                    spine.set_linewidth(2.0)

                # 在每个像素点上添加原始数据值
                #for i in range(centered_particle.shape[0]):
                 #   for j in range(centered_particle.shape[1]):
                  #      if centered_particle[i, j] != 0:
                   #         plt.text(j, i, f'{int(centered_particle[i, j])}', ha='center', va='center', color='black', fontsize=6)

                # 添加标题和颜色条
                # plt.title('particle')
                #plt.colorbar()
                cbar = plt.colorbar(
                    orientation='horizontal',
                    pad=0.12,  # 与主图的距离
                    fraction=0.05
                )

                cbar.set_label('Pixel Value (a.u.)', fontsize=10)
                cbar.ax.tick_params(labelsize=9)

                # 在图像下方显示连通区域面积和粒子中心点位置
                # plt.figtext(0.5, 0.01, f"Particle Area: {area} pixels, Center: ({center_x}, {center_y})", ha="center", fontsize=10)

                # 保存图像
                output_path = os.path.join(output_folder, f"{os.path.basename(file_path)}_{numb}.png")
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close()


                # 保存切割后的粒子 TOT 数据
                centered_tot_data_path = os.path.join(output_folder, f"{os.path.basename(file_path)}_{numb}_TOT.txt")
                np.savetxt(centered_tot_data_path, centered_particle, fmt='%d')

    return particle_count


def plot_statistics(energy_list, time_list, particle_count, save_path=None):
    # 绘制统计图
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(energy_list, time_list, alpha=0.5)
    ax.set_title('Energy vs. Time')
    ax.set_xlabel('Total Energy')
    ax.set_ylabel('Time of Arrival')
    ax.grid(True)
    plt.figtext(0.5, 0.01, f"Total Particles: {particle_count}", ha="center", fontsize=10)

    # 添加鼠标悬停事件处理函数
    def on_motion(event):
        if event.inaxes == ax:
            contains, indices = scatter.contains(event)
            if contains:
                index = indices["ind"][0]
                ax.annotate(str(index + 1), xy=(energy_list[index], time_list[index]), textcoords="offset points", xytext=(0,10), ha='center')
                plt.draw()
            else:
                # 移除所有文本
                for text in ax.texts:
                    text.remove()
                plt.draw()

    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path)

    plt.show()



save_path = r'E:\5.文章\3.探测器标定文章\单粒子图片\原始文件\gsense111单粒子'
# TOA 文件夹路径       此代码中这个路径可以忽略，随便填写一个有效文件夹路径就可以


toa_folder = r'E:\5.文章\3.探测器标定文章\单粒子图片\原始文件\gsense111单粒子'
# 此路径可忽略，随便填写一个有效文件夹路径即可



# 记录能量和时间的列表
energy_list = []
time_list = []
total_particle_count = 0

# 新的输出文件夹路径，保存图片
output_folder = r'E:\5.文章\3.探测器标定文章\单粒子图片\原始文件\gsense111单粒子\输出'

# 处理文件
#for file_path in glob.glob(r'G:\1.日常实验测试\测试粒子每个像素点显示原始数据\中位数处理本底\*.txt'):          #输入文件夹路径修改，以.txt结尾的文件
#   total_particle_count += process_file(file_path, energy_list, time_list, toa_folder)

for file_path in glob.glob(r'E:\5.文章\3.探测器标定文章\单粒子图片\原始文件\gsense111单粒子\*.txt'):          #输入文件夹路径修改，以.txt结尾的文件
    total_particle_count += process_file(file_path, energy_list, time_list, toa_folder)

# 调用plot_statistics函数时，传入保存路径参数
plot_statistics(energy_list, time_list, total_particle_count, save_path="statistics_plot.png")
