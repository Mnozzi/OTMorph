import time  # 导入时间模块，用于计算训练耗时
from options.train_options import TrainOptions  # 从自定义模块导入训练参数配置类
from data.getDatabase import DataProvider  # 导入数据加载器类
from models.models import create_model  # 导入模型创建函数
from util.visualizer import Visualizer  # 导入可视化工具类
from math import *  # 导入数学函数库（如ceil）
from util import util  # 导入自定义工具模块（包含文件操作等）
import os  # 导入操作系统模块（路径操作）

opt = TrainOptions().parse() #  解析训练参数配置（batchSize、学习率等参数）
data_train     = DataProvider(opt.dataroot, mode="train")# 创建训练数据集提供器（加载训练数据）

# 计算总训练样本数和迭代次数
dataset_size   = data_train.n_data# 总样本数
training_iters = int(ceil(data_train.n_data/float(opt.batchSize)))# 总迭代次数（向上取整）
print('#training images = %d' % dataset_size)# 打印训练集大小

total_steps = 0# 初始化全局步数计数器
model = create_model(opt)# 创建模型（根据opt中的配置创建网络结构）
visualizer = Visualizer(opt)# 初始化可视化工具（可能用于TensorBoard等可视化）
# 创建检查点目录和图像保存目录
check_dir = opt.checkpoints_dir# 检查点保存路径
img_dir = os.path.join(check_dir, 'images')# 训练过程图像保存路径
util.mkdirs([check_dir, img_dir])# 递归创建目录

# 初始化最优传输（Optimal Transport）迭代计数器
ot_iter = 1
#以下为主训练循环
for epoch in range(opt.epoch_count, opt.niter + 1):
    epoch_start_time = time.time()#记一下时间

    """ Train """
    for step in range(1, training_iters+1):
        #图像、标签、路径
        ##########注意这里有路径，到时候看看数据集路径该写在哪里##########
        batch_x, batch_y, path = data_train(opt.batchSize)
        data = {'A': batch_x, 'B': batch_y, 'path': path}
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += 1#全局步数+1
        model.set_input(data)#把数据送入模型

        #这里是核心训练步骤——参数优化
        model.optimize_parameters(ot_iter, OTsteps=opt.OTsteps)
        ot_iter += 1 #更新OT迭代
        if ot_iter > opt.OTsteps: #重置OT迭代
            ot_iter = 1

        if step % opt.display_step == 0:
            for label, image_numpy in model.get_current_visuals().items():
                img_path = os.path.join(img_dir, '%s.png' % (label))
                util.save_image(image_numpy, img_path)
            
        if step % opt.plot_step == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, step, training_iters, errors, t, 'Train')

    if epoch % opt.save_epoch_img == 0:
        visualizer.save_current_results(model.get_current_visuals(), epoch, True)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter, time.time() - epoch_start_time))
