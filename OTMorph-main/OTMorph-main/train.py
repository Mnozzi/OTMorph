import os
import time
import torch  # 导入 PyTorch
from options.train_options import TrainOptions
from data.getDatabase import DataProvider
from models.models import create_model
from util.visualizer import Visualizer
from math import *
from util import util

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    opt = TrainOptions().parse()
    data_train = DataProvider(opt.dataroot, mode="train")
    dataset_size = data_train.n_data
    training_iters = int(ceil(data_train.n_data / float(opt.batchSize)))
    print(f'#training images = {dataset_size}')

    # 创建模型并移到 GPU
    model = create_model(opt).to(device)
    visualizer = Visualizer(opt)

    # 主训练循环
    total_steps = 0
    ot_iter = 1
    for epoch in range(opt.epoch_count, opt.niter + 1):
        epoch_start_time = time.time()
        for step in range(1, training_iters + 1):
            batch_x, batch_y, path = data_train(opt.batchSize)
            data = {'A': batch_x, 'B': batch_y, 'path': path}
            total_steps += 1
            model.set_input(data)
            model.optimize_parameters(ot_iter, OTsteps=opt.OTsteps)
            ot_iter += 1
            if ot_iter > opt.OTsteps:
                ot_iter = 1

            # 可视化和日志输出
            if step % opt.display_step == 0:
                for label, image_numpy in model.get_current_visuals().items():
                    img_path = os.path.join(img_dir, f'{label}.png')
                    util.save_image(image_numpy, img_path)
            if step % opt.plot_step == 0:
                errors = model.get_current_errors()
                t = (time.time() - time.time()) / opt.batchSize
                visualizer.print_current_errors(epoch, step, training_iters, errors, t, 'Train')

        # 保存模型和结果
        if epoch % opt.save_epoch_img == 0:
            visualizer.save_current_results(model.get_current_visuals(), epoch, True)
        if epoch % opt.save_epoch_freq == 0:
            print(f'saving model at epoch {epoch}, iters {total_steps}')
            model.save('latest')
            model.save(epoch)

        print(f'End of epoch {epoch}/{opt.niter} \t Time: {time.time() - epoch_start_time:.2f} sec')


if __name__ == "__main__":
    main()  # 直接运行 train.py
