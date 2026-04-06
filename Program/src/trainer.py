from Config import config
from src.losses import compute_angle_mae, compute_regression_mae


def trainer(model, train_loader, criterion, optimizer, device, angle_values=None):
    """
    训练函数：封装训练过程，支持分类和回归任务。
    """
    model.train()
    task = getattr(config, 'task', 'classification')
    max_angle = getattr(config, 'max_angle', 90.0)

    ## 记录变量初始化
    loss_list, true_labels, pred_labels = [], [], []
    total_ae_argmax = 0.0
    total_ae_weighted = 0.0
    total_ae_reg = 0.0
    total_se_reg = 0.0
    total_count = 0

    use_handcrafted_features = config.uses_handcrafted_features()

    for batch in train_loader: # 遍历数据集，从train_loader中获取数据
        if use_handcrafted_features:
            samples, labels, handcrafted_features = batch
            handcrafted_features = handcrafted_features.to(device)
        else:
            samples, labels = batch
            handcrafted_features = None

        samples, labels = samples.to(device), labels.to(device) # 将数据移动到 GPU
        
        ## 前向传播
        optimizer.zero_grad() # 清空梯度
        if use_handcrafted_features:
            out1, out2, out3 = model(samples, handcrafted_features)
        else:
            out1, out2, out3 = model(samples)

        if task == 'regression':
            # out1 是归一化预测值 (batch,)，labels 是归一化角度值
            loss = criterion(out1, labels.float())
            loss_list.append(loss.item())
            # 回归任务的"预测标签"为最近类别角度（等效分类）
            pred_angles = out1.detach() * max_angle
            true_angles = labels.detach() * max_angle
            # 映射到最近类别
            if angle_values is not None:
                av = angle_values.unsqueeze(0)  # (1, K)
                pred_cls = (pred_angles.unsqueeze(1) - av).abs().argmin(dim=1)
                true_cls = (true_angles.unsqueeze(1) - av).abs().argmin(dim=1)
                pred_labels.extend(pred_cls.tolist())
                true_labels.extend(true_cls.tolist())

            # MAE
            reg_mae = compute_regression_mae(out1, labels, max_angle)
            total_ae_reg += reg_mae['ae']
            total_se_reg += reg_mae['se']
            total_count += reg_mae['count']
        else:
            logits, prob, pred = out1, out2, out3
            loss = criterion(logits, labels) # CrossEntropy损失函数会自动将 logits 转换为概率
            loss_list.append(loss.item())
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())

            if angle_values is not None:
                mae_batch = compute_angle_mae(logits, labels, angle_values)
                total_ae_argmax += mae_batch['ae_argmax']
                total_ae_weighted += mae_batch['ae_weighted']
                total_count += mae_batch['count']

        ## 反向传播
        loss.backward()
        optimizer.step()

    train_loss = round(sum(loss_list) / len(loss_list), 5)

    if task == 'regression':
        mae_dict = {
            'mae_argmax': total_ae_reg / max(total_count, 1),
            'mae_weighted': total_ae_reg / max(total_count, 1),
            'rmse': (total_se_reg / max(total_count, 1)) ** 0.5,
        }
    else:
        mae_dict = {'mae_argmax': 0.0, 'mae_weighted': 0.0}
        if angle_values is not None and total_count > 0:
            mae_dict = {
                'mae_argmax': total_ae_argmax / total_count,
                'mae_weighted': total_ae_weighted / total_count,
            }

    return train_loss, true_labels, pred_labels, mae_dict