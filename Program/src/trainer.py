from Config import config
from src.losses import compute_angle_mae


def trainer(model, train_loader, criterion, optimizer, device, angle_values=None):
    """
    训练函数：封装训练过程
    """
    model.train()

    ## 记录变量初始化
    loss_list, true_labels, pred_labels = [], [], []
    total_ae_argmax = 0.0
    total_ae_weighted = 0.0
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
            logits, prob, pred = model(samples, handcrafted_features) # 获取模型输出,概率和预测类别，其中概率暂无用处
        else:
            logits, prob, pred = model(samples) # 获取模型输出, 概率和预测类别
        loss = criterion(logits, labels) # CrossEntropy损失函数会自动将 logits 转换为概率
        ### 记录
        loss_list.append(loss.item()) # 记录损失
        true_labels.extend(labels.tolist()) # 真实标签
        pred_labels.extend(pred.tolist()) # 预测标签

        ### MAE 计算（不参与梯度计算）
        if angle_values is not None:
            mae_batch = compute_angle_mae(logits, labels, angle_values)
            total_ae_argmax += mae_batch['ae_argmax']
            total_ae_weighted += mae_batch['ae_weighted']
            total_count += mae_batch['count']

        ## 反向传播
        loss.backward()
        optimizer.step()

    train_loss = round(sum(loss_list) / len(loss_list), 5) # 计算平均损失

    mae_dict = {'mae_argmax': 0.0, 'mae_weighted': 0.0}
    if angle_values is not None and total_count > 0:
        mae_dict = {
            'mae_argmax': total_ae_argmax / total_count,
            'mae_weighted': total_ae_weighted / total_count,
        }

    return train_loss, true_labels, pred_labels, mae_dict