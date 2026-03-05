import torch
from Config import config


def evaluater(model, test_loader, criterion, device):
    """
    验证函数：封装验证过程
    """
    model.eval()

    ## 记录变量初始化
    loss_list, true_labels, pred_labels = [], [], []

    with torch.no_grad(): # 禁用梯度计算
        use_handcrafted_features = config.uses_handcrafted_features()

        for batch in test_loader: # 遍历数据集，从test_loader中获取数据
            if use_handcrafted_features:
                samples, labels, handcrafted_features = batch
                handcrafted_features = handcrafted_features.to(device)
            else:
                samples, labels = batch
                handcrafted_features = None

            samples, labels = samples.to(device), labels.to(device) # 将数据移动到GPU
            
            ## 前向传播 
            if use_handcrafted_features:
                logits, prob, pred = model(samples, handcrafted_features) 
            else:
                logits, prob, pred = model(samples)
            ### 记录 
            loss = criterion(logits, labels) # 计算损失 
            loss_list.append(loss.item()) # 记录损失 
            true_labels.extend(labels.tolist()) # 真实标签 
            pred_labels.extend(pred.tolist()) # 预测标签 
            
            ## 验证无反向传播过程 
        test_loss = round(sum(loss_list) / len(loss_list), 5) # 计算平均损失 
    return test_loss, true_labels, pred_labels

