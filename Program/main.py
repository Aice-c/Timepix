## 模块
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import importlib
import os

## 评估模块
from matplotlib import pyplot as plt #绘图
from sklearn.metrics import confusion_matrix #混淆矩阵
from sklearn.metrics import classification_report  #分类报告
from sklearn.metrics import accuracy_score #准确率

## 自定义模块
from src.dataset import build_datasets 
from src.trainer import trainer
from src.evaluater import evaluater
from Config import config

## 清空显存
torch.cuda.empty_cache()

## 动态导入模型
def load_model(model_name):
    """
    根据模型名称动态加载模型类
    """
    module_name = f"model.{model_name}"  # 模块路径
    class_name = "Model"  # 假设模型类名统一为 Model
    try:
        module = importlib.import_module(module_name)  # 动态导入模块
        model_class = getattr(module, class_name)  # 获取类
        return model_class
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"无法加载模型 {model_name}: {e}")


## 主函数，封装训练和验证过程并输出结果
def main():
    """
    主函数：封装训练和验证过程并输出结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## 数据集导入和处理
    train_dataset, valid_dataset = build_datasets(
        data_dir=config.data_dir,
        modalities=config.enabled_modalities(),
        train_ratio=config.train_split,
        seed=config.random_seed,
        rotation_enabled=config.rotation,
        feature_flags=config.handcrafted_features,
        crop_size=config.crop_size or 0,
        standardization_mode=config.standardization_mode(),
        per_modality_standardization=config.standardization_settings(),
        handcrafted_standardize=config.handcrafted_standardization_enabled(),
        handcrafted_stats_path=config.handcrafted_stats_path(),
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    ## 模型导入
    model_class = load_model(config.model_name) # 动态导入模型
    model = model_class(num_classes=train_dataset.num_classes).to(device) # 实例化模型

    if config.uses_handcrafted_features() and not getattr(model, 'supports_handcrafted_features', False):
        raise ValueError(
            f"当前模型 {config.model_name} 不支持手工特征，请在 Config.handcrafted_features 中关闭或切换模型。"
        )
    
    weights = torch.tensor(config.weight, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights) # 设定损失函数权重
    #criterion = torch.nn.CrossEntropyLoss() # 设定损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)  # 设定优化器

    ## 训练和验证
    # 记录变量初始化
    best_epoch = 0
    best_vacc = 0
    best_tacc = 0
    best_vreport = None
    best_treport = None
    train_loss_history = [] #训练集损失 
    valid_loss_history = [] #验证集损失
    train_acc_history = [] #训练集准确率
    valid_acc_history = [] #验证集准确率

    for epoch in range(config.epoch):
        print(f'---------Epoch {epoch+1}/{config.epoch}------------')
        ## 训练
        tloss, tlabel, tpred = trainer(model, train_loader, criterion, optimizer, device)
        tacc = accuracy_score(tlabel, tpred) # 计算训练集准确率
        train_loss_history.append(tloss)
        train_acc_history.append(tacc)

        print(f'Train Loss: {tloss}')
        print(f'Train Accuracy: {tacc}')
        print(f'Classification Report:\n{classification_report(tlabel, tpred)}')
        
        ## 验证
        vloss, vlabel, vpred = evaluater(model, valid_loader, criterion, device)
        vacc = accuracy_score(vlabel, vpred)
        valid_loss_history.append(vloss)
        valid_acc_history.append(vacc)

        print(f'Valid Loss: {vloss}')
        print(f'Valid Accuracy: {vacc}')
        print(f'Classification Report:\n{classification_report(vlabel, vpred)}')
        
        ## 保存模型
        if vacc > best_vacc:
            best_epoch = epoch + 1
            best_vacc = vacc
            best_tacc = tacc
            best_treport = classification_report(tlabel, tpred)
            best_vreport = classification_report(vlabel, vpred)
            #torch.save(model.state_dict(), config.output_path + f'/model_epoch_{best_epoch}.pth')
            print(f'Updated best model at epoch {best_epoch} with accuracy {best_vacc}')

    os.makedirs(config.output_path, exist_ok=True)  # 确保输出目录存在
    ## 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config.epoch + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, config.epoch + 1), valid_loss_history, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Valid Loss')
    plt.legend()
    plt.savefig(config.output_path + '/loss_curve.png')

    ## 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config.epoch + 1), train_acc_history, label='Train Accuracy')
    plt.plot(range(1, config.epoch + 1), valid_acc_history, label='Valid Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Valid Accuracy')
    plt.legend()
    plt.savefig(config.output_path + '/accuracy_curve.png')

    ## 打印最佳结果
    print('Train Finished! The best result is: ')
    print(f'Best Epoch: {best_epoch}')
    print(f'Train Acc: {best_tacc}') 
    print(f'Valid Acc: {best_vacc}')
    print(f'Best Train Classification Report: \n{best_treport}')
    print(f'Best Valid Classification Report: \n{best_vreport}')
if __name__ == '__main__':
    main()

