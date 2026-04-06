## 模块
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import os
import csv
import yaml
import numpy as np

## 评估模块
from matplotlib import pyplot as plt #绘图
from sklearn.metrics import confusion_matrix #混淆矩阵
from sklearn.metrics import classification_report  #分类报告
from sklearn.metrics import accuracy_score #准确率

## 自定义模块
from src.dataset import build_datasets 
from src.trainer import trainer
from src.evaluater import evaluater
from src.logger import ExperimentLogger, config_to_dict
from src.losses import build_loss_function
from model.utils import build_model, count_parameters
from Config import config

## 清空显存
torch.cuda.empty_cache()


def run_experiment(overrides: dict = None, save_plots: bool = True, experiment_dir: str = None):
    """
    核心训练/验证流程。

    Parameters
    ----------
    overrides : dict, optional
        用来临时覆盖 Config 中属性的字典
    save_plots : bool
        是否保存 loss/accuracy 曲线图
    experiment_dir : str, optional
        实验结果保存目录。若不指定，使用 config.output_path

    Returns
    -------
    dict
        包含本次实验的关键结果指标
    """
    # ---- 临时覆盖超参 ----
    original_values = {}
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                original_values[key] = getattr(config, key)
                setattr(config, key, value)

    try:
        return _run_experiment_inner(save_plots, experiment_dir)
    finally:
        # ---- 恢复原始值，避免污染后续实验 ----
        for key, value in original_values.items():
            setattr(config, key, value)


def _run_experiment_inner(save_plots: bool, experiment_dir: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task = getattr(config, 'task', 'classification')
    max_angle = getattr(config, 'max_angle', 90.0)

    out_dir = experiment_dir or config.output_path
    os.makedirs(out_dir, exist_ok=True)

    # 数据划分索引路径
    split_indices_path = getattr(config, '_split_indices_path', None)

    ## 数据集导入和处理
    use_three_way = (getattr(config, 'val_split', 0) > 0 and getattr(config, 'test_split', 0) > 0)

    build_kwargs = dict(
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
        task=task,
        max_angle=max_angle,
    )

    if use_three_way:
        build_kwargs['val_ratio'] = config.val_split
        build_kwargs['test_ratio'] = config.test_split
        build_kwargs['split_indices_path'] = split_indices_path
        train_dataset, valid_dataset, test_dataset, label_map = build_datasets(**build_kwargs)
    else:
        train_dataset, valid_dataset, label_map = build_datasets(**build_kwargs)
        test_dataset = None

    print(f'标签映射: {label_map}')
    num_classes = train_dataset.num_classes

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True) if test_dataset else None

    ## 模型导入
    model = build_model(config, num_classes=num_classes, device=device)
    param_count = count_parameters(model)

    if config.uses_handcrafted_features() and not getattr(model, 'supports_handcrafted_features', False):
        raise ValueError(
            f"当前模型 {config.model_name} 不支持手工特征，请在 Config.handcrafted_features 中关闭或切换模型。"
        )
    
    criterion = build_loss_function(config, num_classes=num_classes, label_map=label_map)
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # 学习率调度器
    scheduler = None
    if getattr(config, 'scheduler', 'none') == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epoch, eta_min=1e-6)

    # 构建角度值张量
    angle_values = torch.tensor(
        [float(label_map[i]) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)

    ## 训练和验证
    best_epoch = 0
    best_vacc = 0
    best_tacc = 0
    best_vloss = float('inf')
    best_vmae_argmax = float('inf')
    best_vmae_weighted = float('inf')
    best_vreport = None
    best_treport = None
    best_model_state = None
    patience_counter = 0
    early_stop_patience = getattr(config, 'early_stopping_patience', 0)

    train_loss_history = []
    valid_loss_history = []
    train_acc_history = []
    valid_acc_history = []
    train_mae_argmax_history = []
    train_mae_weighted_history = []
    valid_mae_argmax_history = []
    valid_mae_weighted_history = []

    # CSV 日志
    log_csv_path = os.path.join(out_dir, 'training_log.csv')
    log_fields = ['epoch', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc',
                  'train_mae_argmax', 'train_mae_weighted', 'valid_mae_argmax', 'valid_mae_weighted', 'lr']

    with open(log_csv_path, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    for epoch in range(config.epoch):
        print(f'---------Epoch {epoch+1}/{config.epoch}------------')
        current_lr = optimizer.param_groups[0]['lr']

        ## 训练
        tloss, tlabel, tpred, tmae = trainer(model, train_loader, criterion, optimizer, device, angle_values=angle_values)
        tacc = accuracy_score(tlabel, tpred) if tlabel else 0.0
        train_loss_history.append(tloss)
        train_acc_history.append(tacc)
        train_mae_argmax_history.append(tmae['mae_argmax'])
        train_mae_weighted_history.append(tmae['mae_weighted'])

        print(f'Train Loss: {tloss}')
        print(f'Train Accuracy: {tacc:.4f}')
        print(f'Train MAE (argmax): {tmae["mae_argmax"]:.2f}°  (weighted): {tmae["mae_weighted"]:.2f}°')
        
        ## 验证
        vloss, vlabel, vpred, vmae = evaluater(model, valid_loader, criterion, device, angle_values=angle_values)
        vacc = accuracy_score(vlabel, vpred) if vlabel else 0.0
        valid_loss_history.append(vloss)
        valid_acc_history.append(vacc)
        valid_mae_argmax_history.append(vmae['mae_argmax'])
        valid_mae_weighted_history.append(vmae['mae_weighted'])

        print(f'Valid Loss: {vloss}')
        print(f'Valid Accuracy: {vacc:.4f}')
        print(f'Valid MAE (argmax): {vmae["mae_argmax"]:.2f}°  (weighted): {vmae["mae_weighted"]:.2f}°')
        
        # 写入 CSV
        with open(log_csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                'epoch': epoch + 1,
                'train_loss': tloss,
                'valid_loss': vloss,
                'train_acc': round(tacc, 5),
                'valid_acc': round(vacc, 5),
                'train_mae_argmax': round(tmae['mae_argmax'], 3),
                'train_mae_weighted': round(tmae['mae_weighted'], 3),
                'valid_mae_argmax': round(vmae['mae_argmax'], 3),
                'valid_mae_weighted': round(vmae['mae_weighted'], 3),
                'lr': current_lr,
            })

        # 学习率调度
        if scheduler is not None:
            scheduler.step()
        
        ## 保存最佳模型（分类按准确率，回归按 MAE）
        if task == 'regression':
            is_better = vmae['mae_argmax'] < best_vmae_argmax
        else:
            is_better = vacc > best_vacc

        if is_better:
            best_epoch = epoch + 1
            best_vacc = vacc
            best_tacc = tacc
            best_vloss = vloss
            best_vmae_argmax = vmae['mae_argmax']
            best_vmae_weighted = vmae['mae_weighted']
            if tlabel and tpred:
                best_treport = classification_report(tlabel, tpred, zero_division=0)
            if vlabel and vpred:
                best_vreport = classification_report(vlabel, vpred, zero_division=0)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f'  ▸ 更新最佳模型 epoch={best_epoch}, acc={best_vacc:.4f}, mae={best_vmae_argmax:.2f}°')
        else:
            patience_counter += 1
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                print(f'  ▸ Early stopping at epoch {epoch + 1} (patience={early_stop_patience})')
                break

    # 保存最佳模型
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(out_dir, 'best_model.pth'))
        print(f'最佳模型已保存到 {out_dir}/best_model.pth')

    # 保存配置
    _save_config_yaml(out_dir)

    # ---- 测试集评估（如果有） ----
    test_results = {}
    if test_loader is not None and best_model_state is not None:
        print('\n======== 测试集评估 ========')
        model.load_state_dict(best_model_state)
        model = model.to(device)
        test_loss, test_true, test_pred, test_mae = evaluater(
            model, test_loader, criterion, device, angle_values=angle_values
        )
        test_acc = accuracy_score(test_true, test_pred) if test_true else 0.0
        print(f'Test Loss: {test_loss}, Test Acc: {test_acc:.4f}')
        print(f'Test MAE: {test_mae["mae_argmax"]:.2f}°')

        test_results = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_mae_argmax': test_mae['mae_argmax'],
            'test_mae_weighted': test_mae['mae_weighted'],
            'test_true': test_true,
            'test_pred': test_pred,
        }
        if task == 'regression':
            test_results['test_pred_angles'] = test_mae.get('pred_angles', [])
            test_results['test_true_angles'] = test_mae.get('true_angles', [])

        # 保存测试预测结果
        np.savez(
            os.path.join(out_dir, 'test_predictions.npz'),
            true_labels=np.array(test_true),
            pred_labels=np.array(test_pred),
            pred_angles=np.array(test_results.get('test_pred_angles', [])),
            true_angles=np.array(test_results.get('test_true_angles', [])),
            angle_values=np.array([float(label_map[i]) for i in range(num_classes)]),
            label_map_keys=np.array(list(label_map.keys())),
            label_map_values=np.array(list(label_map.values())),
        )

    if save_plots:
        _save_training_plots(out_dir, config.epoch, train_loss_history, valid_loss_history,
                             train_acc_history, valid_acc_history)

    ## 打印最佳结果
    print('\nTrain Finished! The best result is: ')
    print(f'Best Epoch: {best_epoch}')
    print(f'Valid Acc: {best_vacc:.4f}')
    print(f'Valid MAE (argmax): {best_vmae_argmax:.2f}°  (weighted): {best_vmae_weighted:.2f}°')

    # 释放 GPU 显存
    del model
    torch.cuda.empty_cache()

    return {
        'best_epoch': best_epoch,
        'best_vacc': best_vacc,
        'best_tacc': best_tacc,
        'best_vloss': best_vloss,
        'best_vmae_argmax': best_vmae_argmax,
        'best_vmae_weighted': best_vmae_weighted,
        'param_count': param_count,
        'label_map': label_map,
        **test_results,
    }


def _save_config_yaml(out_dir):
    """将当前配置保存为 YAML 文件。"""
    cfg_dict = {
        'task': getattr(config, 'task', 'classification'),
        'model_name': config.model_name,
        'modalities': list(config.modalities),
        'loss_type': config.loss_type,
        'label_encoding': config.label_encoding,
        'emd_p': config.emd_p,
        'gaussian_sigma': config.gaussian_sigma,
        'epoch': config.epoch,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'batch_size': config.batch_size,
        'dropout_rate': config.dropout_rate,
        'scheduler': getattr(config, 'scheduler', 'none'),
        'early_stopping_patience': getattr(config, 'early_stopping_patience', 0),
        'train_split': config.train_split,
        'val_split': getattr(config, 'val_split', 0),
        'test_split': getattr(config, 'test_split', 0),
        'random_seed': config.random_seed,
        'rotation': config.rotation,
        'max_angle': getattr(config, 'max_angle', 90.0),
    }
    with open(os.path.join(out_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, allow_unicode=True)


def _save_training_plots(out_dir, epochs, train_loss, valid_loss, train_acc, valid_acc):
    """保存训练曲线图。"""
    actual_epochs = len(train_loss)
    x = range(1, actual_epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x, train_loss, label='Train Loss')
    plt.plot(x, valid_loss, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Valid Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(x, train_acc, label='Train Accuracy')
    plt.plot(x, valid_acc, label='Valid Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Valid Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy_curve.png'))
    plt.close()


## 主函数，封装训练和验证过程并输出结果
def main():
    """
    主函数：封装训练和验证过程并输出结果
    """
    run_experiment()

if __name__ == '__main__':
    main()

