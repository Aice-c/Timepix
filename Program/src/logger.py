"""
实验日志模块：自动记录每次实验的超参数和结果到 CSV 文件。
"""

import csv
import os
from datetime import datetime


class ExperimentLogger:
    """将每次实验的配置与结果追加到一个 CSV 文件中。"""

    def __init__(self, log_path):
        self.log_path = log_path
        self._fields = None

    def log(self, config_dict: dict, result_dict: dict):
        """
        记录一条实验记录。

        Parameters
        ----------
        config_dict : dict
            超参数字典，例如 {'learning_rate': 1e-3, 'batch_size': 64, ...}
        result_dict : dict
            结果字典，例如 {'best_epoch': 5, 'best_valid_acc': 0.72, ...}
        """
        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **config_dict,
            **result_dict,
        }

        file_exists = os.path.isfile(self.log_path)
        fieldnames = list(row.keys())

        # 如果文件已存在，读取已有列名并合并（保证新列也能写入）
        if file_exists:
            with open(self.log_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_fields = list(reader.fieldnames or [])
            # 保留已有列顺序，追加新列
            for f_name in fieldnames:
                if f_name not in existing_fields:
                    existing_fields.append(f_name)
            fieldnames = existing_fields

        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def config_to_dict(cfg) -> dict:
    """从 config 对象提取关键超参数为字典。"""
    return {
        'model_name': cfg.model_name,
        'modalities': '+'.join(cfg.modalities),
        'loss_type': getattr(cfg, 'loss_type', 'cross_entropy'),
        'label_encoding': getattr(cfg, 'label_encoding', 'onehot'),
        'emd_p': getattr(cfg, 'emd_p', 2),
        'gaussian_sigma': getattr(cfg, 'gaussian_sigma', 2.0),
        'epoch': cfg.epoch,
        'learning_rate': cfg.learning_rate,
        'weight_decay': cfg.weight_decay,
        'batch_size': cfg.batch_size,
        'dropout_rate': cfg.dropout_rate,
        'kernel_size': cfg.kernel_size,
        'cnn_feature_size': cfg.cnn_feature_size,
        'out_feature_size': cfg.out_feature_size,
        'crop_size': cfg.crop_size,
        'rotation': cfg.rotation,
        'weight': str(cfg.weight),
        'train_split': cfg.train_split,
        'random_seed': cfg.random_seed,
    }
