import os

class config():
    ## 根目录
    root_path = os.getcwd()
    data_dir = os.path.join(root_path, 'data/Alpha0') # alpha粒子数据集路径
    output_path = os.path.join(root_path, 'output')

    ## 数据模态配置
    #modalities = ['ToT', 'ToA']  # 联合模态
    #modalities = ['ToA'] # 单模态 ToA
    modalities = ['ToT'] # 单模态 ToT

    ## 数据集划分
    train_split = 0.8  # 训练集比例
    random_seed = 42  # 随机种子用于可复现的划分

    ## 一般超参
    epoch = 20  # 训练轮数
    #weight = [1.0, 1.0]
    weight = [2.8, 7.1, 6.5, 2.8]  # 类别权重
    learning_rate = 1e-5  # 学习率
    weight_decay = 0  # 权重衰减
    batch_size = 256  # 批大小
    shuffle = True  # 是否打乱数据
    num_workers = 8  # DataLoader 的并行线程数

    ## 数据增强超参
    rotation = True  # 是否使用旋转增强

    ## 模型超参
    model_name = 'Resnet18'  # 模型名称
    kernel_size = 2  # 卷积核大小
    dropout_rate = 0.1  # dropout比率

    ## 特征超参
    inchannel = len(modalities)  # 输入通道数基于模态数量
    feature_size = 32  # 特征图大小
    cnn_feature_size = 256  # 卷积特征大小
    out_feature_size = 512  # 输出特征大小

    ## 手工特征配置（按模态划分）
    handcrafted_features = {
        'ToT': {
            'total_energy': True,
        },
        'ToA': {
            'total_energy': False,
        },
    }

    ## 手工特征标准化
    handcrafted_standardization = True  # 对每个手工特征做 z-score 标准化（基于训练集统计）

    ## 标准化/归一化配置（按模态划分）
        # 仅实现 z-score，预留接口支持未来扩展 min-max 等
        # per_modality 中的 enabled 控制是否对该模态启用标准化
        # log1p 控制是否在标准化前对该模态做 log1p
        # ignore_zero 控制统计均值/方差时是否忽略 0
    standardization = {
        'mode': 'zscore',  # 目前支持 'zscore'
        'per_modality': {
            'ToT': {
                'enabled': True,
                'log1p': False,
                'ignore_zero': True,
            },
            'ToA': {
                'enabled': True,
                'log1p': True,
                'ignore_zero': True,
            },
        }
    }

    @classmethod
    def input_channels(cls):
        return len(cls.modalities)

    @classmethod
    def features_for_modality(cls, modality):
        return cls.handcrafted_features.get(modality, {})

    @classmethod
    def enabled_modalities(cls):
        return list(cls.modalities)

    @classmethod
    def enabled_handcrafted_features(cls):
        enabled = []
        # 仅统计当前启用的模态对应的手工特征
        for modality in cls.modalities:
            flags = cls.handcrafted_features.get(modality, {})
            for name, use_flag in flags.items():
                if use_flag:
                    enabled.append((modality, name))
        return enabled

    @classmethod
    def handcrafted_feature_dim(cls):
        return len(cls.enabled_handcrafted_features())

    @classmethod
    def uses_handcrafted_features(cls):
        return cls.handcrafted_feature_dim() > 0

    @classmethod
    def standardization_mode(cls):
        return cls.standardization.get('mode', 'zscore')

    @classmethod
    def standardization_settings(cls):
        # 返回每个模态的配置，缺失模态回退为禁用
        per = cls.standardization.get('per_modality', {})
        settings = {}
        for m in cls.modalities:
            cfg = per.get(m, {})
            settings[m] = {
                'enabled': bool(cfg.get('enabled', False)),
                'log1p': bool(cfg.get('log1p', False)),
                'ignore_zero': bool(cfg.get('ignore_zero', False)),
            }
        return settings

    @classmethod
    def handcrafted_standardization_enabled(cls):
        return bool(cls.handcrafted_standardization)

    @classmethod
    def handcrafted_stats_path(cls):
        # 保存手工特征标准化统计量的位置
        return os.path.join(cls.output_path, 'handcrafted_feature_stats.json')