import os
import shutil
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from utils import  update_arguments
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from pytorch_lightning.loggers import TensorBoardLogger


def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


yaml.add_constructor('!join', join)


def parser_args():
    parser = argparse.ArgumentParser(description='各个模型公共参数')
    parser.add_argument('--model_type', default="uie", type=str, help='定义模型类型')
    parser.add_argument('--pretrain_path', type=str, default="Langboatmengzi-t5-base", help='定义预训练模型路径')
    parser.add_argument('--lr', default=2e-5, type=float, help='specify the learning rate')
    parser.add_argument('--epoch', default=20, type=int, help='specify the epoch size')
    parser.add_argument('--batch_size', default=6, type=int, help='specify the batch size')
    parser.add_argument('--output_path', default="event_extract", type=str, help='将每轮的验证结果保存的路径')
    parser.add_argument('--float16', default=False, type=bool, help='是否采用浮点16进行半精度计算')
    parser.add_argument('--grad_accumulations_steps', default=3, type=int, help='梯度累计步骤')

    # 不同学习率scheduler的参数
    parser.add_argument('--decay_rate', default=0.999, type=float, help='StepLR scheduler 相关参数')
    parser.add_argument('--decay_steps', default=100, type=int, help='StepLR scheduler 相关参数')
    parser.add_argument('--T_mult', default=1.0, type=float, help='CosineAnnealingWarmRestarts scheduler 相关参数')
    parser.add_argument('--rewarm_epoch_num', default=2, type=int, help='CosineAnnealingWarmRestarts scheduler 相关参数')

    args = parser.parse_args()

    # 根据超参数文件更新参数
    config_file = os.path.join("data", "{}.yaml".format(args.model_type))
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    args = update_arguments(args, config['model_params'])
    args.config_file = config_file

    return args


def main():
    args = parser_args()
    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name=args.model_type)

    from UIE import UIEPytochLighting, UIEDataset, add_special_token_tokenizer, CollateFn
    from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

    tokenizer = add_special_token_tokenizer(args.pretrain_path)

    t5_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_path,cache_dir="./T5")
    t5_model.resize_token_embeddings(len(tokenizer))

    collate_fn = CollateFn(args, tokenizer, t5_model)
    train_dataset = UIEDataset(args, tokenizer, is_training=True)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)

    val_dataset = UIEDataset(args, tokenizer, is_training=False)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False)

    model = UIEPytochLighting(args, tokenizer, t5_model)
    save_temp_model = os.path.join(tb_logger.log_dir, "models")
    shutil.copytree("UIE", save_temp_model)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=8,
        verbose=True,
        monitor='f1',  # 监控val_acc指标
        mode='max',
        save_last=True,
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        every_n_epochs=1,
        # filename = "{epoch:02d}{f1:.3f}{acc:.3f}{recall:.3f}",
        filename="{epoch:02d}{f1:.3f}{acc:.3f}{recall:.3f}{sr_rec:.3f}{sr_acc:.3f}",
    )
    early_stopping_callback = EarlyStopping(monitor="f1",
                                            patience=8,
                                            mode="max",
                                            )
    trainer = pl.Trainer(max_epochs=args.epoch,
                         gpus=[1],
                         logger=tb_logger,
                         # accelerator = 'dp',
                         # plugins=DDPPlugin(find_unused_parameters=True),
                         check_val_every_n_epoch=1,  # 每多少epoch执行一次validation
                         callbacks=[checkpoint_callback,
                                    early_stopping_callback],
                         accumulate_grad_batches=args.grad_accumulations_steps,  # 累计梯度计算
                         precision=16 if args.float16 else 32,  # 半精度训练
                         gradient_clip_val=3,  # 梯度剪裁,梯度范数阈值
                         progress_bar_refresh_rate=5,  # 进度条默认每几个step更新一次
                         # O0：纯FP32训练,
                         # O1：混合精度训练，根据黑白名单自动决定使用FP16（GEMM, 卷积）还是FP32（Softmax）进行计算。
                         # O2：“几乎FP16”混合精度训练，不存在黑白名单，除了Batch norm，几乎都是用FP16计算
                         # O3：纯FP16训练，很不稳定，但是可以作为speed的baseline；
                         amp_level="O1",  # 混合精度训练
                         move_metrics_to_cpu=True,
                         amp_backend="apex",
                         # resume_from_checkpoint =""
                         )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
