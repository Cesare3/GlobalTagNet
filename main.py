import argparse
import os

from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from pl_mltc.datasets.mltc_dataset import MultiLabelTextClassificationDataset
from pl_mltc.models import BertClassifier
from pl_mltc.models import CnnClassifier
from pl_mltc.models import GruClassifier
from pl_mltc.models import LSANClassifier
from pl_mltc.models import LSTMClassifier
from pl_mltc.models import Word2VecClassifier
from pl_mltc.models.gnn_classifier import GnnClassifier
from pl_mltc.utils.console import console
from pl_mltc.utils.env import PROJECT_DIR
from pl_mltc.utils.hparams import load_hparams


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--hp", help="path od the hparam file", required=True)
    parser.add_argument("--exp", help="name of the experiments", required=True)
    parser.add_argument("--version", help="version name", default="v0")

    parser.add_argument("--model", help="which model to use",
                        choices=("word2vec", "cnn", "gru", "lstm", "bert", "lsan", "gnn"),
                        default="word2vec")
    parser.add_argument("--do_evaluate", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--do_debug", action="store_true")
    
    parser.add_argument("--ckpt", help="path of the checkpoint", default=None)
    
    parser.add_argument("--gpus", help="which gpu(s) to use (if it is None, use cpu)", default=None)
    return parser.parse_args()
    
    
def main(args):
    # load hparams
    hp: DictConfig = load_hparams(args.hp)
    print(hp.data.max_seq_len)
    hp.trainer.seed = seed_everything(hp.trainer.seed)
    
    console.log(f"PROJECT_DIR={os.path.abspath(PROJECT_DIR)}")
    console.log(f"Seed is set to [bold green]{hp.trainer.seed}[/bold green].")
    # construct dataset
    console.log(f"Preparing dataset from [bold green]{hp.data.data_dir}[/bold green]...")
    data = MultiLabelTextClassificationDataset(
        data_dir=hp.data.data_dir,
        max_seq_len=hp.data.max_seq_len,
        train_batch_size=hp.data.train_batch_size,
        eval_batch_size=hp.data.eval_batch_size,
        model_name_or_path=getattr(hp.data, "model_name_or_path", None),
    )
    
    # construct model
    console.log(f"Building [bold green]{args.model.upper()}[/bold green]...")
    if args.model == "word2vec":
        model = Word2VecClassifier(
            hp=hp.model,
            vocab=data.vocab,
        )
    elif args.model == "cnn":
        model = CnnClassifier(
            hp=hp.model,
            vocab=data.vocab,
        )
    elif args.model == "gru":
        model = GruClassifier(
            hp=hp.model,
            vocab=data.vocab,
        )
    elif args.model == "lstm":
        model = LSTMClassifier(
            hp=hp.model,
            vocab=data.vocab,
        )
    elif args.model == "bert":
        model = BertClassifier(
            hp=hp.model,
        )
    elif args.model == "lsan":
        model = LSANClassifier(
            hp=hp.model,
            vocab=data.vocab,
        )
    elif args.model == "gnn":
        model = GnnClassifier(
            hp=hp.model,
            vocab=data.vocab,
        )
    else:
        raise NotImplementedError(f"not support model `{args.model}`.")
    
    do_train = not (
        args.do_evaluate or
        args.do_predict
    )
    
    # ----------------------------------
    # 2. INIT EARLY STOPPING
    # ----------------------------------
    if do_train:
        early_stop_callback = EarlyStopping(
            monitor=hp.trainer.early_stopping.monitor,
            min_delta=1e-5,
            patience=hp.trainer.early_stopping.patience,
            verbose=True,
            mode=hp.trainer.early_stopping.metric_mode,
        )
    else:
        early_stop_callback = None
        
    # ----------------------------------
    # 3. INIT LOGGERS
    # ----------------------------------
    experiments = PROJECT_DIR / "experiments"
    version_name = args.version
    if do_train:
        console.log(f"TensorBoardLogger would be built on the directory [bold green]{experiments}[/bold green] "
                    f"with the version [bold red]{version_name}[/bold red].")
        tb_logger = TensorBoardLogger(
            save_dir=experiments,
            name=args.exp,
            version=version_name
        )
    else:
        tb_logger = False
        
    # ----------------------------------
    # 4. INIT MODEL CHECKPOINT CALLBACK
    # ----------------------------------
    if do_train:
        # Model Checkpoint Callback
        ckpt_dir = experiments / args.exp / version_name / "ckpt"
        console.log(f"ModelCheckpoint would be built on the directory [bold green]{ckpt_dir}[/bold green].")
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{step:d}-{train_f1:.4f}-{val_f1:.4f}",
            save_top_k=hp.trainer.checkpoint.save_top_k,
            verbose=True,
            monitor=hp.trainer.checkpoint.monitor,
            mode=hp.trainer.checkpoint.metric_mode,
        )
    else:
        checkpoint_callback = None

    # ----------------------------------
    # 5. INIT TRAINER
    # ----------------------------------
    trainer = Trainer(
        logger=tb_logger,
        enable_checkpointing=True,
        gradient_clip_val=getattr(hp.trainer, "gradient_clip_val", None),
        deterministic=True,
        check_val_every_n_epoch=1,
        fast_dev_run=args.do_debug,
        accumulate_grad_batches=getattr(hp.trainer, "accumulate_grad_batches", None),
        max_epochs=getattr(hp.trainer, "max_epochs", None),
        min_epochs=getattr(hp.trainer, "min_epochs", None),
        max_steps=getattr(hp.trainer, "max_steps", -1),
        val_check_interval=getattr(hp.trainer, "val_check_interval", None),
        callbacks=[
            cb for cb in [RichProgressBar(leave=False), early_stop_callback, checkpoint_callback]
            if cb is not None
        ],
        accelerator='gpu' if args.gpus is not None else None,
        devices=args.gpus,
    )
    
    # evaluate
    if args.do_evaluate:
        # TODO: homework-2
        ...
    
    # predict
    if args.do_predict:
        # TODO: homework-1
        ...
        
    # fit
    if do_train:
        console.log(f'lr={hp.model.optimization.lr}')
        trainer.fit(model, data)
        return
    
    
if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
