import hydra
from omegaconf import DictConfig, OmegaConf
from gomoku_rl import CONFIG_PATH
from gomoku_rl.utils.wandb import init_wandb
from gomoku_rl.runner import IBRRunner


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_sp_ibr")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    init_wandb(cfg=cfg)
    runner = IBRRunner(cfg=cfg)
    runner.run()


if __name__ == "__main__":
    main()
