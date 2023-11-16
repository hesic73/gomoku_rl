import hydra
from omegaconf import DictConfig, OmegaConf
from gomoku_rl import CONFIG_PATH
from gomoku_rl.gui import GomokuBoard
from gomoku_rl.gui import Piece
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import logging
from gomoku_rl.algo import PPOPolicy
from torchrl.data.tensor_specs import DiscreteTensorSpec,CompositeSpec,UnboundedContinuousTensorSpec
import torch

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="demo")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    # pprint(OmegaConf.to_container(cfg))

    # 设计的有点问题，耦合了
    # 有空再改
    action_spec = DiscreteTensorSpec(
        cfg.board_size * cfg.board_size,
        shape=[
            1,
        ],
        device=cfg.device,
    )

    if cfg.get("human_black", True):
        human_color = Piece.BLACK
    else:
        human_color = Piece.WHITE

    model_ckpt_path = cfg.get("model_ckpt_path", None)
    if model_ckpt_path is not None:
        model=PPOPolicy.from_checkpoint(torch.load(model_ckpt_path),cfg=cfg.algo.actor,action_spec=action_spec,device=cfg.device)
    else:
        model = None

    app = QApplication(sys.argv)

    board = GomokuBoard(
        board_size=cfg.get("board_size", 19),
        human_color=human_color,
        model=model,
    )
    window = QMainWindow()
    window.setFixedSize(600, 800)
    window.setCentralWidget(board)
    window.setWindowTitle("demo")
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
