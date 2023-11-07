import hydra
from omegaconf import DictConfig, OmegaConf
from gomoku_rl import CONFIG_PATH
from gomoku_rl.gui import GomokuBoard
from gomoku_rl.gui import Piece
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import logging
from gomoku_rl.algo import get_policy
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
    observation_spec = CompositeSpec(
            {
                "observation": UnboundedContinuousTensorSpec(
                    device=cfg.device,
                    shape=[1, 3, cfg.board_size, cfg.board_size],
                ),
            },
            shape=[
                1,
            ],
            device=cfg.device,
        )

    if cfg.get("human_first", True):
        human_color = Piece.BLACK
    else:
        human_color = Piece.WHITE

    model_ckpt_path = cfg.get("model_ckpt_path", None)
    if model_ckpt_path is not None:
        policy=get_policy(name=cfg.algo.name,cfg=cfg.algo,action_spec=action_spec,observation_spec=observation_spec,device=cfg.device)
        model=policy.get_actor()
        model.load_state_dict(torch.load(model_ckpt_path))
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
