import hydra
from omegaconf import DictConfig, OmegaConf
from gomoku_rl import CONFIG_PATH
from gomoku_rl.gui import GomokuBoard
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="demo")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    # pprint(OmegaConf.to_container(cfg))

    app = QApplication(sys.argv)

    board = GomokuBoard(board_size=cfg.get("board_size", 19))
    window = QMainWindow()
    window.setFixedSize(600, 800)
    window.setCentralWidget(board)
    window.setWindowTitle("demo")
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
