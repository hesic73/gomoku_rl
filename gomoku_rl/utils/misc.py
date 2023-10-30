import os
import logging

def get_checkpoint_dir(root_dir:str,board_size:int):
    tmp=os.path.join(root_dir,f"board_size_{board_size}")
    if not os.path.exists(tmp):
        logging.info(f"{tmp} doesn't exist. Creating it.")
        os.mkdir(tmp)
    return tmp