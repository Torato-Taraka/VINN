import logging
import sys
import os
import json
from .ColoredLogger import ColoredFormatter

with open(os.path.join(os.path.dirname(__file__), './logger_config.json'), encoding="UTF-8") as f:
    config = json.loads(f.read())


def init_logger():
    # 创建logger记录器
    logger = logging.getLogger(config.get("logger_name"))
    logger.setLevel(logging.DEBUG)

    # 创建控制台处理器
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setStream(sys.stdout)

    # 创建formatter格式化器
    formatter = ColoredFormatter(fmt="[%(levelname)s]"
                                     "%(asctime)s"
                                     # "[%(name)s]"
                                     # "(%(filename)s,%(lineno)s)"
                                     " | %(message)s",
                                 use_color=config.get("use_color"),
                                 datafmt="%Y-%m-%d %H:%M:%S"
                                 )

    # 将formatter添加到控制台处理器
    sh.setFormatter(formatter)

    # 将控制台处理器添加到logger
    logger.addHandler(sh)


def get_logger():
    return logging.getLogger(config.get("logger_name"))
