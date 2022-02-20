import logging
import os
import json


class ColoredFormatter(logging.Formatter):
    with open(os.path.join(os.path.dirname(__file__), "color_config.json"), encoding='UTF-8') as f:
        config = json.loads(f.read())

    def __init__(self, fmt, use_color=True, datafmt=None, **kwargs):
        super().__init__(fmt, datafmt, **kwargs)
        self.use_color = use_color

    def format(self, record):
        if self.use_color:
            levelname = record.levelname
            mode = self.config['MODE']
            foreground = self.config['FOREGROUND']
            background = self.config['BACKGROUND']
            prefix = f"\033[{mode[levelname]};{foreground[levelname]};{background[levelname]}m"
            suffix = f"\033[0m"
            record.levelname = f"{prefix}{levelname}{suffix}"
        return super().format(record)
