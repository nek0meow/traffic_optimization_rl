import json
import sys

from src.gui_manager import GUIManager

if __name__ == '__main__':
    config = json.loads(sys.argv[1])
    manager = GUIManager(config)
    manager.run()