import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train_fusion import main


if __name__ == "__main__":
    main()
