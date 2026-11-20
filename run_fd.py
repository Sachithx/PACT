"""FD entry point. Equivalent to ``python run_main.py --dataset fd``.

    python run_fd.py
    python run_fd.py --data_dir /path/to/FD --orders 2,1,0,3
"""

import sys

from run_main import main

if __name__ == "__main__":
    sys.argv.insert(1, "--dataset")
    sys.argv.insert(2, "fd")
    main()
