import os
import importlib.util
import sys
import time

if __name__ == '__main__':
    wget_spec = importlib.util.find_spec("wget")
    found = wget_spec is not None
    if not os.path.exists('./resources/wheels/ptb/'):
        os.makedirs('./resources/wheels/ptb/')
    file_path = './resources/wheels/ptb/latest.txt'
    if not found:
        os.system('python -m pip install wget')
    wget = importlib.import_module('wget')
    spam_spec = importlib.util.find_spec("wget")
    found = spam_spec is not None
    if found:
        print("\nwget Installed and Available continue installation")
    else:
        print("\nDone .... Please Rerun Script to continue!")
        sys.exit(0)
    url = 'https://raw.githubusercontent.com/tedcty/ptb/refs/heads/main/python_lib/ptb_src/dist/latest.txt'
    if os.path.exists(file_path):
        os.remove(file_path)
        os.listdir("./")
        time.sleep(0.1)
    wget.download(url, file_path)

    os.listdir('./resources/wheels/ptb/')
    if os.path.exists(file_path):
        with open(file_path) as f:
            version = f.read()
        print(version)

    latest_ptb = 'https://github.com/tedcty/ptb/raw/refs/heads/main/python_lib/ptb_src/dist/{0}'.format(version)
    os.system('python -m pip install {0}'.format(latest_ptb))