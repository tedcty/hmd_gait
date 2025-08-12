import os
import importlib.util
import sys

class Requirements:
    @staticmethod
    def windows_and_linux():
        print("Installing Requirements")
        os.system('python -m pip install build toml')

        print("\n\tInstalling ABI-MMG's PTB package...")
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
        wget.download(url, file_path)

        os.listdir('./resources/wheels/ptb/')
        if os.path.exists(file_path):
            with open(file_path) as f:
                version = f.read()
            print(version)

        latest_ptb = 'https://github.com/tedcty/ptb/raw/refs/heads/main/python_lib/ptb_src/dist/{0}'.format(version)
        os.system('python -m pip install {0}'.format(latest_ptb))

    @staticmethod
    def macos():
        print("Not Implemented - todo")

if __name__ == '__main__':
    os_platform = sys.platform
    print(f"OS platform: {os_platform}")
    if os_platform == 'win32':
        print("Running on Windows")
        Requirements.windows_and_linux()
    elif os_platform == 'linux':
        print("Running on Linux")
        Requirements.windows_and_linux()
    elif os_platform == 'darwin':
        print("Running on macOS")
        Requirements.macos()
    else:
        print(f"Running on an unknown OS: {os_platform}")

    print("Done!")