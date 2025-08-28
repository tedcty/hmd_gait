import os
import importlib.util
import sys

class Requirements:
    line = "##################################################"
    @staticmethod
    def windows_and_linux():
        print("Installing Requirements:\n\nPlease ensure cuda sdk is not install in your computer. "
              "Cuda sdk from the nvidia website will cause a conflict that will prevent numba from running.\n")
        print("{0}\n\tInstalling tsfresh ...\n{0}".format(Requirements.line))
        os.system("conda install -y conda-forge::tsfresh")

        print("{0}\n\tInstalling Opensim ...\n{0}".format(Requirements.line))
        os.system('conda install -y -c opensim-org opensim')
        os.system("conda install -y pytables")

        os.system('python -m pip install build toml')
        print("{0}\n\tInstalling ABI-MMG's PTB package...\n{0}".format(Requirements.line))
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
            print("\nwget Installed and/or Available ... continuing installation")
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

    print("Done!\nPlease check whether the packages have been installed successfully.")