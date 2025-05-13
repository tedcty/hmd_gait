from ptb.util.data import MYXML
import os
import numpy as np


def edit_range():
    wkdir = "M:/Models/"
    for c in os.listdir(wkdir):
        if c.endswith(".osim"):
            m = MYXML("{0}{1}".format(wkdir, c))
            a = m.tree.getElementsByTagName("Coordinate")
            for aa in a:
                if aa.getAttribute("name") == "pro_sup_r" or aa.getAttribute("name") == "pro_sup_l":
                    b = aa.getElementsByTagName("range")[0].childNodes[0]
                    p = "Before: {0}".format(b.data)
                    b.data = '0 {0}'.format(np.pi)
                    q = aa.getElementsByTagName("range")[0].childNodes[0].data
                    pass
                pass
            m.write("M:/temp/{0}".format(c), pretty=True)
            pass


if __name__ == '__main__':
    edit_range()
