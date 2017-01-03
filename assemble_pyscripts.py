import os
import sys

from csxdata import roots
from csxdata.utilities.misc import dehungarize

root = "E:/PyCharm/" if sys.platform == "win32" else "/data/Prog/PycharmProjects"

pyflz = []
for path, dirs, flz in os.walk(root):
    pyflz += [path + "/" + fl for fl in flz if fl[-3:] == ".py" and "__" not in fl]

print("\n".join(pyflz))
print("Found", len(pyflz), "python filez!")

bigpychain = ""
for pyfl in pyflz:
    try:
        bigpychain += open(pyfl, "r").read()
    except UnicodeDecodeError:
        print("UnicoDecodeError @ ", pyfl)
        print("Skipping...")
    else:
        bigpychain += "##ENDFILE##\n"
    finally:
        bigpychain = dehungarize(bigpychain)

nlines = bigpychain.count("\n") - len(pyflz)
print("Concatenated 'em. Found {} characters in {} lines!".format(len(bigpychain), nlines))
print("Writing to roots[txt]!")

with open(roots["txt"] + "scripts.txt", "w", encoding="utf8") as outfl:
    outfl.write(bigpychain)
