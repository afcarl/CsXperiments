import os

from csxdata.parser.reparse import dehungarize

root = os.path.expanduser("~/Prog/PyCharm/")

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
        print("Dehungarizing...")
        bigpychain = dehungarize(bigpychain)

nlines = bigpychain.count("\n") - len(pyflz)
print("Concatenated 'em. Found {} characters in {} lines!".format(len(bigpychain), nlines))
print("Writing to [txt]!")

with open(os.path.expanduser("~/Prog/data/txt/scripts.txt"), "w", encoding="utf8") as outfl:
    outfl.write(bigpychain)
