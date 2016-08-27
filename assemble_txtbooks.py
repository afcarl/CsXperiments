import os

from csxdata import roots

os.chdir("/data/Megosztott/Ebook/ABC_szerint")

outchain = ""
txts = []

for path, dirs, fls in os.walk("."):
    txts += [path + "/" + fl for fl in fls if ".txt" == fl[-4:]]

assert all(map(lambda x: isinstance(x, str), txts))


def extractinfo(encoding):
    chain = ""
    for flpath in txts:
        with open(flpath, encoding=encoding) as fl:
            try:
                chain += fl.read()
            except:
                pass
            else:
                txts.remove(flpath)
    return chain

for coding in ("utf-8", "utf-8-sig", "cp1250", "latin1", "latin2", "ascii",
               "852", "maclatin2", "U16", "U7"):
    outchain += extractinfo(coding)
    if len(txts) == 0:
        break

with open(roots["txt"] + "books.txt", "w", encoding="utf8") as outfl:
    outfl.write(outchain)
    outfl.close()

print("TXT assembled! Files unprocessed:", len(txts))
print("Finite incantatum!")
