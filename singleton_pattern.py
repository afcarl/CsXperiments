class SingletonBase:

    instances = {}

    def __new__(cls):
        cname = cls.__class__.__name__
        if cname not in cls.instances:
            assert False, "cls() causes infinite recursion of course..."
            SingletonBase.instances[cname] = cls()
        return cls.instances[cname]


class ChildOne(SingletonBase):

    def __init__(self):
        print("I am being created!")


class ChildTwo(SingletonBase):

    def __init__(self):
        print("I am being created!")


if __name__ == '__main__':
    print(SingletonBase.instances)
    c1 = ChildOne()
    c2 = ChildTwo()
    print(SingletonBase.instances)
    c12 = ChildOne()
    c22 = ChildTwo()
    print(SingletonBase.instances)
