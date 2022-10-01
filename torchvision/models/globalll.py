def add(x,y):
    z = x+y
    return z

class ABC():
    def __init__(self):
        self.v = 10
        
if __name__ == '__main__':
    print(add(2,3))
    a = ABC()
    print(a.v) # >>> 10
    print(v) # not defined 