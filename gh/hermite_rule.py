import os

if __name__ == '__main__':

    for i in range(129):
        print(i)
        os.system('./hermite_rule ' + str(i) + ' 0 1 0 data/hermite_' + str(i))
