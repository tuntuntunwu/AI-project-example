import labelme_to_yolo as lty
import sys

if __name__ == '__main__':
    print(sys.argv)
    print(len(sys.argv))
    if len(sys.argv) != 5:
        pass
    else:
        _, method, para1, para2, para3 = sys.argv
        print(method)
        print(para1)
        if method == "lty":
            lty.labelme_to_yolo(para1, para2, para3)

