import sys
from fashion_mnist import mnist_model_proc as mnist_model

if __name__ == '__main__':
    print("---------------------------------------------")
    print("--            AI MODEL Program             --")
    print("--                    program date: 2024.9 --")
    print("--                         program by odin --")
    print("---------------------------------------------")    
    
    if sys.argv[1] == '-h':
        print('arg : -r(run), -m(view moel)')
        print("python3 -r [Model] [train :0 , inference :1 ] epoch_count")
    
    if sys.argv[1] == '-m':
        print("1. fashion mnist")

    if sys.argv[1] == '-r':
        if sys.argv[2] == '1' : ## mnist model
            if sys.argv[3] == '0' : ## train    
                print("----mnist train")            
                mnist_model(0, int(sys.argv[4]))
    


