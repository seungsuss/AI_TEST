import sys
from fashion_mnist import mnist_model_proc as mnist_model

if __name__ == '__main__':
    print("---------------------------------------------")
    print("--            AI MODEL Program             --")
    print("--                    program date: 2024.9 --")
    print("--                         program by odin --")
    print("---------------------------------------------")    
    
    if sys.argv[1] == '-h':
        print('arg : -r(run), -m(view moel) -s(show support model)')
        print("python3 -r [Model] [train :0 , inference :1 ] epoch_count/input_file_path(png)")
    
    if sys.argv[1] == '-m':
        print("1. fashion mnist")

    if sys.argv[1] == '-s':
        print("mnist Model : mnist ")

    if sys.argv[1] == '-r':
        if sys.argv[2] == 'mnist' : ## mnist model
            if sys.argv[3] == '0' : ## train                    
                mnist_model(0, epoch_num = int(sys.argv[4]))
            elif sys.argv[3] == '1' : ## inference
                mnist_model(1, epoch_num = 0, input_file_path= sys.argv[4])
    


