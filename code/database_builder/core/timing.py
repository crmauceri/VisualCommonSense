import time;
start = time.time();
def tic():
    start = time.time();
    
def toc():
    print("elapse time: " + str(time.time()-start));

if __name__ == "__main__":
    tic();
    toc();
    
