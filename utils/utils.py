import os
import time
def get_run_logdir(log_dir='./logs'):
    
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(log_dir, run_id) 
