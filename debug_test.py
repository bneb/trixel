import subprocess
import time
import sys

def run_test():
    p = subprocess.Popen(
        ['cargo', 'test', '-p', 'ternac_solver', '--test', 'z3_tests', '--', '--nocapture'],
        cwd='/Users/kevin/projects/ternac',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    
    start_time = time.time()
    while True:
        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                break
            time.sleep(0.1)
            continue
        
        print(f"[{time.time() - start_time:.2f}s] {line.strip()}")

if __name__ == '__main__':
    run_test()
