import subprocess
import os

def record(seconds=0, out="out.wav", sample_rate=44100, channels=2):
    proc_args = ["arecord", "-D", "hw:1,0", "-f", "S16_LE", "-r", str(sample_rate), "-c", str(channels), out]

    if (seconds > 0):
        proc_args.append("-d")
        proc_args.append(str(seconds))

    proc = subprocess.Popen(proc_args, shell=False, preexec_fn=os.setsid)
    
    print("Starting recording... pid =" + str(proc.pid))

    if (seconds > 0):
        proc.wait()
        print("Recording done")
