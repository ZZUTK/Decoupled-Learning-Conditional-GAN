# check GPU status
from pynvml import *
from time import sleep
from datetime import datetime


def gpu_memory_usage(is_print=True):
    try:
        nvmlInit()
        # version = nvmlSystemGetDriverVersion()
        deviceCount = nvmlDeviceGetCount()
        GPU = {}
        for i in range(deviceCount):
            GPU[i] = {}
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            GPU[i]['total'] = info.total / 1024.0 / 1024.0 / 1024.0
            GPU[i]['free'] = info.free / 1024.0 / 1024.0 / 1024.0
            if is_print:
                print("\nGPU #%d Memory Usage:"
                      "\n\tTotal:\t%4.2fGB\n\tFree:\t%4.2fGB" %
                      (i, GPU[i]['total'], GPU[i]['free']))
                print datetime.now()
        nvmlShutdown()
        return GPU
    except:
        print "Fail to check GPU status!"
        exit(0)


def auto_queue(gpu_memory_require=3.2, interval=1, schedule=None):
    # input arg: schedule = datetime(year, month, day, hour, minute, second)
    if schedule is None:
        schedule = datetime.today()
    else:
        print '\nScheduled time: ', schedule

    # wait until the scheduled time
    now = datetime.today()
    while now.year < schedule.year or now.month < schedule.month or now.day < schedule.day or \
          now.hour < schedule.hour or now.minute < schedule.minute or now.second < schedule.second:
        now = datetime.today()
        sleep(interval)

    gpu_stat = gpu_memory_usage()
    if gpu_stat[0]['total'] < gpu_memory_require:
        print 'Memory requirement is larger than GPU total memory'
        exit(1)
    while gpu_stat[0]['free'] < gpu_memory_require:
        sleep(interval)  # second
        gpu_stat = gpu_memory_usage()
    return gpu_stat


def set_memory_usage(usage=12.0, allow_growth=True):
    auto_queue(gpu_memory_require=usage)
    try:
        import tensorflow as tf
        assert type(usage) is int or float
        assert usage >= 0
       
        config = tf.ConfigProto()
        gpu_stat = gpu_memory_usage()
        total_memory = gpu_stat[0]['total']
        if usage > total_memory:
            usage_percentage = 1.0
        else:
            usage_percentage = usage / total_memory
        config.gpu_options.allow_growth = allow_growth
        config.gpu_options.per_process_gpu_memory_fraction = usage_percentage
        return config
    except:
        print 'Failed to set memory usage!'
        return None


if __name__ == '__main__':
    gpu_memory_usage()
