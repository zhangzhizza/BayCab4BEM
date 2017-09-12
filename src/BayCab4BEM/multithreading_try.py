import threading
import numpy as np
import time
from multiprocessing import Value, Lock

class worker():

    def work(self, globalDict, globalLock, jobID, initValue):
        ret = [];
        toBeAdd = initValue;
        for i in range(10):
            time.sleep(0.1);
            #print (self.id)
            toBeAdd += 1;
            ret.append(toBeAdd);
        globalLock.acquire() # will block if lock is already held
        globalDict[jobID] = ret;
        globalLock.release()
        
globalLock = Lock();
globalDict = {};
totalJobBeDone = 10;
threads = [];

starttime = time.time();
jobID = 0;
while totalJobBeDone > 0:
    if len(threads) < 11:
        thisWorker = worker();
        worker_work = lambda: thisWorker.work(globalDict, globalLock, jobID, jobID);
        thread = threading.Thread(target = (worker_work));
        thread.start();
        threads.append(thread);
        jobID += 1;
    for thread in threads:
        if not thread.isAlive():
            threads.remove(thread);
            totalJobBeDone -= 1;

stoptime = time.time();
print (globalDict)
print ("Time cost: ", stoptime - starttime);