#!/usr/bin/env python
import subprocess, time
from multiprocessing import Pool

DPATH='../../../../simple-examples/data/'
MOD='custom'

HIDDEN_SIZE_LIST=[50, 200]
LR_LIST=[1.0]
LR_DECAY_LIST=[0.5, 0.6, 0.7, 0.8]
EXP_LIST=[1.0, 1.3, 1.5]
BATCH_SIZE_LIST=[20]
MME=40
MAX_EPOCH_LIST=[4, 6, 2]
LFUNC="logistic"
NSTEPS=20


QUEUE_SIZE = 3
SLEEP_TIME = 5 #in minutes
WAIT_TIME = 4*60 #in minutes


max_trial = WAIT_TIME//SLEEP_TIME
def execute_command(command_tuple):
  qsub_command = command_tuple[0]
  command_id = command_tuple[1]
  tmp_file = 'tmp/comm_'+str(command_id)
  trial = 0
  while(True):
    exit_status = subprocess.call(qsub_command, shell=True, stdout=open(tmp_file, 'w'))
    if exit_status is 1:  # Check to make sure the job submitted
      print "Job {0} failed to submit".format(qsub_command)
      return
    line = open(tmp_file).readline()
    if '.sdb' in line:
      l = line.split()
      job = l[0]
      print('Job started: '+job)
      break
    else:
      trial += 1
      time.sleep(SLEEP_TIME*60)
    if trial > max_trial:
      print("Failed to execute command: "+qsub_command)
      return

  time.sleep(SLEEP_TIME*60)
  while(True):
    check_command = 'qstat -n '+job
    with open(tmp_file, 'w') as f:
      exit_status = subprocess.call(check_command, shell=True, stdout=f, stderr=f)
      if exit_status is 1:  # Check to make sure the job submitted
        print "Job {0} failed to submit".format(check_command)
        return
    lines = open(tmp_file).readlines()
    line = ' '.join(lines)
    if 'Job has finished' in line:
        print('Job completed: '+job)
        break
    time.sleep(SLEEP_TIME*60)

  subprocess.call('rm '+tmp_file, shell=True)
    


command_list = []
count = 0
for HS in HIDDEN_SIZE_LIST:
  for LRATE in LR_LIST:
    for LRD in LR_DECAY_LIST:
      for BS in BATCH_SIZE_LIST:
        for ME in MAX_EPOCH_LIST:
          for EX in EXP_LIST:
            OUT='PTB_HS'+str(HS)+'_LR'+str(LRATE)+'_LRD'+str(LRD)+'_BS'+str(BS)+'_ME'+str(ME)+'_EXP'+str(EX)
            qsub_command = "qsub -v DATA_PATH="+DPATH+",MODEL="+MOD+",HIDDEN_SIZE="+str(HS)+",LR="+str(LRATE)+",LR_DECAY="+str(LRD)+",EXP="+str(EX)+",BATCH_SIZE="+str(BS)+",MAX_MAX_EPOCH="+str(MME)+",MAX_EPOCH="+str(ME)+",LOSS_FUNC="+LFUNC+",NUM_STEPS="+str(NSTEPS)+",OUTPUT="+OUT+" gpu_exp.pbs"
            command_list.append((qsub_command, count))
            count += 1



command_exe_pool = Pool(QUEUE_SIZE)
command_exe_pool.map(execute_command, command_list)

