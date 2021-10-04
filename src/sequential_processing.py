
no_of_task=2
jobs_per_task=[1, 1]
operations_per_job=[45, 45]

def isOdd(y,k):
  if(y in [0] + list(range(1,operations_per_job[k],2))):
    return True
  else:
    return False
def isEven(y,k):
  if(y in range(2,operations_per_job[k],2)):
    return True
  else:
    return False

#Processing input file
file1=open('stats_mem_seq.txt','r')
file2=open('freship1.txt','w')

at_time=0
new_at_time=0
for i in range(no_of_task):#Number of tasks
  for k in range(jobs_per_task[i]):#Number of jobs in a task
    for x in range(operations_per_job[i]):#Number of operations per task
      line=file1.readline()
      task,op_type,op,layer_type,at,et=line.strip().split(":")
      new_at_time+=eval(et)
      file2.write(str(round(at_time))+" "+et)
      file2.write("\n")  
file1.close()
file2.close()