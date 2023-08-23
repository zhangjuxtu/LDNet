def print_log(epoch, iter, total_iter, value1, value2=None):
    msg =('epoch:{}, iters:{}/{} '.format(epoch, iter, total_iter))
    for k, v in value1.items():
        msg += '%s: %.4f ' % (k, v)
    if value2:
        msg += '\n'
        for k, v in value2.items():
            msg += '%s:%.3f ' % (k, v)
    if iter == total_iter - 1:
        msg += '\n##########################################'
    print(msg)
    #write_log(msg)

class Logger(object):
   '''Save training process to log file with simple plot function.'''
   def __init__(self, fpath, title=None, resume=False): 
       self.file = None
       self.resume = resume
       self.title = '' if title == None else title
       if fpath is not None:
           if resume: 
               self.file = open(fpath, 'r') 
               name = self.file.readline()
               self.names = name.rstrip().split('\t')
               self.numbers = {}
               for _, name in enumerate(self.names):
                   self.numbers[name] = []

               for numbers in self.file:
                   numbers = numbers.rstrip().split('\t')
                   for i in range(0, len(numbers)):
                       self.numbers[self.names[i]].append(numbers[i])
               self.file.close()
               self.file = open(fpath, 'a')  
           else:
               self.file = open(fpath, 'w')

   def set_names(self, names):
       if self.resume: 
           pass
       # initialize numbers as empty list
       self.numbers = {}
       self.names = names
       for _, name in enumerate(self.names):
           self.file.write(name)
           self.file.write('\t')
           self.numbers[name] = []
       self.file.write('\n')
       self.file.flush()

   def append(self, numbers):
       assert len(self.names) == len(numbers), 'Numbers do not match names'
       #print('logger append')
       for index, num in enumerate(numbers):
           self.file.write("{}".format(num))
           self.file.write('\t')
           self.numbers[self.names[index]].append(num)
       self.file.write('\n')
       self.file.flush()

   def plot(self, names=None):   
       names = self.names if names == None else names
       numbers = self.numbers
       for _, name in enumerate(names):
           x = np.arange(len(numbers[name]))
           plt.plot(x, np.asarray(numbers[name]))
       plt.legend([self.title + '(' + name + ')' for name in names])
       plt.grid(True)

   def close(self):
       if self.file is not None:
           self.file.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        #print('AverageMeter reset')
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
