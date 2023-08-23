#############################################################################################################
##
##  Source code for training. In this source code, there are initialize part, training part, ...
##
#############################################################################################################

import cv2
import math
import torch
#import visdom
import agent
import numpy as np
from data_loader import Generator
from parameters import Parameters
import test
import evaluation
import util_log
from tensorboardX import SummaryWriter

p = Parameters()

###############################################################
##
## Training
## 
###############################################################
def Training(logger):
    print('Training')

    ####################################################################
    ## Hyper parameter
    ####################################################################
    print('Initializing hyper parameter')

    writer = SummaryWriter(log_dir=p.save_path)
    
    #########################################################################
    ## Get dataset
    #########################################################################
    print("Get dataset")
    loader = Generator()

    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')
    if p.resume== True:
        lane_agent = agent.Agent()
        start_epoch = lane_agent.load_weights("model_last.pth")
        #start_epoch = lane_agent.load_weights("model_100epoch.pth")
    else:
        lane_agent = agent.Agent()
        start_epoch = 0

    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_agent.cuda()

    ##############################
    ## Loop for training
    ##############################
    print('Training loop')
    step = 0
    max_iter = math.ceil(loader.size_train / p.batch_size)
    losses = util_log.AverageMeter()
    acc = 0

    for epoch in range(start_epoch, p.n_epoch):
        lane_agent.training_mode()
        iteration = 0
        losses.reset()
        for inputs, target_lanes, target_h in loader.Generate():
            #training
            #print("epoch : " + str(epoch))
            #print("step : " + str(step))
            loss_p, info_dict = lane_agent.train(inputs, target_lanes, target_h, epoch)
            loss_p = loss_p.cpu().data
            lr = lane_agent.get_lr()
            info_dict.update({'lr':lr})
            
            losses.update(info_dict['lane_detection_loss'])
            
            if iteration %50 == 0:
                util_log.print_log(epoch, iteration, max_iter, info_dict)
            iteration = iteration + 1

            step += 1

        lane_agent.scheduler.step() 
        writer.add_scalar('train/loss', losses.avg, epoch)
        lane_agent.save_model(epoch, 'model_last.pth')

        #evaluation
        if epoch > p.dynamic_sample_at and epoch%3 == 0:
            print('lr', lr)
            print("evaluation")
            lane_agent.evaluate_mode()
            test.evaluation_fast(loader, lane_agent)
            result = evaluation.LaneEval.bench_one_submit("test_result.json", "test_label.json")
            Accuracy = result['Accuracy']
            if acc < Accuracy:
                lane_agent.save_model(epoch, 'model_best.pth')
                acc = Accuracy

            logger.append([epoch, info_dict['lr'], losses.avg, Accuracy]) 

        if int(step)>700000:
            writer.close()
            break

    writer.close()

def testing(lane_agent, test_image, meta, step, loss):
    lane_agent.evaluate_mode()

    #_, _, ti = test.test(lane_agent, np.array([test_image]))
    ti, _, _ = test.test(lane_agent, np.array([test_image]), meta)

    #cv2.imwrite('test_result/result_'+str(step)+'_'+str(loss)+'.png', ti[0])

    lane_agent.training_mode()

    
if __name__ == '__main__':
    log_path ='./savefile/training-summary.txt'
    logger = util_log.Logger(fpath=log_path, title='training-summary')
    logger.set_names(['Epoch', 'LR', 'loss', 'acc'])
    Training(logger)
    logger.close()

