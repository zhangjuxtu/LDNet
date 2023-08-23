#############################################################################################################
##
##  Source code for testing
##
#############################################################################################################

import cv2
import matplotlib.pyplot as plt
import json
import torch
import agent
import numpy as np
from copy import deepcopy
from data_loader import Generator
import time
from parameters import Parameters
import util
from tqdm import tqdm
import csaps

p = Parameters()

###############################################################
##
## Training
## 
###############################################################
def Testing():
    print('Testing')
    
    #########################################################################
    ## Get dataset
    #########################################################################
    print("Get dataset")
    loader = Generator()

    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')
    if p.model_path == "":
        lane_agent = agent.Agent()
    else:
        print('instance model')
        lane_agent = agent.Agent()
        #lane_agent.load_weights(640, "tensor(0.2298)")
        #lane_agent.load_weights('model_last.pth')
        lane_agent.load_weights('model_best.pth')
    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_agent.cuda()

    ##############################
    ## testing
    ##############################
    print('Testing loop')
    lane_agent.evaluate_mode()

    if p.mode == 0 : # check model with test data 
        for _, _, _, test_image, meta in loader.Generate():
            gt_img, init_img, masks, ti, _, _ = test(lane_agent, np.array([test_image]), meta)
            fig = plt.figure()
            ax=fig.add_subplot(2, 2, 1)
            ax.imshow(gt_img[0])
            ax=fig.add_subplot(2, 2, 2)
            ax.imshow(init_img[0])
            ax=fig.add_subplot(2, 2, 3)
            ax.imshow(masks[0])
            ax=fig.add_subplot(2, 2, 4)
            ax.imshow(ti[0])
            plt.show()

    elif p.mode == 1: # check model with video
        cap = cv2.VideoCapture("video_path")
        while(cap.isOpened()):
            ret, frame = cap.read()
            prevTime = time.time()
            frame = cv2.resize(frame, (512,256))/255.0
            frame = np.rollaxis(frame, axis=2, start=0)
            _, _, ti = test(lane_agent, np.array([frame])) 
            curTime = time.time()
            sec = curTime - prevTime
            fps = 1/(sec)
            s = "FPS : "+ str(fps)
            cv2.putText(ti[0], s, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            cv2.imshow('frame',ti[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif p.mode == 2: # check model with a picture
        #test_image = cv2.imread(p.test_root_url+"clips/0530/1492720840345996040_0/20.jpg")
        test_data_set = deepcopy(loader.test_data)

        for i in range(1, 500):
            print('i = ', i)
            #i = 27
            test_image = cv2.imread(p.test_root_url+test_data_set[i]['raw_file'])
            ratio_w = p.x_size*1.0/test_image.shape[1]
            ratio_h = p.y_size*1.0/test_image.shape[0]
            test_image = cv2.resize(test_image, (p.x_size, p.y_size))
            gt_image = deepcopy(test_image)
            test_image = np.rollaxis(test_image, axis=2, start=0)/255.0
            lanes = test_data_set[i]['lanes']
            h_samples = np.array(test_data_set[i]['h_samples'])
            gt_image = util.draw_lanes(gt_image, lanes, h_samples, ratio_w, ratio_h)
            gt_image = np.array([gt_image])

            temp_lanes = []
            temp_h = []
            for j in lanes:
                temp = np.array(j)
                temp = temp*ratio_w
                temp_lanes.append( temp )
                temp_h.append(h_samples*ratio_h )
            test_lanes = [np.array(temp_lanes)]
            test_h = [np.array(temp_h)]
            _, meta = util.make_ground_truth_point(test_lanes, test_h, p)
            meta.update({'gt_image':np.array([gt_image]), 'ratio_w':ratio_w, 'ratio_h':ratio_h, 'index':i})
 
            _, _, masks, out_images, x, y, = test(lane_agent, np.array([test_image]), meta)
            gt_image = meta['gt_image']
            x, y = util.convert_to_original_size(x, y, ratio_w, ratio_h)
            x, y = find_target(x, y, h_samples, ratio_w, ratio_h)
            util.visualize_result(test_image, x, y, ratio_w, ratio_h, gt_image, masks, out_images)

    elif p.mode == 3: #evaluation
        print("evaluate")
        evaluation_fast(loader, lane_agent)


############################################################################
## evaluate on the test dataset
############################################################################
def evaluation(loader, lane_agent):
    result_data = deepcopy(loader.test_data)
    for test_image, target_h, meta in loader.Generate_Test():
        ratio_w, ratio_h, index = meta['ratio_w'], meta['ratio_h'], meta['index']
        #print('evaluation: index:', index)
        gt_image = meta['gt_image']
        _, _, masks, out_images, x, y, = test(lane_agent, np.array([test_image]), meta)
        x, y = util.convert_to_original_size(x, y, ratio_w, ratio_h)
        x, y = find_target(x, y, target_h, ratio_w, ratio_h)
        result_data = write_result_json(result_data, x, y, index)
        #util.visualize_result(test_image, x, y, ratio_w, ratio_h, gt_image, masks, out_images)
    save_result(result_data, "test_result.json")

############################################################################
## evaluate on the test dataset
############################################################################
def evaluation_fast(loader, lane_agent):
    result_data = deepcopy(loader.test_data)
    progressbar = tqdm(range(loader.size_test//p.batch_size))
    for test_image, target_h, ratio_w, ratio_h, testset_index, meta in loader.Generate_test_batch():
        x, y = test_fast(lane_agent, test_image, meta)
        x_ = []
        y_ = []
        for i, j in zip(x, y):
            temp_x, temp_y = util.convert_to_original_size(i, j, ratio_w, ratio_h)
            x_.append(temp_x)
            y_.append(temp_y)
        x_, y_ = find_target_fast(x_, y_, target_h, ratio_w, ratio_h)
        #x_, y_ = fitting(x_, y_, target_h, ratio_w, ratio_h)
        result_data = write_result_json_fast(result_data, x_, y_, testset_index)

        #util.visualize_points_origin_size(x_[0], y_[0], test_image[0], ratio_w, ratio_h)
        #print(gt.shape)
        #util.visualize_points_origin_size(gt[0], y_[0], test_image[0], ratio_w, ratio_h)

        progressbar.update(1)
    progressbar.close()
    save_result(result_data, "test_result.json")

############################################################################
## linear interpolation for fixed y value on the test dataset
############################################################################
def find_target(x, y, target_h, ratio_w, ratio_h):
    # find exact points on target_h
    #print('find_target: x ', x)
    out_x = []
    out_y = []
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h
    for i, j in zip(x,y):
        #print('find_target: i', i)
        #print('find_target: j', j)
        min_y = min(j)
        max_y = max(j)
        temp_x = []
        temp_y = []
        for h in target_h:
            temp_y.append(h)
            if h < min_y:
                temp_x.append(-2)
            elif min_y <= h and h <= max_y:
                for k in range(len(j)-1):
                    if j[k] >= h and h >= j[k+1]:
                        #linear regression
                        if i[k] < i[k+1]:
                            temp_x.append(int(i[k+1] - float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                        else:
                            temp_x.append(int(i[k+1] + float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                        break
            else:
                if i[0] < i[1]:
                    l = int(i[1] - float(-j[1] + h)*abs(i[1]-i[0])/abs(j[1]+0.0001 - j[0]))
                    if l > x_size or l < 0 :
                        temp_x.append(-2)
                    else:
                        temp_x.append(l)
                else:
                    l = int(i[1] + float(-j[1] + h)*abs(i[1]-i[0])/abs(j[1]+0.0001 - j[0]))
                    if l > x_size or l < 0 :
                        temp_x.append(-2)
                    else:
                        temp_x.append(l)
        #print('find_target: temp_x len', len(temp_x))
        #print('find_target: temp_x ', temp_x)
        
        if len(temp_x) != len(target_h):
            continue
            #raise Exception('Format of lanes error.')
        out_x.append(temp_x)
        out_y.append(temp_y)
    
    return out_x, out_y

def find_target_fast(x, y, target_h, ratio_w, ratio_h):
    # find exact points on target_h
    out_x = []
    out_y = []
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h
    count = 0
    for x_batch, y_batch in zip(x,y):
        predict_x_batch = []
        predict_y_batch = []
        for i, j in zip(x_batch, y_batch):
            min_y = min(j)
            max_y = max(j)
            temp_x = []
            temp_y = []
            for h in target_h[count]:
                temp_y.append(h)
                if h < min_y:
                    temp_x.append(-2)
                elif min_y <= h and h <= max_y:
                    for k in range(len(j)-1):
                        if j[k] >= h and h >= j[k+1]:
                            #linear regression
                            if i[k] < i[k+1]:
                                temp_x_ = (i[k+1] - float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k]))
                                if np.isnan(temp_x_):
                                    temp_x_ = np.nan_to_num(temp_x_)
                                #temp_x.append(int(i[k+1] - float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                                temp_x.append(int(temp_x_))
                            else:
                                temp_x_ = (i[k+1] + float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k]))
                                if np.isnan(temp_x_):
                                    temp_x_ = np.nan_to_num(temp_x_)
                                temp_x.append(int(temp_x_))
                            break
                else:
                    if i[0] < i[1]:
                        l_ = (i[1] - float(-j[1] + h)*abs(i[1]-i[0])/abs(j[1]+0.0001 - j[0]))
                        if np.isnan(l_):
                            l_ = np.nan_to_num(l_)
                        l = int(l_)
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
                    else:
                        l_ = (i[1] + float(-j[1] + h)*abs(i[1]-i[0])/abs(j[1]+0.0001 - j[0]))
                        if np.isnan(l_):
                            l_ = np.nan_to_num(l_)
                        l = int(l_)
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
            if len(temp_x) != len(target_h[count]):
                continue
            predict_x_batch.append(temp_x)
            predict_y_batch.append(temp_y)                            
        out_x.append(predict_x_batch)
        out_y.append(predict_y_batch)
        count += 1
    
    return out_x, out_y



def fitting(x, y, target_h, ratio_w, ratio_h):
    out_x = []
    out_y = []
    count = 0
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h

    for x_batch, y_batch in zip(x,y):
        predict_x_batch = []
        predict_y_batch = []
        for i, j in zip(x_batch, y_batch):
            min_y = min(j)
            max_y = max(j)
            temp_x = []
            temp_y = []

            jj = []
            pre = -100
            for temp in j[::-1]:
                if temp > pre:
                    jj.append(temp)
                    pre = temp
                else:
                    jj.append(pre+0.00001)
                    pre = pre+0.00001
            sp = csaps.CubicSmoothingSpline(jj, i[::-1], smooth=0.0001)

            last = 0
            last_second = 0
            last_y = 0
            last_second_y = 0
            for h in target_h[count]:
                temp_y.append(h)
                if h < min_y:
                    temp_x.append(-2)
                elif min_y <= h and h <= max_y:
                    temp_x.append( sp([h])[0] )
                    last = temp_x[-1]
                    last_y = temp_y[-1]
                    if len(temp_x)<2:
                        last_second = temp_x[-1]
                        last_second_y = temp_y[-1]
                    else:
                        last_second = temp_x[-2]
                        last_second_y = temp_y[-2]
                else:
                    if last < last_second:
                        l = int(last_second - float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
                    else:
                        l = int(last_second + float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
            predict_x_batch.append(temp_x)
            predict_y_batch.append(temp_y)
        out_x.append(predict_x_batch)
        out_y.append(predict_y_batch) 
        count += 1

    return out_x, out_y


############################################################################
## write result
############################################################################
#def write_result_json(result_data, x, y, testset_index):
#    print('lane {}, num:{}'.format(testset_index, len(x)))
#    for i in x:
#        result_data[testset_index]['lanes'].append(i)
#        result_data[testset_index]['run_time'] = 1
#    return result_data

def write_result_json(result_data, x, y, testset_index):
    #print('lane {}, num:{}'.format(testset_index, len(x)))
    result_data[testset_index]['lanes'] = x
    result_data[testset_index]['run_time'] = 1
    return result_data

def write_result_json_fast(result_data, x, y, testset_index):
    for index, batch_idx in enumerate(testset_index):
        '''
        for i in x[index]:
            result_data[batch_idx]['lanes'] = 
            result_data[batch_idx]['run_time'] = 1
        '''
        result_data[batch_idx]['lanes'] = x[index]
        result_data[batch_idx]['run_time'] = 1
    return result_data

############################################################################
## save result by json form
############################################################################
def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")

############################################################################
## test on the input test image
############################################################################
def test(lane_agent, test_images, meta=None):

    #result = lane_agent.predict_lanes_test(test_images, meta)
    #confidences, heatmaps, lane_sample, lane_preds = lane_agent.predict_lanes_test(test_images, meta)
    confidences, htmap, lane_sample, lane_preds = lane_agent.predict_lanes_test(test_images, meta)
    kp_heatmap = torch.sigmoid(htmap)
    confidences = confidences[-1]
    gt_images=meta['gt_image'] 
    lane_ind = lane_sample['lane_ind']
    lane_init = lane_sample['lane_init']

    kp_num = 8
    xs, ys, scores = util.decode_kp_hm(kp_hm=kp_heatmap, K=kp_num)
    xs = xs.cpu().data.numpy()
    ys = ys.cpu().data.numpy()
    scores = scores.cpu().data.numpy()

    out_x = []
    out_y = []
    out_images = []
    masks = []
    image_inits = []
    endpoints = []

    num_batch = len(test_images)
    for i in range(num_batch):
        # test on test data set
        image = deepcopy(test_images[i])
        image =  np.rollaxis(image, axis=2, start=0)
        image =  np.rollaxis(image, axis=2, start=0)*255.0
        image = image.astype(np.uint8).copy()
        result_image  = deepcopy(image)
        image_init = deepcopy(image)

        confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy()
        mask = torch.sigmoid(confidences[i]).view(p.grid_y, p.grid_x).cpu().data.numpy()
        mask[mask < 0.65] = 0
        masks.append(mask)

        # extract the endpoints from heatmap
        tmp_x = []
        tmp_y = []
        end_x = []
        end_y = []
        for j2 in range(kp_num):
            if scores[i][0][j2] > 0.3:
                x_value = xs[i][0][j2] * p.resize_ratio
                y_value = ys[i][0][j2] * p.resize_ratio
                end_x.append(x_value)
                end_y.append(y_value)
        result_image = util.draw_points(end_x, end_y, result_image, 8, psize=8)

        lane_int = lane_init[lane_ind == i].cpu().data.numpy() 
        for j2, lane in enumerate(lane_int):
            tmp_x = lane[..., 0] * p.resize_ratio
            tmp_y = lane[..., 1] * p.resize_ratio
            tmp_x = np.clip(tmp_x, 0, 511)
            tmp_y = np.clip(tmp_y, 0, 255)
            image_init = util.draw_points(tmp_x, tmp_y, image_init, 3, psize=1) 
            image_inits.append(image_init)  

        lane_pred = lane_preds[lane_ind == i].cpu().data.numpy() 
        for j2, lane in enumerate(lane_pred):
            tmp_x = lane[..., 0] * p.resize_ratio
            tmp_y = lane[..., 1] * p.resize_ratio
            tmp_x = np.clip(tmp_x, 0, 511)
            tmp_y = np.clip(tmp_y, 0, 255)
            out_x.append(tmp_x)
            out_y.append(tmp_y)
            result_image = util.draw_points(tmp_x, tmp_y, result_image, 1, psize=1) 
        out_images.append(result_image)
 
    #return out_x, out_y,  out_images
    return gt_images, image_inits, masks, out_images, out_x, out_y

def test_fast(lane_agent, test_images, meta=None):

    confidences, htmap, lane_sample, lane_preds = lane_agent.predict_lanes_test(test_images, meta)
    lane_ind = lane_sample['lane_ind']

    out_x = []
    out_y = []

    num_batch = len(test_images)
    for i in range(num_batch):
        lane_pred = lane_preds[lane_ind == i].cpu().data.numpy() 
        out_x_ = []
        out_y_ = []
        for j2, lane in enumerate(lane_pred):
            tmp_x = lane[..., 0] * p.resize_ratio
            tmp_y = lane[..., 1] * p.resize_ratio
            tmp_x = np.clip(tmp_x, 0, 511)
            tmp_y = np.clip(tmp_y, 0, 255)
            out_x_.append(tmp_x)
            out_y_.append(tmp_y)
        out_x.append(out_x_)
        out_y.append(out_y_)
 
    #return out_x, out_y,  out_images
    return out_x, out_y

if __name__ == '__main__':
    Testing()
