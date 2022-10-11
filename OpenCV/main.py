import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('test_1.mp4') # 동영상 제목으로
ym_per_pix = 30 / 720
xm_per_pix = 3.7 / 720

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out1 = cv2.VideoWriter('C:\\Users\\rkdgu\\Desktop\\pythonProject\\opencv_youtube.mp4', fourcc, 20.0, frame_size)

def wrapping(image):
    (h, w) = (image.shape[0], image.shape[1]) #928,1916
    #print(h,w)
    #좌표 = 좌상 -> 좌하 -> 우상 -> 우하
    source = np.float32([[927,735],[670,911],[1041,735],[1351,911]])
    destination = np.float32([[100, 0], [100, 800], [500, 0], [500, 800]])

    #변환 전 좌표를 이미지에 표시
    red_color = (255,0,0)# blue = 255,0,0
    green_color = (0,255,0)# green = 0,255,0
    blue_color = (0,0,255)# red = 0,0,255
    black_color = (0,0,0)# black 0,0,0
    # 마지막 -1 = 내부가 채워짐

    #cv2.circle(image, (927, 735), 5, red_color, -1)  # 바꾼빨간점 좌하단
    #cv2.circle(image, (1041, 735), 5, green_color, -1)  # 바꾼초록점 좌상단
    #cv2.circle(image,(670,911),15,blue_color,-1) #
    #cv2.circle(image,(1351,911),15,black_color,-1) #


    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(destination, source) # 마지막에 wrapping된 이미지를 다시 원근감을 주기 위해 반대의 matrix 값을 저장하는 변수
    _image = cv2.warpPerspective(image, transform_matrix, (600, 1100))
    #plt.subplot(121),plt.imshow(image),plt.title('origin')
    #plt.subplot(122),plt.imshow(_image),plt.title('perspective')
    #plt.show()

    return _image, minv

def color_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lower = np.array([0, 50, 30])
    upper = np.array([255, 255, 255])

    yellow_lower = np.array([0, 85, 81])
    yellow_upper = np.array([190, 255, 255])

    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def roi(image):
    x = int(image.shape[1])
    y = int(image.shape[0])

    _shape = np.array(
        [[int(0.1*x), int(y)], [int(0.1*x), int(0.1*y)], [int(0.4*x), int(0.1*y)], [int(0.4*x), int(y)], [int(0.7*x), int(y)], [int(0.7*x), int(0.1*y)],[int(0.9*x), int(0.1*y)], [int(0.9*x), int(y)], [int(0.2*x), int(y)]])

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)


    return masked_image



def plothistogram(image):
    histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
    midpoint = np.int64(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint
    #print(leftbase,rightbase)
    return leftbase, rightbase

def slide_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    nwindows = 7
    window_height = np.int32(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100
    minpix = 50
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2

    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height
        win_y_high = binary_warped.shape[0] - w * window_height
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        # cv2.imshow("out_img", out_img)

        if len(good_left) > minpix:
            left_current = np.int32(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int32(np.mean(nonzero_x[good_right]))

    left_lane = np.concatenate(left_lane)
    right_lane = np.concatenate(right_lane)

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)
    rtx = np.trunc(right_fitx)

    out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color = 'yellow')
    #plt.plot(right_fitx, ploty, color = 'yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.show()

    ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}

    return ret

def draw_lane_lines(original_image, warped_image, Minv, draw_info):
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74)) #cv2.fillpoly(img,pts,color[,lineType[,shift[,offset])
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)

    return pts_mean, result

while True:
    retval, img = cap.read()
    if not retval:
        break

    wrapped_img, minverse = wrapping(img)
    #cv2.imshow('wrapped', wrapped_img)

    w_f_img = color_filter(wrapped_img)
    #cv2.imshow('w_f_img', w_f_img)

    w_f_r_img = roi(w_f_img)
    #cv2.imshow('w_f_r_img', w_f_r_img)

    _gray = cv2.cvtColor(w_f_r_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(_gray, 185, 255, cv2.THRESH_BINARY) #(src,thresh,maxval,type) ->retval, dst// src->input image// thresh-> 임계값// maxval-> 임계값을 넘었을때 적용할 value // type -> threshholding type
    #cv2.imshow('threshold', thresh)

    leftbase, rightbase = plothistogram(thresh)
    # 100에 왼쪽차선, 500에 오른쪽 차선

    ## histogram 기반 window roi 영역
    draw_info = slide_window_search(thresh, leftbase, rightbase)

    ## 원본 이미지에 라인 넣기
    meanPts, result = draw_lane_lines(img, thresh, minverse, draw_info)
    # cv2.imshow("result", result)

    out1.write(result)

    key = cv2.waitKey(25)
    if key == 27:
        break

if cap.isOpened():
    cap.release()

cv2.destroyAllWindows()