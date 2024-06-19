import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops
import sys
import shutil
import os

# Store the initial value of widgets in session state
if "choice" not in st.session_state:
    st.session_state.choice = "subjek1"

st.title('Vascular Health Characterization')

st.header('Video Processing')

subjek = []
for i in range(1,50):
    subjek.append('subjek'+str(i))

choice_video = st.selectbox('Pick one sample video', subjek, key='choice')
st.write('Here video of', st.session_state.choice)
path_video = "./raw_video/%s.mp4" % st.session_state.choice

try :
    st.video(path_video)
    vidcap1 = cv2.VideoCapture(path_video)
    duration1 = vidcap1.get(cv2.CAP_PROP_POS_MSEC)
    frame_count1 = vidcap1.get(cv2.CAP_PROP_FRAME_COUNT)
    st.write('frame_count = ', frame_count1)
except:
    e = sys.exc_info()[0]
    print(st.write(e))

start_processing = st.button('Start Processing')

if start_processing:
    st.success('Image Processing Started')
    expander = st.expander("See process status")
    progress_segment = st.progress(0)
    
    # folder = './output_images'
    # for filename in os.listdir(folder):
    #     file_path = os.path.join(folder, filename)
    #     try:
    #         if os.path.isfile(file_path) or os.path.islink(file_path):
    #             os.unlink(file_path)
    #         elif os.path.isdir(file_path):
    #             shutil.rmtree(file_path)
    #     except Exception as e:
    #         print('Failed to delete %s. Reason: %s' % (file_path, e))

    vidcap = cv2.VideoCapture(path_video)
    duration = vidcap.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    success,image = vidcap.read()

    arr_height_artery = []
    height_artery = 0

    count = 1
    while success:
        expander.success(f"Read frame {count}")
        progress_bar = expander.progress(0)
        
        # save input frames
        cv2.imwrite(f"./raw_images/frame{count}.jpg", image)
        segmented_image = image
        raw_image = image.copy()
        progress_bar.progress(10)

        # crop image
        cropped_value = [109, 204]
        cropped_image = image[cropped_value[0]:(image.shape[0]-148), cropped_value[1]:(image.shape[1]-204)]
        progress_bar.progress(20)

        # circle detection using Circular Hough Transform
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.medianBlur(gray_image, 3)
        rows = blurred_image.shape[0]
        dist_circle = rows/4
        paramh1= 100
        paramh2= 30
        paramh3= 25
        paramh4= 70
        circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, 1,
                                    dist_circle,
                                    param1=paramh1, param2=paramh2,
                                    minRadius=paramh3, maxRadius=paramh4)
        progress_bar.progress(30)

        if circles is not None:
            detected_circles = np.uint16(np.around(circles))

            # choose circle by dark region
            result = []
            for z in range(detected_circles.shape[1]):
                side = round(detected_circles[0][z][2]*np.sqrt(2))

                start_x = detected_circles[0][z][0]-round(side/2)
                start_y = detected_circles[0][z][1]-round(side/2)

                sum = 0
                for x in range(start_x,start_x+side,1):
                    for y in range(start_y,start_y+side,1):
                        if y < blurred_image.shape[0] and x < blurred_image.shape[1]:
                            sum = sum + blurred_image[y][x]

                result.append(sum/(side*side))
            
            result_sorted = result.copy()
            result_sorted.sort()
            index_circle = result.index(result_sorted[0])
            used_detected_circles = detected_circles[0][index_circle]
            progress_bar.progress(40)

            # change all red pixel to white
            # progress_cleared = expander.progress(0)
            count_cleared = 0
            cleared_image = cropped_image.copy()
            for x in range(cleared_image.shape[1]):
                for y in range(cleared_image.shape[0]):
                    # progress_cleared.progress(count_cleared/(cleared_image.shape[0]*cleared_image.shape[1]/100))
                    if list(cleared_image[y][x]) ==  [0,0,255] :
                        cleared_image[y][x] = (255,255,255)
                    count_cleared = count_cleared + 1
            progress_bar.progress(45)

            # draw red circle hough transform
            x = used_detected_circles[0]
            y = used_detected_circles[1]
            r = used_detected_circles[2]
            cv2.circle(cleared_image, (x, y), r, (0, 0, 255), 2)
            progress_bar.progress(50)

            # localized circle
            localized_image = cropped_image.copy()
            for x in range(cleared_image.shape[1]):
                for y in range(cleared_image.shape[0]):
                    if list(cleared_image[y][x]) ==  [0,0,255] :
                        break
                    localized_image[y][x] = (255,255,255)
            for x in range(cleared_image.shape[1]):
                for y in reversed(range(cleared_image.shape[0])):
                    if list(cleared_image[y][x]) ==  [0,0,255] :
                        break
                    localized_image[y][x] = (255,255,255)
            progress_bar.progress(60)

            # contrast streching (adjust gain and bias)
            adjusted_image = localized_image.copy()
            adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=3, beta=80)
            progress_bar.progress(70)

            # otsu threshoding
            gray_adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
            blurred_adjusted_image = cv2.GaussianBlur(gray_adjusted_image,(5,5),0)
            otsu_value,otsu_image = cv2.threshold(blurred_adjusted_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            progress_bar.progress(80)

            # using region_prop measure diameter circle
            label_image = label(otsu_image)
            area = 0
            for region in regionprops(label_image):
                # take largest region
                if region.area > area:
                    area = region.area
                    taken_region = region
            # draw rectangle around segmented 
            minr, minc, maxr, maxc = taken_region.bbox
            minr, minc, maxr, maxc = minr + cropped_value[0], minc + cropped_value[1], maxr + cropped_value[0], maxc + cropped_value[1]
            segmented_image = raw_image.copy()
            cv2.rectangle(segmented_image, (minc, minr), (maxc, maxr), (0, 0, 255), 2)
            progress_bar.progress(90)

            pixel_value = 30/cropped_image.shape[0]
            height_artery = 30/cropped_image.shape[0]*(maxr-minr)
            width_artery = 30/cropped_image.shape[0]*(maxc-minc)
            cv2.putText(segmented_image, 'height=' + str(maxr-minr)+ 'px' + '(' + str(round(height_artery,3)) + 'mm)', (minc, maxr + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(segmented_image, 'width=' + str(maxc-minc)+ 'px' + '(' + str(round(width_artery,3)) + 'mm)', (minc, maxr + 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(segmented_image, 'params_hough_circle=' + str((round(dist_circle,3), paramh1, paramh2, paramh3, paramh4)), (segmented_image.shape[1] - 400, segmented_image.shape[0] -40), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(segmented_image, 'pixel_value=' + str(pixel_value) + 'mm/px', (segmented_image.shape[1] - 400, segmented_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(segmented_image, 'frame=' + str(count), (maxc,maxr), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            progress_bar.progress(98)

        # save artery height if circle none use last height
        arr_height_artery.append(height_artery)
        with open('height_artery.txt', 'a') as f:
            f.write(str(height_artery))
            f.write('\n')

        # save segmented image
        cv2.imwrite(f"./output_images/frame{count}.jpg", segmented_image)
        progress_bar.progress(100)
        # st.image(segmented_image)
        progress_segment.progress(count/frame_count)

        success,image = vidcap.read()

        # if count == 1:
        #     break
            
        count = count + 1
    vidcap.release()
    st.success('Image Processing Completed')

img = cv2.imread(f"./output_images/frame1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
st.image(img)

show_output_video = st.button('Show Output Video')
if show_output_video:
    progress_video = st.progress(0)
    height, width, channels = img.shape
    OUTPUT_FILE = 'output_video.mp4'

    # output video configuration
    FPS = 31
    WIDTH = width
    HEIGHT = height

    # define video writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

    for i in range(1,400):
        filename = f"./output_images/frame{i}.jpg"
        img = cv2.imread(filename)
        writer.write(img)
        progress_video.progress(i/400)

    writer.release()
    st.video('./output_video.mp4')


st.header('Graph Sensor')

file = f"./raw_sensor/{choice_video}.txt"
with open(file) as f:
    lines = f.readlines()
    y = [float(line.split()[1])/100/0.00045/1000 for line in lines]
    y1 = [float(line.split()[1]) for line in lines]
    x1 = [line.split()[0] for line in lines]

df = pd.DataFrame(list(zip(x1, y1)), columns =['Timestamp', 'Contact Force (kPA)'])
# df.head(20)
st.dataframe(df)

end_range_sensor = st.number_input('Pick end range sensor', value= len(y1))
initial_range_sensor = st.number_input('Pick initial range sensor', value= int(end_range_sensor-frame_count1))
y = y[initial_range_sensor:end_range_sensor]

limit = len(y) - 1
x = np.arange(0,limit+1,1)
x_sensor, y_sensor = x, y

fig, ax = plt.subplots()
ax.plot(x, y)
ax.scatter(x, y, c='#d62728')
ax.set_xlabel('index of frames')
ax.set_ylabel('contact force (kPa)')
st.pyplot(fig)


st.header('Sensor X Vessel Radius')

with open('./height_artery.txt') as f:
    lines = f.readlines()
    y = [float(line) for line in lines]
limit = len(y) - 1
x = np.arange(0,limit+1,1)

df = pd.DataFrame(list(zip(y)), columns =['Radius Artery'])
st.dataframe(df)

# fig, ax = plt.subplots()
# # ax.plot(x, y)
# ax.scatter(x, y, c='#d62728')
# st.pyplot(fig)

pick_frame_a = st.slider('Pick frame a', 1, len(y_sensor))
pick_frame_b = st.slider('Pick frame b', 1, len(y_sensor))

col1, col2 = st.columns(2)

with col1:
    y2 = np.arange(0,limit+1,1)
    x2 = np.full(limit+1, pick_frame_a)
    x3 = np.full(limit+1, pick_frame_b)
    x, y = x_sensor, y_sensor
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.scatter(x, y, c='#d62728')
    ax.set_xlabel('index of frames')
    ax.set_ylabel('contact force (kPa)')
    ax.set_ylim(0,25)
    # ax.set_xlim(0,650)
    ax.plot(x2,y2)
    ax.plot(x3,y2)
    st.pyplot(fig)

with col2:
    try :
        st.image(f"./sample_output/{choice_video}/frame{pick_frame_a}.jpg")
        st.image(f"./sample_output/{choice_video}/frame{pick_frame_b}.jpg")
    except:
        e = sys.exc_info()[0]
        print(st.write(e))

st.write(f"Current force at frame {pick_frame_a}: {round(y[pick_frame_a], 3)} kPa (orange line)")
st.write(f"Current force at frame {pick_frame_b}: {round(y[pick_frame_b], 3)} kPa (green line)")