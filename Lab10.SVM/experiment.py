import cv2
from train import processFiles, trainSVM
from detector import Detector
from slidingwindow import display_windows
# Replace these with the directories containing your
# positive and negative sample images, respectively.
pos_dir = "Lab10.Assignment/samples/vehicles"
neg_dir = "Lab10.Assignment/samples/non-vehicles"

# Replace this with the path to your test video file.
video_file = "Lab10.Assignment/videos/test_video.mp4"


def experiment1():
    """
    Train a classifier and run it on a video using default settings
    without saving any files to disk.
    """
    # TODO: You need to adjust hyperparameters
    # Extract HOG features from images in the sample directories and 
    # return results and parameters in a dict.
    feature_data = processFiles(pos_dir, neg_dir, recurse=True,
                                color_space='ycrcb',hog_features=True,spatial_features=True,
                                #hog
                                size=(64,64), pix_per_cell=(8,8), cells_per_block=(2,2), hog_bins=16, #block_stride=(2,2),
                                
                                hist_features=True,spatial_size=(20,20),

                                hist_bins=16
                                
                                ) 


    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(C=2500,feature_data=feature_data, output_file=True,output_filename="weight.pkl")


    # TODO: You need to adjust hyperparameters of loadClassifier() and detectVideo()
    #       to obtain a better performance

    # Instantiate a Detector object and load the dict from trainSVM().
    detector = Detector(init_size=(100,100),x_overlap=0.6,y_step=0.01, scale=1.25,x_range=(0.02, 0.98),y_range=(0.45,0.9)).loadClassifier(classifier_data=classifier_data)
  
    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)

    # Start the detector by supplying it with the VideoCapture object.
    # At this point, the video will be displayed, with bounding boxes
    # drawn around detected objects per the method detailed in README.md.
    detector.detectVideo(video_capture=cap, draw_heatmap_size=0.3,min_bbox=(50,50),threshold=120,num_frames=9)


def experiment2():
    display_windows("Lab10.Assignment/Detection_screenshot_24.11.2022.png")
    

if __name__ == "__main__":
    experiment1()
    #experiment2()
    # experiment3 ...


