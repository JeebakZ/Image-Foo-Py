import cv2
import numpy as np
import streamlit as st
import tempfile
import os

st.title("Football Player and Ball Detection")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    temp_video_path = os.path.join(temp_dir.name, uploaded_file.name)
    
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Create a placeholder to display the video
    stframe = st.empty()

    # Read the video
    vidcap = cv2.VideoCapture(temp_video_path)
    success, image = vidcap.read()

    # Initialize the frame counter
    count = 0

    # Define color ranges in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_red = np.array([0, 31, 255])
    upper_red = np.array([176, 255, 255])

    lower_white = np.array([0, 0, 0])
    upper_white = np.array([0, 0, 255])

    while success:
        # Converting the frame from BGR to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define a mask for the green color range
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply the mask to the original image
        res = cv2.bitwise_and(image, image, mask=mask)

        # Convert the masked image from HSV to BGR and then to grayscale
        res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        res_gray = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2GRAY)

        # Define a kernel for morphological operations
        kernel = np.ones((13, 13), np.uint8)
        
        # Apply thresholding to the grayscale image to create a binary image
        thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Apply morphological closing to remove small holes in the binary image
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Set the font for text display
        font = cv2.FONT_HERSHEY_SIMPLEX

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # Detect players based on the bounding rectangle dimensions
            if h >= (1.5) * w and w > 15 and h >= 15:
                player_img = image[y:y + h, x:x + w]
                player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

                # Detect blue jersey players (France)
                mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
                res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
                res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
                nzCount = cv2.countNonZero(res1)

                # Detect red jersey players (Belgium)
                mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
                res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
                res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
                nzCountred = cv2.countNonZero(res2)

                if nzCount >= 20:
                    # Mark blue jersey players as France
                    cv2.putText(image, 'France', (x - 2, y - 2), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

                if nzCountred >= 20:
                    # Mark red jersey players as Belgium
                    cv2.putText(image, 'Belgium', (x - 2, y - 2), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

            # Detect the football based on the bounding rectangle dimensions
            if 1 <= h <= 30 and 1 <= w <= 30:
                player_img = image[y:y + h, x:x + w]
                player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

                # Detect the white color of the football
                mask1 = cv2.inRange(player_hsv, lower_white, upper_white)
                res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
                res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
                nzCount = cv2.countNonZero(res1)

                if nzCount >= 3:
                    # Mark the football
                    cv2.putText(image, 'football', (x - 2, y - 2), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Write the frame to a temporary file
        temp_frame_path = os.path.join(temp_dir.name, f"frame_{count}.jpg")
        cv2.imwrite(temp_frame_path, image)
        count += 1

        # Display the processed frame in Streamlit
        stframe.image(image, channels="BGR")

        # Read the next frame of the video
        success, image = vidcap.read()

    # Release the video capture object
    vidcap.release()

    # Clean up the temporary directory
    temp_dir.cleanup()
