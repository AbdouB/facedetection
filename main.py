from imutils import paths
import numpy as np
import sys
import imutils
import pickle
import cv2
import os


def extract_embeddings(embeddings_file, min_confidence):
    protoPath = "model/deploy.prototxt"
    modelPath = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"

    faceDetector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    embedder = cv2.dnn.readNetFromTorch("embedder/openface_nn4.small2.v1.t7")
    imagePaths = list(paths.list_images("images"))

    embeddings = []
    names = []

    for (index, imagePath) in enumerate(imagePaths):
        #grabbing the name
        name = imagePath.split(os.path.sep)[-2]
        #loading the image and resizing it to 600 pixels
        image = imutils.resize(cv2.imread(imagePath), width=600)
        (height, width) = image.shape[:2]
        #create a blob from the image 
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1,
            (300, 300),
            (104, 177, 123), #this is the average pixel intensity across all images in the training set for each of the RGB channels
            False,
            False
        )
        #feed the blob to our face detector
        faceDetector.setInput(imageBlob)
        #retrieving the face(s) found in the image using our deep learning face detector
        detections = faceDetector.forward()

        #make sure at least one face is detected in the image before proceeding
        if len(detections) > 0:
            #getting index of the box with the highest confidence value
            index = np.argmax(detections[0, 0, :, 2])
            #retrieving the confidence value of that box
            confidence = detections[0, 0, index, 2]

            #before proceeding we'll make sure that the level of confidence exceeds our minimum confidence value
            if confidence > min_confidence:
                #getting X, Y of box around the face in our orginal image
                (startX, startY, endX, endY) = (detections[0, 0, index, 3:7] * np.array([width, height, width, height])).astype("int")
                #getting the face (o_o) region of interest
                face = image[startY:endY, startX:endX]
                (faceHeight, faceWidth) = face.shape[:2]
                # ensure the face width and height are sufficiently large
                if faceWidth < 20 or faceHeight < 20:
                    continue

                #constructing a face blob
                faceBlob = cv2.dnn.blobFromImage(face,
                    1.0 / 255,
                    (96, 96),
                    (0, 0, 0),
                    True,
                    False
                )
                #feeding blob to our embedder
                embedder.setInput(faceBlob)
                #retrieving 128-d face embeddings to quantify a face
                embeddingsVectors = embedder.forward()

                #append name and embeddings to our arrays
                names.append(name)
                embeddings.append(embeddingsVectors.flatten())

    f = open(embeddings_file, "wb")
    data = {"embeddings": embeddings, "names": names}
    pickle.dump(data, f)
    f.close()


def main():

    if (len(sys.argv) < 2):
        print("Please specify one of [extract, train, recognize] as arguments")
        sys.exit(1)
    
    if (sys.argv[1] == "extract"):
        extract_embeddings(embeddings_file="serialized/embeddings.pickle", min_confidence=0.5)
    if (sys.argv[1] == "train"):
        print("not yet implemented")
        sys.exit(1)
    if (sys.argv[1] == "recognize"):
        print("not yet implemented")
        sys.exit(1)

if __name__ == "__main__":
    main()