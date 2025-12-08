import jetson_inference, jetson_utils

#Step 1: Load image
#Step 2: Loading the model
#Step 3: Enter image 
#Step 4: Get+display results

cameraFeed = jetson_utils.videoSource("/dev/video0")
model = jetson_inference.imageNet(model="resnet18.onnx", labels="labels.txt", input_blob="input_0", output_blob="output_0")
while True:
    frame = cameraFeed.Capture()

    index, confidence = model.Classify(frame)

    label = model.GetClassLabel(index)

    print(label, confidence*100)