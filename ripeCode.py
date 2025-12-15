import jetson_inference, jetson_utils, random

#Step 1: Load image
#Step 2: Loading the model
#Step 3: Enter image 
#Step 4: Get+display results

cameraFeed = jetson_utils.videoSource("/dev/video0")
showFeed = jetson_utils.videoOutput("webrtc://@:8554/output")
prevLabel = None
model = jetson_inference.imageNet(model="resnet18.onnx", labels="labels.txt", input_blob="input_0", output_blob="output_0")
foodType = {"Apples":"fruit",
            "Banana":"fruit",
            "Bellpepper":"vegetable",
            "Carrot":"vegetable",
            "Cucumber":"vegetable",
            "Mango":"fruit",
            "Oranges":"fruit",
            "Potato":"vegetable",
            "Strawberry":"fruit",
            "Tomato":"vegetable",}

recipes = {"fruit":["Cut and eat","Make smoothie","Make juice"],
           "vegetable":["Make a salad","Make a soup","Make a casserole"]}

while True:
    frame = cameraFeed.Capture()
    
    if frame!=None:
        index, confidence = model.Classify(frame)

        label = model.GetClassLabel(index)

        
        showFeed.Render(frame)
        
        if label != prevLabel:
            print(label, confidence*100)
            if label.startswith("Rotten"):
                print("You should probably throw this away.")
            else:
                print(random.choice(recipes[foodType[label[6:]]]))
                
        prevLabel=label
            