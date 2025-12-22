# Ripeness-Detector
Many people don’t know what to do with a fruit once they buy it, and this causes them to go bad before they get a chance to eat it. This AI helps users think about what to do with these fruits and make the most out of their purchase. This can help reduce food waste and reduce total grocery costs overall. I made this because I thought it was interesting to see how an AI would handle this task.

<img width="667" height="319" alt="Screenshot 2025-12-21 at 7 06 38 PM" src="https://github.com/user-attachments/assets/5fe3cb2e-515b-45ce-a069-a5d8955475c5" />

## The Algorithm

The data came from this dataset: https://www.kaggle.com/datasets/ibneshihab/fresh-rotten-fruiters-and-vegetables-classification

Once I had the dataset, I removed the classes of fresh and rotten bittergroud, capsicum, and okra. Then, I balanced out the classes so that a maximum of 550 images were put into training for each class. After training over 16 epochs, the AI had an accuracy of 80%.
By holding up the fruit or vegetable to a connected camera, the AI will detect both the type of fruit/vegetable and whether it is fresh or rotten. From there, the AI will give you a random suggestion based on whether the food item is a fruit or a vegetable.

## Running This Project
1. Install Jetson libraries
2. Attach a camera to your jetson
3. Change directories to "Ripeness-Detector"
4. Grab food item (fruit/vegetable)
5. Run python3 ripeCode.py
6. Stop the code using Control + C
7. Access your feed in the feed.mp4 file
