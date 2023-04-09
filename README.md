# person_tracking_and_face_detection

Detecting and tracking person using deep_sort with yolov3. Resnet50 and pretrained weights are used for feature extraction.
Another feature of my implementation includes face detection with MTCNN

I have no gpu PC.So I implement this in google colab. 

To run project.ipynb, dowload all the folders and upload to your google drive.

In your google colab, folder destination should be like this

<details>
<summary>drive</summary>

- [MyDrive](file1.md)
- [deep_sort_pytorch-master]
- [PyTorch-YOLOv3-master]
- [yolov3.weights]
- [feature_extraction.py]
- [project.ipynb]
</details>

Here is the demo :)

![ezgif-2-eebcc243a2](https://user-images.githubusercontent.com/30900212/230763482-ff01f5ac-bf8f-47bf-b7ed-95d274b018fa.gif)

Code
Deep_sort_pytorch: https://github.com/ZQPei/deep_sort_pytorch
YOLOv3: https://github.com/eriklindernoren/PyTorch-YOLOv3
Resnet50 from torchvision.models

Paper
Deep_sort_pytorch: https://arxiv.org/abs/1703.07402
YOLOv3: https://arxiv.org/abs/1804.02767
Resnet: https://arxiv.org/abs/1512.03385
