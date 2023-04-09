import torch
import torchvision.models as models
import torchvision.transforms as transforms

#load pretrained mdoel
model = models.resnet50(pretrained=True)

#set the model to evaluation mode
model.eval()

#set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

#normalize image
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
    ])
    
#define function to extract features from ROI
def extract_features(image, bbox):
    #extract roi from image
    roi = image.crop(bbox)
    
    
    #apply transform and move to gpu if available
    roi_tensor = transform(roi).unsqueeze(0).to(device)
    
    #pass roi through resnet50 to extract features
    features = model(roi_tensor)
    
    #return feature vector
    return  features.squeeze().cpu().detach().numpy()