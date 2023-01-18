#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import tempfile

import argparse
import logging
import sys

from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

try:
    from PIL import Image
    import boto3
    
    class ImageDataset(torch.utils.data.Dataset):
        # this class is a modified version of a stackoverflow answer to a similar problem
        # here is the source: https://stackoverflow.com/questions/54003052/how-do-i-implement-a-pytorch-dataset-for-use-with-aws-sagemaker
        def __init__(self, bucket_name='dog-images-mle', transform=None, category='train'):
            self.category = category
            self.bucket_name = bucket_name
            self.s3 = boto3.resource('s3')
            self.bucket = self.s3.Bucket(bucket_name)
            self.files = [obj.key for obj in self.bucket.objects.all() if obj.key.__contains__(category) and not
                          obj.key.__contains__('training')]        
            self.transform = transform
            if transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img_name = self.files[idx]

            # infer the label from the filename
            label = int(img_name.split('/')[1].split('.')[0])-1

            # download the file from S3 to a temporary file locally
            # create the local file name
            obj = self.bucket.Object(img_name)
            tmp = tempfile.NamedTemporaryFile()
            tmp_name = '{}.jpg'.format(tmp.name)

            # actually download from S3 to a local place
            with open(tmp_name, 'wb') as f:
                obj.download_fileobj(f)
                f.flush()
                f.close()
                image = Image.open(tmp_name)

            if self.transform:
                image = self.transform(image)
                label = torch.tensor(label)

            return (image, label)
except:
    pass


def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    #print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")
    logger.info("Test set: Average loss: {:.4f}".format(
            total_loss))
    return


def train(model, train_loader, validation_loader, criterion, optimizer, device, epochs):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    #epochs=2
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.register_loss(criterion)
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
                print("START TRAINING")
                if hook:
                    hook.set_mode(modes.TRAIN)
            else:
                model.eval()
                print("START VALIDATING")
                if hook:
                    hook.set_mode(modes.EVAL)
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 2000  == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
                
                #NOTE: Comment lines below to train and test on whole dataset
                if running_samples>(0.2*len(image_dataset[phase].dataset)):
                    break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

        if loss_counter==1:
            break
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model

def create_data_loader(batch_size, category, transform=None):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    assert category in ['train', 'test', 'valid']
    return torch.utils.data.DataLoader(ImageDataset(category=category,
                                                   transform=transform), 
                                       batch_size=batch_size,
                                       shuffle=False)

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#def main(args):
def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
                    "--batch_size",
                    type=int,
                    default=64,
                    metavar="N",
                    help="input batch size for training (default: 64)",
                )
    parser.add_argument(
                    "--test_batch_size",
                    type=int,
                    default=258,
                    metavar="N",
                    help="input batch size for testing (default: 258)",
                )
    parser.add_argument(
                    "--epochs",
                    type=int,
                    default=2,
                    metavar="N",
                    help="number of epochs to train (default: 2)",
                )
    parser.add_argument(
                    "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
                )
    args = parser.parse_args()
    
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs
    lr = args.lr
    
    
    device = get_device()
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    model=model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
    train_loader = create_data_loader(batch_size, 
                                      category='train')
    validation_loader = create_data_loader(batch_size, 
                                           category='valid')
    model=train(model, train_loader, validation_loader, 
                loss_criterion, optimizer, device, epochs)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test_loader = create_data_loader(test_batch_size, category='test')
    #model, test_loader, citerion, device
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    #path = f's3://dog-images-mle-model/model/dog_classification.pt'
    path = '/opt/ml/model/dog_classification.pt'
    torch.save(model.state_dict(), path)

if __name__=='__main__':
    main()
