import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from data_pytorch import Data
from rotnet import RotNet
import time
import shutil
import yaml
from resnet import ResNet

parser = argparse.ArgumentParser(description='Configuration details for training/testing rotation net')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image', type=str)
parser.add_argument('--model_number', type=str, required=True)

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for i, (input, target) in enumerate(train_loader):
    	# Use the usual pytorch implementation of training
		input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
		output = model(input_var)
  		loss = criterion(output, target_var)

		optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     

def validate(val_loader, model, criterion):
	model.eval()
    for i, (input, target) in enumerate(val_loader):
    	# Implement the validation. Remember this is validation and not training
    	# so some things will be different.
     	input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
		output = model(input_var)
        loss = criterion(output, target_var)

def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar', filename2='rotationnetmodelbest.pth.tar'):
	torch.save(state, filename)
	#best_one stores whether your current checkpoint is better than the previous checkpoint
    if best_one:
        shutil.copyfile(filename, filename2)

def main():
	n_epochs = config["num_epochs"]
	model = ResNet()
	criterion = nn.CrossEntropy()
	optimizer = optim.Adam(model.parameters())
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
	train_dataset = Data(args[2])	# Come back to this later
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=4,
                                            shuffle=True)
	val_dataset = Data(args[2])
	val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          	batch_size=4,
                                            shuffle=True)
	best_predict = 0
	for epoch in range(n_epochs):
	 	 #TODO: make your loop which trains and validates. Use the train() func
            train(train_loader, model, criterion, optimizer, epoch)
            predict = validate(val_loader, model, criterion)
            
	 	 #TODO: Save your checkpoint
    		is_best = predict > best_predict
            best_predict = max(predict, best_predict)
			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'best_predict': best_predict,
				'optimizer' : optimizer.state_dict(),
        	}, is_best)
    



if __name__ == "__main__":
    main()
