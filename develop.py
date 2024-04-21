from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as tc
#from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
# f^theta



class Morphism(nn.Module):
    def __init__ (self, name = 'Morphisme R^n --> E', dim_E = 1, neurons = 6):
        print(f'[Model] name : {name}')
        print(f'[Model] dim E : {dim_E}')
        print(f'[Model] no. neurons per layers : {neurons}')
        super(Morphism, self).__init__()
        # layers for plus : E --> E
        self.fc1 = nn.Linear(dim_E, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, neurons)
        self.fc4 = nn.Linear(neurons, dim_E)

        # dropout layer
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = self.fc4(x)
        return output
# f inv theta
class InverseMorphism(nn.Module):
    def __init__ (self, name = 'Inverse E --> R^n', dim_E = 1, neurons = 6):
        
        print(f'[Model] name : {name}')
        print(f'[Model] dim E : {dim_E}')
        print(f'[Model] no. neurons per layers : {neurons}')
        super(InverseMorphism, self).__init__()
        # layers for plus : E --> E
        self.fc1 = nn.Linear(dim_E, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3= nn.Linear(neurons, neurons)
        
        self.fc4 = nn.Linear(neurons, dim_E)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = self.fc4(x)
        return output
    
    
class LoiBinaire(nn.Module):
    def __init__ (self, name = 'Loi binaire ExE-->E', dim_E = 1, neurons = 6):
        
        print(f'[Model] name : {name}')
        print(f'[Model] dim E : {dim_E}')
        print(f'[Model] no. neurons per layers : {neurons}')
        super(LoiBinaire, self).__init__()
        # layers for plus : ExE --> E
        self.fc1 = nn.Linear(2 * dim_E, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, neurons)
        self.fc4 = nn.Linear(neurons, dim_E)
    def forward(self, x, y):
        z = torch.cat([x,y], axis=1) # [K,d], [K,d] ---> [K, 2*d]
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        output = self.fc4(z)
        return output
# scalaire product of structure

class LoiScalaire(nn.Module):
    def __init__ (self, name = 'Loi Scalaire RxE-->E', dim_E = 1, neurons = 6):
        
        print(f'[Model] name : {name}')
        print(f'[Model] dim E : {dim_E}')
        print(f'[Model] no. neurons per layers : {neurons}')
        super(LoiScalaire, self).__init__()
        # layers for plus : KxE --> E
        

        self.fc1 = nn.Linear(dim_E, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, neurons)
        self.fc4 = nn.Linear(neurons, dim_E)
        
        # alpha est un  scalaire,  dim_E est la dimension de l'espace E
        
    def forward(self, alpha, x):
        z = alpha * x # [K,1], [K,d] ---> [K, d]
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        output = self.fc4(z)
        return output
    
    
class Vect_space(nn.Module):
    def __init__ (self, K,  dim_E = 1 , neurons = 6 , name = 'Groupe (E,+)'):
        super(Vect_space, self).__init__()
        self.f    = Morphism(dim_E = dim_E, neurons = neurons)
        self.fi   = InverseMorphism(dim_E = dim_E, neurons = neurons)
        self.plus = LoiBinaire(dim_E = dim_E, neurons = neurons)
        self.scalaire = LoiScalaire(dim_E = dim_E, neurons = neurons)
        # losses
        self.loss_1 = lambda x, y : torch.linalg.vector_norm(self.plus(x , y) - self.f( self.fi(x) + self.fi(y)) )**2
        self.loss_2 = lambda alpha, x : torch.linalg.vector_norm(self.scalaire(alpha , x) - self.f( alpha*self.fi(x)) )**2

        #  Total loss can be weighted 
        self.loss = lambda x, y, alpha : self.loss_1(x, y) + self.loss_2(alpha, x)
        
    def train(self, X, Y,alpha, optimizer, epoch):
        self.f.train()
        self.fi.train()
        self.plus.train()
        self.scalaire.train()
        losses=[]
        for i in range(epoch):
            optimizer.zero_grad()
            L1 = self.loss_1(X, Y)
            L2 = self.loss_2(alpha, X)

            loss = L1 + L2
            #loss = loss.mean()
            if i % 10 == 0:
               print('Epoch {}/{} -\t Loss 1: {:.6f}\t Loss 2: {:.6f}\t Total Loss: {:.6f}'.format(i, epoch, L1.item(), L2.item(), loss.item()))
            
            loss.backward(retain_graph=True)
            losses.append(loss.item())
            optimizer.step()
        return losses
    def forward(self, x):
        return self.fi(x)
            
            
            
def test(self, test_loader):
        pass

# Dataset generation
def line(K): #a=2, b=3):
    
    X = 0.3*torch.randn(K, 2).requires_grad_(False)
    Y = 0.3*torch.randn(K, 2).requires_grad_(False)
    alpha = torch.randn(K, 1).requires_grad_(False)
    epislon = 0.1
    X[:,1] = X[:,0] + epislon * torch.sin(X[:,0] / epislon ) # **3 + a*X[:,0] + b
    Y[:,1] = Y[:,0] + epislon * torch.sin(Y[:,0] / epislon) #**3 + a*Y[:,0] + b
    return X, Y, alpha
#def dataset_parabola(K, a=0.5, b=0.5):
    #X = 0.3*torch.randn(K, 2).requires_grad_(False)
    #Y = 0.3*torch.randn(K, 2).requires_grad_(False)
    #X[:,1] = (X[:,0]-a)**3 + b
    #Y[:,1] = (Y[:,0]-a)**3 + b
    #return X, Y
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                    help='disables macOS GPU training')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--dry-run', action='store_true', default=False,
#                     help='quickly check a single pass')
args = parser.parse_args()
torch.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()
if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
    
# Training datasets
dim = 2
K = 200
X = tc.rand(K, 2)
Y = tc.rand(K, 2)
alpha = tc.rand(K, 1)
#X, Y, alpha = elliptic_curve(K)
X, Y, alpha = line(K)
beta = 1.0  # replace with your value of beta
#f = tc.vmap(lambda X : X[0]**2)

f = tc.vmap(lambda X : (1/beta) * tc.exp(X[0]))
fX = f(X)
fY = f(Y)

# on initalise le Groupe 
G = Vect_space(K, dim_E = dim, neurons = 64)
optimizer = optim.Adadelta(list(G.parameters()), lr=0.1)

losses = G.train(X, Y, alpha, optimizer, args.epochs)
#plt.plot(X[:, 0], X[:, 1], 'x', label='train X')
#plt.plot(Y[:, 0], Y[:, 1], 'o', label='train Y')
#plt.plot(losses);
# Plot training data X
plt.figure(figsize=(6, 4))
plt.plot(X[:, 0], X[:, 1], 'x', label='train X')
plt.title('Training Data X')
plt.legend()
plt.show()
# Plot training data Y
plt.figure(figsize=(6, 4))
plt.plot(Y[:, 0], Y[:, 1], 'o', label='train Y')
plt.title('Training Data Y')
plt.legend()
plt.show()
# Plot losses
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.title('Losses')
plt.show()



Xe = G.fi(G.f(X))
if args.save_model:
    print('Saving model...')
    torch.save(G.state_dict(), "nas_plus.pt")











