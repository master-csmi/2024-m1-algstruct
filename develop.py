from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as tc
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
    def __init__ (self, K, dim_E = 1 , neurons = 6 , name = 'Groupe (E,+)'):
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
            L2 = self.loss_2(alpha, X)#
            loss = L1 + L2
            loss = loss.sum()
            if i % 10 == 0:
                print('Epoch {}/{} -\t Loss 1: {:.6f}\t Loss 2: {:.6f}\t Total Loss: {:.6f}'.format(i, epoch, L1.item(), L2.item(), loss.item()))
            loss.backward(retain_graph=True)
            losses.append(loss.item())
            optimizer.step()
        return losses
    
    
def test(self, test_loader):
    pass
        
    
    
    
    
# Dataset generation
def line(K, epsilon): 
    X = torch.randn(K, 2).requires_grad_(False)
    Y = torch.randn(K, 2).requires_grad_(False)
    alpha = torch.randn(K, 1).requires_grad_(False)
    X[:,1] = X[:,0] + epsilon * torch.sin(X[:,0] / epsilon )
    Y[:,1] = Y[:,0] + epsilon * torch.sin(Y[:,0] / epsilon)
    
    return X, Y, alpha




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
args, unknown = parser.parse_known_args()
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


# dim = 2
# K = 1000
# X, Y, alpha = line(K)
# G = Vect_space(K, dim_E = dim, neurons = 32 )
# train_size = 500
# indices = torch.randperm(K)

# # on sépare les données en données d'entrainement et données de test
# train_indices = indices[:train_size]
# test_indices = indices[train_size:]
# alpha_indices = indices[:train_size]


# # on sélectionne maintenant les données d'entrainement et de test
# X_train = X[train_indices]
# Y_train = Y[train_indices]
# alpha_train = alpha[alpha_indices]

# # pour les données de test
# X_test = X[test_indices]
# Y_test = Y[test_indices]
# alpha_test = alpha[test_indices]




# on entraine le modèle
# optimizer = optim.Adadelta(list(G.parameters()), lr=0.1)
# losses = G.train(X, Y, alpha , optimizer, args.epochs)


# plt.figure(figsize=(6, 4))
# plt.plot(X_train[:, 0], X_train[:, 1], 'o', label='train X')
# plt.title('Training Data X')
# plt.xlim(left=0)  # Définit la limite inférieure de l'axe x à 0
# plt.ylim(bottom=0)  # Définit la limite inférieure de l'axe y à 0
# plt.legend()
# plt.show()


# plt.figure(figsize=(6, 4))
# plt.plot(losses)
# plt.title('Losses')
# plt.show()


# # testons le modèle avec les x_test et y_test
# Xtest = G.f(G.fi(X_test))


# Xtest = Xtest.detach().numpy() 


# plt.figure(figsize=(6, 4))
# plt.plot(Xtest[:,0], Xtest[:,1], '.', label="x-test")
# plt.title('Test Data X')


# plt.legend()
# plt.show()





dim = 2
K = 1000
epsilon = 0.1

V = Vect_space(K, dim_E = dim, neurons=32)
# maintenant que le modèle a été entrainé 



# maintenant initalisons le jeu de données 
X_train, Y_train, alpha  = line (K, epsilon)




# on entraine le modèle
optimizer = optim.Adadelta(list(V.parameters()), lr=0.1)
losses = V.train(X_train, Y_train, alpha,  optimizer, args.epochs)


plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.title('Losses')
plt.show()




plt.figure(figsize=(6, 4))
plt.plot(X_train[:,0], X_train[:,1], '.', label=("x+εsin(x/ε)"))

plt.title('Graphique de x + εsin(x/ε)')

plt.xlim(left=0)  # Définit la limite inférieure de l'axe x à 0
plt.ylim(bottom=0)  # Définit la limite inférieure de l'axe y à 0


X_t , _, _ = line(500, 0.1)
# compare moi le X_t et le X_train 
print(torch.equal(X_t, X_train))





X_test = V.fi(X_t)  

X_test = V.f(X_test) 


X_test = X_test.detach().numpy()   # result 



plt.figure(figsize=(6, 4))
plt.plot(X_test[:,0], X_test[:,1], '+', label="x-test")
plt.title('Test Data X')



plt.legend()
plt.show()





# Xe = G.fi(X)
if args.save_model:
    print('Saving model...')
    torch.save(model.state_dict(), "nas_plus.pt")










