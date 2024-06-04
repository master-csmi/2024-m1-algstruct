from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as tc
from torch.optim.lr_scheduler import StepLR
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
    


# Première proprieté du théorème 
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
    



# Deuxième proprieté du théorème
class LoiScalaire(nn.Module):
    def __init__ (self, name = 'Loi Scalaire RxE-->E', dim_E = 1, neurons = 6):
        
        print(f'[Model] name : {name}')
        print(f'[Model] dim E : {dim_E}')
        print(f'[Model] no. neurons per layers : {neurons}')
        super(LoiScalaire, self).__init__()
        # layers for scaler : KxE --> E
        

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



"""Implementation of the theorem of the vector space with the Train"""
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
            L1 = self.loss_1(X, Y)
            L2 = self.loss_2(alpha, X)
            loss = L1 + L2


            if i % 200 == 0:
                print('Epoch {}/{} -\t Loss 1: {:.6f}\t Loss 2: {:.6f}\t Total Loss: {:.6f}'.format(i, epoch, L1.item(), L2.item(), loss.item()))
            
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        return losses


    def test(self, test_loader,X):

        B,C,alpha = test_loader()

        B_np, C_np, alpha_np =  B.numpy(), C.numpy(), alpha.numpy()

        # Create a DataFrame
        print('Les données Test pour tester le modèle')
        df = pd.DataFrame({
            'B_x': B_np[:, 0],
            'B_y': B_np[:, 1],
            'C_x': C_np[:, 0],
            'C_y': C_np[:, 1],
            'alpha': alpha_np[:, 0]
                        })
        print(df)

        """
            Examinons le comportement des fonctions f, f^{-1}, +, 
            et \odot sur les données de test
        """

        # First propriété
        XXBC =  G.f(G.fi(B) + G.fi(C))
        YYBC = G.plus(B, C)

        # Second propriété
        PXBC =  G.f(alpha * G.fi(C))
        PYBC = G.scalaire(B, C)

        # Calculer les erreur L2 et inf
        Sum_erreur_list_l2 = [torch.norm(XXBC[i] - YYBC[i], p=2).item() for i in range(len(XXBC))]
        Sum_erreur_list_inf = [torch.norm(XXBC[i] - YYBC[i], p=float('inf')).item() for i in range(len(XXBC))]


        dot_erreur_list_l2 = [torch.norm(PXBC[i] - PYBC[i], p=2).item() for i in range(len(XXBC))]
        dot_erreur_list_inf = [torch.norm(PXBC[i] - PYBC[i], p=float('inf')).item() for i in range(len(XXBC))]





        # Ajouter les listes comme nouvelles colonnes dans le DataFrame

        XXBC_list = [x.detach().numpy() for x in XXBC]
        YYBC_list = [y.detach().numpy() for y in YYBC]
        
        # Ajouter la colonne 'Erreur' à la fin du DataFrame


        # Convertir les erreurs en notation scientifique
        Sum_erreur_list_l2 = ['{:.1e}'.format(erreur) for erreur in Sum_erreur_list_l2]
        Sum_erreur_list_inf = ['{:.1e}'.format(erreur) for erreur in Sum_erreur_list_inf]



        ########################
        print('resultat obtenu pour le test de la première propriété du théorème')
        frame_sum = pd.DataFrame({
                'f($f^{-1}(B) + f^{-1}(C)$)': XXBC_list,
                'B ⊕ C': YYBC_list,
                'L^2 erreur': Sum_erreur_list_l2,
                'inf erreur': Sum_erreur_list_inf
                        })
        print(frame_sum)
        



        # Convertir les erreurs en notation scientifique
        # Ajouter les listes comme nouvelles colonnes dans le DataFrame

        PXBC_list = [x.detach().numpy() for x in PXBC]
        PYBC_list = [y.detach().numpy() for y in PYBC]
        dot_erreur_list_l2 = ['{:.1e}'.format(erreur) for erreur in dot_erreur_list_l2]
        dot_erreur_list_inf = ['{:.1e}'.format(erreur) for erreur in dot_erreur_list_inf]
        
        ###########

        # plot sum 
        indice = random.sample(range(B.shape[0]),5)
        print('the plot of sum')
        for i in indice:
            plt.figure()
            plt.plot(X[:, 0], X[:, 1], '.', linewidth = 0.01) 
            plt.plot(B[i, 0], B[i, 1],   'x', color='red',  label=r'$B_{' + str(i+1) + '}$')  
            plt.annotate(r'$B_{' + str(i+1) + '}$', (B[i, 0], B[i, 1] - 0.01))
            plt.plot(C[i, 0], C[i, 1], 'x',  color='black', label=r'$C_{' + str(i+1) + '}$', )
            plt.annotate(r'$C_{' + str(i+1) + '}$', (C[i, 0], C[i, 1] + 0.01))
            # Tracer le point XXBC[i]
            plt.plot(XXBC[i, 0].detach().numpy(), XXBC[i, 1].detach().numpy(), 'o', color='yellow', label=r'f($f^{-1}(B) + f^{-1}(C)$)')
            plt.annotate(r'f($f^{-1}(B) + f^{-1}(C)$)', (XXBC[i, 0].detach().numpy(), XXBC[i, 1].detach().numpy() - 0.1))

            # Tracer le point YYBC[i]
            plt.plot(YYBC[i, 0].detach().numpy(), YYBC[i, 1].detach().numpy(), 'x', color='purple', label=r'$B \oplus C$')
            plt.annotate(r'$B \oplus C$', (YYBC[i, 0].detach().numpy(), YYBC[i, 1].detach().numpy() + 0.01))
            # Ajouter une légende au subplot    

            plt.title(r'$B_{' + str(i+1) + '}$' + f': ({B[i, 0]:.3f}, {B[i, 1]:.3f}), ' + r'$C_{' + str(i+1) + '}$' + f': ({C[i, 0]:.3f}, {C[i, 1]:.3f})', fontsize=10)
            plt.legend()
            plt.show()
            plt.close()
            # plot dot
            
            
            
        ############################
        print('resultat du test de la deuxième propriété du théorème ')
        frame_dot = pd.DataFrame({
                'f(α . f^{-1}(B))': PXBC_list,
                'α ⊙ B': PYBC_list,
                'L^2 erreur': dot_erreur_list_l2,
                'inf erreur': dot_erreur_list_inf
                })
        

        print(frame_dot)
            
            

        """
            affichage des Resultats de la seconde propriété du théorème
        """


        indice = random.sample(range(B.shape[0]),5)
        print('the plot of dot')
        for i in indice:
            plt.figure()
            plt.plot(X[:, 0], X[:, 1], '.', linewidth = 0.01) 
            plt.plot(B[i, 0], B[i, 1],   'x', color='red',  label=f'B_{i+1}')  
            plt.annotate(f'B_{i+1}', (B[i, 0], B[i, 1] - 0.01))
            # Tracer le point XXBC[i]
            plt.plot(PXBC[i, 0].detach().numpy(), PXBC[i, 1].detach().numpy(), 'o', color='yellow', label=r'f($ α \cdot f^{-1}(B)$)')
            plt.annotate(r'f($ α \cdot f^{-1}(B)$)', (XXBC[i, 0].detach().numpy(), XXBC[i, 1].detach().numpy() - 0.1))

            # Tracer le point YYBC[i]
            plt.plot(PXBC[i, 0].detach().numpy(), PXBC[i, 1].detach().numpy(), 'x', color='purple', label=r'$ α \odot B$')
            plt.annotate(r'$ α \odot B$', (PYBC[i, 0].detach().numpy(), PYBC[i, 1].detach().numpy() + 0.01))
            # Ajouter une légende au subplot    
            plt.title(f'Pour B_{i+1}: ({B[i, 0].item():.3f}, {B[i, 1].item():.3f}), α = {alpha[i].item():.3f}', fontsize=10)
            plt.legend()
            plt.show()
            plt.close()
            
        




    
"""On Génère les données pour l'entrainement"""
def line(K, epsilon):
    X = torch.rand(K, 2).requires_grad_(False)
    X[K//2:] *= -1
    Y = torch.randn(K, 2).requires_grad_(False)
    Y[K//2:] *= -1
    # alpha = torch.empty(K, 1).uniform_(-5, 5).requires_grad_(False)
    alpha = torch.randn(K, 1).requires_grad_(False)
    X[:,1] = X[:,0] + epsilon * torch.sin(X[:,0] / epsilon)
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



# on génère les données
K = 2000
epislon = 0.1
X,Y,alpha = line(K, epislon)
dim = 2
# on initialise le vecteur space
G = Vect_space(K, dim_E = dim, neurons = 64)
# on initialise l'optimiseur

optimizer = optim.Adadelta(list(G.parameters()), lr=1e-3)
# la loss
losses = G.train(X,Y, alpha, optimizer, args.epochs)

plt.figure(figsize=(6, 4))
plt.plot(X[:,0], X[:,1], 'x', label='train X')
plt.title('Training Data X')
plt.legend()
plt.show()



# on affiche la loss 
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.title('Losses')
plt.show()



# data test 
def test_laoder():
    K = 10
    B = 0.3*torch.randn((K, 2))
    C = 0.3*torch.randn((K, 2))
    alpha = torch.linspace(-5,5,K).reshape(-1,1)
    for i in range(K):
        B[i,1] = B[i,0] + epislon * torch.sin(B[i,0] / epislon )
        C[i,1] = C[i,0] + epislon * torch.sin(C[i,0] / epislon )
    return B, C, alpha

# result test 
test = G.test(test_laoder,X)
