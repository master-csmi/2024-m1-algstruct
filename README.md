# Mathematical structure detection

## Objective 
Given a dataset $X=(X_1,\ldots,X_n)$ of points in $\mathcal{M}\subset R^d$, and a choice of a mathematical structure (e.g. a group, vector space, ring, Lie Group, etc.), find the correponding laws such that the dataset could be equipped with this mathematical structure.

The difficulty is the formulation of the problem : how to design losses ?

## Two methods ?

First : learn the math structure by its definition. In the case of a group: learn the 4 axioms (zero, commutatitivy, assoc, opposite) as loss to minimize
Second : learn a mapping between R^d and M, and use the morphism property : x(+)y = f( f^{-1}(x) + f^{-1}(y) )  ?

## Milestones 

1. THink about the two methods, which one seems to be the simpliest ? 
2. After choosing the method, try on a very simple case : group, dataset in R^2 with a known function f ...
3. Check the vector space case in a simple case
4. Check more complcated case : in R^d ... 

## Why ?
Transform a nonlinear manifold into a vector space. Reduction modeling.
