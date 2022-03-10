library(rstan)

K <- 3
N <- 3
D_train <- 5
D_test <-2
L <- 4
test_a <- c(0.8, 19, 100)
test_b <- c(0.5, 9, 650)
y_test <- c(1,0)
test_data<-array(c(test_a, test_b), dim = c(3,2))
test_data <- t(test_data)

train_a <- c(0.5,25,150)
train_b <- c(0.1,7,900)
train_c <- c(1.2,13,400)
train_d <- c(0.4,20,350)
train_e <- c(0.9,9,700)
y_train <- c(1,0,1,0,1)
train_data<-array(c(train_a, train_b, train_c, train_d, train_e),dim = c(3,5))
train_data<-t(train_data)

node1<- c(1,2,3,2)
node2<-c(2,4,5,1)
node3<- c(3,6,7,3)

tree_nodes_int<-array(c(node1,node2,node3),dim = c(4,3))
tree_nodes_int <- t(tree_nodes_int)
leafs<-c(4,5,6,7)


data_list <- list(K=K, N=N, L=L, test_data=test_data, 
                  y_test=y_test, train_data=train_data,
                  y_train=y_train, leafs=leafs, tree_nodes_int=tree_nodes_int,
                  D_train=D_train ,D_test=D_test)

# Compiling and producing posterior samples from the model.
stan_samples <- stan(file = 'decision_tree_example.stan', data = data_list,chains = 1)


print(stan_samples)
