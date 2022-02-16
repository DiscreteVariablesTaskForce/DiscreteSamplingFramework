functions {
  real leafsProbabilities(int[] y_train, real[,] train_data, int[,] tree_nodes_int, int[] is_leaf_node, vector theta, int D_train, int N, int[] leafs,  int[] y_test, real[,]test_data, int D_test, int L){
    
    int datapoints_leaf[D_train];
    int position_on_leafs_prob;
    real leafs_prob[4,2];//number of labels, by number of leaf nodes
    int nodeIndex = 1 ;
    int flag = 0;
    int node_feature ;
    int labels[2];
    int number_of_same_labels = 0;
    real division;
    real probability;
    int index=0;
    real logprob = 0;
    int feature_to_compare = 0;
    int flag2 = 0;
    int helper;
    node_feature = tree_nodes_int[nodeIndex][4];

    for (d in 1:D_train){
     nodeIndex = 1;
     flag = 0;
     node_feature = tree_nodes_int[nodeIndex][4];
    
      
      while(flag == 0){
        
        if (is_leaf_node[nodeIndex] == 1){
          flag = 1;
          datapoints_leaf[d] = nodeIndex;
        }
        else if (train_data[d][node_feature] > theta[node_feature]){
          nodeIndex = tree_nodes_int[nodeIndex][3];
        }
        else if (train_data[d][node_feature] < theta[node_feature]){
          nodeIndex = tree_nodes_int[nodeIndex][2];
        }
        for (n in 1:4){//find index if in leafs
            if (nodeIndex == leafs[n]){
              flag2 = 1;
            }
          }
        if (flag2 == 0){//if index not in leafs increase feature
            node_feature = tree_nodes_int[nodeIndex][4];
          }
          flag2 =0;
        
      }
    
    }
    print("datapoints leaf: ", datapoints_leaf);
    
    labels[1]=0;
    labels[2]=1;
    

    for (j in 1:4){//number of leafs
      for (l in 1:2){//number of unique labels
        number_of_same_labels = 0;
        for (k in 1:D_train){//number of classified labels
          if ((datapoints_leaf[k] == leafs[j]) && (labels[l]==y_train[k])){
            number_of_same_labels = number_of_same_labels+1;
            print("label: ", labels[l]);
            print("ison me: ", y_train[k]);
            print("leaf: ", leafs[j]);
            print("ison me: ", datapoints_leaf[k]);
            
          }
        }
      print(number_of_same_labels);
      leafs_prob[j][l] = number_of_same_labels;
      
      }
      
      
    }
    print(leafs_prob);
    for (i in 1:4){//number of leafs
      division = 0;
      for (j in 1:2){//number of unique labels 
      division = leafs_prob[i][j]+division;
      print("division: ", division);
      }
      for (k in 1:2){
        probability = leafs_prob[i][k];
        print("probability: ", probability);
        leafs_prob[i][k] = probability*100/division;
      }
    }
    

    
    flag=1;
    index=1;
    logprob=0;
    flag2 = 0;
    for (i in 1:D_test){
      index=1;
      feature_to_compare = tree_nodes_int[index][4];
      flag = 1;
      flag2 = 0;
      while(flag == 1){
        
        for (n in 1:4){
            if (index == leafs[n]){//find if index in leafs
              flag2 = 1;
            }
          }
          if (flag2 == 1){//if nodex in leafs we are in leaf node
            flag = 0;
            helper = index - L + 1;
            print("helper: ", helper)
            logprob = logprob + leafs_prob[helper][y_test[i]+1];
          }
          flag2 =0;
        
        
        for (p in 1:3){//length of nodes
        print("index: ", index)
          if (tree_nodes_int[p][1] == index && test_data[i][feature_to_compare] > theta[index]){
            index = tree_nodes_int[p][3];
            } 
          else if (tree_nodes_int[p][1] == index && test_data[i][feature_to_compare] < theta[index]){
            index = tree_nodes_int[p][2];
          }
          for (n in 1:4){//find index if in leafs
            if (index == leafs[n]){
              flag2 = 1;
            }
          }
          if (flag2 == 0){//if index not in leafs increase feature
            feature_to_compare = tree_nodes_int[p][4];
          }
          flag2 =0;
        }
      }
    }
    print("here logprob: ", logprob/D_test);
    return logprob;
  }
}

data {
  int K; //Number of features
  int N; //Number of nodes
  int D_train; //Number of train datapoints
  int D_test; ////Number of test datapoints
  int L; //Number of leafs
  int y_test[D_test];//test labels
  int y_train[D_train];//test labels
  int leafs[L];//leafs id
  //vector [L] leaf_prob;//labels
  real test_data[D_test,K];//test datapoints
  real train_data[D_train,K];//train datapoints
  int tree_nodes_int[N,4];
  

}

transformed data {
  int is_leaf_node[7];
  
  for (i in 1:N){
    is_leaf_node[i] = 0;
    
    
  }

  for (i in 1:L){
    is_leaf_node[leafs[i]] = 1;
  }
  //[0,0,0,0,1,1,1]
}

parameters {
  vector [N] theta;
}


//For each MCMC iteration
model {

  theta ~ normal(0,100);
  //leafs probabilities 
  
 {real leafs_prob;
  target += leafsProbabilities(y_train[], train_data[,], tree_nodes_int[,], is_leaf_node[], theta, D_train, N,leafs[], y_test[], test_data[,], D_test, L);
}


}//end for model


