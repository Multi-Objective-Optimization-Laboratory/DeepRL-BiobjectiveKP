# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 12:54:57 2022

@author: Dr. Hadi Charkhgard

Deep Q-learning for solving bi-objective Knapsack Problems

"""
import torch
from torch import nn
from torch.nn import functional as F

import pandas as pd
import os
import numpy as np
import random
import sys
import shutil
import matplotlib.pyplot as plt
import copy
from collections import deque
import math
from timeit import default_timer as timer


######## General USER DEFINED PARAMETERS #########

Training_Active=0 #Choose 1 if you want the model to be trained. Otherwise choose 0, if you wish not. Default: 1.  
Testing_Active=1  #Choose 1 if you want the model to be tested. Otherwise choose 0, if you wish not. If you choose 1, then the algorithm will automatically set "Obj_Constraint_RHS_Is_Given=0". Default: 1.  
Obj_Constraint_RHS_Is_Given=1 #Choose 1 if the right-hand-side value of the objective constraint (i.e., c1-star) is already in the data files. Otherwise, choose 0.  If you choose 1, then the algorithm will automatically set "relative_decay=0". Default: 0.
Number_Of_Instances_For_Active_Learning=50 #Number of instances that need to be genreated during training phase of active learning. Default: 100.
WSM=1 #Choose one if you want the testing to be done based on the weighted sum method. Otherwise, choose zero, and in that case the testing will be done based on pseudo epsilon constraint method. Default: 1

######## USER DEFINED PARAMETERS For Training #########

TrainingDirectory='./TrainingSet/' #Folder name in which the instances that need to be solved are located. 
ClassSizes=[25] #The name of all folders of instances (meaning class sizes) that exist in the ClassDirectory. Names MUST be numeric and MUST show the size of each class.  
Num_of_Episodes=100 #Number of Epsiodes needed for training per C1_star: Default:100 
Number_Of_Mini_Batches=1 #Number of minibatches for training: Default: Number of Trainin Instances Divided by 100 
Maximum_Replay_List_Size=1000000 #This shows the maximum size of expeiences. When we hit this value, the old expeirnces will be replaced with newer ones.
Sample_Size_From_ReplayList=50 #This shows the sample_size. When we hit this value, the old expeirnces will be replaced with newer ones.
Minimum_Replay_list_Size=50 #This must be at least equal to Sample_Size_From_ReplayList. When the number of experiences is at least Minimum_Replay_list_Size, the training process will start. Before that no training will be done. 
Update_Saved_Model_Threshold=10 #This shows how often the so-far trained model will be saved. If you set it to 100, it means the model will be saved every 100 episodes. Default: 100
InFeasible_Reward_Multiplier=-10 # This must be a NEGATIVE value. If at a given step, choosing a varaiable is infeasible. This value will be multiplied by the objective coefficient to generate a NEGATIVE reward for infeasibility.     
NotSelecting_Reward_Multiplier=-10 # This is recommended to be a NON-POSITIVE value. If at a given step, choosing a varaiable is infeasible. This value will be multiplied by the objective coefficient to generate a NEGATIVE reward for infeasibility. 
relative_decay=0 #Choose a value in range (0,1]. This value impacts the change of C1_star (which is the right-hand-side of the first-objective constraint).  The formula is [[C1_star= np.floor(C1_star*relative_decay-absolute_decay)]]. Default: 1
absolute_decay=1 #Choose a non-negative value. This value impacts the change of C1_star (which is the right-hand-side of the first-objective constraint).  The formula is [[C1_star= np.floor(C1_star*relative_decay-absolute_decay)]]. Default: 1
relative_Min_C1_star=0.5 #Choose a value in range (0,1]. First note that Min_C1_star is the minimum C1_start among all instnaces. Now, relative_Min_C1_star*Min_C1_star  defines the LB for the value of C_1_star for training. That means smaller values of C_1_star will not be used for training. Default: 0.5
Training_Iteration_Relative_Decay=1 #Choose a value in range (0,1] larger than or equal to the relative_decay. This shows how the number of training iterations reduces as C1_star gets smaller. Default: relative_decay
Training_Minimum_Iterations=3 #This captures the minimum number of training iterations: Default: 0.9 
Automatic_Lambda=0  #If you choose 1 then augmented scalarization will be 1/(sum(c2_vetctor)+1). However, if you choose the value of 0 then the scalarization value would be set to Manual_Lambda_Value.  
Manual_Lambda_Value= 0.1 #This is a manual scalarization value and woulb eactive if Automatic_Lambda is set to 0.
Shuffle_the_order_of_parameters=0 #If you choose 1 then the order of parameters will chnage at the begining of each episode for each instance. If you choose zero, then the order is always [0,1,...,n]. Default: 1. 
Sequence_of_Percentiles= np.arange(1, 101, 1, dtype=int) #Sequence of Percentiles needed: default [1,....,100].
normalize_obj_vector=1  #If you choose one then scalarized objective coefficients will get normalized, i.e., divided by its maximum. If you choose zero, then no normalization will happen. Default: 1.
normalize_c1_const_vector=1 #If you choose one then c1-constraint coefficients will get normalized, i.e., divided by C1-star. If you choose zero, then no normalization will happen. Default: 1.
normalize_w_const_vector=1 #If you choose one then w-constraint coefficients will get normalized, i.e., divided by W. If you choose zero, then no normalization will happen. Default: 1.
normalize_ratio_vector=0  #If you choose one then the ratios will get normalized, i.e., normalized scalarized objective coefficients will be divided by normalized w-constraint coefficients. If you choose zero, then no normalization will happen. Default: 1.
Add_c1_const_vector_percentiles=1 #If you choose one then the percentiles of c1_based constraint will be added to the state info. If you choose zero, then it will not be added. Default: 1.
Add_w_vector_percentiles=1 #If you choose one then the percentiles of w_based constraint will be added to the state info. If you choose zero, then it will not be added. Default: 1.
Add_obj_vector_percentiles=1 #If you choose one then the percentiles of scalarized objective will be added to the state info. If you choose zero, then it will not be added. Default: 1.
Add_ratio_percentiles=1 #If you choose one then the percentiles of ratios will be  added to the state info. If you choose zero, then it will not be added. Default: 1.
Run_on_CPU=1 #Chooce one if you want CPU to be selected. Zero means GPU will be prefered and will be selected if it is avaiable.  
NumNodesHidden1=100 
NumNodesHidden2=100 
DeepLearningModelPATH='./DeepLearningModel/' #Path to the deep learning model trained so far.
DeepLearninModelgFileNameToLoad='CurrentModel.pt' #Name of the file that the represents the deep learning model that need to be loaded from DeepLearningModelPATH.
CreateNewDeepLearningModel=0 #Choose one if you want a new deep learning model to be trained from scratch each time. Otherwise, by choosing zero, if a partitally trained model already exists in DeepLearningModelPATH it will be loaded. Default 0.
MinEpsGreedy=0.1 #Choose the minimum value of epsilon in the epsilon greedy approach. Default=0.1
Number_Updates_for_Epsilon=72 #We first set the epsilon equal to 1 and then update/decrease it linearly (based on the number of epsiodes explored) until we reach to MinEpsGreedy. The number of updates is shown by Number_Updates_for_Epsilon. default=10.
Pytorch_Learning_rate=1e-3 #Choose the learning rate value for pytorch Adam optimizer. Default=0.1e-3
AvgLossPlot_Activation=1 #Choose the value of one if you want to see the plot of each C_Star. Otherwise, Choose zero. Default=0
Plot_Based_On_Average=0 #Plot Based On Average. Default: 1 
PlotsDirectory='./Plots/' 

######## USER DEFINED PARAMETERS For Testing #########

TestingDirectory='./TestingSet/' #Folder name in which the instances that need to be solved are located. 
Update_Cstar_Based_On_Last_Point=1 #A binary parameter. Update the value of Cstar based on the last point found. Default=1
relative_decay_testing=0 #Choose a value in range (0,1]. This value impacts the change of C1_star (which is the right-hand-side of the first-objective constraint).  The formula is [[C1_star= np.floor(C1_star*relative_decay_testing-absolute_decay_testing)]]. Default: 1
absolute_decay_testing=1 #Choose a non-negative value. This value impacts the change of C1_star (which is the right-hand-side of the first-objective constraint).  The formula is [[C1_star= np.floor(C1_star*relative_decay_testing-absolute_decay_testing)]]. Default: 1
ResultsDirectory='./TestingResults/' #Folder name to which the results will be generated
Beam_Search_Width=20 #Choose the beam search width. It must be a postive integer. Default: 1.  
Sampling_Search_Number=1 #Choose the sampling search size. It must be a postive integer. Default: 1. 
Actual_Number_Of_Items_Is_Given_In_Data=1 #A binary parameter. Choose the value of 1 if the actual number of items is given in the data file. Otherwise set it to zero.  Default: 1. 
Bidirectional_Epsilon_Search=0 #A binary parameter. Choose the value of 1 if you want to use Bidirectional Epsilon Search method. Otherwise set it to zero.  Default: 1. 
Compute_Other_Endpoint=1 #A binary parameter. Choose the value of 1 if you want to compute the other point of the nondominated frontier to find a minimum value for C1star.  Default: 1. 
Bidirectional_Error_Adjustment=0.1 #Choose a small value in range (0,1] to adjust Minimum C-Star when testing active ONLY when Compute_Other_Endpoint==1. The parameter should be close to zero. If you choose 0.1 then Min_C_star will be multiplied by 0.9 to make it smaller due to uncertainity involved in calculating Min_C_star. Default: 0.1
WSM_Loop_Size=100 #Choose the number of iterations of WSM. This should be at least one.

######## Rational Choices #########
if WSM>0:
    WSM=1
    Automatic_Lambda=0
    Manual_Lambda_Value=0
    Update_Cstar_Based_On_Last_Point=0
    relative_decay_testing=0 
    absolute_decay_testing=1 
    Bidirectional_Epsilon_Search=0
    Compute_Other_Endpoint=0
    Add_c1_const_vector_percentiles=0
    Sampling_Search_Number=1
    if WSM_Loop_Size==0:
        WSM_Loop_Size=1
   
if Bidirectional_Epsilon_Search==1:
        Compute_Other_Endpoint=1
    
if Sample_Size_From_ReplayList>Minimum_Replay_list_Size:
   Minimum_Replay_list_Size=Sample_Size_From_ReplayList
   print("WARNING: Minimum_Replay_list_Size was set equal to Sample_Size_From_ReplayList" )
      

if InFeasible_Reward_Multiplier>=0:
    print("WARNING: InFeasible_Reward_Multiplier must be negative. So, we set it to -1" )
    InFeasible_Reward_Multiplier=-1
    
######## GPU or CPU Selection #########

if Run_on_CPU==0 and torch.cuda.is_available(): 
    MyDevice = "cuda:0" 
else: 
    MyDevice = "cpu"   

########  Change_To_Tensor_Function #########

def Change_to_Tensor(Input_Val, Tensor_dim):
     Output_Val=torch.FloatTensor(Input_Val, device=MyDevice)
     if(Tensor_dim==2):
         Output_Val=Output_Val[None, :]
     return Output_Val    

########  Define Loss Function and Pytorch Optimizer #########

Loss_Function=torch.nn.MSELoss()
#Loss_Function=torch.nn.L1Loss()
    
######## USER DEFINED PARAMETERS #########

class Biobjective_KP_Simulator ():
    def __init__(self,filename, n_items,c1_vector,c2_vector,w_vector,W,obj_constraint_rhs):
        self.filename = filename
        self.n_items = n_items
        self.w_vector = np.asarray(w_vector)
        self.W= W
        self.c2_vector = np.maximum(0,np.asarray(c2_vector))
        if Automatic_Lambda==1:
            self.scalarization=  c1_vector + (c2_vector/(sum(self.c2_vector)+1))
        else:
            self.scalarization=  ((1-Manual_Lambda_Value)*c1_vector) + (c2_vector*Manual_Lambda_Value)
        self.obj_constraint_rhs= obj_constraint_rhs 
        self.c1_vector = np.maximum(0,np.asarray(c1_vector))
        
        
    def initialize_the_episode_by_reordering_the_instance_parameters(self,orderlist, C1_star): 
        self.episode_orderlist=orderlist
        self.episode_n_items_left = self.n_items
        self.episode_obj = (np.asarray([self.scalarization[i] for i in orderlist])/((1-normalize_obj_vector) + normalize_obj_vector*max(self.scalarization)))
        self.episode_C1_star_left= C1_star/((1-normalize_c1_const_vector)+normalize_c1_const_vector*max(self.c1_vector))
        self.episode_c1_const_vector = np.asarray([self.c1_vector[i] for i in orderlist])/((1-normalize_c1_const_vector)+normalize_c1_const_vector*max(self.c1_vector))
        self.episode_c1_vector = np.asarray([self.c1_vector[i] for i in orderlist]) #This is needed for only testing
        self.episode_c2_vector = np.asarray([self.c2_vector[i] for i in orderlist]) #This is needed for only testing
        self.episode_w_vector = np.asarray([self.w_vector[i] for i in orderlist])/((1-normalize_w_const_vector)+normalize_w_const_vector*max(self.w_vector))
        self.episode_W_left= self.W/((1-normalize_w_const_vector)+normalize_w_const_vector*max(self.w_vector))
        if (normalize_ratio_vector==1):
            MyConstant=1/self.W
            self.episode_obj_to_w_ratios=  self.episode_obj/(np.maximum(MyConstant,self.episode_w_vector))
        else:
            MyConstant=1
            self.episode_obj_to_w_ratios= np.asarray([self.scalarization[i] for i in orderlist])/(np.maximum(MyConstant,np.asarray([self.w_vector[i] for i in orderlist])))
        self.episode_obj_value=0
        
    def update_the_episode_status (self,C1_star_Deduct,W_Deduct, reward_Add):
        self.episode_n_items_left=self.episode_n_items_left-1
        self.episode_W_left =self.episode_W_left-W_Deduct
        self.episode_C1_star_left=self.episode_C1_star_left-C1_star_Deduct
        self.episode_obj=np.delete(self.episode_obj, 0)  
        self.episode_c1_const_vector=np.delete(self.episode_c1_const_vector, 0) 
        self.episode_c1_vector=np.delete(self.episode_c1_vector, 0)  #This is needed for only testing
        self.episode_c2_vector=np.delete(self.episode_c2_vector, 0)  #This is needed for only testing
        self.episode_w_vector=np.delete(self.episode_w_vector, 0) 
        self.episode_obj_to_w_ratios=np.delete(self.episode_obj_to_w_ratios, 0) 
        self.episode_obj_value=self.episode_obj_value+reward_Add
        
    def get_the_state_vector_of_the_episode (self):
        self.State_Info=[]
        if WSM==0:
           self.State_Info.append(np.array([self.episode_n_items_left, self.episode_W_left, self.episode_C1_star_left, self.episode_obj[0], self.episode_w_vector[0],self.episode_c1_const_vector[0]]))
        else:
            self.State_Info.append(np.array([self.episode_n_items_left, self.episode_W_left, self.episode_obj[0], self.episode_w_vector[0]]))
        if Add_w_vector_percentiles==1:
            self.State_Info.append(np.percentile(self.episode_w_vector,Sequence_of_Percentiles))
        if Add_c1_const_vector_percentiles==1:
            self.State_Info.append(np.percentile(self.episode_c1_const_vector,Sequence_of_Percentiles))
        if Add_obj_vector_percentiles==1:
            self.State_Info.append(np.percentile(self.episode_obj,Sequence_of_Percentiles))            
        if Add_ratio_percentiles==1:     
            self.State_Info.append(np.percentile(self.episode_obj_to_w_ratios,Sequence_of_Percentiles))
        self.State_Info=np.concatenate(self.State_Info) 
        self.State_Info=Change_to_Tensor(self.State_Info,2)
            
    def Is_Selecting_The_First_Item_Feasible (self):
        if self.episode_c1_const_vector[0]> self.episode_C1_star_left:
            return 0
        elif self.episode_w_vector[0]>  self.episode_W_left:
            return 0
        #elif self.episode_obj[0]<0:
        #    return 0
        else:
            return 1

    def Deep_Node_Copy (self, node): 
        self.episode_orderlist= copy.copy(node.episode_orderlist)
        self.episode_n_items_left= copy.copy(node.episode_n_items_left)
        self.episode_obj= copy.copy(node.episode_obj)
        self.episode_C1_star_left= copy.copy(node.episode_C1_star_left)
        self.episode_c1_const_vector= copy.copy(node.episode_c1_const_vector) 
        self.episode_c1_vector=copy.copy(node.episode_c1_vector) 
        self.episode_c2_vector=copy.copy(node.episode_c2_vector) 
        self.episode_w_vector=copy.copy(node.episode_w_vector) 
        self.episode_W_left=copy.copy(node.episode_W_left) 
        self.episode_obj_to_w_ratios= copy.copy(node.episode_obj_to_w_ratios) 
        self.episode_obj_value=copy.copy(node.episode_obj_value)    
        

######## Experience #########
        
class Experience ():
    def __init__(self,CurrentState,NextStateState,CurrentReward, ActionSelected, NonTerminalNode,NextState_is_Feasible):
        self.CurrentState = copy.copy(CurrentState) 
        self.NextState = copy.copy(NextStateState) 
        self.CurrentReward=copy.copy(CurrentReward)  
        self.ActionSelected=copy.copy(ActionSelected)      
        self.NonTerminalNode=copy.copy(NonTerminalNode) 
        self.NextState_is_Feasible=copy.copy(NextState_is_Feasible)    
            

         
######## Read An Instance for Training #########
        
def Read_An_Instance_For_Training(TrainingInstancesOfSameSize, InstanceDirectory,filename,folder):
      df = pd.read_csv(InstanceDirectory)  
      n_items= df.loc[0,"number_of_items"]
      c1_vector=  df.loc[:,"obj1"]
      c2_vector=  df.loc[:,"obj2"]
      w_vector=  df.loc[:,"Constraint"]
      RHS=  df.loc[0,"RHS"]
      obj_constraint_rhs=1e32
      if Obj_Constraint_RHS_Is_Given==1:
          obj_constraint_rhs = df.loc[0,"obj_constraint_RHS"]
      if n_items== folder:
          BOKP_Instance=Biobjective_KP_Simulator(filename, n_items,c1_vector,c2_vector,w_vector,RHS,obj_constraint_rhs)
          TrainingInstancesOfSameSize.append(BOKP_Instance)
      else:
          print('WARNING: %s was discarded because its number of items is %d which is not a match with the class_size %d' %(InstanceDirectory,n_items,folder))
      return 

######## Read An Instance for Testing #########
        
def Read_An_Instance_For_Testing(TestingDirectory,filename, Inverse):
      df = pd.read_csv(TestingDirectory+filename)  
      n_items= df.loc[0,"number_of_items"]
      c1_vector=  df.loc[:,"obj1"]
      c2_vector=  df.loc[:,"obj2"]
      w_vector=  df.loc[:,"Constraint"]
      RHS=  df.loc[0,"RHS"]
      if Actual_Number_Of_Items_Is_Given_In_Data==1:
             Actual_n_items= df.loc[0,"actual_number_of_items"]
      else:  
             Actual_n_items= n_items
      obj_constraint_rhs=1e32
      if Inverse==0:
          TestInstance=Biobjective_KP_Simulator(filename, n_items,c1_vector,c2_vector,w_vector,RHS,obj_constraint_rhs)
      else:
          TestInstance=Biobjective_KP_Simulator(filename, n_items,c2_vector,c1_vector,w_vector,RHS,obj_constraint_rhs)    
      return TestInstance, int(Actual_n_items)
  
######## Read A Class of Instances #########

def Read_all_instances_of_the_class(ClassDirectory,folder):
    TrainingInstancesOfSameSize=[]
    for filename in os.listdir(ClassDirectory):        
           Read_An_Instance_For_Training(TrainingInstancesOfSameSize, ClassDirectory+filename, filename, folder)
    return TrainingInstancesOfSameSize



########  Initialize The Episode For All Instances of The Same Size #########

def Initialize_The_Episode_For_All_Instances_of_The_Same_Size(TrainingInstancesOfSameSize,C1_star):
    for i in range(len(TrainingInstancesOfSameSize)):
          orderlist=list(range(int(TrainingInstancesOfSameSize[i].n_items)))
          if Shuffle_the_order_of_parameters==1:
              random.shuffle(orderlist)
          if Obj_Constraint_RHS_Is_Given==1:
              TrainingInstancesOfSameSize[i].initialize_the_episode_by_reordering_the_instance_parameters(orderlist,TrainingInstancesOfSameSize[i].obj_constraint_rhs)
          else:    
              TrainingInstancesOfSameSize[i].initialize_the_episode_by_reordering_the_instance_parameters(orderlist,C1_star)
          #print(TrainingInstancesOfSameSize[i].obj_constraint_rhs, TrainingInstancesOfSameSize[i].episode_C1_star_left)
           
########  Update Epsilon #########

def  Update_Epsilon (current_episode):
     Temp_Multiplier=math.floor(current_episode*Number_Updates_for_Epsilon/Num_of_Episodes)
     EpsGreedy=1-(Temp_Multiplier*(1-MinEpsGreedy)/(Number_Updates_for_Epsilon-1))
     return EpsGreedy

########  Epsilon Greedy Approach #########

def  Epsilon_Greedy_Selection(Q_Val,EpsGreedy):
        prob = np.random.random()
        if prob < EpsGreedy:
            action = np.random.choice(2)
        else:
            action = np.argmax(Q_Val)
        return action
    
 ########  Compute Q values for all instances that selection of the first remaining item is feasible. Also, take actions for all instances #########

def Sample_From_The_ReplayList(ReplayList):
     Selected_Experience=np.random.choice(len(ReplayList), Sample_Size_From_ReplayList, replace=False)
     FeasibilityList=[]
     for i in Selected_Experience:
             if (i==Selected_Experience[0]):
                 Combined_Current_State_Tensor=ReplayList[i].CurrentState 
                 Combined_Next_State_Tensor=ReplayList[i].NextState
                 Combined_Reward_Tensor=ReplayList[i].CurrentReward
                 Combined_Action_Tensor=ReplayList[i].ActionSelected
                 Combined_Done_Tensor=ReplayList[i].NonTerminalNode 
                 FeasibilityList.append(ReplayList[i].NextState_is_Feasible)
             else:   
                 Combined_Current_State_Tensor=torch.cat((Combined_Current_State_Tensor,ReplayList[i].CurrentState),dim=0) 
                 Combined_Next_State_Tensor=torch.cat((Combined_Next_State_Tensor,ReplayList[i].NextState),dim=0)
                 Combined_Reward_Tensor=torch.cat((Combined_Reward_Tensor,ReplayList[i].CurrentReward),dim=0)
                 Combined_Action_Tensor=torch.cat((Combined_Action_Tensor,ReplayList[i].ActionSelected),dim=0)                                                  
                 Combined_Done_Tensor=torch.cat((Combined_Done_Tensor,ReplayList[i].NonTerminalNode),dim=0)  
                 FeasibilityList.append(ReplayList[i].NextState_is_Feasible)          
     return Combined_Current_State_Tensor,Combined_Next_State_Tensor,Combined_Reward_Tensor,Combined_Action_Tensor,Combined_Done_Tensor,FeasibilityList
   
########  Update Experience List #########

def Update_Experience_List(TrainingInstancesOfSameSize,DeepModel,EpsGreedy,mini,MiniBatches,ReplayList,step,folder):
    for i in MiniBatches[mini]:
            TrainingInstancesOfSameSize[i].get_the_state_vector_of_the_episode()
            Combined_State_Tensor=TrainingInstancesOfSameSize[i].State_Info 
            with torch.no_grad():
                QValues_Tensor=DeepModel(Combined_State_Tensor)
                QValues_Numpy=QValues_Tensor.data.numpy()
                Action=Epsilon_Greedy_Selection(QValues_Numpy,EpsGreedy)
            Reward_Multiplier=Action
            if (TrainingInstancesOfSameSize[i].Is_Selecting_The_First_Item_Feasible()==0 and Action==1):
                if (TrainingInstancesOfSameSize[i].episode_obj[0]>0):
                          Reward_Multiplier= InFeasible_Reward_Multiplier 
                else:
                          Reward_Multiplier= abs(InFeasible_Reward_Multiplier) 
            elif (Action==0): 
                if (TrainingInstancesOfSameSize[i].episode_obj[0]>0):
                          Reward_Multiplier= NotSelecting_Reward_Multiplier
            Rewards_Tensor=Change_to_Tensor(np.asarray([Reward_Multiplier*TrainingInstancesOfSameSize[i].episode_obj[0]]),1)
            Actions_Selected=Change_to_Tensor(np.asarray([Action]),2)
            TrainingInstancesOfSameSize[i].update_the_episode_status(Action*TrainingInstancesOfSameSize[i].episode_c1_const_vector[0],Action*TrainingInstancesOfSameSize[i].episode_w_vector[0],Reward_Multiplier*TrainingInstancesOfSameSize[i].episode_obj[0])     #take action 
            if step==folder-1:
                NonTerminalNode=Change_to_Tensor(np.asarray([0]),1)
                New_Combined_State_Tensor=Combined_State_Tensor
                NextState_is_Feasible=0
            else:
                NonTerminalNode=Change_to_Tensor(np.asarray([1]),1)
                TrainingInstancesOfSameSize[i].get_the_state_vector_of_the_episode()
                NextState_is_Feasible=TrainingInstancesOfSameSize[i].Is_Selecting_The_First_Item_Feasible()
                New_Combined_State_Tensor=TrainingInstancesOfSameSize[i].State_Info            
            NewExperience=Experience(Combined_State_Tensor,New_Combined_State_Tensor,Rewards_Tensor,Actions_Selected,NonTerminalNode,NextState_is_Feasible)
            ReplayList.append(NewExperience)  
    return ReplayList

######## Compute Maximum Possible C_Star  #########

def Compute_Maximum_and_Minimum_C_Star(TrainingInstancesOfSameSize):
    Maximum_C_Star=0
    Minimum_C_Star=1e32
    for i in range(len(TrainingInstancesOfSameSize)):
        Maximum_C_Star= max(Maximum_C_Star,sum(TrainingInstancesOfSameSize[i].c1_vector))
        Minimum_C_Star= min(Minimum_C_Star,sum(TrainingInstancesOfSameSize[i].c1_vector))
    return Maximum_C_Star, Minimum_C_Star 


######## Compute G-Tensor (Total Expected Future Rewards) #########

def Compute_G_Tensor(Combined_Next_State_Tensor,Combined_Done_Tensor,FeasibilityList,TargetDeepModel):
    with torch.no_grad():
         New_QValues_Tensor=TargetDeepModel(Combined_Next_State_Tensor)  
         G_Tensor=torch.max(New_QValues_Tensor,dim=1)[0]
         for i in range(Sample_Size_From_ReplayList):
              if(FeasibilityList[i]==0):
                    G_Tensor[i]=New_QValues_Tensor[i][0]          
         G_Tensor=G_Tensor*Combined_Done_Tensor 
    return G_Tensor

####### Average Loss Calculator #####

def Computing_Avg_Loss_For_Reporting(Denom,SumLoss_SoFar,Current_Loss,LossList_SoFar):
    Denom=Denom+1
    SumLoss_SoFar=SumLoss_SoFar+Current_Loss
    if Plot_Based_On_Average==1:
         LossList_SoFar.append(SumLoss_SoFar/Denom)
    else:
        LossList_SoFar.append(Current_Loss)
    return Denom,SumLoss_SoFar,LossList_SoFar

####### Update The Saved Model #########

def Update_the_Saved_Model(folder,C1_star,current_episode,Update_Saved_Model_Counter,LossList_SoFar,DeepModel):
    if Update_Saved_Model_Counter == Update_Saved_Model_Threshold:
         if Obj_Constraint_RHS_Is_Given==1:
             Save_the_Deep_Learning_Model(folder,current_episode,-1,DeepModel)
         else: 
             Save_the_Deep_Learning_Model(folder,current_episode,C1_star,DeepModel)
         Update_Saved_Model_Counter=1
         if AvgLossPlot_Activation==1:
             Plot_the_average_Loss(C1_star,LossList_SoFar,folder)
    else:
        Update_Saved_Model_Counter=Update_Saved_Model_Counter+1
    return Update_Saved_Model_Counter

######## Create Mini Batches Indicies  #########

def Create_Mini_Batches_Indicies(TrainingInstancesOfSameSize):
        Instances_Indicies=np.arange(len(TrainingInstancesOfSameSize))
        np.random.shuffle(Instances_Indicies)
        MiniBatches=np.array_split(Instances_Indicies,Number_Of_Mini_Batches)
        return MiniBatches   

######## Update Target Network #########

def Update_Target_Network(Update_Target_Network_Counter,TargetDeepModel,Update_Target_Network_threshold,DeepModel):
      if Update_Target_Network_Counter==Update_Target_Network_threshold:
          TargetDeepModel.load_state_dict(DeepModel.state_dict())
          Update_Target_Network_Counter=1
      else:    
          Update_Target_Network_Counter=Update_Target_Network_Counter+1 
      return TargetDeepModel,Update_Target_Network_Counter
  
######## Deep Q-Learning  #########

def Deep_Q_learning(TrainingInstancesOfSameSize,C1_star, folder,DeepModel,Optimizer,Current_Training_Iteration_Decay,ReplayList,TargetDeepModel):
    Denom,SumLoss_SoFar,LossList_SoFar=0,0,[]
    Update_Saved_Model_Counter=1
    Update_Target_Network_Counter=1
    Update_Target_Network_threshold=folder
    Num_of_Training_Episodes=int(max(np.floor(Num_of_Episodes*Current_Training_Iteration_Decay),Training_Minimum_Iterations))
    for current_episode in range(Num_of_Training_Episodes):
        print('We are now explroing Episode %d' %(current_episode)) 
        EpsGreedy=Update_Epsilon(current_episode)
        Initialize_The_Episode_For_All_Instances_of_The_Same_Size(TrainingInstancesOfSameSize,C1_star)  
        MiniBatches=Create_Mini_Batches_Indicies(TrainingInstancesOfSameSize)
        for mini in range(Number_Of_Mini_Batches):
           for step in  range(folder):
               ReplayList=Update_Experience_List(TrainingInstancesOfSameSize,DeepModel,EpsGreedy,mini,MiniBatches,ReplayList,step,folder)
               if(len(ReplayList)>=Minimum_Replay_list_Size): 
                     Combined_Current_State_Tensor,Combined_Next_State_Tensor,Combined_Reward_Tensor,Combined_Action_Tensor,Combined_Done_Tensor,FeasibilityList=Sample_From_The_ReplayList(ReplayList)                    
                     QValues_Tensor=DeepModel(Combined_Current_State_Tensor)
                     Old_Estimate=QValues_Tensor.gather(dim=1,index=Combined_Action_Tensor.long()).squeeze(dim=1)
                     G_Tensor=Compute_G_Tensor(Combined_Next_State_Tensor,Combined_Done_Tensor,FeasibilityList,TargetDeepModel)
                     Target=G_Tensor+Combined_Reward_Tensor
                     loss_value=Loss_Function(Old_Estimate,Target.detach())
                     Optimizer.zero_grad()
                     loss_value.backward()
                     if AvgLossPlot_Activation==1:
                          Denom,SumLoss_SoFar,LossList_SoFar=Computing_Avg_Loss_For_Reporting(Denom,SumLoss_SoFar,loss_value.item(),LossList_SoFar)
                     Optimizer.step() 
                     TargetDeepModel,Update_Target_Network_Counter=Update_Target_Network(Update_Target_Network_Counter,TargetDeepModel,Update_Target_Network_threshold,DeepModel)
        Update_Saved_Model_Counter=Update_the_Saved_Model(folder,C1_star,current_episode,Update_Saved_Model_Counter,LossList_SoFar,DeepModel)                  
    return LossList_SoFar


######## Deep Neural Network Architecture #########

class MyArchitecture (nn.Module):
    
        def __init__(self, NumInputNodes, NumNodesHidden1, NumNodesHidden2):
            super(MyArchitecture,self).__init__()
            #self.multihead_attn = nn.MultiheadAttention(NumInputNodes, num_heads)
            self.l1=nn.Linear(NumInputNodes, NumNodesHidden1)
            self.l2 =nn.Linear(NumNodesHidden1, NumNodesHidden2)
            self.l3 =nn.Linear(NumNodesHidden2, 2)

        def forward(self, x): 
            #x,w =self.multihead_attn(x ,x, x) 
             #x=F.normalize(x, dim=1)
             #x =F.leaky_relu(x)
             x =self.l1(x)
             x =F.relu(x)
             x =self.l2(x)
             x =F.leaky_relu(x)
             y =self.l3(x) 
             return y
               
######## Create or Read the Deep Neural Network Model #########

def Create_or_load_the_Deep_Learning_Model():
    NumInputNodes=6-(2*WSM)+len(Sequence_of_Percentiles)*(Add_c1_const_vector_percentiles+Add_w_vector_percentiles+Add_obj_vector_percentiles+Add_ratio_percentiles)
    DeepModel= MyArchitecture(NumInputNodes, NumNodesHidden1, NumNodesHidden2)  
    if (os.path.exists(DeepLearningModelPATH+DeepLearninModelgFileNameToLoad)==1 and CreateNewDeepLearningModel==0):
        DeepModel.load_state_dict(torch.load(DeepLearningModelPATH+DeepLearninModelgFileNameToLoad))
    TargetDeepModel=copy.deepcopy(DeepModel)  
    TargetDeepModel.load_state_dict(DeepModel.state_dict())
    Optimizer=torch.optim.Adam(DeepModel.parameters(),lr=Pytorch_Learning_rate)
    return DeepModel,Optimizer,TargetDeepModel    

######## Save the Deep Neural Network Model #########

def Save_the_Deep_Learning_Model(folder,episode_num_temp,c1_star_temp,DeepModel):
    if Testing_Active==0:
        torch.save(DeepModel.state_dict(), DeepLearningModelPATH+DeepLearninModelgFileNameToLoad)
        CopyName="ModelForClass_%s_C1Star_%d_Episode_%d_Setting_%d_%d_%d_%d.pt"%(folder,c1_star_temp,episode_num_temp,Add_c1_const_vector_percentiles, Add_w_vector_percentiles,Add_obj_vector_percentiles,Add_ratio_percentiles) 
        torch.save(DeepModel.state_dict(), DeepLearningModelPATH+CopyName) 
    else:
        print("ALART: We do not save the deep learning model as you are in the Active-Learning Mode, i.e., Both Tarining and Testing    ")         


######## Plot the average Loss #########

def Plot_the_average_Loss(C1_star, LossList,folder):
    fig,ax=plt.subplots(1,1)      
    ax.set_xlabel("Epoch for C1_Star of %s"%C1_star) 
    ax.set_ylabel("Avg Loss") 
    ax.scatter(np.arange(len(LossList)),LossList)
    Name_Size="Size_%s_C1star_%s.png" %(folder,C1_star)
    fig.savefig(PlotsDirectory+Name_Size)
     
    Detailed_Loss_df = pd.DataFrame(LossList)  
   
    Loss_CSV_report_file_name= PlotsDirectory+'Loss_Values.csv'
    Detailed_Loss_df.to_csv(Loss_CSV_report_file_name, index=False, header=False)
    return
          
          
######## Temporary Solutions #########

class Partial_Temporary_Point():
      def __init__(self,FirstObjValue,SecondObjValue,Solution,ExpectedReward,SimulatorIndex):
          self.ExpectedReward=ExpectedReward
          self.obj1=FirstObjValue
          self.obj2=SecondObjValue
          self.solution=copy.copy(Solution)  
          self.SimulatorIndex=SimulatorIndex
          self.Selecting_The_Next_Item_Is_Feasible=-1
          
   
######## Test Each Step of the Test Instance #########

def Test_the_Step (TestInstance_List, Current_Beam_Search_List,DeepModel):
    for i in range(len(Current_Beam_Search_List)):
        j=Current_Beam_Search_List[i].SimulatorIndex
        Current_Beam_Search_List[i].Selecting_The_Next_Item_Is_Feasible=TestInstance_List[j].Is_Selecting_The_First_Item_Feasible()
        if i==0:
            TestInstance_List[j].get_the_state_vector_of_the_episode() 
            Combined_State_Tensor=TestInstance_List[j].State_Info
        else:
            TestInstance_List[j].get_the_state_vector_of_the_episode() 
            Combined_State_Tensor=torch.cat((Combined_State_Tensor,TestInstance_List[j].State_Info),dim=0)     
    with torch.no_grad():
         QValues_Tensor=DeepModel(Combined_State_Tensor)
         QValues_Numpy=QValues_Tensor.data.numpy()
  
    return QValues_Numpy


######## Add the new partial solution #########

def Add_the_new_partial_Solution(Partial_Solutions_list,NewSol):  
    Should_be_Added=1
    item=0
    while item<len(Partial_Solutions_list) and Should_be_Added==1:
       if(Partial_Solutions_list[item].ExpectedReward<NewSol.ExpectedReward):
           Partial_Solutions_list.insert(item,NewSol)
           Should_be_Added=0
       else:     
           item=item+1
    if Should_be_Added==1:
       Partial_Solutions_list.append(NewSol) 

######## Update_Current_Beam_Search_List #########

def Update_Current_Beam_Search_List(Current_Beam_Search_List, Partial_Solutions_list, Not_Used_Simulations):
    Current_Beam_Search_List=[]
    i=0
    while i<len(Partial_Solutions_list):
        if i<Beam_Search_Width:
            NewSol=Partial_Temporary_Point(Partial_Solutions_list[i].obj1,Partial_Solutions_list[i].obj2, Partial_Solutions_list[i].solution, Partial_Solutions_list[i].ExpectedReward, Partial_Solutions_list[i].SimulatorIndex)
            Current_Beam_Search_List.append(NewSol)
        else: 
            Not_Used_Simulations.append(Partial_Solutions_list[i].SimulatorIndex)
        i=i+1    
    return Current_Beam_Search_List, Not_Used_Simulations     
######## Beam Search Internal Loop #########

def  Beam_Search_Internal_Loop(Current_Beam_Search_List,TestInstance_List, Not_Used_Simulations,DeepModel): 
      Partial_Solutions_list=[]
      QValues_Numpy=Test_the_Step(TestInstance_List, Current_Beam_Search_List,DeepModel)   
      for i in range(len(Current_Beam_Search_List)):
          j=Current_Beam_Search_List[i].SimulatorIndex    
          ## Creating Selecting Node          
          if Current_Beam_Search_List[i].Selecting_The_Next_Item_Is_Feasible==1:      
              Action_1_FirstObj= Current_Beam_Search_List[i].obj1 + TestInstance_List[j].episode_c1_vector[0]
              Action_1_SecondObj= Current_Beam_Search_List[i].obj2 + TestInstance_List[j].episode_c2_vector[0]
              Action_1_Sol=copy.copy(Current_Beam_Search_List[i].solution)
              Action_1_Sol.append(1)
              Action_1_ExpectedRewards=TestInstance_List[j].episode_obj_value+QValues_Numpy[i,1]
              Action_1_SimulatorIndex=Not_Used_Simulations[0]
              Not_Used_Simulations.pop(0)
              NewSol=Partial_Temporary_Point(Action_1_FirstObj, Action_1_SecondObj,Action_1_Sol, Action_1_ExpectedRewards,Action_1_SimulatorIndex) 
              Add_the_new_partial_Solution(Partial_Solutions_list,NewSol)  
              TestInstance_List[Action_1_SimulatorIndex].Deep_Node_Copy(TestInstance_List[j])
              TestInstance_List[Action_1_SimulatorIndex].update_the_episode_status(TestInstance_List[Action_1_SimulatorIndex].episode_c1_const_vector[0],TestInstance_List[Action_1_SimulatorIndex].episode_w_vector[0],TestInstance_List[Action_1_SimulatorIndex].episode_obj[0])                     
          ## Creating Not_Selecting Node          
          if TestInstance_List[j].episode_c1_const_vector[0]>0:
                 Reward_Multiplier=NotSelecting_Reward_Multiplier
          else:
                 Reward_Multiplier=0
          Action_0_FirstObj= Current_Beam_Search_List[i].obj1 
          Action_0_SecondObj= Current_Beam_Search_List[i].obj2
          Action_0_Sol=copy.copy(Current_Beam_Search_List[i].solution)
          Action_0_Sol.append(0)
          Action_0_ExpectedRewards=TestInstance_List[j].episode_obj_value+QValues_Numpy[i,0]
          Action_0_SimulatorIndex=j
          NewSol=Partial_Temporary_Point(Action_0_FirstObj, Action_0_SecondObj,Action_0_Sol, Action_0_ExpectedRewards,Action_0_SimulatorIndex) 
          Add_the_new_partial_Solution(Partial_Solutions_list,NewSol)  
          TestInstance_List[j].update_the_episode_status(0,0,Reward_Multiplier*TestInstance_List[j].episode_obj[0])  
              
      return Partial_Solutions_list,Not_Used_Simulations
  
######## Initialize All Simulations of the testing set under exploration ######### 
    
def Initialize_All_Simulations(TestInstance_List,orderlist,C1_star) :
    for iterator in range(2*Beam_Search_Width):
        TestInstance_List[iterator].initialize_the_episode_by_reordering_the_instance_parameters(orderlist,C1_star)
     
######## Exploring the test instance using beam search and sampling technique #########

def Test_the_Instance(TestInstance_List,C1_star,Pareto_Optimal_List,Actual_n_items, Inverse,DeepModel):
    Number_of_Items=int(TestInstance_List[0].n_items)
    Temp_Solution=[0]*Number_of_Items
    if WSM==0:
        orderlist=list(range(Number_of_Items))
    else:    
        ratios_inverse=TestInstance_List[0].w_vector/(((1-Manual_Lambda_Value)*TestInstance_List[0].c1_vector)+(Manual_Lambda_Value*TestInstance_List[0].c2_vector))
        orderlist = np.argsort(ratios_inverse)
    Temp_Counter=Sampling_Search_Number
    Estimated_max_obj_main=-1e32
    Estimated_min_obj_other=1e32
    while Temp_Counter>0:
        Current_Beam_Search_List=[]
        NewSol=Partial_Temporary_Point(0,0,[],0,0)
        Current_Beam_Search_List.append(NewSol)
        Initialize_All_Simulations(TestInstance_List,orderlist,C1_star)
        Not_Used_Simulations=list(range(1,2*Beam_Search_Width))
        for step in range(Actual_n_items):
            Partial_Solutions_list,Not_Used_Simulations=Beam_Search_Internal_Loop(Current_Beam_Search_List,TestInstance_List,Not_Used_Simulations,DeepModel)
            Current_Beam_Search_List, Not_Used_Simulations=Update_Current_Beam_Search_List(Current_Beam_Search_List,Partial_Solutions_list,Not_Used_Simulations)   
        Estimated_max_obj_main=max(Estimated_max_obj_main,Partial_Solutions_list[0].obj1)  
        Estimated_min_obj_other=min(Estimated_min_obj_other,Partial_Solutions_list[0].obj2)
        for k in range(len(Current_Beam_Search_List)):
            if Sampling_Search_Number==1 and WSM==0:
                if Inverse==0:
                    Pareto_Optimal_List=Pareto_Optimal_Point(Pareto_Optimal_List,Current_Beam_Search_List[k].obj1,Current_Beam_Search_List[k].obj2,Current_Beam_Search_List[k].solution)   
                else:
                    Pareto_Optimal_List=Pareto_Optimal_Point(Pareto_Optimal_List,Current_Beam_Search_List[k].obj2,Current_Beam_Search_List[k].obj1, Current_Beam_Search_List[k].solution)               
            else:
                for i in range(Actual_n_items):
                    j=orderlist[i]
                    Temp_Solution[j]=Current_Beam_Search_List[k].solution[i]    
                if Inverse==0:    
                   Pareto_Optimal_List=Pareto_Optimal_Point(Pareto_Optimal_List,Current_Beam_Search_List[k].obj1,Current_Beam_Search_List[k].obj2,Temp_Solution)   
                else:
                   Pareto_Optimal_List=Pareto_Optimal_Point(Pareto_Optimal_List,Current_Beam_Search_List[k].obj2,Current_Beam_Search_List[k].obj1,Temp_Solution)    
        if WSM==0:
             random.shuffle(orderlist)   
        Temp_Counter=Temp_Counter-1               
    return Pareto_Optimal_List, Estimated_max_obj_main,(1-Bidirectional_Error_Adjustment)*Estimated_min_obj_other 

######## Pareto Optimal Points #########

class Pareto_Opimal_Point_Class():
      def __init__(self,FirstObjValue,SecondObjValue,Solution):
          self.obj1=FirstObjValue
          self.obj2=SecondObjValue
          self.solution=copy.copy(Solution)

######## Update Pareto Optimal Points #########     
     
def Pareto_Optimal_Point(Pareto_Optimal_List,FirstObjValue,SecondObjValue,Solution):
    Should_Be_Appended=1
    i=0
    while i<len(Pareto_Optimal_List) and Should_Be_Appended==1:
        if FirstObjValue< Pareto_Optimal_List[i].obj1:
             if(SecondObjValue<=Pareto_Optimal_List[i].obj2):
                   Should_Be_Appended=0
             else:
                   i=i+1 
        elif FirstObjValue==Pareto_Optimal_List[i].obj1:
             if(SecondObjValue<=Pareto_Optimal_List[i].obj2):
                   Should_Be_Appended=0
             else:
                   Pareto_Optimal_List.pop(i)
        elif SecondObjValue>=Pareto_Optimal_List[i].obj2:
                Pareto_Optimal_List.pop(i)
        else: 
                New_Pareto=Pareto_Opimal_Point_Class(FirstObjValue,SecondObjValue,Solution)
                Pareto_Optimal_List.insert(i,New_Pareto)
                Should_Be_Appended=0
    if Should_Be_Appended==1:
       New_Pareto=Pareto_Opimal_Point_Class(FirstObjValue,SecondObjValue,Solution)
       Pareto_Optimal_List.append(New_Pareto)      
    return Pareto_Optimal_List

######## Report The Results #########

def report_the_testing_results(filename, NumberOfItems, Pareto_Optimal_List,duration_testing_time):
 
    Number_of_Points_Found=len(Pareto_Optimal_List)
   
    new_row=['obj1', 'obj2']
    for j in range(NumberOfItems):
           new_row.append('x_%d'%j)  
    new_row.append('time')      
    Detailed_df = pd.DataFrame(new_row)  
    Detailed_df=Detailed_df.transpose()    
    for i in range(Number_of_Points_Found):
        new_row=[Pareto_Optimal_List[i].obj1,Pareto_Optimal_List[i].obj2]
        for j in range(NumberOfItems):
             new_row.append(Pareto_Optimal_List[i].solution[j]) 
        if i==0:
            new_row.append(duration_testing_time)   
        else:
            new_row.append("")
        Detailed_df.loc[len(Detailed_df)] =  new_row
    name_main_part = filename[:-4]  #filename without '.csv'
    report_file_name= name_main_part+'_DRL_Results.csv'
    Detailed_df.to_csv(ResultsDirectory+report_file_name, index=False, header=False)
    
######## Main Training LOOP #########
def Training_LOOP():
    print("---- Training Mode ----- ")
    ReplayList= deque(maxlen=Maximum_Replay_List_Size)
    for folder in ClassSizes: 
        DeepModel,Optimizer,TargetDeepModel = Create_or_load_the_Deep_Learning_Model()
        DeepModel.train()
        ClassFolder="%s/" %folder
        TrainingInstancesOfSameSize=Read_all_instances_of_the_class(TrainingDirectory+ClassFolder,folder)
        C1_star,Min_C1_star=Compute_Maximum_and_Minimum_C_Star(TrainingInstancesOfSameSize)
        Current_Training_Iteration_Decay=1
        while C1_star>=1 and C1_star>=Min_C1_star*relative_Min_C1_star:
            print('Now exploring ClassSize %s and C1_star %d' %(folder,C1_star)) 
            LossList=Deep_Q_learning(TrainingInstancesOfSameSize,C1_star, folder,DeepModel,Optimizer,Current_Training_Iteration_Decay,ReplayList,TargetDeepModel)
            if AvgLossPlot_Activation==1:
                  Plot_the_average_Loss(C1_star,LossList,folder)
            C1_star= np.floor(C1_star*relative_decay-absolute_decay)
            Current_Training_Iteration_Decay=Current_Training_Iteration_Decay*Training_Iteration_Relative_Decay
        Save_the_Deep_Learning_Model(folder,Num_of_Episodes,-2,DeepModel) 
    return DeepModel,Optimizer,TargetDeepModel  



def generate_instance_csv_file(ActiveLearningFolderPath,TestInstance_List_Main,Instance_Generator_Name,C1_star, WSM, Number_Of_Instances_For_Active_Learning):
         if WSM==0: 
            scalar_weight=0 
            if Instance_Generator_Name==0:
                C1star_Value=C1_star
            else: 
               C1star_Value=np.round(np.random.uniform(0.1,0.5)*C1_star) 
         else:      
            C1star_Value=C1_star 
            scalar_weight=((Instance_Generator_Name+1)/(Number_Of_Instances_For_Active_Learning+1)) 
         number_of_items= ['' for index in range(int(TestInstance_List_Main[0].n_items))] 
         number_of_items[0]=int(TestInstance_List_Main[0].n_items)
         RHS=['' for index in range(int(TestInstance_List_Main[0].n_items))]
         RHS[0]=TestInstance_List_Main[0].W
         obj_constraint_RHS=['' for index in range(int(TestInstance_List_Main[0].n_items))]
         obj_constraint_RHS[0]=C1star_Value
         NewData={
             'number_of_items': number_of_items,  
             'obj1': ((1-scalar_weight)*TestInstance_List_Main[0].c1_vector)+ (scalar_weight*TestInstance_List_Main[0].c2_vector),
             'obj2': TestInstance_List_Main[0].c2_vector,
             'Constraint': TestInstance_List_Main[0].w_vector,
             'RHS': RHS,
             'obj_constraint_RHS':obj_constraint_RHS
             }
         NewDataFrame=pd.DataFrame(NewData)
         NewDataFrame.to_csv(ActiveLearningFolderPath+str(int(Instance_Generator_Name))+'.csv', index=False)
         return 

def Create_Two_Lists_For_The_Test_Instance(TestingDirectory,filename,Beam_Search_Width,Compute_Other_Endpoint):
        TestInstance_List_Main=[]
        TestInstance_List_Inverse=[]
        for iterator in range(2*Beam_Search_Width):
            TestInstance,Actual_n_items=Read_An_Instance_For_Testing(TestingDirectory,filename,0)
            TestInstance_List_Main.append(TestInstance)  
        if Compute_Other_Endpoint==1:           
           TestInstance_List_Inverse=[]
           for iterator in range(2*Beam_Search_Width):
               TestInstance,Actual_n_items=Read_An_Instance_For_Testing(TestingDirectory,filename,1)
               TestInstance_List_Inverse.append(TestInstance)             
        if Actual_n_items<TestInstance_List_Main[0].n_items and Sampling_Search_Number>1:
            Actual_n_items=int(TestInstance_List_Main[0].n_items)   
            print('WARNING: We modified the value of Actual_n_items and set it to n_items becuase you chose a value larger than 1 for Sampling_Search_Number')
        elif  Actual_n_items>TestInstance_List_Main[0].n_items:     
            Actual_n_items=int(TestInstance_List_Main[0].n_items)
            print('WARNING: We modified the value of Actual_n_items and set it to n_items becuase the value of Actual_n_items is larger than n_items in the data file (which is impossible!)')
        return TestInstance_List_Main, TestInstance_List_Inverse,Actual_n_items   
    
def Create_Traing_Instnaces_For_Active_Learning(TrainingDirectory,TestInstance_List_Main,Number_Of_Instances_For_Active_Learning,C1_star, WSM):
       ActiveLearningFolderName=TrainingDirectory+str(int(TestInstance_List_Main[0].n_items))
       ActiveLearningFolderPath=ActiveLearningFolderName+'/'
       if os.path.isdir(ActiveLearningFolderPath)==1:
           shutil.rmtree(ActiveLearningFolderName)    
       os.mkdir(ActiveLearningFolderPath)
       for Instance_Generator_Name in range(Number_Of_Instances_For_Active_Learning):
             generate_instance_csv_file(ActiveLearningFolderPath,TestInstance_List_Main,Instance_Generator_Name,C1_star, WSM, Number_Of_Instances_For_Active_Learning)

def pseudo_epsilon_Constraint_Method(Pareto_Optimal_List,TestInstance_List_Main,TestInstance_List_Inverse,Update_Cstar_Based_On_Last_Point,relative_decay_testing,absolute_decay_testing,DeepModel,Obj_Constraint_RHS_Is_Given):
           C1_star=sum(TestInstance_List_Main[0].c1_vector) 
           print('Now testing on Instance %s with C1_star %d' %(filename,C1_star)) 
           Pareto_Optimal_List, Estimated_max_obj_1,Estimated_min_obj_2=Test_the_Instance(TestInstance_List_Main,C1_star,Pareto_Optimal_List,Actual_n_items, 0,DeepModel)
           Min_C1_star=1
           if Compute_Other_Endpoint==1:
                C1_star=sum(TestInstance_List_Inverse[0].c1_vector)
                Pareto_Optimal_List, Estimated_max_obj_2,Estimated_min_obj_1=Test_the_Instance(TestInstance_List_Inverse,C1_star,Pareto_Optimal_List,Actual_n_items,1,DeepModel)
                Min_C1_star=Estimated_min_obj_1
           C1_star= Estimated_max_obj_1-1   
           while C1_star>=Min_C1_star:
               print('Now testing on Instance %s with C1_star %d' %(filename,C1_star)) 
               Pareto_Optimal_List, Estimated_max_obj_main,Estimated_min_obj_other=Test_the_Instance(TestInstance_List_Main,C1_star,Pareto_Optimal_List,Actual_n_items,0,DeepModel)
               if Update_Cstar_Based_On_Last_Point==1:
                   C1_star= Estimated_max_obj_main-1
               else:   
                   C1_star= np.floor(C1_star*relative_decay_testing-absolute_decay_testing)
           if Bidirectional_Epsilon_Search==1:
               Min_C1_star=Estimated_min_obj_2
               C1_star=Estimated_max_obj_2-1
               while C1_star>=Min_C1_star:
                   print('Now testing on Instance %s with C1_star %d from Revserse side' %(filename,C1_star)) 
                   Pareto_Optimal_List, Estimated_max_obj_main,Estimated_min_obj_other=Test_the_Instance(TestInstance_List_Inverse,C1_star,Pareto_Optimal_List,Actual_n_items,1,DeepModel)
                   if Update_Cstar_Based_On_Last_Point==1:
                       C1_star= Estimated_max_obj_main-1
                   else:   
                       C1_star= np.floor(C1_star*relative_decay_testing-absolute_decay_testing)
           return  Pareto_Optimal_List         
######## Main LOOP #########

if Training_Active==1 and Testing_Active==0:
    DeepModel,Optimizer,TargetDeepModel=Training_LOOP()
          
if Testing_Active==1:        
      
    if Training_Active==0:    
       DeepModel,Optimizer,TargetDeepModel = Create_or_load_the_Deep_Learning_Model()
       DeepModel.eval()   
       
    for filename in os.listdir(TestingDirectory):         
        TestInstance_List_Main, TestInstance_List_Inverse,Actual_n_items=Create_Two_Lists_For_The_Test_Instance(TestingDirectory,filename,Beam_Search_Width,Compute_Other_Endpoint)
        Pareto_Optimal_List=[]
        start_testing_time=timer()
        
        if Training_Active==1:
            Obj_Constraint_RHS_Is_Given=1
            relative_decay=0
            absolute_decay=1
            ClassSizes=[]
            ClassSizes.append(int(TestInstance_List_Main[0].n_items))
            if WSM==0:
                C1_star=sum(TestInstance_List_Main[0].c1_vector)
            else:  
                C1_star=sum(TestInstance_List_Main[0].c1_vector)+sum(TestInstance_List_Main[0].c2_vector)
            Create_Traing_Instnaces_For_Active_Learning(TrainingDirectory,TestInstance_List_Main,Number_Of_Instances_For_Active_Learning,C1_star, WSM)
            DeepModel,Optimizer,TargetDeepModel=Training_LOOP()   
               
        Obj_Constraint_RHS_Is_Given=0    
        if WSM==0: 
           print("---- Testing Mode:Pseudo_Epsilon_Constraint_Method ----- ")    
           Pareto_Optimal_List=pseudo_epsilon_Constraint_Method(Pareto_Optimal_List,TestInstance_List_Main,TestInstance_List_Inverse,Update_Cstar_Based_On_Last_Point,relative_decay_testing,absolute_decay_testing,DeepModel,Obj_Constraint_RHS_Is_Given)
        else: 
           print("---- Testing Mode: Weighted_Sum_Method ----- ")      
           for increment in range(WSM_Loop_Size):
               Manual_Lambda_Value=((increment+1)/(WSM_Loop_Size+1)) 
               print('Iteration %d of WSM: Instance %s' %(increment+1, filename)) 
               C1_star=sum(TestInstance_List_Main[0].c1_vector)+sum(TestInstance_List_Main[0].c2_vector)
               Pareto_Optimal_List, Estimated_max_obj_main,Estimated_min_obj_other=Test_the_Instance(TestInstance_List_Main,C1_star,Pareto_Optimal_List,Actual_n_items,0,DeepModel)
        end_testing_time=timer()   
        duration_testing_time=end_testing_time-start_testing_time
        report_the_testing_results(filename, Actual_n_items, Pareto_Optimal_List,duration_testing_time)     



     


      
