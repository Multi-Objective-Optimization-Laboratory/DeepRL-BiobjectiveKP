# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:54:01 2022

@author: Dr. Hadi Charkhgard

This code applies the epsilon-constraint method to any bi-objective LP-file 
located in a specific folder  (default: "InstancesToSolve"). The code computes 
the nondominated frontier for each bi-objective LP-file  and report its results
in another specific folder (default: "Results").  After solving each bi-objective LP-file,
Some high-level results will be also reported in an append-based CSV file 
named 'HighLevelResults.csv'.

The code is very generic; It is capable of solving any bi-objective Integer Linear Program
(not just bi-objective knapsack problems). We know that a bi-objective integer linear program
can be stated as follows: 

max cx
max dx
s.t. Ax<=b
    x in Z^n 
    
where c, d, A, b are all assumed to be integers.

The LP file corresponding to the above-mentioned bi-objective integer linear program 
should be in the following format. 
    
max -
s.t. y1=0
     y2=0
     y1=cx
     y2=dx
     AX<=b
     x in Z^n 
     y1, y2 in Z
     
  
In the LP file, y1=0  and y2=0 should have been named as 'obj1' and 'obj2', respectively. 
Otherwise, the code will NOT work. If they are named properly, the code will understand that 
y1 captures the value of the first objective function and y2 captures the value of 
the second objective function, and then it automatically removes y1=0 and y2=0.     j        

"""

import gurobipy as MyGurobi
from gurobipy import GRB
import pandas as pd
import numpy as np
import os
import timeit
import os.path



 
######## USER DEFINED PARAMETERS #########

InstanceDirectory='./InstancesToSolve/' #Folder name in which the instances that need to be solved are located
ResultsDirectory='./Results/' #Folder name to which the results will be generated
ScreenReportingByGurobi=0  #if zero means that logging is NOT allowed, one means otherwise
NumOfThreadsByGurobi=1  #how many threads are allowed to be used by Gurobi? 1 means that arallelization is NOT allowed.
RelativeOptimalityGap=1e-5  #OptimalityGapTolerence This value should be smaller than or equal to 1e-4
epsilon=0.8  #epsilon value in the epsilon the constraint method; Set 0 < epsilon <1. Default is 0.8  
Negative_Infinity=-1e+8
######## Read Model #########

def read_model(InstanceDirectory,filename):
    MyModel=MyGurobi.read(InstanceDirectory+filename)  
    const=MyModel.getConstrByName('obj1') #Getting the constraint correspounding to the First objective 
    obj1=MyModel.getRow(const) #Changing the first constraint into a mathematical expression defining obj1
    MyModel.remove(const) #Removing the constraint correspounding to the First objective
    const=MyModel.getConstrByName('obj2') #Getting the constraint correspounding to the Second objective
    obj2=MyModel.getRow(const)  #Changing the second constraint into a mathematical expression defining obj2
    MyModel.remove(const) #Removing the constraint correspounding to the Second objective
    ModelVars = MyModel.getVars() #Getting all the variables of the model
    return  MyModel, obj1, obj2, ModelVars

######## Initilizing the Parameters of Gurobi #########

def setting_gurobi_parameters(MyModel):
        MyModel.setParam(GRB.Param.LogToConsole, ScreenReportingByGurobi) #Turning off all logging (no progress on the screen will be shown) 
        MyModel.setParam(GRB.Param.Threads, NumOfThreadsByGurobi) #Choosing the number of threads
        MyModel.setParam(GRB.Param.MIPGap, RelativeOptimalityGap) #Choosing the value of the relative optimality gap
        return 

######## Solving The Model Using Gurobi #########

def solve_the_model(MyModel):
           MyModel.optimize()
           if MyModel.Status == GRB.OPTIMAL:
                return MyModel, 1
           elif MyModel.Status == GRB.INF_OR_UNBD:
                print('ATTENTION: Model is infeasible or unbounded')
                return MyModel, 2
           elif MyModel.Status == GRB.INFEASIBLE:
                return MyModel, 2
           elif MyModel.Status == GRB.UNBOUNDED:
                print('ERROR: Model is unbounded')
                return MyModel, 3
           else:
                print('ATTENTION: Optimization ended with status %d' % MyModel.Status)
                return MyModel, 1

######## Solving The Model Using Gurobi #########     
            
def get_optimal_solution_and_objective_values(ModelVars):
      Solution=[]
      obj1_val=obj1.getValue()
      obj2_val=obj2.getValue()
      for i in range(len(ModelVars)):
          Solution.append(ModelVars[i].X)
      return obj1_val, obj2_val, Solution

                        
######## Giving a warm-start feasible solution (primal bound) to Gurobi to improve its performance #########     
            
def provide_warm_start(Solution, ModelVars):
      for i in range(len(ModelVars)):
          ModelVars[i].Start = Solution[i] 
      return       
     
######## Lexicographic Operation #########
def Lexicographic_Optimizer(MyModel, ModelVars, obj1, obj2, LB_Obj2):
     Pareto_Solution=[]
     z_star_1, z_star_2= Negative_Infinity,Negative_Infinity
     LB_obj2_constr=MyModel.addConstr(obj2>=LB_Obj2, "LB_obj2")
     MyModel.setObjective(obj1,GRB.MAXIMIZE)
     MyModel,Status= solve_the_model(MyModel)
     if Status==1:
         z_star_1, z_2, Solution = get_optimal_solution_and_objective_values(ModelVars)
         MyModel.remove(LB_obj2_constr)
         LB_obj1_constr=MyModel.addConstr(obj1>=z_star_1- epsilon, "LB_obj1")
         MyModel.setObjective(obj2,GRB.MAXIMIZE)
         provide_warm_start(Solution, ModelVars)
         MyModel,Status= solve_the_model(MyModel)
         if Status==2:
              print('ERROR: There is a numerical issue because of obj1>=z_star_1 - epsilon')  
              Lexico_Status=3   #Leixcographic Operation Run into a numerical issue
         else:
              Lexico_Status=1   #Leixcographic Operation is Feasible
              z_star_1, z_star_2, Pareto_Solution = get_optimal_solution_and_objective_values(ModelVars)
              MyModel.remove(LB_obj1_constr)
     else: 
         Lexico_Status=2  #Leixcographic Operation is Infeasible 
     return MyModel, ModelVars, obj1, obj2, Lexico_Status, z_star_1, z_star_2, Pareto_Solution
 
    
######## Class of Nondominated Points #########

class Nondominated_Point:
  def __init__(self,obj1,obj2,solution):
    self.obj1 = obj1
    self.obj2 = obj2
    self.solution = solution

######## Report The Results #########
def report_the_results(filename, Nondominated_Frontier,runtime,Variable_Names):
 
    Number_of_Points_Found=len(Nondominated_Frontier)
 
    New_Row = {'File_Name':[filename],
        'run_time':[runtime],
        'Number_Points':[Number_of_Points_Found]
       }
    HighLevel_df = pd.DataFrame(New_Row)
    if os.path.exists('HighLevelResults.csv')==1:
        HighLevel_df.to_csv('HighLevelResults.csv', mode='a', index=False, header=False)
    else:
        HighLevel_df.to_csv('HighLevelResults.csv', mode='a', index=False, header=True)
     
    new_row=['obj1', 'obj2']
    for var in Variable_Names:
           new_row.append(var)     
    Detailed_df = pd.DataFrame(new_row)  
    Detailed_df=Detailed_df.transpose()
    
    Number_of_Variables=len(Variable_Names)
    for i in range(Number_of_Points_Found):
        new_row=[Nondominated_Frontier[i].obj1,Nondominated_Frontier[i].obj2]
        for j in range(Number_of_Variables):
             new_row.append(Nondominated_Frontier[i].solution[j]) 
        Detailed_df.loc[len(Detailed_df)] =  new_row
    name_main_part = filename[:-3]  #filename without '.lp'
    report_file_name= name_main_part+'_Results.csv'
    Detailed_df.to_csv(ResultsDirectory+report_file_name, index=False, header=False)
    return


######## Epsilon Constraint Method #########

def Epsilon_Constraint_Method(filename, MyModel, ModelVars, obj1, obj2):
    start_time = timeit.default_timer()
    Nondominated_Frontier=[]
    LB_Obj2=Negative_Infinity
    Lexico_Status=1
    while(Lexico_Status==1):
         MyModel, ModelVars, obj1, obj2, Lexico_Status, z_star_1, z_star_2, Pareto_Solution= Lexicographic_Optimizer(MyModel, ModelVars, obj1, obj2, LB_Obj2)
         if(Lexico_Status==1):
            New_Point=Nondominated_Point(z_star_1, z_star_2, Pareto_Solution) 
            Nondominated_Frontier.append(New_Point)
            LB_Obj2=z_star_2+epsilon 
    termination_time = timeit.default_timer()
    runtime= termination_time-start_time   
    Variable_Names= MyModel.getAttr('VarName', ModelVars)    
    report_the_results(filename, Nondominated_Frontier,runtime,Variable_Names)
    return
 
######## Main LOOP #########
for filename in os.listdir(InstanceDirectory):           
            MyModel, obj1, obj2, ModelVars= read_model(InstanceDirectory,filename)
            setting_gurobi_parameters(MyModel)
            print('Now Solving %s' %filename)
            Epsilon_Constraint_Method(filename, MyModel, ModelVars, obj1, obj2)
            
           
          
          