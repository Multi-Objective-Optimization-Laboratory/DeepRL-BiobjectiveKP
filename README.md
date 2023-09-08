# DeepRL-BiobjectiveKP

This repository contains Python code and resources for solving bi-objective knapsack problems using deep reinforcement learning techniques. It accompanies the paper titled "Solving Hard Bi-objective Knapsack Problems Using Deep Reinforcement Learning" by Dr. Hadi Charkhgard and co-authors.

## Repository Contents

1. **Exact Augmented Epsilon Constraint Method**
   - Python implementation of the Augmented Epsilon Constraint Method.
   - Utilizes the Gurobi optimizer.
   - Input: LP (Linear Programming) files.
   - LP files can be generated using the tools in the "Instance Generator" folder.

2. **Instance Generator**
   - Python code for generating multiple problem instances.
   - Generates instances in CSV and LP formats.
   - CSV: Suitable for Deep Reinforcement Learning.
   - LP: For the Exact Augmented Epsilon Constraint Method.

3. **Trained Models in PyTorch Format**
   - Pre-trained deep learning models stored in PyTorch's "pt" format.
   - Applicable to both proposed deep RL-based methods discussed in the paper.
   - Use for testing and experimentation.

4. **Main-Proposed Deep RL Methods Implementation**
   - Implementation of the proposed deep RL-based methods in testing mode.
   - Choose and run the code of your desired method.
   - Includes testing instances and pre-trained deep RL models.

## Citation

If you use this repository or the techniques from the paper in your research, kindly cite the original paper authored by Dr. Hadi Charkhgard and co-authors.


