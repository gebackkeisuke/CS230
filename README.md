# CS230

This repository contains the code for the 2025 CS230 project, owned by Keisuke Yamamura, a second-year PhD student at Stanford. The project is organized into five phases. 

1. The first phase automatically generates INPUT files to run thousands of commercial numerical simulations with varying parameters.  
2. The second phase monitors the progress of these simulations on Sherlock, checking whether they have started, completed, or encountered any errors.  
3. The third phase is post-processing: it extracts the INPUT and OUTPUT features needed for deep learning from the simulation results and converts them into binary files for fast read/write operations.  
4. The fourth phase is the deep learning pipeline itself, including dataset creation, dataloader setup, model definition, and training using PyTorch.  
5. The fifth phase covers evaluation and inference.  

In this repository, I have uploaded the code for the last two phases.
