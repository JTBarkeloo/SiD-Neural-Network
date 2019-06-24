# SiD-Neural-Network
First attempt at creating a neural network for measurement of leakage for the Silicon Detector at the International Linear Collider
Needs to have lcgenv.sh and tflow.sh setup to work on ATLAS Tier-3 or lxplus
(or sklearn/keras installed locally, preferably with tensorflow but not necessary)

Will loop over a varying number of input layers of the calorimeter event file, here a pickle file, and do a fit based on given neural network architecture.  

The goal of this project is to accurately identify how much leakage energy exist based on  the least amount of input layers (for cost optimization of the SiD Electromagnetic Calorimeter) as well as be the first introduction to lab undergraduates about how to read and input into pandas dataframes and running various neural networks with straight forward bits to change to improve the network preformance.

