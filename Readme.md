# Vulnerable User Density Exposure Risk Dashboard
### Chenfeng Xiong, Jina Mahmoudi, Mofeng Yang, Weiyu Luo
The Vulnerable User Density Exposure Risk Dashboard provides data and insight on volumes and exposure to transportation safety risks of vulnerable road users, e.g., pedestrians, bicycles, and e-scooters at intersections and roadway segments within Maryland. In this repository, we provide some demo codes of three functions in the dashboard.
## Functions 
1. Crash frequency modeling. 
* Check the folder "Crash_model". Frequency of pedestrian and bicyclist crashes at Maryland intersections and road segments have been estimated using one of the most commonly used crash frequency modeling methodology: Zeroinflated Negative Binomial (ZINB) regression techniques. Please try to run the STATA codes. 
2. Crash model applying. 
* Check the folder "Crash_predict".  After the model training process, we may expand and apply the model on all the intersections and roadway segments in Maryland. Try to copy and paste the coefficients into "coef_int_v1.0.csv" and "coef_link_v1.0.csv" (please keep the original format), then run the python scripts to apply the model and predict. 
* "zinb_intersec_apply_v1.0.py": apply the intersection model. 
* "zinb_link_apply_v1.0.py": apply the link model. 
3. Volume visualization. 
* Vehicle Volume and Pedestrian/Bike Volume is visualized by several steps execution of python scripts. Please run the scripts in the order below: 
* "1-VolumeToGeohash.py": Convert volume data into count data of all geohash boxes in Maryland.
* "2-GeohashToImage.py": Convert geohash counts into images.  

## Reference
* For more information about the dashboard, please click:
https://mti.umd.edu/SDI
