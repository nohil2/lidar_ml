# lidar_ml
This project is an attempt to use a machine learning algorithm to classify lidar points. It uses a random forest classifier to do so, as from research, this seems to be one of the best algorithms to use for this.
The model uses features from the original lidar data, as well as derived data such as statistics gathered through data binning. 

These lidar points are classified following the classification codes in LAS format 1.1-1.4, such as "1" being "ground" and "5" being "high vegetation", etc.

# Installation
This project was written in Python 3.9. The required libraries are in requirements.txt. You can use pip to install them.
```pip install -r requirements.txt```

# Dataset
The dataset used to train and test the classifier in this project was obtained via OpenTopography and is only a subset of the original.

Thompson, S. (2021). Illilouette Creek Basin Lidar Survey, Yosemite Valley, CA 2018. National Center for Airborne Laser Mapping (NCALM). Distributed by OpenTopography. https://doi.org/10.5069/G96M351N.. Accessed: 2023-11-19

# Other citations
Basener, William & Basener, Abigail. (2017). Classification and identification of small objects in complex urban-forested LIDAR data using machine learning. 10.1117/12.2264641. 
