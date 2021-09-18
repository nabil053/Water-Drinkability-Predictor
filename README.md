# Water-Drinkability-Predictor

A simple logistic regression model that predicts whether the sample water is drinkable or not based on a set of attributes.

The dataset used here is called water_potability.csv and it includes descriptions of various chemical properties of water samples. The dataset has 
a total of 9 attributes (class excluded), which are pH values, hardness, solids, chloramines, sulfates, conductivity, organic carbon, trihalomethanes
and turbidity.

The dataset used here contains a total of 3276 sample data. The dataset is further modified at runtime by purging it of data containing null values.
Since class 0 has overwhelmingly more sample data (1200) than class 1 (811) after purging, hence the excess class 0 data (389) are removed, bringing 
the total to 1622. This final dataset is further split into training data (973, 60%) for training the model, validation data (324, 20%) for cross 
validation and testing data (325, 20%) for testing the model's metrics.

The generated model has the ability to predict water's drinkability with an accuracy of 36.00%, and its precision and recall are 34.42% and 94.64% respectively.
