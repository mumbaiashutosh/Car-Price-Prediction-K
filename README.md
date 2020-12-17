# Car-Price-Prediction
After importing the needed libraries - pandas, seaborn, matplotlib, numpy
![Capture](https://user-images.githubusercontent.com/56570977/102513280-06a81580-40b1-11eb-8cc0-c92a4ef105be.JPG)
There are mainly 4 categorical variables - Seller_Type, Transmission, Ownerand Fuel_Type
We also check for missing values
Remove the CarName feature in the new data frame since it will not play  key role
Using the 'Year' feature, we create a new feature to calculate the Age of the car
Removing Year and Current Year since they are not needed anymore
#Converting the three variables into One Hot Encoded
Then we check out the heatmap for the correlation
![HeatMap](https://user-images.githubusercontent.com/56570977/102541677-b774db80-40d6-11eb-8ccc-6e8498b7bff3.JPG)
#independent & dependent features creation - Selling price is the dependent feature
X=final_dataset.iloc[:,1:]
Y=final_dataset.iloc[:,0]
#plot graph of feature importances for better visualization 
#Splitting into Train and Test
We use RandomForestRegressor() and optimize the parameters using RandomizedSearchCV
Then predictions=rf_random.predict(X_test)
sns.distplot(Y_test-predictions) --> plot the SD of the difference between Test and Prediction
![Normal Dis](https://user-images.githubusercontent.com/56570977/102542036-3538e700-40d7-11eb-91ba-f5b4fba08412.JPG)
plt.scatter(Y_test,predictions)
![Linear Dist](https://user-images.githubusercontent.com/56570977/102542055-3b2ec800-40d7-11eb-9720-7458ee9a0ccc.JPG)
Make the pickle file
