# Big Data Final Project

1. I will use Google Natural Language API to classify text on the AI news data that Dr. Eckroth 
	provided us for the CREU project. 
	- running time for this step was approximately 5 hours for the entire dataset of AI news
	
2. After classifying those news articles, I use that data for training and testing sets
	to attempt to train a bag-of-words model for future classification
	- since there is not enough memory to train the model with the entire data set, I try to train 
		the model with a subset of the data. However, the accuracy score is pretty low, 
		below is the table of the accuracy score with training size = number lines in the csv file
		| training size  |  tf-idf |countVectorizer |  
		|----------------|---------|----------------|
		|          3000  |         |                | 		
		|          4000  |         |                | 
		|          5000  |   0.38  |     0.371      |
		|          6000  |   0.329 |     0.328      |
		|          7000  |   0.365 |     0.3859     |
		|          8000  |   0.350 |     0.378      |

	- I figure at this point, increasing the size of the training will not increase the accuracy, so 
		I suspect it has to do with the training data itself, so I did some exploratory checks:
		-- it seems like my training data is pretty skewed, for example, some cateogories would
			 have more than 4000 observations, but some only has 1. 
		-- the categories are also not uniformed, some categories are more detailed than the other,
			for example, one article is classfied as Arts & Entertainment/Fun & Trivia/Flash-Based 
			Entertainment, while some is simply "Reference"
3. Summary of tools:
	- Exploratory analysis: spark
		-- sample code to create a list of trained data from Google API
		-- sample code to train my own model from the trained data
		-- retrieve summary of trained data
	- Distributed workers: spark -- pipe articles to the python code
	- numeric/string processing: spark + google API + sklearn 
	- machine learning: sklearn in spark
