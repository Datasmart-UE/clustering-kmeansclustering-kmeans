Clustering with K-Means - Iris Dataset
This project performs an unsupervised clustering analysis using the K-Means algorithm on the classic Iris dataset.
Overview
The main objective of this project is to group the observations into clusters based on their features without using the labels. The Iris dataset consists of measurements of sepals and petals of three different species of iris flowers.
Steps
	1	Dataset Download: Automatically downloads the Iris dataset from a public repository.
	2	Data Loading: Loads the CSV file into a Pandas DataFrame and assigns appropriate column names.
	3	Data Preparation: Prepares numerical variables for clustering (excluding the species label).
	4	K-Means Clustering: Applies the K-Means algorithm to identify 3 clusters.
	5	Visualization: Plots the resulting clusters to visualize the separation between them.
	6	Model Evaluation: Calculates and displays the Sum of Squared Errors (SSE) within clusters to evaluate the clustering performance.
Technologies Used
	•	Python
	•	Pandas
	•	NumPy
	•	Matplotlib
	•	Scikit-learn
Files
	•	iris_kmeans.py: Python script containing the full code to perform the clustering analysis.
	•	README.md: Project documentation (this file).
Author
Andres Martin Llases 

This project was created for educational purposes and to practice clustering techniques with Python and machine learning libraries.
