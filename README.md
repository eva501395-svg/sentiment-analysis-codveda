#  Social Media Sentiment & Engagement Analysis
### Codveda Technology | Data Science Internship Project

##  Problem Formulation
Social media platforms generate massive volumes of unstructured text and engagement metrics daily. Understanding the underlying sentiment of posts and predicting user engagement (Likes, Retweets) is crucial for brand reputation management and content strategy. 

**Objective:** To build a comprehensive data science pipeline that cleans, explores, and models social media data to predict engagement and classify text sentiment using Machine Learning and Deep Learning techniques.

---

##  Dataset Description
The project utilizes the **Social Media Sentiment Dataset** (`Sentiment_dataset.csv`), containing **733 records** of social media posts. 
- **Features:** Text content, Timestamp, User, Platform, Hashtags, Retweets, Likes, Country, and temporal features (Year, Month, Day, Hour).
- **Target Variables:** `Sentiment` (Categorical: Positive, Negative, Neutral, and fine-grained emotions), `Likes`/`Retweets` (Numerical).

---

## ️ Tech Stack
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (Linear Regression, Random Forest, Logistic Regression, Naive Bayes)
- **Deep Learning:** TensorFlow, Keras
- **NLP:** NLTK, TF-IDF Vectorization

---

##  Project Tasks & Key Findings

###  Level 1: Basic (Data Cleaning & EDA)
- **Data Cleaning:** Handled missing values, removed outliers using the IQR method, applied One-Hot Encoding, and scaled numerical features using `StandardScaler`.
- **EDA Insights:** 
  -  **Correlation Matrix:** Identified a **perfect positive correlation (1.00)** between `Retweets` and `Likes`. Data inspection reveals Likes are consistently exactly double the Retweets.
  -  **Temporal Analysis:** The time of posting (Hour, Month, Year) showed near-zero correlation with post popularity, indicating content quality drives engagement more than timing in this dataset.

###  Level 2: Intermediate (Predictive Modeling & Classification)
- **Regression Analysis:** Predicted engagement metrics using Linear Regression and Random Forest Regressor. Due to the strict 1:2 linear ratio between Retweets and Likes, Linear Regression perfectly captured the underlying mathematical relationship.
- **Classification:** Built a Logistic Regression model to classify post categories.
  -  **Finding:** Achieved an **Accuracy of 92.50%**, a **Recall of 96.77%**, and a near-perfect **ROC-AUC of 99.06%**, demonstrating excellent discriminative power and a well-calibrated decision boundary.

###  Level 3: Advanced (NLP & Neural Networks)
- **NLP Text Classification:** Preprocessed text (tokenization, stopword removal, lemmatization) and converted it to numerical vectors using **TF-IDF**. Trained a **Multinomial Naive Bayes** classifier to predict sentiment.
- **Neural Networks:** Designed a Sequential Feed-Forward Neural Network using **TensorFlow/Keras** (Dense layers, ReLU activation, Dropout for regularization). 
  -  **Finding:** The model successfully learned from the data, reducing loss consistently from `1.15` to `1.02` over 5 epochs and achieving a **Test Accuracy of 52.5%** (significantly beating the 33% random baseline for 3-class classification).

---

##  Business Insights & Recommendations
1. **Handle Multicollinearity:** Because `Likes` and `Retweets` have a 1.00 correlation, one must be dropped before feeding the data into regression models to prevent mathematical instability.
2. **Time is Not a Predictor:** The time of posting (Hour, Month, Year) has almost zero correlation with post popularity in this dataset. Content quality matters more than timing.
3. **Model Selection:** For highly linear engagement data, simple models like Linear Regression are highly effective and computationally cheaper than complex tree-based models.
4. **Future Scope:** To improve the Neural Network and NLP models, I recommend scaling up to a larger dataset, implementing `EarlyStopping` callbacks, and experimenting with Transformer models like BERT for better context understanding.

---





|  Folder / File |  Description |
| :--- | :--- |
| **Root Directory** | |
| `README.md` | Project overview, findings, and instructions |
| `requirements.txt` | List of Python dependencies |
| `Sentiment_dataset.csv` | Raw dataset (733 rows) |
| | |
| ** Level 1: Basic** | |
| `1_Data_Cleaning_Preprocessing.ipynb` | Handling missing values, outliers, and scaling |
| `2_Exploratory_Data_Analysis.ipynb` | Summary stats, histograms, and correlation matrix |
| | |
| ** Level 2: Intermediate** | |
| `1_Predictive_Modeling_Regression.ipynb` | Linear Regression vs Random Forest (R² = 0.98) |
| `2_Classification_Logistic_Regression.ipynb` | Logistic Regression (Accuracy = 92.5%) |
| | |
| ** Level 3: Advanced** | |
| `1_NLP_Text_Classification.ipynb` | Text preprocessing, TF-IDF, and Naive Bayes |
| `2_Neural_Networks_Keras.ipynb` | Feed-forward Neural Network with TensorFlow |
##  Acknowledgments & Thank You
I would like to express my sincere gratitude to **Codveda Technology** for providing me with this incredible Data Science internship opportunity. This project has been a transformative learning experience, allowing me to bridge the gap between theoretical knowledge and real-world application. 

Thank you to the entire Codveda team and my mentors for your continuous guidance, support, and for fostering an environment that encourages innovation and skill development. I am excited to apply these newly acquired skills in Python, Machine Learning, and Deep Learning to future projects!

- **Website:** [www.codveda.com](https://www.codveda.com)
- **Email:** support@codveda.com

