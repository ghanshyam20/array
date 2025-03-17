
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Step 0: Read data into a pandas dataframe
data_banknote = pd.read_csv('data_banknote_authentication.csv')

# Step 1: Pick the column named "class" as target variable y and all other columns as feature variables X
X_banknote = data_banknote.drop(columns=['class'])
y_banknote = data_banknote['class']

# Step 2: Split the data into training and testing sets with 80/20 ratio and random_state=20
X_train_banknote, X_test_banknote, y_train_banknote, y_test_banknote = train_test_split(X_banknote, y_banknote, test_size=0.2, random_state=20)

# Step 3: Use support vector classifier with linear kernel to fit to the training data
svc_linear = SVC(kernel='linear')
svc_linear.fit(X_train_banknote, y_train_banknote)

# Step 4: Predict on the testing data and compute the confusion matrix and classification report
y_pred_linear = svc_linear.predict(X_test_banknote)
print("Linear Kernel SVM")
print(confusion_matrix(y_test_banknote, y_pred_linear))
print(classification_report(y_test_banknote, y_pred_linear))

# Step 5: Repeat steps 3 and 4 for the radial basis function kernel
svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(X_train_banknote, y_train_banknote)
y_pred_rbf = svc_rbf.predict(X_test_banknote)
print("RBF Kernel SVM")
print(confusion_matrix(y_test_banknote, y_pred_rbf))
print(classification_report(y_test_banknote, y_pred_rbf))



# To learn machine learning, you can follow these steps:
# 1. Understand the basics of programming, especially Python.
# 2. Learn about fundamental concepts in statistics and linear algebra.
# 7. Join machine learning communities and participate in competitions like Kaggle.
#    - Kaggle is a great platform to practice your skills, learn from others, and participate in competitions that can help you gain real-world experience.
#    - Engage in forums and discussions to get insights and tips from other practitioners.

# 8. Stay updated with the latest research by reading papers from conferences like NeurIPS, ICML, and CVPR.
#    - These conferences publish cutting-edge research in machine learning and AI. Reading these papers can help you stay informed about the latest advancements.
#    - Follow arXiv.org for preprints of the latest research papers.

# 9. Experiment with different models and hyperparameters to understand their impact.
#    - Try different algorithms and tweak their hyperparameters to see how they affect model performance.
#    - Use tools like GridSearchCV or RandomizedSearchCV in scikit-learn to automate hyperparameter tuning.

# 10. Learn about model evaluation techniques and how to avoid overfitting.
#    - Understand metrics like accuracy, precision, recall, F1-score, ROC-AUC, etc.
#    - Learn techniques like cross-validation, regularization, and early stopping to prevent overfitting.

# 11. Understand the importance of data preprocessing and feature engineering.
#    - Data cleaning, normalization, and transformation are crucial steps before feeding data into a model.
#    - Feature engineering involves creating new features or modifying existing ones to improve model performance.

# 12. Explore advanced topics like deep learning, reinforcement learning, and natural language processing.
#    - Deep learning involves neural networks with many layers and is used for tasks like image and speech recognition.
#    - Reinforcement learning is used for decision-making tasks where an agent learns by interacting with the environment.
#    - Natural language processing (NLP) involves working with text data for tasks like sentiment analysis, translation, and chatbots.

# 13. Collaborate with others on projects to gain different perspectives and insights.
#    - Working in teams can help you learn from others, share knowledge, and tackle more complex problems.
#    - Use platforms like GitHub to collaborate on code and manage projects.

# 14. Keep practicing and never stop learning, as the field of machine learning is constantly evolving.
#    - The field is rapidly changing, with new techniques and tools being developed regularly.
#    - Continuous learning and practice are essential to stay current and improve your skills.

# 15. Use version control systems like Git to manage your code and track changes.
#    - Version control helps you keep track of changes, collaborate with others, and revert to previous versions if needed.
#    - Learn the basics of Git and use platforms like GitHub or GitLab for your projects.

# 16. Learn about cloud platforms like AWS, Google Cloud, and Azure for scalable machine learning solutions.
#    - Cloud platforms offer services for data storage, model training, and deployment at scale.
#    - Familiarize yourself with services like AWS SageMaker, Google AI Platform, and Azure Machine Learning.

# 17. Study the ethical implications of machine learning and AI.
#    - Understand the potential biases and ethical concerns related to AI and machine learning.
#    - Learn about fairness, accountability, and transparency in AI systems.

# 18. Understand the deployment process of machine learning models in production.
#    - Learn how to deploy models as APIs or integrate them into applications.
#    - Understand concepts like model versioning, monitoring, and scaling in production environments.

# 19. Learn about different data visualization techniques to better understand your data.
#    - Visualization helps in exploring data, identifying patterns, and communicating results.
#    - Use libraries like Matplotlib, Seaborn, and Plotly for creating visualizations.

# 20. Participate in hackathons to challenge yourself and improve your skills.
#    - Hackathons are time-bound events where you work on real-world problems and compete with others.
#    - They provide a great opportunity to apply your skills, learn new ones, and network with other professionals.

# 21. Follow influential machine learning practitioners and researchers on social media.
#    - Follow experts on platforms like Twitter, LinkedIn, and Medium to get insights, tips, and updates on the latest trends.
#    - Engage with the community by participating in discussions and sharing your knowledge.

# 22. Take online courses and attend workshops to deepen your knowledge.
#    - Platforms like Coursera, edX, Udacity, and DataCamp offer courses on various machine learning topics.
#    - Attend workshops and webinars to learn from experts and gain hands-on experience.

# 23. Read books on machine learning to get a comprehensive understanding of the field.
#    - Some recommended books include "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron, "Pattern Recognition and Machine Learning" by Christopher Bishop, and "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.

# 24. Practice coding regularly to improve your programming skills.
#    - Regular coding practice helps you become proficient in writing efficient and clean code.
#    - Use platforms like LeetCode, HackerRank, and CodeSignal to practice coding problems.

# 25. Learn about the hardware requirements and optimizations for training large models.
#    - Training large models often requires powerful hardware like GPUs and TPUs.
#    - Understand how to optimize your code and use hardware acceleration to speed up training.

# 26. Understand the role of machine learning in different industries and applications.
#    - Machine learning is used in various industries like healthcare, finance, retail, and manufacturing.
#    - Explore how machine learning is applied in different domains and the impact it has.

# 27. Get familiar with the concept of transfer learning and how to apply it.
#    - Transfer learning involves using a pre-trained model on a new, but related task.
#    - It helps in achieving good performance with limited data and reduces training time.

# 28. Learn about the different types of machine learning, such as supervised, unsupervised, and semi-supervised learning.
#    - Supervised learning involves training a model on labeled data.
#    - Unsupervised learning involves finding patterns in unlabeled data.
#    - Semi-supervised learning uses a combination of labeled and unlabeled data.

# 29. Explore the use of machine learning in big data analytics.
#    - Big data analytics involves processing and analyzing large datasets to extract insights.
#    - Learn about tools and frameworks like Hadoop, Spark, and Flink for big data processing.

# 30. Stay curious and keep exploring new tools and techniques in the field.
#    - The field of machine learning is vast and constantly evolving.
#    - Stay curious, experiment with new tools, and keep learning to stay ahead.

# 31. Understand the importance of reproducibility in machine learning experiments.
#    - Reproducibility ensures that your experiments can be replicated by others.
#    - Use tools like Jupyter Notebooks, Docker, and MLflow to document and share your experiments.

# 32. Learn about the different types of data augmentation techniques.
#    - Data augmentation involves creating new training examples by modifying existing ones.
#    - Techniques include rotation, scaling, cropping, and flipping for images, and adding noise or changing pitch for audio.

# 33. Explore the use of machine learning in edge computing and IoT devices.
#    - Edge computing involves processing data on devices closer to where it is generated, reducing latency and bandwidth usage.
#    - Learn about deploying machine learning models on IoT devices and edge servers.

# 34. Study the impact of machine learning on privacy and security.
#    - Understand the privacy concerns related to data collection and model predictions.
#    - Learn about techniques like differential privacy and federated learning to address these concerns.

# 35. Learn about the different types of optimization algorithms
# 35. Learn about the different types of optimization algorithms used in machine learning.
#    - Optimization algorithms are used to minimize the loss function during model training.
#    - Common algorithms include gradient descent, stochastic gradient descent, and Adam.

# 36. Understand the concept of explainable AI and its importance.
#    - Explainable AI involves making the decisions of machine learning models understandable to humans.
#    - Learn about techniques like LIME, SHAP, and model interpretability methods.

# 37. Explore the use of machine learning in healthcare and other critical domains.
#    - Machine learning is used in healthcare for tasks like disease diagnosis, treatment planning, and drug discovery.
#    - Understand the challenges and ethical considerations in applying machine learning to critical domains.

# 38. Learn about the different types of neural network architectures.
#    - Neural networks come in various architectures like Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformers.
#    - Each architecture is suited for different types of tasks, such as image processing, sequence modeling, and natural language understanding.

# 39. Study the impact of bias and fairness in machine learning models.
#    - Bias in machine learning models can lead to unfair and discriminatory outcomes.
#    - Learn about techniques to detect and mitigate bias, and ensure fairness in your models.

# 40. Understand the importance of cross-validation in model evaluation.
#    - Cross-validation involves splitting the data into multiple folds and training the model on different subsets.
#    - It helps in assessing the model's performance and ensuring it generalizes well to unseen data.










