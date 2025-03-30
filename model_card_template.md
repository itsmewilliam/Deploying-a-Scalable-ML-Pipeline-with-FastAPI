# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

---

## Model Details

- **Model Type**: RandomForestClassifier
- **Library Used**: scikit-learn
- **Trained By**: William Gonzalez (as part of a learning project)
- **Date Trained**: March 29th, 2025
- **Script Used**: `train_model.py`

This model was trained as part of a project to learn how to build, evaluate, and deploy machine learning models using FastAPI.

---

## Intended Use

The model is meant for educational purposes. It predicts if a person earns more than $50K per year based on their personal and job-related information.

This project is part of my learning experience in understanding how ML pipelines work, how to evaluate models, and how to deploy them using APIs.

---

## Training Data

The data comes from the U.S. Census Bureau. It includes over 32,000 records with features like:

- age
- workclass
- education
- marital status
- occupation
- relationship
- race
- sex
- capital gain/loss
- hours-per-week
- native country

The label we’re trying to predict is whether a person earns more than $50K (`>50K`) or not (`<=50K`).

---

## Evaluation Data

I used 20% of the original dataset as the test set to check how well the model performs. I used the same preprocessing steps as the training data using one-hot encoding and label binarization.

---

## Metrics

Here are the main metrics I used to check the model:

- **Precision**: Measures how many of the people the model predicted as `>50K` were actually correct.
- **Recall**: Measures how many of the actual `>50K` people were correctly found by the model.
- **F1 Score**: A balance between precision and recall.

### Overall Model Performance:

- **Precision**: 0.7419  
- **Recall**: 0.6384  
- **F1 Score**: 0.6863

I also checked the model's performance on different groups like race, sex, education, etc., and saved that in `slice_output.txt`.

---

## Ethical Considerations

- The data comes from real census info and might include biases (like gender or racial bias).
- Some groups in the dataset are much smaller than others, which could make the model less accurate for them.
- I’m still learning, so this model is not perfect and shouldn’t be used for real decisions.

---

## Caveats and Recommendations

- This model is for learning only — not for production.
- The dataset is imbalanced (most people earn `<=50K`).
- Some slices (like "Preschool" education) had very few examples, so results may not be reliable there.
- If I continue working on this, I’d like to explore:
  - Balancing the dataset
  - Trying different models
  - Exploring fairness metrics more deeply
