# Employee Sentiment Analysis

This project performs end-to-end sentiment analysis on internal corporate email communications, using modern NLP techniques to assess engagement, flag potential flight risks, and build predictive models. The workflow combines state-of-the-art transformer models, robust feature engineering, rigorous validation, and business-focused analytics.

---

## 📁 Project Structure

```
Employee_Sentiment_Analysis/
├── employee_sentiment_analysis.ipynb    # Main Jupyter notebook with full analysis pipeline
├── sentiment_labeled_data.csv           # All messages with final sentiment labels (for both models)
├── labeled_sentiments_roberta.csv       # Raw output from CardiffNLP Roberta model
├── labeled_sentiments_nlptown.csv       # Raw output from NLP Town BERT model
├── monthly_employee_scores.csv          # Sentiment score per employee per month
├── regression_features.csv              # Feature table for regression modeling
├── figures/                             # PNGs of all figures and plots
├── Final Report.pdf                     # Comprehensive write-up with results and interpretation
├── README.md                            # This file
```

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/HAK978/Springer_Capital_Assessment
cd Employee_Sentiment_Analysis
```

### 2. Create a virtual environment

```bash
conda create -n sentiment python=3.10
conda activate sentiment
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- transformers
- torch >= 2.6.0
- pandas
- matplotlib
- seaborn
- scikit-learn
- nltk

If using GPU, ensure your CUDA and torch versions are compatible (e.g., `torch==2.6.0+cu121`).

## ▶️ Usage

### Run the notebook

```bash
jupyter notebook employee_sentiment_analysis.ipynb
```

Place all input data files (e.g., `emails.csv` or `test.csv`) in the working directory.

**Outputs:**
- `sentiment_labeled_data.csv`: Every message, labeled Positive/Neutral/Negative for both models.
- `labeled_sentiments_roberta.csv`: Initial sentiment results from the CardiffNLP Roberta model.
- `labeled_sentiments_nlptown.csv`: Sentiment results from NLP Town model.
- `monthly_employee_scores.csv`: Aggregate sentiment scores for each employee per month.
- `regression_features.csv`: Feature set for regression modeling.
- `figures/`: Bar plots, trends, rankings, agreement rates, etc.

## 🏅 Top Employees and Flight Risks

### Top 3 Positive Employees (Sample, Jan 2010, CardiffNLP)
1. eric.bass
2. don.baughman
3. kayne.coulter

### Flight Risk Employees

**CardiffNLP Model:**
- don.baughman
- eric.bass
- john.arnold
- rhonda.denton

**NLP Town Model:**
- bobette.riner
- don.baughman
- eric.bass
- john.arnold
- johnny.palmer
- kayne.coulter
- lydia.delgado
- patti.thompson
- rhonda.denton
- sally.beck

## 📌 Key Insights and Recommendations

- Roberta model classifies most messages as neutral (68.7%), NLP Town as positive or negative.
- Flight risk detection consistently flags a small group of employees.
- Regression modeling shows limited but useful predictive power; sentiment ratios and message count are most informative.
- Manual spot checks and cross-model validation add confidence to labeling; further domain fine-tuning is recommended for production use.

## 📊 Methodology Overview

### 1. Sentiment Classification (Dual-Model)
- Used both `cardiffnlp/twitter-roberta-base-sentiment` (RoBERTa, trained on tweets) and `nlptown/bert-base-multilingual-uncased-sentiment` (BERT, trained on reviews) from Hugging Face Transformers.
- For each message, concatenated subject and body fields and ran through both models.
- **Label mappings:**
  - **Roberta:** LABEL_0 → Negative, LABEL_1 → Neutral, LABEL_2 → Positive
  - **NLP Town:** 1-2 stars → Negative, 3 → Neutral, 4-5 → Positive
- Results from both models are compared, and manual spot-checks were performed on cases where models disagreed to ensure quality.

### 2. Exploratory Data Analysis (EDA)
- Visualized sentiment distributions (barplots), time trends (line plots), and message length patterns.
- Summarized data by employee, month, and sentiment label for insight into engagement and tone.

### 3. Monthly Sentiment Scoring
- Each message scored: +1 (Positive), 0 (Neutral), –1 (Negative).
- Aggregated monthly per employee, creating a time series of engagement.

### 4. Employee Ranking System
- Ranked all employees by monthly sentiment scores.
- Generated tables of top 3 and bottom 3 employees each month for both models.
- Useful for highlighting consistent high/low performers.

### 5. Flight Risk Detection
- Flagged employees who sent ≥4 negative messages in any rolling 30-day window (not just calendar months).
- Applied separately for both models; intersection of flagged employees is highlighted for HR action.

### 6. Predictive Modeling
- Built a linear regression model to predict an employee's monthly sentiment score.
- Features included: monthly message count, average message length, positive/negative ratios, and more.
- **Reported performance:**
  - **CardiffNLP Roberta:** MSE = 3.45, R² = 0.18
  - **NLP Town:** MSE = 6.04, R² = 0.08

### 7. Validation and Quality Control
- Manual review (spot-check) of sample messages for labeling accuracy and edge-case handling.
- Differences between model outputs are analyzed and discussed in the final report.

## 📈 Results Overview

- **Messages processed:** 2,191
- **Unique employees:** 226
- **Roberta model (CardiffNLP):** 68.7% Neutral, 24.6% Positive, 6.8% Negative
- **NLP Town model:** 47.5% Positive, 41.7% Negative, 10.8% Neutral
- **Flight risks flagged:** 4 (Roberta), 10 (NLP Town), with overlap
- **Regression R²:** 0.18 (Roberta), 0.08 (NLP Town)

See `Final Report.pdf` for full tables and visualizations.

## 🗂️ File Descriptions

| File | Description |
|------|-------------|
| `employee_sentiment_analysis.ipynb` | Main notebook: complete analysis pipeline |
| `labeled_sentiments_combined.csv` | Sentiment labels for both models (merged) |
| `labeled_sentiments_roberta.csv` | Raw output from CardiffNLP Roberta |
| `labeled_sentiments_nlptown.csv` | Raw output from NLP Town BERT |
| `monthly_employee_scores.csv` | Employee-level sentiment scores by month |
| `regression_features.csv` | Features used for regression model |
| `sentiment_labeled_data.csv` | (if different from above) |
| `test.csv` | Original provided email data |
| `Final Report.pdf` | Comprehensive project report |
| `figures/` | All PNG plots and charts |

## 📝 References

- [CardiffNLP/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)

## 📬 Contact

For any questions, contact: **Harsh Kute** | harshavinashkute@gmail.com
