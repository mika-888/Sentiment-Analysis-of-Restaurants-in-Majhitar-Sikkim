# Sentiment-Analysis-of-Restaurants-in-Majhitar-Sikkim
This project implements and compares two sentiment analysis approaches: 
1. VADER (Lexicon-Based Model)
2. RoBERTa (Transformer-Based SOTA Model)

The objective is to analyze real-world restaurant reviews and evaluate: - Accuracy - Speed - Practical usability


💠Objectives
Perform sentiment classification on real-world textual data
Compare a baseline NLP model with a state-of-the-art transformer model
Generate meaningful insights from customer reviews
Visualize sentiment distribution and restaurant performance
Analyze model performance in terms of efficiency and reliability


📂 Project Structure
```text
TANLP-Sentiment-Analysis/
│
├── data/
│   └── majhitar_active_places_only.csv
│
├── models/
│   ├── vader_model.py
│   ├── roberta_model.py
│
├── analysis/
│   └── comparison_dashboard.py
│
├── results/
│   ├── model1_vader_results.csv
│   ├── model2_roberta_results.csv
│   ├── model1_vader_plots.png
│   ├── model2_roberta_plots.png
│   └── Final_Comparison_Dashboard.png
│
├── requirements.txt
├── run_all.py
└── README.md
```

⚙️ Models Used
🔹 1. VADER (Baseline Model)
Rule-based sentiment analysis
Uses a predefined sentiment lexicon
No training required
Extremely fast and efficient
Works well on social media and short texts
🔹 2. RoBERTa (SOTA Model)
Transformer-based deep learning model
Context-aware understanding of language
Higher accuracy and robustness
Computationally expensive compared to VADER


📊 Features
Sentiment classification (Positive / Neutral / Negative)
Sentiment score distribution analysis
Restaurant ranking based on reviews
Visualization dashboards (histograms, pie charts, bar graphs)
Model comparison (accuracy vs speed)
Extraction of common negative phrases using N-grams


🚀 How to Run the Project
1. Clone the repository
git clone <your-repo-link>
cd TANLP-Sentiment-Analysis
2. Install dependencies
pip install -r requirements.txt
3. Run all modules
python run_all.py


📁 Outputs Generated

The project generates the following outputs:

CSV Files
Model predictions and sentiment scores
Visualization Graphs
Sentiment distribution
Score histograms
Restaurant rankings
Final Dashboard
Comparative analysis between VADER and RoBERTa
Performance and latency comparison
Insight extraction (top negative phrases)


📈 Key Insights
VADER is extremely fast and suitable for real-time applications
RoBERTa provides better contextual understanding and accuracy
There is a clear trade-off between speed and performance
N-gram analysis helps identify common customer complaints


⚖️ Model Comparison Summary
Feature	VADER	RoBERTa
Type	Rule-based	Transformer
Speed	Very Fast	Slower
Accuracy	Moderate	High
Training Needed	No	Pretrained
Interpretability	High	Moderate


🧠 Learning Outcomes
Understanding baseline vs advanced NLP techniques
Hands-on experience with sentiment analysis pipelines
Working with real-world datasets
Visualization and interpretation of results
Performance benchmarking of NLP models

🔍 Future Work
Fine-tuning transformer models on domain-specific data
Multilingual sentiment analysis
Deployment using Flask or FastAPI
Integration with real-time review systems


📌 Conclusion
This project demonstrates the practical implementation of sentiment analysis using both traditional and modern NLP techniques. It highlights the importance of choosing the right model based on use-case requirements such as speed, accuracy, and scalability.

👩‍💻 Author
Name: Subhamika Chhetri
Course: Text Analytics & Natural Language Processing (TANLP)
Program: B.Tech CSE (Data Science)
