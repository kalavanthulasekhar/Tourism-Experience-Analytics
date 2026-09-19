# Tourism Experience Analytics

<p align="center">
	<strong>AI-powered tourism intelligence for understanding visitors, attractions, and travel behavior.</strong>
</p>

<p align="center">
	<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python"></a>
	<a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"></a>
	<a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn"></a>
	<img src="https://img.shields.io/badge/status-academic%20project-1F6FEB" alt="Project status">
</p>

Tourism Experience Analytics is an end-to-end data science project that transforms tourism transaction data into practical insights and predictive experiences. It combines data integration, exploratory analysis, supervised machine learning, recommendation algorithms, SQL analytics, and an interactive Streamlit application in one workflow.

## Project highlights

| Capability | What it does |
| --- | --- |
| **Rating prediction** | Estimates an attraction rating on a 1-5 scale. |
| **Visit mode classification** | Predicts Business, Couples, Family, Friends, or Solo travel. |
| **Personalized recommendations** | Ranks attractions using collaborative, content-based, and popularity signals. |
| **Tourism dashboard** | Presents analytics and predictions through Streamlit. |

## How it works

```text
Raw tourism files -> Data integration -> Cleaning and feature engineering
																							|
								 +----------------------------+----------------------------+
								 v                            v                            v
				Rating regression            Visit classification          Recommendation system
								 +----------------------------+----------------------------+
																							v
																		Streamlit dashboard
```

## Dashboard modules

1. **Analytics dashboard** - view transaction volume, user activity, attraction coverage, ratings, and visit-mode trends.
2. **Visit mode prediction** - classify a visitor into Business, Couples, Family, Friends, or Solo.
3. **Rating prediction** - estimate the expected rating for an attraction.
4. **Recommendations** - discover attractions with a hybrid recommendation strategy.

## Recommendation strategy

- **Collaborative filtering** learns from user-attraction interaction history.
- **Content-based filtering** compares attraction characteristics.
- **Popularity ranking** provides strong general-purpose recommendations.
- **Hybrid ranking** combines these signals for the final result.

## Project structure

```text
Tourism_Analysis/
├── app/
│   └── app.py                         # Streamlit application
├── data/
│   ├── raw/                           # Original Excel datasets
│   └── processed/                     # Cleaned and feature-engineered CSV files
├── models/
│   ├── recommendation/               # Recommendation artifacts
│   ├── rating_prediction_model.pkl
│   ├── visitmode_classes.pkl
│   ├── visitmode_classification_model.pkl
│   └── visitmode_classification_deployment.pkl
├── notebooks/                         # Analysis and modeling workflow
├── reports/                           # Evaluation results and analysis outputs
├── sql/
│   └── tourism_analysis.sql           # Tourism analytics queries
├── compare_classification_models.py   # Compare classification pipelines
├── create_deployment_classifier.py    # Build the deployment classifier
├── requirements.txt
└── README.md
```

## Dataset

The source data consists of tourism transactions and related reference tables:

- `City.xlsx`
- `Continent.xlsx`
- `Country.xlsx`
- `Item.xlsx`
- `Mode.xlsx`
- `Region.xlsx`
- `Transaction.xlsx`
- `Type.xlsx`
- `Updated_Item.xlsx`
- `User.xlsx`

Processed datasets are stored in `data/processed/`, including the cleaned master dataset used by the application: `tourism_master_cleaned.csv`.

## Technology stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost and LightGBM
- Matplotlib and Seaborn
- Plotly
- Joblib
- SQL
- Streamlit
- Jupyter Notebook

## Quick start

### Prerequisites

- Python 3.x
- pip
- The processed datasets and trained model artifacts included in the project

### 1. Open the project

```powershell
cd Tourism_Analysis
```

### 2. Create an environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app/app.py
```

Open the local URL displayed by Streamlit, normally `http://localhost:8501`.

The application loads its data and models at startup. Confirm that these files are available:

Before launching the dashboard, confirm that these paths exist:

```text
data/processed/tourism_master_cleaned.csv
models/visitmode_classification_deployment.pkl
models/visitmode_classes.pkl
models/rating_prediction_model.pkl
models/recommendation/recommendation_artifacts.pkl
```

## Analysis and modeling workflow

The numbered notebooks describe the recommended workflow:

| Step | Notebook | Purpose |
| ---: | --- | --- |
| 01 | `01_data_inspection.ipynb` | Inspect the source data |
| 02-04 | `02_dataset_integration.ipynb` - `04_dataset_integration.ipynb` | Integrate tourism tables |
| 05 | `05_data_cleaning.ipynb` | Clean and validate records |
| 06 | `06_eda.ipynb` | Explore tourism behavior and attraction performance |
| 07 | `07_feature_engineering.ipynb` | Build modeling features |
| 08 | `08_regression.ipynb` | Train and evaluate rating models |
| 09 | `09_classification.ipynb` | Train and evaluate visit-mode models |
| 10 | `10_recommendation.ipynb` | Build and evaluate recommendations |

Open the notebooks in Jupyter or VS Code after installing the project dependencies.

The standalone scripts support the classification workflow:

```bash
python create_deployment_classifier.py
python compare_classification_models.py
```

Run these scripts only after the required processed data and source classification model are available. The scripts use the project-relative `data/` and `models/` directories.

## Reports and evaluation outputs

The `reports/` directory contains generated evaluation results for regression, classification, and recommendation experiments, including:

- classification model comparisons and detailed reports;
- confusion matrices and baseline results;
- regression results;
- recommendation metrics for collaborative, content-based, popularity-based, and hybrid approaches; and
- exploratory analysis tables and rating feature importance.

## SQL analysis

The queries in [`sql/tourism_analysis.sql`](sql/tourism_analysis.sql) cover common tourism metrics, including:

- transaction, user, and attraction counts;
- rating distributions;
- visit mode trends;
- yearly and monthly activity;
- popular attractions;
- attraction performance;
- user activity; and
- rating behavior by visit mode.

## Notes

- Trained model files can be large and may be managed separately from source code. Check the repository contents before deployment.
- The dashboard expects the processed data and model artifact filenames shown above.
- Do not commit credentials or local Streamlit secrets. The repository's `.gitignore` excludes common virtual-environment, cache, secret, log, and generated build files.

## License

No license has been specified for this project yet.
