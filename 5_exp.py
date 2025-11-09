import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

def load_data(path):
    data = pd.read_csv(path)
    X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
    y = data['Chance of Admit']
    return X, y

def build_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse

def show_plots(result):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    bars = plt.bar(result["Model"], result["R2 Score"])
    plt.title("R² Score by Model", fontsize=16, fontweight="bold")
    plt.xlabel("Model", fontsize=14, fontweight="bold")
    plt.ylabel("R² Score", fontsize=14, fontweight="bold")
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0,1.1)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,   
            height + 0.01,                       
            f"{height:.3f}",                   
            ha="center", va="bottom", fontsize=10
        )

    bubble_size = result["RMSE"] * 500 

    plt.subplot(1, 2, 2)
    plt.scatter(result["Model"], result["R2 Score"], s=bubble_size, alpha=0.6)
    plt.title("Model Performance (Bubble = RMSE)", fontsize=16, fontweight="bold")
    plt.xlabel("Model", fontsize=14, fontweight="bold")
    plt.ylabel("R² Score", fontsize=14, fontweight="bold")
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0,1.1)

    for i in range(len(result)):
        plt.text(
            result["Model"][i], 
            result["R2 Score"][i] + 0.01, 
            f"{result['RMSE'][i]}",
            ha="center",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()

def main():
    X, y = load_data("Synthetic_Graduate_Admissions.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Support Vector Regressor": SVR(kernel='rbf'),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5)
    }

    results = []
    for name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        r2, rmse = evaluate_model(pipe, X_test, y_test)
        results.append({"Model": name, "R2 Score": round(r2, 3), "RMSE": round(rmse, 3)})

    results_df = pd.DataFrame(results)
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))

    sample = pd.DataFrame([[320, 110, 4, 4.5, 4.0, 9.0, 1]],
                          columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'])
    best_model = build_pipeline(RandomForestRegressor(random_state=42))
    best_model.fit(X, y)
    pred = best_model.predict(sample)
    print(f"\nPredicted Chance of Admission: {pred[0]*100:.2f}%")
    show_plots(results_df)

if __name__ == "__main__":
    main()
