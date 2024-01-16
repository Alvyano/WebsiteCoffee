import pickle
import pandas as pd
from flask import Flask, render_template, request
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from pmdarima.arima import auto_arima, ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import base64 
from io import BytesIO



app = Flask(__name__)

models = {
    "avocado": pickle.load(open("avocado.pkl", "rb")),
    "choco": pickle.load(open("choco.pkl", "rb")),
    "aren": pickle.load(open("aren.pkl", "rb")),
    "redvelvet": pickle.load(open("redvelvet.pkl", "rb")),
    "taro": pickle.load(open("Taro.pkl", "rb")),
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/prediksi", methods=["GET", "POST"])
def prediksi():
    menu = request.form.get("menu")
    start_year = request.form.get("start_year")
    start_month = request.form.get("start_month")
    end_year = request.form.get("end_year")
    end_month = request.form.get("end_month")

    if start_year and start_month and end_year and end_month:
        # Convert to integers if all values are provided
        start_year = int(start_year)
        start_month = int(start_month)
        end_year = int(end_year)
        end_month = int(end_month)
        # Selecting the appropriate model based on the chosen menu
        model = models.get(menu)
        if not model:
            return render_template("index.html", message="Menu not found!")

        # Load the data used to train the model
        data = pd.read_csv(f"{menu}.csv")
        data["Month"] = pd.to_datetime(data["Month"])
        data.set_index("Month", inplace=True)

        # Create the date range for prediction
        start_date = pd.to_datetime(f"{start_year}-{start_month:02d}-01")
        end_date = pd.to_datetime(f"{end_year}-{end_month:02d}-01") + pd.offsets.MonthEnd(0)
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        # Check if the start date is present in the data index
        if start_date not in data.index:
            return render_template("prediksi.html", message="Start date not found in data.")

        # Perform prediction using the loaded model
        predictions = model.predict(start=start_date, end=end_date, dynamic=True)
        rounded_predictions = predictions.round(0).astype(int)

        # Reindex the rounded_predictions DataFrame to match the date_range
        rounded_predictions = rounded_predictions.reindex(date_range, fill_value=0)

        hasil_prediksi = "Hasil prediksi untuk bulan {} dan menu {}: {}".format(
            start_month, menu, rounded_predictions[0]
        )

        print("Predictions:", predictions)  # Add this line for debugging
        print("Hasil Prediksi:", hasil_prediksi)

        # Get the actual data for the plot
        actual_data = data.loc[start_date:end_date]

       

        # Create a plot of ARIMA predictions and actual data
        plt.figure(figsize=(20, 10))
        plt.plot(actual_data.index, actual_data[menu], label="Actual Data")
        plt.plot(date_range, rounded_predictions.values, label="ARIMA Predictions", linestyle="--")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title(f"ARIMA Predictions and Actual Data for Menu: {menu}")
        plt.legend()
        plt.tight_layout()

        # Save the plot as a base64 encoded image
        with BytesIO() as buffer:
                plt.savefig(buffer, format="png")
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.read()).decode("utf-8")

        plt.close()

        return render_template("prediksi.html", hasil_prediksi=hasil_prediksi, plot_data=plot_data)
    else:
        # Handle the case when any of the date values is missing from the form data
        return render_template("prediksi.html", hasil_prediksi="Please select start and end dates")

    
if __name__ == '__main__':
    app.run(debug=True)
