import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.subplots as sp
from plotly.subplots import make_subplots
from pygments.lexers import go
import statsmodels.tsa.statespace.sarimax
# importing high level interactive plotting libraries
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import subprocess
def df_holiday():
    df_holiday = pd.read_csv("data/holiday.csv")
    df_holiday['date'] = pd.to_datetime(df_holiday['date'])
    df_holiday.set_index('date', inplace=True)
    idx = pd.date_range('2017-01-04', '2018-12-31')
    df_holiday = df_holiday.reindex(idx, fill_value=0)
    df_holiday['is_holiday'] = df_holiday['is_holiday'].astype(int)
    df_holiday.loc[((df_holiday.index.day == 14) & (df_holiday.index.month == 2)), :] = 1
    df_holiday.loc[((df_holiday.index == '2017-11-24') | (df_holiday.index == '2018-11-23')), :] = 1
    holiday_df = df_holiday.loc[df_holiday.index <= '2018-08-14']
    return df_holiday()
def load_data_main(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df_revenue = df.groupby(pd.Grouper(key='date', freq='D'))['revenue'].sum().reset_index()
    df_revenue = df_revenue.loc[df_revenue['date'] >= '2017-01-05']
    df_revenue = df_revenue.loc[df_revenue['date'] <= '2018-08-14']
    df['purchase_year'] = pd.to_datetime(df['date']).dt.year
    df['purchase_month'] = pd.to_datetime(df['date']).dt.month
    df['purchase_MMYYYY'] = pd.to_datetime(df['date']).dt.strftime('%m-%Y')
    df['purchase_week'] = pd.to_datetime(df['date']).dt.isocalendar().week
    df['purchase_dayofweek'] = pd.to_datetime(df['date']).dt.weekday
    df['purchase_dayofmonth'] = pd.to_datetime(df['date']).dt.day
    df_revenue['date'] = pd.to_datetime(df_revenue['date'])
    df_revenue.set_index('date', inplace=True)
    return df, df_revenue
def main():
    st.title("GROUP 11 SALE PREDICTION")
    st.write("This is a simple demo to visualize the result of Group 11")
    side_bar()
def side_bar():
    st.sidebar.image("image/logo_uel.png", use_column_width=True)
    with st.sidebar:
        st.header("Load data")
        uploaded_file = st.file_uploader("Choose file CSV", type="csv")
    if uploaded_file is not None:
        df, df_revenue = load_data_main(uploaded_file)
        st.title("Sale Overview")

        # ========================= CHART1 ======================
        # Vẽ đồ thị dạng đường để có cái nhìn tổng thể về doanh thu hàng ngày
        fig = px.line(df_revenue, x=df_revenue.index, y='revenue')
        # Thiết lập nhãn trục và tiêu đề
        fig.update_layout(
            yaxis_title="Total Revenue earned (Brazilian Real)",
            legend_title="date",
            title="Daily Revenue from Sept 2016 to Aug 2018"
        )
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig)
        # =============== CHART 3 ==========================
        fig = px.histogram(df_revenue, x='revenue')
        fig.update_layout(
            yaxis_title="Frequency",
            xaxis_title="Revenue in Brazilian Real",
            legend_title="",
            title="Daily Revenue distribution from Sept 2016 to Aug 2018"
        )
        st.plotly_chart(fig)
        # =============== CHART 4 ============================
        df1 = df.groupby('product_category_name_english')['revenue'].sum().sort_values(ascending=False)
        fig = px.bar(
            df1,
            x=df1.index,
            y=df1.values,
            labels={'y': 'Sales amount'},
            title='Product Category by sales amount',
            # width=1500,
            # height=700
        )
        fig.update_xaxes(tickangle=-90)
        st.plotly_chart(fig)
        # ================ CHART 5 =========================================================
        unique_products = df.drop_duplicates(subset=['product_id'])
        fig = px.histogram(unique_products, x='price', nbins=50, marginal='rug',
                           title='Price Distribution for Unique Products')
        fig.update_layout(
            xaxis_title="Price",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig)
        average_price = unique_products['price'].mean()
        st.write("Giá trị trung bình của phân bố giá là:", average_price)
        def get_holiday_data():
            df_holiday_data = pd.read_csv("data/holiday.csv")
            df_holiday_data['date'] = pd.to_datetime(df_holiday_data['date'])
            df_holiday_data.set_index('date', inplace=True)
            idx = pd.date_range('2017-01-05', '2018-12-31')
            df_holiday_data = df_holiday_data.reindex(idx, fill_value=0)
            df_holiday_data['is_holiday'] = df_holiday_data['is_holiday'].astype(int)
            df_holiday_data.loc[((df_holiday_data.index.day == 14) & (df_holiday_data.index.month == 2)), :] = 1
            df_holiday_data.loc[((df_holiday_data.index == '2017-11-24') | (df_holiday_data.index == '2018-11-23')),
            :] = 1
            holiday_df = df_holiday_data.loc[df_holiday_data.index <= '2018-08-14']
            return holiday_df
        df_holiday_data = get_holiday_data()
        dfex = pd.concat([df_revenue, df_holiday_data], axis=1)

        st.title("Experiment Tracking")
        make_track = st.checkbox("Experiment Tracking")
        if make_track and 'dfex' in locals():
            dfex_train = dfex['2017-01-05':'2018-07-31']
            dfex_test = dfex['2018-08-01':'2018-08-14']
            track_experiment(dfex_train, dfex_test)
        st.title("Sale Forecast")
        make_forecast = st.checkbox("3. Make Forecast")
        if make_forecast and 'dfex' in locals():
            forecast_period = st.slider("Forecast Period", 1, 14, 7)
            if st.button("Forecast"):
                model = sm.tsa.SARIMAX(dfex['revenue'], exog=dfex['is_holiday'], order=(0, 1, 1),
                                       seasonal_order=(1, 0, 1, 7))
                model_fit = model.fit()
                # Forecast for the selected period
                forecast_period = min(forecast_period, len(dfex_test))
                forecast = model_fit.get_forecast(steps=forecast_period,
                                                  exog=dfex_test.head(forecast_period)['is_holiday'])
                predicted_values = forecast.predicted_mean
                # Display the forecasted values
                st.write("Forecasted Values:")
                st.write(predicted_values)
                # Vẽ biểu đồ dự đoán
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dfex.index, y=dfex['revenue'], mode='lines', name='Actual Sales'))
                fig.add_trace(go.Scatter(x=forecast.predicted_mean.index, y=predicted_values, mode='lines',
                                         name='Predicted Sales'))
                fig.update_xaxes(rangeslider_visible=True)
                fig.update_layout(
                    yaxis_title="Revenue amount",
                    xaxis_title="Date",
                    title="Actual vs Predicted Sales"
                )
                st.plotly_chart(fig)
def metrics(df1, predicted_values):
    y_true = df1['revenue']
    mape = np.mean(np.abs((y_true - predicted_values) / y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, predicted_values))
    mae = mean_absolute_error(y_true, predicted_values)
    difference = y_true.sum() - predicted_values.sum()
    relative_error = (difference / y_true.sum()) * 100
    return mape, rmse, mae, difference, relative_error


def track_experiment(dfex_train, dfex_test):
    model = sm.tsa.SARIMAX(dfex_train['revenue'], exog=dfex_train['is_holiday'], order=(0, 1, 1),
                           seasonal_order=(1, 0, 1, 7))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=len(dfex_test), exog=dfex_test['is_holiday'])
    predicted_values = forecast.predicted_mean

    # Calculate metrics
    mape, rmse, mae, difference, relative_error = metrics(dfex_test, predicted_values)

    # Create a dataframe with the metrics
    metrics_data = {
        'Metric': ['MAPE', 'RMSE', 'MAE', 'Predicted Values Sum', 'Difference', 'Relative Error'],
        'Value': [mape, rmse, mae, predicted_values.sum(), difference, relative_error]
    }
    metrics_df = pd.DataFrame(metrics_data)
    # Display the metrics dataframe
    st.dataframe(metrics_df)
    # Create Figure
    fig, ax = plt.subplots()
    dfex_train_2018 = dfex_train[dfex_train.index.year >= 2018]
    ax.plot(dfex_train_2018.index, dfex_train_2018['revenue'], label='Observed', color='blue')
    ax.plot(dfex_test.index, dfex_test['revenue'], label='Observed Test Data', color='black', linestyle='--')
    ax.plot(dfex_test.index, predicted_values, color='red', label='Forecast')
    ax.fill_between(dfex_test.index, forecast.conf_int()['lower revenue'], forecast.conf_int()['upper revenue'],
                    color='red', alpha=0.2, label='Confidence Interval')
    ax.set_title('SARIMAX Forecast with Confidence Interval')
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue')
    ax.legend()
    plt.tight_layout()
    # Display the plot
    st.pyplot(fig)
    # Return the calculated metrics
    return mape, rmse, mae, difference, relative_error
if __name__ == "__main__":
    main()


