# **CO₂ Emissions Predictor: Insights from World Bank Data**

## **Overview**
The CO₂ Emissions Predictor is a data-driven application designed to analyze and predict carbon dioxide (CO₂) emissions for countries worldwide using World Bank data. This project leverages key environmental, economic, and social indicators to generate insights and deliver predictions through an interactive dashboard.

---

## **Features**
- **Data Insights:**  
  Analyze CO₂ emissions trends based on GDP, renewable energy usage, urban population, and other key indicators.
  
- **Machine Learning Predictions:**  
  Predict CO₂ emissions using a trained Random Forest Regressor model with engineered features.

- **Interactive Dashboard:**  
  An intuitive Streamlit-based dashboard for exploring global trends and generating predictions.

- **Feature Engineering:**  
  Metrics such as Emissions Intensity (emissions per GDP) and Renewable Energy Urban Impact (renewable energy × urban population %) for enhanced analysis.

---

## **Project Structure**
```
CO₂ Emissions Predictor/
├── world_bank_data         # Contains cleaned datasets
├── app.py                  # Streamlit app files for the interactive dashboard
├── notebooks.ipynb         # Jupyter notebooks 
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies

---

## **Setup and Installation**

### **1. Prerequisites**
Ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

### **2. Clone the Repository**
```bash
git clone https://github.com/osisamkay/co2-emissions-predictor.git
cd co2-emissions-predictor
```

### **3. Install Dependencies**
Install required Python packages:
```bash
pip install -r requirements.txt
```

### **4. Launch the Dashboard**
Run the Streamlit app:
```bash
streamlit run dashboard/app.py
```
Open the provided URL in your browser to access the dashboard.

---

## **How It Works**

### **Data Pipeline**
1. **Data Loading:**  
   Load World Bank data for economic and environmental indicators.

2. **Feature Engineering:**  
   - **Emissions Intensity:** CO₂ emissions per GDP.  
   - **Renewable Energy Urban Impact:** Contribution of renewable energy and urbanization.  

3. **Machine Learning:**  
   The Random Forest Regressor is trained to predict CO₂ emissions using the engineered features.  

4. **Interactive Visualization:**  
   A dashboard enables users to explore country-specific trends and predictions.

---

## **Key Insights**
- Countries with higher GDP typically exhibit varying CO₂ emissions, depending on energy sources and urbanization.
- Renewable energy usage reduces emissions intensity, especially in urbanized nations.
- Strong relationships exist between economic indicators and environmental outcomes.

---

## **Technologies Used**
- **Data Processing:** pandas, NumPy  
- **Visualization:** matplotlib, seaborn, Streamlit  
- **Machine Learning:** scikit-learn (Random Forest Regressor)  
- **Deployment:** Streamlit  

---

## **Future Improvements**
- **Dynamic Data Updates:** Real-time World Bank data integration.  
- **Advanced Models:** Incorporate Gradient Boosting or neural networks for enhanced accuracy.  
- **User Customization:** Allow users to add custom datasets or indicators.  
- **Geospatial Analysis:** Visualize data geographically to identify regional trends.

---

## **Contributing**
We welcome contributions! Please:  
1. Fork this repository.  
2. Create a feature branch (`git checkout -b feature-name`).  
3. Commit changes (`git commit -m "Add feature"`).  
4. Push to your fork (`git push origin feature-name`).  
5. Submit a pull request.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## **Contact**
For questions or suggestions, please reach out:  
- **Email:** osisami.oj@gmail.com  
- **GitHub:** [osisamkay](https://github.com/osisamkay)

--- 