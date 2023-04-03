import pickle
from flask import Flask, render_template
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.metrics import accuracy_score, recall_score
app = Flask(__name__,template_folder='/Users/Violine/Desktop/FINAL/FINAL/PREDICTIVE MAINTAINENCE FLASK/Templates')
model = keras.models.load_model('/Users/Violine/Desktop/FINAL/FINAL/PREDICTIVE MAINTAINENCE FLASK/hi.h5')

from flask import request

@app.route('/predictautoencoder')
def predictautoencoder():
    return render_template('predictautoencoder.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/afterautoencoder')
def afterautoencoder():
    return render_template('afterautoencoder.html')

@app.route('/afterlstm')
def afterlstm():
    return render_template('afterlstm.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/selectmodel')
def selectmodel():
    return render_template('selectmodel.html')

@app.route('/predictlstm')
def predictlstm():
    return render_template('predictlstm.html')


@app.route('/predictlstm', methods=['GET', 'POST'])
def predict_lstm():
    if request.method == 'POST':
        # Load the uploaded files
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        # Load the numpy arrays from the uploaded files
        xtest = np.load(file1)
        ytest = np.load(file2)

        # Use the model to make predictions on the test data
        y_pred = model.predict(xtest)

        # Convert probabilities to binary predictions (0 or 1)
        y_pred_binary = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(ytest, y_pred_binary)
        recall = recall_score(ytest, y_pred_binary)

        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a scatter plot of the predicted and expected labels
        ax.scatter(range(len(ytest)), ytest, c='red', label='Expected Labels',s=150)
        ax.scatter(range(len(y_pred_binary)), y_pred_binary, c='blue', label='Predicted Labels',s=50)

        # Set the x and y axis labels and the title
        ax.set_xlabel('Sample Number', fontsize=14)
        ax.set_ylabel('Label', fontsize=14)
        ax.set_title('Expected vs. Predicted Labels', fontsize=16)

        # Remove the gridlines and add a legend
        ax.grid(False)
        ax.legend()

        # Save the plot to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()

        # Convert the buffer content to a string and encode it as base64
        data = base64.b64encode(buf.getbuffer()).decode('ascii')

        # Render the afterlstm.html template and pass the predicted and expected labels and the plot data
        return render_template('afterlstm.html', predicted_labels=y_pred_binary, expected_labels=ytest, accuracy=accuracy, recall=recall, plot_data=data)
    else:
        # Render the predictlstm.html template for GET requests
        return render_template('predictlstm.html')
if __name__ == '__main__':
    app.run(debug=True)

model = keras.models.load_model('/Users/Violine/Desktop/FINAL/FINAL/PREDICTIVE MAINTAINENCE FLASK/model2.h5')



@app.route('/predictautoencoder', methods=['GET', 'POST'])
def predict_autoencoder():
    if request.method == 'POST':

        # Load the uploaded files
        file = request.files['fileInput']  

        #Load the numpy arrays from the uploaded file
        x_test = np.load(file)

        #use the model to make predictions on the test data
        ypred = model.predict(x_test)

        #Calculate the MSE for all sequences and predictions
        import statistics

        #determine median mse to split data into normal vs anomaly
        pred_mse = []   # calculated mse for test data
        pred_res = []   # result: 0 normal, 1 anomaly

        for i in range(len(x_test)):
            pred_mse.append(mean_squared_error(ypred[i,:,:], x_test[i,:,:]))

        for mse in pred_mse:
            if (mse<=0.005):
                pred_res.append(0)
            else:
                pred_res.append(1)             

        normal_disk=pred_res.count(0)
        anomalous_disk=pred_res.count(1)
        return render_template('afterautoencoder.html', normal_disk=normal_disk, anomalous_disk=anomalous_disk )
    else:
        # Render the predictlstm.html template for GET requests
        return render_template('predictautoencoder.html')
    
