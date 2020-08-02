import matplotlib.pyplot as plt 
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from time import time
from tensorflow.keras.models import load_model


def plot_history(history, train_dir):
    # history : dict
    hist = history

    plt.plot(*zip(*hist['loss'].items()))
    plt.plot(*zip(*hist['val_loss'].items()))
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(str(train_dir) + '/history_loss.png')
    plt.show()

    plt.plot(*zip(*hist['event_output_accuracy'].items()))
    plt.plot(*zip(*hist['val_event_output_accuracy'].items()))
    plt.title('Event accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(str(train_dir) + '/history_accuracy.png')
    plt.show()

    plt.plot(*zip(*hist['time_output_loss'].items()))
    plt.plot(*zip(*hist['val_time_output_loss'].items()))
    plt.title('Time Mean Squared Error')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(str(train_dir) + '/history_mse.png')
    plt.show()
    

def plot_history_next_event(history, train_dir):
    # history : dict
    hist = history

    plt.plot(*zip(*hist['loss'].items()))
    plt.plot(*zip(*hist['val_loss'].items()))
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(str(train_dir) + '/history_loss.png')
    plt.show()

    plt.plot(*zip(*hist['accuracy'].items()))
    plt.plot(*zip(*hist['val_accuracy'].items()))
    plt.title('Event accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(str(train_dir) + '/history_accuracy.png')
    plt.show()
    

def plot_history_next_tStamp(history, train_dir):
    # history : dict
    hist = history

    plt.plot(*zip(*hist['loss'].items()))
    plt.plot(*zip(*hist['val_loss'].items()))
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(str(train_dir) + '/history_loss_tStamp.png')
    plt.show()

    plt.plot(*zip(*hist['mae'].items()))
    plt.plot(*zip(*hist['val_mae'].items()))
    plt.title('tstamp mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(str(train_dir) + '/history_mae_tStamp.png')
    plt.show()



def evaluate_vectorized(test_data, train_dir, caseIDs):
    tic = time()
    print("Performing vectorized evaluation")

    filename = 'vectorized_eval.csv'
    X, Y_a, Y_t = test_data
    print(f" -> Loading Model {train_dir}...")
    model = load_model(str(train_dir) + '\\final_model.h5')

    print(" -> Predicting the next events...")
    Y_a_hat, Y_t_hat = model.predict(X)
    print(f"Y_t_hat.shape: {Y_t_hat.shape}")

    true_events = np.empty(shape=(Y_a.shape[0],), dtype=np.uint8)
    np.argmax(Y_a, axis=1, out=true_events)
    print(f"true_events.shape: {true_events.shape}")

    predicted_events = np.empty(shape=(Y_a.shape[0],), dtype=np.uint8)
    np.argmax(Y_a_hat, axis=1, out=predicted_events)
    print(f"predicted_events.shape: {predicted_events.shape}")

    prediction_probability = np.empty(shape=(Y_a.shape[0],), dtype=np.float32)
    np.amax(Y_a_hat, axis=1, out=prediction_probability)
    print(f"prediction_probability.shape: {prediction_probability.shape}")

    print(f"Shapes: \-")
    df = pd.DataFrame({'CaseID': caseIDs,
                       'Prefix Length': np.sum(X[:, :, :25], axis=(2, 1)),
                       'True Event': true_events,
                       'Predicted Event': predicted_events,
                       'Prediction Probability': prediction_probability,
                       'True Timedelta': Y_t,
                       'Predicted Timedelta': Y_t_hat})

    print(f" -> Writing results to {filename}...")
    df.to_csv(train_dir/filename, sep=';', index=False)
    toc = time()
    print(f" -> Finished. Time for vectorized evaluation: {toc-tic:.2f}s")
