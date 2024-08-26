import matplotlib.pyplot as plt
import io
import base64


def plot_training_validation(history):
    # Извлекаем данные истории обучения из словаря 'history'
    history_dict = history['history']
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(history_dict['accuracy']) + 1)

    plt.rcParams["figure.figsize"] = (13, 4)

    # Training and validation loss
    buf = io.BytesIO()
    plt.plot(epochs, loss_values, 'go', label='Training loss')
    plt.plot(epochs, val_loss_values, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(buf, format='png')
    buf.seek(0)
    loss_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    # Training and validation accuracy
    buf = io.BytesIO()
    plt.plot(epochs, history_dict['accuracy'], 'go', label='Training acc')
    plt.plot(epochs, history_dict['val_accuracy'], 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(buf, format='png')
    buf.seek(0)
    acc_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    return loss_image, acc_image


def plot_accuracy(history):
    history_dict = history
    plt.rcParams["figure.figsize"] = (13, 4)
    plt.plot(history_dict['accuracy'])
    plt.plot(history_dict['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    plt.close()

    return image_base64


def plot_loss(history):
    history_dict = history
    plt.rcParams["figure.figsize"] = (13, 4)
    plt.plot(history_dict['loss'])
    plt.plot(history_dict['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    plt.close()

    return image_base64
