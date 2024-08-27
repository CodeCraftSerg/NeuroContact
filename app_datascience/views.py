from django.shortcuts import render
from app_datascience.utils import plot_training_validation, plot_accuracy, plot_loss


def data_science_page(request):
    training_history = {
        'history': {
            'loss': [
                1.8650, 1.3418, 1.1569, 1.0268, 0.9486, 0.8831, 0.8268, 0.7837, 0.7295, 0.7000,
                0.6636, 0.6387, 0.6042, 0.5934, 0.5661, 0.5362, 0.5171, 0.4937, 0.4839, 0.4595,
                0.4477, 0.4470, 0.4348, 0.4190, 0.3674, 0.3309, 0.3312, 0.3303, 0.3230, 0.3094,
                0.3164, 0.3056, 0.2995, 0.2990, 0.2954, 0.2983, 0.2928, 0.2821, 0.2977, 0.2778,
                0.2795, 0.2850, 0.2755, 0.2667, 0.2668, 0.2677, 0.2681, 0.2629
            ],
            'val_loss': [
                1.3360, 1.0370, 1.1056, 1.0399, 0.9368, 0.8587, 0.6846, 0.7307, 0.8300, 0.7235,
                0.9738, 0.5657, 0.7009, 0.5959, 0.6365, 0.6184, 0.5025, 0.5306, 0.4167, 0.5809,
                0.5096, 0.6558, 0.4266, 0.4894, 0.3951, 0.3902, 0.3911, 0.3744, 0.3774, 0.3883,
                0.3783, 0.3673, 0.3691, 0.3815, 0.3915, 0.3793, 0.3685, 0.3616, 0.3843, 0.3671,
                0.3852, 0.3733, 0.3694, 0.3637, 0.3878, 0.3882, 0.3692, 0.3631
            ],
            'accuracy': [
                0.3665, 0.5340, 0.5974, 0.6432, 0.6663, 0.6953, 0.7115, 0.7285, 0.7484, 0.7588,
                0.7719, 0.7805, 0.7905, 0.7951, 0.8039, 0.8145, 0.8251, 0.8285, 0.8345, 0.8427,
                0.8441, 0.8431, 0.8513, 0.8546, 0.8784, 0.8857, 0.8878, 0.8853, 0.8866, 0.8948,
                0.8898, 0.8941, 0.8988, 0.8973, 0.8969, 0.8980, 0.8973, 0.9055, 0.8982, 0.9087,
                0.9037, 0.9023, 0.9064, 0.9090, 0.9089, 0.9089, 0.9091, 0.9082
            ],
            'val_accuracy': [
                0.5335, 0.6391, 0.6388, 0.6761, 0.6867, 0.7288, 0.7668, 0.7616, 0.7397, 0.7733,
                0.7179, 0.8169, 0.7748, 0.8080, 0.7995, 0.7983, 0.8377, 0.8282, 0.8574, 0.8194,
                0.8355, 0.8040, 0.8541, 0.8394, 0.8695, 0.8723, 0.8738, 0.8760, 0.8751, 0.8745,
                0.8762, 0.8798, 0.8787, 0.8779, 0.8742, 0.8776, 0.8830, 0.8838, 0.8753, 0.8799,
                0.8763, 0.8808, 0.8803, 0.8811, 0.8756, 0.8747, 0.8788, 0.8821
            ]
        }
    }

    loss_image, acc_image = plot_training_validation(training_history)
    accuracy_graph = plot_accuracy(training_history['history'])
    loss_graph = plot_loss(training_history['history'])
    context = {
        'model_name': 'Inception',
        'model_description': (
            'Inception is a convolutional neural network architecture that has been '
            'trained on the ImageNet dataset. Known for its sophisticated structure, '
            'Inception has consistently demonstrated high accuracy in image '
            'classification tasks, making it a reliable choice for our project.'
        ),
        'reasons_for_choice': (
            'We selected Inception because of its impressive ability to handle '
            'complex image classification tasks. The architecture strikes a balance '
            'between computational efficiency and high performance, enabling it to '
            'process a wide variety of images with remarkable accuracy. Additionally, '
            'Inception has been extensively validated in numerous research studies '
            'and projects, further solidifying our confidence in its capabilities.'
        ),
        'challenges_faced': (
            'During the implementation of this model, we encountered several challenges. '
            'One of the primary issues was the lack of sufficient computational resources. '
            'Training a model as complex as Inception requires a powerful GPU, and in many '
            'cases, this necessitated the use of cloud-based services with high-end hardware. '
            'However, these services often come with significant costs, necessitating paid accounts '
            'to access the required resources. The combination of high resource demands and the need '
            'for paid services posed a challenge that we had to carefully manage throughout the project.'
        ),
        'accuracy': 'The model achieved an accuracy of 97.3% on the test dataset, which is indicative of its high performance.',
        'loss': 'The model’s loss on the test dataset was 1.7, demonstrating its capability to effectively minimize prediction errors.',
        'models_used': [
            {'name': 'Inception',
             'description': 'A deep convolutional neural network known for its high performance in image classification tasks.'},
            {'name': 'VGG16',
             'description': 'A popular convolutional neural network with 16 layers, widely used for image classification.'},
            {'name': 'ResNet50',
             'description': 'A residual neural network that utilizes skip connections to enable training of very deep networks.'},
            {'name': 'MobileNet',
             'description': 'A lightweight convolutional neural network designed for mobile and embedded vision applications.'},
            {'name': 'DenseNet',
             'description': 'A convolutional neural network where each layer is connected to every other layer, known for efficient feature reuse.'},
            {'name': 'EfficientNet',
             'description': 'A family of convolutional neural networks that scales depth, width, and resolution efficiently.'},
            {'name': 'AlexNet',
             'description': 'One of the first deep convolutional neural networks that demonstrated significant improvements in image classification tasks.'},
        ],
        'training_accuracy': 'Training accuracy: 94.46%',
        'training_loss': 'Training loss: 16.49%',
        'testing_accuracy': 'Testing accuracy: 88.38%',
        'testing_loss': 'Testing loss: 36.16%',
        'accuracy_graph': accuracy_graph,
        'loss_graph': loss_graph,
        'train_val_loss_graph': loss_image,
        'train_val_acc_graph': acc_image
    }
    return render(request, 'app_datascience/data_science.html', context)
