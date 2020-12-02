from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2

def PlantNet(height, width, num_classes):
    
    model = Sequential()
    name = PlantNet.__name__
            
    #Block 1
    model.add(Conv2D(input_shape = (height, width, 3),
                        filters = 128, kernel_size = (5,5), strides = 2, padding = 'Same', name='block1_conv1',
                        activation ='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(filters = 128, kernel_size = (5,5), strides = 2, padding = 'Same', name='block1_conv2',
                        activation ='relu',kernel_initializer='he_normal'))
    model.add(MaxPool2D(strides=(2, 2), name='block1_pool'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    #Block 2
    model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', name='block2_conv1',
                        activation ='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', name='block2_conv2',
                        activation ='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', name='block2_conv3',
                        activation ='relu',kernel_initializer='he_normal'))
    model.add(MaxPool2D(strides=(2, 2), name='block2_pool'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    #Block 3
    model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same', name='block3_conv1',
                        activation ='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same', name='block3_conv2',
                        activation ='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same', name='block3_conv3',
                        activation ='relu',kernel_initializer='he_normal'))
    model.add(MaxPool2D(strides=(2, 2), name='block3_pool'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    #Block 4
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation = "relu", kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,
                    activation = "softmax",
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2()))

    return model, name