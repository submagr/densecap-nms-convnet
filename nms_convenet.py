from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers import Merge
from keras.utils.visualize_util import plot
from keras.optimizers import Adam
import theano
import theano.tensor as T

W = 100
H = 100
gridWidth = 4
w = W/gridWidth
h = H/gridWidth
scoreLayer = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
scoreLayer.add(Convolution2D(512, 11, 11, border_mode='same', input_shape=(2, w, h)))

IoULayer = Sequential()
IoULayer.add(Convolution2D(512, 1, 1, border_mode='same', input_shape=(121, w, h)))

merged = Merge([scoreLayer, IoULayer], mode='concat')

final_model = Sequential()
final_model.add(merged)
final_model.add(Convolution2D(1024, 1, 1))
final_model.add(Convolution2D(512, 1, 1))
final_model.add(Convolution2D(512, 1, 1))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
adam_opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

def myloss(pred_val, true_val, count_pos, count_neg):
    loss = T.log(1.0+T.exp(-1.0*true_val*pred_val))
    cat = (true_val+1)/2
    return cat*((count_neg*loss)/(count_neg+count_pos)) + (1-cat)*((count_pos*loss)/(count_neg+count_pos))

def calcLossAtLocation(i, j, prior_loss_matrix, count_pos, count_neg, y_true, y_pred):
    loss_i_j = myloss(y_pred[i,j], y_true[i,j], count_pos, count_neg) 
    return T.set_subtensor(prior_loss_matrix[i, j], loss_i_j)

def test(y_true, y_pred):
    # Get pos, neg indices
    pos_indices = T.eq(y_true, 1).nonzero()
    neg_indices = T.eq(y_true, -1).nonzero()
    count_pos = T.sum(T.ones_like(pos_indices[0]))
    count_neg = T.sum(T.ones_like(neg_indices[0]))
    # Then get total number of positives and negatives

    pos_loss_results, updates = theano.scan(fn = calcLossAtLocation, outputs_info = T.zeros_like(y_true), sequences = [pos_indices[0], pos_indices[1]], non_sequences = [count_pos, count_neg, y_true, y_pred])

    neg_loss_results, updates = theano.scan(fn = calcLossAtLocation, outputs_info = pos_loss_results[-1], sequences = [neg_indices[0], neg_indices[1]], non_sequences = [count_pos, count_neg, y_true, y_pred])
    return neg_loss_results[-1]

final_model.compile(loss=test, optimizer=adam_opt)

plot(final_model, to_file='model.png', show_shapes=True)
#final_model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)
