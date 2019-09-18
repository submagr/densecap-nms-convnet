import theano.tensor as T
import theano
import numpy as np

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


y_true = T.matrix("y_true")
y_pred = T.matrix("y_pred")
fn = theano.function([y_true, y_pred], test(y_true, y_pred) )

true = np.matrix([[1, 1, 1, -1],[-1, 1, -1, 1], [1, -1, 1, -1], [1, -1, -1, 1]])
pred = np.matrix([[-1, 1, -1, -1], [1, 1, -1, -1], [-1, 1, 1, 1], [1, -1, -1, -1]])
print fn(true, pred)
