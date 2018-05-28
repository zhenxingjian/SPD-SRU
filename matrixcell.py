import tensorflow as tf
import numpy as np
import pdb
from tensorflow.pOthon.util import nest

def FM(A,B,a,n):
    '''
    Compute the Weighted Fréchet mean
    '''
    return tf.add((1.-a)*A,a*B)


def NUS(W_root, A, a_num, tot, n=1):
    '''
    Compute the weighted average on the M -> Y
    '''
    W = tf.pow(W_root,2)
    if a_num==1:
       return (W[0]/tot)*A
    else:
        result = tf.squeeze(tf.slice(A,[0,0,0,0],[-1,1,-1,-1]))*(W[0]/ tot)
        for i in range(1, A.shape[1]):
            result = result + tf.squeeze(tf.slice(A,[0,i,0,0],[-1,1,-1,-1]))*(W[i]/tot)
        return result

def MatrixExp(B,l,n):
    '''
    input a matrix B, and the total length to be calculated, n is the size of B
    output the somehow exp(B) = I + B + B^2 / 2! + B^3 / 3! + ... + B^l / l!
    '''
    
    Result = tf.eye(n)
    return tf.matmul( tf.matrix_inverse(tf.subtract(Result , B)) , tf.add( Result , B) )

def Translation(A,B,n, batch_size):

    '''
    input the matrix A and vector B
    change B to be SO 
    like [[0 ,  1, 2]
          [-1,  0, 3]
          [-2, -3, 0]]
    return B * A * B.T
    '''
    power_matrix = 5
    B = tf.reshape(B,[1,-1])

    #lower_triangel = fill_triangular(B)
    line_B = [tf.zeros([1,n])]
    for i in range (n-1):
        temp_line = tf.concat([ tf.slice(B,[0,i],[1,i+1]) , tf.zeros([1,n-i-1]) ] ,axis = 1)
        line_B.append(temp_line)

    lower_triangel = tf.concat(line_B,axis = 0)

    B_matrix = tf.subtract(lower_triangel, tf.transpose(lower_triangel))
    
    B_matrix = MatrixExp(B_matrix,power_matrix,n)

    B_matrix = tf.tile ( tf.expand_dims(B_matrix,0),[batch_size,1,1] )

 

    Tresult = tf.matmul(B_matrix,A)                              # B * A

    Tresult = tf.matmul(Tresult,tf.transpose(B_matrix,[0,2,1]))      # B * A * B.T
    return Tresult

def Chol_de(A,n,batch_size):
    '''
    input matrix A and it's size n
    decomponent by Cholesky
    return a vector with size n*(n+1)/2
    '''
    #A = tf.add (A , 1e-10 * tf.diag(tf.random_uniform([n])) )
    # A = tf.cond( 
    #     tf.greater( tf.matrix_determinant(A),tf.constant(0.0) ) , 
    #     lambda: A, 
    #     lambda: tf.add (A , 1e-10 * tf.eye(n) ) )
    #L = tf.cholesky(A)

    L = A
    result = tf.slice(L,[0,0,0],[-1,1,1])
    for i in range(1,n):
        j = i
        result = tf.concat( [result , tf.slice(L,[0,i,0],[-1,1,j+1])],axis = 2 )

    result = tf.reshape(result,[-1,n*(n+1)//2])
    return result

def Chol_com(l,n,batch_size):
    '''
    input vector l and target shape n and eps to be the smallest value
    return lower trangle matrix
    '''
    lower_triangle_ = tf.expand_dims(tf.concat([tf.slice(l,[0,0],[-1,1]), tf.zeros((batch_size,n-1))],axis=1),1)
    for i in range(1, n):
        lower_triangle_ = tf.concat([lower_triangle_,tf.expand_dims(tf.concat([tf.slice(l,[0,i*(i+1)//2],[-1,i+1]), tf.zeros((batch_size,n-i-1))],axis=1),1)],axis=1)

    lower_triangle_ = tf.add(lower_triangle_ , tf.tile(tf.expand_dims(tf.eye(n)*1e-2,axis=0),[batch_size,1,1]) )
    result = tf.matmul(lower_triangle_,lower_triangle_,transpose_b=True)
    return result




class SPDSRU(tf.contrib.rnn.RNNCell):
    """
    Implements a simple distribution based recurrent unit that keeps moving
    averages of the mean map embeddings of features of inputs on manifold.
    """

    def __init__(self , alpha , batch_size , matrix_size , eps=1e-10 ):
        self._alpha = alpha
        self._a_num = len(alpha)
        self._batch_size = batch_size
        self._matrix_size = matrix_size
        self._eps = eps

    @property
    def state_size(self):
        return int(self._a_num * self._matrix_size * self._matrix_size)

    @property
    def output_size(self):
        return int(self._matrix_size * self._matrix_size)

    def __call__(self, inputs, state, scope=None):
        a_num = self._a_num
        batch_size = self._batch_size
        eps = self._eps
        n = self._matrix_size
        a = self._alpha

        with tf.variable_scope(scope or type(self).__name__):
            Weights_rnn = {
            'WR_root':tf.get_variable('WR_root' , [a_num] , initializer = tf.random_uniform_initializer() , dtype = np.float32),
            'Wt_root':tf.get_variable('Wt_root' , [1] , initializer = tf.random_uniform_initializer() , dtype = np.float32),
            'Wphi_root':tf.get_variable('Wphi_root' , [1] , initializer = tf.random_uniform_initializer() , dtype = np.float32),
            'Ws_root':tf.get_variable('Ws_root' , [a_num] , initializer = tf.random_uniform_initializer() , dtype = np.float32)
            } # To make every weights to be positive.

            Bias_rnn = {
            'Br':tf.get_variable('Br' , [n*(n-1)//2,1] , initializer = tf.random_uniform_initializer() , dtype = np.float32),
            'Bt':tf.get_variable('Bt' , [n*(n-1)//2,1] , initializer = tf.random_uniform_initializer() , dtype = np.float32),
            'By':tf.get_variable('By' , [n*(n-1)//2,1] , initializer = tf.random_uniform_initializer() , dtype = np.float32)
            }

            Xt = inputs
            Mt_1 = tf.reshape(state , [-1,self._a_num , self._matrix_size , self._matrix_size])

            n_current_X = tf.reshape(Xt,[batch_size,n,n])
            Yt =  NUS(Weights_rnn['WR_root'], Mt_1, a_num, tf.reduce_sum(tf.pow(Weights_rnn['WR_root'],2))+eps, n)
            Rt = Translation( Yt, Bias_rnn['Br'] , n, batch_size )
            Tt = FM(n_current_X, Rt, tf.pow(Weights_rnn['Wt_root'],2)/(tf.reduce_sum([tf.pow(Weights_rnn['Wt_root'],2), tf.pow(Weights_rnn['Wphi_root'],2)])+eps), n)
            Phit = Translation ( Tt, Bias_rnn['Bt'] , n, batch_size )
            
            next_state = []
            for j in range(a_num):
                next_state.append (  tf.expand_dims ( FM ( tf.reshape ( tf.slice(Mt_1,[0,j,0,0],[-1,1,n,n] ) ,[batch_size,n,n]) , Phit, a[j] , n ) , 1 ) )
            Mt = tf.concat(next_state,axis = 1)
            St =  NUS(Weights_rnn['Ws_root'], Mt, a_num, tf.reduce_sum(tf.pow(Weights_rnn['Ws_root'],2))+eps, n)
            Ot = Translation ( St, Bias_rnn['By'] , n, batch_size )
            
            out_state = tf.reshape(Mt , [-1, int(self._a_num * self._matrix_size * self._matrix_size)])

            output = tf.reshape(Ot , [-1, int(self._matrix_size * self._matrix_size)] )

            return (output, out_state)


class CNNRNNCell(tf.contrib.rnn.RNNCell):
    """
    Implements a CNN layer like a RNN to share parameters.
    It will compute the SPD matrix from CNN layer.
    """

    def __init__(self , num_layer , kernel_shape , alpha , batch_size , matrix_size , in_channel , out_channel , reduced_spatial_dim , beta , keep_prob , eps=1e-10 ):
        '''
        kernel_shape is list of list(size 3, width, height, outchannel, like [5,5,15])
        '''
        self._alpha = alpha
        self._a_num = len(alpha)
        self._batch_size = batch_size
        self._matrix_size = matrix_size
        self._eps = eps
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._reduced_spatial_dim = reduced_spatial_dim
        self._beta = beta
        self._num_layer = num_layer
        self._kernel_shape = kernel_shape
        self._keep_prob = keep_prob
        assert num_layer == len(kernel_shape)
        assert kernel_shape[num_layer-1][2] == out_channel
        assert matrix_size == out_channel+1


    @property
    def state_size(self):
        return int(self._a_num * self._matrix_size * self._matrix_size)

    @property
    def output_size(self):
        return int(self._matrix_size * self._matrix_size)

    def __call__(self, inputs, state, scope=None):
        a_num = self._a_num
        batch_size = self._batch_size
        eps = self._eps
        n = self._matrix_size
        a = self._alpha
        in_channel = self._in_channel
        out_channel = self._out_channel
        reduced_spatial_dim = self._reduced_spatial_dim
        beta = self._beta
        num_layer = self._num_layer
        kernel_shape = self._kernel_shape
        keep_prob = self._keep_prob

        with tf.variable_scope(scope or type(self).__name__):

            Weights_cnn = []
            kernel_out_channel = in_channel
            for layer_idx in range(num_layer):
                kernel_width = kernel_shape[layer_idx][0]
                kernel_height = kernel_shape[layer_idx][1]
                kernel_in_channel = kernel_out_channel
                kernel_out_channel = kernel_shape[layer_idx][2]

                Weights_cnn.append(tf.get_variable('W'+str(layer_idx),[kernel_width,kernel_height,kernel_in_channel,kernel_out_channel],
                                                    initializer = tf.random_normal_initializer(stddev=1e-4), 
                                                    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-2),
                                                    dtype = np.float32))
            
            P1 = inputs

            cov_mat = None

            for layer_idx in range(num_layer):
                C1_bn = tf.keras.layers.BatchNormalization()(tf.nn.conv2d(P1,Weights_cnn[layer_idx],[1,1,1,1],'SAME'))
                C1 = tf.nn.relu(C1_bn)
                # C1 = tf.nn.dropout(C1, keep_prob)
                P1 = tf.nn.max_pool(C1,[1,2,2,1],[1,2,2,1],'SAME')
                if keep_prob == 1.:
                    Training = False
                else:
                    Training = True
                # P1 = tf.layers.dropout(P1, rate = 1.-keep_prob,training = Training)

            P2 = tf.transpose(P1,[0,3,2,1])
            Fl = tf.reshape(P2,[batch_size,out_channel,reduced_spatial_dim])
            mean_batch = tf.reduce_mean(Fl,2)   #batch_size x out_channel
            mean_tensor = tf.tile(tf.expand_dims(mean_batch,axis=2),[1,1,reduced_spatial_dim]) #batch_size x out_channel x reduced_spatial_dim
            Fl_m = tf.subtract(Fl,mean_tensor)

            mean_batch = tf.expand_dims(mean_batch,axis=2)
            mean_cov = tf.matmul(mean_batch,mean_batch,transpose_b = True)

            cov_feat = tf.add(tf.matmul(Fl_m, Fl_m, transpose_b=True), beta*beta*mean_cov)

            cov_feat = tf.concat([cov_feat, beta*mean_batch],axis=2)

            mean_batch_t = tf.concat([beta*mean_batch, tf.constant([1.],shape=[batch_size,1,1])],axis=1)
            mean_batch_t = tf.transpose(mean_batch_t,[0,2,1])

            cov_feat = tf.concat([cov_feat, mean_batch_t],axis=1)
            cov_mat = cov_feat

            output = cov_mat
            out_state = state

            return (output, out_state)

class NormalCNNCell(tf.contrib.rnn.RNNCell):
    """
    Implements a CNN layer like a RNN to share parameters.
    """

    def __init__(self , num_layer , kernel_shape , batch_size , in_channel , out_channel , keep_prob , output_dims ):
        '''
        kernel_shape is list of list(size 3, width, height, outchannel, like [5,5,15])
        '''

        self._batch_size = batch_size
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._num_layer = num_layer
        self._kernel_shape = kernel_shape
        self._keep_prob = keep_prob
        self._output_dims = output_dims



    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return int(self._output_dims)

    def __call__(self, inputs, state, scope=None):

        batch_size = self._batch_size
        in_channel = self._in_channel
        out_channel = self._out_channel
        num_layer = self._num_layer
        kernel_shape = self._kernel_shape
        keep_prob = self._keep_prob

        with tf.variable_scope(scope or type(self).__name__):
            Weights_cnn = []
            kernel_out_channel = in_channel
            for layer_idx in range(num_layer):
                kernel_width = kernel_shape[layer_idx][0]
                kernel_height = kernel_shape[layer_idx][1]
                kernel_in_channel = kernel_out_channel
                kernel_out_channel = kernel_shape[layer_idx][2]

                Weights_cnn.append(tf.get_variable('W'+str(layer_idx),[kernel_width,kernel_height,kernel_in_channel,kernel_out_channel],
                                                    initializer = tf.random_normal_initializer(stddev=1e-4), 
                                                    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-2),
                                                    dtype = np.float32))
            
            P1 = inputs


            for layer_idx in range(num_layer):
                C1_bn = tf.keras.layers.BatchNormalization()(tf.nn.conv2d(P1,Weights_cnn[layer_idx],[1,1,1,1],'SAME'))
                C1 = tf.nn.relu(C1_bn)
                P1 = tf.nn.max_pool(C1,[1,2,2,1],[1,2,2,1],'SAME')

            P2 = tf.transpose(P1,[0,3,2,1])
            Fl = tf.reshape(P2,[batch_size,-1])
            
            output = Fl
            out_state = state

            return (output, out_state)