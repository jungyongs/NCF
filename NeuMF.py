import argparse
from keras import initializers, Model
from keras.layers import Input, Embedding, Flatten, Multiply, Dense, Concatenate
from keras.optimizers.legacy import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
import numpy as np
import heapq
from time import time
import scipy.sparse as sp
import GMF,MLP

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=32,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')     
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")               
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='sgd',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()

random_normal = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=42)


def get_model(num_users, num_items, latent_dim, layers, regs=0):
    # Input Layers
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    # Embedding layers
    GMF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'gmf_user_embedding',
                                  embeddings_initializer = random_normal, embeddings_regularizer = l2(regs), input_length=1)
    GMF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'gmf_item_embedding',
                                  embeddings_initializer = random_normal, embeddings_regularizer = l2(regs), input_length=1) 
    
    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = 'mlp_user_embedding',
                                  embeddings_initializer = random_normal, embeddings_regularizer = l2(regs), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'mlp_item_embedding',
                                  embeddings_initializer = random_normal, embeddings_regularizer = l2(regs), input_length=1) 

    # Flatten
    GMF_user_latent = Flatten()(GMF_Embedding_User(user_input))
    GMF_item_latent = Flatten()(GMF_Embedding_Item(item_input))
    MLP_user_latent = Flatten()(MLP_Embedding_User(user_input))
    MLP_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    

    # Element-wise product
    GMF_vector = Multiply()([GMF_user_latent, GMF_item_latent])

    # Concatenate
    MLP_vector = Concatenate()([MLP_user_latent, MLP_item_latent])

    # MLP layers
    for i in range(1, len(layers)):
        layer = Dense(layers[i], kernel_regularizer=l2(regs), activation='relu', name = 'layer%d' %i)
        MLP_vector = layer(MLP_vector)
    
    vector = Concatenate()([GMF_vector, MLP_vector])

    # Output
    output = Dense(1, activation = 'sigmoid', kernel_initializer='lecun_uniform', name = 'output')(vector)

    model = Model(inputs=[user_input, item_input], outputs=output)
    
    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    #GMF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('gmf_user_embedding').set_weights(gmf_user_embeddings)
    model.get_layer('gmf_item_embedding').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_user_embedding').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_item_embedding').set_weights(mlp_item_embeddings)

    # MLP layers
    for i in range(1,num_layers):
        mlp_layers_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layers_weights)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('output').get_weights()
    mlp_prediction = mlp_model.get_layer('output').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]))
    new_bias = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('output').set_weights([0.5*new_weights, 0.5*new_bias])    
    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_items = train.shape[1]
    for (u,i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u,j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

#################### Dataset ####################

def load_train(filename):
    train_num_users, train_num_items, cnt = 0, 0, 0
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            u, i = int(arr[0]), int(arr[1])
            train_num_users = max(train_num_users, u)
            train_num_items = max(train_num_items, i)
            cnt += 1
            line = f.readline()
    # Construct matrix
    mat = sp.dok_matrix((train_num_users+1, train_num_items+1), dtype=np.float32)
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if (rating > 0):
                mat[user, item] = 1.0
            line = f.readline()    
    return mat
        
def load_test(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList

def load_negatives(filename):
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1: ]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList

#################### Evaluation ####################

def gethits(gtItem,ranklist):
    if gtItem in ranklist:
        return 1
    return 0

def getndcg(gtItem,ranklist):
    if gtItem in ranklist:
        index = ranklist.index(gtItem)
        return (1/(np.log2(index+2)))
    return 0

def eval_one_rating(idx):
    rating = testRatings[idx]
    items = testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)

    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    predictions = model.predict([users, np.array(items)], 
                                 batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
    hr = gethits(gtItem, ranklist)
    ndcg = getndcg(gtItem, ranklist)
    return (hr, ndcg)

def evaluate_model(model, testRatings, testNegatives, K):
    hits, ndcgs = [], []
    for idx in range(len(testRatings)):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits,ndcgs)

if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    regs = args.reg_mf
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain

    topK = 10
    print("NeuMF arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_NeuMF_%d_%s.h5' %(args.dataset, mf_dim, args.layers)

    # Loading data
    t1 = time()
    train = load_train(args.path + args.dataset+'.train.rating')
    testRatings = load_test(args.path + args.dataset+'.test.rating')
    testNegatives = load_negatives(args.path + args.dataset+'.test.negative')
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(num_users, num_items, mf_dim, layers, regs)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(learning_rate=learning_rate), loss='binary_crossentropy')
    print(model.summary())

    # Load pretrain model
    if mf_pretrain != '' and mlp_pretrain != '':
        gmf_model = GMF.get_model(num_users,num_items,mf_dim)
        gmf_model.load_weights(mf_pretrain)
        mlp_model = MLP.get_model(num_users,num_items, layers, reg_layers)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
        print("Load pretrained GMF (%s) and MLP (%s) models done. " %(mf_pretrain, mlp_pretrain))

    # Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        user_input, item_input, labels = get_train_instances(train, num_negatives)

        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                            np.array(labels), # labels 
                            batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
        t2 = time()

        # Evaluation
        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK)
        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
            % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    