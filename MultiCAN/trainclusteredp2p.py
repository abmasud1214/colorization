import clusteredp2p
import tensorflow as tf

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    ClusteredModel = clusteredp2p.ClusteredP2PGan(5, "./Models/ClusteredClassifier", (256, 256, 3), 0)
    ClusteredModel.train("../data_imagenet_small/train", "./Models", n_epochs=3, n_batch=10)

