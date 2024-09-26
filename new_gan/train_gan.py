import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from utils.config import parse_args
from utils.data_loader import get_data_loader

from models.gan import GAN
from models.dcgan import DCGAN_MODEL
from models.wgan_clipping import WGAN_CP
from models.wgan_gradient_penalty import WGAN_GP
#from models.wgan_c import WGAN_GP
from models.wgan_fmnist import WGAN_FMNIST
from models.wgan_C_fmnist import WGAN_C_FMNIST
from models.wgan_gp_fmnist import WGAN_GP_FMNIST
from models.wgan_C_cifar import WGAN_C_GP
from models.wgan_c_svhn import WGAN_C_SVHN

def main(args):
    # Load datasets to train and test loaders

    nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #labels = ["0","1","2","3" ,"4", "5", "6" , "7", "8", "9"]
    labels = ["airplane" ,"automobile" ,"bird","cat" , "deer" , "dog", "frog" , "horse", "ship" , "truck"]
    for i in range(10):
        model = None
        if args.model == 'GAN':
            model = GAN(args)
        elif args.model == 'DCGAN':
            model = DCGAN_MODEL(args)
        elif args.model == 'WGAN-CP':
            model = WGAN_CP(args)
        elif args.model == 'WGAN-GP':
            model = WGAN_GP(args)
#            model = WGAN_C_GP(args)
        elif args.model == 'WGAN_FMNIST':
            #model = WGAN_FMNIST(args)
            #model = WGAN_C_FMNIST(args)
            model = WGAN_GP_FMNIST(args)
        elif args.model == 'WGAN_SVHN':
            model = WGAN_C_SVHN(args)
        else:
            print("Model type non-existing. Try again.")
            exit(-1)
            
        print('start to load data of label: {}'.format(labels[i]))
        train_loader, test_loader, label = get_data_loader(args, nums[i], labels[i])
        #feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)
        print('successful load data')

        # Start model training
        if args.is_train == 'True':
            model.train(train_loader,label)

        # start evaluating on test data
        else:
            model.evaluate(test_loader, args.load_D, args.load_G)
            # for i in range(50):
            #    model.generate_latent_walk(i)


if __name__ == '__main__':
    args = parse_args()
    print(args.cuda)
    main(args)
