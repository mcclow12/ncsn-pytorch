from utils import *
from trainer import NCSNTrainer
from model import CondRefineNetDilated

#Adam optim params
lr = 1e-3
beta1 = 0.9
beta2 = 0.999

batch_size = 8 #128 too much memory for me
num_iters = 20000

#Lagenvian params
lang_eps = 2e-5
lang_T = 1000

#data params
dataset = "MNIST"
image_size = 32
channels = 1
logit_transform = False
random_flip = False
num_classes = 10
ngf = 64

#save params
checkpoints_folder = './checkpoints'
save_every = 1000

mnist_config = config(
            dataset, 
            image_size, 
            channels, 
            logit_transform, 
            random_flip,
            num_classes,
            ngf
)

ncsn = CondRefineNetDilated(mnist_config)

dataloader = get_train_set(batch_size)

trainer = NCSNTrainer(ncsn, lr, dataloader, num_iters, beta1, beta2, checkpoints_folder, save_every)
trainer.train_ncsn()

model_path = './run/logs/mnist/checkpoint.pth'

state_dict = torch.load(model_path, map_location=torch.device('cpu'))[0]
trained_ncsn = CondRefineNetDilated(mnist_config)
trained_ncsn = torch.nn.DataParallel(trained_ncsn)
trained_ncsn.load_state_dict(state_dict)
trained_ncsn.eval()
eps = 1e-5
T = 100
num_samples = 1
nrow = 1
sample = trainer.save_samples(num_samples, nrow, trained_ncsn, eps, T)

