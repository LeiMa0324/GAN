from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
import keras.layers as layers
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# 实验二: batch size 128

# Load data
(X_train, _), (_, _) = mnist.load_data()

# Preprocessing
X_train = X_train.reshape(-1, 784)   # -1,自动根据第二个参数计算行数
X_train = X_train.astype('float32')/255 # 转换RGB颜色为0-1之间

# Set the dimensions of the noise
z_dim = 100     # 输入noise vector的dimension

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)  # 使用adam做gradient ascent

# Generator
g = Sequential()    # sequential是一个网络，可以包含多个layer
g.add(Dense(256,input_dim = z_dim))     #定义第一层，Dense函数是一个NN网络，第一层需要指定输入的dimension
                                        # 第一个参数是neurons的个数，也就是输出的dimension
g.add(layers.LeakyReLU(alpha=0.3))    # 对上述输出加一个actionvation，可以直接在dense上加参数activation
g.add(Dense(784, activation='sigmoid')) # 生成28*28的图片

# Discrinimator
d = Sequential()    #初始化一个NN网络
d.add(Dense(256,input_dim=784))   # 输入为784维，28*28的图片,256个neurons
d.add(layers.LeakyReLU(alpha=0.3))

d.add(Dense(1, activation='sigmoid'))  # 输出一个0-1之间的sclar
# 定义lossfunction，优化器等，实现training
d.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# GAN
d.trainable = False #冻结discriminator，防止在generator训练期间更新参数
inputs = Input(shape=(z_dim, ))

# hidden层为generator
hidden = g(inputs)
# output层为dscriminator
output = d(hidden)
gan = Model(inputs, output)     #给定一个输入张量和输出张量，可以实例化一个model,输入的shape和途中的网络
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# Training
def train(epochs=1, plt_frq=1, BATCH_SIZE=128):
    # 将训练集分为468个batchsize
    batchCount = int(X_train.shape[0] / BATCH_SIZE)
    print('Epochs:', epochs)
    print('Batch size:', BATCH_SIZE)
    print('Batches per epoch:', batchCount)

    # 使用一个batch做一次迭代训练网络
    for i in range(batchCount):
        # Create a batch by drawing random index numbers from the training set
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)]
        # Create noise vectors for the generator
        noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))

        # Generate the images from the noise
        generated_images = g.predict(noise)
        X = np.concatenate((image_batch, generated_images))

        # Create labels
        y = np.zeros(2 * BATCH_SIZE)
        y[:BATCH_SIZE] = 1  # 前半部分为正例，后半部分为负例

        # Train discriminator on generated images
        d.trainable = True
        # 仅训练discriminator，使之判断最准确
        d_loss = d.train_on_batch(X, y)  #运行一次更新，返回误差


        # Train generator
        noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
        y2 = np.ones(BATCH_SIZE)
        d.trainable = False
        # 固定住discriminator，训练整个gan网络（generator)
        # 使fake image的score最高（伪装正例，骗过discriminator）。
        g_loss = gan.train_on_batch(noise, y2)

        #记录最后一次loss，返回该epoch的loss
        if i == batchCount-1:
            return [g_loss[0],d_loss[0]]


epochCount = 200
g_loss_arr = []
d_loss_arr = []

for i in range(epochCount):
    loss_arr = train(i)
    g_loss_arr.append(loss_arr[0])
    d_loss_arr.append(loss_arr[1])

# plot loss
LossFigure = plt.figure()
plt.xlabel(u'Epochs')

plt.ylabel(u'Loss')

my_x_ticks = np.arange(0,epochCount,20)
plt.xticks(my_x_ticks)

plt.plot(np.arange(0,epochCount),g_loss_arr)
plt.plot(np.arange(0,epochCount),d_loss_arr)
plt.legend(["g__training_loss","d_training_loss"],loc=2)
LossFigure.savefig("LossFigures/exp2_loss.png")
plt.show()

#================================================
# 测试阶段
#================================================
# Generate images
np.random.seed(504)
h = w = 28
num_gen = 25

z = np.load("FixNoise.npy")
generated_images = g.predict(z)

# plot of generation
n = np.sqrt(num_gen).astype(np.int32)
I_generated = np.empty((h*n, w*n))
for i in range(n):
    for j in range(n):
        I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = generated_images[i*n+j, :].reshape(28, 28)

GeneratedImages = plt.figure(figsize=(4, 4))
plt.axis("off")
plt.imshow(I_generated, cmap='gray')
GeneratedImages.savefig("GeneratedImages/exp2_res.png")
plt.show()


# # serialize model to JSON
# model_json = g.to_json()
# with open("generator.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# g.save_weights("generator.h5")
# print("Saved model to disk")
# for i in range(1,20):
#     train(epochs=i)